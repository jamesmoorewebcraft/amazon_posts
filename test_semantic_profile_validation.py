import unittest
from get_response_from_openai import (
    _deterministic_semantic_claim_violations,
    _deterministic_remove_cross_model_observations,
    _deterministic_repair_repeated_measurements,
    _enforce_evidence_weight_conflicts,
    _model_semantic_violation_is_actionable as actionable,
    _semantic_repair_candidate_made_progress,
)

NAME = "Example Pack 25L"
PROFILE = {
    "claim_provenance": [{"product": NAME, "distinctive_markers": ["10kg"]}],
    "products": [{"name": NAME, "facts": {
        "load_support": {"canonical_value": "up to 10 kg", "requires_attribution": True, "value_status": "confirmed", "forbidden_terms": []},
        "features": {"canonical_value": "sleeve", "requires_attribution": False, "value_status": "confirmed", "forbidden_terms": []},
        "dimensions": {"canonical_value": "", "requires_attribution": False, "value_status": "unresolved", "forbidden_terms": []},
        "durability_history": {"canonical_value": "long-lasting", "requires_attribution": True, "value_status": "confirmed", "forbidden_terms": ["3 years"]},
        "weight": {"canonical_value": "680 g", "safe_wording": "One source reports: Weighs 680 g (24 oz).", "requires_attribution": True, "value_status": "confirmed", "forbidden_terms": []},
        "fit_or_compatibility": {"canonical_value": "unisex", "safe_wording": "One source reports: Unisex design.", "requires_attribution": True, "value_status": "confirmed", "forbidden_terms": []},
        "water_protection": {"canonical_value": "water-resistant", "safe_wording": "Water-resistant but not fully waterproof.", "requires_attribution": False, "value_status": "confirmed", "forbidden_terms": ["waterproof"]},
        "access_or_security": {"canonical_value": "none", "safe_wording": "One source reports: No anti-theft features; front pockets exposed.", "requires_attribution": True, "value_status": "confirmed", "forbidden_terms": []},
    }}],
}


def finding(attribute, passage, reason, repair=""):
    return actionable({"product": NAME, "attribute": attribute, "passage": passage, "reason": reason, "repair": repair}, PROFILE, {})


class SemanticProfileValidationTests(unittest.TestCase):
    def test_profile_overrules_false_attribution_requirement(self):
        self.assertFalse(finding("features", "The pack includes a sleeve.", "The profile requires attribution.", "Add attribution."))

    def test_unresolved_disclaimer_is_not_a_violation(self):
        self.assertFalse(finding("dimensions", "We could not reliably confirm its dimensions.", "Dimensions are unconfirmed.", "No repair needed; this is correct."))

    def test_owned_measurement_overrules_false_cross_model_finding(self):
        self.assertFalse(finding("load_support", "One source reports comfort around 10 kg.", "The 10 kg evidence belongs only to another model.", "Remove it."))

    def test_forbidden_owned_value_remains_actionable(self):
        self.assertTrue(finding("durability_history", "It still looks new after 3 years.", "The 3-year evidence belongs only to another model.", "Remove it."))

    def test_missing_required_attribution_remains_actionable(self):
        self.assertTrue(finding(
            "weight",
            "The product weighs 680 g.",
            "The canonical profile requires attribution for this value.",
            "Add attribution.",
        ))

    def test_keep_single_attributed_observation_is_not_a_conflict(self):
        self.assertFalse(finding(
            "fit_or_compatibility",
            "One reviewer found the product too long for their frame.",
            "This observation is correctly attributed but should not be generalized.",
            "Keep as a single user report, not a universal statement.",
        ))

    def test_exact_safe_wording_is_never_a_model_conflict(self):
        self.assertFalse(finding(
            "water_protection",
            "Water-resistant but not fully waterproof.",
            "This is a correct statement, but the article also uses stronger wording.",
            "Clarify another passage.",
        ))

    def test_repetition_is_editorial_not_factual(self):
        self.assertFalse(finding(
            "access_or_security",
            "One source reports: No anti-theft features; front pockets exposed.",
            "This is correctly attributed, but the article repeats it multiple times, which is redundant.",
            "Keep one instance and remove duplicates.",
        ))

    def test_confirmed_fact_with_confusing_phrasing_is_not_actionable(self):
        self.assertFalse(finding(
            "access_or_security",
            "The 25L version lacks this option entirely.",
            "The 25L has no lockable zipper; the lack is confirmed, but the phrasing may confuse.",
            "Clarify that the 25L does not have a lockable zipper.",
        ))

    def test_section_placement_objection_is_not_a_fact_violation(self):
        self.assertFalse(finding(
            "features",
            "The pack is hydration compatible.",
            "This appears in the Airline Compatibility section but is unrelated to dimensions.",
            "Move it to another section.",
        ))

    def test_measurement_suffix_is_not_counted_as_a_duplicate(self):
        html = (
            "<p>The panel measures 26 cm by 40 cm with a 6 cm gap at the top.</p>"
        )
        violations = _deterministic_semantic_claim_violations(
            html, PROFILE, NAME, {"semantic_fact_audit": {}}
        )
        self.assertFalse(any(
            item.get("attribute") == "repeated_measurement" for item in violations
        ))
    def test_conflicting_exact_product_weights_stay_unresolved(self):
        products = [{
            "name": NAME,
            "facts": {"weight": {
                "canonical_value": "680 g",
                "safe_wording": "One source reports a weight of 680 g.",
                "confidence": "medium",
                "value_status": "confirmed",
                "source_type": "specialist_review",
                "source_count": 1,
                "exact_source_count": 1,
                "conflicting_values": [],
            }},
        }]
        ledger = [
            {"product": NAME, "evidence_excerpt": "The pack weighs 680 g (24 oz)."},
            {"product": NAME, "evidence_excerpt": "At 640 g (1.4 lb), it is easy to carry."},
            {"product": NAME, "evidence_excerpt": "The holders fit a 32 oz water bottle."},
            {"product": NAME, "evidence_excerpt": "Loads above 10 kg can be tiring."},
        ]
        changed = _enforce_evidence_weight_conflicts(products, ledger)
        fact = products[0]["facts"]["weight"]
        self.assertEqual(1, changed)
        self.assertEqual("", fact["canonical_value"])
        self.assertEqual("source_conflict", fact["value_status"])
        self.assertEqual("low", fact["confidence"])
        self.assertEqual(["680 g", "640 g"], fact["conflicting_values"])
        self.assertIn("single figure could not be verified", fact["safe_wording"])


    def test_true_repeated_measurement_is_still_detected(self):
        html = "<p>The product weighs 680 g, giving it a weight of 680 g.</p>"
        violations = _deterministic_semantic_claim_violations(
            html, PROFILE, NAME, {"semantic_fact_audit": {}}
        )
        self.assertTrue(any(
            item.get("attribute") == "repeated_measurement" for item in violations
        ))


    def test_repeated_threshold_measurement_uses_non_numeric_implication(self):
        passage = (
            "No hip belt, so loads above 10 kg may strain shoulders over time "
            "One source reports the pack is comfortable up to about 10 kg."
        )
        html = (
            "<p>Keep this introduction. No hip belt, so loads above 10 kg may "
            "strain shoulders over time\nOne source reports the pack is "
            "comfortable up to about 10 kg. Keep this conclusion.</p>"
        )
        output, repaired = _deterministic_repair_repeated_measurements(
            html,
            [{
                "attribute": "repeated_measurement",
                "passage": passage,
                "reason": "The measurement '10kg' is stated twice in one sentence.",
            }],
        )
        self.assertEqual({passage}, repaired)
        self.assertIn("heavier loads may strain shoulders", output)
        self.assertIn("Keep this introduction.", output)
        self.assertIn("Keep this conclusion.", output)
        self.assertEqual(1, output.count("10 kg"))

    def test_semantic_candidate_can_advance_when_all_cited_passages_are_fixed(self):
        self.assertTrue(_semantic_repair_candidate_made_progress(
            [{"passage": "one"}],
            [{"passage": "new one"}, {"passage": "new two"}],
            ["one"],
            1,
        ))
        self.assertTrue(_semantic_repair_candidate_made_progress(
            [{"passage": "one"}, {"passage": "two"}],
            [{"passage": "two"}],
            ["one", "two"],
            1,
        ))
        self.assertFalse(_semantic_repair_candidate_made_progress(
            [{"passage": "one"}],
            [{"passage": "one"}],
            ["one"],
            0,
        ))

    def test_cross_model_observation_with_explicit_owner_is_removed(self):
        passage = "The 25L felt too long for the reviewer, making the 23L a better fit."
        html = f"<h2>Fit</h2><p>{passage}</p>"
        output, repaired = _deterministic_remove_cross_model_observations(
            html,
            [{
                "product": "Example Pack 25L",
                "attribute": "claim_provenance",
                "passage": passage,
                "reason": "The observation belongs to Example Pack 23L.",
                "repair": "Remove the observation.",
                "evidence_owner": "Example Pack 23L",
            }],
        )
        self.assertNotIn(passage, output)
        self.assertEqual({passage}, repaired)

    def test_model_identified_cross_model_sentence_is_removed_without_owner_field(self):
        passage = "The 25L felt too long for a 5-foot-1 reviewer."
        output, repaired = _deterministic_remove_cross_model_observations(
            f"<p>{passage}</p>",
            [{
                "product": NAME,
                "attribute": "dimensions",
                "passage": passage,
                "reason": "This observation is from the review of the Women's 23L, not from evidence for the 25L.",
                "repair": "Remove this sentence or attribute it to the Women's 23L.",
            }],
        )
        self.assertNotIn(passage, output)
        self.assertEqual({passage}, repaired)

    def test_general_rewording_request_does_not_trigger_cross_model_removal(self):
        passage = "The product may suit shorter users."
        output, repaired = _deterministic_remove_cross_model_observations(
            f"<p>{passage}</p>",
            [{
                "product": NAME,
                "attribute": "fit_or_compatibility",
                "passage": passage,
                "reason": "The wording could be clearer.",
                "repair": "Rewrite this sentence.",
            }],
        )
        self.assertIn(passage, output)
        self.assertFalse(repaired)


if __name__ == "__main__":
    unittest.main()
