import json
import unittest
from unittest.mock import patch

from get_response_from_openai import (
    _deterministic_remove_explicit_omissions,
    _deterministic_repair_with_canonical_safe_wording,
    _deterministic_repair_required_attribution,
    _deterministic_repair_repeated_measurements,
    _deterministic_semantic_claim_violations,
    _is_related_profile_product,
    _source_claim_provenance,
    audit_semantic_claim_consistency,
    annotate_source_product_boundaries,
    build_canonical_product_profile,
    configure_runtime_category,
    keyword_product_identity_tokens,
    pick_specific_from_whitelist,
    reviewed_product_matches_keyword,
)
from insert_amazon_links_images import (
    _canonical_heading_numeric_tokens,
    reviewed_product_matches_keyword as downstream_identity_match,
)


CFG = {
    "review_identity": {
        "lock_keyword_model_tokens": True,
        "allow_product_retargeting": False,
        "block_on_mismatch": True,
    },
    "canonical_facts": {
        "related_product_min_shared_tokens": 2,
        "single_source_hard_attributes": [
            "weight", "dimensions", "device_fit", "compatibility",
            "airline_compatibility",
        ],
        "family_identity_generic_words": [
            "pack", "bag", "backpack", "rucksack", "adult", "unisex",
        ],
    },
    "identity_word_aliases": {
        "backpack": "pack",
        "backpacks": "pack",
        "rucksack": "pack",
        "rucksacks": "pack",
    },
}


class GenericIdentityControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_runtime_category(CFG)

    def test_size_token_is_locked_but_budget_number_is_not(self):
        self.assertEqual(
            {"25l"},
            keyword_product_identity_tokens("Patagonia Black Hole 25L review"),
        )
        self.assertEqual(
            set(),
            keyword_product_identity_tokens("best backpacks under £100"),
        )

    def test_related_size_cannot_replace_reviewed_size(self):
        keyword = "Patagonia Black Hole 25L review"
        self.assertTrue(
            reviewed_product_matches_keyword(
                "Patagonia Black Hole Pack 25L", keyword, CFG
            )
        )
        self.assertFalse(
            reviewed_product_matches_keyword(
                "Patagonia Black Hole Pack 32L", keyword, CFG
            )
        )
        self.assertFalse(
            downstream_identity_match(
                "Patagonia Black Hole Pack 32L", keyword, CFG
            )
        )

    def test_whitelist_rescue_obeys_keyword_identity(self):
        selected = pick_specific_from_whitelist(
            [
                "Patagonia Black Hole Pack 32L",
                "Patagonia Black Hole Pack 25L",
            ],
            brand_lexicon=["Patagonia"],
            cfg=CFG,
            keyword="Patagonia Black Hole 25L review",
        )
        self.assertEqual("Patagonia Black Hole Pack 25L", selected)

    def test_related_models_are_allowed_but_ui_or_legal_entities_are_not(self):
        primary = "Patagonia Black Hole Pack 25L"
        self.assertTrue(
            _is_related_profile_product(
                primary, "Patagonia Black Hole Pack 32L", CFG
            )
        )
        self.assertFalse(
            _is_related_profile_product(primary, "Sponsored Products", CFG)
        )
        self.assertFalse(
            _is_related_profile_product(primary, "Patagonia Europe Co Ltd", CFG)
        )

    def test_source_h1_controls_following_claims_until_next_boundary(self):
        dataset = (
            "H1 Gear Review: Example Women’s Backpack 23L\n"
            "Text: It fits under an airplane seat and accepts a small lock.\n"
            "H1 Example Pack 25L Review\n"
            "Text: It has a 25 litre capacity.\n"
        )
        annotated = annotate_source_product_boundaries(dataset)
        self.assertIn(
            "[[SOURCE_BLOCK_PRODUCT: Gear Review: Example Women’s Backpack 23L]]",
            annotated,
        )
        self.assertLess(
            annotated.index("Example Women’s Backpack 23L"),
            annotated.index("accepts a small lock"),
        )
        self.assertLess(
            annotated.index("[[SOURCE_BLOCK_PRODUCT: Example Pack 25L Review]]"),
            annotated.index("25 litre capacity"),
        )

    def test_conflicting_values_are_not_synthesized_into_range(self):
        primary = "Example Pack 25L"
        response = {
            "schema_version": 2,
            "products": [{
                "name": primary,
                "generation_or_style": "unknown",
                "current_status": "current",
                "facts": {
                    "weight": {
                        "canonical_value": "640-680 g",
                        "safe_wording": "The pack weighs 640-680 g.",
                        "confidence": "high",
                        "conflicting_values": [],
                        "forbidden_terms": [],
                        "basis": "Two sources give different weights.",
                        "product_scope": "exact_product",
                        "source_count": 2,
                        "exact_source_count": 0,
                        "requires_attribution": False,
                        "value_status": "explicit_range",
                        "evidence_excerpt": "One source: 640 g; another source: 680 g",
                        "source_type": "specialist_review",
                        "source_date": "unknown",
                    },
                    "dimensions": {
                        "canonical_value": "",
                        "safe_wording": "Designed to fit under airplane seats.",
                        "confidence": "low",
                        "conflicting_values": [],
                        "forbidden_terms": [],
                        "basis": "No dimensions found.",
                        "product_scope": "exact_product",
                        "source_count": 0,
                        "exact_source_count": 0,
                        "requires_attribution": False,
                        "value_status": "unresolved",
                        "evidence_excerpt": "",
                        "source_type": "unknown",
                        "source_date": "unknown",
                    },
                    "durability_history": {
                        "canonical_value": "3-5 years",
                        "safe_wording": "Users report 3-5 years of daily use.",
                        "confidence": "high",
                        "conflicting_values": [],
                        "forbidden_terms": [],
                        "basis": "One source aggregates ownership reports.",
                        "product_scope": "exact_product",
                        "source_count": 2,
                        "exact_source_count": 1,
                        "requires_attribution": False,
                        "value_status": "explicit_range",
                        "evidence_excerpt": "Users report 3-5 years.",
                        "source_type": "specialist_review",
                        "source_date": "unknown",
                    },
                },
            }],
        }
        config = {
            **CFG,
            "canonical_facts": {
                **CFG["canonical_facts"],
                "enabled": True,
                "primary_product_only": True,
                "max_products": 1,
                "dynamic_attributes": True,
                "max_dynamic_attributes": 6,
            },
        }
        with patch(
            "get_response_from_openai.deepseek_generate",
            lambda *_args, **_kwargs: json.dumps(response),
        ):
            profile = build_canonical_product_profile(
                f"H1 {primary}\nText: source evidence",
                "example pack 25l review",
                primary,
                config,
                product_whitelist=[primary],
            )
        facts = profile["products"][0]["facts"]
        self.assertEqual("", facts["weight"]["canonical_value"])
        self.assertEqual("source_conflict", facts["weight"]["value_status"])
        self.assertIn("640 g", facts["weight"]["safe_wording"])
        self.assertIn("680 g", facts["weight"]["safe_wording"])
        self.assertEqual(
            "We could not reliably confirm the dimensions for this exact product.",
            facts["dimensions"]["safe_wording"],
        )
        self.assertTrue(facts["durability_history"]["requires_attribution"])
        self.assertNotIn("3-5 years", facts["durability_history"]["safe_wording"])
        self.assertIn(
            "precise duration is not independently verified",
            facts["durability_history"]["safe_wording"],
        )
        self.assertIn("3-5 years", facts["durability_history"]["forbidden_terms"])

    def test_canonical_profile_coalesces_word_order_and_pack_backpack_aliases(self):
        primary = "Patagonia Black Hole Pack 25L"
        alias = "Patagonia Black Hole 25L Backpack"
        related = "Patagonia Black Hole 32L"
        response = {
            "schema_version": 2,
            "products": [
                {
                    "name": primary,
                    "generation_or_style": "unknown",
                    "current_status": "current",
                    "facts": {
                        "capacity": {
                            "canonical_value": "25 L",
                            "safe_wording": "Has a 25 litre capacity.",
                            "confidence": "high",
                            "conflicting_values": [],
                            "forbidden_terms": [],
                            "basis": "Exact model evidence.",
                            "product_scope": "exact_product",
                            "source_count": 2,
                            "requires_attribution": False,
                            "evidence_excerpt": "25L",
                            "source_type": "specialist_review",
                            "source_date": "unknown",
                        }
                    },
                },
                {
                    "name": alias,
                    "generation_or_style": "unknown",
                    "current_status": "current",
                    "facts": {
                        "weight": {
                            "canonical_value": "640 g",
                            "safe_wording": "One source reports 640 g.",
                            "confidence": "medium",
                            "conflicting_values": [],
                            "forbidden_terms": [],
                            "basis": "Alias source.",
                            "product_scope": "exact_product",
                            "source_count": 1,
                            "requires_attribution": True,
                            "evidence_excerpt": "640g",
                            "source_type": "specialist_review",
                            "source_date": "unknown",
                        }
                    },
                },
                {
                    "name": related,
                    "generation_or_style": "unknown",
                    "current_status": "current",
                    "facts": {
                        "capacity": {
                            "canonical_value": "32 L",
                            "safe_wording": "Has a 32 litre capacity.",
                            "confidence": "high",
                            "conflicting_values": [],
                            "forbidden_terms": [],
                            "basis": "Exact model evidence.",
                            "product_scope": "exact_product",
                            "source_count": 2,
                            "requires_attribution": False,
                            "evidence_excerpt": "32L",
                            "source_type": "specialist_review",
                            "source_date": "unknown",
                        }
                    },
                },
            ],
        }
        prompts = []

        def fake_generate(prompt_text, **_kwargs):
            prompts.append(prompt_text)
            return json.dumps(response)

        config = {
            **CFG,
            "canonical_facts": {
                "enabled": True,
                "primary_product_only": False,
                "max_products": 4,
                "dynamic_attributes": True,
                "max_dynamic_attributes": 6,
                "related_product_min_shared_tokens": 2,
            },
        }
        dataset = (
            f"[PRODUCT] {primary}\n"
            f"[PRODUCT] {alias}\n"
            f"[PRODUCT] {related}\n"
        )
        with patch("get_response_from_openai.deepseek_generate", fake_generate):
            profile = build_canonical_product_profile(
                dataset,
                "patagonia black hole 25l review",
                primary,
                config,
                product_whitelist=[primary, alias, related],
            )

        self.assertEqual(
            [primary, related],
            [item["name"] for item in profile["products"]],
        )
        self.assertEqual(primary, profile["aliases"][alias])
        self.assertIn(f'"{alias}": "{primary}"', prompts[0])

    def test_heading_numbers_require_primary_unattributed_high_confidence_fact(self):
        profile = {
            "primary_product": "Example Pack 25L",
            "products": [{
                "name": "Example Pack 25L",
                "facts": {
                    "device_fit": {
                        "canonical_value": "15-inch laptop",
                        "safe_wording": "Fits a 15-inch laptop.",
                        "confidence": "high",
                        "requires_attribution": False,
                    },
                    "weight": {
                        "canonical_value": "1045 g",
                        "safe_wording": "One reviewer reports 1045 g.",
                        "confidence": "high",
                        "requires_attribution": True,
                    },
                    "dimensions": {
                        "canonical_value": "48 x 28 x 24 cm",
                        "safe_wording": "Measures 48 x 28 x 24 cm.",
                        "confidence": "medium",
                        "requires_attribution": False,
                    },
                },
            }],
        }
        allowed = _canonical_heading_numeric_tokens(profile)
        self.assertIn("25l", allowed)
        self.assertIn("15inch", allowed)
        self.assertNotIn("1045g", allowed)
        self.assertNotIn("48", allowed)


    def test_source_provenance_preserves_exact_product_boundaries(self):
        products = ["Example Pack 25L", "Example Pack 23L"]
        dataset = (
            "H1 Example Pack 25L Review\nThe pack weighs 680 g.\n"
            "H1 Example Pack 23L Review\nA 5'1 reviewer tested a 13-inch laptop."
        )
        ledger = _source_claim_provenance(dataset, products, CFG)
        related = [item for item in ledger if item["product"] == products[1]]
        markers = {marker for item in related for marker in item["distinctive_markers"]}
        self.assertIn("13inch", markers)
        self.assertIn("5'1", markers)

    def test_deterministic_audit_carries_previous_model_context(self):
        primary = "Example Pack 25L"
        related = "Example Pack 32L"
        profile = {
            "primary_product": primary,
            "aliases": {},
            "products": [
                {"name": primary, "facts": {}},
                {"name": related, "facts": {}},
            ],
            "claim_provenance": [{
                "product": related,
                "attribute": "weight",
                "evidence_excerpt": "It weighs 1045 g.",
                "distinctive_markers": ["1045g"],
            }],
        }
        html = ("<p>The Example Pack 32L is larger. One source reports it weighs " "1045 g, making it larger than the 25L version. The extra space remains useful.</p>")
        violations = _deterministic_semantic_claim_violations(
            html, profile, primary, CFG
        )
        self.assertFalse(violations)

    def test_deterministic_audit_blocks_cross_product_user_observation(self):
        primary = "Example Pack 25L"
        related = "Example Pack 23L"
        profile = {
            "primary_product": primary,
            "aliases": {},
            "products": [
                {"name": primary, "facts": {}},
                {"name": related, "facts": {
                    "device_fit": {
                        "evidence_excerpt": "A 13-inch laptop in a soft case required effort.",
                    },
                }},
            ],
            "claim_provenance": [{
                "product": related,
                "attribute": "device_fit",
                "evidence_excerpt": "A 13-inch laptop in a soft case required effort.",
                "distinctive_markers": ["13inch"],
            }],
        }
        html = (
            "<p>One user found that fitting a 13-inch laptop in a soft case "
            "required significant effort.</p>"
        )
        violations = _deterministic_semantic_claim_violations(
            html, profile, primary, CFG
        )
        self.assertEqual("claim_provenance", violations[0]["attribute"])
        self.assertEqual(related, violations[0]["evidence_owner"])

    def test_deterministic_audit_allows_explicit_true_evidence_owner(self):
        primary = "Example Pack 25L"
        related = "Example Pack 23L"
        profile = {
            "primary_product": primary,
            "aliases": {},
            "products": [
                {"name": primary, "facts": {}},
                {"name": related, "facts": {}},
            ],
            "claim_provenance": [{
                "product": related,
                "attribute": "device_fit",
                "evidence_excerpt": "A 13-inch laptop required effort.",
                "distinctive_markers": ["13inch"],
            }],
        }
        html = "<p>For the Example Pack 23L, a 13-inch laptop required effort.</p>"
        violations = _deterministic_semantic_claim_violations(
            html, profile, primary, CFG
        )
        self.assertFalse(violations)

    def test_deterministic_audit_blocks_superlative_with_missing_value(self):
        primary = "Example Pack 25L"
        profile = {
            "primary_product": primary,
            "aliases": {},
            "products": [
                {"name": primary, "facts": {"weight": {
                    "canonical_value": "680 g", "value_status": "confirmed",
                    "product_scope": "exact_product",
                }}},
                {"name": "Example Pack 32L", "facts": {"weight": {
                    "canonical_value": "1045 g", "value_status": "confirmed",
                    "product_scope": "exact_product",
                }}},
                {"name": "Example Pack 23L", "facts": {"weight": {
                    "canonical_value": "", "value_status": "unresolved",
                    "product_scope": "exact_product",
                }}},
            ],
        }
        violations = _deterministic_semantic_claim_violations(
            "<p>The 25L remains the lightest option.</p>", profile, primary, CFG
        )
        self.assertEqual("incomplete_superlative", violations[0]["attribute"])


    def test_semantic_audit_discards_self_declared_non_violation(self):
        primary = "Example Pack 25L"
        profile = {
            "primary_product": primary,
            "aliases": {},
            "products": [{"name": primary, "facts": {}}],
        }
        response = {"violations": [{
            "product": primary,
            "attribute": "weight",
            "passage": "The Example Pack 25L weighs 640 g.",
            "reason": "No violation. This rounding is acceptable.",
            "repair": "No repair needed.",
        }]}
        with patch(
            "get_response_from_openai.deepseek_generate",
            return_value=json.dumps(response),
        ):
            violations, _report = audit_semantic_claim_consistency(
                "<p>The Example Pack 25L weighs 640 g.</p>",
                profile,
                "example pack 25l review",
                primary,
                {"semantic_fact_audit": {"enabled": True}},
            )
        self.assertFalse(violations)
    def test_deterministic_editorial_fact_controls(self):
        primary = "Example Pack 25L"
        profile = {
            "primary_product": primary,
            "aliases": {},
            "products": [{"name": primary, "facts": {}}],
        }
        html = (
            "<h2>Water-Resistant Shell and Exposed Pockets</h2>"
            "<p>The Example Pack 25L fits a 15-inch laptop for everyday travel.</p>"
            "<p>At 680 g, this pack weighs 680 g.</p>"
            "<p>This pack is built to outlast the competition.</p>"
            "<p>The Example Pack 25L works for most airlines.</p>"
        )
        violations = _deterministic_semantic_claim_violations(
            html, profile, primary, CFG
        )
        attributes = {item["attribute"] for item in violations}
        self.assertTrue({
            "section_lead_mismatch",
            "repeated_measurement",
            "unsupported_comparative",
            "universal_compatibility",
        }.issubset(attributes))

    def test_aligned_section_lead_is_not_flagged(self):
        primary = "Example Pack 25L"
        profile = {
            "primary_product": primary,
            "aliases": {},
            "products": [{"name": primary, "facts": {}}],
        }
        html = (
            "<h2>Water-Resistant Shell and Exposed Pockets</h2>"
            "<p>The water-resistant shell sheds rain, while exposed pockets "
            "remain less protected.</p>"
        )
        violations = _deterministic_semantic_claim_violations(
            html, profile, primary, CFG
        )
        self.assertNotIn(
            "section_lead_mismatch",
            {item["attribute"] for item in violations},
        )
    def test_context_recognizes_reordered_equivalent_related_name(self):
        primary = "Patagonia Black Hole Pack 25L"
        related = "Patagonia Women's Blackhole Backpack 23L"
        profile = {
            "primary_product": primary,
            "aliases": {},
            "products": [
                {"name": primary, "facts": {}},
                {"name": "Patagonia Black Hole 32L", "facts": {}},
                {"name": related, "facts": {}},
            ],
            "claim_provenance": [{
                "product": related,
                "attribute": "device_fit",
                "evidence_excerpt": "It fits a 13-inch laptop snugly.",
                "distinctive_markers": ["13inch"],
            }],
        }
        html = (
            "<p>The Patagonia Black Hole 32L provides more room.</p>"
            "<p>The Patagonia Black Hole 23L (Women's) has a shorter torso. "
            "One source reports it fits a 13-inch laptop snugly.</p>"
        )
        violations = _deterministic_semantic_claim_violations(
            html, profile, primary, CFG
        )
        self.assertFalse(violations)

    def test_repeated_adjectival_measurement_has_deterministic_fallback(self):
        passage = (
            "The sleeve fits most 15-inch laptops, though one source advises "
            "testing a 15-inch device before purchase."
        )
        html, repaired = _deterministic_repair_repeated_measurements(
            f"<p>{passage}</p>",
            [{
                "attribute": "repeated_measurement",
                "passage": passage,
                "reason": "The measurement '15inch' is stated twice in one sentence.",
            }],
        )
        self.assertEqual(1, len(repaired))
        self.assertEqual(1, html.count("15-inch"))
        self.assertIn("testing a device before purchase", html)
    def test_h2_lead_can_align_with_child_heading_topics(self):
        primary = "Example Pack 25L"
        profile = {
            "primary_product": primary,
            "aliases": {},
            "products": [{"name": primary, "facts": {}}],
        }
        html = (
            "<h2>Travel and Practical Features</h2>"
            "<p>The shell is water-resistant and the zippers have no dedicated locks.</p>"
            "<h3>Water Resistance and Security Features</h3>"
            "<p>Details.</p>"
            "<h3>Airline Compatibility and Carry-On Suitability</h3>"
            "<p>Details.</p>"
        )
        violations = _deterministic_semantic_claim_violations(
            html, profile, primary, CFG
        )
        self.assertNotIn(
            "section_lead_mismatch",
            {item["attribute"] for item in violations},
        )

    def test_required_attribution_has_deterministic_fallback(self):
        passage = "At 680 g, the Example Pack 25L is lightweight."
        html, repaired = _deterministic_repair_required_attribution(
            f"<p>{passage}</p>",
            [{
                "attribute": "weight",
                "passage": passage,
                "reason": "The fact has medium confidence and requires attribution.",
                "repair": "Add attribution.",
            }],
        )
        self.assertEqual({passage}, repaired)
        self.assertIn("One source reports: At 680 g", html)
    def test_semantic_conflict_uses_exact_products_safe_wording(self):
        profile = {
            "products": [
                {"name": "Example Pack 25L", "facts": {"load_support": {
                    "safe_wording": "One source reports: Comfortable up to about 10 kg.",
                }}},
                {"name": "Example Pack 32L", "facts": {"load_support": {
                    "safe_wording": "One source reports: Comfortable up to about 9 kg.",
                }}},
            ],
        }
        first = "The 25L's load support is not independently confirmed."
        second = "A reviewer found 10 kg comfortable in the 32L."
        html, repaired = _deterministic_repair_with_canonical_safe_wording(
            f"<p>{first}</p><p>{second}</p>",
            [
                {
                    "product": "Example Pack 25L",
                    "attribute": "load_support",
                    "passage": first,
                    "reason": "This contradicts the canonical profile.",
                },
                {
                    "product": "Example Pack 32L",
                    "attribute": "load_support",
                    "passage": second,
                    "reason": "The 10 kg observation belongs to the 25L.",
                },
            ],
            profile,
        )
        self.assertEqual({first, second}, repaired)
        self.assertIn("up to about 10 kg", html)
        self.assertIn("up to about 9 kg", html)
        self.assertNotIn("not independently confirmed", html)

    def test_safe_wording_preserves_required_attribution(self):
        passage = "The pack has no hip belt; comfortable up to about 10 kg."
        profile = {
            "products": [{"name": "Example Pack 25L", "facts": {"load_support": {
                "safe_wording": "Comfortable up to about 10 kg; no hip belt.",
                "requires_attribution": True,
            }}}],
        }
        html, repaired = _deterministic_repair_with_canonical_safe_wording(
            f"<p>{passage}</p>",
            [{
                "product": "Example Pack 25L",
                "attribute": "load_support",
                "passage": passage,
                "reason": "The canonical profile requires attribution.",
                "repair": "Rephrase to 'Comfortable up to about 10 kg; no hip belt.'",
            }],
            profile,
        )
        self.assertEqual({passage}, repaired)
        self.assertIn(
            "One source reports: Comfortable up to about 10 kg; no hip belt.",
            html,
        )

    def test_semantic_audit_deduplicates_and_respects_attributed_observations(self):
        primary = "Example Pack 25L"
        profile = {
            "primary_product": primary,
            "aliases": {},
            "products": [{"name": primary, "facts": {}}],
        }
        load = "The pack carries an intended load range."
        observation = "One reviewer found the bag suitable for an overhead locker."
        disclaimer = "Exact dimensions are unconfirmed, so check airline rules."
        response = {"violations": [
            {"product": primary, "attribute": "load_support", "passage": load, "reason": "Unsupported range.", "repair": "Remove it."},
            {"product": primary, "attribute": "load_support", "passage": load, "reason": "The range is unsupported.", "repair": "Remove it."},
            {"product": primary, "attribute": "dimensions", "passage": observation, "reason": "A single observation is not universal.", "repair": "Keep it attributed."},
            {"product": primary, "attribute": "dimensions", "passage": disclaimer, "reason": "This disclaimer is acceptable.", "repair": "Keep the disclaimer."},
        ]}
        html = f"<p>{load}</p><p>{observation}</p><p>{disclaimer}</p>"
        with patch(
            "get_response_from_openai.deepseek_generate",
            return_value=json.dumps(response),
        ):
            violations, _report = audit_semantic_claim_consistency(
                html,
                profile,
                "example pack 25l review",
                primary,
                {"semantic_fact_audit": {"enabled": True, "deterministic_checks": False}},
            )
        self.assertEqual(1, len(violations))
        self.assertEqual("load_support", violations[0]["attribute"])
    def test_explicit_omit_instruction_removes_only_cited_passage(self):
        passage = "This product is designed with travel in mind."
        retained = "Its exact dimensions are unconfirmed."
        html, repaired = _deterministic_remove_explicit_omissions(
            f"<p>{passage}</p><p>{retained}</p>",
            [{
                "attribute": "dimensions",
                "passage": passage,
                "repair": "Omit or replace with a supported feature.",
            }],
        )
        self.assertEqual({passage}, repaired)
        self.assertNotIn(passage, html)
        self.assertIn(retained, html)
        self.assertNotIn("<p></p>", html)
if __name__ == "__main__":
    unittest.main()
