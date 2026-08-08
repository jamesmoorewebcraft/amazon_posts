import unittest

from get_response_from_openai import (
    audit_canonical_conflicts,
    find_unresolved_canonical_conflicts,
)


PRODUCT = "Patagonia Black Hole 25L"
PROFILE = {
    "products": [
        {
            "name": PRODUCT,
            "facts": {
                "water_protection": {
                    "canonical_value": "water-resistant",
                    "safe_wording": (
                        "The backpack is water-resistant, not waterproof; it repels "
                        "rain and moisture but is not submersible and water can enter "
                        "through zippers in heavy rain."
                    ),
                    "conflicting_values": ["waterproof", "weatherproof"],
                    "forbidden_terms": ["waterproof", "fully waterproof"],
                }
            },
        }
    ]
}


class CanonicalConflictAuditTests(unittest.TestCase):
    def audit(self, text):
        return audit_canonical_conflicts(f"<p>{PRODUCT} {text}</p>", PROFILE)

    def test_negated_conflicting_term_is_warning_not_blocker(self):
        blocking, warnings = self.audit(
            "is water-resistant, not waterproof. Water can enter through its zippers."
        )
        self.assertEqual([], blocking)
        self.assertEqual("negated", warnings[0]["passages"][0]["reason"])

    def test_lack_of_conflicting_feature_is_warning_not_blocker(self):
        profile = {
            "products": [{
                "name": PRODUCT,
                "facts": {"access_or_security": {
                    "canonical_value": "no anti-theft features",
                    "safe_wording": "The backpack has no anti-theft features.",
                    "conflicting_values": [],
                    "forbidden_terms": ["anti-theft"],
                }},
            }]
        }
        html = f"<p>{PRODUCT} has a lack of anti-theft features in its front pockets.</p>"
        blocking, warnings = audit_canonical_conflicts(html, profile)
        self.assertEqual([], blocking)
        self.assertEqual("negated", warnings[0]["passages"][0]["reason"])

    def test_approved_safe_wording_is_trusted(self):
        safe_wording = PROFILE["products"][0]["facts"]["water_protection"]["safe_wording"]
        blocking, warnings = self.audit(safe_wording)
        self.assertEqual([], blocking)
        self.assertTrue(warnings)
        self.assertTrue(all(
            passage["reason"] == "approved_safe_wording"
            for passage in warnings[0]["passages"]
        ))

    def test_question_and_negated_answer_do_not_block(self):
        blocking, warnings = self.audit(
            "Is the backpack waterproof? It is water-resistant, not waterproof."
        )
        self.assertEqual([], blocking)
        reasons = {passage["reason"] for passage in warnings[0]["passages"]}
        self.assertEqual({"question", "negated"}, reasons)

    def test_faq_json_ld_question_does_not_block(self):
        html = (
            '<script type="application/ld+json">'
            '{"name":"Is the Patagonia Black Hole 25L waterproof?",'
            '"text":"It is water-resistant, not waterproof."}'
            '</script>'
        )
        blocking, warnings = audit_canonical_conflicts(html, PROFILE)
        self.assertEqual([], blocking)
        self.assertTrue(warnings)

    def test_affirmative_conflict_blocks(self):
        blocking, warnings = self.audit("is waterproof and suitable for prolonged rain.")
        self.assertEqual("water_protection", blocking[0]["attribute"])
        self.assertIn("waterproof", blocking[0]["values"])
        self.assertEqual([], warnings)

    def test_positive_claim_after_negated_sentence_still_blocks(self):
        blocking, warnings = self.audit(
            "is not waterproof. Despite that caveat, the backpack is waterproof."
        )
        self.assertTrue(blocking)
        self.assertTrue(warnings)

    def test_resolved_source_comparison_is_warning(self):
        profile = {
            "products": [{
                "name": PRODUCT,
                "facts": {"weight": {
                    "canonical_value": "650 g",
                    "safe_wording": "The backpack weighs approximately 650 g.",
                    "conflicting_values": ["680 g"],
                    "forbidden_terms": [],
                }},
            }]
        }
        html = (
            f"<p>{PRODUCT}: one reviewer listed 680 g, but the current "
            "manufacturer specification is 650 g.</p>"
        )
        blocking, warnings = audit_canonical_conflicts(html, profile)
        self.assertEqual([], blocking)
        self.assertEqual(
            "resolved_source_comparison",
            warnings[0]["passages"][0]["reason"],
        )

    def test_value_contained_in_canonical_range_is_not_a_conflict(self):
        profile = {
            "products": [{
                "name": PRODUCT,
                "facts": {"load_support": {
                    "canonical_value": "Comfortable up to ~9-10 kg",
                    "safe_wording": "Comfortable with loads up to about 9-10 kg.",
                    "conflicting_values": ["10KG", ">10kg"],
                    "forbidden_terms": [],
                }},
            }]
        }
        html = f"<p>{PRODUCT} is comfortable with loads up to about 9-10 kg.</p>"
        blocking, warnings = audit_canonical_conflicts(html, profile)
        self.assertEqual([], blocking)
        self.assertEqual([], warnings)
    def test_backward_compatible_wrapper_returns_only_blockers(self):
        html = f"<p>{PRODUCT} is water-resistant, not waterproof.</p>"
        self.assertEqual([], find_unresolved_canonical_conflicts(html, PROFILE))


    def test_related_observation_before_table_header_is_not_primary_conflict(self):
        primary = "Patagonia Black Hole Pack 25L"
        related = "Patagonia Women's Blackhole Backpack 23L"
        profile = {
            "products": [
                {
                    "name": primary,
                    "facts": {"ventilation": {
                        "canonical_value": "limited",
                        "safe_wording": "The back panel offers limited ventilation.",
                        "conflicting_values": ["breathable"],
                        "forbidden_terms": [],
                    }},
                },
                {"name": related, "facts": {}},
            ],
        }
        html = (
            "The Patagonia Black Hole 23L (Women's) has a shorter torso. "
            "One source describes its back panel as breathable and padded.\n"
            "<table><thead><tr><th>Feature</th>"
            f"<th>{primary}</th><th>{related}</th></tr></thead></table>"
        )
        blocking, _warnings = audit_canonical_conflicts(html, profile)
        self.assertEqual([], blocking)
if __name__ == "__main__":
    unittest.main()
