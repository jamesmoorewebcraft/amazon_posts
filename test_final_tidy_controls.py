import unittest

from bs4 import BeautifulSoup

from final_tidy_up import (
    apply_post_tidy_final_qa,
    apply_last_pass_qa,
    normalize_editorial_capitals,
    remove_redundant_leading_strong_headings,
)


CFG = {
    "final_cleanup": {
        "remove_redundant_strong_headings": True,
        "strong_heading_max_words": 12,
        "normalize_multiword_capitals": True,
        "capital_preserve_tokens": ["TPU", "YKK", "DWR"],
    }
}


class FinalTidyControlTests(unittest.TestCase):
    def test_removes_standalone_strong_heading_but_preserves_inline_lead_in(self):
        soup = BeautifulSoup(
            "<h2>Materials and Weight</h2>"
            "<p><strong>Materials, Durability and Weight</strong>\n\n"
            "The shell is recycled polyester.</p>"
            "<p><strong>Practical note:</strong> Keep electronics dry.</p>",
            "html.parser",
        )
        removed = remove_redundant_leading_strong_headings(soup, CFG)
        self.assertEqual(1, removed)
        self.assertNotIn("Materials, Durability and Weight", str(soup))
        self.assertIn("<strong>Practical note:</strong>", str(soup))

    def test_normalizes_multiword_capitals_and_preserves_configured_acronyms(self):
        soup = BeautifulSoup(
            "<p>This design holds MORE GEAR and uses TPU FILM with YKK ZIPPERS.</p>",
            "html.parser",
        )
        changed = normalize_editorial_capitals(soup, CFG)
        self.assertEqual(3, changed)
        text = soup.get_text(" ", strip=True)
        self.assertIn("more gear", text)
        self.assertIn("TPU film", text)
        self.assertIn("YKK zippers", text)


    def test_unverified_zipper_claim_replaces_the_whole_sentence_and_syncs_faq(self):
        profile = {
            "primary_product": "Example Pack 25L",
            "products": [{"name": "Example Pack 25L", "facts": {
                "water_protection": {
                    "canonical_value": "water-resistant",
                    "safe_wording": "The pack is water-resistant but not fully waterproof.",
                }
            }}],
        }
        fallback = (
            "Available sources did not reliably confirm this model's zipper "
            "performance in sustained heavy rain, so sensitive equipment may "
            "still benefit from additional protection."
        )
        soup = BeautifulSoup(
            "<p>The pack is not fully waterproof, but water can enter through "
            "the zippers during heavy downpours.</p>"
            "<div class='faq-qa'><p><strong>Q1. Is it waterproof?</strong><br/>"
            "The pack is water-resistant, but Available sources did not reliably "
            "confirm this model's zipper performance in sustained heavy rain, so "
            "sensitive equipment may still benefit from additional protection. "
            "during heavy downpours.</p></div>"
            "<script type='application/ld+json'>{\"@type\": \"FAQPage\", \"mainEntity\": []}</script>",
            "html.parser",
        )
        report = apply_last_pass_qa(soup, profile, "Example Pack 25L")
        output = str(soup)
        self.assertNotIn("but Available sources", output)
        self.assertNotIn("during heavy downpours", output)
        self.assertEqual(3, output.count(fallback))
        self.assertEqual(2, report["zipper"])
        self.assertIn(fallback, soup.find("script").string)

    def test_device_storage_heading_gets_a_canonical_lead(self):
        profile = {
            "primary_product": "Example Pack 25L",
            "products": [{"name": "Example Pack 25L", "facts": {
                "device_fit": {"canonical_value": "15-inch laptop"},
                "feature_presence": {
                    "canonical_value": "laptop sleeve, tablet sleeve, top pocket"
                },
            }}],
        }
        soup = BeautifulSoup(
            "<h2>Padded 15-Inch Laptop Sleeve and Tablet Pocket</h2>"
            "<p>A zippered top pocket keeps chargers and keys within reach.</p>",
            "html.parser",
        )
        report = apply_last_pass_qa(soup, profile, "Example Pack 25L")
        output = str(soup)
        self.assertIn(
            "Example Pack 25L includes a laptop sleeve and a tablet sleeve, "
            "and it fits most 15-inch laptops.",
            output,
        )
        self.assertLess(output.index("includes a laptop sleeve"), output.index("A zippered top pocket"))
        self.assertEqual(1, report["device_section_leads"])

    def test_raw_canonical_fragments_are_smoothed(self):
        soup = BeautifulSoup(
            "<p>One source reports: Comfortable with loads up to about 10 kg. "
            "One source reports: Limited ventilation; not ideal for sweaty hikes.</p>",
            "html.parser",
        )
        report = apply_last_pass_qa(soup, {}, "")
        output = soup.get_text(" ", strip=True)
        self.assertIn("One source found loads of around 10 kg manageable.", output)
        self.assertIn("A source noted limited ventilation during warm or strenuous use.", output)
        self.assertEqual(2, report["source_fragments"])

    def test_verdict_does_not_promise_a_precise_lifespan(self):
        soup = BeautifulSoup(
            "<p>It is a solid choice that will serve you for years.</p>",
            "html.parser",
        )
        report = apply_last_pass_qa(soup, {}, "")
        self.assertNotIn("serve you for years", str(soup))
        self.assertIn("strong long-term choice", str(soup))
        self.assertEqual(1, report["lifespan"])

    def test_post_tidy_final_qa_runs_cleanup_and_safety_gate(self):
        profile = {
            "primary_product": "Example Pack 25L",
            "products": [{"name": "Example Pack 25L", "facts": {
                "water_protection": {
                    "canonical_value": "water-resistant",
                    "safe_wording": "Water-resistant but not fully waterproof.",
                    "forbidden_terms": ["waterproof"],
                }
            }}],
        }
        cfg = {
            **CFG,
            "canonical_facts": {"require_profile": True},
            "review_identity": {"block_on_mismatch": True},
            "final_trust_controls": {
                "enabled": True,
                "restrained_style_rewrites": [{
                    "pattern": r"\ba handful of almonds\b",
                    "replacement": "very little",
                }],
            },
            "semantic_fact_audit": {"deterministic_checks": True},
        }
        output, report = apply_post_tidy_final_qa(
            "<p>Example Pack 25L has pockets holding a handful of almonds.</p>",
            cfg,
            profile,
            "example pack 25l review",
        )
        self.assertIn("very little", output)
        self.assertFalse(report["blocked"])
        self.assertEqual([], report["identity_issues"])

    def test_post_tidy_final_qa_blocks_affirmative_canonical_conflict(self):
        profile = {
            "primary_product": "Example Pack 25L",
            "products": [{"name": "Example Pack 25L", "facts": {
                "water_protection": {
                    "canonical_value": "water-resistant",
                    "safe_wording": "Water-resistant but not fully waterproof.",
                    "forbidden_terms": ["waterproof"],
                }
            }}],
        }
        cfg = {
            **CFG,
            "canonical_facts": {"require_profile": True},
            "review_identity": {"block_on_mismatch": True},
            "final_trust_controls": {"enabled": True},
            "semantic_fact_audit": {"deterministic_checks": True},
        }
        _output, report = apply_post_tidy_final_qa(
            "<p>Example Pack 25L is waterproof.</p>",
            cfg,
            profile,
            "example pack 25l review",
        )
        self.assertTrue(report["blocked"])
        self.assertTrue(report["canonical_conflicts"])

    def test_unconfirmed_luggage_pass_through_blocks_publication(self):
        profile = {
            "primary_product": "Example Pack 25L",
            "products": [{"name": "Example Pack 25L", "facts": {
                "feature_presence": {
                    "canonical_value": "laptop sleeve and top pocket",
                    "value_status": "confirmed",
                }
            }}],
        }
        cfg = {
            **CFG,
            "canonical_facts": {"require_profile": True},
            "review_identity": {"block_on_mismatch": True},
            "final_trust_controls": {"enabled": True},
            "semantic_fact_audit": {"deterministic_checks": True},
        }
        _output, report = apply_post_tidy_final_qa(
            "<p>Example Pack 25L has a third strap on its back panel that "
            "slides over a roller suitcase handle.</p>",
            cfg,
            profile,
            "example pack 25l review",
        )
        self.assertTrue(report["blocked"])
        self.assertEqual(
            "luggage_pass_through",
            report["unverified_binary_travel_feature_claims"][0]["attribute"],
        )

if __name__ == "__main__":
    unittest.main()
