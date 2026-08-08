import unittest

from get_response_from_openai import _looks_like_ui_phrase, configure_runtime_category
from insert_amazon_links_images import (
    _heading_has_unsupported_primary_claim,
    apply_final_generic_editorial_controls,
    build_config_for_category,
    load_category_db,
)


PROFILE = {
    "primary_product": "Example Pack 25L",
    "products": [{
        "name": "Example Pack 25L",
        "facts": {
            "water_protection": {
                "canonical_value": "water-resistant",
                "safe_wording": "The pack is water-resistant but not fully waterproof.",
                "evidence_excerpt": "TPU laminate sheds light rain.",
            },
            "ventilation": {
                "canonical_value": "limited",
                "safe_wording": (
                    "One source reports: The back panel offers limited ventilation; "
                    "it may feel warm during intense activity."
                ),
                "evidence_excerpt": "limited airflow",
            },            "load_support": {
                "canonical_value": "up to 10 kg",
                "safe_wording": "One reviewer reports loads around 10 kg are manageable; there is no hip belt.",
                "evidence_excerpt": "loads around 10 kg remain comfortable",
                "requires_attribution": True,
            },
        },
    }],
}


class DownstreamTrustControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_runtime_category({
            "canonical_facts": {
                "editorial_product_name_prefixes": ["the bottom line", "our verdict"],
            }
        })

    def test_editorial_bottom_line_is_not_a_product(self):
        self.assertTrue(_looks_like_ui_phrase("The Bottom Line: Example Pack 25L"))

    def test_heading_claim_requires_exact_product_support(self):
        controls = {
            "heading_claim_guard": {
                "enabled": True,
                "risk_groups": [{
                    "name": "water_ingress",
                    "patterns": [r"\bseep\w*\b", r"\bleak\w*\b"],
                    "support_terms": ["seep", "leak", "water can enter"],
                }],
            }
        }
        unsupported, group = _heading_has_unsupported_primary_claim(
            "TPU Laminate Sheds Rain but Zippers Seep", PROFILE, controls
        )
        self.assertTrue(unsupported)
        self.assertEqual("water_ingress", group)

        supported_profile = {
            **PROFILE,
            "products": [{
                **PROFILE["products"][0],
                "facts": {
                    **PROFILE["products"][0]["facts"],
                    "zipper_ingress": {
                        "safe_wording": "One source reports that water can enter through the zippers."
                    },
                },
            }],
        }
        unsupported, _ = _heading_has_unsupported_primary_claim(
            "Zippers Can Seep in Rain", supported_profile, controls
        )
        self.assertFalse(unsupported)

    def test_final_controls_fix_trust_mapping_value_and_verdict(self):
        html = (
            "<h2>Who is this for?</h2>"
            "<p>We have tested its durability and features to see whether it works.</p>"
            "<h2>TPU Laminate Sheds Rain</h2>"
            "<p>The Example Pack 25L fits most 15-inch laptops.</p>"
            "<p>The TPU laminate sheds light rain, but the pack is not waterproof.</p>"
            "<h2>Security</h2>"
            "<p>One source reports: The pack has no dedicated anti-theft protection.</p>"
            "<h2>Value for Money and Final Verdict</h2>"
            "<p>It balances quality and function without breaking the bank.</p>"
            "<h2>Frequently Asked Questions</h2>"
        )
        output, report = apply_final_generic_editorial_controls(
            html,
            {"final_trust_controls": {"enabled": True}},
            canonical_profile=PROFILE,
            primary_product="Example Pack 25L",
        )
        self.assertNotIn("We have tested", output)
        self.assertIn("We examined its durability, features, and real-world owner feedback", output)
        self.assertNotIn("fits most 15-inch laptops", output)
        self.assertIn("TPU laminate sheds light rain", output)
        self.assertIn("One source reports that the pack", output)
        self.assertNotIn("without breaking the bank", output)
        self.assertIn("durability central to its long-term value", output)
        self.assertIn("limited ventilation", output)
        self.assertEqual(1, report["misplaced_summaries_removed"])
        self.assertEqual(1, report["verdict_drawbacks_added"])


    def test_final_controls_clean_machine_fragments_and_source_language(self):
        cfg = build_config_for_category("backpacks", load_category_db())
        html = (
            "<h2>Durability and Weight</h2>"
            "<p>The pack has a laptop sleeve, water bottle pockets, and a sternum strap.</p>"
            "<p>Includes laptop sleeve, water bottle pockets, and sternum strap. "
            "One source reports: Weighs 680 g (24 oz). "
            "One reviewer says it has held well through performance without marks or areas of default. "
            "The coating handles scrapes that would damage lesser fabrics.</p>"
        )
        output, report = apply_final_generic_editorial_controls(
            html, cfg, canonical_profile=PROFILE, primary_product="Example Pack 25L"
        )
        self.assertNotIn("Includes laptop sleeve", output)
        self.assertIn("One source lists its weight as 680 g (24 oz)", output)
        self.assertIn("held up well without obvious damage or defects", output)
        self.assertIn("adds resistance to everyday scrapes and abrasion", output)
        self.assertEqual(1, report["raw_feature_fragments_removed"])
        self.assertEqual(1, report["source_fragment_rewrites"])
        self.assertEqual(1, report["malformed_source_rewrites"])
        self.assertEqual(1, report["unsupported_comparative_rewrites"])

    def test_final_controls_remove_invented_security_compensation(self):
        cfg = build_config_for_category("backpacks", load_category_db())
        html = (
            "<h2>Security</h2><p>The pack has no dedicated anti-theft features. "
            "The front pockets sit exposed, though the side flap provides some visual cover "
            "for smaller items. The sternum strap helps keep the bag close to your body "
            "in crowded spaces.</p>"
        )
        output, report = apply_final_generic_editorial_controls(
            html, cfg, canonical_profile=PROFILE, primary_product="Example Pack 25L"
        )
        self.assertIn("no dedicated anti-theft features", output)
        self.assertNotIn("visual cover", output)
        self.assertNotIn("crowded spaces", output)
        self.assertEqual(2, report["unsupported_security_benefits_removed"])

    def test_water_prefix_does_not_validate_waterproof_heading_summary(self):
        cfg = build_config_for_category("backpacks", load_category_db())
        html = (
            "<h2>Weather-Resistant Laminate but Not Waterproof</h2>"
            "<p>The product includes two water bottle pockets and a laptop sleeve.</p>"
            "<p>The laminate sheds light rain, but the product is not fully waterproof.</p>"
        )
        output, report = apply_final_generic_editorial_controls(
            html, cfg, canonical_profile=PROFILE, primary_product="Example Pack 25L"
        )
        self.assertNotIn("water bottle pockets", output)
        self.assertIn("laminate sheds light rain", output)
        self.assertEqual(1, report["misplaced_summaries_removed"])

    def test_shared_qa_removes_raw_material_and_ventilation_duplicates(self):
        cfg = build_config_for_category("backpacks", load_category_db())
        html = (
            "<h2>Materials</h2><p>Made from recycled polyester ripstop with TPU laminate. "
            "The product uses recycled polyester ripstop with a TPU laminate for durability.</p>"
            "<h2>Comfort</h2><p>One source reports: Limited ventilation; not ideal for sweaty hikes. "
            "One source reports ventilation is limited, so it may feel sweaty during activity.</p>"
        )
        output, report = apply_final_generic_editorial_controls(
            html, cfg, canonical_profile=PROFILE, primary_product="Example Pack 25L"
        )
        self.assertNotIn("Made from recycled", output)
        self.assertNotIn("One source reports: Limited ventilation", output)
        self.assertIn("uses recycled polyester", output)
        self.assertIn("ventilation is limited", output)
        self.assertEqual(2, report["near_duplicate_sentences_removed"])

    def test_unique_subjectless_material_fragment_becomes_grammatical(self):
        cfg = build_config_for_category("backpacks", load_category_db())
        output, report = apply_final_generic_editorial_controls(
            "<p>Made from recycled aluminium with a protective surface finish.</p>",
            cfg,
            canonical_profile=PROFILE,
            primary_product="Example Pack 25L",
        )
        self.assertIn("It is made from recycled aluminium", output)
        self.assertEqual(1, report["raw_feature_fragments_rewritten"])

    def test_single_water_word_cannot_validate_security_section_summary(self):
        cfg = build_config_for_category("backpacks", load_category_db())
        html = (
            "<h2>Water Resistance and Security Features</h2>"
            "<p>The product includes two water bottle pockets.</p>"
            "<p>The shell is water-resistant, while no dedicated anti-theft protection is provided.</p>"
        )
        output, report = apply_final_generic_editorial_controls(
            html, cfg, canonical_profile=PROFILE, primary_product="Example Pack 25L"
        )
        self.assertNotIn("water bottle pockets", output)
        self.assertIn("water-resistant", output)
        self.assertEqual(1, report["misplaced_summaries_removed"])

    def test_single_source_measurements_are_attributed_in_prose_and_tables(self):
        cfg = build_config_for_category("backpacks", load_category_db())
        html = (
            "<p>The weight becomes noticeable near its 10 kg comfort threshold.</p>"
            "<table><tr><th>Model</th><th>Load</th></tr>"
            "<tr><td>Example Pack 25L</td><td>10 kg</td></tr></table>"
        )
        output, report = apply_final_generic_editorial_controls(
            html, cfg, canonical_profile=PROFILE, primary_product="Example Pack 25L"
        )
        self.assertNotIn("comfort threshold", output)
        self.assertIn("One reviewer reports loads around 10 kg are manageable", output)
        self.assertIn("Reported manageable around 10 kg", output)
        self.assertEqual(2, report["required_attribution_repairs"])

    def test_vague_table_values_and_quirky_metaphors_are_cleaned(self):
        cfg = build_config_for_category("backpacks", load_category_db())
        html = (
            "<p>It avoids tiny compartments that hold little more than a handful of almonds.</p>"
            "<table><tr><th>Model</th><th>Comfortable Load</th></tr>"
            "<tr><td>Alternative</td><td>Full load</td></tr></table>"
        )
        output, report = apply_final_generic_editorial_controls(
            html, cfg, canonical_profile=PROFILE, primary_product="Example Pack 25L"
        )
        self.assertNotIn("handful of almonds", output)
        self.assertIn("limited practical use", output)
        self.assertIn("Not quantified", output)
        self.assertEqual(1, report["restrained_style_rewrites"])
        self.assertEqual(1, report["vague_table_values_replaced"])

    def test_repeated_single_source_fact_is_kept_once_in_best_section(self):
        cfg = build_config_for_category("backpacks", load_category_db())
        sentence = "One reviewer reports loads around 10 kg are manageable; there is no hip belt."
        html = (
            f"<h2>Materials and Weight</h2><p>{sentence}</p>"
            f"<h2>Comfort and Carrying Performance</h2><p>{sentence}</p>"
        )
        output, report = apply_final_generic_editorial_controls(
            html, cfg, canonical_profile=PROFILE, primary_product="Example Pack 25L"
        )
        self.assertEqual(1, output.count(sentence))
        self.assertEqual(1, report["repeated_canonical_facts_removed"])

    def test_compound_water_bottle_phrase_does_not_validate_weather_summary(self):
        cfg = build_config_for_category("backpacks", load_category_db())
        cfg.setdefault("final_trust_controls", {})["section_alignment_ignored_phrases"] = [
            "water bottle pockets"
        ]
        html = (
            "<h2>Water-Resistant Zippers and Exposed Pockets</h2>"
            "<p>Travel features include twin water bottle pockets and a laptop sleeve.</p>"
            "<p>The shell resists light rain, but water may enter through exposed zippers.</p>"
        )
        output, report = apply_final_generic_editorial_controls(
            html, cfg, canonical_profile=PROFILE, primary_product="Example Pack 25L"
        )
        self.assertNotIn("Travel features include", output)
        self.assertEqual(1, report["misplaced_summaries_removed"])

    def test_related_model_uncertainty_cannot_create_primary_buying_advice(self):
        cfg = build_config_for_category("backpacks", load_category_db())
        profile = {
            **PROFILE,
            "products": PROFILE["products"] + [{
                "name": "Example Pack 23L",
                "facts": {"device_fit": {"canonical_value": "13-inch laptop"}},
            }],
        }
        html = (
            "<p>The primary sleeve fits most 15-inch laptops. "
            "A reviewer of the Example Pack 23L was uncertain whether all 15-inch models fit. "
            "Testing in a shop before purchase is advisable for 15-inch laptop owners.</p>"
        )
        output, report = apply_final_generic_editorial_controls(
            html, cfg, canonical_profile=profile, primary_product="Example Pack 25L"
        )
        self.assertNotIn("uncertain whether", output)
        self.assertNotIn("before purchase is advisable", output)
        self.assertEqual(2, report["cross_model_inferences_removed"])


    def test_varied_attributed_measurement_phrasings_count_as_one_fact(self):
        cfg = build_config_for_category("backpacks", load_category_db())
        html = (
            "<h2>Materials and Weight</h2>"
            "<p>One source reports that the pack is comfortable with loads up to about 10 kg.</p>"
            "<h2>Comfort and Carrying</h2>"
            "<p>One reviewer notes loads of around 10 kg remain manageable.</p>"
            "<h2>Everyday Use</h2>"
            "<p>A source states that loads near 10 kg can be carried comfortably.</p>"
        )
        output, report = apply_final_generic_editorial_controls(
            html, cfg, canonical_profile=PROFILE, primary_product="Example Pack 25L"
        )
        self.assertEqual(1, output.count("10 kg"))
        self.assertEqual(2, report["repeated_canonical_facts_removed"])

    def test_adjacent_ventilation_restatement_is_consolidated(self):
        cfg = build_config_for_category("backpacks", load_category_db())
        controls = cfg.setdefault("final_trust_controls", {})
        controls["canonical_fact_topic_terms"] = {
            **controls.get("canonical_fact_topic_terms", {}),
            "ventilation": ["ventilation", "airflow", "sweaty", "warm"],
        }
        controls["adjacent_attribute_dedupe_attributes"] = ["ventilation"]
        html = (
            "<p>The back panel offers limited ventilation and may feel sweaty during strenuous activity. "
            "The panel sits close to the body, so airflow is limited and it may feel sweaty during strenuous activity.</p>"
        )
        output, report = apply_final_generic_editorial_controls(
            html, cfg, canonical_profile=PROFILE, primary_product="Example Pack 25L"
        )
        self.assertEqual(1, output.casefold().count("sweaty"))
        self.assertEqual(1, report["adjacent_attribute_repetitions_removed"])

    def test_vague_daily_use_table_value_becomes_not_quantified(self):
        cfg = build_config_for_category("backpacks", load_category_db())
        controls = cfg.setdefault("final_trust_controls", {})
        controls["vague_table_values"] = ["daily use"]
        controls["vague_table_value_replacement"] = "Not quantified"
        output, report = apply_final_generic_editorial_controls(
            "<table><tr><th>Model</th><th>Load comfort</th></tr>"
            "<tr><td>Related model</td><td>Daily use</td></tr></table>",
            cfg,
            canonical_profile=PROFILE,
            primary_product="Example Pack 25L",
        )
        self.assertIn("Not quantified", output)
        self.assertEqual(1, report["vague_table_values_replaced"])

    def test_measurement_after_related_model_context_names_its_owner(self):
        cfg = build_config_for_category("backpacks", load_category_db())
        profile = {
            "primary_product": "Example Pack 25L",
            "products": [
                {
                    "name": "Example Pack 25L",
                    "facts": {
                        "weight": {
                            "canonical_value": "640 g",
                            "safe_wording": "One source reports that the pack weighs 640 g.",
                            "evidence_excerpt": "weighs 640 g",
                            "requires_attribution": True,
                        }
                    },
                },
                {
                    "name": "Example Pack 23L",
                    "facts": {
                        "weight": {
                            "canonical_value": "",
                            "safe_wording": "Weight not quantified.",
                        }
                    },
                },
            ],
        }
        html = (
            "<p>The Example Pack 23L is designed for shorter torsos.</p>"
            "One source reports that the pack weighs 640 g."
        )
        output, report = apply_final_generic_editorial_controls(
            html, cfg, canonical_profile=profile, primary_product="Example Pack 25L"
        )
        self.assertIn("reports that Example Pack 25L weighs 640 g", output)
        self.assertEqual(1, report["ambiguous_measurements_scoped"])

    def test_collective_material_claim_is_qualified_when_specs_differ(self):
        cfg = build_config_for_category("backpacks", load_category_db())
        profile = {
            "primary_product": "Example Pack 25L",
            "products": [
                {"name": "Example Pack 25L", "facts": {
                    "material": {"canonical_value": "recycled polyester ripstop"}
                }},
                {"name": "Example Pack 32L", "facts": {
                    "material": {"canonical_value": "300D polyester ripstop"}
                }},
            ],
        }
        output, report = apply_final_generic_editorial_controls(
            "<p>Both models share the same weather-resistant ripstop fabric family.</p>",
            cfg,
            canonical_profile=profile,
            primary_product="Example Pack 25L",
        )
        self.assertIn("specifications vary by model", output)
        self.assertEqual(1, report["collective_material_claims_qualified"])


if __name__ == "__main__":
    unittest.main()