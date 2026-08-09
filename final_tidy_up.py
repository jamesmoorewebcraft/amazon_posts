import re
import csv
import json
import sys
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString

def read_keyword_and_country():
    try:
        with open("config/current_keyword.csv", "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                if not row or all(not c.strip() for c in row):
                    continue  # skip blank lines
                parts = [c.strip() for c in row]

                # Skip a header row if present
                if len(parts) >= 2 and parts[0].lower() in {"keyword", "kw"} and parts[1].lower() in {"country", "co"}:
                    continue

                # Accept 2 or 3+ columns; ignore extras like 'site'
                if len(parts) >= 2:
                    return parts[0], parts[1].upper()
    except Exception as e:
        print(f"[ERROR] Failed to read current keyword: {e}")
    return "", ""



def read_category_from_current_keyword() -> str:
    try:
        with open("config/current_keyword.csv", "r", encoding="utf-8") as file:
            for row in csv.reader(file):
                if not row or all(not str(cell).strip() for cell in row):
                    continue
                if str(row[0]).strip().lower() in {"keyword", "kw"}:
                    continue
                return str(row[3]).strip() if len(row) >= 4 else ""
    except Exception:
        pass
    return ""


def _deep_merge(*parts):
    result = {}
    for part in parts:
        if not isinstance(part, dict):
            continue
        for key, value in part.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = _deep_merge(result[key], value)
            else:
                result[key] = value
    return result


def load_tidy_config(category: str) -> dict:
    path = Path("config/category_config.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    selected = {}
    for key, value in data.items():
        if key.casefold() == (category or "").casefold():
            selected = value
            break
    return _deep_merge(data.get("default", {}), selected)


def remove_redundant_leading_strong_headings(soup: BeautifulSoup, cfg: dict) -> int:
    """Remove standalone bold pseudo-headings left inside a paragraph."""
    controls = cfg.get("final_cleanup") or {}
    if controls.get("remove_redundant_strong_headings", True) is False:
        return 0
    maximum_words = int(controls.get("strong_heading_max_words", 12))
    removed = 0
    for paragraph in soup.find_all("p"):
        first = next(
            (
                child for child in paragraph.children
                if not isinstance(child, NavigableString) or str(child).strip()
            ),
            None,
        )
        if getattr(first, "name", None) != "strong":
            continue
        label = re.sub(r"\s+", " ", first.get_text(" ", strip=True)).strip()
        if (
            not label
            or label.casefold().startswith(("q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9"))
            or len(label.split()) > maximum_words
            or re.search(r"[.!?]$", label)
        ):
            continue
        following = first.next_sibling
        # Generated subsection remnants are standalone bold lines separated from
        # the body by blank lines. Inline bold lead-ins remain untouched.
        if not isinstance(following, NavigableString) or not re.match(
            r"^\s*(?:\r?\n\s*){2,}",
            str(following),
        ):
            continue
        following.replace_with(re.sub(r"^\s+", "", str(following)))
        first.decompose()
        removed += 1
    return removed


def normalize_editorial_capitals(soup: BeautifulSoup, cfg: dict) -> int:
    """Sentence-case unexplained multiword capitals in editorial text nodes."""
    controls = cfg.get("final_cleanup") or {}
    if controls.get("normalize_multiword_capitals", True) is False:
        return 0
    preserve = {
        str(value).upper()
        for value in (controls.get("capital_preserve_tokens") or [])
        if str(value).strip()
    }
    pattern = re.compile(r"\b[A-Z]{3,}(?:\s+[A-Z]{3,})+\b")
    changed = 0

    def replace_phrase(match):
        nonlocal changed
        changed += 1
        return " ".join(
            word if word.upper() in preserve else word.lower()
            for word in match.group(0).split()
        )

    for node in list(soup.find_all(string=True)):
        parent_name = getattr(node.parent, "name", "")
        if parent_name in {"script", "style", "code", "pre", "h1", "h2", "h3", "h4"}:
            continue
        revised = pattern.sub(replace_phrase, str(node))
        if revised != str(node):
            node.replace_with(revised)
    return changed


def _replace_unverified_zipper_sentences(text: str) -> tuple[str, int]:
    """Replace an unsupported zipper claim at sentence scope, never as a phrase."""
    fallback = (
        "Available sources did not reliably confirm this model's zipper "
        "performance in sustained heavy rain, so sensitive equipment may "
        "still benefit from additional protection."
    )
    marker = (
        r"(?:\bwater\s+can\s+enter\s+through\s+(?:the\s+)?zippers?\b|"
        r"\b(?:the\s+)?zippers?\s+remain\s+the\s+(?:primary|main)\s+entry\s+point\b|"
        r"\bespecially\s+around\s+(?:the\s+)?zippers?\b|"
        r"\bavailable\s+sources\s+did\s+not\s+reliably\s+confirm\s+this\s+model'?s\s+zipper\s+performance\b)"
    )
    sentence_rx = re.compile(
        r"(?P<leading>(?:^|(?<=[.!?])\s*))"
        r"(?P<sentence>[^.!?]*?" + marker
        + r"[^.!?]*(?:[.!?](?:\s+(?:during|in|for|with|and|but|so)\b[^.!?]*[.!?])?|$))",
        re.I,
    )
    normalized_fallback = re.sub(r"\s+", " ", fallback).strip().casefold()
    changes = 0

    def replace(match):
        nonlocal changes
        sentence = re.sub(r"\s+", " ", match.group("sentence")).strip()
        if sentence.casefold() == normalized_fallback:
            return match.group(0)
        changes += 1
        return match.group("leading") + fallback

    return sentence_rx.sub(replace, str(text or "")), changes


def _repair_device_storage_section_leads(
    soup: BeautifulSoup,
    record: dict,
    primary_product: str,
) -> int:
    """Restore a missing laptop/tablet lead only from confirmed canonical facts."""
    facts = (record or {}).get("facts") or {}
    device = facts.get("device_fit") or {}
    features = facts.get("feature_presence") or {}
    device_value = str(device.get("canonical_value") or "").strip()
    if re.search(r"\b\d+(?:\.\d+)?[- ]inch laptop$", device_value, re.I):
        device_value += "s"
    feature_text = " ".join(
        str(features.get(key) or "")
        for key in ("canonical_value", "safe_wording", "evidence_excerpt")
    ).casefold()
    if (
        not primary_product
        or not device_value
        or "laptop sleeve" not in feature_text
        or "tablet sleeve" not in feature_text
    ):
        return 0

    repaired = 0
    for heading in soup.find_all("h2"):
        heading_text = re.sub(
            r"\s+", " ", heading.get_text(" ", strip=True)
        ).casefold()
        if not all(term in heading_text for term in ("laptop", "sleeve", "tablet")):
            continue
        lead = heading.find_next_sibling("p")
        if lead is None:
            continue
        lead_text = lead.get_text(" ", strip=True).casefold()
        if all(term in lead_text for term in ("laptop", "sleeve", "tablet")):
            continue
        replacement = soup.new_tag("p")
        replacement.string = (
            f"{primary_product} includes a laptop sleeve and a tablet sleeve, "
            f"and it fits most {device_value}."
        )
        lead.insert_before(replacement)
        repaired += 1
    return repaired


def _unverified_binary_travel_feature_claims(
    html: str,
    canonical_profile: dict,
    primary_product: str,
) -> list[dict]:
    """Find luggage-pass-through claims without an exact-product confirmation."""
    record = next(
        (
            item for item in (canonical_profile or {}).get("products", [])
            if str(item.get("name") or "").casefold()
            == str(primary_product or "").casefold()
        ),
        {},
    )
    supporting_text = " ".join(
        " ".join([
            str(attribute),
            str((fact or {}).get("canonical_value") or ""),
            str((fact or {}).get("safe_wording") or ""),
        ])
        for attribute, fact in (record.get("facts") or {}).items()
        if isinstance(fact, dict)
        and str((fact or {}).get("value_status") or "confirmed").casefold()
        == "confirmed"
    ).casefold()
    support_rx = re.compile(
        r"\b(?:luggage|roller|trolley|suitcase)\b.{0,80}"
        r"\b(?:pass[- ]?through|sleeve|strap|handle)\b",
        re.I,
    )
    if support_rx.search(supporting_text):
        return []
    claim_rx = re.compile(
        r"\b(?:third|rear|back)\s+(?:strap|sleeve)\b.{0,160}"
        r"\b(?:slides?\s+over|roller\s+(?:luggage|suitcase)\s+handle)\b|"
        r"\b(?:luggage|roller|trolley|suitcase)\s+pass[- ]?through\b",
        re.I,
    )
    issues = []
    soup = BeautifulSoup(html or "", "html.parser")
    for node in soup.find_all(["p", "li"]):
        passage = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
        if claim_rx.search(passage):
            issues.append({
                "attribute": "luggage_pass_through",
                "product": primary_product,
                "passage": passage[:500],
                "reason": "No confirmed exact-product luggage pass-through fact.",
            })
    return issues


def apply_last_pass_qa(soup: BeautifulSoup, canonical_profile: dict | None, primary_product: str) -> dict:
    """Fix deterministic final-stage defects without adding claims."""
    report = {"lists": 0, "faq_synced": 0, "orphaned": 0, "source_status": 0, "source_fragments": 0, "lifespan": 0, "zipper": 0, "load_repeat": 0, "device_section_leads": 0}
    # Convert runs of two or more raw hyphen bullets in a paragraph into real lists.
    bullet_run = re.compile(r"(?m)(?:^[ \t]*-\s+[^\r\n]+(?:\r?\n|$)){2,}")
    for paragraph in list(soup.find_all("p")):
        inner = paragraph.decode_contents()
        matches = list(bullet_run.finditer(inner))
        if not matches:
            continue
        for match in reversed(matches):
            items = re.findall(r"(?m)^[ \t]*-\s+([^\r\n]+)", match.group(0))
            inner = inner[:match.start()] + "</p><ul>" + "".join(f"<li>{item.strip()}</li>" for item in items) + "</ul><p>" + inner[match.end():]
            report["lists"] += 1
        fragment = BeautifulSoup(f"<p>{inner}</p>", "html.parser")
        paragraph.replace_with(*list(fragment.contents))
    record = next((x for x in (canonical_profile or {}).get("products", []) if str(x.get("name", "")).casefold() == str(primary_product).casefold()), {})
    water = (record.get("facts") or {}).get("water_protection") or {}
    zipper_supported = "zipper" in " ".join(str(water.get(k) or "") for k in ("canonical_value", "safe_wording", "basis", "evidence_excerpt")).casefold()
    report["device_section_leads"] += _repair_device_storage_section_leads(
        soup, record, primary_product
    )
    seen_load = 0
    for node in list(soup.find_all(string=True)):
        if not isinstance(node, NavigableString) or getattr(node.parent, "name", "") in {"script", "style", "noscript"}:
            continue
        text = str(node)
        text, n = re.subn(r"(?i)\s*they recommend testing in a store if you use a larger device\.\s*", " ", text); report["orphaned"] += n
        text, n = re.subn(r"(?i)\bone source reports:\s+(we could not reliably confirm\b)", r"\1", text); report["source_status"] += n
        for pattern, replacement in (
            (r"(?i)\bthe last daypack you.{0,3}ll ever need to buy\b", "a strong long-term choice for everyday carry"),
            (r"(?i)\bthe last daypack they will need to buy\b", "a strong long-term choice for everyday carry"),
            (r"(?i)\blong lifespan\b", "positive long-term-use reports"),
            (r"(?i)\bthrough years of use\b", "through regular use"),
            (r"(?i)\bis a (?:solid|reliable) choice that will serve you for years\b", "is a strong long-term choice"),
        ):
            text, n = re.subn(pattern, replacement, text); report["lifespan"] += n
        if not zipper_supported:
            text, n = _replace_unverified_zipper_sentences(text)
            report["zipper"] += n
        for pattern, replacement in (
            (r"(?i)\bone source reports:\s*comfortable with loads up to about\s+(\d+(?:\.\d+)?\s*(?:kg|lb|lbs))", r"One source found loads of around \1 manageable"),
            (r"(?i)\bone source reports:\s*limited ventilation;\s*not ideal for sweaty hikes", "A source noted limited ventilation during warm or strenuous use"),
        ):
            text, n = re.subn(pattern, replacement, text)
            report["source_fragments"] += n
        kept = []
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if re.search(r"(?i)\b10\s*kg\b", sentence):
                seen_load += 1
                if seen_load > 1:
                    report["load_repeat"] += 1
                    continue
            kept.append(sentence)
        new_text = " ".join(kept)
        if new_text != str(node):
            node.replace_with(NavigableString(new_text))
    # The visible FAQ is the final editable source of truth; regenerate FAQPage answers from it.
    visible = []
    for block in soup.select("div.faq-qa"):
        p = block.find("p"); strong = p.find("strong") if p else None
        if not p or not strong:
            continue
        br = p.find("br")
        answer_html = "".join(str(x) for x in br.next_siblings) if br else ""
        visible.append((re.sub(r"^Q\d+\.\s*", "", strong.get_text(" ", strip=True)), BeautifulSoup(answer_html, "html.parser").get_text(" ", strip=True) or "?"))
    if visible:
        for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
            try: payload = json.loads(script.string or script.get_text())
            except (TypeError, json.JSONDecodeError): continue
            if payload.get("@type") != "FAQPage": continue
            payload["mainEntity"] = [{"@type":"Question", "name":q, "acceptedAnswer":{"@type":"Answer", "text":a}} for q, a in visible]
            script.clear(); script.append(json.dumps(payload, ensure_ascii=False)); report["faq_synced"] += 1
    return report


def apply_post_tidy_final_qa(
    content: str,
    cfg: dict,
    canonical_profile: dict | None,
    keyword: str,
) -> tuple[str, dict]:
    """Run shared cleanup and deterministic safety gates on the final body."""
    from get_response_from_openai import (
        _deterministic_semantic_claim_violations,
        audit_canonical_conflicts,
    )
    from insert_amazon_links_images import (
        apply_final_generic_editorial_controls,
        reviewed_product_matches_keyword,
    )

    profile = canonical_profile or {}
    primary = str(profile.get("primary_product") or "").strip()
    qa_soup = BeautifulSoup(content or "", "html.parser")
    last_pass_report = apply_last_pass_qa(qa_soup, profile, primary)
    content = str(qa_soup)
    cleaned, cleanup_report = apply_final_generic_editorial_controls(
        content,
        cfg,
        canonical_profile=profile,
        primary_product=primary,
    )
    # Generic editorial controls can still change visible FAQ copy, so sync the
    # structured data once more after every downstream rewrite.
    final_soup = BeautifulSoup(cleaned or "", "html.parser")
    post_control_report = apply_last_pass_qa(final_soup, profile, primary)
    cleaned = str(final_soup)
    for key, value in post_control_report.items():
        last_pass_report[key] = last_pass_report.get(key, 0) + value
    canonical_conflicts, canonical_warnings = audit_canonical_conflicts(
        cleaned, profile
    ) if profile else ([], [])
    deterministic_violations = _deterministic_semantic_claim_violations(
        cleaned, profile, primary, cfg
    ) if profile and primary else []
    binary_travel_claims = _unverified_binary_travel_feature_claims(
        cleaned, profile, primary
    ) if profile and primary else []

    visible_text = BeautifulSoup(cleaned or "", "html.parser").get_text(" ", strip=True)
    identity_issues = []
    if (cfg.get("canonical_facts") or {}).get("require_profile", True) and not profile:
        identity_issues.append("missing_canonical_profile")
    if profile and not primary:
        identity_issues.append("missing_primary_product")
    if primary and primary.casefold() not in visible_text.casefold():
        identity_issues.append("reviewed_product_missing_from_final_body")
    if primary and keyword and not reviewed_product_matches_keyword(primary, keyword, cfg):
        identity_issues.append("reviewed_product_keyword_mismatch")

    report = {
        "schema_version": 1,
        "cleanup": cleanup_report,
        "last_pass_qa": last_pass_report,
        "canonical_conflicts": canonical_conflicts,
        "canonical_warnings": canonical_warnings,
        "deterministic_semantic_violations": deterministic_violations,
        "identity_issues": identity_issues,
        "unverified_binary_travel_feature_claims": binary_travel_claims,
        "blocked": bool(
            canonical_conflicts or deterministic_violations or identity_issues or binary_travel_claims
        ),
    }
    return cleaned, report

def remove_heading_numbers(content):
    content = re.sub(r'<h2>\s*\d+(\.\d+)*\s+(.*?)</h2>', r'<h2>\2</h2>', content)
    content = re.sub(r'<h3>\s*\d+(\.\d+)*\s+(.*?)</h3>', r'<h3>\2</h3>', content)
    content = re.sub(r'<h4>\s*\d+(\.\d+)*\s+(.*?)</h4>', r'<h4>\2</h4>', content)
    return content

def process_blog_file(
    input_file: Path,
    output_file: Path,
    country: str,
    cfg: dict | None = None,
    keyword: str = "",
    canonical_profile: dict | None = None,
    final_qa_report_path: Path | None = None,
):
    if not input_file.exists():
        print(f"[SKIP] File not found: {input_file}")
        return False

    try:
        lines = input_file.read_text(encoding='utf-8').splitlines()
        cleaned_lines = []
        skip_next = False

        for i in range(len(lines)):
            if skip_next:
                skip_next = False
                continue

            if re.match(r'<h3>.*</h3>', lines[i].strip()) and i + 1 < len(lines) and lines[i + 1].strip() == '<p>[Content Missing]</p>':
                skip_next = True
            else:
                cleaned_lines.append(lines[i])

        content = '\n'.join(cleaned_lines)

        # Remove heading numbers
        content = remove_heading_numbers(content)

        # Formatting cleanup
        content = re.sub(r'<p>####\s*(.*?)</p>', r'<h4>\1</h4>', content)
        content = re.sub(r'<p>###\s*(.*?)</p>', r'<h4>\1</h4>', content)
        content = re.sub(r'^####\s*(.*?)$', r'<h4>\1</h4>', content, flags=re.MULTILINE)
        content = re.sub(r'^###\s*(.*?)$', r'<h4>\1</h4>', content, flags=re.MULTILINE)
        content = re.sub(r'<h4>\s*#\s*(.*?)</h4>', r'<h4>\1</h4>', content)
        content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
        content = re.sub(r'```html|```', '', content)

        # Remove '&lt;' entities
        content = content.replace('&lt;', '')

        # Remove only a genuinely standalone "p" line. Never remove the letter
        # p globally: doing so corrupts <p class="..."> and every </p> closing tag.
        content = re.sub(r'^\s*p\s*$', '', content, flags=re.MULTILINE)

        # Remove malformed empty tags left by older generated files.
        content = re.sub(r'<>|</\s*>', '', content)

        # Normalize invalid nested paragraphs such as <p><p>...</p></p>.
        # Unwrap only the inner paragraph so all text and inline links survive.
        soup = BeautifulSoup(content, "html.parser")
        for paragraph in list(soup.find_all("p")):
            if paragraph.find_parent("p") is not None:
                paragraph.unwrap()
        for paragraph in list(soup.find_all("p")):
            if not paragraph.get_text(" ", strip=True) and not paragraph.find(
                ["img", "a", "br"]
            ):
                paragraph.decompose()
        cleanup_cfg = cfg or {}
        removed_strong = remove_redundant_leading_strong_headings(soup, cleanup_cfg)
        normalized_caps = normalize_editorial_capitals(soup, cleanup_cfg)
        if removed_strong or normalized_caps:
            print(
                f"[CLEANUP] removed strong headings={removed_strong}; "
                f"normalized capital phrases={normalized_caps}"
            )
        content = str(soup)
        final_qa_report = None
        if canonical_profile is not None:
            content, final_qa_report = apply_post_tidy_final_qa(
                content,
                cleanup_cfg,
                canonical_profile,
                keyword,
            )
            if final_qa_report_path is not None:
                final_qa_report_path.write_text(
                    json.dumps(final_qa_report, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            print(
                "[FINAL_QA] "
                f"blocked={final_qa_report.get('blocked', False)}; "
                f"cleanup={final_qa_report.get('cleanup', {})}; "
                f"canonical={len(final_qa_report.get('canonical_conflicts', []))}; "
                f"semantic={len(final_qa_report.get('deterministic_semantic_violations', []))}; "
                f"identity={len(final_qa_report.get('identity_issues', []))}"
            )

            # Print actionable details for any deterministic semantic blockers so
            # a failed batch run can be diagnosed without opening the JSON report.
            semantic_issues = final_qa_report.get(
                "deterministic_semantic_violations",
                [],
            )
            for index, issue in enumerate(semantic_issues, start=1):
                print(
                    f"[FINAL_QA:{index}] "
                    f"{issue.get('attribute', 'unknown')}: "
                    f"{issue.get('reason', 'No reason supplied')}"
                )
                passage = issue.get("passage")
                if passage:
                    print(f"    Passage: {passage}")
                repair = issue.get("repair")
                if repair:
                    print(f"    Repair: {repair}")

        # Remove empty lines that might result
        content = re.sub(r'^\s*$', '', content, flags=re.MULTILINE)

        # Add metadata
        metadata = f"\n\n"
        content = metadata + content

        output_file.write_text(content, encoding='utf-8')
        print(f"[OK] Saved cleaned blog to {output_file}")
        if final_qa_report and final_qa_report.get("blocked"):
            print("[ERROR] Final editorial QA blocked publication; inspect the QA report.")
            return False
        return True
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        return False
def main():
    keyword, country = read_keyword_and_country()
    category = read_category_from_current_keyword()
    cfg = load_tidy_config(category)
    if not keyword or country not in ["US", "UK", "CA"]:
        print("[ERROR] Invalid or missing keyword/country in current_keyword.csv.")
        return

    safe_keyword = keyword.replace(" ", "_")
    safe_keyword_country = f"{safe_keyword}_{country}"
    output_dir = Path("output") / safe_keyword_country

    input_path = output_dir / f"processed_blog_final_updated_{country}.txt"
    output_path = output_dir / f"processed_blog_final_{country}.txt"

    print(f"Final tidy-up for: {keyword} ({country})")
    canonical_profile_path = output_dir / f"canonical_product_profile_{country}.json"
    try:
        canonical_profile = json.loads(
            canonical_profile_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        print(f"[ERROR] Could not load canonical profile: {exc}")
        canonical_profile = {}

    success = process_blog_file(
        input_path,
        output_path,
        country,
        cfg=cfg,
        keyword=keyword,
        canonical_profile=canonical_profile,
        final_qa_report_path=output_dir / f"final_editorial_qa_{country}.json",
    )
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()