"""
Script Purpose:
---------------
This script processes blog content to extract, clean, deduplicate, and update product names using DeepSeek's language model.
It aims to identify product mentions, standardize their format, and replace the original mentions in the content with cleaned names.

Input Files:
------------
1. config/current_keyword.csv → Specifies keyword, country, and optionally site
   (e.g., "smartphones,US,https://example.com"). The script ignores the site column.
2. output/{keyword}/content_{country}.txt → Blog content from which product names will be extracted.
3. config/deepseek_api_key.txt       → DeepSeek API key for querying DeepSeek-V4-Pro.

Output Files:
-------------
1. output/{keyword}/extracted_product_names_{country}_1.csv → Cleaned and unique product names.
2. output/{keyword}/content_{country}_updated.txt           → Updated blog content with cleaned names.
3. output/{keyword}/_debug_discarded_product_names.txt      → Log of duplicates removed.
4. output/{keyword}/_debug_original_to_cleaned_map.txt      → Mapping from raw to cleaned names.
5. output/{keyword}/no_products.txt                         → Flag file when nothing valid is found.
6. logs/{keyword}/amend_product_names.log                   → Log file for processing steps and issues.
"""

import json
import logging
import os
from pathlib import Path
import re
from difflib import SequenceMatcher

import pandas as pd
from openai import OpenAI as DeepSeekClient
from internal_links import log_deepseek_usage

CURRENT_KEYWORD_FILE = Path("config/current_keyword.csv")
DEEPSEEK_API_KEY_FILE = Path("config/deepseek_api_key.txt")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
client = None
CATEGORY_CONFIG_CACHE = {}



def _config_value_to_strings(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    return []


def _iter_category_config_files():
    config_dir = Path("config")
    if not config_dir.exists():
        return []

    exact = config_dir / "category_config.json"
    others = sorted(config_dir.glob("category_config_*.json"))

    files = []
    if exact.exists():
        files.append(exact)
    for candidate in others:
        if candidate not in files:
            files.append(candidate)
    return files


def _score_category_match(keyword: str, topic_key: str, category_name: str, rules: dict) -> int:
    keyword_norm = normalize_text_for_search(keyword)
    keyword_safe = _topic_key_from_keyword(keyword)
    score = 0

    category_tokens = []
    for source in (
        category_name,
        rules.get("topic_key"),
        rules.get("category"),
        *(rules.get("include_keywords") or []),
        *(rules.get("title_required_terms") or []),
        *(rules.get("generic_tails") or []),
        *(rules.get("category_core_tokens") or []),
        *(rules.get("query_must_include") or []),
    ):
        category_tokens.extend(_config_value_to_strings(source))

    seen = set()
    ordered_tokens = []
    for token in category_tokens:
        token_norm = normalize_text_for_search(token)
        if token_norm and token_norm not in seen:
            seen.add(token_norm)
            ordered_tokens.append(token_norm)

    category_name_norm = normalize_text_for_search(category_name)
    topic_key_norm = normalize_text_for_search(str(rules.get("topic_key") or ""))

    if category_name == topic_key:
        score += 200
    if topic_key_norm and normalize_text_for_search(topic_key) == topic_key_norm:
        score += 200
    if keyword_safe == category_name:
        score += 150
    if topic_key_norm and keyword_safe == _topic_key_from_keyword(topic_key_norm):
        score += 150
    if category_name_norm and category_name_norm in keyword_norm:
        score += 80
    if topic_key_norm and topic_key_norm.replace("_", " ") in keyword_norm:
        score += 80

    for token in ordered_tokens:
        if token and token in keyword_norm:
            score += max(8, min(len(token), 40))

    return score


MODEL_TOKEN_RX = re.compile(r"\b(?:[A-Z]{1,4}[A-Z0-9-]{3,}|\d+[A-Z][A-Z0-9-]{0,})\b")


def setup_logging(safe_keyword, country):
    safe_keyword_country = f"{safe_keyword}_{country}"
    log_dir = Path("logs") / safe_keyword_country
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "amend_product_names.log"

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=log_file,
        filemode="a",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("==== Started amend_product_names ====")


def read_current_keyword_and_country():
    try:
        with open(CURRENT_KEYWORD_FILE, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            if not line:
                return None, None
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                keyword = parts[0]
                country = parts[1].upper()
                return keyword, country
    except Exception as e:
        logging.error(f"Failed to read current_keyword.csv: {e}")
    return None, None


def query_deepseek(prompt, attempt_label="initial"):
    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2500,
            temperature=0,
            extra_body={"thinking": {"type": "disabled"}},
        )
        log_deepseek_usage(
            response,
            label=f"amend_product_names:{attempt_label}",
            requested_model=DEEPSEEK_MODEL,
        )
        content = response.choices[0].message.content
        logging.debug(f"{attempt_label.upper()} DeepSeek response: {repr(content)}")
        return content
    except Exception as e:
        logging.error(f"{attempt_label.upper()} DeepSeek request failed: {e}")
        return ""


def normalize_text_for_search(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = (
        text.replace("“", '"').replace("”", '"')
        .replace("‘", "'").replace("’", "'")
        .replace("–", "-").replace("‐", "-").replace("—", "-")
        .replace("″", '"')
        .replace("\u00a0", " ")
        .replace("\t", " ")
        .strip()
    )
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_text(s: str) -> str:
    return normalize_text_for_search(s)


def make_safe_keyword(keyword: str, country: str) -> str:
    k = (keyword or "").strip()
    if country:
        c = re.escape(country.strip())
        k = re.sub(rf"\s*\(\s*{c}\s*\)\s*$", "", k, flags=re.I)
        k = re.sub(rf"\s+{c}\s*$", "", k, flags=re.I)

    k = re.sub(r"[^A-Za-z0-9]+", "_", k)
    k = re.sub(r"_+", "_", k).strip("_")
    return k.lower()


def _topic_key_from_keyword(keyword: str, country: str = "") -> str:
    return make_safe_keyword(keyword or "", country or "")


def load_category_rules(keyword: str, country: str = "") -> dict:
    topic_key = _topic_key_from_keyword(keyword, country)
    if topic_key in CATEGORY_CONFIG_CACHE:
        return CATEGORY_CONFIG_CACHE[topic_key]

    config_files = _iter_category_config_files()
    if not config_files:
        logging.warning("No category config file found. Category-based filtering will be skipped.")
        CATEGORY_CONFIG_CACHE[topic_key] = {}
        return {}

    best_rules = None
    best_score = -1
    best_source = None
    keyword_norm = normalize_text_for_search(keyword)

    for config_path in config_files:
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as e:
            logging.warning(f"Failed to read category config {config_path}: {e}")
            continue

        if not isinstance(payload, dict):
            logging.warning(f"Category config is not a JSON object: {config_path}")
            continue

        for category_name, category_rules in payload.items():
            if not isinstance(category_rules, dict):
                continue

            score = _score_category_match(keyword_norm, topic_key, str(category_name), category_rules)
            if score > best_score:
                best_score = score
                best_rules = json.loads(json.dumps(category_rules))
                best_source = (config_path, category_name)

    if best_rules and best_score > 0:
        logging.info(
            f"Loaded category config: {best_source[0]} -> {best_source[1]} (score={best_score})"
        )
        CATEGORY_CONFIG_CACHE[topic_key] = best_rules
        return best_rules

    logging.warning(
        f"No matching category config found for keyword={keyword!r}, country={country!r}. "
        "Category-based filtering will be skipped."
    )
    CATEGORY_CONFIG_CACHE[topic_key] = {}
    return {}


def _build_union_regex(values, prefix=False):
    parts = [re.escape(str(v).strip()) for v in (values or []) if str(v).strip()]
    if not parts:
        return None
    body = "|".join(sorted(parts, key=len, reverse=True))
    pattern = rf"(?i)^(?:{body})(?:\b|$)" if prefix else rf"(?i)\b(?:{body})\b"
    return re.compile(pattern)


def get_category_patterns(keyword: str, country: str = "") -> dict:
    rules = load_category_rules(keyword, country)
    extraction_rules = rules.get("product_name_extraction", {})
    if not isinstance(extraction_rules, dict):
        extraction_rules = {}
    bad_suffix_pattern = rules.get("bad_suffix_rx") if isinstance(rules, dict) else None
    bad_suffix_rx = None
    if bad_suffix_pattern:
        try:
            bad_suffix_rx = re.compile(bad_suffix_pattern)
        except re.error as e:
            logging.warning(f"Invalid bad_suffix_rx in category config: {e}")

    return {
        "rules": rules,
        "has_rules": bool(rules),
        "noise_exact": {normalize_text_for_search(x) for x in rules.get("high_signal_noise_exact", [])},
        "bad_exact": {normalize_text_for_search(x) for x in rules.get("bad_exact_headings", [])},
        "bad_suffix_rx": bad_suffix_rx,
        "editorial_bad_rx": _build_union_regex(rules.get("editorial_bad_words")),
        "generic_heading_rx": _build_union_regex(rules.get("generic_heading_prefixes"), prefix=True),
        "product_head_noun_rx": _build_union_regex(rules.get("product_head_nouns") or rules.get("generic_nouns")),
        "generic_modifier_words": {
            normalize_text_for_search(x)
            for x in (
                list(rules.get("generic_adjectives") or [])
                + list(extraction_rules.get("generic_modifier_words") or [])
            )
            if str(x).strip()
        },
        "extraction": extraction_rules,
    }


def _strip_json_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\s*```$", "", s).strip()
    return s


def _keyword_category_terms(keyword: str, country: str = "") -> set:
    words = re.findall(r"[a-z0-9]+", normalize_text_for_search(keyword))
    rules = load_category_rules(keyword, country)
    extraction_rules = rules.get("product_name_extraction", {})
    stop_words = {
        normalize_text_for_search(word)
        for word in extraction_rules.get("keyword_stop_words", [])
        if str(word).strip()
    }
    terms = set()
    for word in words:
        if word in stop_words or len(word) <= 2:
            continue
        terms.add(word)
        if word.endswith("ies") and len(word) > 4:
            terms.add(word[:-3] + "y")
        elif word.endswith("s") and len(word) > 4:
            terms.add(word[:-1])
    return terms


def _contains_keyword_category_term(text: str, keyword: str) -> bool:
    terms = _keyword_category_terms(keyword)
    if not terms:
        return False
    tokens = set(re.findall(r"[a-z0-9]+", normalize_text_for_search(text)))
    expanded = set(tokens)
    for token in tokens:
        if token.endswith("ies") and len(token) > 4:
            expanded.add(token[:-3] + "y")
        elif token.endswith("s") and len(token) > 4:
            expanded.add(token[:-1])
    return bool(terms & expanded)


def _looks_like_standalone_product_line(line: str, keyword: str = "", country: str = "") -> bool:
    patterns = get_category_patterns(keyword, country)
    extraction_rules = patterns["extraction"]
    s = (line or "").strip()
    if not s or s.startswith(("URL:", "Text:", "H1 ", "H2 ", "H3 ")):
        return False
    if len(s) > int(extraction_rules.get("max_candidate_characters", 140)):
        return False
    if re.match(r"(?i)^\d+\s+best\b", s):
        return False
    token_count = len(re.findall(r"[A-Za-z0-9]+", s))
    if token_count < int(extraction_rules.get("min_candidate_words", 2)):
        return False
    if token_count > int(extraction_rules.get("max_candidate_words", 12)):
        return False

    n = normalize_text_for_search(s)
    bad_fragments = tuple(
        normalize_text_for_search(item)
        for item in extraction_rules.get("standalone_exclude_fragments", [])
        if str(item).strip()
    )
    if any(fragment in n for fragment in bad_fragments):
        return False

    question_prefixes = extraction_rules.get("question_prefixes", [])
    if question_prefixes:
        prefix_body = "|".join(re.escape(str(item)) for item in question_prefixes)
        if re.match(rf"(?i)^(?:{prefix_body})\b", s):
            return False

    excluded_words_rx = _build_union_regex(extraction_rules.get("standalone_exclude_words"))
    if excluded_words_rx and excluded_words_rx.search(s):
        return False

    has_brand_like = bool(re.search(r"\b[A-Z][A-Za-z0-9&'-]{2,}\b", s))
    has_keyword_term = _contains_keyword_category_term(s, keyword)
    has_product_noun = bool(
        patterns["product_head_noun_rx"] and patterns["product_head_noun_rx"].search(s)
    )
    if extraction_rules.get("accept_brand_model_names"):
        return has_brand_like
    return has_brand_like and (has_keyword_term or has_product_noun)


def extract_high_signal_title_lines(text: str, keyword: str = "", country: str = "") -> str:
    heading_candidates = []
    standalone_candidates = []
    patterns = get_category_patterns(keyword, country)
    noise_exact = patterns["noise_exact"]

    for line in text.splitlines():
        s = line.strip()
        if not s or not s.startswith(("H1 ", "H2 ", "H3 ")):
            continue

        heading = s[3:].strip()
        heading = re.sub(r"(?i)^product summary:\s*", "", heading).strip()
        if not heading:
            continue

        h_norm = normalize_text_for_search(heading)
        if h_norm in noise_exact:
            continue
        if "captcha" in h_norm:
            continue
        if "verify" in h_norm and "visitor" in h_norm:
            continue

        if s.startswith("H1 "):
            heading_candidates.append(heading)
            continue

        if re.match(r"^\d{1,2}[\.)]?\s*\S+", heading):
            heading_candidates.append(heading)
            continue

        has_brand_like = bool(re.search(r"\b[A-Z][A-Za-z&]+\b", heading))
        has_model_like = bool(re.search(r"\b(?:\d+[A-Za-z]|[A-Za-z]+\d)\w*\b", heading))
        has_category_noun = bool(patterns["product_head_noun_rx"] and patterns["product_head_noun_rx"].search(heading))
        if has_brand_like and (has_model_like or has_category_noun):
            heading_candidates.append(heading)

    for line in text.splitlines():
        s = line.strip()
        if _looks_like_standalone_product_line(s, keyword, country):
            standalone_candidates.append(s)

    out = standalone_candidates + heading_candidates
    seen = set()
    deduped = []
    for item in out:
        key = normalize_text_for_search(item)
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return "\n".join(deduped)


def extract_product_names_from_file(keyword, country):
    safe_keyword = make_safe_keyword(keyword, country)
    file_path = Path(f"output/{safe_keyword}_{country}/content_{country}.txt")

    if not file_path.exists():
        logging.error(f"Input file not found: {file_path}")
        return []

    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception as e:
        logging.error(f"Failed to read file {file_path}: {e}")
        return []

    candidate_lines = extract_high_signal_title_lines(text, keyword, country)
    if not candidate_lines.strip():
        logging.warning("No high-signal lines found to send to the model.")
        return []

    prompt = (
        "From the following lines, extract ONLY real product names.\n"
        "Rules:\n"
        "- Must be a product title, not page sections like 'From the manufacturer' or 'Product Information'.\n"
        "- A brand plus a named product range/model is valid even when it has no number.\n"
        "- Return JSON array of unique strings.\n\n"
        f"{candidate_lines}"
    )

    raw = query_deepseek(prompt)
    if not raw:
        return []

    cleaned_raw = _strip_json_fences(raw)
    names = []
    try:
        parsed = json.loads(cleaned_raw)
        if isinstance(parsed, list):
            names = [str(x) for x in parsed if isinstance(x, (str, int, float))]
    except Exception:
        match = re.search(r"\[[\s\S]*\]", cleaned_raw)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    names = [str(x) for x in parsed if isinstance(x, (str, int, float))]
            except Exception:
                names = []

    return names or []


def extract_model_codes(text: str):
    if not text:
        return set()
    t = text.upper()
    patterns = [
        r"\b[A-Z]{1,5}\d{2,6}[A-Z]{0,5}(?:/\d{1,3})?\b",
        r"\b[A-Z]{1,4}(?:-[A-Z0-9]{1,6}){1,4}\b",
        r"\b\d{1,3}-IN-\d{1,3}\b",
        r"\b\d{1,3}\s*IN\s*\d{1,3}\b",
    ]
    codes = set()
    for pat in patterns:
        for code in re.findall(pat, t):
            code = re.sub(r"\s+", "", code)
            code = code.replace("IN", "-IN-") if re.fullmatch(r"\d{1,3}IN\d{1,3}", code) else code
            if len(code) >= 4:
                codes.add(code)
    return codes


def are_similar(name1, name2, threshold=0.8):
    c1 = extract_model_codes(name1 or "")
    c2 = extract_model_codes(name2 or "")
    if (c1 or c2) and c1 != c2:
        return False
    return SequenceMatcher(None, (name1 or "").lower(), (name2 or "").lower()).ratio() >= threshold


def _looks_like_plain_product_phrase(n: str, keyword: str = "", country: str = "") -> bool:
    if not n:
        return False

    patterns = get_category_patterns(keyword, country)
    extraction_rules = patterns["extraction"]

    if patterns["generic_heading_rx"] and patterns["generic_heading_rx"].search(n):
        return False
    if patterns["editorial_bad_rx"] and patterns["editorial_bad_rx"].search(n):
        return False

    token_count = len(n.split())
    if token_count < int(extraction_rules.get("min_product_name_words", 2)):
        return False
    if token_count > int(extraction_rules.get("max_product_name_words", 12)):
        return False

    if patterns["product_head_noun_rx"] and patterns["product_head_noun_rx"].search(n):
        alpha_tokens = re.findall(r"[a-z0-9&'-]+", n)
        informative = [
            token for token in alpha_tokens
            if token not in patterns["generic_modifier_words"] and len(token) > 2
        ]
        return len(informative) >= 2

    if _contains_keyword_category_term(n, keyword):
        alpha_tokens = re.findall(r"[a-z0-9&'-]+", n)
        informative = [
            token for token in alpha_tokens
            if token not in patterns["generic_modifier_words"] and len(token) > 2
        ]
        return len(informative) >= 2

    has_brand_like = bool(re.search(r"\b[a-z][a-z&'-]{2,}\b", n))
    has_model_like = bool(re.search(r"\b(?:\d+[a-z]|[a-z]+\d)[a-z0-9-]*\b", n))
    return has_brand_like and has_model_like


def _looks_like_configured_brand_model(name: str, patterns: dict) -> bool:
    extraction_rules = patterns["extraction"]
    if not extraction_rules.get("accept_brand_model_names"):
        return False

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9&'-]*", name or "")
    distinctive = [
        token for token in tokens
        if token[0].isupper()
        or token.isupper()
        or (
            any(character.isupper() for character in token[1:])
            and any(character.islower() for character in token)
        )
    ]
    required = int(extraction_rules.get("brand_model_min_distinctive_tokens", 2))
    return len(distinctive) >= required


def is_valid_product_candidate(name: str, keyword: str = "", country: str = "") -> bool:
    if not name:
        return False
    patterns = get_category_patterns(keyword, country)
    n = normalize_text_for_search(name).strip()
    if len(n.split()) < int(patterns["extraction"].get("min_product_name_words", 2)):
        return False
    if n in patterns["bad_exact"]:
        return False
    if patterns["bad_suffix_rx"] and patterns["bad_suffix_rx"].search(n):
        return False
    if patterns["generic_heading_rx"] and patterns["generic_heading_rx"].search(n):
        return False
    if patterns["editorial_bad_rx"] and patterns["editorial_bad_rx"].search(n):
        return False
    if MODEL_TOKEN_RX.search(name):
        return True
    if _looks_like_configured_brand_model(name, patterns):
        return True
    return _looks_like_plain_product_phrase(n, keyword, country)


def clean_product_names(product_names, keyword, country):
    safe_keyword = make_safe_keyword(keyword, country)
    discarded = {"duplicates": []}

    try:
        final_replacement_map = {}
        truly_unique_names = []

        for original_name in product_names:
            cleaned_name = str(original_name).strip('"\', \n\t').replace('\\"', '"')

            max_words = int(
                get_category_patterns(keyword, country)["extraction"].get(
                    "max_product_name_words", 12
                )
            )
            if len(cleaned_name.split()) > max_words:
                logging.debug(f"Discarded overlong candidate: {cleaned_name!r}")
                continue
            if not is_valid_product_candidate(cleaned_name, keyword, country):
                logging.debug(f"Discarded invalid candidate: {cleaned_name!r}")
                continue

            normalized_cleaned_name = normalize_text_for_search(cleaned_name)
            comma_index = normalized_cleaned_name.find(',')
            comma_cut = normalized_cleaned_name[:comma_index] if comma_index != -1 else normalized_cleaned_name
            first_7_words = " ".join(normalized_cleaned_name.split()[:7])
            shortened = first_7_words if len(first_7_words) < len(comma_cut) else comma_cut
            shortened = shortened.strip()

            if not shortened or len(shortened.split()) < 2:
                continue

            is_duplicate = False
            for existing in truly_unique_names:
                if are_similar(shortened, existing):
                    discarded["duplicates"].append(shortened)
                    final_replacement_map[original_name] = existing
                    is_duplicate = True
                    break

            if not is_duplicate:
                truly_unique_names.append(shortened)
                final_replacement_map[original_name] = shortened

        discard_path = Path(f"output/{safe_keyword}_{country}/_debug_discarded_product_names.txt")
        with open(discard_path, "w", encoding="utf-8") as f:
            if discarded["duplicates"]:
                f.write("Duplicates:\n" + "\n".join(discarded["duplicates"]) + "\n")

        logging.info(f"Final shortened names for CSV: {len(truly_unique_names)}")
        logging.info(f"Final mapping entries (for replacement): {len(final_replacement_map)}")
        return truly_unique_names, final_replacement_map
    except Exception as e:
        logging.error(f"Error during cleaning/deduplication: {e}")
        return [], {}


def replace_using_placeholders(keyword, country, name_map):
    safe_keyword = make_safe_keyword(keyword, country)
    safe_keyword_country = f"{safe_keyword}_{country}" if country else safe_keyword
    out_dir = os.path.join("output", safe_keyword_country)
    content_path = os.path.join(out_dir, f"content_{country}.txt")
    output_path = os.path.join(out_dir, f"content_{country}_updated.txt")

    if not os.path.exists(content_path):
        logging.warning(f"Content file not found: {content_path}")
        return

    with open(content_path, "r", encoding="utf-8") as f:
        content = f.read()

    if not content or not name_map:
        logging.warning("No content or no name_map provided; skipping replacements.")
        return

    canonical_names = sorted({c for c in name_map.values() if c and str(c).strip()})
    canon_to_variants = {c: {c} for c in canonical_names}
    for variant, canon in name_map.items():
        if canon in canon_to_variants and variant and str(variant).strip():
            canon_to_variants[canon].add(str(variant).strip())

    def _variant_sort_key(s: str):
        toks = re.findall(r"[A-Za-z0-9]+", s or "")
        return (-len(s or ""), -len(toks), s or "")

    def build_anywhere_pattern(s: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9]+", s or "")
        if not tokens:
            return ""
        joiner = r"[ \t\r\n\-–—_/+&.,:;()\[\]{}\"'’]*"
        inner = joiner.join(map(re.escape, tokens))
        return rf"(?i)(?<![A-Za-z0-9]){inner}(?![A-Za-z0-9])"

    used_infos = []
    for canon, variants in canon_to_variants.items():
        first_pos = None
        total_hits = 0
        for variant in variants:
            pat = build_anywhere_pattern(variant)
            if not pat:
                continue
            for match in re.finditer(pat, content):
                total_hits += 1
                if first_pos is None or match.start() < first_pos:
                    first_pos = match.start()
        if total_hits > 0 and first_pos is not None:
            used_infos.append((first_pos, normalize_text(canon), canon))
    used_infos.sort()
    used_canons = [canon for _, _, canon in used_infos]

    canonical_to_placeholder = {
        canon: f"<<PRODUCT_{i}>>" for i, canon in enumerate(used_canons, start=1)
    }

    lines = content.splitlines(keepends=True)
    total_replacements = 0

    def _is_protected_line(line: str) -> bool:
        s = (line or "").lstrip()
        return s.startswith(("URL:", "H1 ", "H2 ", "H3 "))

    for canon in used_canons:
        placeholder = canonical_to_placeholder[canon]
        variants = sorted(canon_to_variants[canon], key=_variant_sort_key, reverse=True)
        for variant in variants:
            pat = build_anywhere_pattern(variant)
            if not pat:
                continue
            new_lines = []
            for line in lines:
                if _is_protected_line(line):
                    new_lines.append(line)
                    continue
                replaced_line, count = re.subn(pat, placeholder, line, flags=re.IGNORECASE)
                if count:
                    total_replacements += count
                new_lines.append(replaced_line)
            lines = new_lines

    updated = "".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(updated)

    logging.info(f"Updated content written to: {output_path}")
    logging.info(f"Total actual replacements made: {total_replacements}")


def expand_to_full_product_phrases(
    content: str,
    extracted_names,
    keyword: str = "",
    country: str = "",
    max_length: int = 120,
):
    if not extracted_names:
        return []

    patterns = get_category_patterns(keyword, country)
    phrase_patterns = patterns["extraction"].get("product_phrase_patterns", [])
    if not phrase_patterns:
        return [str(name).strip() for name in extracted_names if str(name).strip()]

    results = []
    for name in extracted_names:
        name_str = str(name).strip()
        if not name_str:
            continue

        matches = list(re.finditer(re.escape(name_str), content, flags=re.IGNORECASE))
        if not matches:
            results.append(name_str)
            continue

        best = None
        for match in matches:
            start = max(0, match.start() - 60)
            end = min(len(content), match.end() + 60)
            window = re.sub(r"\s+", " ", content[start:end]).strip()

            candidate = None
            for phrase_pattern in phrase_patterns:
                phrase_match = re.search(phrase_pattern, window, flags=re.IGNORECASE)
                if phrase_match:
                    candidate = phrase_match.group(0)
                    break
            if not candidate:
                candidate = name_str

            candidate = re.sub(r"\s+", " ", candidate).strip(" .,:;!?\"'")
            if len(candidate) > max_length:
                continue
            if patterns["editorial_bad_rx"] and patterns["editorial_bad_rx"].search(candidate):
                continue

            score = len(candidate)
            if best is None or score > best[0]:
                best = (score, candidate)

        results.append(best[1] if best else name_str)

    return results


def process_keyword_products(keyword, country):
    import traceback

    safe_keyword = make_safe_keyword(keyword, country)
    setup_logging(safe_keyword, country)

    try:
        logging.info(f"--- Starting product processing for {keyword} ({country}) ---")
        extracted = extract_product_names_from_file(keyword, country)

        content_path = Path(f"output/{safe_keyword}_{country}/content_{country}.txt")
        try:
            content_text = content_path.read_text(encoding="utf-8")
            extracted = expand_to_full_product_phrases(
                content_text, extracted, keyword, country
            )
            logging.info("Expanded product names to full descriptive phrases.")
        except Exception as e:
            logging.warning(f"Failed to expand product names: {e}")

        if not extracted:
            logging.warning("No product names extracted.")
            flag_file = Path(f"output/{safe_keyword}_{country}/no_products.txt")
            try:
                flag_file.write_text("No products found", encoding="utf-8")
                logging.info(f"Flag file written: {flag_file}")
            except Exception as e:
                logging.error(f"Failed to write no_products flag file: {e}")
            return

        logging.info(f"Extracted {len(extracted)} product names.")
        cleaned, name_map = clean_product_names(extracted, keyword, country)
        if not cleaned:
            logging.warning("All product names discarded during cleaning.")
            flag_file = Path(f"output/{safe_keyword}_{country}/no_products.txt")
            try:
                flag_file.write_text("No valid products found after cleaning", encoding="utf-8")
                logging.info(f"Flag file written: {flag_file}")
            except Exception as e:
                logging.error(f"Failed to write no_products flag file: {e}")
            return

        logging.info(f"Cleaned down to {len(cleaned)} unique names.")
        output_path = Path(f"output/{safe_keyword}_{country}/extracted_product_names_{country}_1.csv")
        pd.DataFrame(cleaned, columns=["product_name"]).to_csv(output_path, index=False)
        logging.info(f"Saved cleaned product names to: {output_path}")

        mapping_path = Path(f"output/{safe_keyword}_{country}/_debug_original_to_cleaned_map.txt")
        with open(mapping_path, "w", encoding="utf-8") as f:
            for orig, new in name_map.items():
                f.write(f"{orig} → {new}\n")
        logging.info(f"Saved name mapping to: {mapping_path}")

        replace_using_placeholders(keyword, country, name_map)
    except Exception as e:
        logging.critical(f"Unexpected error: {e}")
        logging.critical(traceback.format_exc())
        print("Critical error occurred. Check the log for details.")


if __name__ == "__main__":
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        try:
            api_key = DEEPSEEK_API_KEY_FILE.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            print(f"❌ DeepSeek API key file not found: {DEEPSEEK_API_KEY_FILE}")
            raise SystemExit(1)
    if not api_key:
        print("❌ DeepSeek API key is empty.")
        raise SystemExit(1)

    client = DeepSeekClient(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    keyword, country = read_current_keyword_and_country()
    if keyword and country:
        process_keyword_products(keyword, country)
    else:
        print("❌ Failed to identify keyword and country from config.")
