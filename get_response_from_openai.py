"""
Category-agnostic review generator
----------------------------------
Refactors the pipeline to work for any product category by reading
category rules from category_config.json.

Backwards compatible with insert_amazon_links_images.py.
"""

import re
import time
import logging
import os
import json
import csv
from shared_company_names import load_company_names
import hashlib
from collections import Counter
from pathlib import Path
from collections import defaultdict
from urllib.parse import urlparse
from urllib.parse import unquote_plus
from typing import Optional, List
from datetime import date

from bs4 import BeautifulSoup
import tiktoken
import sys
import secrets
from html import unescape, escape
# Shared DeepSeek client and model configuration:
from internal_links import (
    DEEPSEEK_MODEL,
    DEEPSEEK_PRO_MODEL,
    _get_deepseek_client,
    _slugify,
    insert_internal_link_slots,
    log_deepseek_usage,
)
from difflib import SequenceMatcher

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Config loading & routing
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _deep_merge_config(*sources: dict) -> dict:
    """Recursively merge config dictionaries from left to right."""
    merged = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = _deep_merge_config(merged[key], value)
            else:
                merged[key] = value
    return merged


def _normalize_topic_key(value: str) -> str:
    return re.sub(r"[\s-]+", "_", (value or "").strip().casefold())


def load_topic_config(path: str, topic_key: str | None = None) -> tuple[dict, str]:
    with open(path, "r", encoding="utf-8-sig") as f:
        root = json.load(f)

    if not isinstance(root, dict):
        raise SystemExit(f"Config must be an object: {path}")

    # Supported formats:
    # 1) Multi-category: {"default": {...}, "backpacks": {...}, ...}
    # 2) Single wrapper: {"backpacks": {...}}
    # 3) Flat category: {"topic_key": "backpacks", "generic_tails": [...], ...}
    is_flat_category = any(
        key in root for key in ("topic_key", "generic_tails", "generic_adjectives")
    )

    if is_flat_category:
        cfg = dict(root)
        selected_topic = str(cfg.get("topic_key") or topic_key or Path(path).stem)
    else:
        category_entries = {
            str(key): value
            for key, value in root.items()
            if str(key).casefold() != "default" and isinstance(value, dict)
        }
        requested = _normalize_topic_key(topic_key or "")
        key_lookup = {
            _normalize_topic_key(key): key
            for key in category_entries
        }

        if requested:
            selected_key = key_lookup.get(requested)
            if selected_key is None:
                available = ", ".join(sorted(category_entries)) or "[none]"
                raise SystemExit(
                    f"Unknown topic '{topic_key}' in {path}. Available topics: {available}"
                )
        elif len(category_entries) == 1:
            selected_key = next(iter(category_entries))
        else:
            available = ", ".join(sorted(category_entries)) or "[none]"
            raise SystemExit(
                "Category was not specified for a multi-category config. "
                "Set ORCH_CATEGORY/TOPIC_KEY or provide column 4 in "
                f"config/current_keyword.csv. Available topics: {available}"
            )

        default_cfg = root.get("default")
        selected_cfg = category_entries[selected_key]
        cfg = _deep_merge_config(
            default_cfg if isinstance(default_cfg, dict) else {},
            selected_cfg,
        )
        selected_topic = str(selected_cfg.get("topic_key") or selected_key)

    if not isinstance(cfg, dict):
        raise SystemExit(f"Config for topic '{selected_topic}' must be an object: {path}")

    # global-ish defaults
    cfg.setdefault("explicit_removals", [])
    cfg.setdefault("noisy_keywords", [])
    cfg.setdefault("pros_triggers", ["why we like it","what we like","strengths","advantages","pros"])
    cfg.setdefault("cons_triggers", ["flaws","weaknesses","what we donâ€™t like","cons","drawbacks","issues"])
    cfg.setdefault("generic_headings_to_strip", ["purpose of the review","final verdict","value for money"])
    cfg.setdefault("acronym_allowlist", [])

    # required per topic
    for key in ("generic_tails", "generic_adjectives"):
        if not isinstance(cfg.get(key), list) or not cfg[key]:
            raise SystemExit(
                f"Topic '{selected_topic}' must define non-empty '{key}'"
            )

    # normalize
    cfg["generic_tails"] = [
        str(value).strip().lower()
        for value in cfg["generic_tails"]
        if str(value).strip()
    ]
    cfg["generic_adjectives"] = [
        str(value).strip().lower()
        for value in cfg["generic_adjectives"]
        if str(value).strip()
    ]

    # optional: normalize these too if present
    for key in (
        "generic_nouns",
        "category_only_blocklist",
        "category_only_bad_words",
        "section_label_left_sides",
    ):
        if isinstance(cfg.get(key), list):
            cfg[key] = [
                str(value).strip().lower()
                for value in cfg[key]
                if str(value).strip()
            ]

    return cfg, selected_topic


def _get_bad_exact(cfg: dict | None = None) -> set[str]:
    cfg = cfg or {}
    base = {
        "from the manufacturer", "from the brand", "product information",
        "product description", "product guidance & documents",
        "product guidance and documents", "about us", "customer reviews",
        "similar item to consider", "options available",
        "safety and product resources", "skip to",
        "similar brands on amazon", "customers also viewed these products",
        "different look, same great performance", "quick look",
        "key specifications", "key features", "specifications",
        "overview", "key terms",
    }
    extra = {str(x).strip().lower() for x in (cfg.get("bad_exact") or []) if str(x).strip()}
    return base | extra



_WORD_TOKEN_RX = re.compile(r"[a-z0-9]+", re.I)

def _tokenize(s: str) -> list[str]:
    return _WORD_TOKEN_RX.findall((s or "").lower())



# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Logging & IO helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def setup_logging(keyword, country):
    safe_keyword = keyword.replace(" ", "_")
    safe_keyword_country = f"{safe_keyword}_{country}"
    log_dir = os.path.join("logs", safe_keyword_country)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "deepseek_content.log")
    logging.basicConfig(filename=log_file, level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s",force=True)

def read_keyword_from_file(file_name="config/current_keyword.csv"):
    try:
        with open(file_name, "r", encoding='utf-8') as file:
            line = file.readline().strip()
            if not line:
                return "", ""
            parts = [p.strip() for p in next(csv.reader([line]))]
            # Accept 2 or 3+ columns; ignore extras like 'site'
            if len(parts) >= 2:
                keyword = parts[0].lstrip("\ufeff").strip()
                country = parts[1].upper()
                # Keywords are short search phrases, not prose, file names, or
                # task instructions. Fail fast instead of creating output under
                # an accidentally pasted sentence.
                words = re.findall(r"\b[\w'-]+\b", keyword)
                if (
                    not (1 <= len(words) <= 15)
                    or len(keyword) > 120
                    or re.search(r"(?i)(?:\.py|\.json|\.csv)\b", keyword)
                    or re.search(r"(?i)\b(?:attached\s+module|output\s+file|extracts?\s+a\s+top\s+pick)\b", keyword)
                    or re.search(r"[.!?]\s*$", keyword)
                ):
                    logging.error(f"Rejected malformed keyword value: {keyword!r}")
                    return "", country
                return keyword, country
    except Exception as e:
        logging.error(f"Failed to read keyword file: {e}")
    return "", ""


def read_category_from_file(file_name="config/current_keyword.csv"):
    """Return the optional category stored in column four of the keyword CSV."""
    try:
        with open(file_name, "r", encoding="utf-8-sig", newline="") as file:
            line = file.readline().strip()
        if not line:
            return ""
        parts = [part.strip() for part in next(csv.reader([line]))]
        return parts[3] if len(parts) >= 4 else ""
    except Exception as exc:
        logging.warning(f"Failed to read category from keyword file: {exc}")
        return ""


def read_file(file_path):
    try:
        with open(file_path, "r", encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return ""


_LIMITED_SOURCE_DATA_RX = re.compile(
    r"Limited source data was available for .*?[.!?]",
    re.IGNORECASE | re.DOTALL,
)


def get_top_pick_report_value(
    selected_top_pick: str,
    *,
    cleaned_content_path: str | None = None,
) -> str:
    """
    Return the value that should be written to the top-pick report.

    If the cleaned content file contains a "Limited source data was available for ..."
    message, that exact message is written instead of the usual top-pick string.
    """
    selected_top_pick = (selected_top_pick or "").strip()
    cleaned_content_path = (cleaned_content_path or "").strip()

    if cleaned_content_path and os.path.exists(cleaned_content_path):
        try:
            cleaned_content = read_file(cleaned_content_path)
            match = _LIMITED_SOURCE_DATA_RX.search(cleaned_content or "")
            if match:
                limited_message = re.sub(r"\s+", " ", match.group(0)).strip()
                logging.info(
                    f"ðŸ“Š Top pick report using limited-source message from {cleaned_content_path}: {limited_message}"
                )
                return limited_message
        except Exception as e:
            logging.warning(
                f"Could not inspect cleaned content file '{cleaned_content_path}' for top pick report override: {e}"
            )

    return selected_top_pick


def export_top_pick_report(
    keyword: str,
    selected_top_pick: str,
    report_dir: str = "output",
    *,
    cleaned_content_path: str | None = None,
    country: str = "",
    site: str = "",
    category: str = "",
    processed_date: str | None = None,
) -> str:
    """
    Upsert the CSV report row for a successful top pick while preserving
    the expanded schema used by run_batch.py.

    Columns:
      - processed_date
      - keyword
      - country
      - site
      - category
      - status
      - selected_top_pick
      - reason

    If the cleaned content file contains a limited-source-data message, that
    message is exported instead of the normal selected top pick.
    """
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "top_pick_report.csv")

    keyword = (keyword or "").strip()
    country = (country or "").strip().upper()
    site = (site or "").strip()
    category = (category or "").strip()
    processed_date = (processed_date or date.today().isoformat()).strip()

    report_value = get_top_pick_report_value(
        selected_top_pick,
        cleaned_content_path=cleaned_content_path,
    )

    fieldnames = [
        "processed_date",
        "keyword",
        "country",
        "site",
        "category",
        "status",
        "selected_top_pick",
        "reason",
    ]

    rows = []
    found = False
    this_key = (keyword.lower(), country, site, category)

    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_key = (
                        (row.get("keyword") or "").strip().lower(),
                        (row.get("country") or "").strip().upper(),
                        (row.get("site") or "").strip(),
                        (row.get("category") or "").strip(),
                    )

                    normalized = {k: (row.get(k) or "").strip() for k in fieldnames}
                    normalized["country"] = normalized["country"].upper()

                    if row_key == this_key:
                        normalized.update({
                            "processed_date": processed_date,
                            "keyword": keyword,
                            "country": country,
                            "site": site,
                            "category": category,
                            "status": "success",
                            "selected_top_pick": report_value,
                            "reason": "",
                        })
                        found = True

                    rows.append(normalized)
        except Exception as e:
            logging.warning(f"Could not read existing top pick report '{report_path}': {e}")

    if not found:
        rows.append({
            "processed_date": processed_date,
            "keyword": keyword,
            "country": country,
            "site": site,
            "category": category,
            "status": "success",
            "selected_top_pick": report_value,
            "reason": "",
        })

    rows.sort(key=lambda r: (
        (r.get("processed_date") or ""),
        (r.get("keyword") or "").lower(),
        (r.get("country") or "").upper(),
        (r.get("site") or ""),
        (r.get("category") or ""),
    ))

    with open(report_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logging.info(f"ðŸ“Š Top pick report exported to {report_path}")
    return report_path


def _normalize_minhash_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"'`]+", "", text)
    return text.strip()


def _char_ngrams(text: str, n: int = 5) -> set[str]:
    text = _normalize_minhash_text(text)
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def _stable_hash(text: str, seed: int) -> int:
    payload = f"{seed}:{text}".encode("utf-8", errors="ignore")
    return int(hashlib.md5(payload).hexdigest(), 16)


def _minhash_signature(text: str, num_hashes: int = 64, ngram_size: int = 5) -> tuple[int, ...]:
    shingles = _char_ngrams(text, n=ngram_size)
    if not shingles:
        return tuple()

    signature = []
    for seed in range(num_hashes):
        mins = min(_stable_hash(shingle, seed) for shingle in shingles)
        signature.append(mins)
    return tuple(signature)


def _estimated_jaccard_from_signatures(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def remove_minhash_near_duplicate_blocks(
    text: str,
    similarity_threshold: float = 0.82,
    num_hashes: int = 64,
    ngram_size: int = 5,
    min_block_len: int = 120,
    length_ratio_floor: float = 0.75,
):
    """
    Remove near-duplicate paragraphs/blocks using MinHash-style similarity.

    Speed improvement:
    - skips MinHash comparisons when block lengths are too different

    Returns:
        cleaned_text, stats_dict
    """

    if not text:
        return text, {
            "original_blocks": 0,
            "kept_blocks": 0,
            "removed_blocks": 0,
            "original_chars": 0,
            "kept_chars": 0,
            "removed_chars": 0,
            "removed_pct": 0.0,
            "comparisons": 0,
            "comparisons_skipped_by_length": 0,
        }

    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

    kept_blocks = []
    kept_signatures = []
    kept_lengths = []

    removed_blocks = 0
    removed_chars = 0
    comparisons = 0
    comparisons_skipped_by_length = 0

    for block in blocks:
        norm = _normalize_minhash_text(block)
        block_len = len(norm)

        # keep short blocks unless exact duplicate
        if block_len < min_block_len:
            if not any(_normalize_minhash_text(k) == norm for k in kept_blocks):
                kept_blocks.append(block)
                kept_signatures.append(tuple())
                kept_lengths.append(block_len)
            else:
                removed_blocks += 1
                removed_chars += len(block)
            continue

        sig = _minhash_signature(
            block,
            num_hashes=num_hashes,
            ngram_size=ngram_size
        )

        is_dup = False

        for kept_sig, kept_len in zip(kept_signatures, kept_lengths):
            if not kept_sig:
                continue

            # Fast skip: blocks with very different lengths are unlikely duplicates
            shorter = min(block_len, kept_len)
            longer = max(block_len, kept_len)
            length_ratio = shorter / max(longer, 1)

            if length_ratio < length_ratio_floor:
                comparisons_skipped_by_length += 1
                continue

            comparisons += 1
            similarity = _estimated_jaccard_from_signatures(sig, kept_sig)

            if similarity >= similarity_threshold:
                is_dup = True
                break

        if is_dup:
            removed_blocks += 1
            removed_chars += len(block)
        else:
            kept_blocks.append(block)
            kept_signatures.append(sig)
            kept_lengths.append(block_len)

    cleaned_text = "\n\n".join(kept_blocks)

    original_chars = len(text)
    kept_chars = len(cleaned_text)

    removed_pct = 0
    if original_chars > 0:
        removed_pct = round((removed_chars / original_chars) * 100, 1)

    stats = {
        "original_blocks": len(blocks),
        "kept_blocks": len(kept_blocks),
        "removed_blocks": removed_blocks,
        "original_chars": original_chars,
        "kept_chars": kept_chars,
        "removed_chars": removed_chars,
        "removed_pct": removed_pct,
        "comparisons": comparisons,
        "comparisons_skipped_by_length": comparisons_skipped_by_length,
    }

    return cleaned_text, stats

def _normalize_for_dedupe(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"<[^>]+>", " ", text)      # strip html tags if any leaked in
    text = re.sub(r"\s+", " ", text)          # normalize whitespace
    text = re.sub(r"[\"'`]+", "", text)       # normalize quotes
    return text.strip()


def _is_similar(a: str, b: str, threshold: float = 0.92) -> bool:
    """
    Returns True if two text blocks are near-duplicates.
    """
    if not a or not b:
        return False

    a_n = _normalize_for_dedupe(a)
    b_n = _normalize_for_dedupe(b)

    if a_n == b_n:
        return True

    shorter, longer = (a_n, b_n) if len(a_n) <= len(b_n) else (b_n, a_n)
    if shorter and shorter in longer and (len(shorter) / max(len(longer), 1)) >= 0.85:
        return True

    return SequenceMatcher(None, a_n, b_n).ratio() >= threshold


def remove_duplicate_content(
    text: str,
    paragraph_threshold: float = 0.92,
    sentence_threshold: float = 0.96,
    min_paragraph_len: int = 80
) -> str:
    """
    Removes exact and near-duplicate paragraphs/sentences while preserving order
    AND preserving paragraph structure.

    Strategy:
    1. Split into paragraphs on blank lines.
    2. Remove paragraphs that are exact or near-duplicates.
    3. Within each retained paragraph, remove repeated / near-duplicate sentences.
    4. Rebuild the text with paragraph breaks preserved.
    """
    if not text:
        return text

    # --- paragraph-level dedupe first ---
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    kept_paragraphs = []

    for p in raw_paragraphs:
        p_norm = _normalize_for_dedupe(p)

        # Short blocks are often headings or labels; only remove exact duplicates
        if len(p_norm) < min_paragraph_len:
            if not any(_normalize_for_dedupe(existing) == p_norm for existing in kept_paragraphs):
                kept_paragraphs.append(p)
            continue

        is_dup = any(
            _is_similar(p, existing, threshold=paragraph_threshold)
            for existing in kept_paragraphs
        )

        if not is_dup:
            kept_paragraphs.append(p)

    # --- sentence-level cleanup inside each kept paragraph ---
    cleaned_paragraphs = []

    for para in kept_paragraphs:
        sentence_candidates = re.split(r"(?<=[.!?])\s+", para)
        kept_sentences = []

        for s in sentence_candidates:
            s = s.strip()
            if not s:
                continue

            s_norm = _normalize_for_dedupe(s)

            # For short fragments, only remove exact duplicates
            if len(s_norm) < 25:
                if not any(_normalize_for_dedupe(existing) == s_norm for existing in kept_sentences):
                    kept_sentences.append(s)
                continue

            is_dup = any(
                _is_similar(s, existing, threshold=sentence_threshold)
                for existing in kept_sentences
            )

            if not is_dup:
                kept_sentences.append(s)

        cleaned_para = " ".join(kept_sentences).strip()
        if cleaned_para:
            cleaned_paragraphs.append(cleaned_para)

    return "\n\n".join(cleaned_paragraphs)
    
def remove_boilerplate_blocks(text: str) -> str:
    """
    Removes common boilerplate paragraphs that appear in scraped articles.
    """

    if not text:
        return text

    boilerplate_patterns = [
        r"this article may contain affiliate links",
        r"as an amazon associate",
        r"subscribe to receive",
        r"never miss an update",
        r"read more about",
        r"follow us on",
        r"sign up for",
        r"newsletter",
        r"posted on",
        r"last updated",
        r"thanks to .* for providing",
        r"more info on my policies page",
    ]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    kept = []

    for p in paragraphs:
        p_lower = p.lower()

        if any(re.search(pattern, p_lower) for pattern in boilerplate_patterns):
            continue

        kept.append(p)

    return "\n\n".join(kept)
def _looks_like_product_paragraph(text: str, brand_lexicon: set[str] | None = None) -> bool:
    """
    Returns True if the paragraph looks product-specific.

    Strong signals:
    - model-like tokens with digits (e.g. TP07, AP-1512HH, Core 300-P)
    - known brand mentions
    - multiple Title Case tokens that resemble a product name
    """
    if not text:
        return False

    t = re.sub(r"\s+", " ", text).strip()
    tl = t.lower()

    # 1) Model-like token with digits
    if re.search(r"\b[A-Za-z]*\d+[A-Za-z0-9\-\/]*\b", t):
        return True

    # 2) Known brand mention
    if brand_lexicon:
        for brand in brand_lexicon:
            if brand and brand.lower() in tl:
                return True

    # 3) Product-title-like phrase:
    #    2-6 consecutive Title Case / ALLCAPS / model-ish words
    if re.search(
        r"\b(?:[A-Z][A-Za-z0-9+\-\/]*|[A-Z]{2,}[A-Za-z0-9+\-\/]*)"
        r"(?:\s+(?:[A-Z][A-Za-z0-9+\-\/]*|[A-Z]{2,}[A-Za-z0-9+\-\/]*)){1,5}\b",
        t
    ):
        return True

    return False


_ACTIVE_CATEGORY_CONFIG: dict = {}


def configure_runtime_category(cfg: dict | None) -> None:
    """Expose the selected data-only category rules to legacy helper call paths."""
    global _ACTIVE_CATEGORY_CONFIG
    _ACTIVE_CATEGORY_CONFIG = dict(cfg or {})


def _runtime_category_config(cfg: dict | None = None) -> dict:
    return cfg if isinstance(cfg, dict) else _ACTIVE_CATEGORY_CONFIG


def _identity_word_aliases(cfg: dict | None = None) -> dict[str, str]:
    values = _runtime_category_config(cfg).get("identity_word_aliases") or {}
    return {
        str(key).casefold(): str(value).casefold()
        for key, value in values.items()
        if str(key).strip() and str(value).strip()
    }


def _is_generic_review_paragraph(text: str, cfg: dict | None = None) -> bool:
    """
    Returns True for category-level / generic review prose that is usually safe to drop
    when it does not mention a specific product.
    """
    if not text:
        return False

    tl = re.sub(r"\s+", " ", text).strip().lower()

    generic_patterns = [
        r"\bwhen choosing\b",
        r"\bwhat to look for\b",
        r"\bbefore you buy\b",
        r"\bbuying guide\b",
        r"\bhow to choose\b",
        r"\bvalue for money\b",
        r"\bfinal verdict\b",
        r"\bour verdict\b",
        r"\bthis type of product\b",
        r"\bmost products\b",
        r"\bmany models\b",
        r"\bthese products\b",
        r"\bthis can help you\b",
        r"\bfor your family\b",
        r"\bwe tested\b",
        r"\bhow we tested\b",
        r"\bi like to think of myself\b",
        r"\badventurous spirit\b",
    ]
    generic_patterns.extend(
        str(pattern)
        for pattern in (_runtime_category_config(cfg).get("generic_review_patterns") or [])
        if str(pattern).strip()
    )
    return any(re.search(pattern, tl) for pattern in generic_patterns)


def remove_generic_review_text(
    text: str,
    brand_lexicon: set[str] | None = None,
    min_para_len: int = 60,
    cfg: dict | None = None,
) -> tuple[str, dict]:
    """
    Product-aware filtering:
    - preserve product-specific paragraphs
    - remove generic review text when it lacks product signals

    Returns:
        cleaned_text, stats_dict
    """
    if not text:
        return text, {
            "original_paragraphs": 0,
            "kept_paragraphs": 0,
            "removed_paragraphs": 0,
            "original_chars": 0,
            "kept_chars": 0,
            "removed_chars": 0,
            "removed_pct": 0.0,
        }

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    kept = []
    removed_paragraphs = 0
    removed_chars = 0

    for para in paragraphs:
        para_norm = re.sub(r"\s+", " ", para).strip()

        # Keep short headings/labels unless obviously generic
        if len(para_norm) < min_para_len:
            if _looks_like_product_paragraph(para_norm, brand_lexicon=brand_lexicon):
                kept.append(para)
            elif _is_generic_review_paragraph(para_norm, cfg):
                removed_paragraphs += 1
                removed_chars += len(para)
            else:
                kept.append(para)
            continue

        has_product_signal = _looks_like_product_paragraph(
            para_norm, brand_lexicon=brand_lexicon
        )
        is_generic = _is_generic_review_paragraph(para_norm, cfg)

        # Only remove when it's generic AND not product-specific
        if is_generic and not has_product_signal:
            removed_paragraphs += 1
            removed_chars += len(para)
            continue

        kept.append(para)

    cleaned_text = "\n\n".join(kept)
    original_chars = len(text)
    kept_chars = len(cleaned_text)
    removed_pct = round((removed_chars / original_chars) * 100, 1) if original_chars > 0 else 0.0

    stats = {
        "original_paragraphs": len(paragraphs),
        "kept_paragraphs": len(kept),
        "removed_paragraphs": removed_paragraphs,
        "original_chars": original_chars,
        "kept_chars": kept_chars,
        "removed_chars": removed_chars,
        "removed_pct": removed_pct,
    }

    return cleaned_text, stats    

def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]


def _norm_block_for_compare(text: str) -> str:
    return _normalize_for_dedupe(text)


def _collect_removed_blocks(before_text: str, after_text: str) -> list[str]:
    """
    Returns paragraphs/blocks that were present in before_text but not retained
    in after_text, using normalized exact matching.
    """
    before_blocks = _split_paragraphs(before_text)
    after_norms = {_norm_block_for_compare(b) for b in _split_paragraphs(after_text)}

    removed = []
    seen = set()

    for block in before_blocks:
        norm = _norm_block_for_compare(block)
        if norm and norm not in after_norms and norm not in seen:
            removed.append(block)
            seen.add(norm)

    return removed


def _write_removed_text_report(report_path: str, sections: list[tuple[str, list[str]]]) -> None:
    """
    Writes removed text sections to a text file.
    sections = [("Section title", [block1, block2, ...]), ...]
    """
    lines = []

    for title, blocks in sections:
        lines.append(f"===== {title} =====")
        lines.append(f"Removed blocks: {len(blocks)}")
        lines.append("")

        if not blocks:
            lines.append("(none)")
            lines.append("")
            continue

        for i, block in enumerate(blocks, 1):
            lines.append(f"--- Removed block {i} ---")
            lines.append(block)
            lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")
        
def preprocess_raw_dataset(
    raw_dataset: str,
    logging_prefix: str = "[RAW_PREP]",
    brand_lexicon: set[str] | None = None,
    minhash_threshold: float = 0.82,
    minhash_num_hashes: int = 64,
    minhash_ngram_size: int = 5,
    minhash_min_block_len: int = 120,
    removed_text_path: str | None = None,
    cfg: dict | None = None,
):
    """
    Run all raw dataset preprocessing steps with logging.

    Stages:
    1. Exact / local near-duplicate removal
    2. Boilerplate removal
    3. Product-aware generic review text removal
    4. MinHash similarity dedupe

    If removed_text_path is provided, write out the text removed by:
    - duplicate removal
    - MinHash similarity dedupe
    """

    if not raw_dataset:
        logging.info(f"{logging_prefix} empty dataset")
        if removed_text_path:
            _write_removed_text_report(
                removed_text_path,
                [
                    ("Removed by duplicate removal", []),
                    ("Removed by MinHash similarity dedupe", []),
                ],
            )
        return raw_dataset

    def count_paragraphs(text):
        return len([p for p in re.split(r"\n\s*\n", text or "") if p.strip()])

    original_len = len(raw_dataset)
    original_paragraphs = count_paragraphs(raw_dataset)

    # â”€â”€ Stage 1: duplicate removal â”€â”€
    stage1 = remove_duplicate_content(raw_dataset)
    removed_by_dedup = _collect_removed_blocks(raw_dataset, stage1)

    stage1_len = len(stage1)
    stage1_paragraphs = count_paragraphs(stage1)

    removed_pct = round((1 - stage1_len / original_len) * 100, 1) if original_len else 0

    logging.info(
        f"{logging_prefix}[DEDUP] chars {original_len} â†’ {stage1_len} "
        f"({removed_pct}% removed) | paragraphs {original_paragraphs} â†’ {stage1_paragraphs}"
    )

    # â”€â”€ Stage 2: boilerplate removal â”€â”€
    stage2 = remove_boilerplate_blocks(stage1)

    stage2_len = len(stage2)
    stage2_paragraphs = count_paragraphs(stage2)

    removed_pct = round((1 - stage2_len / stage1_len) * 100, 1) if stage1_len else 0

    logging.info(
        f"{logging_prefix}[BOILERPLATE] chars {stage1_len} â†’ {stage2_len} "
        f"({removed_pct}% removed) | paragraphs {stage1_paragraphs} â†’ {stage2_paragraphs}"
    )

    # â”€â”€ Stage 3: product-aware generic text removal â”€â”€
    stage3, generic_stats = remove_generic_review_text(
        stage2,
        brand_lexicon=brand_lexicon,
        min_para_len=60,
        cfg=cfg,
    )

    logging.info(
        f"{logging_prefix}[GENERIC] chars {generic_stats['original_chars']} â†’ "
        f"{generic_stats['kept_chars']} ({generic_stats['removed_pct']}% removed) | "
        f"paragraphs {generic_stats['original_paragraphs']} â†’ {generic_stats['kept_paragraphs']} "
        f"(removed {generic_stats['removed_paragraphs']})"
    )

    # â”€â”€ Stage 4: MinHash similarity dedupe â”€â”€
    stage4, stats = remove_minhash_near_duplicate_blocks(
        stage3,
        similarity_threshold=minhash_threshold,
        num_hashes=minhash_num_hashes,
        ngram_size=minhash_ngram_size,
        min_block_len=minhash_min_block_len,
    )
    removed_by_minhash = _collect_removed_blocks(stage3, stage4)

    logging.info(
        f"{logging_prefix}[MINHASH] chars {stats['original_chars']} â†’ "
        f"{stats['kept_chars']} ({stats['removed_pct']}% removed) | "
        f"blocks {stats['original_blocks']} â†’ {stats['kept_blocks']} "
        f"(removed {stats['removed_blocks']}) | "
        f"comparisons={stats.get('comparisons', 0)} "
        f"skipped_by_length={stats.get('comparisons_skipped_by_length', 0)}"
    )

    final_len = len(stage4)
    final_paragraphs = count_paragraphs(stage4)

    total_removed = round((1 - final_len / original_len) * 100, 1) if original_len else 0

    logging.info(
        f"{logging_prefix}[FINAL] chars {original_len} â†’ {final_len} "
        f"({total_removed}% removed total) | paragraphs {original_paragraphs} â†’ {final_paragraphs}"
    )

    if removed_text_path:
        _write_removed_text_report(
            removed_text_path,
            [
                ("Removed by duplicate removal", removed_by_dedup),
                ("Removed by MinHash similarity dedupe", removed_by_minhash),
            ],
        )
        logging.info(f"{logging_prefix}[REMOVED_TEXT] saved to {removed_text_path}")

    return stage4
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Safeguard helpers: product whitelist, subsetting, post-checks
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_product_likely_slice(text: str, *, brand_lexicon=None, max_chars: int = 18000) -> str:
    """
    Bias the content toward lines that contain actual product models:
    - lines with digits / model codes
    - [H1]/[H2]/[H3]/[PRODUCT] tags
    - lines containing known brands (if provided)
    Falls back to the first max_chars if nothing is found.
    """
    if not text:
        return ""

    brand_lexicon = brand_lexicon or set()
    lines = text.splitlines()

    modelish = re.compile(r"[A-Za-z]*\d+[A-Za-z0-9\-\/]*")  # token with digits
    tagged = re.compile(r"^\s*(?:\[H1\]|\[H2\]|\[H3\]|\[PRODUCT\])", re.I)

    keep = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue

        if tagged.match(s):
            keep.append(s)
            continue

        if modelish.search(s):
            keep.append(s)
            continue

        if brand_lexicon:
            sl = s.lower()
            for b in list(brand_lexicon)[:2000]:
                if b and b.lower() in sl:
                    keep.append(s)
                    break

    blob = "\n".join(keep).strip()
    if len(blob) < 400:
        blob = text

    return blob[:max_chars]

_TESTIMONIAL_HEADING_RX = re.compile(
    r"(?i)\b("
    r"exceeded\s+my\s+expectations|"
    r"highly\s+recommended|"
    r"excellent\s+service|"
    r"very\s+impressed|"
    r"outstanding\s+service|"
    r"impressive\s+quality|"
    r"my\s+experience\s+with|"
    r"i\s+absolutely\s+love|"
    r"i\s+recently\s+purchased|"
    r"i\s+purchased|"
    r"i\s+can'?t\s+express|"
    r"five-?star"
    r")\b"
)

def looks_like_testimonial_heading(s: str) -> bool:
    """
    Trustburn/Trustpilot-style review headings & sentiment lines (NOT product names).
    """
    t = (s or "").strip()
    if not t:
        return True

    # sentiment/experience phrasing
    if _TESTIMONIAL_HEADING_RX.search(t):
        return True

    # title-like sentence (often starts with pronoun) and contains service/process words
    tl = t.lower()
    if re.search(r"(?i)^(i|my|our|we)\b", tl) and re.search(r"(?i)\b(service|team|installation|professional|delivery|staff)\b", tl):
        return True

    return False

def looks_like_rating_label(s: str) -> bool:
    """
    Detect rating labels like:
      Overall Rating: 4.7 / 5
      Rating 4.5 out of 5
      4.7/5 stars
    """
    if not s:
        return False

    t = s.lower().strip()

    if re.search(r"\b\d(\.\d+)?\s*/\s*5\b", t):
        return True
    if re.search(r"\b\d(\.\d+)?\s*out\s+of\s+5\b", t):
        return True
    if "overall rating" in t:
        return True

    return False
    
def looks_like_date_label(s: str) -> bool:
    """
    Reject date-like strings such as:
      Selasa, 16 Januari 2024
      Tuesday, 16 January 2024
      16 January 2024
      Jan 16, 2024
    Covers English + common Indonesian day/month names seen in scraped pages.
    """
    if not s:
        return False

    t = re.sub(r"\s+", " ", s).strip().lower()

    # day/month lexicons
    day_words = (
        "monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        "senin|selasa|rabu|kamis|jumat|jum'at|sabtu|minggu"
    )
    month_words = (
        "january|february|march|april|may|june|july|august|september|october|november|december|"
        "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|"
        "januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember"
    )

    # e.g. "Selasa, 16 Januari 2024" / "Tuesday, 16 January 2024"
    if re.search(rf"\b(?:{day_words})\b,?\s+\d{{1,2}}\s+\b(?:{month_words})\b\s+\d{{4}}\b", t, re.I):
        return True

    # e.g. "16 January 2024" / "16 Januari 2024"
    if re.search(rf"\b\d{{1,2}}\s+\b(?:{month_words})\b\s+\d{{4}}\b", t, re.I):
        return True

    # e.g. "January 16, 2024"
    if re.search(rf"\b(?:{month_words})\b\s+\d{{1,2}},?\s+\d{{4}}\b", t, re.I):
        return True

    return False
    
def looks_like_blog_metadata(s: str) -> bool:
    """
    Reject common blog metadata lines that appear in scraped content.
    Examples:
        Selasa, 16 Januari 2024
        Posted on January 16, 2024
        By John Smith
        Author: Jane Doe
        Updated March 2023
    """
    if not s:
        return False

    t = re.sub(r"\s+", " ", s).strip().lower()

    if looks_like_date_label(t):
        return True

    if re.search(r"\bposted\s+on\b", t):
        return True

    if re.search(r"\bupdated\b", t):
        return True

    if re.search(r"\bby\s+[a-z]", t):
        return True

    if re.search(r"\bauthor\b", t):
        return True

    return False

_LEADING_VERB_PREFIX_RX = re.compile(
    r"""(?ix)^\s*
    (?:i\s+)?                              # optional "I "
    (?:tried|tested|reviewed|used|bought|purchased|installed|had|own(?:ed)?|got)\b
    \s+
    """
)

_LEADING_ARTICLE_RX = re.compile(r"(?ix)^\s*(?:the|a|an)\s+")

def strip_sentence_tail(s: str) -> str:
    """
    Remove descriptive tail text from a product-like phrase.
    Keeps the LEFT side only.
    Example: "The Dyson Hot+Cool HF1 is a sleek..." -> "The Dyson Hot+Cool HF1"
    """
    if not s:
        return s

    x = re.sub(r"\s+", " ", s).strip()

    # Cut at first sentence punctuation
    x = re.split(r"[\.!\?]\s+", x, maxsplit=1)[0].strip()

    # Cut at common description introducers
    x = re.split(
        r"(?i)\b("
        r"is|are|was|were|"
        r"has|have|had|"
        r"features|feature|offers|offer|includes|include|"
        r"with|including|"
        r"that|which|"
        r"out\s+for\s+a\s+month|out\s+for\s+a\s+week|out\s+for\s+a\s+day"
        r")\b",
        x,
        maxsplit=1
    )[0].strip()

    return x.strip(" -â€“â€”:;,.")  # final cleanup

def strip_leading_review_verbs(s: str) -> str:
    if not s:
        return s
    return _LEADING_VERB_PREFIX_RX.sub("", s).strip()

def strip_leading_definite_article(s: str) -> str:
    if not s:
        return s
    return _LEADING_ARTICLE_RX.sub("", s).strip()

# âœ… Backwards-compatible alias (fixes your crash)
def trim_product_sentence_tail(s: str) -> str:
    return strip_sentence_tail(s)


def looks_like_company_review_page(cleaned_text: str) -> bool:
    """
    Detects Trustburn-style company review pages that are not product model pages.
    This prevents DeepSeek from returning review headings as 'products'.
    """
    if not cleaned_text:
        return False
    t = cleaned_text.lower()

    # Strong trustburn/company-review indicators from your example content
    if "trustburn.com/reviews/" in t and "start collecting reviews today" in t:
        return True
    if "claim your business" in t and "start collecting reviews today" in t:
        return True
    if "find out what customers think about" in t and "reviews" in t:
        return True

    return False


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def find_verbatim_evidence_for_product(content: str, product: str, *, max_chars: int = 240) -> Optional[str]:
    """
    Find a short excerpt from `content` that contains `product` verbatim (same words, same order).

    Prefer the exact line/chunk containing the product, because broad context windows can
    contaminate evidence with nearby prose and cause downstream realignment to pick a
    descriptive sentence instead of the product heading itself.
    """
    if not content or not product:
        return None

    content_n = content
    product_n = _normalize_ws(product)
    needle = product_n.lower()

    # Prefer exact line/chunk matches first. This keeps the evidence as tight as possible.
    chunks: List[str] = re.split(r"[\r\n]+|â€¢|\u2022|\||\t", content_n)
    for ch in chunks:
        ch_n = _normalize_ws(ch)
        if not ch_n:
            continue
        if needle in ch_n.lower():
            return ch_n[:max_chars]

    # Fallback: narrow neighborhood slice from the full content.
    lower = content_n.lower()
    idx = lower.find(needle)
    if idx != -1:
        start = max(0, idx - 60)
        end = min(len(content_n), idx + len(product_n) + 60)
        excerpt = _normalize_ws(content_n[start:end])
        return excerpt[:max_chars]

    return None


def looks_like_reversed_category_brand(name: str, *, cfg: dict | None = None) -> bool:
    """
    Reject reversed category-brand strings like "Steam Mop Goblin" where a generic
    category tail appears first and a short brand/name fragment follows.
    """
    s = re.sub(r"\s+", " ", (name or "")).strip()
    if not s:
        return False

    cfg = cfg or {}
    tails = [str(t).strip().lower() for t in (cfg.get("generic_tails") or []) if str(t).strip()]
    s_l = s.lower()

    for tail in tails:
        if s_l.startswith(tail + " "):
            remainder = s[len(tail):].strip(" -â€“â€”:;,.")
            if remainder and len(remainder.split()) <= 2:
                return True
    return False


def contains_complete_term(text: str, term: str) -> bool:
    """Match a deny term as complete word(s), never inside a product word."""
    text = re.sub(r"\s+", " ", (text or "")).strip()
    term = re.sub(r"\s+", " ", (term or "")).strip()
    if not text or not term:
        return False
    pattern = re.escape(term).replace(r"\ ", r"\s+")
    return bool(re.search(rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])", text, re.IGNORECASE))


def contains_any_complete_term(text: str, terms) -> bool:
    return any(contains_complete_term(text, term) for term in (terms or []) if term)


def has_top_pick_deny_signal(name: str, *, cfg: dict | None = None) -> bool:
    """
    Central deny-term check for accessory / compatibility titles that should never win
    the top-pick slot. Includes a few hard-coded phrases that are too important to rely
    on config updates for.
    """
    s = re.sub(r"\s+", " ", (name or "")).strip().lower()
    if not s:
        return False

    deny_terms = {
        str(x).strip().lower()
        for x in ((cfg or {}).get("top_pick_deny_terms") or [])
        if str(x).strip()
    }
    deny_terms |= {
        str(x).strip().lower()
        for x in ((cfg or {}).get("exclude_in_title_strict") or [])
        if str(x).strip()
    }
    deny_terms |= {"to fit", "fits", "fit for", "compatible replacement", "spare"}

    return contains_any_complete_term(s, deny_terms)


def looks_like_marker_contaminated_product(name: str) -> bool:
    s = re.sub(r"\s+", " ", (name or "")).strip().lower()
    if not s:
        return False
    return bool(re.match(r"^(?:text:|h[1-6]|\[product\]|product\])", s))
def evidence_contains_product_verbatim(evidence: str, product: str) -> bool:
    """
    Enforce: product name must appear in evidence in the same word order.
    Implemented as a normalized substring check, with punctuation-normalization
    so harmless formatting differences don't cause false rejects.

    Examples that should pass:
      evidence: "Dyson Pure Humidify + Cool PH01 Air Purifier"
      product:  "Dyson Pure Humidify Cool PH01"
    """
    if not evidence or not product:
        return False

    def _canon(s: str) -> str:
        s = _normalize_ws(s).lower()

        # Treat common joiners as spaces (keeps word order, avoids "+" mismatch)
        s = s.replace("+", " ")
        s = s.replace("&", " ")
        s = s.replace("/", " ")

        # Normalize dash variants to spaces too
        s = re.sub(r"[â€-â€’â€“â€”-]+", " ", s)

        # Collapse whitespace again after replacements
        s = re.sub(r"\s+", " ", s).strip()
        return s

    ev = _canon(evidence)
    pr = _canon(product)
    if not ev or not pr:
        return False

    return pr in ev

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Heading marker cleanup (fix "H1 Product Name..." leaking into top_pick)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_HEADING_MARKER_PREFIX_RX = re.compile(
    r"(?i)^\s*(?:h[1-6]\s+|\[h[1-6]\]\s+|<\s*h[1-6][^>]*>\s*|\[product\]\s*|product\]\s*|text:\s*)"
)

def strip_heading_marker_prefix(s: str) -> str:
    """
    Removes leading structural markers that sometimes leak from HTML->text:
      - "H1 DKNY ..." / "h2 ..." (bare)
      - "[H1] DKNY ..." (tagged)
      - "<h1>DKNY ..." (rare partial)
      - "[PRODUCT] ..." / "PRODUCT] ..."
      - "Text: ..."
    """
    x = (s or "").strip()
    if not x:
        return ""
    prev = None
    while x and x != prev:
        prev = x
        x = _HEADING_MARKER_PREFIX_RX.sub("", x).strip()
    return x
    

_URL_RX = re.compile(r"(?i)\bhttps?://\S+\b")
_AMAZON_CHROME_RX = re.compile(
    r"""(?ix)
    \b(
        amazon\ global\ store|
        purchase\ options\ and\ add-?ons|
        add-?ons|
        customers|
        customer\ reviews?|
        option[s]?\ available|
        brand
    )\b
    """
)

def strip_url_and_retailer_chrome(s: str) -> str:
    """
    Remove leaked URL / retailer chrome from candidate product names.
    Example:
      "URL: https://... H3 Amazon Global Store H1 Thule Subterra Carry On Spinner"
      -> "Thule Subterra Carry On Spinner"
    """
    x = re.sub(r"\s+", " ", (s or "")).strip()
    if not x:
        return x

    x = re.sub(r"(?i)\burl\s*:\s*", " ", x)
    x = _URL_RX.sub(" ", x)
    x = _AMAZON_CHROME_RX.sub(" ", x)

    parts = re.split(r"(?i)\bH[1-6]\s+", x)
    if len(parts) > 1:
        x = parts[-1]

    x = re.sub(r"(?is)</?h[1-6][^>]*>", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    x = x.strip(" \t\r\n-â€“â€”:;,.|")
    return x


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Placeholder-aware grounding for Amazon "Product Summary: <<PRODUCT_2>> ..." lines
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_PLACEHOLDER_PRODUCT_RX = re.compile(r"<<\s*PRODUCT[_\-\s]*\d+\s*>>", re.I)

def _compact_alnum(s: str) -> str:
    """Lowercase and strip to only [a-z0-9] for robust substring matching."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def evidence_has_placeholder_marker(evidence: str) -> bool:
    return bool(evidence and _PLACEHOLDER_PRODUCT_RX.search(evidence))

def evidence_matches_product_by_model_token(evidence: str, product: str) -> bool:
    """
    Allows grounding when evidence uses placeholders (e.g. <<PRODUCT_2>>) but includes
    a distinctive model token that matches the product.
    Example evidence: "... Core300-P, White"  product: "LEVOIT Core 300-P"
    """
    if not evidence or not product:
        return False
    if not evidence_has_placeholder_marker(evidence):
        return False

    ev_c = _compact_alnum(evidence)

    # Model-ish tokens: any token containing a digit (keeps "300-P", "Core300-P", "AC0820/30", etc.)
    # Use the ORIGINAL product string so hyphens/slashes remain part of tokens.
    model_tokens = set(re.findall(r"[A-Za-z]*\d+[A-Za-z0-9\-\/]*", product))
    for tok in model_tokens:
        tok_c = _compact_alnum(tok)
        if len(tok_c) >= 3 and tok_c in ev_c:
            return True

    return False


def whitelist_note(products: list[str]) -> str:
    if not products:
        return ""
    items = "\n".join(f"- {p}" for p in products)
    return (
        "You may ONLY mention products from this whitelist (verbatim full names):\n"
        f"{items}\n"
        "If a product is not on this list, do not mention it."
    )

_CAPITALIZED_MODEL_RX = re.compile(
    r"\b([A-Z][A-Za-z0-9]+(?: [A-Z0-9][A-Za-z0-9\-]+){1,6})\b"
)
# Common UI/editorial phrases that are capitalized but are NOT products.
# Keep these short and broad; add more as you see false positives in logs.
_UI_PHRASE_ALLOWLIST = {
    "quick verdict",
    "final verdict",
    "value for money",
    "purpose of the review",
    "frequently asked questions",
    "comparison table",
    "key differences",
    "see todayâ€™s price on amazon",
    "amazon price link",
    "product link",
    "quick verdict box",
    "quick verdict title",
    "quick verdict cta",
    "best overall",
    "pros",
    "cons",
    "general view",
    "customers bought together",
    "products customers bought together",
    "overview",
    "summary",
}

MODEL_TOKEN_RX = re.compile(r"\b[A-Z]{1,4}[A-Z0-9-]{4,}\b")

def evidence_matches_product_by_model_token_any(evidence: str, product: str) -> bool:
    """
    Accept grounding if evidence contains at least one strong model-like token
    that also appears in the product (e.g. AX71-304GY), even if the evidence
    doesn't include the full title tail like 'Air Purifier'.
    """
    if not evidence or not product:
        return False

    ev = _normalize_ws(evidence)
    pr = _normalize_ws(product)

    ev_models = set(MODEL_TOKEN_RX.findall(ev))
    pr_models = set(MODEL_TOKEN_RX.findall(pr))

    return bool(ev_models and pr_models and (ev_models & pr_models))

def _looks_like_ui_phrase(s: str) -> bool:
    """
    Returns True if the candidate looks like a UI/editorial/promo phrase we should ignore.
    """
    if not s:
        return True

    t = re.sub(r"\s+", " ", s).strip().lower()

    editorial_prefixes = [
        str(value).strip().casefold()
        for value in (
            ((_runtime_category_config().get("canonical_facts") or {})
             .get("editorial_product_name_prefixes") or [
                 "the bottom line", "bottom line", "our verdict",
                 "final verdict", "editor's verdict", "editors' verdict",
             ])
        )
        if str(value).strip()
    ]
    if any(
        re.match(rf"^(?:the\s+)?{re.escape(prefix)}(?:\b|\s*[:\-])", t)
        for prefix in editorial_prefixes
    ):
        return True

    # Common search widgets, CTA headings, and editorial title suffixes must
    # never be treated as products or fuzzy-whitelist destinations.
    if re.search(r"(?i)^people\s+also\s+ask(?:\s*\(faqs?\))?$", t):
        return True
    if re.search(r"(?i)^check\s+out\b.*\b(?:current\s+)?prices?\b", t):
        return True
    if re.search(r"(?i)\bpros\s*(?:&|and)\s*cons\b", t):
        return True
    if re.search(r"(?i)\breviewed\b", t):
        return True
    if re.search(r"(?i)\b(?:running\s+costs?|cost\s+per\s+hour|ofgem|p\s*/\s*kwh)\b", t):
        return True
    
    # ---- Common SERP / publisher UI junk we must never treat as products ----
    if re.search(r"(?i)\bdownload\s+the\s+.+\s+app\b", t):
        return True

    # e.g. "We Scanned895products" / "We Scanned 895 products"
    if re.search(r"(?i)^\s*we\s+scanned\s*\d+\s*products?\b", t):
        return True

    # e.g. "YOU CAN ALSO READ"
    if re.search(r"(?i)^\s*you\s+can\s+also\s+read\b", t):
        return True

    # Common â€œapp CTAâ€ variants
    if re.search(r"(?i)\b(cnn|nytimes|bbc|guardian)\s+app\b", t):
        return True

    # Exact allowlist hit
    if t in _UI_PHRASE_ALLOWLIST:
        return True
        
    # âœ… NEW: "Section Label: Topic" patterns (common false positives)
    # Examples: "Purifier: Set-up", "Filter: Replacement", "Controls: Overview"
    if ":" in s:
        left, right = [x.strip().lower() for x in s.split(":", 1)]
        # Very common section-label left sides across retailer/editorial pages
        common_left_sides = {
            "controls", "setup", "set up", "set-up", "features", "specs",
            "performance", "maintenance", "conclusion",
        }
        configured_left_sides = {
            str(value).strip().casefold()
            for value in (_runtime_category_config().get("section_label_left_sides") or [])
            if str(value).strip()
        }
        if left in common_left_sides | configured_left_sides:
            return True
        # If the right side looks like a how-to / section topic, treat as UI even if left is unknown
        if re.search(r"(?i)\b(set[-\s]?up|how to|overview|guide|manual|instructions|maintenance|faq|specs)\b", right):
            return True


    # âœ… NEW: promo/CTA patterns (common false positives in SERP/Amazon blocks)
    # Examples: "Unlock 5% savings", "Save 20%", "Claim discount"
    if "%" in t:
        return True

    # Match complete promo terms/phrases. Raw substring checks create false
    # positives inside valid product words (for example, "ideal" contains
    # "deal").
    if re.search(
        r"\b(?:unlock|save|savings|discount|deal|offer|coupon|promo|promotion|"
        r"limited\s+time|prime|subscribe|free\s+delivery)\b",
        t,
    ):
        return True

    # Starts-with patterns that often appear as UI text
    if t.startswith(("see ", "read ", "view ", "click ", "related ", "deals ", "buy ")):
        return True

    # Generic section labels / headings that look like products to the regex
    # Match complete heading terms. In particular, substring matching for
    # "table" incorrectly classified every product containing "Portable" as
    # UI text and triggered evidence-line rescue.
    if re.search(r"\b(?:verdict|review|summary|faq|comparison|table)\b", t):
        return True

    # Short Title-Case phrases with no digits/hyphens/acronyms can be headings OR real products.
    # Old logic falsely rejected things like "Coway Mighty Air Purifier".
    has_distinctive = bool(re.search(r"(\d|[A-Z]{2,}|\w+-\w+)", s))
    words = t.split()

    if not has_distinctive and len(words) <= 4:
        # If it contains *explicit* heading/UI terms, it's likely not a product.
        if any(w in {"overview", "summary", "view", "general", "verdict", "pros", "cons"} for w in words):
            return True

        # If it looks like a product title (>=2 Title-Case-ish tokens), DO NOT flag as UI.
        # Example: "Coway Mighty Air Purifier" should pass.
        titlecase_tokens = sum(
            1 for w in s.split()
            if re.match(r"^[A-Z][A-Za-z0-9]+$", w) and not w.isupper()
        )
        if titlecase_tokens >= 2:
            return False

        # Otherwise, treat very short, non-distinctive phrases as UI-ish
        return True


    return False

import re

def is_category_only_title(name: str, *, cfg: dict | None = None) -> bool:
    """
    Returns True if the title looks like it's just the category/topic name
    (e.g., "air purifier", "air purifiers") rather than a real product model/name.

    Config-driven:
      - generic_tails: list[str] (exact-match category tails)
      - category_only_blocklist: list[str] (exact-match category-only titles)
      - category_only_bad_words: list[str] (for very-short titles made only of these tokens)
    """
    cfg = cfg or {}

    s = (name or "").strip()
    if not s:
        return True

    s_l = s.lower()

    def _norm_list(key: str) -> list[str]:
        vals = cfg.get(key) or []
        if not isinstance(vals, list):
            return []
        out: list[str] = []
        for x in vals:
            x = str(x).strip().lower()
            if x and x not in out:
                out.append(x)
        return out

    generic_tails = _norm_list("generic_tails")
    blocklist = _norm_list("category_only_blocklist")
    bad_words = set(_norm_list("category_only_bad_words"))

    # Exact category-only matches (prefer explicit blocklist, but allow generic_tails too)
    if s_l in blocklist or s_l in generic_tails:
        return True

    # Very short + contains only configured generic words
    toks = re.findall(r"[a-z0-9]+", s_l)
    if bad_words and len(toks) <= 3:
        if toks and all(t in bad_words for t in toks):
            return True

    return False



def violates_whitelist(text: str, products: list[str]) -> list[str]:
    if not text:
        return []
    wl = {p.lower(): p for p in (products or [])}

    candidates = set(_CAPITALIZED_MODEL_RX.findall(text))

    # âœ… NEW: drop obvious UI/editorial phrases
    candidates = {c for c in candidates if not _looks_like_ui_phrase(c)}

    # Keep only things that still look like product-ish text
    bad = [c for c in candidates if c.lower() not in wl]
    return bad

_LEGAL_ENTITY_NAME_RX = re.compile(
    r"(?i)\b(?:corporation|corp\.?|llc|ltd\.?|limited|inc\.?|incorporated|plc|gmbh|"
    r"holdings?\s+(?:llc|ltd\.?|limited|inc\.?|plc))\s*$"
)

_EDITORIAL_PRODUCT_NAME_SUFFIX_RX = re.compile(
    r"(?i)\s*(?:[:|\-]\s*)"
    r"(?:performance\s+over\s+time|long[- ]term\s+performance|hands[- ]on\s+review|"
    r"full\s+review|review|final\s+verdict|verdict|specifications?|specs|features?|"
    r"pros?\s+(?:and|&)\s+cons?|who\s+is\s+it\s+for|"
    r"(?:buying\s+)?guide(?:\s*\(?20\d{2}\)?)?)\s*$"
)

# Category/type synonyms are supplied by identity_word_aliases in JSON.

def looks_like_legal_entity_name(name: str) -> bool:
    """Identify legal company headings that are not specific product names."""
    return bool(_LEGAL_ENTITY_NAME_RX.search((name or "").strip().rstrip(" .,:;!?")))

def strip_editorial_product_name_suffix(name: str) -> str:
    """Remove section labels appended to an otherwise valid product name."""
    s = (name or "").strip()
    previous = None
    while s and s != previous:
        previous = s
        s = _EDITORIAL_PRODUCT_NAME_SUFFIX_RX.sub("", s).strip().rstrip(" .,:;!?")
    return s

def canonical_product_identity_key(name: str) -> tuple[str, ...]:
    """Order-insensitive key using category-configured type-word aliases."""
    cleaned = strip_editorial_product_name_suffix(name)
    words = re.findall(r"[a-z0-9]+", cleaned.lower())
    aliases = _identity_word_aliases()
    words = [aliases.get(word, word) for word in words]
    return tuple(sorted(words))

def keyword_product_identity_tokens(text: str) -> set[str]:
    """Return model/spec tokens that must survive reviewed-product selection."""
    normalized = (text or "").casefold()
    normalized = re.sub(
        r"(?<=\d)\s+(?=(?:l|ml|cl|kg|g|mg|w|kw|v|mah|inch|inches|cm|mm)\b)",
        "",
        normalized,
    )
    tokens = set()
    token_rx = re.compile(r"\b(?:[a-z]+\d+[a-z0-9]*|\d+(?:\.\d+)?[a-z]+|\d+)\b")
    for match in token_rx.finditer(normalized):
        token = match.group(0)
        if re.fullmatch(r"(?:19|20)\d{2}", token):
            continue
        prefix = normalized[max(0, match.start() - 24):match.start()]
        if token.isdigit() and re.search(
            r"(?:\bunder|\bbelow|\bover|\babove|\bless than|\bmore than|"
            r"\bbudget|[\u00a3$\u20ac])\s*$",
            prefix,
        ):
            continue
        tokens.add(token)
    return tokens


def reviewed_product_matches_keyword(
    product: str,
    keyword: str,
    cfg: dict | None = None,
) -> bool:
    """Protect brand/model intent; affiliate availability cannot retarget a review."""
    controls = (cfg or {}).get("review_identity") or {}
    if controls.get("allow_product_retargeting", False):
        return True
    if controls.get("lock_keyword_model_tokens", True) is False:
        return True
    required = keyword_product_identity_tokens(keyword)
    if not required:
        return True
    present = keyword_product_identity_tokens(product)
    return required.issubset(present)


def canonicalize_product_tags_in_text(cleaned_text: str, cfg: dict | None = None) -> str:
    """Demote company headings and merge obvious product aliases in tagged evidence."""
    if not cleaned_text:
        return cleaned_text

    product_rx = re.compile(r"(?im)^\s*\[PRODUCT\]\s*(.+?)\s*$")
    canonical_by_key = {}
    aliases = {}

    for match in product_rx.finditer(cleaned_text):
        raw = match.group(1).strip()
        cleaned = normalize_product_name(
            raw, cfg=cfg, try_map_to_whitelist=False, max_words=None
        )
        cleaned = strip_editorial_product_name_suffix(cleaned)
        if not cleaned or looks_like_legal_entity_name(cleaned):
            continue
        key = canonical_product_identity_key(cleaned)
        canonical = canonical_by_key.setdefault(key, cleaned)
        aliases[raw] = canonical
        aliases[cleaned] = canonical

    def replace_tag(match):
        raw = match.group(1).strip()
        cleaned = strip_editorial_product_name_suffix(raw)
        if looks_like_legal_entity_name(cleaned):
            return f"[H2] {cleaned}"
        canonical = aliases.get(raw) or aliases.get(cleaned) or cleaned
        return f"[PRODUCT] {canonical}"

    result = product_rx.sub(replace_tag, cleaned_text)
    for alias, canonical in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if alias == canonical:
            continue
        result = re.sub(
            rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
            canonical,
            result,
            flags=re.I,
        )
    return result

def subset_for_heading(dataset_text: str, heading: str, products: list[str], min_distinct:int=2, max_chars:int=12000) -> str:
    """Build a topic-relevant evidence slice without injecting unrelated products."""
    lines = dataset_text.splitlines()
    stopwords = {
        "a", "an", "and", "are", "as", "at", "being", "by", "for", "from",
        "in", "is", "it", "models", "of", "on", "other", "over", "product",
        "products", "review", "reviewed", "special", "the", "to", "with",
        "features", "performance", "time",
    }
    heading_tokens = {
        token for token in re.findall(r"[a-z0-9]+", (heading or "").lower())
        if len(token) >= 3 and token not in stopwords
    }

    selected = set()
    for index, line in enumerate(lines):
        line_tokens = set(re.findall(r"[a-z0-9]+", line.lower()))
        if heading_tokens and heading_tokens.intersection(line_tokens):
            for nearby in range(max(0, index - 1), min(len(lines), index + 2)):
                # Do not cross into a neighbouring product/section merely
                # because it follows a relevant evidence line.
                if nearby != index and re.match(r"^\s*\[(?:PRODUCT|H[1-6])\]", lines[nearby], re.I):
                    continue
                selected.add(nearby)

    # Add evidence for the primary product only. Arbitrary extra whitelist entries
    # must not turn a one-product section into a fabricated comparison.
    primary = next((p for p in (products or []) if p), "")
    if primary:
        primary_low = primary.lower()
        for index, line in enumerate(lines):
            if primary_low in line.lower():
                selected.add(index)

    if not selected:
        return dataset_text[:max_chars]
    return "\n".join(lines[index] for index in sorted(selected))[:max_chars]

def regenerate_if_bad(html: str, heading: str, recommended_product: str, product_whitelist: list[str], hybrid_dataset: str, style_guide: str, keyword: str, cfg: dict):
    """
    If the model mentions out-of-whitelist products OR a comparison section lacks 2 products,
    re-run with full context and a strong whitelist instruction.
    """
    if not html:
        return html

    # check for non-whitelisted mentions
    bad = violates_whitelist(html, product_whitelist)

    # check if a comparison-like heading contains at least 2 whitelisted products
    needs_comparison = any(k in (heading or "").lower() for k in ["compare", "versus", "vs", "difference"])
    mentioned = [p for p in product_whitelist if p.lower() in html.lower()]
    too_few = needs_comparison and (len(set(mentioned)) < 2)

    if bad:
        logging.warning(
            "[SECTION_VALIDATION] Keeping section without generic whitelist rerun: "
            "heading=%r candidates=%r",
            heading,
            bad[:8],
        )

    # Only comparison headings receive a paid retry, because they have an
    # objective requirement for at least two named products. Generic candidate
    # detection previously reran every detail section and added no clear value.
    if too_few:
        return generate_detailed_subheading_content(
            heading,
            hybrid_dataset[:12000],
            (style_guide or "") + "\n\n" + whitelist_note(product_whitelist),
            keyword,
            recommended_product,
            cfg,
            product_whitelist=product_whitelist,
            label=f"comparison-rerun:{heading}",
            max_tokens=900,
        )
    return html
    
def strip_best_prefixes(html: str, whitelist: list[str] | None = None) -> str:
    """
    Remove editorial 'Best ' prefixes from product mentions in generated HTML.

    What it fixes:
    - Table cells: <td>Best Product Name</td>
    - Prose: "the best Product Name is/was/offers/has..."
    - Attributes: data-product="Best Product Name"
    - Anchor text: >Best Product Name</a>

    Safety:
    - If whitelist is provided, ONLY strip when the remainder exactly matches a whitelisted product
      (case-insensitive). This prevents accidental edits of normal sentences.
    """
    if not html:
        return html or ""

    def _ok(after: str) -> bool:
        after_clean = re.sub(r"\s+", " ", (after or "")).strip()
        if not whitelist:
            return True
        return any(after_clean.lower() == w.lower() for w in whitelist)

    # 1) <td>Best X</td>
    td_rx = re.compile(r"(?is)(<td\b[^>]*>\s*)(Best\s+)([^<]+?)(\s*</td>)")

    def _td_repl(m: re.Match) -> str:
        before = m.group(1)
        product = (m.group(3) or "").strip()
        after = m.group(4)
        return f"{before}{product}{after}" if _ok(product) else m.group(0)

    html = td_rx.sub(_td_repl, html)

    # 2) Prose: "the best X is/was/offers/has/fits/..."
    # Keep this conservative so we don't rewrite random phrases like "best way to..."
    prose_rx = re.compile(
        r"(?i)\b(the\s+)?best\s+([A-Z][A-Za-z0-9].{2,80}?)\b"
        r"(?=\s+(is|was|offers|has|fits|packs|rolls|makes|feels|handles|works)\b)"
    )

    def _prose_repl(m: re.Match) -> str:
        candidate = (m.group(2) or "").strip()
        return candidate if _ok(candidate) else m.group(0)

    html = prose_rx.sub(_prose_repl, html)

    # 3) data-product="Best X"
    attr_rx = re.compile(r'(?is)(\bdata-product\s*=\s*")\s*(Best\s+)([^"]+?)\s*(")')

    def _attr_repl(m: re.Match) -> str:
        prod = (m.group(3) or "").strip()
        if _ok(prod):
            return f'{m.group(1)}{prod}{m.group(4)}'
        return m.group(0)

    html = attr_rx.sub(_attr_repl, html)

    # 4) Anchor text: >Best X</a>
    a_text_rx = re.compile(r"(?is)(>)(\s*Best\s+)([^<]+?)(\s*</a>)")

    def _a_text_repl(m: re.Match) -> str:
        prod = (m.group(3) or "").strip()
        if _ok(prod):
            return f"{m.group(1)}{prod}{m.group(4)}"
        return m.group(0)

    html = a_text_rx.sub(_a_text_repl, html)

    return html
    


def extract_h2_candidates(cleaned_text: str) -> list[str]:
    out = []
    for ln in (cleaned_text or "").splitlines():
        if not ln.startswith("[H2] "):
            continue
        h2 = ln.replace("[H2] ", "").strip()

        # NEW: skip listicle/year headings
        if re.match(r"(?i)^(best|top|our\s+picks|recommended)\b.*\b(19\d{2}|20\d{2})\b$", h2):
            continue

        out.append(h2)
    return out



def strip_sentence_tail(s: str) -> str:
    """
    Remove descriptive tail text from a product-like phrase.

    Examples:
      "The Dyson Hot+Cool HF1 is a sleek and smart fan heater" -> "The Dyson Hot+Cool HF1"
      "Dyson Pure Cool ... â€” Great for bedrooms" -> "Dyson Pure Cool ..."
      "Winix 5500-2 for large rooms" -> "Winix 5500-2"
    """
    if not s:
        return s

    x = re.sub(r"\s+", " ", s).strip()
    if not x:
        return x

    # 1) Cut at punctuation sentence breaks first
    x = re.split(r"[\.!\?]\s+", x, maxsplit=1)[0].strip()

    # 2) Cut at common description introducers
    # NOTE: includes for/to (your trim_product_sentence_tail had these; keep them!)
    m = re.search(
        r"(?i)\s+\b("
        r"is|are|was|were|"
        r"has|have|had|"
        r"features?|offers?|includes?|"
        r"with|including|"
        r"that|which|"
        r"for|to"
        r")\b\s+",
        x
    )
    if m and m.start() >= 8:
        x = x[:m.start()].strip(" -â€“â€”:;,.")
        return x

    # 3) Cut at dashy descriptors ( "â€” Great for ..." / "- Best ..." )
    m = re.search(r"\s*[â€“â€”-]\s+", x)
    if m and m.start() >= 8:
        return x[:m.start()].strip(" -â€“â€”:;,.")  # keep left side

    # 4) Soft cap extremely long strings
    parts = x.split()
    if len(parts) > 12:
        x = " ".join(parts[:12]).strip(" -â€“â€”:;,.")
    return x





def promote_h2_to_product_tags(
    cleaned_text: str,
    *,
    whitelist: list[str] | None = None,
    brand_lexicon: set[str] | None = None,
    cfg=None,
) -> str:
    """
    Converts:
      [H2] Candidate
    into:
      [PRODUCT] Canonical Name
    ONLY when the H2 looks like a real product name.

    - Uses your existing normalize_product_name + strip_serp_editorial_wrappers.
    - Uses looks_like_unique_product gate (brand/model cues).
    - If whitelist provided, maps to closest whitelist match.
    """
    if not cleaned_text:
        return cleaned_text or ""

    out = []
    for ln in cleaned_text.splitlines():
        if not ln.startswith("[H2] "):
            out.append(ln)
            continue

        raw = ln.replace("[H2] ", "").strip()

        cand = normalize_product_name(
            raw,
            whitelist=whitelist or None,
            cfg=cfg,
            allow_strip_best_prefix=True,
            try_map_to_whitelist=bool(whitelist),
            max_words=12,
        )
        cand = strip_serp_editorial_wrappers(cand)
        # Reject promo/UI headings as "products"
        if _looks_like_ui_phrase(cand):
            out.append(f"[H2] {raw}")
            continue

        # Promotion gate: must look like a unique product name
        if cand and looks_like_unique_product(cand, brand_lexicon=brand_lexicon, cfg=cfg):
            out.append(f"[PRODUCT] {cand}")
        else:
            # keep as H2 if not promotable (still useful as structure)
            out.append(f"[H2] {raw}")

    return "\n".join(out)

def trim_to_category_tail(name: str, *, cfg: dict | None = None) -> str:
    """
    If a product title contains a *generic category tail* (e.g. "Air Purifier", "Humidifier")
    and everything after that looks like marketing copy, trim to the end of the tail.

    CRITICAL: Do NOT trim if the remainder looks like it continues the product name,
    e.g. "Activated Carbon Fan and Purifier" should NOT be trimmed at "Activated Carbon".
    """
    cfg = cfg or {}
    s = (name or "").strip()
    if not s:
        return s

    generic_tails = [str(x).strip() for x in (cfg.get("generic_tails") or []) if str(x).strip()]
    if not generic_tails:
        return s

    s_l = s.lower()

    # Only allow trimming if the remainder begins with punctuation / separators (NOT just anything).
    allowed_after_prefixes = (
        ",", " -", " â€“", " â€”", ":", ";", "|", "â€¢", "(", "[", "{"
    )

    best_end = None

    for tail in generic_tails:
        t = tail.strip()
        if not t:
            continue
        t_l = t.lower()

        start = 0
        while True:
            idx = s_l.find(t_l, start)
            if idx == -1:
                break

            end = idx + len(t_l)
            remainder = s[end:].strip()

            # Exact tail at end is always safe
            if not remainder:
                best_end = end
                break

            # Preserve a short model/capacity extension placed after the category
            # tail, then drop the comma-separated marketing features.
            # Example: "Portable Air Conditioner 10000 BTU, 4-in-1 Cooling".
            raw_remainder = s[end:]
            identity_match = re.match(r"\s+([^,]{1,60}),", raw_remainder)
            if identity_match:
                identity_extension = identity_match.group(1).strip()
                if len(identity_extension.split()) <= 5 and re.search(r"\d", identity_extension):
                    best_end = end + identity_match.end(1)

            # Only trim if remainder clearly looks like a descriptor clause
            rem_l = remainder.lower()
            if remainder.startswith(allowed_after_prefixes) or rem_l.startswith(("for ", "with ", "featuring ", "includes ", "up to ", "covers ")):
                best_end = end

            start = idx + 1

    if best_end is not None:
        return s[:best_end].strip().rstrip(" .,:;!?â€“â€”-").strip()

    return s



def strip_editorial_suffixes(s: str) -> str:
    """
    Removes common editorial suffixes accidentally appended to product titles
    while keeping the real product identity intact.

    Examples:
      "Russell Hobbs RHSM1001-G: an In-Depth Look" -> "Russell Hobbs RHSM1001-G"
      "Russell Hobbs RHSM1001-G Review" -> "Russell Hobbs RHSM1001-G"
    """
    if not s:
        return s

    out = re.sub(r"\s+", " ", s).strip()
    if not out:
        return out

    # 1) Strong colon / dash editorial tails
    #    Keep this conservative and only trim when the right side is a known
    #    editorial phrase, not when it could still be part of the product name.
    editorial_tail_rx = re.compile(
        r"""(?ix)
        ^\s*(.*?)\s*
        (?:[:\-â€“â€”]\s*)
        (
            an?\s+in[-\s]?depth\s+look|
            in[-\s]?depth\s+look|
            first\s+look|
            hands[-\s]?on|
            overview|
            review|
            buying\s+guide|
            complete\s+guide|
            detailed\s+review|
            expert\s+review|
            full\s+review
        )\s*$
        """
    )
    m = editorial_tail_rx.match(out)
    if m:
        out = m.group(1).strip(" -â€“â€”:;,.")

    # 2) Whole-word / end-of-string suffixes (case-insensitive)
    suffixes = [
        "quality",
        "alternatives",
        "alternative",
        "review",
        "pros & cons",
        "pros and cons",
        "price",
        "key specs",
        "specs",
        "performance",
        "design and size",
        "filters",
        "room coverage",
        "sound",
        "power consumption",
        "customer service",
        "additional features",
        "in-depth look",
        "first look",
        "overview",
    ]

    lowered = out.lower().strip()

    # Remove repeatedly (handles "... Quality Alternatives" etc.)
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            suf_l = suf.lower()
            if lowered.endswith(": " + suf_l):
                out = out[: -(len(suf) + 2)].strip()
                lowered = out.lower().strip()
                changed = True
            elif lowered.endswith(" - " + suf_l) or lowered.endswith(" â€“ " + suf_l) or lowered.endswith(" â€” " + suf_l):
                out = re.sub(r"(?i)\s*[-â€“â€”]\s*" + re.escape(suf) + r"\s*$", "", out).strip()
                lowered = out.lower().strip()
                changed = True
            elif lowered.endswith(" " + suf_l):
                out = out[: -(len(suf) + 1)].strip()
                lowered = out.lower().strip()
                changed = True
            elif lowered == suf_l:
                out = ""
                lowered = ""
                changed = True

    return out
def _norm_simple(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\r\n-â€“â€”:â€¢|,.;")
    return s

def is_category_only_product_name(name: str, *, cfg: dict | None = None) -> bool:
    """
    True if the "name" is basically just the category label, not a specific model.

    Config-driven (no hardcoded defaults):
      - category_only_blocklist: list[str] (preferred explicit category-only labels)
      - generic_tails: list[str] (also treated as category-only when alone)
    """
    cfg = cfg or {}
    s = _norm_simple(name)
    if not s:
        return True

    # Prefer explicit per-topic blocklist; fall back to generic_tails (which should exist per topic)
    block = [
        _norm_simple(x)
        for x in (cfg.get("category_only_blocklist") or [])
        if str(x).strip()
    ]
    tails = [
        _norm_simple(x)
        for x in (cfg.get("generic_tails") or [])
        if str(x).strip()
    ]

    # Exact category-only
    if s in block or s in tails:
        return True

    # Handle simple pluralization against either list
    if s.endswith("s"):
        base = s[:-1]
        if base in block or base in tails:
            return True

    return False

def is_bad_top_pick_candidate(name: str, *, brand_lexicon=None, cfg=None) -> bool:
    """
    Final guardrail so headings/UI/rating labels never become top picks.
    """
    if not name:
        return True

    s = re.sub(r"\s+", " ", name).strip()

    if re.search(r"https?://", s, re.I):
        return True
        
    if looks_like_date_label(s):
        return True

    if looks_like_rating_label(s):
        return True

    if _looks_like_ui_phrase(s):
        return True

    if looks_like_testimonial_heading(s):
        return True

    if is_category_only_product_name(s, cfg=cfg):
        return True

    if not looks_like_unique_product(s, brand_lexicon=brand_lexicon, cfg=cfg):
        return True

    return False

def pick_specific_from_whitelist(
    product_whitelist,
    brand_lexicon=None,
    cfg=None,
    keyword: str = "",
):
    cfg = cfg or {}
    best = ""
    best_score = float("-inf")

    brand_set = {str(b).lower() for b in (brand_lexicon or []) if b}

    def _has_known_brand(x: str) -> bool:
        toks = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", x or "")
        if not toks or not brand_set:
            return False
        return toks[0].lower() in brand_set

    def _score_candidate(x: str) -> float:
        s = re.sub(r"\s+", " ", (x or "")).strip()
        if not s:
            return float("-inf")
            
        if looks_like_date_label(s):
            return float("-inf")

        score = 0.0
        words = s.split()
        
        # Slight penalty for ALL CAPS titles (often scraped headings)
        if s.isupper():
            score -= 2

        # Strong reward if any known brand appears in the name
        if brand_set:
            s_lower = s.lower()
            if any(b in s_lower for b in brand_set):
                score += 10.0

        if re.search(r"\d", s):
            score += 4.0

        if re.search(r"\b[A-Za-z]*\d+[A-Za-z0-9\-\/]*\b", s):
            score += 4.0

        if _has_strong_product_cues(s, brand_lexicon=brand_lexicon, cfg=cfg):
            score += 4.0

        if looks_like_unique_product(s, brand_lexicon=brand_lexicon, cfg=cfg):
            score += 4.0

        if len(words) < 2:
            score -= 8
        elif 3 <= len(words) <= 8:
            score += 3

        return score

    for p in (product_whitelist or []):
        if not p:
            continue

        cand = str(p).strip()
        cand = strip_serp_editorial_wrappers(cand, cfg=cfg).strip()
        cand = strip_section_wrapper_prefix(cand, cfg=cfg).strip()
        cand = strip_leading_review_verbs(cand).strip()
        cand = strip_sentence_tail(cand).strip()
        cand = strip_leading_article(cand).strip()
        cand = strip_editorial_suffixes(cand).strip()
        cand = strip_url_and_retailer_chrome(cand).strip()
        cand = smart_title_case(cand).strip()

        if not cand:
            continue
            
        # NEW: reject date-like strings before scoring
        if looks_like_date_label(cand):
            continue

        if is_bad_top_pick_candidate(cand, brand_lexicon=brand_lexicon, cfg=cfg):
            continue
        if keyword and not reviewed_product_matches_keyword(cand, keyword, cfg):
            continue

        score = _score_candidate(cand)

        if score > best_score:
            best = cand
            best_score = score

    return best
    
def find_best_product_phrase_in_line(
    line: str,
    brand_lexicon=None,
    cfg=None,
    max_words: int = 10
) -> str | None:
    """
    Extract a product-looking phrase from a single evidence line.

    IMPORTANT:
    - We must return something that appears VERBATIM in the evidence line.
    - Avoid returning sentence-like chunks ("X is a sleek ...").
    """
    if not line:
        return None
    cfg = cfg or {}

    # 1) Hard stop at first sentence break (keeps verbatim)
    #    e.g. "The Dyson ... heater." -> only consider left side.
    hard_split = re.split(r"[\.!\?]\s+", line, maxsplit=1)
    cand_line = hard_split[0].strip() if hard_split else line.strip()

    # 2) Soft stop at common â€œsentence tailâ€ introducers.
    #    Keep the LEFT side only so any chunk we return remains verbatim.
    #    Example: "The Dyson Hot+Cool HF1 is a sleek..." -> "The Dyson Hot+Cool HF1"
    cand_line = re.split(
        r"(?i)\b("
        r"is|are|was|were|"
        r"has|have|had|"
        r"features|feature|offers|offer|includes|include|"
        r"with|including|"
        r"that|which"
        r")\b",
        cand_line,
        maxsplit=1
    )[0].strip()

    if not cand_line:
        return None

    tokens = cand_line.split()
    if len(tokens) < 2:
        return None

    # Prefer longer phrases first
    for n in range(min(max_words, len(tokens)), 2, -1):
        for i in range(0, len(tokens) - n + 1):
            chunk = " ".join(tokens[i:i+n]).strip().strip(".,;:!?()[]{}\"'")

            if not chunk:
                continue
            if looks_like_generic_headline(chunk, brand_lexicon=brand_lexicon, cfg=cfg):
                continue
            if _looks_like_ui_phrase(chunk):
                continue

            # Must be product-ish
            if (
                looks_like_product_title(chunk, brand_lexicon=brand_lexicon, cfg=cfg)
                or looks_like_unique_product(chunk, brand_lexicon=brand_lexicon, cfg=cfg)
            ):
                return chunk

    return None

def rescue_product_from_evidence_line(
    evidence: str,
    *,
    brand_lexicon=None,
    cfg=None,
    max_words: int = 8
) -> str:
    """
    Last-chance rescue: if DeepSeek returned a weak/non-grounded product, try to
    recover a real product phrase directly from the repaired evidence line.
    """
    if not evidence:
        return ""

    cand = find_best_product_phrase_in_line(
        evidence,
        brand_lexicon=brand_lexicon,
        cfg=cfg,
        max_words=max_words,
    ) or ""

    cand = strip_url_and_retailer_chrome(cand).strip()
    cand = strip_heading_marker_prefix(cand).strip()
    cand = strip_sentence_tail(cand).strip()
    cand = normalize_all_caps_product(cand).strip()

    if not cand:
        return ""

    if looks_like_blog_metadata(cand) or is_bad_top_pick_candidate(
        cand, brand_lexicon=brand_lexicon, cfg=cfg
    ):
        return ""

    return cand

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Text utilities
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# (Regex constants for headline/listicle rejection are defined in the
# 'Category-agnostic generic headline / listicle rejection' section below.)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# NEW: PDP/Amazon title tail trimming
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_MARKETING_TAIL_RX = re.compile(
    r"""(?ix)
    \b(
        removing\s+up\s+to|
        removes?\s+up\s+to|
        removes?\s+up\b|
        eliminates?\b|
        gets?\s+rid\s+of\b|
        helps?\s+(?:remove|reduce)\b|
        filter(?:s|ing)?\s+out\b|
        covers?\s+up\s+to\b|
        up\s+to\s+\d+|
        # common truncation words at the end of broken titles
        up\s+to\s*$
    )
    """,
)

_SEP_SPLIT_RX = re.compile(r"\s*(\||â€¢)\s*")  # split on pipes/bullets used in Amazon titles

def trim_product_title_tail(name: str, *, max_comma_segments: int = 2) -> str:
    """
    Trim Amazon/PDP-style marketing tails from titles.

    Rules:
    - If there's a '|' (or bullet), keep only left side.
    - If marketing tail phrase appears, cut at its start.
    - If there are many commas, keep only first N segments (default 2).
      (This preserves: 'Brand Model, Variant Name' but drops feature lists.)
    - Final cleanup of trailing punctuation and whitespace.
    """
    if not name:
        return ""

    s = re.sub(r"\s+", " ", str(name)).strip()

    # 1) Split on pipes/bullets: "Product | Removes Up To ..." -> "Product"
    parts = _SEP_SPLIT_RX.split(s, maxsplit=1)
    if parts:
        s = parts[0].strip()

    # 2) Kill obvious marketing tails like "Removing up to", "Removes up to", etc.
    m = _MARKETING_TAIL_RX.search(s)
    if m:
        s = s[:m.start()].strip()

    # 3) Comma-heavy titles usually turn into feature lists. Keep only first 2 segments.
    # Example:
    # "Home APO50371DG, Connected 4-stage HEPA Air Filter Cleaner Removing Up to"
    # -> keep 2 segments: "Home APO50371DG, Connected 4-stage HEPA Air Filter Cleaner"
    comma_count = s.count(",")
    if comma_count >= 3:
        max_keep = 2  # very likely a feature list
    elif comma_count >= 1:
        max_keep = max_comma_segments
    else:
        max_keep = 999

    segs = [p.strip() for p in s.split(",") if p.strip()]
    if len(segs) > max_keep:
        s = ", ".join(segs[:max_keep]).strip()


    # 4) Also trim dash tails that often introduce marketing blurbs
    # (Only if the right side looks like marketing copy.)
    for dash in [" â€” ", " â€“ ", " - "]:
        if dash in s:
            left, right = s.split(dash, 1)
            r = right.strip().lower()
            if any(k in r for k in ["remov", "eliminat", "up to", "filter", "mode", "pet", "odor", "coverage"]):
                s = left.strip()

    return s.strip().rstrip(" .,:;!?â€“â€”-").strip()

import re

def drop_redundant_comma_category(name: str, *, cfg: dict | None = None) -> str:
    """
    If a title looks like:
        "<Brand Model>, <Category plural> for <marketing...>"
    and the left side already contains the same category concept,
    drop the comma clause entirely.

    Config-driven:
      - generic_tails: list[str] (REQUIRED per topic; no hardcoded fallbacks)
    """
    cfg = cfg or {}
    s = (name or "").strip()
    if "," not in s:
        return s

    left, right = [p.strip() for p in s.split(",", 1)]
    if not left or not right:
        return s

    generic_tails = [
        str(x).strip().lower()
        for x in (cfg.get("generic_tails") or [])
        if str(x).strip()
    ]

    # No topic-specific hardcoded fallback: if not configured, do nothing
    if not generic_tails:
        return s

    left_l = left.lower()
    right_l = right.lower()

    # If the comma clause starts with a generic tail (often category text),
    # and the left already contains that same tail (singular/plural), drop it.
    for t in generic_tails:
        if right_l.startswith(t):
            t_sing = t[:-1] if t.endswith("s") else t
            t_plur = t if t.endswith("s") else (t + "s")
            if (t_sing in left_l) or (t_plur in left_l):
                return left

    return s


def clean_generated_text(text, heading=None, cfg=None):
    cfg = cfg or {}
    lines = text.strip().splitlines()
    cleaned = []
    normalized_heading = heading.strip().lower() if heading else ""

    generic_strip = set([h.lower() for h in cfg.get("generic_headings_to_strip", [])])

    for line in lines:
        line_stripped = line.strip()
        # Remove full heading tags like <h1>Heading</h1>
        if re.match(r"<h[1-6]>.*</h[1-6]>", line_stripped, re.IGNORECASE):
            continue
        # Remove isolated heading tags
        line_stripped = re.sub(r"</?h[1-6]>", "", line_stripped, flags=re.IGNORECASE).strip()
        lower_line = line_stripped.lower()
        # Remove markdown headings
        if re.match(r"^#{1,6}\s", line_stripped):
            continue
        # Drop if equals the current heading
        if heading and lower_line == normalized_heading:
            continue
        # Drop generic, config-driven headings
        if lower_line in generic_strip:
            continue
        # Remove registered trademark symbol
        line_stripped = line_stripped.replace("Â®", "")
        cleaned.append(line_stripped)

    return "\n".join(cleaned).strip()

def smart_title_case(name):
    lowercase_exceptions = {'of', 'and', 'in', 'on', 'the', 'for', 'with', 'a', 'an', 'to'}
    def title_word(word, is_first):
        if re.fullmatch(r'[A-Z0-9\-]+', word):
            return word
        elif not is_first and word.lower() in lowercase_exceptions:
            return word.lower()
        else:
            return word[:1].upper() + word[1:]
    words = re.split(r'(\s+)', name)
    titled = [title_word(word, i == 0) if word.strip() else word for i, word in enumerate(words)]
    return ''.join(titled)

def normalize_all_caps_product(name: str) -> str:
    """
    Convert ALL-CAPS product titles into normal title case
    without breaking model codes.

    Example:
        FLOYD CABIN BAG -> Floyd Cabin Bag
        LEVOIT CORE 300-P -> Levoit Core 300-P
        AX71-304GY -> AX71-304GY
    """
    if not name:
        return name

    s = name.strip()

    # Only transform if the whole string is uppercase
    if not s.isupper():
        return s

    words = []
    for w in s.split():
        # Preserve model codes containing digits or hyphens
        if re.search(r"\d", w) or "-" in w:
            words.append(w)
        else:
            words.append(w.capitalize())

    return " ".join(words)
    
def strip_leading_index(name: str) -> str:
    """
    Remove leading list indices like '1.', '2)', '3 Alaska', '2Mzoo' etc.

    BUT preserve genuine model patterns like:
      '2 In 1 ...'
      '3-in-1 ...'
      '4 in 1 ...'
    """
    if not name:
        return name

    s = name.strip()

    # âœ… PROTECT: do NOT strip leading numbers that form "X in Y" / "X-in-Y" model phrases.
    if re.match(r"(?i)^\s*\d+\s*(?:-?\s*in\s*-?\s*)\d+\b", s):
        return s

    # Strip "1 ", "2.", "3)" etc with a real separator
    s2 = re.sub(r'^\s*\d+[\.\)]?\s+', '', s)

    # Strip mashed-together variants like "1alaska", "2Mzoo"
    s2 = re.sub(r'^\s*\d+(?=[A-Za-z])', '', s2)

    return s2.strip()

    
def normalize_product_name(
    name: str,
    whitelist: list[str] | None = None,
    *,
    cfg: dict | None = None,
    allow_strip_best_prefix: bool = True,
    try_map_to_whitelist: bool = True,
    max_words: int | None = 12,
) -> str:
    """
    Canonicalize a product name safely.

    Goals:
    - Remove indexing artifacts ("1.", "2)", "3Alaska")
    - Remove editorial award labels ("Best Overall: X", "Top Pick: X")
    - Remove trailing regional fluff ("in the UK", "UK")
    - Remove leading "Best " ONLY if remainder is a real product (whitelist match) when whitelist provided
    - Optionally map fuzzy names to closest whitelist match (token overlap)

    Never deletes meaningful model words (no category-word nuking).
    """
    if not name:
        return ""

    cfg = cfg or {}
    s = str(name).strip()
    s = strip_editorial_product_name_suffix(s)
    
    # 0) Strip literal heading tokens that sometimes leak from evidence like "H1 Product Title"
    s = re.sub(r"(?i)^\s*(?:h[1-6]\s+)+", "", s).strip()

    # 0) Remove placeholders like <<PRODUCT_1>> BEFORE any other logic (incl. whitelist mapping)
    s = strip_product_placeholders(s)

    # 0a) Remove SERP/editorial wrappers early (before other logic)
    s = strip_serp_editorial_wrappers(s,cfg=cfg)

    # 0b) Trim Amazon/PDP marketing tails and comma feature lists
    s = trim_product_title_tail(s, max_comma_segments=2)

    # âœ… NEW: drop redundant comma category clauses like:
    # "AIRTOK Air Purifier, Air Purifiers for Bedroom Home" -> "AIRTOK Air Purifier"
    # This runs AFTER tail trimming and BEFORE other wrapper stripping / mapping.
    s = drop_redundant_comma_category(s, cfg=cfg)

    # 0c) Strip wrapper/section prefixes like "Conclusion: ..." etc.
    s = strip_section_wrapper_prefix(s, cfg=cfg)

    # 1) Remove leading indices
    s = strip_leading_index(s)

    # 2) Strip award-label prefixes like "Best Overall: X", "Also Great: X" -> X
    if ":" in s:
        left, right = s.split(":", 1)
        if re.search(r"\b(best|top|our|also)\b", left, re.I):
            # specifically allow "also great", "also good", etc.
            s = right.strip()


    # 3) Remove common editorial prefixes that are NOT part of the product name
    # Covers:
    # - "Best Overall: X"
    # - "Top Pick - X"
    # - "Also Great: X"
    # - "Also Great X"
    s = re.sub(
        r"(?i)^(?:"
        r"(best|top|our)\s+(overall|pick|choice|budget|premium|upgrade|runner[-\s]?up)"
        r"|also\s+(great|good|solid|notable|recommended)"
        r")\b[:\-â€“â€”]?\s*",
        "",
        s,
    ).strip()


    # 4) Remove trailing region tags (common in SERP headings)
    s = re.sub(r"(?i)\b(in the uk|in uk|uk)\b$", "", s).strip()

    # 5) Remove leading "Best " only if safe
    if allow_strip_best_prefix and re.match(r"(?i)^best\s+", s):
        remainder = re.sub(r"(?i)^best\s+", "", s).strip()
        if whitelist:
            if any(remainder.lower() == w.lower() for w in whitelist):
                s = remainder
        else:
            s = remainder

    # 6) Strip trailing punctuation noise
    s = s.strip().rstrip(" .,:;!?â€“â€”-").strip()

    # âœ… Save â€œbest cleanedâ€ baseline BEFORE whitelist mapping / max_words truncation
    baseline = s

    # 7) Optional: map fuzzy to closest whitelist match (useful for DeepSeek output)
    if whitelist and try_map_to_whitelist:
        mapped = closest_whitelist_match(s, whitelist, cfg=cfg)
        if mapped:
            mapped_s = str(mapped).strip()

            # âœ… NEW: don't let mapping turn a product into an editorial headline
            bad_map = False

            if ":" in mapped_s:
                left = mapped_s.split(":", 1)[0].strip().lower()
                if not re.search(r"(?i)\b(best|top|our)\b", left):
                    bad_map = True

            if re.search(r"(?i)\b(smell\s+test|can\s+the|should\s+you|your\s+next\s+read|"
                         r"release\s+date|price|key\s+specs|power\s+consumption|"
                         r"how\s+i\s+tested|how\s+we\s+tested|testing|test)\b", mapped_s):
                bad_map = True

            if not bad_map:
                s = mapped_s
            else:
                # revert to baseline (the pre-mapping cleaned product string)
                s = baseline


    # 8) Optional: cap word count (avoid DeepSeek rambles)
    if max_words is not None:
        words = s.split()
        if len(words) > max_words:
            s = " ".join(words[:max_words]).strip(" .,:;!?â€“â€”-")

    # âœ… Revised seatbelt: if mapping/truncation collapsed to 1 token, revert to baseline
    # BUT only when baseline has model/distinctive cues (digits or MODEL_TOKEN_RX).
    if s and len(s.split()) == 1:
        if re.search(r"\d", baseline) or MODEL_TOKEN_RX.search(baseline):
            s = baseline

    return s.strip()


def clean_product_name(name: str, cfg: dict | None = None) -> str:
    """
    Backwards-compatible wrapper.
    Your pipeline still calls clean_product_name() in a few places,
    but the canonical cleaner is normalize_product_name().
    """
    return normalize_product_name(
        name or "",
        whitelist=None,
        cfg=cfg or {},
        allow_strip_best_prefix=True,
        try_map_to_whitelist=False,
        max_words=14,  # slightly generous for structured titles
    ).strip()

    
def strip_best_prefix_from_name(name: str, whitelist: list[str] | None = None) -> str:
    """
    Remove a leading 'Best ' ONLY when it cleanly maps to a real product name.
    - If whitelist is provided: only strip when remainder exactly equals a whitelist item (case-insensitive).
    - If whitelist is not provided: strip unconditionally when it starts with 'Best '.
    """
    if not name:
        return name

    s = name.strip()
    if not re.match(r"(?i)^best\s+", s):
        return s

    remainder = re.sub(r"(?i)^best\s+", "", s).strip()

    # If we have a whitelist, only strip if remainder is an exact product
    if whitelist:
        if any(remainder.lower() == w.lower() for w in whitelist):
            return remainder
        return s  # don't strip if it doesn't become a real product

    # No whitelist: assume it's editorial and strip
    return remainder

# Token normalization for whitelist matching
_NORM_TOKEN_RX = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?")

def norm_tokens(s: str) -> list[str]:
    """
    Tokenize product-ish strings for fuzzy whitelist matching.
    Keeps hyphenated tokens like 'wh-1000xm5' or '3-in-1'.
    """
    if not s:
        return []
    toks = [t.lower().strip("-") for t in _NORM_TOKEN_RX.findall(str(s))]
    # Keep this conservative: just drop ultra-common glue words
    drop = {"the", "and", "or", "for", "with", "a", "an", "of", "to", "in", "on"}
    return [t for t in toks if t and t not in drop]

_DISTINCTIVE_TOKEN_RX = re.compile(
    r"(\d|[A-Za-z]+\d|\d+[A-Za-z]|[A-Za-z0-9]+-\d+[A-Za-z0-9-]*)"
)

def is_distinctive_token(tok: str) -> bool:
    """
    Heuristic: token looks like a model/variant cue (digits, alnum mixes, hyphen+digits).
    """
    return bool(_DISTINCTIVE_TOKEN_RX.search(tok or ""))



def closest_whitelist_match(
    name: str,
    whitelist: List[str],
    *,
    cfg: dict | None = None,
) -> Optional[str]:
    """
    Find closest match in whitelist based on token overlap.

    âœ… Key behavior:
    - If the input contains distinctive/model tokens (digits, model codes, hyphenated codes),
      require a candidate to share at least one of those tokens.
    - If multiple candidates share the SAME (brand + distinctive/model token set),
      prefer the SHORTER (less editorial) variant, even if the longer one has
      slightly higher raw token overlap (e.g. "... Highlights", "... Review").
    """
    cfg = cfg or {}

    n_tokens = norm_tokens(name)
    if not n_tokens:
        return None

    n_set = set(n_tokens)

    # Distinctive tokens present in the *input* name (e.g., HE400, 3-in-1, mk2, etc.)
    name_distinctive = {t for t in n_set if is_distinctive_token(t)}

    # Brand + model-key for the input, when possible
    input_brand = n_tokens[0]
    input_key = None
    if name_distinctive:
        input_key = (input_brand, tuple(sorted(name_distinctive)))

    best = None
    best_overlap = 0
    best_distinctive_overlap = 0
    best_tokens_len = None
    best_model_exact = False  # whether best shares the same brand+model-key as input

    for w in whitelist:
        w_raw = (w or "").strip()
        if not w_raw:
            continue

        # Never fuzzy-map a valid product name to a CTA, search widget, or
        # editorial heading merely because several product tokens overlap.
        if _looks_like_ui_phrase(w_raw):
            continue

        # âœ… never map TO obvious editorial/headline-ish items
        # (keep conservative; we only block very common wrappers)
        if re.search(
            r"(?i)\b("
            r"smell\s+test|the\s+smell\s+test|"
            r"can\s+the|should\s+you|your\s+next\s+read|"
            r"release\s+date|price|key\s+specs|specs|"
            r"power\s+consumption|is\s+it\s+worth|worth\s+the\s+price|"
            r"how\s+i\s+tested|how\s+we\s+tested|testing|test|"
            r"highlights|highlight\b"
            r")\b",
            w_raw,
        ):
            continue

        # reject colon-headlines unless they are award labels like "Best Overall: X"
        if ":" in w_raw:
            left = w_raw.split(":", 1)[0].strip().lower()
            if not re.search(r"(?i)\b(best|top|our)\b", left):
                continue

        w_tokens = norm_tokens(w_raw)
        if not w_tokens:
            continue

        w_set = set(w_tokens)
        overlap_set = n_set & w_set
        overlap = len(overlap_set)
        if overlap < 2:
            continue

        # If input has distinctive/model tokens, candidate must share at least one of them.
        if name_distinctive and not (overlap_set & name_distinctive):
            continue

        brand_overlap = (n_tokens[0] == w_tokens[0])
        distinctive_overlap = sum(1 for t in overlap_set if is_distinctive_token(t))
        if not brand_overlap and distinctive_overlap == 0:
            continue

        # Does this candidate share the *same* brand+model-key as the input?
        cand_distinctive = {t for t in w_set if is_distinctive_token(t)}
        cand_key = None
        if cand_distinctive:
            cand_key = (w_tokens[0], tuple(sorted(cand_distinctive)))

        model_exact = bool(input_key and cand_key == input_key)

        w_len = len(w_tokens)

        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Selection / tie-break rules
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # 1) Prefer candidates that match the same brand+model-key as the input.
        #    Among those, prefer the shorter name (less editorial).
        if model_exact:
            if (not best_model_exact) or (best_tokens_len is not None and w_len < best_tokens_len):
                best = w
                best_model_exact = True
                best_overlap = overlap
                best_distinctive_overlap = distinctive_overlap
                best_tokens_len = w_len
            continue

        # 2) If best is already model-exact, never replace it with a non-exact match.
        if best_model_exact:
            continue

        # 3) Otherwise fall back to overlap scoring: overlap, then distinctive overlap, then shorter.
        if (
            overlap > best_overlap
            or (overlap == best_overlap and distinctive_overlap > best_distinctive_overlap)
            or (
                overlap == best_overlap
                and distinctive_overlap == best_distinctive_overlap
                and (best is None or (best_tokens_len is not None and w_len < best_tokens_len))
            )
        ):
            best = w
            best_overlap = overlap
            best_distinctive_overlap = distinctive_overlap
            best_tokens_len = w_len
            best_model_exact = False

    return best




   

def _has_non_generic_model_word(tokens: list[str], cfg: dict | None = None) -> bool:
    """
    Returns True if `tokens` contains at least one "meaningful" (non-generic) word.

    Category-agnostic:
    - Generic adjectives and generic nouns are config-driven.
    - Model-ish tokens (digits, hyphenated model codes, acronyms) count as meaningful.
    - Otherwise, require a token with some length (default >=4) that's not generic.

    Config keys (all optional):
      - generic_adjectives: ["best","great","lightweight",...]
      - generic_nouns: ["luggage","suitcase","vacuum",...]
      - min_meaningful_token_len: 4
      - acronym_allowlist: ["tsa","hepa","usb",...]  (lowercased recommended)
    """
    cfg = cfg or {}

    generic_adjectives = {
        str(x).strip().lower()
        for x in (cfg.get("generic_adjectives") or [])
        if str(x).strip()
    }
    generic_nouns = {
        str(x).strip().lower()
        for x in (cfg.get("generic_nouns") or [])
        if str(x).strip()
    }
    min_len = int(cfg.get("min_meaningful_token_len") or 4)

    acronym_allowlist = {
        str(x).strip().lower()
        for x in (cfg.get("acronym_allowlist") or [])
        if str(x).strip()
    }

    def _is_modelish(tok: str) -> bool:
        if not tok:
            return False
        # digits anywhere (versions/sizes/models)
        if any(ch.isdigit() for ch in tok):
            return True
        # hyphenated token (common model/standard marker)
        if "-" in tok:
            return True
        # category-specific acronyms (HEPA, TSA, USB-C, etc.) after normalization
        if tok.lower() in acronym_allowlist:
            return True
        # conservative fallback: short consonant-heavy acronyms (optional)
        if tok.isalpha() and 2 <= len(tok) <= 4 and not re.search(r"[aeiou]", tok.lower()):
            return True
        return False

    for t in tokens or []:
        tl = str(t).strip().lower()
        if not tl:
            continue

        # model-ish tokens are meaningful even if short (e.g., "X5", "HEPA", "USB-C")
        if _is_modelish(t.strip()):
            return True

        # skip configured generic words
        if tl in generic_adjectives:
            continue
        if tl in generic_nouns:
            continue

        # otherwise require some length to avoid noise like "pro", "max" unless you want to config them
        if len(tl) >= min_len:
            return True

    return False

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Category-agnostic generic headline / listicle rejection
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_YEAR_RX = re.compile(r"\b(19\d{2}|20\d{2})\b")

_LISTICLE_WORDS_RX = re.compile(
    r"""(?ix)
    \b(
        best|top|our\s+picks?|picks?|recommended|favorites|
        buying\s+guide|buyer(?:'|â€™)s\s+guide|guide|
        roundup|comparison|vs|versus|
        to\s+buy|you\s+can\s+buy|worth\s+it|
        for\s+\d{4}|in\s+\d{4}
    )\b
    """
)

# Stronger â€œheadline-ishâ€ patterns
_BUY_IN_YEAR_RX = re.compile(r"(?ix)\b(to\s+buy|you\s+can\s+buy|buy)\b.*\b(19\d{2}|20\d{2})\b")
_BEST_FOR_YEAR_RX = re.compile(r"(?ix)\b(best|top|our\s+picks?)\b.*\b(for|in)\b.*\b(19\d{2}|20\d{2})\b")

# Pure â€œCategory + Yearâ€ (e.g., â€œAir Purifiers 2025â€, â€œRobot Vacuums 2026â€)
_CATEGORY_YEAR_ONLY_RX = re.compile(r"(?i)^\s*[A-Za-z][A-Za-z &\-']+\s+(19\d{2}|20\d{2})\s*$")


def _has_strong_product_cues(name: str, brand_lexicon=None, cfg=None) -> bool:
    """
    Strong cues = evidence that this is a specific product, not a heading.

    âœ… Adds support for Brand + category-core token + generic tail
       e.g. "Mila Air Purifier" (no digits/model codes required).
    """
    if not name:
        return False
    cfg = cfg or {}

    s = re.sub(r"\s+", " ", str(name)).strip()
    if not s:
        return False

    tokens = _WORD_RX_SIMPLE.findall(s)
    if len(tokens) < 2:
        return False

    # Original strong cues: model tokens / distinctive patterns
    if MODEL_TOKEN_RX.search(s):
        return True
    if _DISTINCTIVE_RX.search(s):
        return True

    # âœ… NEW: Brand + category-core token + generic tail (3 tokens)
    category_core = {str(x).strip().lower() for x in (cfg.get("category_core_tokens") or []) if str(x).strip()}
    generic_tails = [str(x).strip().lower() for x in (cfg.get("generic_tails") or []) if str(x).strip()]
    generic_nouns = {str(x).strip().lower() for x in (cfg.get("generic_nouns") or []) if str(x).strip()}

    lower = s.lower()
    last = tokens[-1].lower()

    ends_generic = (last in generic_nouns) or any(lower.endswith(t) for t in generic_tails)

    if ends_generic and len(tokens) == 3:
        first = tokens[0].lower()
        mid = tokens[1].lower()

        def _heuristic_brand(tok: str) -> bool:
            return tok[:1].isalpha() and len(tok) >= 3 and tok not in generic_nouns

        in_lex = False
        if brand_lexicon:
            try:
                in_lex = first in {b.lower() for b in brand_lexicon}
            except Exception:
                in_lex = False

        has_brand = in_lex or _heuristic_brand(first)

        # âœ… "Mila Air Purifier" passes because mid=="air" is in cfg["category_core_tokens"]
        if has_brand and (mid in category_core):
            return True

    return False




def _has_min_product_identity_cues(
    name: str,
    *,
    brand_lexicon: set[str] | None = None,
    cfg: dict | None = None
) -> bool:
    """
    Minimal identity cues (brand/model-ish) WITHOUT calling looks_like_generic_headline().
    This exists to avoid recursion loops.
    """
    cfg = cfg or {}
    s = (name or "").strip()
    if not s:
        return False

    if _looks_like_ui_phrase(s):
        return False

    tokens = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", s)
    if len(tokens) < 2:
        return False

    # brand hit on first token
    has_brand = False
    if brand_lexicon and tokens:
        first = tokens[0].lower()
        has_brand = first in {b.lower() for b in brand_lexicon}

    # model-ish cues
    has_modelish = bool(re.search(r"\b[A-Za-z]+\d+[A-Za-z0-9]*\b|\b\d+[A-Za-z]+\b", s))
    has_hyphen_model = bool(re.search(r"\b[A-Za-z0-9]+-\d+[A-Za-z0-9-]*\b", s)) or (s.count("-") >= 2)

    # allowlisted acronyms (optional)
    acronym_allowlist = {str(x).strip().lower() for x in (cfg.get("acronym_allowlist") or []) if str(x).strip()}
    has_allow_acronym = any(tok.lower() in acronym_allowlist for tok in tokens)

    return bool(has_modelish or has_hyphen_model or (has_brand and (has_modelish or has_hyphen_model or has_allow_acronym)))

def looks_like_company_review_page(cleaned_text: str) -> bool:
    """
    Detect Trustburn/Trustpilot-style company review pages (not product pages).
    """
    t = (cleaned_text or "").lower()

    # Strong Trustburn / review-platform signals (from your example)
    signals = [
        "trustburn.com/reviews/",
        "start collecting reviews today",
        "claim your business",
        "business transparency",
        "try our chrome extension",
        "trustburn widget",
        "find out what customers think about",
        "get a comprehensive view of the company",
    ]
    hits = sum(1 for s in signals if s in t)
    if hits >= 2:
        return True

    # Another pattern: lots of "Reviews" + lots of short H2 headings + no model-ish tokens
    if "reviews" in t:
        # If there are many H2 headings and no digits/model tokens, likely testimonials
        h2_count = len(re.findall(r"(?im)^\s*h2\s+", cleaned_text or ""))
        has_modelish = bool(re.search(r"[A-Za-z]*\d+[A-Za-z0-9\-\/]*", cleaned_text or ""))
        if h2_count >= 3 and not has_modelish:
            return True

    return False

def looks_like_generic_headline(
    name: str,
    *,
    brand_lexicon: set[str] | None = None,
    cfg: dict | None = None
) -> bool:
    """
    Category-agnostic rejector for generic listicle/headline strings.

    Returns True when:
    - The string looks like a roundup/listicle/guide (often scraped as a â€œproductâ€)
    - AND it lacks strong product cues (brand/model signals)

    Examples rejected:
      - "Air Purifiers to Buy in 2026"
      - "Best Robot Vacuums 2025"
      - "Coffee Makers 2026"
      - "Buying Guide: Key Features to Look For"

    Examples NOT rejected:
      - "Sony WH-1000XM5"
      - "Levoit Core 600S Smart Air Purifier"
      - "Shark HP150UK"
    """
    s = (name or "").strip()
    if not s:
        return True

    t = re.sub(r"\s+", " ", s).strip().lower()

    # Strong headline patterns (to buy / best for year)
    if _BUY_IN_YEAR_RX.search(t) or _BEST_FOR_YEAR_RX.search(t):
        if not _has_min_product_identity_cues(s, brand_lexicon=brand_lexicon, cfg=cfg):
            return True

    # â€œCategory + Yearâ€ only (very common false positive)
    if _CATEGORY_YEAR_ONLY_RX.match(s):
        if not _has_min_product_identity_cues(s, brand_lexicon=brand_lexicon, cfg=cfg):
            return True

    # Broad listicle language + year
    if _YEAR_RX.search(t) and _LISTICLE_WORDS_RX.search(t):
        if not _has_min_product_identity_cues(s, brand_lexicon=brand_lexicon, cfg=cfg):
            return True

    # Listicle language even without a year (guide/roundup/comparison/etc.)
    if _LISTICLE_WORDS_RX.search(t):
        if not _has_min_product_identity_cues(s, brand_lexicon=brand_lexicon, cfg=cfg):
            return True

    return False



# Stronger "model cue" detector:
# - digits anywhere (X2, 28", 360, 55cm)
# - acronyms (TSA, ABS, PC)
# - alphanumeric model tokens (X100, S23)
# - hyphenated tokens ONLY if they include a digit (e.g., WH-1000XM5)
_DISTINCTIVE_RX = re.compile(r"(\d|[A-Z]{2,}|[A-Za-z]+\d|\d+[A-Za-z]+|[A-Za-z]+-\d+[A-Za-z0-9]*)")

_WORD_RX_SIMPLE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?")

def build_brand_lexicon_from_names(names: list[str], min_freq: int = 2) -> set[str]:
    counts: dict[str, int] = {}
    for name in names or []:
        s = (name or "").strip()
        if not s or is_obviously_not_a_product(s):
            continue
        tokens = _WORD_RX_SIMPLE.findall(s)
        if len(tokens) < 2:
            continue
        first = tokens[0].lower()
        counts[first] = counts.get(first, 0) + 1
    return {b for b, c in counts.items() if c >= min_freq}
 
def build_brand_lexicon_from_structured_products(structured_products, min_freq: int = 2) -> set[str]:
    counts: dict[str, int] = {}

    for e in structured_products or []:
        name = (e.get("product") or "").strip()
        if not name:
            continue

        # Avoid learning from Amazon headings / junk
        if is_obviously_not_a_product(name):
            continue

        # Relaxed: allow normal names like "Ninja Foodi Air Fryer"
        tokens = _WORD_RX_SIMPLE.findall(name)
        if len(tokens) < 2:
            continue

        first = tokens[0].strip()
        if len(first) < 2:
            continue

        key = first.lower()
        counts[key] = counts.get(key, 0) + 1

    return {b for b, c in counts.items() if c >= min_freq}

_SECTION_WRAPPER_PREFIX_RX_CACHE: dict[tuple[str, ...], re.Pattern] = {}

def _build_section_wrapper_prefix_rx(cfg: dict | None) -> re.Pattern:
    cfg = cfg or {}

    defaults = [
        "conclusion",
        "final verdict",
        "verdict",
        "summary",
        "recommendation",
        "our verdict",
        "closing thoughts",
        "also great",
        "great value",
        "also consider",
        "runner-up",
        "upgrade pick",
        "budget pick",
        "top pick",
        "best overall",
    ]

    labels = list(cfg.get("section_label_left_sides") or [])
    labels = [str(x or "").strip().lower() for x in labels if str(x or "").strip()]
    labels = list(dict.fromkeys(labels + defaults))  # cfg first, then defaults, dedup

    if not labels:
        return re.compile(r"^\b$")  # match nothing

    labels.sort(key=len, reverse=True)
    cache_key = tuple(labels)
    if cache_key in _SECTION_WRAPPER_PREFIX_RX_CACHE:
        return _SECTION_WRAPPER_PREFIX_RX_CACHE[cache_key]

    alts = "|".join(re.escape(x) for x in labels)

    rx = re.compile(
        rf"""(?ix)
        ^\s*
        (?:the\s+)?                # optional "the"
        (?:{alts})
        \s*[:\-â€“â€”]+\s*             # separator :, -, en-dash, em-dash
        """
    )
    _SECTION_WRAPPER_PREFIX_RX_CACHE[cache_key] = rx
    return rx


def looks_like_unique_product(
    name: str,
    brand_lexicon: set[str] | None = None,
    cfg: dict | None = None
) -> bool:
    """
    Decide if `name` looks like a specific product (Brand + Model), not a generic category phrase.
    Safe defaults are pulled from cfg inside the function.
    """
    cfg = cfg or {}

    def _norm_list(key: str) -> list[str]:
        v = cfg.get(key, [])
        if not isinstance(v, list):
            v = []
        return [str(x).strip().lower() for x in v if str(x).strip()]

    generic_tails = _norm_list("generic_tails")
    generic_nouns = set(_norm_list("generic_nouns"))
    generic_adjectives = set(_norm_list("generic_adjectives"))
    acronym_allowlist = set(_norm_list("acronym_allowlist"))

    s = (name or "").strip()
    if not s:
        return False
    if looks_like_legal_entity_name(s):
        return False


    # 0) Hard reject: generic headlines/listicles
    if looks_like_generic_headline(s, brand_lexicon=brand_lexicon, cfg=cfg):
        return False

    # 1) Reject question/guide starters
    if re.match(
        r"(?i)^(consider|how to|how|why|what|when|where|which|tips|tip|guide|choosing|"
        r"does|do|did|is|are|was|were|testing|test|reviewing|review|about)\b",
        s
    ):
        return False

    # 2) Reject wrapper/award/section prefixes (cfg-driven; no globals)
    try:
        section_rx = _build_section_wrapper_prefix_rx(cfg)
        if section_rx and section_rx.match(s):
            return False
    except Exception:
        pass

    tokens = _WORD_RX_SIMPLE.findall(s)
    if len(tokens) < 2:
        return False

    has_distinctive = bool(_DISTINCTIVE_RX.search(s))
    has_model_token = bool(MODEL_TOKEN_RX.search(s))

    first_tok = (tokens[0] or "").strip()
    first_low = first_tok.lower()

    def _heuristic_brand(tok: str) -> bool:
        return (
            tok[:1].isalpha()
            and len(tok) >= 3
            and tok not in generic_adjectives
            and tok not in generic_nouns
        )

    in_lex = False
    if brand_lexicon:
        try:
            in_lex = first_low in {b.lower() for b in brand_lexicon}
        except Exception:
            in_lex = False

    has_brand = in_lex or _heuristic_brand(first_low)

    # If it contains prepositions AND lacks brand/model cues, likely not product
    if re.search(r"(?i)\b(for|with|of)\b", s) and not has_brand and not has_model_token:
        return False

    if s.count(",") >= 1 and not has_brand and not has_model_token:
        return False

    # Sentence-like glue words without distinctive/model cues => reject
    if (not has_distinctive) and re.search(r"(?i)\b(of|your|for|with|without|and)\b", s):
        return False

    lower = s.lower().strip()
    ends_generic_tail = any(lower.endswith(t) for t in generic_tails)
    last_tok = tokens[-1].lower() if tokens else ""
    ends_generic_noun = (last_tok in generic_nouns)
    ends_generic = ends_generic_tail or ends_generic_noun
    
    # âœ… NEW: allow Brand + category-core token + generic tail (3 tokens)
    # e.g. "Mila Air Purifier" (mid token "Air" is only 3 chars)
    if ends_generic and has_brand and len(tokens) == 3:
        category_core = {str(x).strip().lower() for x in (cfg.get("category_core_tokens") or []) if str(x).strip()}
        mid = tokens[1].lower()
        if mid in category_core:
            return True


    # Allow: Brand + enough tokens + generic tail (e.g. "Levoit Core Mini Air Purifier")
    if ends_generic and has_brand and len(tokens) >= 4:
        return True

    if ends_generic and not _has_non_generic_model_word(tokens, cfg=cfg):
        return False

    if not has_distinctive:
        if not (has_brand and _has_non_generic_model_word(tokens[1:], cfg=cfg)):
            return False

    return True




def prune_product_whitelist(
    whitelist: list[str],
    *,
    brand_lexicon: set[str] | None = None,
    cfg: dict | None = None
) -> list[str]:
    """
    Clean and de-duplicate a product whitelist, keeping only names that pass
    the identity gate (looks_like_unique_product).

    Adds extra guards to prevent obvious SECTION HEADINGS from entering the whitelist.
    """
    cfg = cfg or {}
    out: list[str] = []
    seen: set[str] = set()

    # âœ… NEW: reject obvious non-product/section-heading phrases
    # (keep conservative; these frequently appear as H2/H3 headings)
    HEADING_PREFIX_RX = re.compile(
        "(?i)^(how|why|what|when|where|which|tips?|guide|choosing|overview|summary|conclusion|"
        r"final verdict|value for money|does|do|did|is|are|was|were|testing|test|about)\b"
    )
    HEADING_PHRASE_RX = re.compile(
        r"(?i)\b("
        r"key features|features\b|specs\b|specifications\b|at a glance|"
        r"how i tested|how do\b|how easy is it|how many\b|should you buy|"
        r"pros and cons|verdict|review\b|comparison\b|vs\b|versus\b"
        r")\b"
    )

    for p in (whitelist or []):
        raw = (p or "").strip()
        if not raw:
            continue

        # remove placeholders + leading bullets/dashes early
        cleaned = strip_product_placeholders(raw)
        cleaned = cleaned.lstrip(" \t\r\n-â€“â€”:â€¢|")

        # existing cleaning steps
        cleaned = strip_serp_editorial_wrappers(cleaned)
        cleaned = cleaned.strip().rstrip(" .,:;!?â€“â€”-")
        cleaned = trim_product_title_tail(cleaned, max_comma_segments=2)
        cleaned = smart_title_case(cleaned).strip()

        if not cleaned:
            continue

        # hard reject heading-style labels
        if re.match(r"(?i)^(for|best\s+for)\b", cleaned):
            continue

        # âœ… NEW: reject obvious section headings
        if HEADING_PREFIX_RX.match(cleaned):
            continue
        if HEADING_PHRASE_RX.search(cleaned):
            continue

        # extra guard: reject if still starts with a dash after all cleanup
        if cleaned.startswith(("-", "â€“", "â€”")):
            continue

        # cfg-driven identity gate
        if not looks_like_unique_product(cleaned, brand_lexicon=brand_lexicon, cfg=cfg):
            continue

        key = cleaned.lower()
        if key in seen:
            continue

        seen.add(key)
        out.append(cleaned)

    return out



def strip_serp_editorial_wrappers(name: str, cfg: dict | None = None) -> str:
    if not name:
        return name
    cfg = cfg or {}
    s = name.strip()

    # strip leading "Also Great: " / "Verdict: " etc. using cfg labels
    s = strip_section_wrapper_prefix(s, cfg=cfg)

    # Leading SERP wrappers
    s = re.sub(r"(?i)^\s*related\s+deals?\s+you\s+might\s+like\s+for\s+", "", s).strip()
    s = re.sub(r"(?i)^\s*deals?\s+you\s+might\s+like\s+for\s+", "", s).strip()
    s = re.sub(r"(?i)^\s*related\s+deals?\s+for\s+", "", s).strip()
    
    # Common review-page wrapper that isn't part of the product name
    # e.g. "Additional Details About the Lipault Lost in Berlin Cabin 2.0"
    s = re.sub(r"(?i)^\s*additional\s+details\s+about\s+(?:the\s+)?", "", s).strip()

    # Common "Review" wrappers (keep conservative)
    s = re.sub(r"(?i)^\s*review\s+of\s+", "", s).strip()

    # trailing wrappers
    s = re.sub(r"(?i)\s+review\s+summary\s*$", "", s).strip()
    s = re.sub(r"(?i)\s+summary\s*$", "", s).strip()
    s = re.sub(r"(?i)\s+review\s*$", "", s).strip()
    s = re.sub(r"(?i)\s+hands[-\s]?on\s+review\s*$", "", s).strip()

    return s.strip().rstrip(" .,:;!?â€“â€”-").strip()




def drop_empty_lists_and_fix_cells(html: str) -> str:
    # Remove empty ULs (whitespace or zero LIs)
    html = re.sub(r"<ul>\s*(?:<li>\s*</li>\s*)*\s*</ul>", "", html, flags=re.I)
    # Remove stray Pros/Cons labels that arenâ€™t followed by a list item
    html = re.sub(r"(?:<p>\s*(Pros|Cons)\s*:\s*</p>)", "", html, flags=re.I)
    # Replace empty table cells with em dash
    html = re.sub(r"<td>\s*</td>", "<td>â€”</td>", html, flags=re.I)
    return html

def remove_table_claims_without_table(html: str) -> str:
    """
    If the text explicitly introduces a comparison table but no table exists,
    soften ONLY those intro phrases.

    Minor correctness improvements:
    - Only targets explicit "here's a comparison table" type phrasing (not any 'table' mention).
    - Detects approved comparison table regardless of wrapper presence.
    - Treats markdown pipe tables as a "table exists" signal too (so we don't rewrite).
    """
    if not html:
        return html or ""

    # Detect an actual rendered comparison table (wrapped or not)
    has_html_table = bool(re.search(
        r'(?is)<table\b[^>]*\bclass=["\'][^"\']*\bcomparison-table\b[^"\']*["\'][^>]*>',
        html
    ))

    # Detect markdown pipe tables that may have slipped through
    has_md_table = bool(_MD_TABLE_BLOCK.search(html))

    has_any_table = has_html_table or has_md_table
    if has_any_table:
        return html

    # Only soften explicit â€œtable-introâ€ claims
    intro_rx = re.compile(
        r"(?is)\b("
        r"here(?:â€™|')?s\s+(?:a\s+)?(?:quick\s+)?comparison\s+table\s*:?"
        r"|comparison\s+table\s*:?"
        r"|the\s+table\s+below\s+shows\s*:?"
        r"|in\s+the\s+table\s+below\s*,?\s*"
        r"|see\s+the\s+table\s+below\s*:?"
        r")\b"
    )

    return intro_rx.sub("Here are the key differences:", html)


_MD_TABLE_BLOCK = re.compile(
    r"""(?mx)
    ^\s*\|(?P<header>.+?)\|\s*$\s*          # header row
    ^\s*\|(?P<sep>\s*[:-]+(?:\s*\|\s*[:-]+)+)\s*\|\s*$\s*  # --- style separator row
    (?P<body>(?:^\s*\|.+?\|\s*$\s*)+)       # one or more body rows
    """
)

def _split_md_row(line: str) -> list[str]:
    # split a markdown row | a | b | c |  -> ["a","b","c"]
    cells = [c.strip() for c in line.strip().strip('|').split('|')]
    return [c if c else "â€”" for c in cells]

def _strip_inline_lists(s: str) -> str:
    # Convert any <ul><li>..</li></ul> inside a cell to comma-separated text
    s = re.sub(r'(?is)</?ul[^>]*>', '', s)
    s = re.sub(r'(?is)</?li[^>]*>', ', ', s)
    s = re.sub(r'\s*,\s*,\s*', ', ', s)
    return re.sub(r'\s+', ' ', s).strip(' ,')

def _wrap_approved_inner(inner_html: str) -> str:
    # inner_html should be THEAD+TBODY only (no <table> tags)
    return (
        '<div class="comparison-table-wrap">'
        '<table class="comparison-table">'
        f'{inner_html}'
        '</table>'
        '</div>'
    )

def _wrap_existing_comparison_table(table_html: str) -> str:
    # table_html should already be <table class="comparison-table">...</table>
    return f'<div class="comparison-table-wrap">{table_html}</div>'

def wrap_unwrapped_comparison_tables(html: str) -> str:
    if not html:
        return html or ""

    wrapped_rx = re.compile(
        r'(?is)<div\s+class="comparison-table-wrap"[^>]*>\s*'
        r'<table[^>]*\bclass=["\'][^"\']*\bcomparison-table\b[^"\']*["\'][^>]*>.*?</table>\s*'
        r'</div>'
    )

    table_rx = re.compile(
        r'(?is)<table[^>]*\bclass=["\'][^"\']*\bcomparison-table\b[^"\']*["\'][^>]*>.*?</table>'
    )

    # Shield existing wrapped blocks
    kept = []
    def _shield(m):
        kept.append(m.group(0))
        return f"__WRAPPED_TABLE_{len(kept)-1}__"
    html = wrapped_rx.sub(_shield, html)

    # Wrap any remaining comparison tables
    html = table_rx.sub(lambda m: _wrap_existing_comparison_table(m.group(0)), html)

    # Restore shielded blocks
    for i, frag in enumerate(kept):
        html = html.replace(f"__WRAPPED_TABLE_{i}__", frag)

    return html


def _ensure_em_dash(cell_html: str) -> str:
    inner = re.sub(r'(?is)<[^>]+>', '', cell_html).strip()
    return cell_html if inner else "â€”"

def convert_markdown_tables_to_html(html: str) -> str:
    """
    Find markdown pipe tables and convert them into the APPROVED structure:
    <div class="comparison-table-wrap"><table class="comparison-table"><thead>..</thead><tbody>..</tbody></table></div>
    Keeps only 3â€“6 header columns and 2â€“4 body rows (else leave the block as-is).
    """
    def _repl(m: re.Match) -> str:
        header = _split_md_row(m.group('header'))
        body_raw = [ln for ln in m.group('body').splitlines() if ln.strip()]
        rows = [_split_md_row(ln) for ln in body_raw]

        # Guard rails
        if not (3 <= len(header) <= 6):
            return m.group(0)  # donâ€™t convert, leave text as-is
        # truncate/normalize rows to header length
        norm_rows = []
        for r in rows:
            if len(r) < len(header):
                r = r + ["â€”"] * (len(header) - len(r))
            elif len(r) > len(header):
                r = r[:len(header)]
            norm_rows.append(r)
        # keep 2â€“4 rows max
        norm_rows = norm_rows[:4]
        if len(norm_rows) < 2:
            return m.group(0)

        thead = "<thead><tr>" + "".join(f"<th>{escape(h or 'â€”')}</th>" for h in header) + "</tr></thead>"
        tb_rows = []
        for r in norm_rows:
            cells = "".join(f"<td>{escape(c or 'â€”')}</td>" for c in r)
            tb_rows.append(f"<tr>{cells}</tr>")
        tbody = "<tbody>" + "".join(tb_rows) + "</tbody>"
        return _wrap_approved_inner(thead + tbody)

    # Replace *each* markdown table block
    return _MD_TABLE_BLOCK.sub(_repl, html)


def upgrade_html_tables_to_approved(html: str) -> str:
    """
    Upgrade plain/legacy <table>â€¦</table> to approved wrapper/class, and ensure <thead>/<tbody>.
    Leaves non-table content untouched.
    """
    def _fix_table(m: re.Match) -> str:
        tbl = m.group(0)

        # If already has the approved table class, DO NOT add another <table>.
        # Wrapping can be handled in a separate pass if needed.
        if re.search(r'(?is)<table[^>]*\bclass=["\'][^"\']*\bcomparison-table\b[^"\']*["\']', tbl):
            return tbl


        # Extract rows
        rows = re.findall(r'(?is)<tr[^>]*>(.*?)</tr>', tbl)
        if not rows:
            return ""  # drop empty table

        # Extract cells per row
        parsed = []
        for r in rows:
            cells = re.findall(r'(?is)<t[hd][^>]*>(.*?)</t[hd]>', r)
            if not cells:
                cells = re.findall(r'(?is)<td[^>]*>(.*?)</td>', r)
            if not cells:
                continue
            parsed.append([_strip_inline_lists(c).strip() or "â€”" for c in cells])

        if len(parsed) < 2:
            return ""  # not enough rows to be a comparison

        header = parsed[0]
        body = parsed[1:]
        # Normalize header size: 3â€“6 cols
        if len(header) < 3:
            return ""  # too few columns to be useful
        header = header[:6]
        new_body = []
        for r in body[:4]:
            r = (r + ["â€”"] * len(header))[:len(header)]
            new_body.append(r)

        thead = "<thead><tr>" + "".join(f"<th>{escape(h or 'â€”')}</th>" for h in header) + "</tr></thead>"
        tb_rows = []
        for r in new_body:
            cells = "".join(f"<td>{escape(c or 'â€”')}</td>" for c in r)
            tb_rows.append(f"<tr>{cells}</tr>")
        tbody = "<tbody>" + "".join(tb_rows) + "</tbody>"

        return _wrap_approved_inner(thead + tbody)

    # 1) Upgrade any bare tables
    html = re.sub(r'(?is)<table\b[^>]*>.*?</table>', _fix_table, html)

    # 1b) Ensure comparison tables are wrapped exactly once
    html = wrap_unwrapped_comparison_tables(html)

    # 2) Replace any empty <td> with em dash
    html = re.sub(r'(?is)<td>\s*</td>', '<td>â€”</td>', html)
    # 3) Remove any residual markdown separator lines like |---|---|
    html = re.sub(r'(?m)^\s*\|\s*[:\-]+(?:\s*\|\s*[:\-]+)+\s*\|\s*$', '', html)
    return html


def fix_and_normalize_tables(html: str) -> str:
    if not html:
        return html or ""
    html = convert_markdown_tables_to_html(html)
    html = upgrade_html_tables_to_approved(html)   # <-- put this back
    html = re.sub(r'(?is)<td>\s*</td>', '<td>â€”</td>', html)
    return html


def _count_tables(html: str) -> int:
    return len(re.findall(r'(?is)<table\b', html))
    
def _count_md_tables(text: str) -> int:
    if not text:
        return 0
    try:
        return len(_MD_TABLE_BLOCK.findall(text))
    except Exception:
        return 0

def _log_table_state(label: str, html: str):
    logging.info(
        f"[TABLE_DEBUG] {label} | html_tables={_count_tables(html)} | md_tables={_count_md_tables(html)} | chars={len(html or '')}"
    )


def _maybe_wrap_paragraph(html: str) -> str:
    """
    Wraps in <p>â€¦</p> only if the content looks inline.
    Avoids wrapping when there are block elements like <div>, <table>, lists, or headings.
    """
    if not html:
        return ""
    if re.search(r'(?is)<(div|table|ul|ol|h[1-6]|section|article|header|footer)\b', html):
        return html
    return f"<p>{html}</p>"



_PLACEHOLDER_RX = re.compile(r"<<\s*PRODUCT_\d+\s*>>\s*,?\s*", re.IGNORECASE)

def strip_product_placeholders(name: str) -> str:
    if not name:
        return ""
    # Remove <<PRODUCT_1>>, <<PRODUCT_2>> prefixes (and any trailing comma/space)
    name = _PLACEHOLDER_RX.sub("", name).strip()
    # Normalize trailing punctuation the rest of your code already trims in places
    name = name.strip().rstrip(" .,:;!?")
    return name

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Cleaning & structuring scraped dataset
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    
def clean_scraped_dataset(raw_text, flagged_lines, cfg):
    lines = raw_text.splitlines()
    clean_lines, removed_lines, seen_lines = [], [], set()
    buffer = []
    in_h5_block = False
    current_tag = None


   

    noisy_keywords = [
        "privacy","consent","data processing","advertising","subscribe","comment",
        "terms and conditions","policy","vendors"
    ] + [kw.lower() for kw in cfg.get("noisy_keywords", [])]

    # Generic explicit patterns + category additions
    explicit_patterns = [
        r"^by[a-z]+",
        r"jump to products",
        r"save article",
        r"read more",
        r"^h[234]\s+(about the author|strictly necessary cookies|unknown cookies|cookie list|connect with us|shipping & returns)$",
        r"do not sell or share my personal data",
        r"visit us on (facebook|instagram|twitter|pinterest|youtube)",
        r"^text:\s*(always active|unknown cookies|switch labellabel|web id|original price|\d stars\d)",
        r"^text:\s*\d\.\d/5",
        r"\d+\s*star ratings",
        r"ratings distribution",
        r"(?:^|\s)(5|4|3|2|1)\s*stars?\b.*\d",
    ] + [pat for pat in cfg.get("explicit_removals", [])]

    pros_triggers = cfg.get("pros_triggers", [])
    cons_triggers = cfg.get("cons_triggers", [])

    def should_remove_price(line):
        price_matches = re.findall(r"[$Â£â‚¬]\s?\d+[\d.,]*", line)
        if len(price_matches) >= 3:
            removed_lines.append(line)
            logging.info(f"Removed line due to multiple price mentions: '{line}'")
            return True
        return False

    def should_remove_star_line(line):
        if "*" in line and len(line.split()) <= 3:
            removed_lines.append(line)
            logging.info(f"Removed short line with star: '{line}'")
            return True
        return False

    def is_date_only(line):
        if re.match(r"^(h5\s+)?(january|february|march|april|may|june|july|august|september|october|november|december)?\s*\d{1,2},?\s*\d{4}$", line.strip(), re.IGNORECASE):
            removed_lines.append(line)
            logging.info(f"Removed date-only line: '{line}'")
            return True
        return False

    def is_catalog_style(line):
        if not line or re.search(r"[.!?,:;]", line):
            return False
        if re.match(r"^Text:\s+", line):
            line = line[5:].strip()
        word_count = len(line.split())
        return 1 <= word_count <= 8 and not re.search(r"\b(is|are|was|were|has|have|had|does|do|did|buy|get|save)\b", line.lower())

    def should_keep_line(line):
        line_stripped = line.strip()
        if not line_stripped:
            return False
        lower = line_stripped.lower()

        if is_date_only(line_stripped):
            return False
        if should_remove_price(line_stripped):
            return False
        if should_remove_star_line(line_stripped):
            return False

        if lower.startswith("text:") and not line_stripped[5:].strip():
            removed_lines.append(line)
            logging.info(f"Removed empty text line: '{line}'")
            return False

        for term in noisy_keywords:
            if term in lower:
                flagged_lines.append(f"[KEYWORD: {term}] {line}")
                removed_lines.append(line)
                logging.info(f"Removed noisy keyword line: '{line}'")
                return False

        if re.match(r"^h6\b", lower) or len(line.split()) == 1:
            removed_lines.append(line)
            logging.info(f"Removed H6 or single-word line: '{line}'")
            return False

        for pattern in explicit_patterns:
            if re.search(pattern, lower, flags=0):
                removed_lines.append(line)
                logging.info(f"Removed explicit pattern match line: '{line}'")
                return False

        return True

    def flush_buffer():
        nonlocal buffer
        if sum(is_catalog_style(l) for l in buffer) >= 3:
            for l in buffer:
                removed_lines.append(l)
                logging.info(f"Removed H5 index line: '{l}'")
        else:
            for l in buffer:
                clean_lines.append(l)
        buffer = []

    for original_line in lines:
        line = original_line.strip()
        if not should_keep_line(line):
            continue

        if re.match(r"^h5\s+", line, re.IGNORECASE):
            flush_buffer()
            buffer = [line]
            in_h5_block = True
            continue

        if in_h5_block:
            if re.match(r"^h[1-4]\s+", line, re.IGNORECASE):
                flush_buffer()
                in_h5_block = False
            elif is_catalog_style(line) or line.lower().startswith("text:"):
                buffer.append(line)
                continue
            else:
                flush_buffer()
                in_h5_block = False

        if in_h5_block:
            buffer.append(line)
            continue

        if re.match(r"^h2\s+(.*)", line, re.IGNORECASE):
            h2_text = re.match(r"^h2\s+(.*)", line, re.IGNORECASE).group(1).strip()

            h2_text = normalize_product_name(
                h2_text,
                whitelist=None,
                allow_strip_best_prefix=True,
                try_map_to_whitelist=False,
                max_words=14,   # allow a bit more room at this stage
            )
            h2_text = strip_serp_editorial_wrappers(h2_text)

            tagged = f"[H2] {h2_text}"

            if tagged not in seen_lines:
                clean_lines.append(tagged)
                seen_lines.add(tagged)

            current_tag = None
            continue


        # Configurable detection of pros/cons triggers
        if any(re.search(rf"\b{re.escape(t)}\b", line, re.IGNORECASE) for t in pros_triggers):
            current_tag = "PROS"; continue
        if any(re.search(rf"\b{re.escape(t)}\b", line, re.IGNORECASE) for t in cons_triggers):
            current_tag = "CONS"; continue

        elif re.match(r"^h[1-5]\s+", line, re.IGNORECASE):
            current_tag = None
            if line not in seen_lines:
                clean_lines.append(line)
                seen_lines.add(line)
            continue

        if current_tag:
            tagged = f"[{current_tag}] {line}"
            if tagged not in seen_lines:
                clean_lines.append(tagged)
                seen_lines.add(tagged)
        else:
            if line not in seen_lines:
                clean_lines.append(line)
                seen_lines.add(line)

    flush_buffer()
    return "\n".join(clean_lines), "\n".join(removed_lines)
    
def prune_bad_product_tags(cleaned_text: str, brand_lexicon: set[str] | None = None, cfg: dict | None = None) -> str:
    out = []
    for ln in (cleaned_text or "").splitlines():
        if ln.startswith("[PRODUCT] "):
            name = ln.replace("[PRODUCT] ", "").strip()
            if not looks_like_unique_product(name, brand_lexicon=brand_lexicon, cfg=cfg):
                continue
        out.append(ln)
    return "\n".join(out)


def extract_products_to_json(cleaned_text: str):
    """
    Parse the cleaned dataset into structured product JSON.

    Expects lines like:
      [PRODUCT] Name
      [PROS] ...
      [CONS] ...
      URL: ...

    Fix: remove editorial prefixes like "Best " from product names (and other award-label noise)
    so you don't end up with products like "Best Samsonite Airea Spinner" in structured output.
    """
    lines = (cleaned_text or "").splitlines()
    product_map = defaultdict(lambda: {"pros": [], "cons": [], "other": []})
    current_product = None
    current_source = None

    def extract_domain_name(url: str) -> str:
        try:
            hostname = urlparse(url).hostname
            if hostname:
                return hostname.replace("www.", "").split(".")[0].capitalize()
        except Exception:
            pass
        return "Unknown"

    for line in lines:
        line = (line or "").strip()
        if not line:
            continue

        # Track current source domain (optional but useful)
        if line.startswith("URL:"):
            url = line.replace("URL:", "").strip()
            current_source = extract_domain_name(url)
            continue

        # New product block
        if line.startswith("[PRODUCT] "):
            raw_name = line.replace("[PRODUCT] ", "").strip()

            # âœ… Canonicalize product name so "Best X" doesn't become the stored product name.
            # If you have a product whitelist available at this stage, pass it in here instead of None.
            current_product = normalize_product_name(
                raw_name,
                whitelist=None,                # no whitelist available in this function signature
                allow_strip_best_prefix=True,  # strips leading "Best "
                try_map_to_whitelist=False,    # can't map without a whitelist
                max_words=12,
            )
            
            current_product = strip_serp_editorial_wrappers(current_product)


            # If normalization fails or it's clearly not a product, skip until a valid product appears
            if (
                not current_product
                or len(current_product.split()) < 2
                or _looks_like_ui_phrase(current_product)
                or is_obviously_not_a_product(current_product, brand_lexicon=None, cfg={})
            ):
                current_product = None

            continue

        if current_product is None:
            continue

        # Attach tagged lines
        if line.startswith("[PROS] "):
            product_map[current_product]["pros"].append({
                "text": line.replace("[PROS] ", "").strip(),
                "source": current_source or "Unknown"
            })
        elif line.startswith("[CONS] "):
            product_map[current_product]["cons"].append({
                "text": line.replace("[CONS] ", "").strip(),
                "source": current_source or "Unknown"
            })
        else:
            product_map[current_product]["other"].append({
                "text": line.strip(),
                "source": current_source or "Unknown"
            })

    return [
        {"product": p, "pros": v["pros"], "cons": v["cons"], "other": v["other"]}
        for p, v in product_map.items()
    ]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Token budgeting & hybrid prompt
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_hybrid_dataset(cleaned_path, structured_path, max_tokens=100000):
    def count_tokens(text):
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    try:
        with open(cleaned_path, "r", encoding="utf-8") as f:
            cleaned_lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logging.error(f"Error reading cleaned content: {e}")
        cleaned_lines = []

    try:
        with open(structured_path, "r", encoding="utf-8") as f:
            structured_json = json.load(f)
    except Exception as e:
        logging.error(f"Error reading structured JSON: {e}")
        structured_json = []

    structured_str = f"Structured Product Data:\n{json.dumps(structured_json, indent=2)}\n\nCleaned User Review Content:\n"
    current_token_count = count_tokens(structured_str)

    included_lines = []
    for line in cleaned_lines:
        line_tokens = count_tokens(line + "\n")
        if current_token_count + line_tokens > max_tokens:
            break
        included_lines.append(line)
        current_token_count += line_tokens

    hybrid_prompt = structured_str + "\n".join(included_lines)
    logging.info(f"Final hybrid prompt token count: {current_token_count}")
    return hybrid_prompt

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# DeepSeek helper
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def annotate_source_product_boundaries(dataset: str) -> str:
    """Make source-page H1/product ownership explicit without changing evidence."""
    output = []
    boundary_rx = re.compile(
        r"(?i)^\s*(?:\[?H1\]?\s*[:\-]?|<h1[^>]*>)\s*(.+?)(?:</h1>)?\s*$"
    )
    product_rx = re.compile(r"(?i)^\s*\[PRODUCT\]\s*(.+?)\s*$")
    for line in str(dataset or "").splitlines():
        match = boundary_rx.match(line) or product_rx.match(line)
        if match:
            title = re.sub(r"<[^>]+>", " ", match.group(1))
            title = re.sub(r"\s+", " ", title).strip()
            if title:
                output.append(f"[[SOURCE_BLOCK_PRODUCT: {title}]]")
        output.append(line)
    return "\n".join(output)


def _dataset_cache_prefix(dataset: str) -> str:
    """Return a dataset-first prefix with explicit source-product ownership."""
    annotated = annotate_source_product_boundaries(dataset)
    return (
        "REFERENCE DATASET (facts only; do not invent):\n"
        "SOURCE OWNERSHIP RULE: Every claim after [[SOURCE_BLOCK_PRODUCT: X]] "
        "belongs to X until the next boundary. A passing mention of another model "
        "does not switch ownership. Never transfer a claim across boundaries.\n"
        + annotated.strip()
    )


def dynamic_backoff(attempt, base_time=2):
    return min(base_time * (1.5 ** attempt), 60)


def deepseek_generate(
    prompt,
    retries=3,
    model=DEEPSEEK_MODEL,
    label="",
    max_tokens=1500,
    temperature=0.8,
    request_timeout=120,
    cache_prefix=None,
):
    def count_tokens(text):
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    token_count = count_tokens(prompt)
    if cache_prefix:
        token_count += count_tokens(cache_prefix)
    log_prefix = f"[{label}]" if label else "[deepseek_generate]"

    messages = [{"role": "user", "content": prompt}]
    if cache_prefix:
        messages = [
            {
                "role": "system",
                "content": (
                    "Use the supplied reference dataset as the sole factual source. "
                    "Follow the final user message for the requested output."
                ),
            },
            {"role": "user", "content": cache_prefix},
            {"role": "user", "content": prompt},
        ]

    logging.info(f"{log_prefix} Prompt token count: {token_count}")

    # Cosmetic warning only (raised threshold)
    if token_count > 60000:
        logging.warning(f"{log_prefix} Prompt is very long. Consider truncating it.")

    for attempt in range(retries):
        try:
            logging.info(
                f"{log_prefix} DeepSeek call start | attempt={attempt + 1} | "
                f"model={model} | max_tokens={max_tokens} | temperature={temperature} | "
                f"timeout={request_timeout}s"
            )

            # Reuse the shared DeepSeek client.
            client = _get_deepseek_client()

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=request_timeout,
                extra_body={"thinking": {"type": "disabled"}},
            )
            log_deepseek_usage(
                response,
                label=label or "deepseek_generate",
                requested_model=model,
            )

            choice = response.choices[0]
            output = (choice.message.content or "").strip()
            finish_reason = str(getattr(choice, "finish_reason", "") or "")

            logging.info(
                f"{log_prefix} DeepSeek call success | attempt={attempt + 1} | "
                f"response_chars={len(output)} | finish_reason={finish_reason or 'unknown'}"
            )
            if finish_reason.lower() in {"length", "max_tokens"}:
                logging.warning(
                    f"{log_prefix} Response reached the output-token limit and may be truncated."
                )
            return output

        except Exception as e:
            logging.exception(
                f"{log_prefix} DeepSeek error on attempt {attempt + 1}: {e}"
            )
            time.sleep(dynamic_backoff(attempt))

    logging.error(f"{log_prefix} Exhausted retries; returning fallback response.")
    return "Unable to generate content."


    
def enforce_list_table_limits(html: str, max_lists: int = 12, max_tables: int = 12) -> str:
    if not html:
        return html or ""

    # --- Patterns ---
    ul_pattern = re.compile(r"<ul\b[^>]*>.*?</ul>", flags=re.DOTALL | re.IGNORECASE)

    # APPROVED fragment uses the NEW classes:
    approved_table_pattern = re.compile(
        r'(?is)<div\s+class="comparison-table-wrap"[^>]*>\s*'
        r'<table\b[^>]*\bclass=["\'][^"\']*\bcomparison-table\b[^"\']*["\'][^>]*>.*?</table>\s*'
        r'</div>'
    )


    # Non-approved tables (anything else)
    any_table_pattern = re.compile(r"(?is)<table\b[^>]*>.*?</table>")

    # Helper: count placeholder cells (â€” or &mdash;) inside <td>...</td>
    td_rx = re.compile(r"(?is)<td\b[^>]*>(.*?)</td>")

    def _placeholder_td_count(table_html: str) -> int:
        count = 0
        for m in td_rx.finditer(table_html or ""):
            inner = re.sub(r"<[^>]+>", "", m.group(1)).strip()
            if inner in {"â€”", "&mdash;"}:
                count += 1
        return count

    def _td_count(table_html: str) -> int:
        return len(td_rx.findall(table_html or ""))

    # --- Keep only first N ULs ---
    uls = list(ul_pattern.finditer(html))
    if len(uls) > max_lists:
        logging.info(f"[TABLE_DEBUG] enforce_list_table_limits :: ul_count={len(uls)} cap={max_lists} :: removing={len(uls)-max_lists}")
        for m in reversed(uls[max_lists:]):
            start, end = m.span()
            html = html[:start] + html[end:]

    # --- Find approved tables ---
    approved_matches = list(approved_table_pattern.finditer(html))
    logging.info(f"[TABLE_DEBUG] enforce_list_table_limits :: approved_tables_found={len(approved_matches)} cap={max_tables}")

    # Filter: drop any approved table with >2 placeholder TDs
    approved_filtered = []
    dropped_placeholder = 0
    for idx, m in enumerate(approved_matches):
        frag = m.group(0)
        ph = _placeholder_td_count(frag)
        td = _td_count(frag)

        logging.info(
            f"[TABLE_DEBUG] enforce_list_table_limits :: approved_table#{idx} "
            f"| td={td} | placeholders={ph} | rule=placeholders<=2"
        )

        if ph <= 2:
            approved_filtered.append(frag)
        else:
            dropped_placeholder += 1
            logging.info(
                f"[TABLE_DEBUG] enforce_list_table_limits :: approved_table#{idx} DROPPED "
                f"| reason=too_many_placeholders | td={td} | placeholders={ph}"
            )

    # Apply table cap AFTER filtering by placeholder rule
    to_keep = approved_filtered[:max_tables]
    dropped_cap = max(0, len(approved_filtered) - len(to_keep))
    if dropped_cap:
        logging.info(
            f"[TABLE_DEBUG] enforce_list_table_limits :: approved_tables_after_placeholder={len(approved_filtered)} "
            f"cap={max_tables} :: dropping_due_to_cap={dropped_cap}"
        )

    # Shield kept approved tables with placeholders
    for i, frag in enumerate(to_keep):
        ph = f"__TABLE_KEEP_{i}__"
        html = html.replace(frag, ph, 1)
        logging.info(f"[TABLE_DEBUG] enforce_list_table_limits :: approved_table_kept#{i} shield={ph}")

    # Remove any remaining approved fragments (beyond the cap or failed the placeholder check)
    html = approved_table_pattern.sub("", html)

    # Log non-approved tables before removing them
    non_approved_tables = list(any_table_pattern.finditer(html))
    if non_approved_tables:
        logging.info(f"[TABLE_DEBUG] enforce_list_table_limits :: non_approved_tables_found={len(non_approved_tables)} :: removing_all=1")

    # Remove ANY other <table>â€¦</table> (non-approved tables)
    html = any_table_pattern.sub("", html)

    # Restore kept approved tables
    for i, frag in enumerate(to_keep):
        ph = f"__TABLE_KEEP_{i}__"
        html = html.replace(ph, frag, 1)

    # Remove EMPTY wrappers that might be left after table stripping
    html = re.sub(
        r'<div\s+class="comparison-table-wrap"\s*>\s*</div>',
        "",
        html,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Strip self-check comment if you ever add one
    html = re.sub(
        r"<!--\s*UL_COUNT\s*:\s*\d+\s*TABLE_COUNT\s*:\s*\d+\s*-->",
        "",
        html,
        flags=re.IGNORECASE
    )

    # Final summary line for quick scanning
    logging.info(
        f"[TABLE_DEBUG] enforce_list_table_limits :: summary "
        f"| approved_found={len(approved_matches)} "
        f"| dropped_placeholder={dropped_placeholder} "
        f"| kept={len(to_keep)} "
        f"| dropped_cap={dropped_cap}"
    )

    return html






# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Summarization & name helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def summarize_dataset(dataset, keyword):
    prompt = f"""
    Summarize the following dataset related to "{keyword}".

    Extract key insights, comparisons, notable features, strengths, weaknesses, and any trends.

    Output:
    - Under 500 words.
    - Group similar insights.
    - Mention product names where possible.
    - Exclude irrelevant or repetitive information.
    - Focus on helping a reader choose the best product.

    The reference dataset is supplied separately.
    """
    return deepseek_generate(
        prompt,
        label="dataset_summary_fallback",
        cache_prefix=_dataset_cache_prefix(dataset),
    )




def _canonical_profile_prompt_block(profile: dict | None) -> str:
    if not isinstance(profile, dict) or not profile.get("products"):
        return (
            "CANONICAL PRODUCT PROFILE: unavailable. Be conservative: omit "
            "disputed exact specifications and qualify uncertain compatibility claims."
        )
    return (
        "CANONICAL PRODUCT PROFILE (authoritative for this article):\n"
        + json.dumps(profile, ensure_ascii=False, indent=2)
        + "\nUse canonical_value and safe_wording for factual claims. "
          "The profile overrides conflicting wording in the raw dataset. "
          "Never treat source_conflict as a true range. Compatibility requires "
          "direct evidence, not a proxy such as capacity. Security requires "
          "verified access control, locking or theft deterrence."
    )


_GENERIC_FAMILY_IDENTITY_WORDS = {
    "review", "reviews", "best", "top", "new", "latest", "product", "model",
}


def _family_identity_tokens(name: str, cfg: dict | None = None) -> set[str]:
    words = re.findall(r"[a-z0-9]+", (name or "").casefold())
    aliases = _identity_word_aliases(cfg)
    words = [aliases.get(word, word) for word in words]
    configured_generic = {
        str(value).casefold()
        for value in (
            ((_runtime_category_config(cfg).get("canonical_facts") or {})
             .get("family_identity_generic_words") or [])
        )
    }
    words = [
        word for word in words
        if word not in (_GENERIC_FAMILY_IDENTITY_WORDS | configured_generic)
        and not any(ch.isdigit() for ch in word)
    ]
    tokens = set(words)
    tokens.update(
        words[index] + words[index + 1]
        for index in range(len(words) - 1)
    )
    return tokens


def _is_related_profile_product(
    primary: str,
    candidate: str,
    cfg: dict | None = None,
) -> bool:
    if not primary or not candidate:
        return False
    if canonical_product_identity_key(primary) == canonical_product_identity_key(candidate):
        return True
    if looks_like_legal_entity_name(candidate) or _looks_like_ui_phrase(candidate):
        return False
    shared = _family_identity_tokens(primary, cfg) & _family_identity_tokens(candidate, cfg)
    minimum = int(((cfg or {}).get("canonical_facts") or {}).get(
        "related_product_min_shared_tokens", 2
    ))
    return len(shared) >= max(1, minimum)


def _evidence_product_heading_candidates(dataset: str) -> list[str]:
    candidates = []
    patterns = (
        r"(?im)^\s*(?:\[PRODUCT\]|\[?H1\]?|H1)\s*[:\-]?\s*(.+?)\s*$",
        r"(?im)^\s*URL:\s*https?://[^\s]*/([^\s?#]+)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, dataset or ""):
            raw = match.group(1).replace("-", " ")
            raw = re.sub(
                r"(?i)^\s*(?:gear|product|hands on|in depth|long term)?\s*review\s*[:\-]?\s*",
                "",
                raw,
            )
            raw = re.sub(
                r"(?i)\s+(?:buying\s+)?guide(?:\s*\(?20\d{2}\)?)?\s*$",
                "",
                raw,
            )
            cleaned = normalize_product_name(
                raw,
                cfg=None,
                try_map_to_whitelist=False,
                max_words=12,
            )
            cleaned = strip_editorial_product_name_suffix(cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;!?")
            if cleaned:
                editorial_comparison = bool(re.search(
                    r"(?i)\b(?:differences?|similarities|comparison|versus|vs\.?|"
                    r"models?\s+being\s+reviewed)\b",
                    cleaned,
                ))
                if not editorial_comparison and not _looks_like_ui_phrase(cleaned):
                    candidates.append(cleaned)
    return list(dict.fromkeys(candidates))


def _normalize_provenance_marker(value: str) -> str:
    """Normalize a distinctive measured/user-reported literal for comparison."""
    value = str(value or "").casefold()
    value = value.replace("\u2019", "'").replace("\u2032", "'")
    value = value.replace("\u201d", '"').replace("\u2033", '"')
    if re.fullmatch(r'\d+(?:\.\d+)?\s*"', value.strip()):
        value = re.sub(r'\s*"$', "inch", value.strip())
    value = re.sub(r"(?<=\d)\s*[-\u2013\u2014]\s*(?=[a-z%])", "", value)
    value = re.sub(r"(?<=\d)\s+(?=[a-z%])", "", value)
    return re.sub(r"\s+", " ", value).strip(" .,;:()[]")


def _provenance_markers(text: str, cfg: dict | None = None) -> set[str]:
    """Extract exact, portable markers that can establish claim ownership."""
    controls = _runtime_category_config(cfg).get("semantic_fact_audit") or {}
    patterns = controls.get("provenance_marker_patterns") or [
        r"(?<!\w)\d+(?:\.\d+)?\s*(?:-|\u2013|\u2014)?\s*(?:%|[a-zA-Z]{1,12})(?!\w)",
        r"(?<!\w)\d+\s*['\u2019\u2032]\s*\d+(?:\s*[\"\u201d\u2033])?(?!\w)",
    ]
    markers = set()
    for pattern in patterns:
        try:
            matches = re.finditer(str(pattern), text or "", flags=re.I)
        except re.error:
            logging.warning("[SEMANTIC_AUDIT] Ignoring invalid provenance regex: %s", pattern)
            continue
        for match in matches:
            marker = _normalize_provenance_marker(match.group(0))
            if marker and len(marker) >= 2:
                markers.add(marker)
    return markers


def _source_claim_provenance(
    dataset: str,
    products: list[str],
    cfg: dict,
) -> list[dict]:
    """Build a product-owned literal ledger directly from marked source blocks."""
    if not dataset or not products:
        return []

    identity_markers = set()
    for product in products:
        identity_markers.update(_provenance_markers(product, cfg))

    def source_owner(title: str) -> str:
        exact = canonical_product_identity_key(title)
        for product in products:
            if exact == canonical_product_identity_key(product):
                return product
        title_numbers = keyword_product_identity_tokens(title)
        ranked = []
        for product in products:
            product_numbers = keyword_product_identity_tokens(product)
            number_overlap = len(title_numbers & product_numbers)
            if title_numbers and product_numbers and not number_overlap:
                continue
            if (
                not number_overlap
                and not _is_related_profile_product(product, title, cfg)
            ):
                continue
            family_overlap = len(
                _family_identity_tokens(title, cfg)
                & _family_identity_tokens(product, cfg)
            )
            ranked.append((number_overlap * 10 + family_overlap, product))
        ranked.sort(reverse=True)
        if not ranked or ranked[0][0] <= 0:
            return ""
        if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
            return ""
        return ranked[0][1]

    boundary_rx = re.compile(r"^\[\[SOURCE_BLOCK_PRODUCT:\s*(.+?)\]\]$")
    blocks = []
    owner = ""
    lines = []
    for line in annotate_source_product_boundaries(dataset).splitlines():
        match = boundary_rx.match(line.strip())
        if match:
            if owner and lines:
                blocks.append((owner, " ".join(lines)))
            owner = source_owner(match.group(1).strip())
            lines = []
            continue
        if owner:
            lines.append(line.strip())
    if owner and lines:
        blocks.append((owner, " ".join(lines)))

    per_product_limit = max(
        1,
        int((cfg.get("canonical_facts") or {}).get(
            "max_source_provenance_entries_per_product", 24
        )),
    )
    counts = {product: 0 for product in products}
    ledger = []
    seen = set()
    for product, block in blocks:
        for passage in re.split(r"(?<=[.!?])\s+|[\r\n]+", block):
            passage = re.sub(r"\s+", " ", passage).strip()
            passage_product = source_owner(passage) or product
            markers = sorted(
                marker for marker in _provenance_markers(passage, cfg)
                if marker not in identity_markers
            )
            if not passage or not markers or counts.get(passage_product, 0) >= per_product_limit:
                continue
            key = (canonical_product_identity_key(passage_product), tuple(markers), passage.casefold())
            if key in seen:
                continue
            seen.add(key)
            ledger.append({
                "product": passage_product,
                "attribute": "source_observation",
                "evidence_excerpt": passage[:300],
                "distinctive_markers": markers,
            })
            counts[passage_product] = counts.get(passage_product, 0) + 1
    return ledger

def _enforce_evidence_weight_conflicts(
    products: list[dict],
    claim_provenance: list[dict],
) -> int:
    """Downgrade a hard weight fact when exact-product sources disagree.

    A model should not select between competing source figures. This compact
    provenance check deliberately considers only mass-like measurements and
    skips bottle/load capacities, which prevents an unrelated capacity from
    being mistaken for a product weight.
    """
    mass_rx = re.compile(
        r"(?<!\w)(\d+(?:\.\d+)?)\s*(kg|g|lb|lbs|oz)(?!\w)", re.I
    )
    unit_priority = {"g": 0, "kg": 1, "lb": 2, "lbs": 2, "oz": 3}
    reconciled = 0
    for product in products or []:
        if not isinstance(product, dict):
            continue
        fact = (product.get("facts") or {}).get("weight")
        if not isinstance(fact, dict):
            continue
        if str(fact.get("source_type") or "").casefold() in {
            "manufacturer", "retail_listing"
        }:
            continue
        product_key = canonical_product_identity_key(str(product.get("name") or ""))
        values = []
        for item in claim_provenance or []:
            if not isinstance(item, dict) or (
                canonical_product_identity_key(str(item.get("product") or ""))
                != product_key
            ):
                continue
            excerpt = str(item.get("evidence_excerpt") or "")
            if re.search(
                r"\b(?:(?:water\s*)?bottles?|holders?|loads?|payload|capacity|hip support|comfort threshold)\b",
                excerpt, re.I,
            ):
                continue
            matches = mass_rx.findall(excerpt)
            if not matches:
                continue
            # Prefer one metric measurement per source passage. A paired
            # imperial value (for example 680 g / 24 oz) is not a conflict.
            number, unit = min(
                matches, key=lambda match: unit_priority[match[1].casefold()]
            )
            value = f"{number} {unit.casefold()}"
            if value.casefold() not in {known.casefold() for known in values}:
                values.append(value)
        if len(values) < 2:
            continue
        joined = " and ".join(values[:2]) if len(values) == 2 else ", ".join(values[:3])
        fact["canonical_value"] = ""
        fact["safe_wording"] = (
            f"Published source figures differ on the exact weight ({joined}), "
            "so a single figure could not be verified."
        )
        fact["confidence"] = "low"
        fact["value_status"] = "source_conflict"
        fact["requires_attribution"] = True
        fact["conflicting_values"] = list(dict.fromkeys([
            *(fact.get("conflicting_values") or []),
            *values,
        ]))[:12]
        fact["source_count"] = max(
            int(fact.get("source_count") or 0), len(values)
        )
        fact["exact_source_count"] = 0
        fact["basis"] = (
            "Exact-product source figures conflict; no primary specification "
            "was available to resolve them."
        )
        reconciled += 1
    return reconciled


def build_canonical_product_profile(
    dataset: str,
    keyword: str,
    recommended_product: str,
    cfg: dict,
    product_whitelist: list[str] | None = None,
) -> dict:
    """Consolidate conflicting source claims once for every article section."""
    controls = cfg.get("canonical_facts") or {}
    if controls.get("enabled", True) is False:
        return {}

    attribute_schema = cfg.get("canonical_attributes") or {}
    dynamic_attributes = bool(controls.get("dynamic_attributes", True))
    max_dynamic_attributes = max(0, int(controls.get("max_dynamic_attributes", 6)))
    max_products = max(1, int(controls.get("max_products", 4)))
    # Build the scope before prompting so UI/legal headings cannot consume the
    # model's limited product slots. Evidence H1/PRODUCT markers recover related
    # variants that the earlier whitelist extractor may have missed.
    products = [recommended_product] if recommended_product else []
    canonical_by_identity = {
        canonical_product_identity_key(recommended_product): recommended_product
    } if recommended_product else {}
    source_aliases = {}
    if not controls.get("primary_product_only", True):
        candidates = list(dict.fromkeys([
            *(product_whitelist or []),
            *_evidence_product_heading_candidates(dataset),
        ]))
        candidates.sort(
            key=lambda value: (
                0 if keyword_product_identity_tokens(str(value)) else 1,
                len(str(value)),
            )
        )
        for candidate in candidates:
            candidate = re.sub(r"\s+", " ", str(candidate or "")).strip()
            if (
                not candidate
                or not _is_related_profile_product(recommended_product, candidate, cfg)
            ):
                continue
            identity = canonical_product_identity_key(candidate)
            existing = canonical_by_identity.get(identity)
            if existing:
                if (
                    candidate.casefold() != existing.casefold()
                    and not any(alias.casefold() == candidate.casefold()
                                for alias in source_aliases)
                ):
                    source_aliases[candidate] = existing
                continue
            canonical_by_identity[identity] = candidate
            products.append(candidate)
            if len(products) >= max_products:
                break

    prompt = f"""
Create a canonical product fact profile for a category-agnostic buying review.

Article topic: {keyword}
Primary product: {recommended_product}
Products in scope: {json.dumps(products, ensure_ascii=False)}
Source-name aliases to canonical product names:
{json.dumps(source_aliases, ensure_ascii=False, indent=2)}
Configured attribute schema:
{json.dumps(attribute_schema, ensure_ascii=False, indent=2)}
Dynamic decision attributes enabled: {dynamic_attributes}
Maximum additional decision attributes per product: {max_dynamic_attributes}
Primary-product-only mode: {bool(controls.get("primary_product_only", True))}
Source-priority guidance:
{json.dumps(controls.get("source_priority") or [], ensure_ascii=False, indent=2)}

Return STRICT JSON only:
{{
  "schema_version": 2,
  "products": [
    {{
      "name": "full product name",
      "generation_or_style": "style number, generation or date range, or unknown",
      "current_status": "current|legacy|unknown",
      "facts": {{
        "attribute_id": {{
          "canonical_value": "short normalized value, or empty if unresolved",
          "safe_wording": "one consumer-facing sentence that preserves necessary qualifications",
          "confidence": "high|medium|low",
          "conflicting_values": ["exact conflicting strings found in the evidence"],
          "forbidden_terms": ["stronger or misleading terms that must not be used"],
          "basis": "brief explanation of how the evidence was reconciled",
          "product_scope": "exact_product|variant_or_generation|ambiguous",
          "source_count": 1,
          "exact_source_count": 1,
          "requires_attribution": true,
          "value_status": "confirmed|explicit_range|source_conflict|unresolved",
          "evidence_excerpt": "short source wording supporting this exact product",
          "source_type": "manufacturer|retail_listing|specialist_review|user_report|unknown",
          "source_date": "YYYY-MM-DD, year, or unknown"
        }}
      }}
    }}
  ]
}}

Rules:
- Use only the supplied evidence.
- Preserve product editions, generations, sizes and variants; never merge their specifications.
- Bind every fact to the exact named product whose local source context supports it. A fact found near Product B must never be assigned to Product A merely because both products share a brand or family name.
- Treat source-page titles, H1 headings, explicit model names and immediately surrounding paragraphs as product boundaries. Generic headings such as "Final Thoughts", "Overview" or "Specifications" are never product identities.
- Return profiles ONLY for the exact names in Products in scope. Do not spend product slots on companies, advertisers, headings or newly invented alternatives.
- Treat every Source-name alias as the mapped canonical product, combine its evidence into that one record, and reconcile conflicting alias evidence rather than creating another product.
- Record style number, generation or evidence date when available. Never merge current and legacy specifications into one product record.
- Current manufacturer specifications and exact current retail listings outrank older third-party reviews for hard specifications.
- A dated or generation-unknown specialist/user report may describe real-world experience, but must not become an unqualified current specification.
- NEVER infer, scale, convert or borrow a specification from another size, variant, edition or related model. If the exact product's value is not independently supported, leave canonical_value empty, set confidence to low and state that it could not be reliably confirmed.
- Prefer an explicitly current manufacturer or exact retail listing for hard specifications.
- Use specialist reviews and user reports for real-world fit, comfort and performance.
- Treat exact dimensions, weight, capacity, electrical ratings, compatibility limits and safety limits as hard specifications.
- When evidence conflicts and no reliable current value wins, leave canonical_value empty and make safe_wording qualified or omit the exact number.
- Never turn different source values into a numerical range. Use value_status source_conflict, put the reported values in conflicting_values, and explain that published figures differ.
- Use value_status explicit_range only when one source explicitly reports a range or mapped variants establish its endpoints.
- source_count counts sources supporting the general attribute. exact_source_count counts only independent sources supporting the exact number, range, lifespan or compatibility limit.
- A precise value supported by one non-manufacturer source requires attribution even when other sources support the general attribute.
- Distinguish material construction from real-world performance. A mesh panel can be breathable yet still have limited airflow.
- Distinguish stated compatibility from real-world fit. When user reports vary, say that actual fit depends on physical dimensions, cases, variants or conditions; do not turn a nominal rating into a universal guarantee.
- Capacity, power, price or another proxy never establishes physical fit, regulatory compliance, airline acceptance or safety. Compatibility needs direct evidence for the exact product.
- Security means verified access control, locking hardware, concealment or theft deterrence. Visibility, padding, ordinary storage and bottle retention are not security features.
- Never upgrade claim strength. For example, evidence of rain resistance does not establish waterproofing.
- Populate conflicting_values ONLY with mutually exclusive factual alternatives.
- Do not put synonyms, compatible observations, unit conversions, the canonical value with extra units, or wording already approved by safe_wording in conflicting_values.
- forbidden_terms are terms that would mislead when asserted positively; they may still appear in an explicit negation or buyer-exclusion statement.
- For the primary product, add a related model's distinctive literal claim to forbidden_terms when the evidence ties that claim only to the related model and transferring it would mislead. This applies even when the primary product's canonical_value is unresolved.
- Cover every configured attribute that has evidence.
- When dynamic decision attributes are enabled, add up to {max_dynamic_attributes} important evidence-backed attributes not covered by the configured schema. Use category-neutral IDs such as fit_or_compatibility, feature_presence, access_or_security, durability_history, controls_or_interface, maintenance, or safety.
- For a comparison product, include only facts needed for claims or comparisons likely to appear in the article.
- If a claim is supported by only one reviewer or one unverified secondary source, set requires_attribution to true and make safe_wording explicitly say "one reviewer reports" or "one source reports". Never generalize it to "users report".
- Set product_scope to ambiguous and leave canonical_value empty when the evidence cannot be tied to one exact product, variant or generation.
- evidence_excerpt must be short and must identify why the fact belongs to this product rather than a related model.
- Keep each safe_wording sentence under 28 words and each basis under 20 words.
- Include at most 5 conflicting_values and 5 forbidden_terms per attribute.
- Always return the primary product first. Return only products explicitly listed in Products in scope, up to the configured product limit.
- Do not include prices, promotional claims or generic praise.
""".strip()

    max_output_tokens = int(controls.get("max_tokens", 4200))
    raw = deepseek_generate(
        prompt,
        model=DEEPSEEK_MODEL,
        label="canonical_product_profile",
        max_tokens=max_output_tokens,
        temperature=float(controls.get("temperature", 0.1)),
        cache_prefix=_dataset_cache_prefix(dataset),
    )

    raw_response_path = str(cfg.get("_canonical_raw_response_path") or "").strip()
    if raw_response_path:
        try:
            with open(raw_response_path, "w", encoding="utf-8") as raw_file:
                raw_file.write(raw)
        except Exception as exc:
            logging.warning("[CANONICAL_FACTS] Could not save raw response: %s", exc)

    try:
        parsed = _json_from_text_block(raw)
    except Exception as first_exc:
        logging.warning(
            "[CANONICAL_FACTS] Initial JSON parse failed (%s); requesting compact repair.",
            first_exc,
        )
        repair_prompt = prompt + """
        
REPAIR REQUIREMENTS:
- The previous response was invalid or truncated.
- Return a complete, compact JSON object for the products in scope.
- Include configured attributes plus only the most important dynamic decision attributes.
- Use at most 20 words for safe_wording and 12 words for basis.
- Use at most 3 conflicting_values and 3 forbidden_terms per attribute.
- Do not use Markdown. Close every JSON object and array.
""".rstrip()
        repaired_raw = deepseek_generate(
            repair_prompt,
            model=DEEPSEEK_MODEL,
            label="canonical_product_profile_repair",
            max_tokens=int(controls.get("repair_max_tokens", max(4200, max_output_tokens))),
            temperature=0.0,
            cache_prefix=_dataset_cache_prefix(dataset),
        )
        repair_response_path = str(
            cfg.get("_canonical_repair_response_path") or ""
        ).strip()
        if repair_response_path:
            try:
                with open(repair_response_path, "w", encoding="utf-8") as repair_file:
                    repair_file.write(repaired_raw)
            except Exception as exc:
                logging.warning(
                    "[CANONICAL_FACTS] Could not save repair response: %s", exc
                )
        try:
            parsed = _json_from_text_block(repaired_raw)
        except Exception as repair_exc:
            logging.error(
                "[CANONICAL_FACTS] Could not parse initial or repaired profile: "
                "initial=%s repair=%s",
                first_exc,
                repair_exc,
            )
            return {}

    if not isinstance(parsed, dict) or not isinstance(parsed.get("products"), list):
        logging.error("[CANONICAL_FACTS] Invalid profile structure.")
        return {}

    cleaned_products = []
    allowed_confidence = {"high", "medium", "low"}
    for product in parsed.get("products", [])[:max_products]:
        if not isinstance(product, dict):
            continue
        name = re.sub(r"\s+", " ", str(product.get("name") or "")).strip()
        facts = product.get("facts")
        matched_product = next(
            (
                candidate for candidate in products
                if canonical_product_identity_key(name)
                == canonical_product_identity_key(candidate)
            ),
            "",
        )
        allowed_product = bool(matched_product)
        if not name or not isinstance(facts, dict) or not allowed_product:
            if name:
                logging.warning(
                    "[CANONICAL_FACTS] Dropped out-of-scope product record: %s", name
                )
            continue
        name = matched_product
        if any(
            canonical_product_identity_key(existing.get("name", ""))
            == canonical_product_identity_key(name)
            for existing in cleaned_products
        ):
            logging.warning(
                "[CANONICAL_FACTS] Dropped duplicate alias product record: %s",
                product.get("name"),
            )
            continue
        generation_or_style = re.sub(
            r"\s+", " ", str(product.get("generation_or_style") or "unknown")
        ).strip()
        current_status = str(product.get("current_status") or "unknown").strip().lower()
        if current_status not in {"current", "legacy", "unknown"}:
            current_status = "unknown"
        cleaned_facts = {}
        for attribute, fact in facts.items():
            if not isinstance(fact, dict):
                continue
            safe_wording = re.sub(r"\s+", " ", str(fact.get("safe_wording") or "")).strip()
            canonical_value = re.sub(r"\s+", " ", str(fact.get("canonical_value") or "")).strip()
            if not safe_wording and not canonical_value:
                continue
            confidence = str(fact.get("confidence") or "low").lower()
            basis = re.sub(r"\s+", " ", str(fact.get("basis") or "")).strip()
            # A related model may inform comparison prose, but it must never
            # supply a hard specification for the reviewed product.
            inferred_from_other_model = bool(re.search(
                r"(?i)\b(?:inferred|estimated|extrapolated|borrowed|scaled)\b.{0,100}"
                r"\b(?:model|variant|version|size|edition)\b|"
                r"\b(?:model|variant|version|size|edition)\b.{0,100}"
                r"\b(?:not confirmed|unconfirmed|not explicitly|inferred)\b",
                basis,
            ))
            if inferred_from_other_model:
                canonical_value = ""
                confidence = "low"
                safe_wording = (
                    f"We could not reliably confirm the {str(attribute).replace('_', ' ')} "
                    f"for the specific {name} version."
                )
            normalized_canonical = re.sub(
                r"\s+", "", canonical_value.casefold()
            )
            normalized_safe = re.sub(r"\s+", " ", safe_wording.casefold()).strip()
            cleaned_conflicts = []
            for value in (fact.get("conflicting_values") or []):
                value = re.sub(r"\s+", " ", str(value)).strip()
                if not value:
                    continue
                normalized_value = re.sub(r"\s+", "", value.casefold())
                value_in_safe_wording = (
                    re.sub(r"\s+", " ", value.casefold()).strip() in normalized_safe
                )
                equivalent_expansion = bool(
                    normalized_canonical
                    and (
                        normalized_value in normalized_canonical
                        or normalized_canonical in normalized_value
                    )
                )
                if value_in_safe_wording or equivalent_expansion:
                    continue
                cleaned_conflicts.append(value)

            try:
                source_count = max(0, int(fact.get("source_count") or 0))
            except (TypeError, ValueError):
                source_count = 0
            try:
                exact_source_count = max(0, int(fact.get("exact_source_count") or 0))
            except (TypeError, ValueError):
                exact_source_count = 0
            value_status = str(
                fact.get("value_status")
                or ("confirmed" if canonical_value else "unresolved")
            ).strip().lower()
            if value_status not in {
                "confirmed", "explicit_range", "source_conflict", "unresolved"
            }:
                value_status = "confirmed" if canonical_value else "unresolved"
            default_scope = (
                "exact_product"
                if canonical_product_identity_key(name)
                == canonical_product_identity_key(recommended_product)
                else "ambiguous"
            )
            product_scope = str(fact.get("product_scope") or default_scope).strip().lower()
            if product_scope not in {"exact_product", "variant_or_generation", "ambiguous"}:
                product_scope = "ambiguous"
            requires_attribution = bool(fact.get("requires_attribution", source_count <= 1))
            evidence_excerpt = re.sub(
                r"\s+", " ", str(fact.get("evidence_excerpt") or "")
            ).strip()
            source_type = str(fact.get("source_type") or "unknown").strip().lower()
            if source_type not in {
                "manufacturer", "retail_listing", "specialist_review",
                "user_report", "unknown",
            }:
                source_type = "unknown"
            source_date = re.sub(
                r"\s+", " ", str(fact.get("source_date") or "unknown")
            ).strip()
            cleaned_forbidden_terms = [
                re.sub(r"\s+", " ", str(value)).strip()
                for value in (fact.get("forbidden_terms") or [])
                if str(value).strip()
            ]

            hard_single_source_attributes = set(controls.get(
                "single_source_hard_attributes",
                [
                    "weight", "dimensions", "device_fit", "compatibility",
                    "electrical_rating", "power", "safety_limit",
                ],
            ))
            attribute_id = str(attribute).strip()
            precision_limited_attributes = {
                str(value).casefold()
                for value in controls.get(
                    "single_source_duration_attributes",
                    ["durability_history", "lifespan", "service_life", "longevity"],
                )
            }
            duration_rx = re.compile(
                r"\b\d+(?:\.\d+)?(?:\s*(?:-|\u2013|\u2014|to)\s*\d+(?:\.\d+)?)?"
                r"\s*(?:years?|months?|weeks?|days?|hours?)\b",
                re.I,
            )
            duration_terms = list(dict.fromkeys(
                duration_rx.findall(" ".join([safe_wording, canonical_value, evidence_excerpt]))
            ))
            if (
                attribute_id.casefold() in precision_limited_attributes
                and duration_terms
                and exact_source_count < int(
                    controls.get("minimum_exact_sources_for_duration", 2)
                )
                and source_type not in {"manufacturer", "retail_listing"}
            ):
                cleaned_forbidden_terms.extend(duration_terms)
                safe_wording = str(controls.get(
                    "single_source_duration_safe_wording",
                    "One source reports evidence of long-term use, but the precise duration is not independently verified.",
                )).strip()
                if duration_rx.search(canonical_value):
                    canonical_value = "long-term use"
                requires_attribution = True
                confidence = "medium" if confidence == "high" else confidence

            precise_value = bool(re.search(r"\d", canonical_value))
            basis_reports_conflict = bool(re.search(
                r"(?i)\b(?:different|differ|conflict|vary|varies|inconsistent)\b",
                basis,
            ))
            looks_like_range = bool(re.search(
                r"\d\s*(?:-|â€“|â€”|to)\s*\d",
                canonical_value,
            ))
            if looks_like_range and basis_reports_conflict:
                value_status = "source_conflict"
            if value_status == "source_conflict":
                measurement_rx = re.compile(
                    r"\b\d+(?:\.\d+)?\s*(?:kg|g|mg|lb|lbs|oz|l|ml|cm|mm|m|"
                    r"inches?|inch|years?|months?|w|kw|v|mah|%)\b",
                    re.I,
                )
                reported_values = []
                for value in measurement_rx.findall(evidence_excerpt):
                    normalized_value = re.sub(r"\s+", " ", value).strip()
                    if normalized_value not in reported_values:
                        reported_values.append(normalized_value)
                if not reported_values:
                    reported_values = list(dict.fromkeys(cleaned_conflicts))
                if reported_values:
                    if len(reported_values) == 2:
                        joined_values = f"{reported_values[0]} and {reported_values[1]}"
                    else:
                        joined_values = ", ".join(reported_values[:3])
                    safe_wording = (
                        f"Published figures for the {attribute_id.replace('_', ' ')} "
                        f"vary: {joined_values}; the exact version is unclear."
                    )
                    cleaned_conflicts = list(dict.fromkeys(
                        [*cleaned_conflicts, *reported_values]
                    ))
                canonical_value = ""
                confidence = "medium" if confidence == "high" else confidence
                requires_attribution = True
            if (
                attribute_id in hard_single_source_attributes
                and source_count <= 1
                and source_type not in {"manufacturer", "retail_listing"}
                and confidence == "high"
            ):
                confidence = "medium"
            if (
                precise_value
                and exact_source_count <= 1
                and source_type not in {"manufacturer", "retail_listing"}
            ):
                requires_attribution = True
                if confidence == "high":
                    confidence = "medium"
            if (
                attribute_id in hard_single_source_attributes
                and not canonical_value
                and (confidence == "low" or source_count <= 0)
            ):
                safe_wording = (
                    f"We could not reliably confirm the "
                    f"{attribute_id.replace('_', ' ')} for this exact product."
                )
                value_status = "unresolved"
            if requires_attribution and safe_wording and not re.search(
                r"(?i)\b(?:one|a)\s+(?:source|reviewer|review|user|owner|tester)\b|"
                r"\baccording to\b",
                safe_wording,
            ):
                safe_wording = f"One source reports: {safe_wording}"

            # Ambiguous product ownership cannot license an affirmative exact
            # claim. Keep the warning sentence, but prevent article generation
            # from treating it as a settled product fact.
            if product_scope == "ambiguous":
                canonical_value = ""
                confidence = "low"

            cleaned_facts[str(attribute).strip()] = {
                "canonical_value": canonical_value,
                "safe_wording": safe_wording,
                "confidence": confidence if confidence in allowed_confidence else "low",
                "conflicting_values": list(dict.fromkeys(cleaned_conflicts))[:12],
                "forbidden_terms": list(dict.fromkeys(cleaned_forbidden_terms))[:12],
                "basis": basis[:500],
                "product_scope": product_scope,
                "source_count": source_count,
                "exact_source_count": exact_source_count,
                "requires_attribution": requires_attribution,
                "value_status": value_status,
                "evidence_excerpt": evidence_excerpt[:300],
                "source_type": source_type,
                "source_date": source_date[:40],
            }
        if cleaned_facts:
            cleaned_products.append({
                "name": name,
                "generation_or_style": generation_or_style[:120],
                "current_status": current_status,
                "facts": cleaned_facts,
            })

    claim_provenance = _source_claim_provenance(dataset, products, cfg)
    _enforce_evidence_weight_conflicts(cleaned_products, claim_provenance)
    for product in cleaned_products:
        for attribute, fact in product.get("facts", {}).items():
            excerpt = str(fact.get("evidence_excerpt") or "").strip()
            markers = sorted(_provenance_markers(excerpt, cfg))
            if excerpt and markers:
                claim_provenance.append({
                    "product": product.get("name", ""),
                    "attribute": attribute,
                    "evidence_excerpt": excerpt,
                    "distinctive_markers": markers,
                })

    profile = {
        "schema_version": 2,
        "primary_product": recommended_product,
        "conflict_policy": controls.get("conflict_policy", "qualify_or_omit"),
        "aliases": source_aliases,
        "claim_provenance": claim_provenance,
        "products": cleaned_products,
    }
    logging.info(
        "[CANONICAL_FACTS] Built profile for %d product(s), %d fact(s).",
        len(cleaned_products),
        sum(len(product["facts"]) for product in cleaned_products),
    )
    return profile


def _heading_requests_comparison(heading: str, cfg: dict) -> bool:
    controls = cfg.get("editorial_controls") or {}
    terms = controls.get("comparison_heading_terms") or [
        "compare", "comparison", "versus", " vs ", "difference",
        "alternatives", "other models", "models being reviewed",
    ]
    normalized = f" {(heading or '').casefold()} "
    return any(str(term).casefold() in normalized for term in terms)


def _article_memory_block(cfg: dict) -> str:
    controls = cfg.get("editorial_controls") or {}
    if controls.get("use_article_memory", True) is False:
        return ""
    entries = cfg.get("_article_memory") or []
    if not entries:
        return ""
    max_sections = max(1, int(controls.get("memory_sections", 6)))
    max_chars = max(500, int(controls.get("memory_chars", 5000)))
    text = "\n\n".join(str(value) for value in entries[-max_sections:])
    return (
        "ALREADY COVERED IN EARLIER SECTIONS:\n"
        + text[-max_chars:]
        + "\nDo not repeat these conclusions unless the current heading requires them."
    )


def remember_generated_section(cfg: dict, heading: str, html: str) -> None:
    text = unescape(re.sub(r"<[^>]+>", " ", html or ""))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return
    memory = cfg.setdefault("_article_memory", [])
    memory.append(f"{heading}: {text[:1800]}")
    del memory[:-10]


def _normalize_nested_paragraphs(html: str) -> str:
    """Collapse invalid nested paragraphs without discarding their content."""
    soup = BeautifulSoup(html or "", "html.parser")
    # Work from inner nodes outward. Unwrapping keeps text and inline markup.
    for paragraph in list(soup.find_all("p")):
        if paragraph.find_parent("p") is not None:
            paragraph.unwrap()
    for paragraph in list(soup.find_all("p")):
        if not paragraph.get_text(" ", strip=True) and not paragraph.find(
            ["img", "a", "br"]
        ):
            paragraph.decompose()
    return str(soup)


def _soften_unsupported_editorial_claims(html: str) -> str:
    """Remove unsupported superiority/absolute language from visible prose."""
    soup = BeautifulSoup(html or "", "html.parser")
    for node in list(soup.find_all(string=True)):
        if node.find_parent(["script", "style", "code", "pre"]):
            continue
        text = str(node)
        revised = text
        revised = re.sub(
            r"(?i)\s+in\s+a\s+way\s+that\s+few\s+competitors\s+match"
            r"(?:\s+at\s+this\s+size)?",
            ", making it a strong option for its intended use",
            revised,
        )
        revised = re.sub(
            r"(?i)\bfew\s+competitors\s+(?:can\s+)?match(?:\s+it)?"
            r"(?:\s+at\s+this\s+size)?",
            "it is a strong option for its intended use",
            revised,
        )
        revised = re.sub(
            r"(?i)\bbetter\s+than\s+most\s+rivals\b",
            "well suited to its intended use",
            revised,
        )
        revised = re.sub(
            r"(?i)\b(?:unmatched|unrivalled|unrivaled|second\s+to\s+none|"
            r"best[- ]in[- ]class)\b",
            "strong",
            revised,
        )
        revised = re.sub(
            r"(?i)\s+without\s+(?:any\s+)?(?:hesitation|reservation)\b",
            "",
            revised,
        )
        revised = re.sub(r"[ \t]{2,}", " ", revised)
        if revised != text:
            node.replace_with(revised)
    return str(soup)


def clean_generated_document_html(html: str) -> str:
    """Normalize model HTML and apply narrow editorial safety controls."""
    cleaned = html or ""
    cleaned = re.sub(r"(?i)\x60{3}(?:html)?", "", cleaned)
    cleaned = re.sub(r"</\s*>", "", cleaned)
    cleaned = _normalize_nested_paragraphs(cleaned)
    cleaned = _soften_unsupported_editorial_claims(cleaned)
    return cleaned.strip()


def _html_heading_signature(html: str, tag: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", match))).strip()
        for match in re.findall(rf"(?is)<{tag}\b[^>]*>.*?</{tag}>", html or "")
    ]


def run_final_editorial_pass(
    html: str,
    canonical_profile: dict,
    keyword: str,
    recommended_product: str,
    cfg: dict,
) -> tuple[str, dict]:
    """Reconcile facts and repetition while preserving the article structure."""
    original = clean_generated_document_html(html)
    controls = cfg.get("final_editorial_pass") or {}
    report = {
        "enabled": bool(controls.get("enabled", True)),
        "applied": False,
        "reason": "",
        "original_chars": len(original),
    }
    if controls.get("enabled", True) is False or not canonical_profile:
        report["reason"] = "disabled_or_no_profile"
        return original, report

    prompt = f"""
Edit the supplied HTML article fragment for factual consistency and concise editorial quality.

Article topic: {keyword}
Primary product: {recommended_product}

Hard requirements:
- Return ONLY the revised HTML fragment. No Markdown fence or explanation.
- Preserve every H2 and H3 heading exactly, in the same order.
- Preserve product identities. Do not invent products, specifications or experiences.
- Treat each product record in the canonical profile as authoritative only for that exact product, size, variant and generation.
- Before retaining any sentence that names a product, verify that every factual claim in that sentence is licensed by that same product's profile record.
- Never transfer fit, compatibility, feature presence, dimensions, performance, security, durability or user experience from one related product to another.
- If a related product is mentioned but has no matching profile record, retain only its identity and omit unsupported specifications or comparisons.
- Facts marked requires_attribution must remain explicitly attributed to one reviewer or one source; never rewrite them as broad user consensus.
- Replace conflicting specifications and claim strength with canonical safe_wording.
- If a canonical value is unresolved or below high confidence, remove unsupported exact specifications rather than estimating them.
- Never infer one model's dimensions, weight, ratings or compatibility from another model.
- If exact dimensions are not high-confidence, do not promise universal airline, installation or physical compatibility. Use conditional wording and tell the reader to check the current provider/manufacturer limits.
- Reconcile apparently conflicting fit reports explicitly: nominal size labels do not guarantee fit because physical dimensions, protective cases, body shape and conditions vary.
- Keep useful positive and negative evidence.
- Remove vague claims such as "the company listened to feedback" unless the concrete change is stated.
- Remove repeated model-selection conclusions already made elsewhere.
- Treat adjacent H3 sections under the same H2 as distinct editorial jobs. Assign each fact, threshold, drawback and user example to the single subsection whose heading fits it best.
- When one sibling H3 gives a broad feature inventory and another covers a specialised feature or real-world use, keep the shared inventory only in the broad section. The specialised section must focus on heading-specific implications, fit, positioning, padding, limitations or scenarios.
- If adjacent subsections repeat the same evidence or conclusion, keep the stronger version once and shorten the overlap by roughly 15-25%.
- Do not repeat the same pocket list, device example, body-fit example, comfort threshold, missing feature or ventilation point in two adjacent subsections.
- Remove low-information attribution fragments such as "One source reports the pack is durable" unless they are replaced by a concrete supported observation.
- Rewrite colon-led source fragments into natural prose, for example "One source reports that..." or "One source notes that...".
- Remove orphaned feature fragments such as standalone sentences beginning "Made from", "Includes" or "Comfortable with" when the same fact is already explained in the surrounding section.
- Do not describe ordinary storage pockets, laptop sleeves or sternum straps as anti-theft features. State plainly when there are no dedicated anti-theft features.
- Capacity alone never proves airline compatibility. Use attributed exact-product travel experience and require readers to check current carrier dimensions.
- Replace unsupported comparative superiority such as "few competitors match", "better than most rivals", "unmatched" or "best-in-class" with a defensible product-specific benefit.
- Remove tautologies that state the same measurement twice; state it once and add one useful implication.
- A precise lifespan or ownership-duration range from one secondary source must become attributed nonnumeric long-term wording.
- Never use a related model's complaint or test result to qualify the reviewed product without independent evidence for the reviewed product.
- Ensure the first summary paragraph after every concrete H2 directly addresses that H2 using verified facts.
- Replace broad compatibility claims such as "most airlines" with attributed exact-product experience plus a check-the-rules qualification.
- Do not use absolute recommendations such as "without hesitation", "without reservation", "must-buy" or "no-brainer"; state the intended buyer or use case instead.
- Reserve broad "choose X for...; choose Y for..." guidance for comparison sections and the final verdict.
- Do not add links, images, scripts, prices, headings, tables or lists.
- You may remove a redundant table, but never add a table or increase table count.
- Keep the article between 70% and 105% of its original length.
""".strip()

    reference = _canonical_profile_prompt_block(canonical_profile) + "\n\nHTML ARTICLE TO EDIT:\n" + original
    raw = deepseek_generate(
        prompt,
        model=DEEPSEEK_MODEL,
        label="final_editorial_consistency",
        max_tokens=int(controls.get("max_tokens", 10000)),
        temperature=float(controls.get("temperature", 0.1)),
        request_timeout=int(controls.get("request_timeout", 180)),
        cache_prefix=reference,
    )
    candidate = clean_generated_document_html(raw)

    failures = []
    if _html_heading_signature(candidate, "h2") != _html_heading_signature(original, "h2"):
        failures.append("h2_signature_changed")
    if _html_heading_signature(candidate, "h3") != _html_heading_signature(original, "h3"):
        failures.append("h3_signature_changed")
    if recommended_product and recommended_product.casefold() not in candidate.casefold():
        failures.append("primary_product_missing")
    ratio = len(candidate) / max(len(original), 1)
    if not 0.70 <= ratio <= 1.05:
        failures.append(f"length_ratio_{ratio:.2f}")
    if len(re.findall(r"(?is)<table\b", candidate)) > len(re.findall(r"(?is)<table\b", original)):
        failures.append("table_count_increased")
    if len(re.findall(r"(?is)<script\b", candidate)) > len(re.findall(r"(?is)<script\b", original)):
        failures.append("script_count_increased")

    if failures:
        report["reason"] = ",".join(failures)
        logging.warning("[EDITORIAL_PASS] Rejected candidate: %s", report["reason"])
        return original, report

    report.update({
        "applied": True,
        "reason": "accepted",
        "final_chars": len(candidate),
        "length_ratio": round(ratio, 3),
    })
    return candidate, report


def _visible_article_text(
    html: str,
    *,
    include_tables: bool = True,
) -> str:
    """Return normalized visible article text.

    Tables remain included by default because model-assisted audits and identity
    checks benefit from seeing the whole article. Deterministic prose provenance
    checks can set ``include_tables=False`` so row/column values are not flattened
    into a text stream and accidentally assigned to the wrong product.
    """
    soup = BeautifulSoup(html or "", "html.parser")
    for node in soup.find_all(["script", "style", "code", "pre"]):
        node.decompose()

    if not include_tables:
        for table in soup.find_all("table"):
            table.decompose()

    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()


def _product_mention_position(
    passage: str,
    product: str,
    canonical_profile: dict,
) -> int | None:
    passage_folded = re.sub(r"\s+", " ", str(passage or "").casefold())
    variants = [product]
    for alias, canonical in (canonical_profile.get("aliases") or {}).items():
        if canonical_product_identity_key(str(canonical)) == canonical_product_identity_key(product):
            variants.append(str(alias))
    positions = []
    passage_markers = _provenance_markers(passage)
    for variant in variants:
        normalized = re.sub(r"\s+", " ", variant.casefold()).strip()
        if normalized:
            position = passage_folded.find(normalized)
            if position >= 0:
                positions.append(position)
        for value in re.findall(r"(?<!\w)\d+(?:\.\d+)?\s*[a-zA-Z]+(?!\w)", variant):
            marker = _normalize_provenance_marker(value)
            if marker and marker in passage_markers:
                marker_position = passage_folded.find(marker)
                if marker_position < 0:
                    compact = re.sub(r"\s*[-\u2013\u2014]\s*", "", passage_folded)
                    marker_position = compact.find(marker)
                positions.append(marker_position if marker_position >= 0 else len(passage_folded))
    return min(positions) if positions else None


def _product_context_position(
    passage: str,
    product: str,
    canonical_profile: dict,
) -> int | None:
    """Return only mentions strong enough to change the active product subject."""
    passage_folded = re.sub(r"\s+", " ", str(passage or "").casefold()).strip()
    passage_folded = passage_folded.replace("\u2019", "'").replace("\u2018", "'")
    variants = [product]
    for alias, canonical in (canonical_profile.get("aliases") or {}).items():
        if canonical_product_identity_key(str(canonical)) == canonical_product_identity_key(product):
            variants.append(str(alias))
    positions = []
    for variant in variants:
        normalized = re.sub(r"\s+", " ", variant.casefold()).strip()
        normalized = normalized.replace("\u2019", "'").replace("\u2018", "'")
        if normalized:
            position = passage_folded.find(normalized)
            if position >= 0:
                positions.append(position)
    product_markers = sorted(_provenance_markers(product), key=len, reverse=True)
    # Generated copy may reorder or lightly normalize a full product name (for
    # example, moving a demographic qualifier or joining a compound model word).
    # Treat that as a subject-setting mention only when a product identity marker
    # and enough non-numeric family terms occur together. Requiring both avoids
    # allowing shorthand such as "the 25L version" to steal the active context.
    passage_markers = _provenance_markers(passage)
    shared_family = _family_identity_tokens(passage) & _family_identity_tokens(product)
    context_min_shared = int(
        ((_runtime_category_config().get("semantic_fact_audit") or {})
         .get("context_identity_min_shared_tokens", 2))
    )
    if len(shared_family) >= max(1, context_min_shared):
        compact_passage = re.sub(r"\s*[-\u2013\u2014]\s*", "", passage_folded)
        for marker in product_markers:
            if marker not in passage_markers:
                continue
            marker_position = compact_passage.find(marker)
            positions.append(marker_position if marker_position >= 0 else 0)
    for marker in product_markers:
        if re.match(
            rf"^(?:the\s+)?{re.escape(marker)}\b",
            _normalize_provenance_marker(passage_folded),
        ):
            positions.append(0)
    return min(positions) if positions else None

def _product_mentioned_in_passage(
    passage: str,
    product: str,
    canonical_profile: dict,
) -> bool:
    return _product_mention_position(passage, product, canonical_profile) is not None

def _fact_for_attribute(product: dict, attribute: str) -> dict:
    target = str(attribute or "").casefold()
    for fact_id, fact in (product.get("facts") or {}).items():
        if str(fact_id).casefold() == target and isinstance(fact, dict):
            return fact
    return {}


def _profile_product_for_name(canonical_profile: dict, product_name: str) -> dict:
    """Return the canonical product record matching one audit subject."""
    target = canonical_product_identity_key(str(product_name or ""))
    if not target:
        return {}
    for product in canonical_profile.get("products") or []:
        if (
            isinstance(product, dict)
            and canonical_product_identity_key(str(product.get("name") or "")) == target
        ):
            return product
    return {}


def _model_semantic_violation_is_actionable(
    item: dict,
    canonical_profile: dict,
    cfg: dict,
) -> bool:
    """Validate a model finding against the deterministic canonical ledger."""
    passage = re.sub(r"\s+", " ", str((item or {}).get("passage") or "")).strip()
    reason = re.sub(r"\s+", " ", str((item or {}).get("reason") or "")).strip()
    repair = re.sub(r"\s+", " ", str((item or {}).get("repair") or "")).strip()
    product_name = str((item or {}).get("product") or "").strip()
    attribute = str((item or {}).get("attribute") or "").strip()
    if not passage or not reason:
        return False
    if re.search(r"\bno repair (?:is )?needed\b|\bno change (?:is )?needed\b", repair, re.I):
        return False

    product = _profile_product_for_name(canonical_profile, product_name)
    fact = _fact_for_attribute(product, attribute) if product else {}

    def claim_literal(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()

    safe_wording = str(fact.get("safe_wording") or "").strip()
    if safe_wording and claim_literal(passage) == claim_literal(safe_wording):
        return False

    explicitly_attributed = bool(re.match(
        r"^(?:according to|one|a|the)\s+(?:source|reviewer|user|owner|tester)\b",
        passage,
        re.I,
    ))
    universal_scope = bool(re.search(
        r"\b(?:all|always|every|universally|guaranteed|without exception)\b",
        passage,
        re.I,
    ))
    if (
        re.match(r"^keep\b", repair, re.I)
        and explicitly_attributed
        and not universal_scope
    ):
        return False

    nonfactual_editorial_objection = bool(re.search(
        r"\b(?:redundan\w*|repeat(?:ed|s|ing)?(?:\s+it)?\s+(?:multiple|several)|"
        r"overly cautious|understat(?:e|es|ed|ing)|section placement|"
        r"appears? in (?:the )?.{0,60}\bsection|"
        r"phrasing (?:may|might|could) (?:confuse|be confusing))\b",
        reason,
        re.I,
    ))
    if nonfactual_editorial_objection:
        return False

    cited_passage_approved = bool(re.search(
        r"\b(?:this|the (?:claim|statement|sentence|phrase|wording))\s+is\s+"
        r"(?:correct|consistent|acceptable|correctly attributed)\b|"
        r"\bis correctly attributed\b",
        reason,
        re.I,
    ))
    objection_points_elsewhere = bool(re.search(
        r"\b(?:but|however)\b.*\b(?:article|another|elsewhere|later|second)|"
        r"\b(?:article|another passage|elsewhere|later)\s+(?:also\s+)?(?:says|states|claims|uses|repeats)",
        reason,
        re.I,
    ))
    if cited_passage_approved and objection_points_elsewhere:
        return False

    attribution_dispute = bool(re.search(
        r"\b(?:requires?|needs?) attribution\b|\bwithout attribution\b|"
        r"\badd attribution\b|\battribute (?:the|this)\b",
        f"{reason} {repair}",
        re.I,
    ))
    already_attributed = bool(re.match(
        r"^(?:according to|one|a|the)\s+(?:source|reviewer|user|owner|tester)\b",
        passage,
        re.I,
    ))
    if attribution_dispute and fact:
        if not bool(fact.get("requires_attribution")) or already_attributed:
            return False

    value_status = str(fact.get("value_status") or "").casefold()
    canonical_value = str(fact.get("canonical_value") or "").strip()
    uncertainty_disclaimer = bool(re.search(
        r"\b(?:could not|cannot|can't|unable to|not)\b.{0,45}"
        r"\b(?:confirm(?:ed)?|verify|verified|reliably establish)\b|"
        r"\b(?:unconfirmed|not reliably confirmed|unknown)\b",
        passage,
        re.I,
    ))
    if fact and uncertainty_disclaimer and (
        value_status == "unresolved" or not canonical_value
    ):
        return False

    ownership_dispute = bool(re.search(
        r"\bbelongs? (?:only )?to\b|\bevidence (?:is|was|comes?) from\b|"
        r"\bnot (?:confirmed|supported) for\b|\bbelongs to another\b",
        reason,
        re.I,
    ))
    if ownership_dispute and product:
        passage_markers = {
            _normalize_provenance_marker(value)
            for value in _provenance_markers(passage, cfg)
        }
        passage_markers.discard("")
        owned_markers = set()
        target_key = canonical_product_identity_key(product_name)
        for provenance in canonical_profile.get("claim_provenance") or []:
            if canonical_product_identity_key(
                str((provenance or {}).get("product") or "")
            ) != target_key:
                continue
            markers = (provenance or {}).get("distinctive_markers") or _provenance_markers(
                str((provenance or {}).get("evidence_excerpt") or ""), cfg
            )
            owned_markers.update(
                _normalize_provenance_marker(value) for value in markers
            )
        owned_markers.discard("")
        forbidden_markers = set()
        for value in [
            *(fact.get("conflicting_values") or []),
            *(fact.get("forbidden_terms") or []),
        ]:
            forbidden_markers.update(_provenance_markers(str(value), cfg))
        supported_markers = passage_markers & owned_markers
        if supported_markers and not (supported_markers & forbidden_markers):
            return False

    return True

def _section_lead_alignment_violations(html: str, canonical_profile: dict, recommended_product: str, cfg: dict) -> list[dict]:
    """Find concrete H2 headings whose first summary paragraph is off-topic."""
    controls = cfg.get("semantic_fact_audit") or {}
    if controls.get("check_section_lead_alignment", True) is False:
        return []
    stop_words = {str(value).casefold() for value in controls.get(
        "section_alignment_stop_words",
        ["and", "or", "the", "a", "an", "for", "with", "without", "of", "to", "in", "on", "its", "this", "that", "product", "model", "review", "performance", "features", "design"],
    )}
    product_words = set()
    for product in canonical_profile.get("products") or []:
        product_words.update(re.findall(r"[a-z0-9]+", str(product.get("name") or "").casefold()))

    def tokens(value: str) -> set[str]:
        words = re.findall(r"[a-z0-9]+", re.sub(r"^\s*\d+(?:\.\d+)?\s*", "", value.casefold()))
        normalized = set()
        for word in words:
            if word in stop_words or word in product_words or len(word) < 3:
                continue
            if word.endswith("ies") and len(word) > 4:
                word = word[:-3] + "y"
            elif word.endswith("s") and len(word) > 4:
                word = word[:-1]
            normalized.add(word)
        return normalized

    exclusions = [str(value).casefold() for value in controls.get(
        "section_alignment_excluded_headings",
        ["who is it for", "user experience", "value for money", "final verdict", "frequently asked questions", "other models"],
    )]
    soup = BeautifulSoup(html or "", "html.parser")
    violations = []
    for heading in soup.find_all("h2"):
        heading_text = re.sub(r"\s+", " ", heading.get_text(" ", strip=True)).strip()
        if any(value in heading_text.casefold() for value in exclusions):
            continue
        lead = None
        node = heading.find_next_sibling()
        while node is not None:
            if getattr(node, "name", None) in {"h2", "h3"}:
                break
            if getattr(node, "name", None) == "p":
                candidate = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
                if candidate:
                    lead = candidate
                    break
            node = node.find_next_sibling()
        if not lead:
            continue
        heading_tokens = tokens(heading_text)
        topic_tokens = set(heading_tokens)
        if controls.get("section_alignment_include_child_headings", True):
            child = heading.find_next_sibling()
            while child is not None and getattr(child, "name", None) != "h2":
                if getattr(child, "name", None) == "h3":
                    topic_tokens.update(tokens(child.get_text(" ", strip=True)))
                child = child.find_next_sibling()
        minimum = max(1, int(controls.get("section_alignment_min_heading_tokens", 2)))
        lead_tokens = tokens(lead)
        related_topic = bool(topic_tokens & lead_tokens) or any(
            len(left) >= 5 and len(right) >= 5 and left[:5] == right[:5]
            for left in topic_tokens
            for right in lead_tokens
        )
        if len(heading_tokens) < minimum or related_topic:
            continue
        violations.append({
            "product": recommended_product,
            "attribute": "section_lead_mismatch",
            "passage": lead[:700],
            "reason": f"The first summary under '{heading_text}' does not address the heading's concrete topics.",
            "repair": "Rewrite only this summary sentence using verified facts that directly match the heading.",
        })
    return violations

def _deterministic_semantic_claim_violations(
    html: str,
    canonical_profile: dict,
    recommended_product: str,
    cfg: dict,
) -> list[dict]:
    """Catch source-ownership and incomplete-comparison errors without an LLM."""
    controls = cfg.get("semantic_fact_audit") or {}
    if controls.get("deterministic_checks", True) is False:
        return []

    products = [
        product for product in (canonical_profile.get("products") or [])
        if isinstance(product, dict) and str(product.get("name") or "").strip()
    ]
    if not products:
        return []

    primary = str(recommended_product or canonical_profile.get("primary_product") or "")

    # Comparison tables need row/column context. Flattening them into prose can
    # make a value from one product column appear to belong to the active prose
    # subject (for example, a 32L weight being assigned to the reviewed 25L).
    # Keep tables visible to the model-assisted audit, but exclude them from this
    # deterministic prose-provenance scan.
    prose_text = _visible_article_text(html, include_tables=False)
    passages = [
        value.strip()
        for value in re.split(r"(?<=[.!?])\s+|[\r\n]+", prose_text)
        if value.strip()
    ]
    passage_contexts = []
    active_product = primary
    for passage in passages:
        named_products = [
            str(product.get("name") or "")
            for product in products
            if _product_context_position(
                passage, str(product.get("name") or ""), canonical_profile
            ) is not None
        ]
        if named_products:
            active_product = min(
                named_products,
                key=lambda product: _product_context_position(
                    passage, product, canonical_profile
                ),
            )
        passage_contexts.append(active_product)

    identity_markers = set()
    for product in products:
        identity_markers.update(_provenance_markers(str(product.get("name") or ""), cfg))
    for alias in (canonical_profile.get("aliases") or {}):
        identity_markers.update(_provenance_markers(str(alias), cfg))

    ledger = canonical_profile.get("claim_provenance") or []
    if not ledger:
        ledger = []
        for product in products:
            for attribute, fact in (product.get("facts") or {}).items():
                excerpt = str((fact or {}).get("evidence_excerpt") or "")
                if excerpt:
                    ledger.append({
                        "product": product.get("name", ""),
                        "attribute": attribute,
                        "evidence_excerpt": excerpt,
                        "distinctive_markers": sorted(_provenance_markers(excerpt, cfg)),
                    })

    marker_owners = {}
    for item in ledger:
        owner = str((item or {}).get("product") or "").strip()
        attribute = str((item or {}).get("attribute") or "claim_provenance").strip()
        markers = (item or {}).get("distinctive_markers") or _provenance_markers(
            str((item or {}).get("evidence_excerpt") or ""), cfg
        )
        for marker in markers:
            normalized = _normalize_provenance_marker(marker)
            if normalized and normalized not in identity_markers:
                marker_owners.setdefault(normalized, set()).add((owner, attribute))

    violations = []
    seen = set()
    for passage_index, passage in enumerate(passages):
        passage_markers = _provenance_markers(passage, cfg)
        for marker in sorted(passage_markers):
            owners = marker_owners.get(marker) or set()
            owner_names = {owner for owner, _attribute in owners if owner}
            if len(owner_names) != 1:
                continue
            owner = next(iter(owner_names))
            if canonical_product_identity_key(owner) == canonical_product_identity_key(primary):
                continue
            if _product_mentioned_in_passage(passage, owner, canonical_profile):
                continue

            context_product = passage_contexts[passage_index]
            if canonical_product_identity_key(context_product) == canonical_product_identity_key(owner):
                continue
            mentioned = [
                str(product.get("name") or "")
                for product in products
                if _product_mentioned_in_passage(
                    passage, str(product.get("name") or ""), canonical_profile
                )
            ]
            assigned = mentioned[0] if len(mentioned) == 1 else (context_product or primary)
            if canonical_product_identity_key(assigned) == canonical_product_identity_key(owner):
                continue
            key = ("claim_provenance", passage.casefold(), owner.casefold())
            if key in seen:
                continue
            seen.add(key)
            attribute = next(
                (value for product_name, value in owners if product_name == owner),
                "claim_provenance",
            )
            violations.append({
                "product": assigned or primary,
                "attribute": "claim_provenance",
                "passage": passage[:700],
                "reason": (
                    f"The distinctive evidence marker '{marker}' belongs only to "
                    f"{owner}, but this passage assigns or implies it for {assigned or primary}."
                ),
                "repair": (
                    f"Remove the observation or explicitly attribute it to {owner}; "
                    "do not transfer it to the reviewed product."
                ),
                "evidence_owner": owner,
                "evidence_attribute": attribute,
            })

    superlatives = controls.get("comparative_superlatives") or {
        "lightest": "weight",
        "heaviest": "weight",
    }
    comparison_cues = [
        str(value).casefold()
        for value in (controls.get("comparison_scope_cues") or [
            "option", "model", "product", "choice", "of the", "among", "overall",
        ])
    ]
    if len(products) > 1:
        for passage in passages:
            folded = passage.casefold()
            for term, attribute in superlatives.items():
                term_folded = str(term).casefold()
                if term_folded not in folded or not any(cue in folded for cue in comparison_cues):
                    continue
                missing = []
                for product in products:
                    fact = _fact_for_attribute(product, str(attribute))
                    if (
                        not str(fact.get("canonical_value") or "").strip()
                        or str(fact.get("value_status") or "").lower()
                        in {"unresolved", "source_conflict"}
                        or str(fact.get("product_scope") or "").lower() == "ambiguous"
                    ):
                        missing.append(str(product.get("name") or "unknown product"))
                if not missing:
                    continue
                key = ("incomplete_superlative", passage.casefold(), term_folded)
                if key in seen:
                    continue
                seen.add(key)
                violations.append({
                    "product": primary,
                    "attribute": "incomplete_superlative",
                    "passage": passage[:700],
                    "reason": (
                        f"'{term}' requires a confirmed {attribute} value for every "
                        f"compared product; missing or unresolved: {', '.join(missing)}."
                    ),
                    "repair": (
                        "Replace the superlative with a supported pairwise comparison "
                        "or omit the ranking."
                    ),
                })

    measurement_verbs = [str(value).casefold() for value in controls.get(
        "tautological_measurement_verbs",
        ["weigh", "measure", "capacity", "runtime", "run time", "power", "consume", "draw", "output", "hold", "store"],
    )]
    comparative_terms = [str(value).casefold() for value in controls.get(
        "unsupported_comparative_terms",
        ["outlast the competition", "outlast competitors", "industry-leading", "best-in-class", "superior to competitors"],
    )]
    compatibility_terms = [str(value).casefold() for value in controls.get(
        "universal_compatibility_terms",
        [
            "most airlines",
            "all airlines",
            "airline approved",
            "guaranteed carry-on",
            "guaranteed to fit",
            "typical cabin baggage limits",
            "standard carry-on use",
        ],
    )]
    for passage_index, passage in enumerate(passages):
        folded = passage.casefold()
        compact = re.sub(r"(?<=\d)\s*[-\u2013\u2014]?\s*(?=[a-z%])", "", folded)
        matched_comparative = next((term for term in comparative_terms if term and term in folded), "")
        if matched_comparative:
            key = ("unsupported_comparative", passage.casefold(), matched_comparative)
            if key not in seen:
                seen.add(key)
                violations.append({
                    "product": passage_contexts[passage_index] or primary,
                    "attribute": "unsupported_comparative",
                    "passage": passage[:700],
                    "reason": f"'{matched_comparative}' claims competitor superiority without a complete named comparison.",
                    "repair": "Replace it with a product-specific, non-comparative benefit.",
                })
        matched_compatibility = next((term for term in compatibility_terms if term and term in folded), "")
        if matched_compatibility and not passage.rstrip().endswith("?"):
            key = ("universal_compatibility", passage.casefold(), matched_compatibility)
            if key not in seen:
                seen.add(key)
                violations.append({
                    "product": passage_contexts[passage_index] or primary,
                    "attribute": "universal_compatibility",
                    "passage": passage[:700],
                    "reason": f"'{matched_compatibility}' is a broad compatibility claim not established by one report or a proxy specification.",
                    "repair": "Use attributed exact-product experience and tell readers to check the applicable dimensions or rules.",
                })

    soup = BeautifulSoup(html or "", "html.parser")
    for node in soup.find_all(["p", "li"]):
        if node.find_parent(["table", "script", "style"]):
            continue
        node_text = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
        for passage in re.split(r"(?<=[.!?])\s+|[\r\n]+", node_text):
            passage = passage.strip()
            if not passage:
                continue
            folded = passage.casefold()
            compact = re.sub(r"(?<=\d)\s*[-\u2013\u2014]?\s*(?=[a-z%])", "", folded)
            for marker in _provenance_markers(passage, cfg):
                if marker in identity_markers:
                    continue
                marker_occurrences = re.findall(
                    rf"(?<![a-z0-9.]){re.escape(marker)}(?![a-z0-9])",
                    compact,
                    re.I,
                )
                if len(marker_occurrences) >= 2 and any(
                    verb in folded for verb in measurement_verbs
                ):
                    key = ("repeated_measurement", passage.casefold(), marker)
                    if key not in seen:
                        seen.add(key)
                        named = [
                            str(product.get("name") or "")
                            for product in products
                            if _product_mentioned_in_passage(
                                passage, str(product.get("name") or ""), canonical_profile
                            )
                        ]
                        violations.append({
                            "product": named[0] if len(named) == 1 else primary,
                            "attribute": "repeated_measurement",
                            "passage": passage[:700],
                            "reason": f"The measurement '{marker}' is stated twice in one sentence.",
                            "repair": "State the measurement once and keep one useful consumer implication.",
                        })
                    break
    for item in _section_lead_alignment_violations(html, canonical_profile, primary, cfg):
        key = (item.get("attribute"), item.get("passage", "").casefold())
        if key not in seen:
            seen.add(key)
            violations.append(item)
    limit = max(1, int(controls.get("max_deterministic_violations", 12)))
    return violations[:limit]


def audit_semantic_claim_consistency(
    html: str,
    canonical_profile: dict,
    keyword: str,
    recommended_product: str,
    cfg: dict,
) -> tuple[list[dict], dict]:
    """Model-assisted audit limited to objective atomic product claims."""
    controls = cfg.get("semantic_fact_audit") or {}
    report = {
        "enabled": bool(controls.get("enabled", True)),
        "attempted": False,
        "reason": "",
        "violations": [],
    }
    if controls.get("enabled", True) is False or not canonical_profile:
        report["reason"] = "disabled_or_no_profile"
        return [], report

    report["attempted"] = True
    visible_text = _visible_article_text(html)
    deterministic_violations = _deterministic_semantic_claim_violations(
        html,
        canonical_profile,
        recommended_product,
        cfg,
    )
    prompt = f"""
Audit objective factual consistency in a product review.

Article topic: {keyword}
Primary reviewed product: {recommended_product}
Configured generic controls:
{json.dumps({
    "collective_claim_terms": controls.get("collective_claim_terms") or [],
    "unsupported_claim_strength_terms": controls.get("unsupported_claim_strength_terms") or [],
    "comparative_superlatives": controls.get("comparative_superlatives") or {},
    "unsupported_comparative_terms": controls.get("unsupported_comparative_terms") or [],
    "universal_compatibility_terms": controls.get("universal_compatibility_terms") or [],
}, ensure_ascii=False, indent=2)}

Return STRICT JSON only:
{{
  "violations": [
    {{
      "product": "exact product name",
      "attribute": "atomic attribute ID",
      "passage": "exact complete sentence copied from the visible article",
      "reason": "short factual explanation",
      "repair": "safe replacement instruction"
    }}
  ]
}}

Audit ONLY:
- claims assigned to the wrong product, size, variant or generation;
- feature present versus absent contradictions;
- dedicated hardware versus an ordinary component that can serve a similar purpose;
- incompatible hard numbers or universal compatibility claims;
- current versus legacy specification mixing;
- hard factual claims not licensed by the matching canonical product record;
- conflicting statements about locking, security, pass-throughs, compatibility,
  dimensions, weight, capacity, ratings or safety limits.
- distinctive measurements or user observations whose evidence belongs only to
  another product, even when the article says only "one reviewer" or "one user";
- superlatives such as lightest or heaviest when any product in the stated
  comparison lacks a confirmed value for the relevant attribute;
- quantified feature claims such as "both alternatives", "the other two" or
  "all models" unless every included product record affirmatively supports it;
- strength upgrades such as heavy-duty, heavy-load or premium comfort when the
  canonical evidence supports only moderate loads or ordinary comfort;
- airline, regulatory or physical-fit conclusions inferred only from capacity.
- repeated or tautological measurements within one sentence;
- unsupported competitor superiority without a complete named comparison;
- a recommendation or warning transferred from a related model to the primary product, even when the source model is named;
- a concrete H2 whose first summary sentence discusses a different feature;
- broad compatibility language such as "most airlines" without exact-product evidence sufficient for that scope.

Rules:
- Treat the canonical profile as authoritative and product-scoped.
- Names listed in the profile's aliases map are the same product identity; do not flag an alias merely because its word order or generic type word differs.
- A fact in Product B's record never licenses the same claim for Product A.
- Advice, warnings and buying qualifications are claims too. Do not use Product B's user experience to qualify Product A unless Product A has independent supporting evidence.
- The first summary sentence after a concrete H2 must directly address that heading, not an unrelated feature from another section.
- Do not return an item as a violation when its own reasoning says no violation or no repair is needed.
- Treat claim_provenance entries and their distinctive_markers as exclusive to
  the named evidence owner unless another product record independently supports them.
- A collective superlative requires a comparable confirmed value for every model
  in that comparison. Recommend a supported pairwise statement when one is unknown.
- A claim about several products requires separate affirmative evidence for each;
  silence or an unresolved fact is not confirmation.
- Keep load and comfort language at or below the strength of safe_wording.
- A single attributed real-world compatibility report is not a universal specification.
- Capacity or another proxy does not establish dimensions, compliance, airline acceptance or fit.
- Do not classify visibility, padding, ordinary pockets or bottle retention as security.
- Do not flag subjective opinions, normal pros/cons balance, writing style or repetition.
- Do not flag compatible nuances, such as comfortable at one load and manageable
  briefly at a slightly higher load.
- Ordinary zipper pulls accepting a padlock is compatible with no dedicated
  lockable hardware only when the article explicitly distinguishes them.
- A general sleeve is not the same as a dedicated separate compartment.
- Copy each passage exactly from visible article text.
- Return at most 12 violations. Return an empty list when none exist.

CANONICAL PROFILE:
{json.dumps(canonical_profile, ensure_ascii=False, indent=2)}

VISIBLE ARTICLE:
{visible_text}
""".strip()

    try:
        raw = deepseek_generate(
            prompt,
            model=DEEPSEEK_MODEL,
            label="semantic_fact_audit",
            max_tokens=int(controls.get("max_tokens", 2600)),
            temperature=float(controls.get("temperature", 0.0)),
            request_timeout=int(controls.get("request_timeout", 180)),
            cache_prefix=_dataset_cache_prefix(visible_text),
        )
        raw_path = str(cfg.get("_semantic_audit_raw_response_path") or "").strip()
        if raw_path:
            with open(raw_path, "w", encoding="utf-8") as raw_file:
                raw_file.write(raw)
        try:
            parsed = _json_from_text_block(raw)
        except ValueError as parse_exc:
            if "No valid JSON found" not in str(parse_exc):
                raise
            retry_raw = deepseek_generate(
                prompt + "\n\nYour previous response was invalid. Return the JSON object only, with no explanation.",
                model=DEEPSEEK_MODEL,
                label="semantic_fact_audit_retry",
                max_tokens=int(controls.get("max_tokens", 2600)),
                temperature=float(controls.get("temperature", 0.0)),
                request_timeout=int(controls.get("request_timeout", 180)),
                cache_prefix=_dataset_cache_prefix(visible_text),
            )
            if raw_path:
                with open(raw_path, "w", encoding="utf-8") as raw_file:
                    raw_file.write(retry_raw)
            parsed = _json_from_text_block(retry_raw)
    except Exception as exc:
        report["reason"] = f"exception:{exc}"
        logging.exception("[SEMANTIC_AUDIT] Audit failed: %s", exc)
        if controls.get("block_on_audit_failure", True):
            failure = {
                "product": recommended_product,
                "attribute": "audit_failure",
                "passage": "",
                "reason": "Semantic fact audit could not be completed.",
                "repair": "Inspect the raw audit response and rerun.",
            }
            combined = [*deterministic_violations, failure]
            report["deterministic_violation_count"] = len(deterministic_violations)
            report["violations"] = combined
            return combined, report
        report["deterministic_violation_count"] = len(deterministic_violations)
        report["violations"] = deterministic_violations
        return deterministic_violations, report

    violations = list(deterministic_violations)
    seen_violations = {
        (
            canonical_product_identity_key(str(item.get("product") or recommended_product)),
            str(item.get("attribute") or "").casefold(),
            re.sub(r"\s+", " ", str(item.get("passage") or "").casefold()).strip(),
        )
        for item in violations
    }
    parsed_items = (parsed.get("violations") or []) if isinstance(parsed, dict) else []
    for item in parsed_items:
        if len(violations) >= 12:
            break
        if not isinstance(item, dict):
            continue
        passage = re.sub(r"\s+", " ", str(item.get("passage") or "")).strip()
        reason = re.sub(r"\s+", " ", str(item.get("reason") or "")).strip()
        if not passage or not reason:
            continue
        if re.search(r"(?i)\bno violation\b|\bno repair needed\b", reason):
            continue
        # Reject invented audit passages. Minor quote normalization is allowed,
        # but the model must point to text that is actually visible.
        probe = passage.casefold().strip(" \"'")
        article_folded = visible_text.casefold()
        if probe not in article_folded:
            prefix = probe[:80].strip()
            if len(prefix) < 30 or prefix not in article_folded:
                continue
        product_name = re.sub(
            r"\s+", " ", str(item.get("product") or recommended_product)
        ).strip()
        attribute = re.sub(
            r"\s+", "_", str(item.get("attribute") or "unknown")
        ).strip("_")
        repair = re.sub(
            r"\s+", " ", str(item.get("repair") or "")
        ).strip()
        # Validate model findings against the canonical ledger before allowing
        # them to drive another rewrite round.
        validation_item = {
            "product": product_name,
            "attribute": attribute,
            "passage": passage,
            "reason": reason,
            "repair": repair,
        }
        if not _model_semantic_violation_is_actionable(
            validation_item, canonical_profile, cfg
        ):
            continue
        # Reject duplicate records and passages the auditor itself says to keep.
        if (
            re.search(r"\b(?:sentence|statement|passage|disclaimer) is acceptable\b", reason, re.I)
            and re.search(r"\bkeep\b|\bno change\b", repair, re.I)
        ):
            continue
        attributed_observation_attributes = {
            str(value).casefold()
            for value in controls.get(
                "attributed_observation_attributes",
                ["dimensions", "compatibility", "airline_compatibility", "carry_on_compatibility", "underseat_compatibility"],
            )
        }
        explicitly_attributed = bool(re.match(
            r"^(?:one|a|the)\s+(?:reviewer|source|user|owner|tester)\b",
            passage,
            re.I,
        ))
        universal_scope = bool(re.search(
            r"\b(?:all|always|every|guaranteed|approved|universally|"
            r"meets? (?:all|every)|within (?:all|every))\b",
            passage,
            re.I,
        ))
        if (
            attribute.casefold() in attributed_observation_attributes
            and explicitly_attributed
            and not universal_scope
        ):
            continue
        violation_key = (
            canonical_product_identity_key(product_name),
            attribute.casefold(),
            passage.casefold(),
        )
        if violation_key in seen_violations:
            continue
        seen_violations.add(violation_key)
        violations.append({
            "product": product_name,
            "attribute": attribute,
            "passage": passage[:700],
            "reason": reason[:500],
            "repair": repair[:500],
        })

    report["reason"] = "completed"
    report["deterministic_violation_count"] = len(deterministic_violations)
    report["violations"] = violations
    return violations, report


def _deterministic_remove_explicit_omissions(
    html: str,
    violations: list[dict],
) -> tuple[str, set[str]]:
    """Remove a cited passage only when its repair explicitly starts with Omit."""
    working = str(html or "")
    repaired_passages: set[str] = set()
    for violation in violations or []:
        passage = str((violation or {}).get("passage") or "").strip()
        repair = str((violation or {}).get("repair") or "").strip()
        if not passage or passage not in working or not re.match(r"^omit\b", repair, re.I):
            continue
        paragraph_pattern = re.compile(
            rf"<p\b[^>]*>\s*{re.escape(passage)}\s*</p>",
            re.I,
        )
        working, paragraph_count = paragraph_pattern.subn("", working)
        if not paragraph_count:
            working = working.replace(passage, "")
        repaired_passages.add(passage)
    return working, repaired_passages

def _deterministic_remove_cross_model_observations(
    html: str,
    violations: list[dict],
) -> tuple[str, set[str]]:
    """Remove cited observations whose evidence belongs to another product.

    The semantic model can keep paraphrasing a cross-model observation while
    preserving the same ownership error. Once the validated audit explicitly
    identifies another evidence owner, omission is safer than another rewrite.
    """
    working = str(html or "")
    repaired_passages: set[str] = set()
    source_scope_rx = re.compile(
        r"\b(?:belongs? (?:only )?to|belongs? to another|"
        r"from (?:the )?review of|not from evidence for|"
        r"evidence (?:belongs? to|is for) another|"
        r"observation (?:belongs? to|from) another product)\b",
        re.I,
    )
    removal_rx = re.compile(r"\b(?:remove|omit|delete)\b", re.I)
    for violation in violations or []:
        item = violation or {}
        passage = str(item.get("passage") or "").strip()
        product = str(item.get("product") or "").strip()
        owner = str(item.get("evidence_owner") or "").strip()
        attribute = str(item.get("attribute") or "").strip().casefold()
        reason = str(item.get("reason") or "")
        repair = str(item.get("repair") or "")
        owner_mismatch = bool(
            owner
            and canonical_product_identity_key(owner)
            != canonical_product_identity_key(product)
        )
        explicit_provenance_error = attribute == "claim_provenance" and owner_mismatch
        model_scoped_removal = bool(
            source_scope_rx.search(reason) and removal_rx.search(repair)
        )
        if (
            not passage
            or passage not in working
            or not (explicit_provenance_error or model_scoped_removal)
        ):
            continue
        paragraph_pattern = re.compile(
            rf"<p\b[^>]*>\s*{re.escape(passage)}\s*</p>",
            re.I,
        )
        working, paragraph_count = paragraph_pattern.subn("", working)
        if not paragraph_count:
            working = working.replace(passage, "")
        repaired_passages.add(passage)
    return working, repaired_passages


def _deterministic_repair_with_canonical_safe_wording(
    html: str,
    violations: list[dict],
    canonical_profile: dict,
) -> tuple[str, set[str]]:
    """Replace a disputed claim with the exact product fact's safe wording."""
    working = str(html or "")
    repaired_passages: set[str] = set()
    products = [
        item for item in (canonical_profile.get("products") or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    for violation in violations or []:
        passage = str((violation or {}).get("passage") or "").strip()
        product_name = str((violation or {}).get("product") or "").strip()
        attribute = str((violation or {}).get("attribute") or "").strip()
        if not passage or passage not in working or not product_name or not attribute:
            continue
        product_key = canonical_product_identity_key(product_name)
        product = next(
            (
                item for item in products
                if canonical_product_identity_key(str(item.get("name") or ""))
                == product_key
            ),
            None,
        )
        if not product:
            continue
        fact = _fact_for_attribute(product, attribute)
        safe_wording = re.sub(
            r"\s+", " ", str((fact or {}).get("safe_wording") or "")
        ).strip()
        if not safe_wording or safe_wording.casefold() == passage.casefold():
            continue
        reason = str((violation or {}).get("reason") or "")
        repair = str((violation or {}).get("repair") or "")
        safe_probe = re.sub(r"[^a-z0-9]+", " ", safe_wording.casefold()).strip()
        repair_probe = re.sub(r"[^a-z0-9]+", " ", repair.casefold()).strip()
        canonical_contradiction = bool(re.search(
            r"\b(?:contradict\w*|belongs? to|not (?:supported|confirmed)|unsupported|canonical profile (?:confirms|states|lists))\b",
            reason,
            re.I,
        ))
        safe_wording_requested = bool(
            safe_probe and len(safe_probe) >= 20 and safe_probe in repair_probe
        )
        if not (canonical_contradiction or safe_wording_requested):
            continue
        # A safe wording still needs the provenance required by its canonical
        # fact. Otherwise a deterministic repair can create a new audit
        # failure by replacing an attributed reviewer observation with a
        # general product claim.
        if (
            (fact or {}).get("requires_attribution") is True
            and not re.match(
                r"^(?:according to|one source|a source|one reviewer|a reviewer|"
                r"the manufacturer|the product listing)\b",
                safe_wording,
                re.I,
            )
        ):
            safe_wording = "One source reports: " + safe_wording
        working = working.replace(passage, safe_wording)
        repaired_passages.add(passage)
    return working, repaired_passages


def _deterministic_repair_disputed_weight_claims(
    html: str,
    violations: list[dict],
) -> tuple[str, set[str]]:
    """Attribute conflicting reported weights instead of stating one as definitive."""
    working = str(html or "")
    repaired_passages: set[str] = set()
    measurement_rx = re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:g|kg|oz|lb|lbs)\b", re.I
    )
    disagreement_rx = re.compile(
        r"(?:another|different|conflict\w*)\s+source.*reports?", re.I
    )
    for violation in violations or []:
        item = violation or {}
        if str(item.get("attribute") or "").strip().casefold() != "weight":
            continue
        passage = str(item.get("passage") or "").strip()
        reason = str(item.get("reason") or "")
        if not passage or passage not in working or not disagreement_rx.search(reason):
            continue
        measurements: list[str] = []
        for value in measurement_rx.findall(passage + " " + reason):
            normalized = re.sub(r"\s+", " ", value).strip()
            if normalized.casefold() not in {entry.casefold() for entry in measurements}:
                measurements.append(normalized)
        if len(measurements) < 2:
            continue
        repaired = (
            "Sources report different weights for this model, including "
            + " and ".join(measurements[:2])
            + "."
        )
        working = working.replace(passage, repaired)
        repaired_passages.add(passage)
    return working, repaired_passages


def _deterministic_repair_required_attribution(
    html: str,
    violations: list[dict],
) -> tuple[str, set[str]]:
    """Add explicit source attribution when an audit requires it."""
    working = str(html or "")
    repaired_passages: set[str] = set()
    attribution_rx = re.compile(
        r"\b(?:according to|one source|a source|one reviewer|a reviewer|"
        r"the manufacturer|the product listing)\b",
        re.I,
    )
    for violation in violations or []:
        passage = str((violation or {}).get("passage") or "").strip()
        reason = str((violation or {}).get("reason") or "")
        repair = str((violation or {}).get("repair") or "")
        requires_attribution = (
            "requires attribution" in reason.casefold()
            or "add attribution" in repair.casefold()
        )
        if (
            not requires_attribution
            or not passage
            or passage not in working
            or attribution_rx.search(passage[:100])
        ):
            continue
        attributed = "One source reports: " + passage
        working = working.replace(passage, attributed)
        repaired_passages.add(passage)
    return working, repaired_passages

def _deterministic_repair_repeated_measurements(
    html: str,
    violations: list[dict],
) -> tuple[str, set[str]]:
    """Remove a repeated adjectival measurement without changing its claim."""
    working = str(html or "")
    repaired_passages: set[str] = set()
    for violation in violations or []:
        if str((violation or {}).get("attribute") or "") != "repeated_measurement":
            continue
        passage = str((violation or {}).get("passage") or "").strip()
        reason = str((violation or {}).get("reason") or "")
        marker_match = re.search(r"measurement\s+['\"]([^'\"]+)['\"]", reason, re.I)
        if not passage or not marker_match:
            continue
        marker = _normalize_provenance_marker(marker_match.group(1))
        parts = re.fullmatch(r"(\d+(?:\.\d+)?)([a-z%]{1,12})", marker)
        if not parts:
            continue
        number, unit = parts.groups()
        if unit in {"inch", "inches"}:
            unit_pattern = r"inch(?:es)?"
        elif unit in {"foot", "feet"}:
            unit_pattern = r"(?:foot|feet)"
        else:
            unit_pattern = rf"{re.escape(unit)}s?"
        literal_pattern = re.compile(
            rf"(?<!\w){re.escape(number)}\s*(?:[-\u2013\u2014]\s*)?{unit_pattern}(?!\w)",
            re.I,
        )
        matches = list(literal_pattern.finditer(passage))
        if len(matches) < 2:
            continue
        repaired = passage
        changed = False
        for match in reversed(matches[1:]):
            # Removing a repeated adjectival literal is safe ("15-inch device"
            # becomes "device"). Standalone measurements still go to the model.
            if not re.match(r"\s+[A-Za-z]", repaired[match.end():]):
                continue
            repaired = repaired[:match.start()] + repaired[match.end():]
            changed = True
        if not changed:
            # A common tautology combines a threshold implication with the
            # canonical measurement in the same paragraph, for example
            # "loads above 10 kg ... comfortable up to 10 kg". Keep the
            # licensed measurement and make the threshold implication
            # deliberately non-numeric.
            threshold_pattern = re.compile(
                rf"\b(load|loads)\s+(?:above|over|beyond)\s+"
                rf"{re.escape(number)}\s*(?:[-\u2013\u2014]\s*)?{unit_pattern}(?!\w)",
                re.I,
            )
            repaired, threshold_count = threshold_pattern.subn(
                lambda match: (
                    "heavier loads"
                    if match.group(1).casefold().endswith("s")
                    else "a heavier load"
                ),
                repaired,
                count=1,
            )
            changed = threshold_count > 0
        repaired = re.sub(r" {2,}", " ", repaired)
        if changed and repaired != passage:
            if passage in working:
                working = working.replace(passage, repaired)
                repaired_passages.add(passage)
                continue
            # Audit passages are normalized visible text. When the source
            # paragraph contains line breaks, its normalized passage will not
            # be a literal HTML substring. Safely replace a matching plain-text
            # paragraph/list item without flattening nested markup.
            normalized_passage = re.sub(r"\s+", " ", passage).strip()
            soup = BeautifulSoup(working, "html.parser")
            for node in soup.find_all(["p", "li"]):
                if node.find(True) is not None:
                    continue
                node_text = re.sub(
                    r"\s+", " ", node.get_text(" ", strip=True)
                ).strip()
                if normalized_passage not in node_text:
                    continue
                repaired_node_text = node_text.replace(
                    normalized_passage, repaired, 1
                )
                node.clear()
                node.append(repaired_node_text)
                working = str(soup)
                repaired_passages.add(passage)
                break
    return working, repaired_passages


def _semantic_repair_candidate_made_progress(
    input_violations: list[dict],
    candidate_violations: list[dict],
    cited_passages: list[str],
    passages_fixed: int,
) -> bool:
    """Accept a repair that reduces violations or fully resolves its inputs.

    A deterministic safe-wording repair can remove every cited violation and
    expose different violations that were already present in the article. Do
    not discard that safe progress merely because the newly visible audit set
    is larger; accepting it allows the next bounded round to repair the new,
    narrower findings. Never accept a candidate that leaves a cited passage
    unchanged.
    """
    if cited_passages and passages_fixed <= 0:
        return False
    if len(candidate_violations) < len(input_violations):
        return True
    return bool(cited_passages) and passages_fixed == len(cited_passages)


def repair_semantic_claim_conflicts(
    html: str,
    violations: list[dict],
    canonical_profile: dict,
    keyword: str,
    recommended_product: str,
    cfg: dict,
) -> tuple[str, dict]:
    """Repair only semantic-audit passages while preserving document structure."""
    original = clean_generated_document_html(html)
    controls = cfg.get("semantic_fact_audit") or {}
    report = {
        "enabled": bool(controls.get("auto_repair", True)),
        "attempted": False,
        "applied": False,
        "reason": "",
        "initial_violation_count": len(violations or []),
    }
    if not violations or controls.get("auto_repair", True) is False:
        report["reason"] = "disabled_or_no_violations"
        return original, report

    report["attempted"] = True
    omitted_passages: set[str] = set()
    if controls.get("deterministic_explicit_omission_repairs", True):
        original, omitted_passages = _deterministic_remove_explicit_omissions(
            original, violations
        )
    cross_model_passages: set[str] = set()
    if controls.get("deterministic_cross_model_observation_removals", True):
        original, cross_model_passages = (
            _deterministic_remove_cross_model_observations(original, violations)
        )
    safe_wording_passages: set[str] = set()
    if controls.get("deterministic_safe_wording_repairs", True):
        original, safe_wording_passages = (
            _deterministic_repair_with_canonical_safe_wording(
                original, violations, canonical_profile
            )
        )
    original, attributed_passages = _deterministic_repair_required_attribution(
        original, violations
    )
    original, repeated_passages = _deterministic_repair_repeated_measurements(
        original, violations
    )
    repaired_passages = (
        omitted_passages
        | cross_model_passages
        | safe_wording_passages
        | attributed_passages
        | repeated_passages
    )
    report["deterministic_explicit_omission_repairs"] = len(omitted_passages)
    report["deterministic_cross_model_observation_removals"] = len(
        cross_model_passages
    )
    report["deterministic_safe_wording_repairs"] = len(safe_wording_passages)
    report["deterministic_attribution_repairs"] = len(attributed_passages)
    report["deterministic_repeated_measurement_repairs"] = len(repeated_passages)
    report["deterministic_pre_repairs"] = len(repaired_passages)
    if repaired_passages:
        violations = [
            item for item in violations
            if str((item or {}).get("passage") or "").strip() not in repaired_passages
        ]
        if not violations:
            report.update({
                "applied": True,
                "reason": "deterministic_repairs_ready_for_reaudit",
            })
            return original, report

    prompt = f"""
Repair only the listed factual violations in this HTML product review.

Article topic: {keyword}
Primary reviewed product: {recommended_product}

Requirements:
- Return only the complete revised HTML fragment.
- Preserve every H2 and H3 exactly and in the same order.
- Treat each canonical product record as applying only to that exact model,
  size, variant and generation.
- Names in the profile's aliases map refer to the mapped canonical product; normalize them when useful and never treat them as separate variants.
- Remove a claim when the matching product profile does not license it.
- Preserve explicit attribution for single-source reports.
- Distinguish a general feature from dedicated hardware.
- Do not add facts, products, links, images, scripts, prices, headings, tables or lists.
- For claim_provenance violations, remove the observation or name its true evidence owner; never leave it implied for the primary product.
- Replace an incomplete superlative with a supported pairwise comparison or omit the ranking.
- Rewrite collective feature claims so each named product is supported independently.
- Reduce load, comfort, compatibility and airline language to the canonical safe_wording strength.
- Remove repeated measurements and unsupported competitor-superiority phrases.
- For section_lead_mismatch, rewrite only the cited lead sentence so it directly summarizes the existing H2.
- Do not use a related model's complaint to advise buyers of the primary product unless the primary record independently supports it.
- Change only sentences needed to resolve the listed violations.

CANONICAL PROFILE:
{json.dumps(canonical_profile, ensure_ascii=False, indent=2)}

VIOLATIONS:
{json.dumps(violations, ensure_ascii=False, indent=2)}

HTML:
{original}
""".strip()
    try:
        raw = deepseek_generate(
            prompt,
            model=DEEPSEEK_MODEL,
            label="semantic_fact_repair",
            max_tokens=int(controls.get("repair_max_tokens", 10000)),
            temperature=float(controls.get("repair_temperature", 0.0)),
            request_timeout=int(controls.get("request_timeout", 180)),
            cache_prefix=_dataset_cache_prefix(original),
        )
        raw_path = str(cfg.get("_semantic_repair_raw_response_path") or "").strip()
        if raw_path:
            with open(raw_path, "w", encoding="utf-8") as raw_file:
                raw_file.write(raw)
    except Exception as exc:
        report["reason"] = f"exception:{exc}"
        logging.exception("[SEMANTIC_REPAIR] Repair failed: %s", exc)
        return original, report

    candidate = clean_generated_document_html(raw)
    failures = []
    if _html_heading_signature(candidate, "h2") != _html_heading_signature(original, "h2"):
        failures.append("h2_signature_changed")
    if _html_heading_signature(candidate, "h3") != _html_heading_signature(original, "h3"):
        failures.append("h3_signature_changed")
    if recommended_product and recommended_product.casefold() not in candidate.casefold():
        failures.append("primary_product_missing")
    ratio = len(candidate) / max(len(original), 1)
    if not 0.75 <= ratio <= 1.05:
        failures.append(f"length_ratio_{ratio:.2f}")
    if len(re.findall(r"(?is)<table\b", candidate)) > len(re.findall(r"(?is)<table\b", original)):
        failures.append("table_count_increased")
    if len(re.findall(r"(?is)<script\b", candidate)) > len(re.findall(r"(?is)<script\b", original)):
        failures.append("script_count_increased")
    if failures:
        report["reason"] = ",".join(failures)
        return original, report

    report.update({
        "applied": True,
        "reason": "candidate_ready_for_reaudit",
        "length_ratio": round(ratio, 3),
    })
    return candidate, report


def audit_canonical_conflicts(
    html: str,
    canonical_profile: dict,
) -> tuple[list[dict], list[dict]]:
    """Return blocking conflicts and non-blocking contextual mentions.

    A conflicting literal blocks publication only when it is used as an affirmative
    claim. Negated uses, questions, approved safe wording, and source comparisons
    resolved to the canonical value remain visible as warnings instead.
    """
    if not canonical_profile:
        return [], []

    fragments = re.split(r"(?is)\n\s*\n|</(?:p|li|tr|div)>", html or "")
    text_fragments = [
        re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()
        for fragment in fragments
    ]

    def normalized_literal(value: str) -> str:
        value = str(value or "").casefold()
        for source, replacement in (
            ("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"')
        ):
            value = value.replace(source, replacement)
        value = re.sub(r"(?<=\d)\s+(?=[a-z])", "", value)
        return re.sub(r"\s+", " ", value).strip()

    def literal_spans(text: str, value: str) -> list[tuple[int, int]]:
        literal = normalized_literal(value)
        if not literal:
            return []
        pattern = re.escape(literal)
        if literal[0].isalnum():
            pattern = r"(?<!\w)" + pattern
        if literal[-1].isalnum():
            pattern += r"(?!\w)"
        return [match.span() for match in re.finditer(pattern, text)]

    def overlap(span: tuple[int, int], other: tuple[int, int]) -> bool:
        return span[0] < other[1] and other[0] < span[1]

    profile_product_names = [
        str(item.get("name") or "").strip()
        for item in (canonical_profile.get("products") or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]

    def occurrence_context_product(text: str, position: int) -> str:
        """Return the nearest preceding product subject for one literal."""
        prefix = text[:max(0, position)]
        segments = re.split(r"(?<=[.!?])\s+|[\r\n]+", prefix)
        for segment in reversed(segments[-12:]):
            mentions = []
            for product_name in profile_product_names:
                mention_position = _product_context_position(
                    segment, product_name, canonical_profile
                )
                if mention_position is not None:
                    mentions.append((mention_position, product_name))
            if mentions:
                return max(mentions, key=lambda item: item[0])[1]
        return ""

    def occurrence_reason(
        text: str,
        span: tuple[int, int],
        canonical_value: str,
        safe_spans: list[tuple[int, int]],
    ) -> str | None:
        if any(overlap(span, safe_span) for safe_span in safe_spans):
            return "approved_safe_wording"

        prefix = text[max(0, span[0] - 80):span[0]]
        prefix = re.split(r"[.!?;]", prefix)[-1]
        if re.search(
            r"\b(?:not|never|no|neither|isn't|isnt|aren't|arent|wasn't|"
            r"wasnt|weren't|werent|doesn't|doesnt|don't|dont)\b"
            r"(?:[\W_]+\w+){0,4}[\W_]*$",
            prefix,
        ):
            return "negated"
        # Terms such as "lack of anti-theft features" deny the property even
        # though they do not use a standalone negative token immediately
        # before the conflicting literal.
        if re.search(
            r"\b(?:lack|lacks|lacking)\s+of\s+$",
            prefix,
        ):
            return "negated"


        # A forbidden term can be used safely to contrast the canonical claim
        # with a stronger claim (for example, "water-resistant rather than
        # fully waterproof"). Treat the contrast as a denial, not as an
        # assertion of the forbidden property.
        contrast_prefix = text[max(0, span[0] - 140):span[0]]
        contrast_prefix = re.split(r"[.!?;]", contrast_prefix)[-1]
        if re.search(
            r"\b(?:rather\s+than|instead\s+of|as\s+opposed\s+to)\s+"
            r"(?:fully\s+)?$",
            contrast_prefix,
        ):
            return "contrastive_negation"

        context = text[max(0, span[0] - 120):span[1] + 180]
        if re.search(
            r"\b(?:if|for)\s+(?:you|buyers?|people|users?|those)\s+"
            r"(?:who\s+)?(?:need|want|require)\b",
            context,
        ) and re.search(
            r"\b(?:look elsewhere|choose another|not (?:suitable|ideal|designed)|"
            r"avoid|lacks?|without)\b",
            context,
        ):
            return "conditional_exclusion"

        suffix = text[span[1]:span[1] + 180]
        question_mark = suffix.find("?")
        sentence_stops = [
            position
            for position in (suffix.find("."), suffix.find("!"))
            if position >= 0
        ]
        if question_mark >= 0 and (
            not sentence_stops or question_mark < min(sentence_stops)
        ):
            return "question"

        canonical_literal = normalized_literal(canonical_value)
        context = text[max(0, span[0] - 180):span[1] + 180]
        if (
            canonical_literal
            and canonical_literal in text
            and re.search(
                r"\b(?:but|however|although|whereas|rather than|instead)\b",
                context,
            )
            and re.search(
                r"\b(?:source|reviewer|review|manufacturer|reported|reports|"
                r"listed|lists|claimed|claims|according to)\b",
                context,
            )
        ):
            return "resolved_source_comparison"
        return None

    blocking = []
    warnings = []
    for product in canonical_profile.get("products", []):
        name = str(product.get("name") or "").strip()
        if not name:
            continue
        product_fragments = [
            fragment
            for fragment in text_fragments
            if name.casefold() in fragment.casefold()
        ]
        for attribute, fact in (product.get("facts") or {}).items():
            canonical_value = str(fact.get("canonical_value") or "").strip()
            safe_wording = str(fact.get("safe_wording") or "").strip()
            canonical_literal = normalized_literal(canonical_value)
            safe_literal = normalized_literal(safe_wording)

            # Model-provided conflicting_values occasionally contain a unit-expanded
            # canonical value (e.g. "680 g (24 oz)") or a compatible observation
            # explicitly approved by safe_wording (e.g. "sweaty" with limited airflow).
            # These are not mutually exclusive facts. forbidden_terms remain active
            # because their safety depends on local negation/conditional context.
            raw_conflicts = [
                str(value).strip()
                for value in (fact.get("conflicting_values") or [])
            ]
            true_conflicts = []
            for value in raw_conflicts:
                value_literal = normalized_literal(value)
                if not value_literal:
                    continue
                equivalent_expansion = bool(
                    canonical_literal
                    and (
                        value_literal in canonical_literal
                        or canonical_literal in value_literal
                    )
                )
                explicitly_approved = bool(
                    safe_literal and value_literal in safe_literal
                )
                if equivalent_expansion or explicitly_approved:
                    continue
                true_conflicts.append(value)

            conflicting_values = list(dict.fromkeys([
                *true_conflicts,
                *[
                    str(value).strip()
                    for value in (fact.get("forbidden_terms") or [])
                    if str(value).strip()
                ],
            ]))
            # A primary value may be unresolved while a related-model claim is
            # still known to be false or unsupported for this product. Audit
            # forbidden terms even when there is no replacement canonical value.
            if not conflicting_values:
                continue

            blocking_occurrences = []
            warning_occurrences = []
            for fragment in product_fragments:
                normalized_fragment = normalized_literal(fragment)
                safe_spans = literal_spans(normalized_fragment, safe_wording)
                for value in conflicting_values:
                    for span in literal_spans(normalized_fragment, value):
                        occurrence_owner = occurrence_context_product(
                            normalized_fragment, span[0]
                        )
                        if (
                            occurrence_owner
                            and canonical_product_identity_key(occurrence_owner)
                            != canonical_product_identity_key(name)
                        ):
                            continue
                        reason = occurrence_reason(
                            normalized_fragment,
                            span,
                            canonical_value,
                            safe_spans,
                        )
                        occurrence = {"value": value, "passage": fragment[:500]}
                        if reason:
                            occurrence["reason"] = reason
                            warning_occurrences.append(occurrence)
                        else:
                            blocking_occurrences.append(occurrence)

            def summarized_occurrences(occurrences: list[dict]) -> list[dict]:
                summarized = []
                seen = set()
                for occurrence in occurrences:
                    key = (
                        normalized_literal(occurrence["value"]),
                        occurrence["passage"],
                        occurrence.get("reason"),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    summarized.append(occurrence)
                return summarized[:8]

            if blocking_occurrences:
                blocking_values = list(dict.fromkeys(
                    occurrence["value"] for occurrence in blocking_occurrences
                ))
                blocking.append({
                    "product": name,
                    "attribute": attribute,
                    "canonical_value": canonical_value,
                    "values": [
                        value for value in [canonical_value, *blocking_values] if value
                    ],
                    "passages": summarized_occurrences(blocking_occurrences),
                })
            if warning_occurrences:
                warnings.append({
                    "product": name,
                    "attribute": attribute,
                    "canonical_value": canonical_value,
                    "values": list(dict.fromkeys(
                        occurrence["value"] for occurrence in warning_occurrences
                    )),
                    "passages": summarized_occurrences(warning_occurrences),
                })
    return blocking, warnings


def deterministically_remove_unsupported_durability_durations(html: str, conflicts: list[dict]) -> str:
    """Remove unsupported exact durability durations left by a repair model."""
    def normalized_literal(value: str) -> str:
        value = str(value or "").casefold()
        value = re.sub(r"(?<=\d)\s+(?=[a-z])", "", value)
        return re.sub(r"\s+", " ", value).strip()

    duration_conflicts = []
    for conflict in conflicts or []:
        if str(conflict.get("attribute") or "") != "durability_history":
            continue
        canonical_value = normalized_literal(str(conflict.get("canonical_value") or ""))
        for value in conflict.get("values") or []:
            literal = normalized_literal(str(value or ""))
            if literal and literal != canonical_value and re.fullmatch(r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*years?", literal):
                duration_conflicts.append(str(conflict.get("product") or "").strip())
    if not duration_conflicts:
        return html

    def replace_in_block(match: re.Match) -> str:
        block = match.group(0)
        if not any(product and product.casefold() in block.casefold() for product in duration_conflicts):
            return block
        block = re.sub(r"\bafter\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+years?(?:\s+of\s+(?:daily|regular)\s+(?:use|wear))?\b", "after extended use", block, flags=re.IGNORECASE)
        return re.sub(r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+years?(?:\s+of\s+(?:daily|regular)\s+(?:use|wear))?\b", "an extended period", block, flags=re.IGNORECASE)

    return re.sub(r"(?is)<(?:p|li|script)\b[^>]*>.*?</(?:p|li|script)>", replace_in_block, html or "")



def repair_unresolved_canonical_conflicts(
    html: str,
    conflicts: list[dict],
    canonical_profile: dict,
    keyword: str,
    recommended_product: str,
    cfg: dict,
) -> tuple[str, dict]:
    """Repair only audit-confirmed conflicts, then let the deterministic audit decide."""
    original = clean_generated_document_html(html)
    controls = cfg.get("canonical_facts") or {}
    report = {
        "enabled": bool(controls.get("auto_repair_conflicts", True)),
        "attempted": False,
        "applied": False,
        "reason": "",
        "initial_conflict_count": len(conflicts or []),
    }
    if not conflicts or controls.get("auto_repair_conflicts", True) is False:
        report["reason"] = "disabled_or_no_conflicts"
        return original, report

    report["attempted"] = True
    prompt = f"""
Repair factual conflicts in an HTML product review about "{keyword}".

Primary product: {recommended_product}

Hard requirements:
- Return ONLY the complete revised HTML fragment.
- Preserve every H2 and H3 exactly and in the same order.
- Change only sentences needed to resolve the listed audit conflicts.
- Treat the canonical profile as authoritative.
- Never transfer a specification, test result or user observation from another
  size, variant, generation or model to the primary product.
- If a disputed observation belongs to another model or cannot be tied to the
  exact product, remove it rather than weakening it with vague wording.
- Prefer canonical safe_wording when a sentence must be replaced.
- Preserve useful surrounding evidence and qualifications.
- Do not add facts, links, images, scripts, prices, headings, tables or lists.
- Remove every listed conflicting literal, including spelling variants, from visible HTML and JSON-LD FAQ answers.
- Do not retain a precise duration when the canonical safe wording says that duration is unverified.
""".strip()

    reference = (
        _canonical_profile_prompt_block(canonical_profile)
        + "\n\nAUDIT-CONFIRMED CONFLICTS:\n"
        + json.dumps(conflicts, ensure_ascii=False, indent=2)
        + "\n\nHTML ARTICLE TO REPAIR:\n"
        + original
    )
    try:
        raw = deepseek_generate(
            prompt,
            model=DEEPSEEK_MODEL,
            label="canonical_conflict_repair",
            max_tokens=int(controls.get("conflict_repair_max_tokens", 8000)),
            temperature=float(controls.get("conflict_repair_temperature", 0.0)),
            request_timeout=int(controls.get("conflict_repair_timeout", 180)),
            cache_prefix=reference,
        )
    except Exception as exc:
        report["reason"] = f"model_exception:{exc}"
        logging.exception("[CANONICAL_REPAIR] Model call failed: %s", exc)
        return original, report

    raw_path = str(cfg.get("_canonical_conflict_repair_response_path") or "").strip()
    if raw_path:
        try:
            with open(raw_path, "w", encoding="utf-8") as raw_file:
                raw_file.write(raw)
        except Exception as exc:
            logging.warning("[CANONICAL_REPAIR] Could not save raw response: %s", exc)

    candidate = clean_generated_document_html(raw)
    failures = []
    if _html_heading_signature(candidate, "h2") != _html_heading_signature(original, "h2"):
        failures.append("h2_signature_changed")
    if _html_heading_signature(candidate, "h3") != _html_heading_signature(original, "h3"):
        failures.append("h3_signature_changed")
    if recommended_product and recommended_product.casefold() not in candidate.casefold():
        failures.append("primary_product_missing")
    ratio = len(candidate) / max(len(original), 1)
    if not 0.80 <= ratio <= 1.05:
        failures.append(f"length_ratio_{ratio:.2f}")
    if len(re.findall(r"(?is)<table\b", candidate)) > len(re.findall(r"(?is)<table\b", original)):
        failures.append("table_count_increased")
    if len(re.findall(r"(?is)<script\b", candidate)) > len(re.findall(r"(?is)<script\b", original)):
        failures.append("script_count_increased")

    remaining, _warnings = audit_canonical_conflicts(candidate, canonical_profile)
    if len(remaining) >= len(conflicts):
        deterministic_candidate = deterministically_remove_unsupported_durability_durations(candidate, remaining)
        deterministic_remaining, _warnings = audit_canonical_conflicts(deterministic_candidate, canonical_profile)
        if len(deterministic_remaining) < len(remaining):
            candidate = deterministic_candidate
            remaining = deterministic_remaining
            report["deterministic_duration_cleanup"] = True
    report["remaining_conflict_count"] = len(remaining)
    if len(remaining) >= len(conflicts):
        failures.append("conflicts_not_reduced")

    if failures:
        report["reason"] = ",".join(failures)
        logging.warning("[CANONICAL_REPAIR] Rejected repair: %s", report["reason"])
        return original, report

    report.update({
        "applied": True,
        "reason": "accepted",
        "length_ratio": round(ratio, 3),
    })
    logging.info(
        "[CANONICAL_REPAIR] Applied repair; conflicts %d -> %d.",
        len(conflicts),
        len(remaining),
    )
    return candidate, report


def find_unresolved_canonical_conflicts(html: str, canonical_profile: dict) -> list[dict]:
    """Backward-compatible wrapper returning only affirmative conflicts."""
    blocking, _warnings = audit_canonical_conflicts(html, canonical_profile)
    return blocking

def get_recommended_product_name(keyword, country):
    safe_keyword_country = f"{keyword.replace(' ', '_')}_{country}"
    json_file = Path(f"output/{safe_keyword_country}/product_names_{country}.json")
    if json_file.exists():
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                names = json.load(f)
                if isinstance(names, dict) and "top_pick" in names:
                    return strip_product_placeholders(
                        smart_title_case(str(names["top_pick"]).strip().rstrip(" .,:;!?"))
                    )
                else:
                    logging.warning(f"Unexpected format or missing 'top_pick': {names}")
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {e}")
    return "one of the products reviewed"

def enforce_product_mention(prompt_func, heading, recommended_product, dataset=None):
    """
    Only enforce that the recommended product is mentioned if the dataset contains it.
    If the dataset does NOT contain the recommended product, return the first output
    without retrying/forcing a mention (prevents hallucinated recommendations).

    Returns HTML wrapped in <p> when appropriate.
    """
    # If we can't verify dataset support, be conservative: do NOT force.
    ds_has_reco = (
        bool(recommended_product)
        and bool(dataset)
        and (recommended_product.lower() in str(dataset).lower())
    )

    max_attempts = 3
    output = ""

    # If dataset doesn't support the product name, do NOT enforce mention.
    if not ds_has_reco:
        output = prompt_func()
        return _maybe_wrap_paragraph(output)

    # Dataset supports it â†’ enforce mention with retries.
    for attempt in range(max_attempts):
        output = prompt_func()
        if recommended_product.lower() in str(output).lower():
            return _maybe_wrap_paragraph(output)
        logging.warning(f"[{heading}] Attempt {attempt + 1}: Missing recommended product.")

    logging.error(f"[{heading}] Final attempt failed to include recommended product: {recommended_product}")
    return _maybe_wrap_paragraph(output)

def _strong_whitelist(wl: list[str] | None, *, brand_lexicon=None, cfg=None) -> list[str]:
    cfg = cfg or {}
    if not wl:
        return []
    out = []
    for w in wl:
        if not w:
            continue
        if _has_strong_product_cues(w, brand_lexicon=brand_lexicon, cfg=cfg) and looks_like_unique_product(w, brand_lexicon=brand_lexicon, cfg=cfg):
            out.append(w)
    return out



def find_primary_product(cleaned_text, keyword, product_whitelist=None, brand_lexicon=None, cfg=None, audit=None):
    cfg = cfg or {}
    rejection_reasons = []

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # âœ… HINTS: compute + log FIRST so it appears even if Amazon H1 returns
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    hint_names = [p for p in (product_whitelist or []) if p][:25]
    hint_block = "\n".join(f"- {p}" for p in hint_names) if hint_names else "(none)"

    if hint_names:
        logging.info(f"[TOP_PICK_HINTS] count={len(hint_names)} names={hint_names}")
    else:
        logging.info("[TOP_PICK_HINTS] count=0 names=[]")
        
    # âœ… Skip DeepSeek on company-review pages (Trustburn/Trustpilot-style)
    # These pages contain testimonials/headlines, not product model names.
    if looks_like_company_review_page(cleaned_text):
        logging.info("[SKIP_DEEPSEEK] reason=company_review_page")

        if audit is not None:
            audit.setdefault("deepseek", {})
            audit["deepseek"]["raw_response"] = ""
            audit["deepseek"]["cleaned_candidate"] = ""
            audit["deepseek"]["rejection_reasons"] = ["company_review_page"]
            audit["deepseek"]["accepted"] = False

            audit.setdefault("selection", {})
            audit["selection"]["source"] = "company_review_page_skip_gpt"

        # Return "" so the caller uses fallback/structured selection instead
        return ""



    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # âœ… Prefer Amazon PDP H1 titles if present (strongest â€œverbatimâ€ signal)
    # BUT: reject accessories/filters/packs using ONLY cfg (no hard-coded deny list)
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    try:
        amazon_titles = extract_amazon_dp_h1_titles(cleaned_text, cfg=cfg)

        deny_terms = [str(x).strip().lower() for x in (cfg.get("exclude_in_title_strict") or []) if str(x).strip()]
        deny_terms += [str(x).strip().lower() for x in (cfg.get("deny_browse_nodes") or []) if str(x).strip()]
        deny_terms = list(dict.fromkeys(deny_terms))

        include_keywords = [str(x).strip().lower() for x in (cfg.get("include_keywords") or []) if str(x).strip()]

        for t in amazon_titles:
            cand = strip_product_placeholders(t)
            cand = (cand or "").strip().lstrip(" \t\r\n-â€“â€”:â€¢|")
            if not cand:
                continue

            cand_l = cand.lower()

            if re.match(r"(?i)^(for|best\s+for)\b", cand):
                continue
            if looks_like_generic_headline(cand, brand_lexicon=brand_lexicon, cfg=cfg):
                continue

            if deny_terms and contains_any_complete_term(cand_l, deny_terms):
                logging.info(f"[AMZ_H1_REJECT] reason=deny_term title='{cand}'")
                continue

            if include_keywords and not any(k in cand_l for k in include_keywords):
                logging.info(f"[AMZ_H1_REJECT] reason=missing_include_keyword title='{cand}'")
                continue

            # âœ… NEW: never accept category-only titles from Amazon H1 (e.g. "Air Purifier")
            if is_category_only_product_name(cand, cfg=cfg):
                logging.info(f"[AMZ_H1_REJECT] reason=category_only title='{cand}'")
                continue

            if looks_like_product_title(cand, brand_lexicon=brand_lexicon, cfg=cfg):
                if audit is not None:
                    audit.setdefault("deepseek", {})
                    audit["deepseek"]["raw_response"] = ""
                    audit["deepseek"]["cleaned_candidate"] = cand
                    audit["deepseek"]["rejection_reasons"] = []
                    audit["deepseek"]["accepted"] = True
                    audit["deepseek"]["evidence_repaired"] = False

                    audit.setdefault("selection", {})
                    audit["selection"]["source"] = "amazon_pdp_h1"
                    audit["selection"]["verbatim_title"] = cand

                logging.info(f"Amazon PDP H1 selected top pick: {cand}")
                return cand


    except Exception:
        pass
    store_names = load_company_names()
    
    content_for_gpt = build_product_likely_slice(cleaned_text, brand_lexicon=brand_lexicon, max_chars=18000)



    prompt = f"""
You are extracting ONE exact product name from user review content for: "{keyword}".

We already extracted likely product names elsewhere (context only, do not invent):
{hint_block}

CRITICAL RULES:
- The product name MUST appear VERBATIM in the Content (same words, same order).
- Do NOT return testimonial headings or company-review headlines (e.g. "â€¦ exceeded my expectations", "My experience with â€¦").
- Do NOT return category-only labels from this configured list: {json.dumps((cfg.get("category_only_blocklist") or cfg.get("generic_tails") or [])[:20], ensure_ascii=False)}.
- If there is no clear product model in the Content, return: {{"product":"","evidence":""}}
- Return ONLY JSON (no extra text).
- Do NOT return section labels/descriptors like:
  "For Large Rooms", "Best for", "Top pick", "Final verdict", "Value for money", etc.
- The name must look like "Brand + Model" (at least 2 tokens).
- Also return the EXACT evidence line from the Content where the product appears.


OUTPUT FORMAT (STRICT JSON ONLY):
{{"product":"...","evidence":"..."}}

Content:
{content_for_gpt}
""".strip()


    raw = deepseek_generate(
        prompt,
        model=DEEPSEEK_PRO_MODEL,
        label="top_product_selection",
    ).strip()

    if audit is not None:
        audit.setdefault("deepseek", {})
        audit["deepseek"]["raw_response"] = raw
        audit["deepseek"]["prompt_excerpt"] = (prompt[:800] + "â€¦") if len(prompt) > 800 else prompt

    # Parse JSON (tolerant)
    obj = {}
    m = re.search(r"\{[\s\S]*?\}", raw)
    if m:
        try:
            obj = json.loads(m.group(0))
        except Exception:
            obj = {}

    result = (obj.get("product") or "").strip()
    evidence = (obj.get("evidence") or "").strip()
    # âœ… NEW: strip leaked heading markers like "H1 " from both fields
    result = strip_heading_marker_prefix(result)
    evidence = strip_heading_marker_prefix(evidence)
    
    # âœ… Prevent full sentences becoming "product names" (DO THIS EARLY)
    result = strip_sentence_tail(result).strip()

    # âœ… NEW: prevent full sentences/prefixes from becoming product names (BEFORE evidence repair)
    result = strip_sentence_tail(result)
    result = strip_leading_review_verbs(result)        # "tried/tested/reviewed ..."
    result = strip_leading_definite_article(result)    # "The Dyson ..." -> "Dyson ..."
    result = result.strip()


    # --- Fix 4: evidence repair (verbatim grounding) ---
    evidence_raw = evidence
    if result and evidence and not evidence_contains_product_verbatim(evidence, result):
        repaired = find_verbatim_evidence_for_product(cleaned_text or "", result, max_chars=240)
        if repaired and evidence_contains_product_verbatim(repaired, result):
            logging.info("ðŸ”§ DeepSeek evidence repaired using local content match")
            evidence = repaired
    # --- end Fix 4 ---

    result = unquote_plus(result).strip()
    result = strip_product_placeholders(result).strip()

    # Normalize / map to whitelist
    result = normalize_product_name(
        result,
        whitelist=product_whitelist or None,
        cfg=cfg,
        allow_strip_best_prefix=True,
        try_map_to_whitelist=True,
        max_words=12,
    )
    # âœ… MUST pass cfg so cfg-driven prefix stripping works ("Also Great:", etc.)
    result = strip_serp_editorial_wrappers(result, cfg=cfg).strip()

    # Remove editorial suffixes ("... Quality"/"... Review"/etc.)
    result = strip_editorial_suffixes(result).strip()

    # âœ… Final cleanup again AFTER mapping/stripping
    # Order matters: sentence tails can reveal verb/article prefixes after trimming.
    result = strip_sentence_tail(result).strip()
    result = strip_leading_review_verbs(result).strip()

    # Use the function name you actually have in your module:
    # - if you implemented strip_leading_definite_article(), call that
    # - if your module uses strip_leading_article(), keep this line as-is
    result = strip_leading_article(result, brand_lexicon=brand_lexicon, cfg=cfg).strip()  # "The Dyson ..." -> "Dyson ..."

    # cfg-driven award/section label stripping ("Also Great:", "Great Value:", etc.)
    result = strip_section_wrapper_prefix(result, cfg=cfg).strip()

    # One more tail trim after wrapper removal (sometimes wrapper removal exposes "is/with/..." splits)
    result = strip_sentence_tail(result).strip()

    result = smart_title_case(result).strip()

    if result and looks_like_marker_contaminated_product(result):
        rejection_reasons.append("marker_contaminated_product")

    if result and looks_like_reversed_category_brand(result, cfg=cfg):
        rejection_reasons.append("reversed_category_brand")

    if result and has_top_pick_deny_signal(result, cfg=cfg):
        rejection_reasons.append("top_pick_deny_term")

    # âœ… Reject testimonial/company-review headings even if verbatim
    if result and looks_like_testimonial_heading(result):
        rejection_reasons.append("testimonial_heading")

    # âœ… Reject category-only
    if result and is_category_only_product_name(result, cfg=cfg):
        rejection_reasons.append("category_only_title")

    # âœ… Require strong product cues for DeepSeek output too (prevents headlines passing)
    if result:
        grounded = False
        try:
            grounded = evidence_contains_product_verbatim(evidence, result) or evidence_matches_product_by_model_token(evidence, result)
        except Exception:
            grounded = False

        if (not grounded) and (not _has_strong_product_cues(result, brand_lexicon=brand_lexicon, cfg=cfg)):
            rejection_reasons.append("missing_strong_product_cues")





    # HARDENING A: strip leading punctuation/bullets/dashes
    result = (result or "").lstrip(" \t\r\n-â€“â€”:â€¢|")


    # Evidence checks (grounding) â€” robust + allows aligning to exact in-text product mention
    if evidence:
        ct = cleaned_text or ""

        ev_n = _normalize_ws(evidence).lower()
        ct_n = _normalize_ws(ct).lower()

        # Evidence must come from the content (or be repairable)
        if ev_n not in ct_n:
            repaired = find_verbatim_evidence_for_product(ct, result, max_chars=240) if result else None
            if repaired:
                evidence = repaired
                ev_n = _normalize_ws(evidence).lower()
                ct_n = _normalize_ws(ct).lower()

        if ev_n not in ct_n:
            rejection_reasons.append("evidence_not_in_content")
        else:
            # Product must appear verbatim in evidence (use the stronger helper)
            if result and not evidence_contains_product_verbatim(evidence, result):

                # âœ… NEW: First try a shorter identity form (trim generic category tails)
                # e.g. "AEG AX71-304GY Air Purifier" -> "AEG AX71-304GY Air Purifier" (unchanged)
                # or if you later add tail trimming, this prevents false rejects.
                trimmed_result = trim_to_category_tail(result, cfg=cfg) if (cfg and result) else result
                if trimmed_result and evidence_contains_product_verbatim(evidence, trimmed_result):
                    result = trimmed_result
                    if audit is not None:
                        audit.setdefault("deepseek", {})
                        audit["deepseek"]["tail_trimmed_for_evidence_match"] = True

                else:
                    # âœ… FIX A (existing): Amazon placeholder evidence ("<<PRODUCT_2>>") often omits brand
                    # but still includes distinctive model tokens.
                    if evidence_matches_product_by_model_token(evidence, result):
                        if audit is not None:
                            audit.setdefault("deepseek", {})
                            audit["deepseek"]["placeholder_model_grounding"] = True
                        # accept; do not append product_not_in_evidence

                    # âœ… NEW: Non-placeholder evidence (normal prose) â€” accept if model token overlaps
                    elif evidence_matches_product_by_model_token_any(evidence, result):
                        if audit is not None:
                            audit.setdefault("deepseek", {})
                            audit["deepseek"]["model_token_grounding"] = True
                        # accept; do not append product_not_in_evidence

                    else:
                        # Existing behavior: try to align result to what actually appears in evidence
                        aligned = None
                        aligned = find_best_product_phrase_in_line(
                            evidence,
                            brand_lexicon=brand_lexicon,
                            cfg=cfg,
                            max_words=10,
                        ) if result else None

                        if aligned and evidence_contains_product_verbatim(evidence, aligned):
                            aligned_clean = (aligned or "").strip().lstrip(" \t\r\n-â€“â€”:â€¢|")
                            if (
                                looks_like_unique_product(aligned_clean, brand_lexicon=brand_lexicon, cfg=cfg)
                                and _has_strong_product_cues(aligned_clean, brand_lexicon=brand_lexicon, cfg=cfg)
                                and not looks_like_generic_headline(aligned_clean, brand_lexicon=brand_lexicon, cfg=cfg)
                            ):
                                result = aligned_clean
                            else:
                                rejection_reasons.append("product_not_in_evidence")
                        else:
                            rejection_reasons.append("product_not_in_evidence")


    else:
        rejection_reasons.append("missing_evidence")



    # UI/CTA detector (avoid false positives for real product-looking titles)
    if result and _looks_like_ui_phrase(result):

        has_hints = bool(product_whitelist)  # TOP_PICK_HINTS / extracted names

        # âœ… Strong grounding override: if the result is grounded in evidence, never reject as UI/CTA
        grounded_in_evidence = False
        try:
            grounded_in_evidence = (
                evidence_contains_product_verbatim(evidence, result)
                or evidence_matches_product_by_model_token(evidence, result)
            )
        except Exception:
            grounded_in_evidence = False

        # âœ… Product-looking escape hatch: allow real product titles even if UI heuristic fires
        productish = False
        try:
            if looks_like_unique_product(result, brand_lexicon=brand_lexicon, cfg=cfg):
                productish = True
            elif looks_like_product_title(result, brand_lexicon=brand_lexicon, cfg=cfg):
                productish = True
        except Exception:
            productish = False

        if grounded_in_evidence or productish:
            pass  # keep it
        else:
            if has_hints:
                # With hints available, be stricter
                rejection_reasons.append("ui_or_cta_phrase")
            else:
                # Without hints, only reject if it's *clearly* UI/CTA
                if re.search(
                    r"(?i)\b("
                    r"general view|overview|table of contents|contents|"
                    r"top \d+|best\b|buy\b|shop\b|add to (cart|basket)|"
                    r"customers bought together|related products|you may also like|"
                    r"read more|see more|learn more|click here"
                    r")\b",
                    result.strip()
                ):
                    rejection_reasons.append("ui_or_cta_phrase")


    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # âœ… Strong whitelist enforcement (THIS is the requested change)
    # Only enforce overlap/whitelist if we have strong, product-like whitelist entries.
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _strong_whitelist(wl: list[str] | None, *, brand_lexicon=None, cfg=None) -> list[str]:
        cfg = cfg or {}
        if not wl:
            return []
        out = []
        for w in wl:
            if not w:
                continue
            if _has_strong_product_cues(w, brand_lexicon=brand_lexicon, cfg=cfg) and looks_like_unique_product(w, brand_lexicon=brand_lexicon, cfg=cfg):
                out.append(w)
        return out


    strong_wl = _strong_whitelist(product_whitelist, brand_lexicon=brand_lexicon, cfg=cfg)


    # Soft grounding: overlap only against strong_wl
    def _model_digit_tokens(s: str) -> set[str]:
        # tokens like "112", "35", "112uv" -> keep any token containing a digit
        return {t for t in _tokenize(s) if any(ch.isdigit() for ch in t)}


    # Soft grounding: overlap only against strong_wl (model-aware, but avoids false rejects)

    # âœ… Determine grounding NOW (do not rely on audit, because audit override is set later)
    grounded_in_evidence = False
    try:
        grounded_in_evidence = (
            evidence_contains_product_verbatim(evidence, result)
            or evidence_matches_product_by_model_token(evidence, result)
        )
    except Exception:
        grounded_in_evidence = False

    # If itâ€™s verbatim grounded, skip overlap enforcement entirely
    if strong_wl and result and not grounded_in_evidence:
        r_tok = set(_tokenize(result))
        r_digits = _model_digit_tokens(result)

        best_overlap = 0
        best_digit_overlap = 0
        whitelist_has_same_model_digits = False

        for h in strong_wl:
            h_tok = set(_tokenize(h))
            best_overlap = max(best_overlap, len(r_tok & h_tok))

            if r_digits:
                h_digits = _model_digit_tokens(h)
                if h_digits & r_digits:
                    whitelist_has_same_model_digits = True
                    best_digit_overlap = max(best_digit_overlap, len(h_digits & r_digits))

        if r_digits:
            if whitelist_has_same_model_digits and best_digit_overlap < 1:
                rejection_reasons.append("no_overlap_with_extracted_products")
        else:
            if best_overlap < 2:
                rejection_reasons.append("no_overlap_with_extracted_products")

    brand_lexicon = brand_lexicon or set()

    # Whitelist membership: enforce against a CLEANED version of the whitelist
    def _clean_wl_item(w: str) -> str:
        w = (w or "").strip()
        if not w:
            return ""

        # 1) Remove SERP/editorial wrappers early
        w = strip_serp_editorial_wrappers(w).strip()

        # 2) Drop obvious headline/test/guide entries BEFORE product cleaning
        # (These were poisoning whitelist/mapping: "The Smell Test: ...", "Can the Dyson ...", etc.)
        if re.search(r"(?i)\b("
                     r"smell\s+test|the\s+smell\s+test|"
                     r"can\s+the|should\s+you|your\s+next\s+read|"
                     r"release\s+date|price|key\s+specs|specs|"
                     r"power\s+consumption|is\s+it\s+worth|worth\s+the\s+price|"
                     r"how\s+i\s+tested|how\s+we\s+tested|testing|test|"
                     r"review\b|reviews\b|roundup|comparison|vs|versus|guide\b|buying\s+guide"
                     r")\b", w):
            return ""

        # 3) Colon titles are often headings ("Smell Test: ...", "Key Specs: ...")
        # Drop them unless the left side looks like an award label, OR the whole string
        # still looks like a product title after cleaning.
        if ":" in w:
            left = w.split(":", 1)[0].strip().lower()

            # allow obvious award labels
            if re.search(r"(?i)\b(best|top|our)\b", left):
                pass
            else:
                # if left is short/generic, treat as heading
                if left in {
                    "overview", "summary", "verdict", "conclusion", "key takeaways",
                    "key specs", "specs", "controls", "setup", "set-up", "set up",
                    "performance", "design", "value", "price"
                } or len(left.split()) <= 3:
                    return ""


        # 4) Now apply canonical product cleaning
        w = clean_product_name(w, cfg=cfg).strip()

        # 5) Strip wrapper/section prefixes like "Conclusion: ..." etc.
        w = strip_section_wrapper_prefix(w, cfg=cfg).strip()


        # 6) Final hygiene: kill UI phrases / generic headlines (belt-and-suspenders)
        if not w:
            return ""
        if _looks_like_ui_phrase(w):
            return ""
        if looks_like_generic_headline(w, brand_lexicon=brand_lexicon, cfg=cfg):
            return ""

        # 7) Normalize leading junk
        w = w.lstrip(" \t\r\n-â€“â€”:â€¢|").strip()
        
        if is_category_only_product_name(w, cfg=cfg):
            return ""

        return w


    clean_strong_wl = [cw for cw in (_clean_wl_item(w) for w in (strong_wl or [])) if cw]
    clean_strong_wl = prune_product_whitelist(clean_strong_wl, brand_lexicon=brand_lexicon, cfg=cfg)


    # Whitelist membership: enforce against a CLEANED version of the whitelist,
    # but accept "contained" matches (short name vs canonical long name).
    #
    # If DeepSeek pick is VERBATIM-grounded in evidence, do NOT fail just because
    # the whitelist is incomplete/noisy. This prevents good picks being replaced by fallback junk.
    if clean_strong_wl and result:
        res_digits = _model_digit_tokens(result)
        r_norm = _normalize_ws(result).lower()

        def _wl_match_ok(w: str) -> bool:
            w_norm = _normalize_ws(w).lower()
            # exact match
            if r_norm == w_norm:
                return True
            # allow contained match either direction (short vs long canonical)
            if r_norm in w_norm or w_norm in r_norm:
                return True
            return False

        wl_match = any(_wl_match_ok(w) for w in clean_strong_wl)

        # Strong grounding override: verbatim evidence (or placeholder+model grounding)
        grounded_in_evidence = False
        try:
            grounded_in_evidence = (
                evidence_contains_product_verbatim(evidence, result)
                or evidence_matches_product_by_model_token(evidence, result)
            )
        except Exception:
            grounded_in_evidence = False

        if not wl_match and not grounded_in_evidence:
            if res_digits:
                wl_has_same_model = any(_model_digit_tokens(w) & res_digits for w in clean_strong_wl)
                # Only enforce digit-overlap if whitelist actually contains same-model entries
                if wl_has_same_model:
                    rejection_reasons.append("not_in_whitelist")
            else:
                rejection_reasons.append("not_in_whitelist")
        elif (not wl_match) and grounded_in_evidence:
            # record for audit/debugging
            if audit is not None:
                audit.setdefault("deepseek", {})
                audit["deepseek"]["whitelist_override"] = "verbatim_evidence"

    # Hard reject obvious junk before identity / whitelist casing logic
    result = strip_url_and_retailer_chrome(result).strip()
    result = strip_heading_marker_prefix(result).strip()
    result = strip_sentence_tail(result).strip()
    
    if looks_like_blog_metadata(result):
        rejection_reasons.append("blog_metadata")

    if is_bad_top_pick_candidate(result, brand_lexicon=brand_lexicon, cfg=cfg):
        rescued = rescue_product_from_evidence_line(
            evidence,
            brand_lexicon=brand_lexicon,
            cfg=cfg,
            max_words=8,
        )
        if rescued:
            result = rescued
        else:
            rejection_reasons.append("bad_top_pick_candidate")

    # Exact whitelist match (case-insensitive) + preserve canonical casing
    mapped_exact = None
    if product_whitelist and result:
        for w in product_whitelist:
            if result.lower() == w.lower():
                mapped_exact = w
                break
        if mapped_exact:
            result = mapped_exact

    # Store-prefix cleanup after junk filtering, before identity/store checks
    result = strip_leading_store_name(result, store_names)
    result = strip_url_and_retailer_chrome(result).strip()

    # Identity gate
    passes_gate = looks_like_unique_product(result, brand_lexicon=brand_lexicon, cfg=cfg)
    if not (passes_gate or mapped_exact):
        rejection_reasons.append("failed_identity_gate")

    # Other rejects
    if not result:
        rejection_reasons.append("empty")
    if "?" in result:
        rejection_reasons.append("contains_question_mark")
    if re.search(r"\bhttps?://|\burl\s*:", result, re.I):
        rejection_reasons.append("contains_url")

    lower_result = result.lower()
    for store in store_names:
        if lower_result.startswith(store.lower() + " "):
            rejection_reasons.append(f"store_name_detected:{store}")
            break

    if audit is not None:
        audit["deepseek"]["cleaned_candidate"] = result
        audit["deepseek"]["rejection_reasons"] = rejection_reasons[:]
        audit["deepseek"]["raw_evidence"] = evidence_raw
        audit["deepseek"]["final_evidence"] = evidence
        audit["deepseek"]["evidence_repaired"] = (_normalize_ws(evidence_raw).lower() != _normalize_ws(evidence).lower())

    if (not result or len(result.split()) < 2) or rejection_reasons:
        logging.warning(f"DeepSeek product rejected; reasons={rejection_reasons or ['invalid_name']}")
        if audit is not None:
            audit["deepseek"]["accepted"] = False
        return "one of the products reviewed"

    if audit is not None:
        audit["deepseek"]["accepted"] = True

    logging.info(f"DeepSeek suggested top pick: {result}")
    return result


EMOJI_RE = re.compile(
    "["                               # broad emoji ranges
    "\U0001F300-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "\U0001F1E6-\U0001F1FF"           # flags
    "]+",
    flags=re.UNICODE
)

HTML_TAG_RE = re.compile(r"<[^>]+>")

CURRENCY_RE = re.compile(r"[$â‚¬Â£Â¥â‚¹]|(?:\bUSD|\bEUR|\bGBP|\bJPY|\bINR)\b", re.I)

def _clean_line(s: str) -> str:
    # Remove HTML tags/entities and emoji
    s = HTML_TAG_RE.sub("", s)
    s = unescape(s)
    s = EMOJI_RE.sub("", s)
    # Remove control chars
    s = re.sub(r"[\x00-\x1F\x7F]", " ", s)
    # Collapse whitespace and trim punctuation noise
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(" \t\r\n-â€“â€”:.")
    # Kill markdown/formatting
    s = re.sub(r"[*_#`~]", "", s)
    return s

def _word_count_ok(s: str) -> bool:
    words = re.findall(r"[A-Za-z0-9â€™']+", s)
    return 8 <= len(words) <= 14

def _mentions_price_or_currency(s: str) -> bool:
    return bool(CURRENCY_RE.search(s)) or bool(re.search(r"\b\d+\s*(?:bucks|dollars|pounds|euros|yen|rupees?)\b", s, re.I))

def _is_single_sentence(s: str) -> bool:
    # Consider overly long or multiple sentences as invalid
    return s.count(".") <= 1 and not any(x in s for x in [";", "â€¢", "â€“", "â€”", "â€¢", "Key Benefits", "http", "www"])

def _truncate_to_14_words(s: str) -> str:
    words = re.findall(r"\S+", s)
    return " ".join(words[:14]).strip(" .,:;â€“â€”")

def _fallback_line(recommended_product: str) -> str:
    # Category-neutral because the product name is already shown beside this line.
    return "Balances practical features, dependable performance, and everyday ease for most buyers."

def _passes_rules(s: str) -> bool:
    return (
        _is_single_sentence(s)
        and _word_count_ok(s)
        and not _mentions_price_or_currency(s)
    )

def generate_quick_verdict_tagline(
    dataset,
    keyword,
    recommended_product,
    style_guide,
    cfg,
    approved_article=None,
):
    """Derive one concrete tagline from verified benefits; omit unsafe fallbacks."""
    controls = cfg.get("quick_verdict") or {}
    if controls.get("derive_from_verified_benefits", True) is False:
        return ""

    canonical_profile = cfg.get("_canonical_profile") or {}
    canonical_block = _canonical_profile_prompt_block(canonical_profile)
    generic_rx = re.compile(
        r"(?i)\b(?:practical features|dependable performance|everyday ease|"
        r"great value|most buyers|excellent choice|strongest verified|"
        r"informed (?:everyday )?buying|this review|the article)\b"
    )

    def acceptable(raw: str) -> str:
        candidate = _clean_line(raw)
        if (
            _passes_rules(candidate)
            and not generic_rx.search(candidate)
            and recommended_product.casefold() not in candidate.casefold()
        ):
            return candidate
        return ""

    prompt = f"""
Write one concrete Quick Verdict tagline for "{recommended_product}".

Rules:
- Return only one sentence with 8-14 words.
- State two or three of the strongest verified benefits for the likely buyer.
- Include an intended user or use case and explain why one verified feature helps.
- Use only high- or medium-confidence facts from the matching product profile.
- Write a product verdict, never a description of the review or its methodology.
- Combine a concrete feature, its buyer benefit and an intended use.
- Do not mention price, Amazon, alternatives, ratings or drawbacks.
- Do not use generic phrases such as "practical features", "dependable performance",
  "everyday ease", "great value", "most buyers", "strongest verified",
  "informed buying" or "excellent choice".
- Do not use the product name, HTML, labels, bullets, a colon, semicolon or dash.

{canonical_block}

APPROVED ARTICLE:
{str(approved_article or "")[:12000]}
""".strip()
    try:
        raw = deepseek_generate(
            prompt,
            model=DEEPSEEK_MODEL,
            label="quick_verdict_verified",
            max_tokens=80,
            temperature=float(controls.get("temperature", 0.15)),
            cache_prefix=_dataset_cache_prefix(dataset),
        )
        candidate = acceptable(raw)
        if candidate:
            return candidate
    except Exception as exc:
        logging.warning("[QUICK_VERDICT] Verified tagline generation failed: %s", exc)

    if controls.get("retry_on_invalid", True):
        primary_facts = {}
        for product in canonical_profile.get("products", []):
            if canonical_product_identity_key(product.get("name", "")) == canonical_product_identity_key(recommended_product):
                primary_facts = {
                    key: value
                    for key, value in (product.get("facts") or {}).items()
                    if (value or {}).get("confidence") in {"high", "medium"}
                    and (value or {}).get("canonical_value")
                    and not (value or {}).get("requires_attribution")
                }
                break
        retry_prompt = f"""
Return exactly one 8-14 word product verdict and nothing else.
Use only these verified facts:
{json.dumps(primary_facts, ensure_ascii=False, indent=2)}
It must combine a concrete feature, buyer benefit and intended use.
Do not use the product name, a colon, semicolon, dash, price, methodology language,
generic praise or unsupported claims.
""".strip()
        try:
            raw = deepseek_generate(
                retry_prompt,
                model=DEEPSEEK_MODEL,
                label="quick_verdict_verified_retry",
                max_tokens=60,
                temperature=0.0,
                cache_prefix=_dataset_cache_prefix(dataset),
            )
            candidate = acceptable(raw)
            if candidate:
                return candidate
        except Exception as exc:
            logging.warning("[QUICK_VERDICT] Retry failed: %s", exc)

    logging.warning(
        "[QUICK_VERDICT] No verified buyer-focused tagline passed validation; omitting tagline."
    )
    return ""


def render_quick_verdict_html(recommended_product, verdict_line):
    safe_product_text = (recommended_product or "").strip()
    safe_line_text = _clean_line(unescape(verdict_line or ""))
    verdict_suffix = (
        ' <span class="dash"> &mdash; </span>' + escape(safe_line_text)
        if safe_line_text else ""
    )

    title_html = '<div class="quick-verdict-title">Quick Verdict</div>'

    if not safe_product_text:
        body = escape(safe_line_text) if safe_line_text else "Verified product summary unavailable."
        return (
            title_html
            + '<div class="quick-verdict-box" role="note" aria-label="Quick Verdict">'
            + '<strong>Quick Verdict:</strong> '
            + body
            + '</div>'
        )

    safe_product_attr = escape(safe_product_text, quote=True)
    box_html = (
        '<div class="quick-verdict-box" role="note" aria-label="Quick Verdict">'
        '&#127942; <strong>Best Overall:</strong> '
        f'<a class="product-link" data-product="{safe_product_attr}">{escape(safe_product_text)}</a>'
        f'{verdict_suffix}'
        '<div class="quick-verdict-cta">&#128293; '
        f'<a class="amazon-price-link" data-product="{safe_product_attr}">See Today&apos;s Price on Amazon &rarr;</a>'
        '</div>'
        '</div>'
    )
    return title_html + box_html

def normalize_quick_verdict_classes(html: str) -> str:
    if not html:
        return html
    soup = BeautifulSoup(html, "html.parser")
    mapping = {
        "quick-Verdict-box": "quick-verdict-box",
        "quick-Verdict-title": "quick-verdict-title",
        "quick-Verdict-cta": "quick-verdict-cta",
    }
    for el in soup.find_all(class_=True):
        classes = el.get("class", [])
        changed = False
        for i, c in enumerate(classes):
            if c in mapping:
                classes[i] = mapping[c]
                changed = True
        if changed:
            el["class"] = classes
    return str(soup)


def inject_quick_verdict_box(html, box_html, where="before_h2"):
    """
    Insert the Quick Verdict box:
      - "before_h2": immediately BEFORE the first <h2>
      - "top": at the very beginning of the document
    Removes any existing box first (idempotent).
    """
    if not html or not box_html:
        return html or ""

    # Remove any existing box
    html = re.sub(
        r'(?is)<div\s+class="quick-verdict-title"[^>]*>.*?</div>\s*'
        r'<div\s+class="quick-verdict-box"[^>]*>[\s\S]*?</div>\s*',
        '',
        html
    )


    if where == "top":
        return box_html + "\n" + html

    # Default: before first <h2>
    m = re.search(r'(?is)<h2\b[^>]*>[\s\S]*?</h2>', html)
    if m:
        return html[:m.start()] + box_html + "\n" + html[m.start():]

    # Fallback: top
    return box_html + "\n" + html


def inject_verdict_inline_into_first_h2(html, inline_html):
    """
    Insert a block-like inline verdict at the START of the first <h2>'s content,
    so it renders above the heading text but remains inside the H2 element.
    Idempotent: removes any existing .quick-verdict-inline first.
    """
    if not html or not inline_html:
        return html or ""

    # Remove any existing inline verdict to avoid duplicates
    html = re.sub(
        r'<span\s+class="quick-verdict-inline"[^>]*>[\s\S]*?</span>\s*',
        '',
        html,
        flags=re.IGNORECASE
    )

    m = re.search(r'(?is)(<h2\b[^>]*>)([\s\S]*?)(</h2>)', html)
    if not m:
        return html  # no H2 found; do nothing (keeps sanitizer happy)

    open_tag, inner, close_tag = m.groups()
    new_h2 = open_tag + inline_html + inner + close_tag
    return html[:m.start()] + new_h2 + html[m.end():]

def strip_section_wrapper_prefix(s: str, cfg: dict | None = None) -> str:
    """
    Strip prefixes like:
      "Also Great: ...", "Great Value â€” ...", "Verdict: ...", "Setup: ...", etc.
    Driven by cfg["section_label_left_sides"] via _build_section_wrapper_prefix_rx(cfg).
    """
    if not s:
        return s
    rx = _build_section_wrapper_prefix_rx(cfg)
    return rx.sub("", s).strip()





def normalize_name(s: str) -> str:
    s = unquote_plus(s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_amazon_dp_h1_titles(cleaned_text: str, cfg: dict | None = None) -> list[str]:
    """
    Extract Amazon PDP H1 titles from cleaned_text, but filter out obvious
    accessory/pack/replacement titles using cfg-driven rules.

    Returns a list of candidate titles (already normalized).
    """
    cfg = cfg or {}
    titles: list[str] = []
    lines = (cleaned_text or "").splitlines()
    current_url = None
    inside_amazon_dp = False

    # These patterns are local because the cleaned dataset uses explicit
    # "URL:" and "H1" line markers. The previous undefined names caused this
    # fast path to raise NameError and silently fall through to DeepSeek.
    URL_RX = re.compile(r"(?i)^\s*URL:\s*(https?://\S+)\s*$")
    AMAZON_DP_RX = re.compile(r"(?i)https?://(?:www\.)?amazon\.[^/\s]+/.+/(?:dp|gp/product)/")
    H1_RX = re.compile(r"(?i)^\s*(?:\[H1\]|H1)\s+(.+?)\s*$")

    # cfg-driven terms
    include_keywords = [str(x).strip().lower() for x in (cfg.get("include_keywords") or []) if str(x).strip()]
    deny_terms = [str(x).strip().lower() for x in (cfg.get("exclude_in_title_strict") or []) if str(x).strip()]

    # Optional: treat deny browse nodes as additional deny terms (often contains â€œfiltersâ€, â€œpartsâ€, etc.)
    deny_terms += [str(x).strip().lower() for x in (cfg.get("deny_browse_nodes") or []) if str(x).strip()]

    # Safe defaults (category-agnostic-ish, still relevant for most Amazon accessories)
    deny_terms += [
        "replacement", "replace", "refill", "cartridge", "compatible with",
        "accessory", "accessories", "spare", "spares", "kit",
        "filter", "filters", "pre-filter", "prefilter",
        "pack", "2-pack", "3-pack", "4-pack", "multipack", "multi-pack"
    ]
    deny_terms = list(dict.fromkeys(deny_terms))  # de-dupe preserve order

    def _looks_like_accessory_title(t: str) -> bool:
        tl = t.lower()
        if re.match(r"(?i)^\s*with\b", t):
            return True
        if re.search(r"(?i)\b(\d+\s*-\s*pack|\d+\s*pack|multi\s*pack|multipack|pack\s+of\s+\d+)\b", t):
            return True
        if contains_any_complete_term(tl, deny_terms):
            return True
        # If we have category include keywords, require at least one to appear
        # (this prevents accessory-only titles from being accepted)
        if include_keywords and not any(k in tl for k in include_keywords):
            return True
        return False

    for line in lines:
        m = URL_RX.match(line)
        if m:
            current_url = m.group(1)
            inside_amazon_dp = bool(AMAZON_DP_RX.search(current_url))
            continue

        if inside_amazon_dp:
            h1 = H1_RX.match(line)
            if h1:
                title = normalize_name(h1.group(1))
                title = trim_product_title_tail(title, max_comma_segments=2)

                # âœ… NEW: filter accessory/pack titles here
                if title and not _looks_like_accessory_title(title):
                    titles.append(title)

                inside_amazon_dp = False

    return titles



# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Top-pick audit helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _get_bad_exact(cfg: dict | None = None) -> set[str]:
    """
    Return a normalized set of exact phrases that are clearly NOT product names.
    cfg can optionally extend/override via:
      - cfg["bad_exact_headings"] (list/set of strings)
      - cfg["bad_exact"]          (list/set of strings)  [optional alias]
    """
    cfg = cfg or {}

    base = {
        "from the manufacturer",
        "from the brand",
        "product information",
        "product description",
        "product guidance & documents",
        "product guidance and documents",
        "about us",
        "customer reviews",
        "similar item to consider",
        "options available",
        "safety and product resources",
        "skip to",
        "similar brands on amazon",
        "customers also viewed these products",
        "different look, same great performance",
        "quick look",
        "key specifications",
        "key features",
        "specifications",
        "overview",
        "key terms",
    }

    extra = set()
    for k in ("bad_exact_headings", "bad_exact"):
        vals = cfg.get(k)
        if isinstance(vals, (list, tuple, set)):
            extra |= {str(x).strip().lower() for x in vals if str(x).strip()}

    return base | extra
    
def strip_leading_store_name(name: str, store_names) -> str:
    """
    Remove a leading retailer/store token when it is acting as a seller prefix,
    not part of the actual product identity.

    Examples:
        'Aldi Skylite 56cm Spinner Carry On' -> 'Skylite 56cm Spinner Carry On'
        "ALDI's Skylite 56cm Spinner Carry On" -> 'Skylite 56cm Spinner Carry On'
    """
    s = re.sub(r"\s+", " ", (name or "")).strip()
    if not s:
        return s

    for store in sorted((store_names or []), key=len, reverse=True):
        store = str(store).strip()
        if not store:
            continue

        if re.match(rf"(?i)^{re.escape(store)}(?:['â€™]s)?\s+", s):
            return re.sub(rf"(?i)^{re.escape(store)}(?:['â€™]s)?\s+", "", s).strip()

    return s    

def strip_leading_article(
    s: str,
    *,
    brand_lexicon: set[str] | None = None,
    cfg: dict | None = None
) -> str:
    """
    Cosmetic: remove a leading determiner ("The", optionally "A/An")
    when it is *very likely* just a sentence/article, not part of the product name.

    Examples:
      "The Dyson Hot+Cool HF1" -> "Dyson Hot+Cool HF1"
      "the levoit core 400s"   -> "levoit core 400s"
    """
    if not s:
        return s

    cfg = cfg or {}
    x = re.sub(r"\s+", " ", s).strip()

    m = re.match(r"(?i)^(the|a|an)\s+(.+)$", x)
    if not m:
        return x

    remainder = m.group(2).strip()
    if not remainder:
        return x

    # Look at the first token after the article
    toks = _WORD_RX_SIMPLE.findall(remainder)
    if not toks:
        return x
    first = toks[0].lower()

    # If we have a brand lexicon and it recognizes the next token as a brand -> strip
    if brand_lexicon:
        try:
            if first in {b.lower() for b in brand_lexicon}:
                return remainder
        except Exception:
            pass

    # Heuristic: don't strip if the "next token" is obviously generic
    generic_adj = {str(v).strip().lower() for v in (cfg.get("generic_adjectives") or []) if str(v).strip()}
    generic_noun = {str(v).strip().lower() for v in (cfg.get("generic_nouns") or []) if str(v).strip()}
    if first in generic_adj or first in generic_noun:
        return x

    # Heuristic: if remainder looks like a product (brand+model cues), it's safe to strip
    # Use your existing function if present; otherwise do a minimal check.
    try:
        if looks_like_unique_product(remainder, brand_lexicon=brand_lexicon, cfg=cfg):
            return remainder
    except Exception:
        pass

    # Minimal fallback: if there's a model token/digit later, assume it's producty
    if re.search(r"\d", remainder) or MODEL_TOKEN_RX.search(remainder):
        return remainder

    # Otherwise: leave it (avoid harming true names like "The Frame", "The One", etc.)
    return x




def _get_bad_suffix_rx(cfg: dict | None = None) -> re.Pattern:
    """
    Return a compiled regex that matches generic non-product suffixes.
    cfg can optionally provide:
      - cfg["bad_suffix_rx"] (string pattern)
    """
    cfg = cfg or {}
    pat = cfg.get("bad_suffix_rx")
    if isinstance(pat, str) and pat.strip():
        try:
            return re.compile(pat.strip(), re.I)
        except re.error:
            # fall through to default
            pass

    return re.compile(
        r"(?i)\b("
        r"price|room coverage|sound|power consumption|additional features|customer service|"
        r"quality|reviews?|alternatives|price comparison|deals?"
        r")\b\s*$"
    )

def is_obviously_not_a_product(
    name: str,
    *,
    brand_lexicon: set[str] | None = None,
    cfg: dict | None = None
) -> bool:
    """
    Fast, conservative filter for strings that are clearly NOT product names.
    Safe even if BAD_EXACT / BAD_SUFFIX_RX are not in scope.
    """
    n = normalize_name(name).lower()
    if not n:
        return True

    cfg = cfg or {}

    # exact headings from cfg (preferred), else globals, else empty
    bad_exact = cfg.get("bad_exact_headings")
    if not isinstance(bad_exact, (set, list, tuple)):
        bad_exact = globals().get("BAD_EXACT", set()) or set()
    bad_exact = {str(x).strip().lower() for x in bad_exact if str(x).strip()}

    # suffix regex from cfg string, else compiled global
    bad_suffix_rx = None
    cfg_rx = cfg.get("bad_suffix_rx")
    if isinstance(cfg_rx, str) and cfg_rx.strip():
        try:
            bad_suffix_rx = re.compile(cfg_rx, re.I)
        except Exception:
            bad_suffix_rx = None
    if bad_suffix_rx is None:
        bad_suffix_rx = globals().get("BAD_SUFFIX_RX", None)

    if n in bad_exact:
        return True

    if bad_suffix_rx is not None:
        try:
            if bad_suffix_rx.search(n):
                return True
        except Exception:
            pass

    if _looks_like_ui_phrase(name):
        return True

    if looks_like_generic_headline(name, brand_lexicon=brand_lexicon, cfg=cfg):
        return True

    return False






    
def looks_like_product_title(
    name: str,
    brand_lexicon: set[str] | None = None,
    cfg: dict | None = None
) -> bool:
    """
    Stricter-but-safe title detector for Amazon PDP H1 titles.

    Adds cfg-driven accessory filtering and (optionally) requires at least one
    include keyword (e.g., "air purifier") if include_keywords is configured.
    """
    cfg = cfg or {}
    s = normalize_name(name)
    if not s:
        return False

    # ðŸš« Reject promo/CTA headings like "Unlock 5% Savings"
    if _looks_like_ui_phrase(s):
        return False

    # hard reject: sentence-like / marketing copy / headings with punctuation
    if re.search(r"[.!?]", s):
        return False

    # tokenize conservatively (keeps hyphenated model tokens)
    tokens = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", s)
    if len(tokens) < 2:
        return False

    # reject obvious non-products (your existing rules)
    if is_obviously_not_a_product(s, brand_lexicon=brand_lexicon, cfg=cfg):
        return False

    lower = s.lower()

    # reject common heading-ish starters (very common in scraped Amazon blocks)
    if re.match(r"(?i)^(about|overview|highlights|features|specifications|specs|description|details|information|warranty|reviews?|questions|q&a|compare)\b", s):
        return False

    # reject if it contains these "section glue" phrases
    if any(x in lower for x in [
        "from the manufacturer", "from the brand", "product information", "customer reviews",
        "important information", "compare with similar items", "looking for specific info",
        "videos", "sponsored", "advertisement"
    ]):
        return False

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # âœ… NEW: accessory/pack/replacement rejects (cfg-driven + safe defaults)
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    deny_terms = [str(x).strip().lower() for x in (cfg.get("exclude_in_title_strict") or []) if str(x).strip()]
    deny_terms += [str(x).strip().lower() for x in (cfg.get("deny_browse_nodes") or []) if str(x).strip()]
    deny_terms += [
        "replacement", "replace", "refill", "cartridge", "compatible with",
        "accessory", "accessories", "spare", "spares", "kit",
        "filter", "filters", "pre-filter", "prefilter",
    ]
    deny_terms = list(dict.fromkeys(deny_terms))

    if re.match(r"(?i)^\s*with\b", s):
        return False
    if re.search(r"(?i)\b(\d+\s*-\s*pack|\d+\s*pack|multi\s*pack|multipack|pack\s+of\s+\d+)\b", s):
        return False
    if contains_any_complete_term(lower, deny_terms):
        return False

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # âœ… NEW: if include_keywords is configured, require at least one
    # This prevents accessory-only titles from winning the Amazon H1 fast-path.
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    include_keywords = [str(x).strip().lower() for x in (cfg.get("include_keywords") or []) if str(x).strip()]
    if include_keywords and not any(k in lower for k in include_keywords):
        return False

    # ---- Strong product cues ----
    has_digit = bool(re.search(r"\d", s))
    has_alnum_model = bool(re.search(r"\b[A-Za-z]+\d+[A-Za-z0-9]*\b|\b\d+[A-Za-z]+\b", s))
    has_model_hyphen = bool(re.search(r"\b[A-Za-z0-9]+-\d+[A-Za-z0-9-]*\b", s)) or (s.count("-") >= 2)
    has_acronym = bool(re.search(r"\b[A-Z]{2,}\b", s))

    strong_cue = has_digit or has_alnum_model or has_model_hyphen or has_acronym

    # ---- Brand + title-like structure fallback (stricter) ----
    has_brand = False
    if brand_lexicon:
        first = tokens[0].lower()
        has_brand = first in {b.lower() for b in brand_lexicon}

    if len(tokens) > 12:
        return False

    glue_words = {"how", "why", "what", "when", "where", "which", "your", "best", "top", "guide", "tips"}
    if any(t.lower() in glue_words for t in tokens[:3]):
        return False

    titlecase_like = sum(
        1 for t in tokens[:6]
        if re.match(r"^[A-Z][A-Za-z0-9]+$", t) and not t.isupper()
    ) >= 2

    fallback_ok = has_brand and titlecase_like
    return bool(strong_cue or fallback_ok)




def _now_iso():
    try:
        import datetime as _dt
        return _dt.datetime.now(_dt.UTC).isoformat().replace("+00:00", "Z")
    except Exception:
        return ""


def init_top_pick_audit(keyword, country):
    return {
        "timestamp": _now_iso(),
        "keyword": keyword,
        "country": country,
        "method": None,                      # "deepseek" or "fallback"
        "deepseek": {
            "prompt_excerpt": "",
            "raw_response": "",
            "cleaned_candidate": "",
            "truncated_to_8_words": False,
            "rejection_reasons": [],        # e.g., ["empty","contains_question_mark","store_name_detected"]
            "accepted": False
        },
        "fallback": {
            "candidates": [],               # [{product, pros_count, cons_count, score, cleaned_name, valid}]
            "winner": ""
        },
        "canonicalization": {
            "final_name_before_titlecase": "",
            "final_name": ""
        },
        "enforcement": {
            "purpose_contains_final_name": None,
            "abandoned_due_to_missing_purpose_mention": None
        },
        "files": {
            "saved_top_pick_json": "",
            "audit_json": ""
        }
    }

def save_top_pick_audit(audit, output_dir, country):
    try:
        path = os.path.join(output_dir, f"top_pick_decision_{country}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2, ensure_ascii=False)
        logging.info(f"ðŸ§¾ Top-pick audit saved: {path}")
        audit["files"]["audit_json"] = path
    except Exception as e:
        logging.error(f"Failed to write top-pick audit: {e}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Canonicalisation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def canonicalize_structured_products(structured_products):
    result = []
    by_identity = {}
    for entry in structured_products:
        entry = dict(entry)
        pname = strip_editorial_product_name_suffix(
            str(entry.get("product", "")).strip().rstrip(" .,:;!?")
        )
        if not pname or looks_like_legal_entity_name(pname):
            continue
        pname = smart_title_case(pname)
        entry["product"] = pname
        identity = canonical_product_identity_key(pname)
        existing = by_identity.get(identity)
        if existing is not None:
            for field, value in entry.items():
                if field == "product" or value in (None, "", [], {}):
                    continue
                if isinstance(value, list):
                    target = existing.setdefault(field, [])
                    if isinstance(target, list):
                        for item in value:
                            if item not in target:
                                target.append(item)
                elif not existing.get(field):
                    existing[field] = value
            continue
        by_identity[identity] = entry
        result.append(entry)
    return result

def build_canonical_product_map(structured_products, recommended_product):
    canonical_map = {}
    for entry in structured_products:
        pname = str(entry.get("product", "")).strip().rstrip(" .,:;!?")
        if not pname:
            continue
        canonical = smart_title_case(pname)
        canonical_map[pname.lower()] = canonical
    if recommended_product:
        rp = recommended_product.strip().rstrip(" .,:;!?")
        canonical_map[rp.lower()] = smart_title_case(rp)
    return canonical_map



_SCRIPT_BLOCK_RX = re.compile(r"(?is)<script\b[^>]*>.*?</script>")


def normalize_product_mentions(text: str, canonical_map: dict[str, str]) -> str:
    if not text or not canonical_map:
        return text

    nonce = secrets.token_hex(8)
    kept_scripts = []

    def _shield_script(m: re.Match) -> str:
        kept_scripts.append(m.group(0))
        return f"__SCRIPT_BLOCK_{nonce}_{len(kept_scripts)-1}__"

    shielded = _SCRIPT_BLOCK_RX.sub(_shield_script, text)

    items = sorted(
        ((k, v) for k, v in canonical_map.items() if (k or "").strip()),
        key=lambda kv: len(kv[0]),
        reverse=True
    )

    out = shielded
    for lower_name, canonical in items:
        pattern = re.compile(rf"(?<!\w){re.escape(lower_name)}(?!\w)", re.IGNORECASE)
        out = pattern.sub(lambda m, c=canonical: c, out)
    for i, block in enumerate(kept_scripts):
        out = out.replace(f"__SCRIPT_BLOCK_{nonce}_{i}__", block)

    return out

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Prompt generators (category-agnostic)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _full_name_instruction():
    return ("Always refer to products by their **full product names** "
            "(e.g., 'Brand Model X10 Product Type') instead of brand names "
            "or partial names. This ensures clarity and precision.")

def generate_introduction(heading, dataset, style_guide, keyword, country, recommended_product, cfg, product_whitelist=None):
    def prompt_func():
        prompt = f"""
        You are writing the introduction to a product review article in the {keyword} category.

        The section titled '{heading}' should explain the purpose of the review: to help readers choose the right option(s) based on performance, value, features, and user experience.

        Hard requirements (no exceptions):
        - 3â€“4 sentences total. Short, active sentences (<20 words).
        - Use transitions only when they improve the logical flow; do not apply a quota.
        - 5th-grade reading level.
        - Open with a strong hook that highlights a common consumer frustration or need (e.g., wasted money, poor durability, difficult usability) and immediately suggest how this type of product solves it.
         - Include emotional hooks to reduce anxiety and create relief. 
         - Lead with a benefit, not just the problem. Example: â€œTired of gadgets that die halfway through the day? The right device could be your daily lifesaver.â€
        - Scope must match the dataset: if only one product exists, talk only about that product; if multiple exist, mention that the review compares them.
        - Make it scannable and practical for online shoppers.
        - No exact prices.
        - Do NOT output any lists, tables, or empty HTML.
        - End by clearly recommending only this product: "{recommended_product}" â€” using its full product name, and no other.

        If the dataset does not contain "{recommended_product}" or has no usable fields, write a single 2-sentence neutral intro and STOP (no recommendation, no lists, no tables).

        {_full_name_instruction()}

        {_canonical_profile_prompt_block(cfg.get("_canonical_profile"))}

        The reference dataset is supplied separately.

        Style Guide:
        {style_guide}

        """
        raw_output = deepseek_generate(
            prompt,
            label=f"introduction:{heading}",
            cache_prefix=_dataset_cache_prefix(dataset),
        )

        # â¬‡ï¸ ADD THESE LINES HERE
        cleaned_output = clean_generated_text(raw_output, heading, cfg).strip()
        cleaned_output = clean_generated_document_html(cleaned_output)
        cleaned_output = enforce_list_table_limits(cleaned_output, max_lists=1, max_tables=0)
        cleaned_output = drop_empty_lists_and_fix_cells(cleaned_output)
        cleaned_output = remove_table_claims_without_table(cleaned_output)
        cleaned_output = strip_best_prefixes(cleaned_output, whitelist=product_whitelist)

        # â¬†ï¸ END INSERT

        return cleaned_output
    return enforce_product_mention(prompt_func, heading, recommended_product, dataset=dataset)



def generate_overview(heading, dataset, style_guide, keyword, country, recommended_product, cfg, product_whitelist=None, label="overview", max_tokens=120):
    wl_text = whitelist_note(product_whitelist or [])
    canonical_block = _canonical_profile_prompt_block(cfg.get("_canonical_profile"))
    prompt = f"""
    Write exactly one product-specific verdict sentence for the main section titled: '{heading}'.

    {canonical_block}

    Hard requirements:
    - Begin with the full product name: "{recommended_product}".
    - State one concrete finding supported by the dataset.
    - Use 12â€“30 words in one sentence.
    - Do not introduce the general subject or explain why the feature matters.
    - Do not preview what the section will discuss.
    - Do not address the reader.
    - Do not use emotional hooks, rhetorical questions, transitions, lists, tables, buying advice, or generic category statements.
    - Do not repeat the heading in different words.
    - Do not mention exact prices or apply bold or italic formatting.
    - If no relevant product-specific evidence exists, return an empty string.

    {wl_text}

    Supporting section evidence:
    {dataset}

    Style Guide:
    {style_guide}

    """
    raw_output = deepseek_generate(prompt, label=label, max_tokens=max_tokens, temperature=0.35)

    cleaned_output = clean_generated_text(raw_output, heading, cfg).strip()
    cleaned_output = enforce_list_table_limits(cleaned_output, max_lists=0, max_tables=0)
    cleaned_output = drop_empty_lists_and_fix_cells(cleaned_output)
    cleaned_output = remove_table_claims_without_table(cleaned_output)
    cleaned_output = strip_best_prefixes(cleaned_output, whitelist=product_whitelist)

    # A broad H2 is only a container for its detailed H3 sections. If the
    # model ignores the verdict format, omit the bridge rather than publish a
    # generic scene-setting paragraph.
    overview_text = unescape(re.sub(r"<[^>]+>", " ", cleaned_output or ""))
    overview_text = re.sub(r"\s+", " ", overview_text).strip()
    product_name = re.sub(r"\s+", " ", recommended_product or "").strip()
    if not overview_text or not product_name:
        return ""
    if not overview_text.casefold().startswith(product_name.casefold()):
        logging.warning("[OVERVIEW] Dropped non-product-specific H2 overview for %r", heading)
        return ""
    if len(re.findall(r"[.!?](?:\s|$)", overview_text)) != 1:
        logging.warning("[OVERVIEW] Dropped multi-sentence H2 overview for %r", heading)
        return ""
    word_count = len(re.findall(r"\b[\w'-]+\b", overview_text))
    if not 12 <= word_count <= 30:
        logging.warning(
            "[OVERVIEW] Dropped H2 overview outside 12-30 words for %r: %d",
            heading,
            word_count,
        )
        return ""

    return _maybe_wrap_paragraph(cleaned_output)


def generate_detailed_subheading_content(
    heading,
    dataset,
    style_guide,
    keyword,
    recommended_product,
    cfg,
    product_whitelist=None,
    label="detail",
    max_tokens=900
):
    heading = (heading or "").strip()
    wl_text = whitelist_note(product_whitelist or [])
    canonical_block = _canonical_profile_prompt_block(cfg.get("_canonical_profile"))
    memory_block = _article_memory_block(cfg)
    controls = cfg.get("editorial_controls") or {}
    comparison_mode = _heading_requests_comparison(heading, cfg)

    if comparison_mode:
        comparison_rules = """
- This is a dedicated comparison section. Compare only material buying differences supported by evidence.
- If 2 or more products are discussed and a table adds new information, include at most ONE comparison table.
- A table is optional; do not repeat a table or model-selection conclusion already covered earlier.
"""
    else:
        comparison_rules = """
- This is a feature-focused section, not a comparison hub.
- Focus on the recommended product. Mention at most one alternative only when it adds a new, directly relevant buying distinction.
- Do not end with broad "choose X for...; choose Y for..." guidance.
- Do not include a comparison table in this section.
"""

    prompt = f"""
You are a skilled product review writer. Give shoppers specific, evidence-based analysis for the subheading "{heading}".

{canonical_block}

{memory_block}

{wl_text}

Rules:
- Write only about products present in the evidence. Do not invent specifications, models or experiences.
- The canonical product profile overrides conflicting raw evidence.
- Copy canonical safe_wording whenever a fact needs qualification.
- Company names, publishers, retailers and section headings are not products.
- Keep paragraphs to 2-4 short sentences and keep total words minimal.
- Discuss no more than 3 products.
- Include "{recommended_product}" when relevant to the heading.
- Do not use exact prices, bold keywords or generic scene-setting introductions.
- Do not say a manufacturer "listened to feedback", "improved" something, or that users "praised" it unless the concrete supported change or outcome is stated.
- Do not mechanically repeat transitions such as Therefore, However, Also or In short.
{comparison_rules}

Lists:
- Include at most two short lists, only when a list improves scanning.
- Never emit empty lists or unsupported Pros/Cons labels.

Table format, only when comparison mode explicitly permits one:
<div class="comparison-table-wrap">
<table class="comparison-table">
<thead><tr><!-- 3-6 short Title Case headers --></tr></thead>
<tbody><!-- 2-4 complete product rows --></tbody>
</table>
</div>
- Use canonical values. Use "â€”" for a genuinely unavailable cell.
- Never leave a cell empty, use inline styles, or nest lists in cells.

Quality gates:
- No empty tags, stray closing tags, unsupported superlatives or vague feedback claims.
- Do not repeat a buying conclusion found in ALREADY COVERED content.
- Give this subsection a distinct job from the adjacent subsection: do not reuse the same threshold, drawback, user example or recommendation unless the current heading cannot be answered without it.
- Avoid unsupported comparative superiority and absolute recommendations. State the supported benefit and intended use instead.

{_full_name_instruction()}

Section evidence:
{dataset}

Style Guide:
{style_guide}
""".strip()

    raw_output = deepseek_generate(
        prompt,
        label=label,
        max_tokens=max_tokens,
        temperature=float(controls.get("detail_temperature", 0.45)),
    )

    html_tables = len(re.findall(r"(?is)<table\\b", raw_output))
    md_tables = len(_MD_TABLE_BLOCK.findall(raw_output))
    logging.info(
        "[TABLE_DEBUG] %s :: raw_output | comparison_mode=%s | html_tables=%d | md_tables=%d | chars=%d",
        heading,
        comparison_mode,
        html_tables,
        md_tables,
        len(raw_output),
    )

    cleaned_output = clean_generated_text(raw_output, heading, cfg).strip()
    cleaned_output = clean_generated_document_html(cleaned_output)
    before = _count_tables(cleaned_output)
    cleaned_output = fix_and_normalize_tables(cleaned_output)
    after = _count_tables(cleaned_output)
    logging.info("[%s] Table normalize: %d -> %d", heading, before, after)

    max_lists = max(0, int(controls.get("max_lists_per_section", 2)))
    cleaned_output = enforce_list_table_limits(
        cleaned_output,
        max_lists=max_lists,
        max_tables=1 if comparison_mode else 0,
    )
    cleaned_output = drop_empty_lists_and_fix_cells(cleaned_output)
    cleaned_output = remove_table_claims_without_table(cleaned_output)
    cleaned_output = strip_best_prefixes(cleaned_output, whitelist=product_whitelist)
    return _maybe_wrap_paragraph(cleaned_output)


def generate_value_for_money_verdict(
    heading,
    dataset,
    style_guide,
    keyword,
    country,
    recommended_product,
    cfg,
    product_whitelist=None,
    label="verdict",
    max_tokens=800,
):
    """
    Verdict section should never include tables or lists.

    IMPORTANT behavior:
    - If the dataset includes the recommended product name, we must recommend it.
    - If the dataset does NOT include the recommended product name, write a neutral summary
      with NO product recommendation and DO NOT try to force-mention it.
    """
    import re
    from urllib.parse import unquote_plus

    wl_text = whitelist_note(product_whitelist or [])

    def _norm_presence(s: str) -> str:
        """
        Normalize strings for presence checks:
        - Decode plus-encoding
        - Treat '+' as spaces
        - Lowercase
        - Remove non-alphanumerics
        """
        s = unquote_plus(str(s or ""))
        s = s.replace("+", " ")
        s = s.lower()
        s = re.sub(r"[^a-z0-9]+", "", s)
        return s

    prompt = f"""
You are writing the final value-for-money verdict for a product review about "{keyword}".

Section title: "{heading}"

Hard rules (no exceptions):
- Output ONLY 2 short paragraphs (2â€“4 sentences each).
- Active voice, 5th-grade reading.
- Use transitions naturally; do not apply a quota or repeat the same opener.
- NO bullet lists. NO numbered lists. NO tables. NO HTML lists (<ul>/<ol>) and NO <table>.
- No exact prices; use general price language.
- No bold/italic keywords.

Product rules:
- If the dataset includes "{recommended_product}", recommend exactly: "{recommended_product}" (full name) and nothing else.
- If the dataset does NOT include "{recommended_product}", write a neutral summary with NO product recommendation.
- Do NOT say "this option", "this product", or "the unit" without naming the product (when recommending).

{_canonical_profile_prompt_block(cfg.get("_canonical_profile"))}

{wl_text}

{_full_name_instruction()}

The reference dataset is supplied separately.

Style Guide:
{style_guide}
""".strip()

    raw_output = deepseek_generate(
        prompt,
        label=label,
        max_tokens=max_tokens,
        temperature=0.6,
        cache_prefix=_dataset_cache_prefix(dataset),
    )

    cleaned_output = clean_generated_text(raw_output, heading, cfg).strip()
    cleaned_output = clean_generated_document_html(cleaned_output)

    # Strip any lists/tables aggressively (verdict must be plain paragraphs)
    cleaned_output = re.sub(r"(?is)<table\b[^>]*>.*?</table>", "", cleaned_output)
    cleaned_output = re.sub(r"(?is)<ul\b[^>]*>.*?</ul>", "", cleaned_output)
    cleaned_output = re.sub(r"(?is)<ol\b[^>]*>.*?</ol>", "", cleaned_output)

    # Remove stray Pros/Cons label paragraphs left behind, fix empty <td> if any leaked
    cleaned_output = drop_empty_lists_and_fix_cells(cleaned_output)

    # If it still claims a table exists, soften wording
    cleaned_output = remove_table_claims_without_table(cleaned_output)

    cleaned_output = strip_best_prefixes(cleaned_output, whitelist=product_whitelist)

    # --- IMPORTANT: do NOT enforce product mention if dataset doesn't contain it ---
    ds_has_reco = False
    if recommended_product:
        ds_has_reco = _norm_presence(recommended_product) in _norm_presence(dataset or "")

    # If you have a whitelist, it can be an additional strong signal the reco is "real"
    if (not ds_has_reco) and recommended_product and product_whitelist:
        ds_has_reco = any(
            _norm_presence(recommended_product) == _norm_presence(p) for p in (product_whitelist or [])
        )

    if ds_has_reco:
        # If dataset supports it, ensure the recommendation appears in the verdict.
        # Re-ask once if missing; then deterministically guarantee mention.
        if _norm_presence(recommended_product) not in _norm_presence(cleaned_output):
            retry_prompt = prompt + "\n\nReminder: You MUST explicitly recommend the exact product name once."
            retry_raw = deepseek_generate(
                retry_prompt,
                label=f"{label}-retry",
                max_tokens=max_tokens,
                temperature=0.5,
                cache_prefix=_dataset_cache_prefix(dataset),
            )
            cleaned_output = clean_generated_text(retry_raw, heading, cfg).strip()
            cleaned_output = re.sub(r"(?is)<table\b[^>]*>.*?</table>", "", cleaned_output)
            cleaned_output = re.sub(r"(?is)<ul\b[^>]*>.*?</ul>", "", cleaned_output)
            cleaned_output = re.sub(r"(?is)<ol\b[^>]*>.*?</ol>", "", cleaned_output)
            cleaned_output = drop_empty_lists_and_fix_cells(cleaned_output)
            cleaned_output = remove_table_claims_without_table(cleaned_output)
            cleaned_output = strip_best_prefixes(cleaned_output, whitelist=product_whitelist)

        # Deterministic guarantee (prevents "this option" / generic anchor text issues)
        if recommended_product and _norm_presence(recommended_product) not in _norm_presence(cleaned_output):
            cleaned_output = (
                f"Therefore, {recommended_product} is the best value pick for most people.\n\n"
                + cleaned_output.strip()
            )

        return _maybe_wrap_paragraph(cleaned_output)

    # Dataset doesn't contain the recommended product -> neutral summary, no forced mention
    # If the model still mentioned the product anyway, remove it defensively.
    if recommended_product:
        cleaned_output = re.sub(
            rf"(?i)\b{re.escape(recommended_product)}\b",
            "this option",
            cleaned_output,
        )

    return _maybe_wrap_paragraph(cleaned_output)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# FAQ generation (new)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _json_from_text_block(text):
    """
    Extract the first JSON array/object from a text blob (tolerant of code fences).
    """
    # Try fenced blocks first
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
        return json.loads(candidate)

    # Fallback: find first { ... } or [ ... ]
    start = min([i for i in [text.find("{"), text.find("[")] if i != -1], default=-1)
    if start != -1:
        # crude but robust: walk until JSON parses or we run out
        for end in range(len(text), start + 1, -1):
            chunk = text[start:end].strip()
            try:
                return json.loads(chunk)
            except Exception:
                continue
    raise ValueError("No valid JSON found in model output.")

def generate_faq_json(
    dataset,
    keyword,
    recommended_product,
    style_guide,
    cfg,
    max_q=10,
    approved_article=None,
):
    """Generate pre-purchase FAQs from approved article facts, not raw conflicts."""
    canonical_block = _canonical_profile_prompt_block(cfg.get("_canonical_profile"))
    approved_source = approved_article or dataset
    prompt = f"""
You are writing a Pre-Purchase FAQ for a product review about "{keyword}".

{canonical_block}

FOCUS:
- Include only questions a shopper asks before buying.
- Cover suitability, fit, dimensions, specifications, compatibility, materials, durability, included features, model differences and value positioning when supported.

EXCLUDE COMPLETELY:
- Troubleshooting, setup, installation, pairing, firmware, cleaning, maintenance, repair, replacement parts and customer-service procedures.

OUTPUT RULES:
- Return STRICT JSON ONLY: an array of objects with keys "q" and "a".
- Return 6-10 Q&As with 2-4 short sentences per answer.
- Use only facts already present in the canonical profile or approved article.
- The canonical profile overrides any conflicting wording in the approved article.
- Preserve every qualification in canonical safe_wording.
- Do not introduce a new specification, compatibility promise or performance claim.
- Do not use exact prices, bullets or HTML.
- Use full product names when needed.
- If a hard specification is not high-confidence for the exact reviewed model, OMIT that question entirely; do not publish an estimate, inferred value or related-model measurement.
- For nominal compatibility or fit, reconcile varying reports by explaining that physical dimensions, cases, variants or conditions can change the result.
- If a non-specification fact is uncertain, say so briefly instead of selecting the strongest interpretation.

Style Guide:
{style_guide}

Recommended product:
{recommended_product}
""".strip()

    reference = (
        canonical_block
        + "\n\nAPPROVED ARTICLE CONTENT:\n"
        + str(approved_source or "")
    )
    raw = deepseek_generate(
        prompt,
        model=DEEPSEEK_MODEL,
        label="faq",
        max_tokens=1800,
        temperature=0.2,
        cache_prefix=reference,
    )

    try:
        parsed = _json_from_text_block(raw)
        if not isinstance(parsed, list):
            raise ValueError("FAQ JSON must be a list.")
    except Exception:
        logging.error("FAQ parse failed; falling back to empty FAQ.")
        parsed = []

    non_prepurchase_patterns = [
        r"\bhow\s+to\b", r"\bset\s*up\b", r"\binstall(?:ation)?\b",
        r"\bupdate(?:s|d)?\b", r"\bfirmware\b", r"\breset\b",
        r"\btroubleshoot(?:ing)?\b", r"\brepair\b", r"\bfix\b",
        r"\bclean(?:ing)?\b", r"\bmaintenance\b", r"\bpair(?:ing)?\b",
        r"\b(?:connect|sync)\b", r"\berror\s+code\b",
        r"\breplace(?:ment)?\s+(?:part|wheel|zipper|battery)\b",
        r"\bcustomer\s+service\b", r"\bsupport\b",
    ]

    def is_prepurchase(question, answer):
        text = f"{question} {answer}".lower()
        return not any(re.search(pattern, text) for pattern in non_prepurchase_patterns)

    faq_controls = cfg.get("faq_controls") or {}
    confidence_rank = {"low": 1, "medium": 2, "high": 3}
    min_confidence = str(
        faq_controls.get("minimum_specification_confidence", "high")
    ).lower()
    min_rank = confidence_rank.get(min_confidence, 3)
    primary_facts = {}
    profile = cfg.get("_canonical_profile") or {}
    primary_name = str(profile.get("primary_product") or recommended_product or "").casefold()
    for product in profile.get("products") or []:
        if str(product.get("name") or "").casefold() == primary_name:
            primary_facts = product.get("facts") or {}
            break

    hard_spec_terms = {
        "dimensions": ("dimension", "dimensions", "size", "measure", "measurement"),
        "weight": ("weight", "weigh", "mass"),
        "capacity": ("capacity", "litre", "liter", "volume"),
        "device_fit": ("fit", "fits", "compatibility", "compatible", "laptop", "device"),
        "electrical_rating": ("voltage", "wattage", "amps", "power rating"),
        "safety_limit": ("maximum load", "weight limit", "safety limit"),
    }

    def uses_unconfirmed_hard_spec(question, answer):
        if not faq_controls.get("omit_unconfirmed_specification_questions", True):
            return False
        text = f"{question} {answer}".casefold()
        for attribute, fact in primary_facts.items():
            attr_key = str(attribute).casefold()
            terms = hard_spec_terms.get(attr_key)
            if terms is None:
                if any(marker in attr_key for marker in (
                    "dimension", "weight", "capacity", "size", "fit",
                    "compat", "rating", "voltage", "watt", "limit"
                )):
                    terms = tuple(re.sub(r"[_-]+", " ", attr_key).split())
                else:
                    continue
            confidence = str((fact or {}).get("confidence") or "low").lower()
            canonical_value = str((fact or {}).get("canonical_value") or "").strip()
            if (not canonical_value or confidence_rank.get(confidence, 1) < min_rank):
                if any(re.search(rf"(?i)\b{re.escape(term)}\b", text) for term in terms):
                    return True
                literals = [
                    canonical_value,
                    *((fact or {}).get("conflicting_values") or []),
                ]
                if any(str(value).strip().casefold() in text for value in literals if str(value).strip()):
                    return True
        return False

    faqs = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        question = re.sub(r"<[^>]+>", "", str(item.get("q", ""))).strip()
        answer = re.sub(r"<[^>]+>", "", str(item.get("a", ""))).strip()
        if (
            question
            and answer
            and is_prepurchase(question, answer)
            and not uses_unconfirmed_hard_spec(question, answer)
        ):
            faqs.append({"q": question, "a": answer})
        if len(faqs) >= max_q:
            break
    return faqs


def render_faq_html(faqs, faq_level_label="Frequently Asked Questions", level_number="11.0"):
    """
    Render FAQ as plain Q&A blocks (always visible) + FAQPage JSON-LD.
    Avoids <details>/<summary> so answers won't be hidden by theme/CMS.
    """
    if not faqs:
        return ""

    # Visible HTML (no <ul>/<table>, so your global caps won't strip it)
    html_parts = [f'<h2>{escape(level_number)} {escape(faq_level_label)}</h2>']

    for i, qa in enumerate(faqs, start=1):
        q_raw = (qa.get("q", "") or "").strip()
        a_raw = (qa.get("a", "") or "").strip()

        # Defensive: if an answer somehow comes back empty, use an em dash
        if not a_raw:
            a_raw = "â€”"

        q = escape(q_raw)
        a = escape(a_raw)

        html_parts.append(
            f'<div class="faq-qa">'
            f'<p><strong>Q{i}. {q}</strong><br>{a}</p>'
            f'</div>'
        )

    # FAQPage JSON-LD (keeps answers in structured data for SEO)
    json_ld = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": (qa.get("q", "") or "").strip(),
                "acceptedAnswer": {"@type": "Answer", "text": ((qa.get("a", "") or "").strip() or "â€”")}
            } for qa in faqs
        ]
    }

    html_parts.append(
        '<script type="application/ld+json">' +
        json.dumps(json_ld, ensure_ascii=False) +
        '</script>'
    )


    return "\n".join(html_parts)



# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Outline parsing & top pick selection
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def parse_outline(file_name):
    try:
        with open(file_name, "r", encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        headings = {}
        current_main = None
        for line in lines:
            match = re.match(r"^(\d+\.\d*)\s+(.+)", line)
            if match:
                level, title = match.groups()
                if level.endswith(".0"):
                    headings[level] = {"title": title, "subheadings": {}}
                    current_main = level
                elif current_main:
                    headings[current_main]["subheadings"][level] = {"title": title}
        return headings
    except Exception as e:
        logging.error(f"Error parsing outline: {e}")
        return {}

def select_top_product(structured_products, brand_lexicon=None, cfg=None, audit=None):
    cfg = cfg or {}

    def clean_product_name(name: str) -> str:
        name = (name or "").strip()
        if not name:
            return ""

        # "Best X: PRODUCT" â†’ PRODUCT
        if ":" in name:
            left, right = name.split(":", 1)
            if re.search(r"\b(best|top|our)\b", left, re.I):
                name = right.strip()

        name = re.sub(r"\b(in the uk|in uk|uk)\b$", "", name, flags=re.I).strip()
        name = re.sub(r"^(Top|Best|Excellent|Great|The)\s+", "", name, flags=re.I).strip()
        name = re.sub(r"\s*[-â€“â€”:]+\s*(Our\s+)?(Top\s+)?Pick.*$", "", name, flags=re.I).strip()
        name = re.sub(r"(?i)\s+(reviews?|review|price comparison|alternatives)\b$", "", name).strip()

        # âœ… NEW: strip common editorial suffixes that leak in as â€œproduct namesâ€
        name = re.sub(
            r"(?i)\s+("
            r"quality|alternatives?|price|key specs|specs|performance|design( and size)?|filters?|"
            r"room coverage|sound|power consumption|customer service|additional features"
            r")\b$",
            "",
            name
        ).strip()

        return name

import re

def is_valid_product_name(name: str) -> bool:
    # Must be at least 2 tokens (prevents "Purifier", "Dyson", etc.)
    if not name or len(name.split()) < 2:
        return False

    n = (name or "").strip()
    n_l = n.lower()

    # Basic rejects
    if n.endswith("?"):
        return False

    # Reject section-label colon headings like "Filter: Maintenance"
    if ":" in n:
        left, right = [p.strip() for p in n.split(":", 1)]
        left_l, right_l = left.lower(), right.lower()

        # âœ… Config-driven: common section-label left sides
        section_left_sides = [
            str(x).strip().lower()
            for x in (cfg.get("section_label_left_sides") or [])
            if str(x).strip()
        ]
        if left_l in set(section_left_sides):
            return False

        # Right side looks like a how-to / section topic (generic; keep in code)
        if re.search(r"(?i)\b(set[-\s]?up|how to|overview|guide|manual|instructions|maintenance|faq|specs)\b", right_l):
            return False

        # Conservative: short left side (<=2 words) is almost always a heading label
        if len(left.split()) <= 2:
            return False

    # Reject question/explainer prefixes
    if any(n_l.startswith(prefix) for prefix in [
        "why ", "how ", "what ", "should ", "is ", "are ", "does ", "do ", "did ",
        "testing ", "test ", "about "
    ]):
        return False

    if n_l.startswith("text:"):
        return False

    # âœ… Config-driven: category-only reject (no hardcoded regex like air purifiers)
    if is_category_only_product_name(n, cfg=cfg):
        return False

    # UI / generic headline filters
    if _looks_like_ui_phrase(n):
        return False
    if looks_like_generic_headline(n, brand_lexicon=brand_lexicon, cfg=cfg):
        return False

    # Must look like a specific product (brand/model cues etc.)
    if not looks_like_unique_product(n, brand_lexicon=brand_lexicon, cfg=cfg):
        return False

    # Require strong product cues so headings don't pass
    if not _has_strong_product_cues(n, brand_lexicon=brand_lexicon, cfg=cfg):
        return False

    # ðŸš« Amazon/retailer section headings (common false positives) â€” exact match
    bad = {
        "from the manufacturer",
        "from the brand",
        "product information",
        "product description",
        "about us",
        "customer reviews",
        "product guidance & documents",
        "product guidance and documents",

        # Amazon UI/merch modules that leak into candidates
        "products related to this item",
        "featured items you may like",
        "customers who viewed this item also viewed",
        "customers bought together",
        "products customers bought together",
        "general view",
        "you may also be interested in",
    }
    if re.sub(r"\s+", " ", n).strip().lower() in bad:
        return False

    # ðŸš« Generic meta headings (match anywhere, not just end)
    if re.search(
        r"(?i)\b("
        r"reviews?|review|price comparison|alternatives?|"
        r"room coverage|coverage|cadr|clean air delivery rate|effectiveness|"
        r"power consumption|customer service|sound|noise|"
        r"should you buy|key specs|specs|performance|design and size|design|filters|additional features|"
        r"you may also be interested in|products related to this item|featured items you may like|"
        r"customers who viewed this item also viewed|"
        r"customers bought together|products customers bought together|general view"
        r")\b",
        n
    ):
        return False

    return True


    for entry in structured_products:
        raw_name = (entry.get("product") or "").strip()
        cleaned_name = clean_product_name(raw_name)

        pros_count = len(entry.get("pros", []) or [])
        cons_count = len(entry.get("cons", []) or [])
        score = pros_count - cons_count

        # âœ… NEW: Early UI/CTA filter at ingestion (donâ€™t rely on other gates)
        if _looks_like_ui_phrase(cleaned_name):
            rejected_log.append({
                "product": raw_name,
                "cleaned_name": cleaned_name,
                "reason": "ui_or_cta_phrase",
                "pros_count": pros_count,
                "cons_count": cons_count,
                "score": score
            })
            continue

        # --- Fix 3: filter at ingestion + keep rejected audit trail ---
        reason = None

        if is_obviously_not_a_product(cleaned_name, brand_lexicon=brand_lexicon, cfg=cfg):
            reason = "obviously_not_product"
        else:
            # âœ… Consolidate gating inside is_valid_product_name (avoids double looks_like_unique_product)
            valid = is_valid_product_name(cleaned_name)
            if not valid:
                reason = "invalid_product_name"

        if reason:
            rejected_log.append({
                "product": raw_name,
                "cleaned_name": cleaned_name,
                "reason": reason,
                "pros_count": pros_count,
                "cons_count": cons_count,
                "score": score
            })
            continue

        # If we got here, it's a real candidate
        candidates_log.append({
            "product": raw_name,
            "cleaned_name": cleaned_name,
            "valid": True,
            "pros_count": pros_count,
            "cons_count": cons_count,
            "score": score
        })

        if score > best_score:
            best_score = score
            best_product = cleaned_name

    # Sort candidates in audit by score desc, then pros desc, then cons asc
    candidates_log.sort(key=lambda x: (x["score"], x["pros_count"], -x["cons_count"]), reverse=True)

    if audit is not None:
        audit.setdefault("fallback", {})
        audit["fallback"]["candidates"] = candidates_log
        audit["fallback"]["rejected"] = rejected_log
        audit["fallback"]["winner"] = best_product or ""
        audit["fallback"]["counts"] = {
            "candidates": len(candidates_log),
            "rejected": len(rejected_log)
        }

    if best_product:
        logging.info(f"Fallback selected top pick: '{best_product}' (score={best_score})")
        return best_product

    logging.warning("Fallback could not find a valid product; defaulting to generic phrase.")
    return "one of the products reviewed"



    
def strip_existing_faq_sections(html: str) -> str:
    if not html:
        return html or ""

    # Remove visible FAQ section: from a <h2> that contains "Frequently Asked Questions"
    # through to just before the next <h2> (or end of doc)
    html = re.sub(
        r'(?is)<h2\b[^>]*>[^<]*Frequently\s+Asked\s+Questions[^<]*</h2>[\s\S]*?(?=(?:<h2\b[^>]*>)|$)',
        '',
        html
    )

    # Remove FAQPage JSON-LD blocks only (leave other JSON-LD alone)
    def _rm_faq_jsonld(m):
        return '' if 'FAQPage' in m.group(0) else m.group(0)

    html = re.sub(
        r'(?is)<script[^>]+type=["\']application/ld\+json["\'][^>]*>[\s\S]*?</script>',
        _rm_faq_jsonld,
        html
    )
    return html
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# NEW: INTERNAL LINKING & METADATA HELPERS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



def build_outline_from_html(html: str):
    """
    Parse H2 sections in render order.
    Returns: [{"index": i, "title": "1.0 Purpose...", "slug": "purpose-of-the-review"}]
    """
    outline = []
    for i, m in enumerate(re.finditer(r'(?is)(<h2\b[^>]*>)([\s\S]*?)(</h2>)', html)):
        inner = re.sub(r'(?is)</?h2\b[^>]*>', '', m.group(0)).strip()
        title = re.sub(r'<[^>]+>', '', inner).strip()
        outline.append({"index": i, "title": title, "slug": _slugify(title)})
    return outline


def _extract_keywords_from_text(text: str, k: int = 10):
    """
    Lightweight keyword guesser (fallback if model call fails).
    """
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())
    stop = set("""
        the a an and or for with from into on to of in is are be this that these those it its they them we you i our your by
        as but if then than also very more most some any each other about over under within without not no yes can will may
    """.split())
    counts = {}
    for w in words:
        if w in stop:
            continue
        counts[w] = counts.get(w, 0) + 1
    kw = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return [w for w, _ in kw]

def build_post_metadata(html: str, keyword: str):
    """
    Produces: title, summary (1â€“2 sentences), keywords (5â€“12), anchor_candidates (4â€“10).
    Uses DeepSeek if available; gracefully falls back to heuristics.
    """
    plain = re.sub(r"<[^>]+>", " ", html)
    plain = re.sub(r"\s+", " ", plain).strip()

    # Default title from first H2 or keyword
    first_h2 = re.search(r'(?is)<h2\b[^>]*>[\s\S]*?</h2>', html)
    default_title = keyword.strip().title()
    if first_h2:
        t = re.sub(r'(?is)</?h2\b[^>]*>', '', first_h2.group(0)).strip()
        t = re.sub(r'<[^>]+>', '', t).strip()
        # If the first H2 is "1.0 ..." (purpose), prefer keyword
        if not re.match(r"^\d+(\.\d+)?\s+", t, flags=re.IGNORECASE):
            default_title = t

    # Try DeepSeek for crisp metadata
    try:
        prompt = f"""
Summarize and tag an article draft.

Return STRICT JSON with keys:
- "title": human title â‰¤ 70 chars
- "summary": 1â€“2 sentences, â‰¤ 320 chars total
- "keywords": 5â€“12 short phrases (no hashtags, no quotes)
- "anchor_candidates": 4â€“10 natural phrases that would make good internal-link anchor text

Context keyword: "{keyword}"

Text:
{plain}
"""
        raw = deepseek_generate(prompt, model=DEEPSEEK_MODEL, label="post_metadata")
        meta = _json_from_text_block(raw)
        if not isinstance(meta, dict):
            raise ValueError("bad meta json")
        title = str(meta.get("title") or default_title).strip()[:70]
        summary = str(meta.get("summary") or "").strip()
        keywords = [k.strip() for k in meta.get("keywords", []) if str(k).strip()]
        anchors = [a.strip() for a in meta.get("anchor_candidates", []) if str(a).strip()]
    except Exception:
        title = default_title
        summary = (plain[:300] + "...") if len(plain) > 303 else plain
        keywords = _extract_keywords_from_text(plain, k=10)
        anchors = list({*(keywords[:6]), *(re.findall(r"\b[A-Za-z]+(?:\s+[A-Za-z0-9\-]+){1,3}\b", title)[:4])})

    # Final trims
    summary = re.sub(r"\s+", " ", summary).strip()
    if len(summary) > 320:
        summary = summary[:317].rstrip() + "..."
    # ensure distinct, shortish anchors
    anchors = [a for a in anchors if 2 <= len(a.split()) <= 6][:10]

    return {
        "title": title,
        "summary": summary,
        "keywords": keywords[:12],
        "anchor_candidates": anchors
    }
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Internal linking helpers (during generation)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_WORD_RX = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")
_TAG_RX  = re.compile(r"<[^>]+>")
_WS_RX   = re.compile(r"\s+")
_PARA_RX = re.compile(r"(?is)<p\b[^>]*>.*?</p>")

STOPWORDS = {
    "a","an","the","and","or","of","for","to","in","on","with","by","about","is",
    "are","was","were","be","being","been","it","its","this","that","these","those",
    "as","at","from","into","over","under","up","down","you","your","our","their","there","here",
}

def _strip_html(s: str) -> str:
    return _WS_RX.sub(" ", _TAG_RX.sub(" ", s or "")).strip()

def _tokens(text: str) -> set:
    return {w.lower().strip("-'") for w in _WORD_RX.findall(text or "") if w.lower() not in STOPWORDS}




# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Main
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    keyword, country = read_keyword_from_file()
    if not keyword or country not in ["US", "UK", "CA"]:
        raise SystemExit("Invalid or missing keyword/country.")

    setup_logging(keyword, country)

    # Load & choose category config
    cfg_path = os.getenv("TOPIC_CONFIG", "config/category_config.json")
    topic_hint = (
        os.getenv("ORCH_CATEGORY")
        or os.getenv("TOPIC_KEY")
        or read_category_from_file()
    )
    cfg, cfg_name = load_topic_config(cfg_path, topic_key=topic_hint)
    configure_runtime_category(cfg)

    logging.info(f"Using category config: {cfg_name}")


    safe_keyword = keyword.replace(" ", "_")
    safe_keyword_country = f"{safe_keyword}_{country}"
    output_dir = os.path.join("output", safe_keyword_country)
    output_path = os.path.join(output_dir, f"generated_blog_content_{country}.txt")

    os.makedirs(output_dir, exist_ok=True)

    abandon_path = os.path.join(output_dir, "abandon_post.txt")
    try:
        if os.path.exists(abandon_path):
            os.remove(abandon_path)
    except Exception as e:
        logging.warning(f"Could not remove stale abandon_post.txt: {e}")

    style_guide = read_file("input/style_guide.txt")
    full_outline = parse_outline("input/improved_content_outline_reduced.txt")


    # Step 1: Read and clean raw dataset
    # NOTE: We want the pipeline to keep running even when the dataset is tiny.
    # A short dataset is still useful (it can produce a single-product review or
    # a more cautious, neutral article). Only a *missing/empty* dataset gets a
    # minimal fallback scaffold.
    raw_dataset_path = os.path.join(output_dir, f"content_{country}_updated.txt")
    raw_dataset = read_file(raw_dataset_path)

    removed_dedupe_path = os.path.join(
        output_dir,
        f"removed_dedupe_and_minhash_{country}.txt"
    )

    raw_dataset = preprocess_raw_dataset(
        raw_dataset,
        brand_lexicon=None,   # or pass a known set later if available
        removed_text_path=removed_dedupe_path,
        cfg=cfg,
    )

    raw_len = len((raw_dataset or "").strip())
    if raw_len == 0:
        logging.warning(f"âš ï¸ Dataset missing/empty: {raw_dataset_path} â€” creating a minimal fallback dataset.")
        raw_dataset = (
            f"h2 {keyword}\n"
            f"Text: Limited source data was available for {keyword}.\n"
        )
    elif raw_len < 20:
        logging.warning(f"âš ï¸ Dataset is very short ({raw_len} chars): {raw_dataset_path} â€” proceeding anyway.")
        raw_dataset = (raw_dataset.strip() + f"\nText: Dataset is short; write cautiously and avoid adding facts.\n")
    
    flagged_lines_path = os.path.join(output_dir, f"flagged_noisy_keywords_{country}.txt")
    flagged_lines = []
    cleaned_dataset, removed_dataset = clean_scraped_dataset(raw_dataset, flagged_lines, cfg)
    if not cleaned_dataset.strip():
        # If the cleaner stripped everything (common with tiny inputs), keep the run alive
        # with a conservative fallback dataset.
        logging.warning("âš ï¸ Cleaned dataset is empty after cleaning; using a minimal fallback cleaned dataset.")
        cleaned_dataset = (
            f"[H2] {keyword}\n"
            f"Text: Limited usable review content was found for {keyword}.\n"
        )


    # Save cleaned & removed
    cleaned_path = os.path.join(output_dir, f"cleaned_content_{country}.txt")
    with open(cleaned_path, "w", encoding="utf-8") as f:
        f.write(cleaned_dataset)
    logging.info(f"âœ… Cleaned content saved to {cleaned_path}")

    # â”€â”€ Phase B: Promote [H2] headings to [PRODUCT] where appropriate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    h2_candidates = extract_h2_candidates(cleaned_dataset)

    # Build a brand lexicon from H2 candidates (early signal)
    h2_brand_lexicon = build_brand_lexicon_from_names(h2_candidates, min_freq=2)

    # Promote H2 -> PRODUCT using the identity gate
    cleaned_dataset = promote_h2_to_product_tags(
        cleaned_dataset,
        whitelist=None,
        brand_lexicon=h2_brand_lexicon,
        cfg=cfg
    )

    cleaned_dataset = canonicalize_product_tags_in_text(cleaned_dataset, cfg=cfg)
    logging.info("Canonicalized product aliases and demoted legal-company headings.")
    # Save promoted version (so downstream reads [PRODUCT] blocks)
    with open(cleaned_path, "w", encoding="utf-8") as f:
        f.write(cleaned_dataset)
    logging.info("âœ… Phase B: promoted [H2] -> [PRODUCT] and re-saved cleaned content.")

    logging.info(f"Remaining [H2] headings: {cleaned_dataset.count('[H2] ')}")
    logging.info(f"Final [PRODUCT] count: {cleaned_dataset.count('[PRODUCT] ')}")

    removed_path = os.path.join(output_dir, f"removed_content_{country}.txt")
    with open(removed_path, "w", encoding="utf-8") as f:
        f.write(removed_dataset)
    if flagged_lines:
        with open(flagged_lines_path, "w", encoding="utf-8") as f:
            f.write("\n".join(flagged_lines))
        logging.info(f"ðŸš§ Flagged keyword lines saved to {flagged_lines_path}")

    # Step 2: Structure + canonicalise
    structured_products_raw = extract_products_to_json(cleaned_dataset)
    structured_products = canonicalize_structured_products(structured_products_raw)

    # Brand lexicon from initial structure
    brand_lexicon = build_brand_lexicon_from_structured_products(structured_products, min_freq= 1)


    # Prune bad [PRODUCT] tags (ONLY ONCE)
    before = cleaned_dataset.count("[PRODUCT] ")
    cleaned_dataset = prune_bad_product_tags(cleaned_dataset, brand_lexicon=brand_lexicon, cfg=cfg)
    after = cleaned_dataset.count("[PRODUCT] ")

    with open(cleaned_path, "w", encoding="utf-8") as f:
        f.write(cleaned_dataset)

    logging.info(f"âœ… Pruned [PRODUCT] tags: {before} -> {after}; re-saved to {cleaned_path}")

    # Re-extract after pruning
    structured_products_raw = extract_products_to_json(cleaned_dataset)
    structured_products = canonicalize_structured_products(structured_products_raw)

    # Recompute brand lexicon (recommended)
    brand_lexicon = build_brand_lexicon_from_structured_products(structured_products, min_freq=1)


    structured_path = os.path.join(output_dir, f"structured_products_{country}.json")
    with open(structured_path, "w", encoding="utf-8") as f:
        json.dump(structured_products, f, indent=2, ensure_ascii=False)
    logging.info(f"âœ… Structured product data saved to {structured_path}")

    # Top pick (DeepSeek â†’ fallback) with full audit
    audit = init_top_pick_audit(keyword, country)

    brand_lexicon = build_brand_lexicon_from_structured_products(structured_products, min_freq=1)

    def _merge_whitelist(
        base: list[str],
        additions: list[str],
        *,
        brand_lexicon: set[str] | None,
        cfg: dict
    ) -> list[str]:
        """
        Single whitelist update path:
        - strip placeholders
        - strip leading bullets/dashes
        - title-case
        - reject 'For ...' / 'Best for ...'
        - reject obvious SECTION HEADINGS (key features/specs/how-to/etc.)
        - de-dupe
        - prune via prune_product_whitelist (identity gate)
        """
        merged = list(base or [])
        seen_local = set(p.lower() for p in merged if p)

        # âœ… NEW: reject obvious non-product/section-heading phrases
        HEADING_PREFIX_RX = re.compile(
            "(?i)^(how|why|what|when|where|which|tips?|guide|choosing|overview|summary|conclusion|"
            r"final verdict|value for money|does|do|did|is|are|was|were|testing|test|about)\b"
        )
        HEADING_PHRASE_RX = re.compile(
            r"(?i)\b("
            r"key features|features\b|specs\b|specifications\b|at a glance|"
            r"how i tested|how do\b|how easy is it|how many\b|should you buy|"
            r"pros and cons|verdict|review\b|comparison\b|vs\b|versus\b"
            r")\b"
        )

        for a in (additions or []):
            if not a:
                continue

            x = strip_product_placeholders(str(a))
            x = (x or "").strip().lstrip(" \t\r\n-â€“â€”:â€¢|")

            # âœ… MUST pass cfg so prefix stripper uses cfg labels
            x = strip_serp_editorial_wrappers(x, cfg=cfg).strip()

            x = trim_product_title_tail(x, max_comma_segments=2)
            x = smart_title_case(x).strip()

            if not x:
                continue

            # reject heading-style labels
            if re.match(r"(?i)^(for|best\s+for)\b", x):
                continue

            # âœ… reject obvious section headings
            if HEADING_PREFIX_RX.match(x):
                continue
            if HEADING_PHRASE_RX.search(x):
                continue

            # âœ… NEW: prevent category-only labels entering via additions
            if is_category_only_product_name(x, cfg=cfg):
                continue

            # âœ… NEW: prevent testimonial headings entering via additions
            if looks_like_testimonial_heading(x):
                continue

            # âœ… NEW: cfg-aware hard block of obvious non-products
            if is_obviously_not_a_product(x, brand_lexicon=brand_lexicon, cfg=cfg):
                continue

            # âœ… Keep only real product-looking entries
            if not (_has_strong_product_cues(x, brand_lexicon=brand_lexicon, cfg=cfg) and
                    looks_like_unique_product(x, brand_lexicon=brand_lexicon, cfg=cfg)):
                continue

            k = x.lower()
            if k in seen_local:
                continue

            # NEW: block date/author/blog-metadata strings before they enter the whitelist
            if looks_like_blog_metadata(x):
                continue

            seen_local.add(k)
            merged.append(x)

            if looks_like_blog_metadata(x):
                continue

        return prune_product_whitelist(merged, brand_lexicon=brand_lexicon, cfg=cfg)


    # â”€â”€ Build whitelist EARLY (before top pick selection) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Reject common section-heading patterns at the point of ingestion so they
    # never become TOP_PICK_HINTS.
    HEADING_PREFIX_RX_EARLY = re.compile(
        r"""(?ix)
        ^\s*(?: 
            how|why|what|when|where|which|
            tips?|guide|choosing|overview|summary|conclusion|
            final\s+verdict|value\s+for\s+money|
            does|do|did|is|are|was|were|
            unboxing|setting\s+up|setup|how\s+to|hands\s+on|first\s+impressions|
            testing|test|about
        )\b
        """
    )



    HEADING_PHRASE_RX_EARLY = re.compile(
        r"(?i)\b("
        r"key features|features\b|key specs|specs\b|specifications\b|at a glance|"
        r"how i tested|testing\b|test results|"
        r"pros and cons|verdict|review\b|"
        r"comparison\b|compare\b|vs\b|versus\b|alternatives\b|"
        r"room coverage|cadr|"
        r"price\b|design\b|size\b|filters\b|sound\b|"
        r"power consumption|customer service|additional features"
        r"|featured items|you may like|products related to this item|related to this item|products related|"
        r"customers bought together|products customers bought together|general view"
        r")\b"
    )

    product_whitelist = []
    seen = set()
    for e in structured_products:
        raw_p = (e.get("product") or "").strip()
        if not raw_p:
            continue

        p = clean_product_name(raw_p, cfg=cfg).strip()
        p = smart_title_case(p).strip()
        
        # âœ… ADD THIS LINE
        p = strip_serp_editorial_wrappers(p, cfg=cfg).strip()



        # (optional) re-title-case if your stripper changes spacing/case
        p = smart_title_case(p).strip()

        # âœ… reject "For ..." / "Best for ..." headings so they never enter the whitelist
        if re.match(r"(?i)^(for|best\s+for)\b", p):
            continue

        # âœ… NEW: reject obvious UI/editorial phrases and section headings
        if _looks_like_ui_phrase(p):
            continue
        if HEADING_PREFIX_RX_EARLY.match(p):
            continue
        if HEADING_PHRASE_RX_EARLY.search(p):
            continue

        if p:
            # Normalize once for checks (prevents "Also Great:", "Verdict:", etc. poisoning hints)
            p_clean = strip_serp_editorial_wrappers(p, cfg=cfg).strip()
            if not p_clean:
                continue
                
            if looks_like_blog_metadata(p_clean):
                continue

            p_key = p_clean.lower()
            if p_key in seen:
                continue

            # ðŸš« Never allow category-only labels into hints
            if is_category_only_product_name(p_clean, cfg=cfg):
                continue

            # ðŸš« Never allow Trustburn-style testimonial headings into hints
            if looks_like_testimonial_heading(p_clean):
                continue

            # ðŸš« Never allow obvious headings/UI into hints (cfg-aware)
            if is_obviously_not_a_product(p_clean, brand_lexicon=brand_lexicon, cfg=cfg):
                continue

            # âœ… Only keep entries that look like real products (strong cues)
            if _has_strong_product_cues(p_clean, brand_lexicon=brand_lexicon, cfg=cfg) and \
               looks_like_unique_product(p_clean, brand_lexicon=brand_lexicon, cfg=cfg):
                seen.add(p_key)
                product_whitelist.append(p_clean)



    product_whitelist = _merge_whitelist(
        product_whitelist,
        additions=[],
        brand_lexicon=brand_lexicon,
        cfg=cfg
    )


    deepseek_pick = find_primary_product(
        cleaned_dataset,
        keyword,
        product_whitelist=product_whitelist,
        brand_lexicon=brand_lexicon,
        cfg=cfg,
        audit=audit
    )

    audit["method"] = "deepseek"

    if not deepseek_pick or "one of the products" in deepseek_pick.lower():
        logging.warning("Fallback: DeepSeek could not extract a clean product name. Using structured data.")
        audit["method"] = "fallback"
        fallback_pick = select_top_product(structured_products, brand_lexicon=brand_lexicon, cfg=cfg, audit=audit)
        raw_final = fallback_pick
    else:
        raw_final = deepseek_pick
        
    source = (audit or {}).get("selection", {}).get("source", "")
    if source == "amazon_pdp_h1" and raw_final:
        product_whitelist = _merge_whitelist(
            product_whitelist,
            additions=[raw_final],
            brand_lexicon=brand_lexicon,
            cfg=cfg
        )


    audit["canonicalization"]["final_name_before_titlecase"] = strip_url_and_retailer_chrome(
        strip_product_placeholders(
            strip_heading_marker_prefix(str(raw_final or "").strip().rstrip(" .,:;!?"))
        )
    )

    if audit["canonicalization"]["final_name_before_titlecase"].lower().startswith(("h1 ", "h2 ", "h3 ")):
        logging.warning("[TOP_PICK] heading marker still present after stripping (unexpected)")

    source = (audit or {}).get("selection", {}).get("source", "")
    allow_fuzzy_map = (source != "amazon_pdp_h1")

    recommended_product = normalize_product_name(
        audit["canonicalization"]["final_name_before_titlecase"],
        whitelist=product_whitelist,
        cfg=cfg,
        allow_strip_best_prefix=True,
        try_map_to_whitelist=allow_fuzzy_map,
        max_words=None,
    )
    
    recommended_product = normalize_all_caps_product(recommended_product)
    
    if is_bad_top_pick_candidate(recommended_product, brand_lexicon=brand_lexicon, cfg=cfg):
        rescue = pick_specific_from_whitelist(
            product_whitelist,
            brand_lexicon=brand_lexicon,
            cfg=cfg,
            keyword=keyword,
        )
        if rescue:
            recommended_product = normalize_all_caps_product(rescue)
            audit.setdefault("selection", {})
            audit["selection"]["source"] = "whitelist_rescue"
            audit["selection"]["verbatim_title"] = rescue

    # Keep your category-tail trimming behavior, but do it BEFORE the final cleaner
    recommended_product = strip_product_placeholders(recommended_product).strip()
    recommended_product = strip_section_wrapper_prefix(recommended_product, cfg=cfg).strip()
    recommended_product = strip_sentence_tail(recommended_product).strip()
    recommended_product = strip_leading_review_verbs(recommended_product).strip()
    recommended_product = strip_leading_article(recommended_product, brand_lexicon=brand_lexicon, cfg=cfg).strip()
    recommended_product = strip_editorial_suffixes(recommended_product).strip()
    recommended_product = trim_to_category_tail(recommended_product, cfg=cfg)
    recommended_product = recommended_product.strip()

    # âœ… Don't remap again (prevents drifting)
    recommended_product = normalize_product_name(
        recommended_product,
        whitelist=product_whitelist,
        cfg=cfg,
        allow_strip_best_prefix=True,
        try_map_to_whitelist=False,
        max_words=None,
    )

    def _final_clean_product_name(x: str) -> str:
        """
        Canonical final cleanup stack (ONE place so rescue + main path match).
        """
        x = strip_url_and_retailer_chrome(x).strip()
        x = strip_heading_marker_prefix(x).strip()
        x = strip_product_placeholders(x).strip()
        x = strip_serp_editorial_wrappers(x, cfg=cfg).strip()
        x = strip_section_wrapper_prefix(x, cfg=cfg).strip()  # "Also Great:", etc.
        x = strip_sentence_tail(x).strip()
        x = strip_leading_review_verbs(x).strip()
        x = strip_leading_article(x).strip()                  # "The Dyson ..." -> "Dyson ..."
        x = strip_leading_definite_article(x).strip()         # extra safety for matched editorial titles
        x = strip_editorial_suffixes(x).strip()
        x = strip_leading_definite_article(x).strip()         # e.g. "The Russell Hobbs ...: an In-Depth Look"
        x = trim_to_category_tail(x, cfg=cfg)
        x = strip_sentence_tail(x).strip()                    # one more pass after trimming
        x = smart_title_case(x).strip()
        return x

    marker_or_deny_bad = (
        looks_like_marker_contaminated_product(recommended_product)
        or looks_like_reversed_category_brand(recommended_product, cfg=cfg)
        or has_top_pick_deny_signal(recommended_product, cfg=cfg)
    )
    if marker_or_deny_bad:
        rescue = pick_specific_from_whitelist(
            product_whitelist,
            brand_lexicon=brand_lexicon,
            cfg=cfg,
            keyword=keyword,
        )
        if rescue:
            recommended_product = rescue
            audit.setdefault("selection", {})
            audit["selection"]["source"] = "whitelist_rescue"
            audit["selection"]["verbatim_title"] = rescue

    # âœ… FINAL safety net (before rescue)
    recommended_product = _final_clean_product_name(recommended_product)

    # âœ… FINAL SAFETY NET: if we still ended up with a category-only label, rescue from whitelist
    if is_category_only_product_name(recommended_product, cfg=cfg):
        rescue = pick_specific_from_whitelist(
            product_whitelist,
            brand_lexicon=brand_lexicon,
            cfg=cfg,
            keyword=keyword,
        )
        if rescue:
            logging.warning(f"[TOP_PICK_RESCUE] category_only='{recommended_product}' -> '{rescue}'")
            recommended_product = _final_clean_product_name(rescue)

            if audit is not None:
                audit.setdefault("selection", {})
                audit["selection"]["source"] = "whitelist_rescue"
                audit["selection"]["verbatim_title"] = rescue
        else:
            logging.warning(
                f"[TOP_PICK_RESCUE_FAILED] category_only='{recommended_product}' and no whitelist rescue available"
            )

    # âœ… Write final_name ONCE (after rescue)
    # Final reviewed-product identity gate. Never change model or size merely
    # because a different affiliate item or whitelist entry is available.
    identity_controls = cfg.get("review_identity") or {}
    if not reviewed_product_matches_keyword(recommended_product, keyword, cfg):
        identity_rescue = pick_specific_from_whitelist(
            product_whitelist,
            brand_lexicon=brand_lexicon,
            cfg=cfg,
            keyword=keyword,
        )
        if identity_rescue and reviewed_product_matches_keyword(identity_rescue, keyword, cfg):
            logging.warning(
                "[REVIEW_IDENTITY] Replaced mismatched reviewed product '%s' with '%s'.",
                recommended_product,
                identity_rescue,
            )
            recommended_product = _final_clean_product_name(identity_rescue)
            audit.setdefault("selection", {})
            audit["selection"]["source"] = "keyword_identity_rescue"
            audit["selection"]["verbatim_title"] = identity_rescue
        elif identity_controls.get("block_on_mismatch", True):
            msg = (
                "Post abandoned: reviewed product does not match keyword model identity.\n"
                f"- Keyword: {keyword} ({country})\n"
                f"- Candidate: {recommended_product}\n"
                "Action: supply evidence for the exact requested model; affiliate substitutes "
                "must remain downstream choices."
            )
            with open(abandon_path, "w", encoding="utf-8") as abandon_file:
                abandon_file.write(msg)
            logging.error(msg)
            audit.setdefault("enforcement", {})["review_identity_match"] = False
            save_top_pick_audit(audit, output_dir, country)
            sys.exit(4)

    audit.setdefault("enforcement", {})["review_identity_match"] = True

    audit["canonicalization"]["final_name"] = recommended_product


    if is_category_only_product_name(recommended_product, cfg=cfg):
        audit["canonicalization"]["final_name"] = recommended_product  # update after rescue



    
    # âœ… single whitelist update path (no legacy rebuild)
    product_whitelist = _merge_whitelist(
        product_whitelist,
        additions=[recommended_product],
        brand_lexicon=brand_lexicon,
        cfg=cfg
    )
    recommended_identity = canonical_product_identity_key(recommended_product)
    product_whitelist = [recommended_product] + [
        product for product in product_whitelist
        if canonical_product_identity_key(product) != recommended_identity
        and not looks_like_legal_entity_name(product)
    ]



    # Save top pick for compatibility
    product_name_file = os.path.join(output_dir, f"product_names_{country}.json")
    with open(product_name_file, "w", encoding="utf-8") as f:
        json.dump({"top_pick": recommended_product}, f, indent=2, ensure_ascii=False)
    logging.info(f"ðŸ† Final saved top pick: {recommended_product}")
    audit["files"]["saved_top_pick_json"] = product_name_file
    top_pick_report_path = export_top_pick_report(
        keyword,
        recommended_product,
        report_dir="output",
        cleaned_content_path=cleaned_path,
        country=country,
        site=os.environ.get("ORCH_SITE", ""),
        category=os.environ.get("ORCH_CATEGORY", ""),
    )
    audit["files"]["top_pick_report_csv"] = top_pick_report_path

    canonical_map = build_canonical_product_map(structured_products, recommended_product)

    # Step 3: Hybrid dataset â†’ normalise casing
    hybrid_dataset = build_hybrid_dataset(cleaned_path, structured_path, max_tokens=115000)
    hybrid_dataset = normalize_product_mentions(hybrid_dataset, canonical_map)

    # Build one authoritative interpretation of conflicting specifications and
    # compatibility claims before any article section is written. Preserve raw
    # model output so a parse failure is diagnosable rather than leaving only {}.
    cfg["_canonical_raw_response_path"] = os.path.join(
        output_dir, f"canonical_product_profile_raw_{country}.txt"
    )
    cfg["_canonical_repair_response_path"] = os.path.join(
        output_dir, f"canonical_product_profile_repair_raw_{country}.txt"
    )
    cfg["_canonical_conflict_repair_response_path"] = os.path.join(
        output_dir, f"canonical_conflict_repair_raw_{country}.txt"
    )
    cfg["_semantic_audit_raw_response_path"] = os.path.join(
        output_dir, f"semantic_fact_audit_raw_{country}.txt"
    )
    cfg["_semantic_repair_raw_response_path"] = os.path.join(
        output_dir, f"semantic_fact_repair_raw_{country}.txt"
    )
    canonical_profile = build_canonical_product_profile(
        hybrid_dataset,
        keyword,
        recommended_product,
        cfg,
        product_whitelist=product_whitelist,
    )
    cfg["_canonical_profile"] = canonical_profile
    cfg["_article_memory"] = []

    canonical_profile_path = os.path.join(
        output_dir,
        f"canonical_product_profile_{country}.json",
    )
    with open(canonical_profile_path, "w", encoding="utf-8") as profile_file:
        json.dump(canonical_profile, profile_file, indent=2, ensure_ascii=False)
    audit["files"]["canonical_product_profile"] = canonical_profile_path

    canonical_controls = cfg.get("canonical_facts") or {}
    if (
        canonical_controls.get("enabled", True)
        and canonical_controls.get("require_profile", True)
        and not canonical_profile.get("products")
    ):
        msg = (
            "Post abandoned: canonical product profile could not be built.\n"
            f"- Keyword: {keyword} ({country})\n"
            f"- Top pick: {recommended_product}\n"
            "Action: inspect the canonical_product_profile model response and source evidence."
        )
        with open(abandon_path, "w", encoding="utf-8") as abandon_file:
            abandon_file.write(msg)
        logging.error(msg)
        save_top_pick_audit(audit, output_dir, country)
        sys.exit(5)

    # â”€â”€ Summarize once (extractive) to shrink later prompts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    summary_prompt = f"""
    Summarize the dataset EXTRACTIVELY for writing reviews of "{keyword}".
    Return STRICT JSON: {{"summary": "...", "products": [{{"name":"...", "notable": ["...","..."]}}]}}
    - Use only facts present in the dataset.
    - Include every product name you can find (brand + model) in "products".
    - No opinions not grounded in the text.

    The reference dataset is supplied separately.
    """
    try:
        summary_raw = deepseek_generate(
            summary_prompt,
            model=DEEPSEEK_MODEL,
            label="dataset_summary",
            max_tokens=1000,
            temperature=0.3,
            cache_prefix=_dataset_cache_prefix(hybrid_dataset),
        )
        m = re.search(r"\{.*\}", summary_raw, flags=re.S)
        summary_obj = json.loads(m.group(0)) if m else {}
        dataset_summary = str(summary_obj.get("summary","")).strip()
        summary_products = []
        for p in summary_obj.get("products", []):
            n_raw = p.get("name", "")
            n = normalize_product_name(
                n_raw,
                whitelist=None,
                cfg=cfg,
                allow_strip_best_prefix=True,
                try_map_to_whitelist=False,   # important: don't map here
                max_words=12,
            )
            n = smart_title_case(n)
            if n:
                summary_products.append(n)

        # after building summary_products
        product_whitelist = _merge_whitelist(
            product_whitelist,
            additions=summary_products,
            brand_lexicon=brand_lexicon,
            cfg=cfg
        )

        sus = [p for p in product_whitelist if len(p.split()) < 2]
        logging.info(f"[WHITELIST_DEBUG] short_entries={sus}")


    except Exception:
        # fallback to your existing summarizer if strict JSON parse fails
        dataset_summary = summarize_dataset(hybrid_dataset, keyword)

    # â”€â”€ Generator wrappers with selective context + safe fallbacks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def gen_intro(title):
        logging.info(f"[SECTION] intro start: {title}")

        # Keep FULL context for Purpose (safeguard)
        html = generate_introduction(
            title,
            hybrid_dataset,
            style_guide,
            keyword,
            country,
            recommended_product,
            cfg,
            product_whitelist=product_whitelist,
        )

        remember_generated_section(cfg, title, html)
        logging.info(f"[SECTION] intro complete: {title} | chars={len(html or '')}")
        return html


    def gen_overview(title):
        logging.info(f"[SECTION] overview start: {title}")

        # Prefer summary to shrink prompt, fallback to subset if summary too short
        ctx = dataset_summary if len(dataset_summary) > 400 else subset_for_heading(
            hybrid_dataset,
            title,
            product_whitelist,
            min_distinct=1
        )

        html = generate_overview(
            title,
            ctx,
            style_guide,
            keyword,
            country,
            recommended_product,
            cfg,
            product_whitelist=product_whitelist,
            label=f"overview:{title}",
            max_tokens=120,
        )

        remember_generated_section(cfg, title, html)
        logging.info(f"[SECTION] overview complete: {title} | chars={len(html or '')}")
        return html


    def gen_detail(title):
        logging.info(f"[SECTION] detail start: {title}")

        ds = subset_for_heading(
            hybrid_dataset,
            title,
            product_whitelist,
            min_distinct=2
        )

        html = generate_detailed_subheading_content(
            title,
            ds,
            style_guide,
            keyword,
            recommended_product,
            cfg,
            product_whitelist=product_whitelist,
            label=f"detail:{title}",
            max_tokens=900,
        )

        logging.info(f"[SECTION] detail generated: {title} | chars={len(html or '')}")

        # quick post-check + optional re-run with full context
        html_checked = regenerate_if_bad(
            html,
            title,
            recommended_product,
            product_whitelist,
            hybrid_dataset,
            style_guide,
            keyword,
            cfg,
        )

        if html_checked != html:
            logging.info(f"[SECTION] detail regenerated (fix applied): {title}")

        remember_generated_section(cfg, title, html_checked)
        logging.info(f"[SECTION] detail complete: {title} | chars={len(html_checked or '')}")
        return html_checked


    def gen_verdict(title):
        logging.info(f"[SECTION] verdict start: {title}")

        # Keep FULL context for Verdict (safeguard)
        html = generate_value_for_money_verdict(
            title,
            hybrid_dataset,
            style_guide,
            keyword,
            country,
            recommended_product,
            cfg,
            product_whitelist=product_whitelist,
        )

        remember_generated_section(cfg, title, html)
        logging.info(f"[SECTION] verdict complete: {title} | chars={len(html or '')}")
        return html


    main_content_dict, subheading_content_dict = {}, {}

    for main_level, main_data in full_outline.items():
        main_title_text = f"{main_level} {main_data['title']}"
        heading_clean = main_data["title"]
        main_heading_html = f"<h2>{main_title_text}</h2>"

        if main_level == "1.0":
            content = gen_intro(heading_clean)
        elif "value for money" in heading_clean.lower() and "verdict" in heading_clean.lower():
            content = gen_verdict(heading_clean)
        else:
            if not main_data["subheadings"]:
                content = gen_detail(heading_clean)
            else:
                content = gen_overview(heading_clean)

        content = normalize_product_mentions(content, canonical_map)
        main_content_dict[main_level] = f"{main_heading_html}\n{content}"

        for sub_level, sub_data in main_data["subheadings"].items():
            sub_title_text = f"{sub_level} {sub_data['title']}"
            sub_heading_html = f"<h3>{sub_title_text}</h3>"
            sub_content = gen_detail(sub_data["title"])
            sub_content = normalize_product_mentions(sub_content, canonical_map)
            subheading_content_dict[sub_level] = f"{sub_heading_html}\n{sub_content}"

    generated_content = ""
    for main_level, main_data in full_outline.items():
        generated_content += main_content_dict.get(main_level, f"<h2>{main_level} {main_data['title']}</h2><p>Content not available.</p>") + "\n"
        for sub_level, sub_data in main_data["subheadings"].items():
            generated_content += subheading_content_dict.get(sub_level, f"<h3>{sub_level} {sub_data['title']}</h3><p>Content not available.</p>") + "\n"

    generated_content = normalize_product_mentions(generated_content, canonical_map)

    # --- PURPOSE-ONLY VALIDATION (unchanged, but category-agnostic) ---
    def _strip_html_to_text(s: str) -> str:
        return re.sub(r"<[^>]+>", " ", s)
    def _normalize_alnum_lower(s: str) -> str:
        return re.sub(r"[^0-9a-z]+", "", s.lower())
    def _extract_h2_sections(html: str):
        parts = re.split(r'(?is)(<h2\b[^>]*>[\s\S]*?</h2>)', html)
        sections, current = [], None
        for part in parts:
            if not part:
                continue
            if re.match(r'(?is)<h2\b[^>]*>[\s\S]*?</h2>', part):
                if current:
                    sections.append(current)
                title = re.sub(r'(?is)</?h2\b[^>]*>', '', part).strip()
                current = {"title": title, "body": ""}
            else:
                if current:
                    current["body"] += part
        if current:
            sections.append(current)
        return sections


    purpose_section_body = None
    for sec in _extract_h2_sections(generated_content):
        if "purpose" in sec["title"].lower():
            purpose_section_body = sec["body"]; break
    if purpose_section_body is None:
        for sec in _extract_h2_sections(generated_content):
            if sec["title"].lower().startswith("1.0 "):
                purpose_section_body = sec["body"]; break

    if purpose_section_body is None:
        msg = ("Post abandoned: Purpose section not found.\n"
               f"- Keyword: {keyword} ({country})\n"
               f"- Top pick: {recommended_product}\n"
               "Action: check outline generation and re-run.")
        with open(abandon_path, "w", encoding="utf-8") as f: f.write(msg)
        logging.error(msg)
        sys.exit(3)

    purpose_text_norm = _normalize_alnum_lower(_strip_html_to_text(purpose_section_body))
    top_pick_norm    = _normalize_alnum_lower(recommended_product)

    logging.debug(f"[Purpose check] top_pick_norm='{top_pick_norm}'")
    logging.debug(f"[Purpose check] purpose_contains={top_pick_norm in purpose_text_norm}")

    contains = (top_pick_norm in purpose_text_norm)
    logging.debug(f"[Purpose check] contains={contains}")

    # Record enforcement result
    try:
        audit["enforcement"]["purpose_contains_final_name"] = bool(contains)
    except Exception:
        pass

    if not contains:
        msg = ("Post abandoned: top pick not mentioned in Purpose section.\n"
               f"- Keyword: {keyword} ({country})\n"
               f"- Top pick: {recommended_product}\n"
               "Action: ensure the Purpose text explicitly contains the full product name and re-run.")
        with open(abandon_path, "w", encoding="utf-8") as f: f.write(msg)
        logging.error(msg)
        debug_out = os.path.join(output_dir, f"generated_blog_content_{country}_UNPUBLISHED.txt")
        with open(debug_out, "w", encoding="utf-8") as f: f.write(re.sub(r"</?p>", "", generated_content))
        try:
            audit["enforcement"]["abandoned_due_to_missing_purpose_mention"] = True
            save_top_pick_audit(audit, output_dir, country)
        except Exception:
            pass
        sys.exit(3)
    else:
        try:
            audit["enforcement"]["abandoned_due_to_missing_purpose_mention"] = False
        except Exception:
            pass

    # --- END PURPOSE VALIDATION ---
    

    # ... existing code that assembles `generated_content` ...
    # Reconcile canonical facts, vague claims and repeated buying conclusions
    # before the FAQ is generated from the approved article.
    generated_content = normalize_product_mentions(generated_content, canonical_map)
    try:
        generated_content, editorial_report = run_final_editorial_pass(
            generated_content,
            canonical_profile,
            keyword,
            recommended_product,
            cfg,
        )
    except Exception as exc:
        logging.exception("[EDITORIAL_PASS] Failed; retaining pre-pass content: %s", exc)
        generated_content = clean_generated_document_html(generated_content)
        editorial_report = {
            "enabled": True,
            "applied": False,
            "reason": f"exception:{exc}",
        }

# â”€â”€ NEW: Append FAQ section (unchanged logic below) â”€â”€
    try:
        generated_content = strip_existing_faq_sections(generated_content)
        def _next_top_number(headings_dict):
            try:
                tops = [int(lvl.split(".")[0]) for lvl in headings_dict.keys()]
                nxt = max(tops) + 1 if tops else 11
            except Exception:
                nxt = 11
            return f"{nxt}.0"
        faq_level_num = _next_top_number(full_outline)
        faqs = generate_faq_json(
            hybrid_dataset,
            keyword,
            recommended_product,
            style_guide,
            cfg,
            approved_article=generated_content,
        )
        if not any(qa.get("a", "").strip() for qa in faqs):
            logging.warning("FAQ generated with empty answers; rendering will include em dashes as fallback.")
        faq_html = render_faq_html(faqs, "Frequently Asked Questions", faq_level_num)
        faq_html = normalize_product_mentions(faq_html, canonical_map)
        if faq_html:
            generated_content = f"{generated_content}\n{faq_html}\n"
            logging.info("âœ… FAQ section appended (single instance).")
    except Exception as e:
        logging.error(f"âŒ FAQ generation failed: {e}")

    # Final deterministic HTML and canonical-conflict gate.
    generated_content = clean_generated_document_html(generated_content)
    initial_conflicts, initial_conflict_warnings = audit_canonical_conflicts(
        generated_content,
        canonical_profile,
    )
    conflict_repair_report = {
        "enabled": bool(canonical_controls.get("auto_repair_conflicts", True)),
        "attempted": False,
        "applied": False,
        "reason": "no_initial_conflicts",
        "initial_conflict_count": len(initial_conflicts),
    }
    if initial_conflicts:
        generated_content, conflict_repair_report = repair_unresolved_canonical_conflicts(
            generated_content,
            initial_conflicts,
            canonical_profile,
            keyword,
            recommended_product,
            cfg,
        )

    unresolved_conflicts, conflict_warnings = audit_canonical_conflicts(
        generated_content,
        canonical_profile,
    )
    consistency_report = {
        "audit_schema_version": 3,
        "canonical_profile_file": canonical_profile_path,
        "editorial_pass": editorial_report,
        "conflict_auto_repair": conflict_repair_report,
        "initial_unresolved_conflicts": initial_conflicts,
        "initial_conflict_warnings": initial_conflict_warnings,
        "unresolved_conflicts": unresolved_conflicts,
        "conflict_warnings": conflict_warnings,
        "invalid_empty_closing_tags": len(re.findall(r"</\s*>", generated_content)),
    }
    consistency_report_path = os.path.join(
        output_dir,
        f"content_consistency_audit_{country}.json",
    )
    with open(consistency_report_path, "w", encoding="utf-8") as report_file:
        json.dump(consistency_report, report_file, indent=2, ensure_ascii=False)
    audit["files"]["content_consistency_audit"] = consistency_report_path

    if conflict_warnings:
        logging.warning(
            "[CANONICAL_AUDIT] Recorded %d non-blocking contextual mention(s).",
            len(conflict_warnings),
        )

    if unresolved_conflicts and canonical_controls.get(
        "block_on_unresolved_conflicts",
        True,
    ):
        msg = (
            "Post abandoned: unresolved canonical fact conflicts remain.\n"
            f"- Keyword: {keyword} ({country})\n"
            f"- Conflicts: {len(unresolved_conflicts)}\n"
            f"- Report: {consistency_report_path}"
        )
        with open(abandon_path, "w", encoding="utf-8") as abandon_file:
            abandon_file.write(msg)
        unpublished_path = os.path.join(
            output_dir,
            f"generated_blog_content_{country}_UNPUBLISHED.txt",
        )
        with open(unpublished_path, "w", encoding="utf-8") as unpublished_file:
            unpublished_file.write(generated_content)
        logging.error(msg)
        save_top_pick_audit(audit, output_dir, country)
        sys.exit(5)

    # Enforce global list/table caps
    editorial_controls = cfg.get("editorial_controls") or {}
    before_all = _count_tables(generated_content)
    generated_content = fix_and_normalize_tables(generated_content)
    after_all = _count_tables(generated_content)
    logging.info(f"[document] Table normalize: {before_all} -> {after_all}")
    generated_content = enforce_list_table_limits(
        generated_content,
        max_lists=int(editorial_controls.get("max_lists_per_post", 10)),
        max_tables=int(editorial_controls.get("max_comparison_tables_per_post", 2)),
    )
    # Inject quick verdict BOX (with title above and CTA below)
    try:
        quick_line = generate_quick_verdict_tagline(
            hybrid_dataset,
            keyword,
            recommended_product,
            style_guide,
            cfg,
            approved_article=generated_content,
        )
        quick_box = render_quick_verdict_html(recommended_product, quick_line)
        quick_box = normalize_product_mentions(quick_box, canonical_map)
        quick_box = normalize_quick_verdict_classes(quick_box)
        generated_content = inject_quick_verdict_box(generated_content, quick_box, where="before_h2")
        logging.info("âœ… Quick Verdict box injected before first <h2> (with title + Amazon CTA).")
    except Exception as e:
        logging.error(f"âŒ Quick Verdict box placement failed: {e}")

    
    generated_content = clean_generated_document_html(generated_content)

    # Audit the complete article after all generated prose, including the Quick
    # Verdict. Accept a repair only when an independent re-audit reduces the
    # number of objective factual violations.
    semantic_controls = cfg.get("semantic_fact_audit") or {}
    initial_semantic_violations, semantic_audit_report = audit_semantic_claim_consistency(
        generated_content,
        canonical_profile,
        keyword,
        recommended_product,
        cfg,
    )
    unresolved_semantic_violations = list(initial_semantic_violations)
    semantic_reaudit_report = {
        "enabled": bool(semantic_controls.get("enabled", True)),
        "attempted": False,
        "reason": "not_needed",
        "violations": unresolved_semantic_violations,
    }
    semantic_repair_rounds = []
    max_semantic_repair_rounds = max(
        0,
        int(semantic_controls.get("max_repair_rounds", 3)),
    )
    original_audit_raw_path = cfg.get("_semantic_audit_raw_response_path")
    original_repair_raw_path = cfg.get("_semantic_repair_raw_response_path")

    def _round_artifact_path(path_value, round_number, label):
        path_value = str(path_value or "").strip()
        if not path_value:
            return ""
        root, ext = os.path.splitext(path_value)
        return f"{root}_round_{round_number}_{label}{ext or '.txt'}"

    for semantic_round in range(1, max_semantic_repair_rounds + 1):
        if (
            not unresolved_semantic_violations
            or semantic_controls.get("auto_repair", True) is False
        ):
            break

        cfg["_semantic_repair_raw_response_path"] = _round_artifact_path(
            original_repair_raw_path,
            semantic_round,
            "repair",
        )
        semantic_candidate, round_repair_report = repair_semantic_claim_conflicts(
            generated_content,
            unresolved_semantic_violations,
            canonical_profile,
            keyword,
            recommended_product,
            cfg,
        )
        round_record = {
            "round": semantic_round,
            "repair": round_repair_report,
            "input_violations": unresolved_semantic_violations,
        }
        if not round_repair_report.get("applied"):
            round_record["accepted"] = False
            round_record["reason"] = "repair_candidate_rejected"
            semantic_repair_rounds.append(round_record)
            break

        # Count progress against the exact passages this round was asked to fix.
        # A later audit may uncover different issues; that must not undo useful
        # corrections already made.
        candidate_visible = re.sub(
            r"\s+",
            " ",
            _visible_article_text(semantic_candidate).casefold(),
        ).strip()
        cited_passages = [
            re.sub(r"\s+", " ", str(item.get("passage") or "").casefold()).strip()
            for item in unresolved_semantic_violations
            if str(item.get("passage") or "").strip()
        ]
        remaining_cited = [
            passage for passage in cited_passages
            if passage and passage in candidate_visible
        ]
        passages_fixed = len(cited_passages) - len(remaining_cited)

        cfg["_semantic_audit_raw_response_path"] = _round_artifact_path(
            original_audit_raw_path,
            semantic_round,
            "reaudit",
        )
        candidate_violations, semantic_reaudit_report = audit_semantic_claim_consistency(
            semantic_candidate,
            canonical_profile,
            keyword,
            recommended_product,
            cfg,
        )
        round_record.update({
            "passages_fixed": passages_fixed,
            "remaining_original_passages": len(remaining_cited),
            "output_violations": candidate_violations,
            "reaudit": semantic_reaudit_report,
        })
        reaudit_reduced_violations = (
            len(candidate_violations) < len(unresolved_semantic_violations)
        )
        round_record["reaudit_reduced_violations"] = reaudit_reduced_violations
        semantic_progress = _semantic_repair_candidate_made_progress(
            unresolved_semantic_violations,
            candidate_violations,
            cited_passages,
            passages_fixed,
        )
        round_record["semantic_progress"] = semantic_progress
        if not semantic_progress:
            round_repair_report["applied"] = False
            round_repair_report["reason"] = "rejected_no_semantic_reduction"
            round_record["accepted"] = False
            round_record["reason"] = "no_semantic_reduction"
            semantic_repair_rounds.append(round_record)
            break

        generated_content = semantic_candidate
        unresolved_semantic_violations = candidate_violations
        round_repair_report["reason"] = "accepted_cited_passages_repaired"
        round_record["accepted"] = True
        round_record["reason"] = "cited_passages_repaired"
        semantic_repair_rounds.append(round_record)

    # The final model pass can expose a source-disputed weight after the bounded
    # repair budget is exhausted. Resolve this narrow, audit-provided case without
    # another whole-article rewrite, then independently re-audit it.
    if unresolved_semantic_violations:
        terminal_candidate, terminal_fixed_passages = (
            _deterministic_repair_disputed_weight_claims(
                generated_content, unresolved_semantic_violations
            )
        )
        if terminal_fixed_passages:
            terminal_violations, terminal_reaudit_report = (
                audit_semantic_claim_consistency(
                    terminal_candidate,
                    canonical_profile,
                    keyword,
                    recommended_product,
                    cfg,
                )
            )
            terminal_record = {
                "round": "terminal_deterministic_disputed_weight_repair",
                "input_violations": unresolved_semantic_violations,
                "passages_fixed": len(terminal_fixed_passages),
                "output_violations": terminal_violations,
                "reaudit": terminal_reaudit_report,
                "reaudit_reduced_violations": (
                    len(terminal_violations) < len(unresolved_semantic_violations)
                ),
            }
            if terminal_record["reaudit_reduced_violations"]:
                generated_content = terminal_candidate
                unresolved_semantic_violations = terminal_violations
                terminal_record["accepted"] = True
                terminal_record["reason"] = "disputed_weight_repaired"
            else:
                terminal_record["accepted"] = False
                terminal_record["reason"] = "no_semantic_reduction"
            semantic_repair_rounds.append(terminal_record)
    cfg["_semantic_audit_raw_response_path"] = original_audit_raw_path
    cfg["_semantic_repair_raw_response_path"] = original_repair_raw_path
    any_semantic_repair_applied = any(
        item.get("accepted") for item in semantic_repair_rounds
    )
    semantic_repair_report = {
        "enabled": bool(semantic_controls.get("auto_repair", True)),
        "attempted": bool(semantic_repair_rounds),
        "applied": any_semantic_repair_applied,
        "reason": (
            "completed"
            if not unresolved_semantic_violations
            else "max_rounds_or_no_progress"
            if semantic_repair_rounds
            else "no_initial_violations"
        ),
        "initial_violation_count": len(initial_semantic_violations),
        "final_violation_count": len(unresolved_semantic_violations),
        "max_rounds": max_semantic_repair_rounds,
        "rounds": semantic_repair_rounds,
    }

    # A late repair or Quick Verdict must not bypass the deterministic canonical
    # checks that ran earlier. If a semantic rewrite introduced a new literal
    # conflict, allow one narrowly scoped canonical repair and audit it again.
    final_canonical_conflicts, final_canonical_warnings = audit_canonical_conflicts(
        generated_content,
        canonical_profile,
    )
    final_canonical_repair_report = {
        "enabled": bool(canonical_controls.get("auto_repair_conflicts", True)),
        "attempted": False,
        "applied": False,
        "reason": "no_late_conflicts",
        "initial_conflict_count": len(final_canonical_conflicts),
    }
    if final_canonical_conflicts:
        generated_content, final_canonical_repair_report = (
            repair_unresolved_canonical_conflicts(
                generated_content,
                final_canonical_conflicts,
                canonical_profile,
                keyword,
                recommended_product,
                cfg,
            )
        )
        final_canonical_conflicts, final_canonical_warnings = audit_canonical_conflicts(
            generated_content,
            canonical_profile,
        )
        final_canonical_repair_report["remaining_conflict_count"] = len(
            final_canonical_conflicts
        )
    consistency_report.update({
        "semantic_fact_audit": semantic_audit_report,
        "semantic_fact_repair": semantic_repair_report,
        "semantic_fact_reaudit": semantic_reaudit_report,
        "unresolved_semantic_violations": unresolved_semantic_violations,
        "final_canonical_repair": final_canonical_repair_report,
        "final_unresolved_conflicts": final_canonical_conflicts,
        "final_conflict_warnings": final_canonical_warnings,
    })
    with open(consistency_report_path, "w", encoding="utf-8") as report_file:
        json.dump(consistency_report, report_file, indent=2, ensure_ascii=False)

    block_semantic = (
        unresolved_semantic_violations
        and semantic_controls.get("block_on_unresolved", True)
    )
    block_final_canonical = (
        final_canonical_conflicts
        and canonical_controls.get("block_on_unresolved_conflicts", True)
    )
    if block_semantic or block_final_canonical:
        msg = (
            "Post abandoned: final product-fact consistency checks failed.\n"
            f"- Keyword: {keyword} ({country})\n"
            f"- Semantic conflicts: {len(unresolved_semantic_violations)}\n"
            f"- Canonical conflicts: {len(final_canonical_conflicts)}\n"
            f"- Report: {consistency_report_path}"
        )
        with open(abandon_path, "w", encoding="utf-8") as abandon_file:
            abandon_file.write(msg)
        unpublished_path = os.path.join(
            output_dir,
            f"generated_blog_content_{country}_UNPUBLISHED.txt",
        )
        with open(unpublished_path, "w", encoding="utf-8") as unpublished_file:
            unpublished_file.write(generated_content)
        logging.error(msg)
        save_top_pick_audit(audit, output_dir, country)
        sys.exit(6)

    # Insert internal link slots based on the final structure
    generated_content = insert_internal_link_slots(generated_content)

    # â”€â”€ Build outline (+ add a pre-conclusion slot now if not present) â”€â”€
    outline = build_outline_from_html(generated_content)
    if "<!-- INTERNAL_LINK_SLOT:pre-conclusion -->" not in generated_content:
        # last safeguard
        generated_content += "\n<!-- INTERNAL_LINK_SLOT:pre-conclusion -->\n"

    # â”€â”€ Build metadata â”€â”€
    metadata = build_post_metadata(generated_content, keyword)
    # add a few useful fields for downstream
    metadata.update({
        "keyword": keyword,
        "country": country,
        "recommended_product": recommended_product,
        "word_count": len(re.findall(r'\w+', re.sub(r"<[^>]+>", " ", generated_content))),
        "outline_h2_count": len(outline)
    })

    # Final identity gate: neither metadata generation nor downstream affiliate
    # availability may silently change the model/size being reviewed.
    identity_controls = cfg.get("review_identity") or {}
    identity_tokens = keyword_product_identity_tokens(keyword)
    title_tokens = keyword_product_identity_tokens(str(metadata.get("title") or ""))
    visible_article = _visible_article_text(generated_content)
    identity_issues = []
    if not reviewed_product_matches_keyword(recommended_product, keyword, cfg):
        identity_issues.append("recommended_product_mismatch")
    if identity_tokens and not identity_tokens.issubset(title_tokens):
        identity_issues.append("metadata_title_mismatch")
    if recommended_product.casefold() not in visible_article.casefold():
        identity_issues.append("reviewed_product_missing_from_body")
    audit.setdefault("enforcement", {})["final_review_identity"] = {
        "keyword_identity_tokens": sorted(identity_tokens),
        "recommended_product": recommended_product,
        "metadata_title": metadata.get("title"),
        "issues": identity_issues,
    }
    if identity_issues and identity_controls.get("block_on_mismatch", True):
        msg = (
            "Post abandoned: reviewed-product identity changed before publication.\n"
            f"- Keyword: {keyword} ({country})\n"
            f"- Reviewed product: {recommended_product}\n"
            f"- Issues: {', '.join(identity_issues)}"
        )
        with open(abandon_path, "w", encoding="utf-8") as abandon_file:
            abandon_file.write(msg)
        unpublished_path = os.path.join(
            output_dir,
            f"generated_blog_content_{country}_UNPUBLISHED.txt",
        )
        with open(unpublished_path, "w", encoding="utf-8") as unpublished_file:
            unpublished_file.write(generated_content)
        logging.error(msg)
        save_top_pick_audit(audit, output_dir, country)
        sys.exit(7)

    # Save legacy .txt (backward compatible)
    try:
        cleaned_output = generated_content
        # convert closing paragraphs to newlines
        cleaned_output = re.sub(r'(?i)</p\s*>', '\n', cleaned_output)
        # remove opening paragraph tags
        cleaned_output = re.sub(r'(?i)<p\b[^>]*>', '', cleaned_output)
        # convert <br> to newlines as well
        cleaned_output = re.sub(r'(?i)<br\s*/?>', '\n', cleaned_output)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned_output)
        logging.info(f"âœ… Content saved to {output_path}")
    except Exception as e:
        logging.error(f"âŒ Error writing output file: {e}")


    # Save NEW metadata + structured payload for insert_amazon_links_images.py
    try:
        meta_path = os.path.join(output_dir, f"post_metadata_{country}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logging.info(f"âœ… Post metadata saved to {meta_path}")

        payload = {
            "body_html": generated_content,
            "metadata": metadata,
            "outline": outline
        }
        payload_path = os.path.join(output_dir, f"generated_post_payload_{country}.json")
        with open(payload_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logging.info(f"âœ… Structured payload saved to {payload_path}")
    except Exception as e:
        logging.error(f"âŒ Error writing metadata/payload: {e}")


    # Finally, persist the top-pick audit for this run
    try:
        save_top_pick_audit(audit, output_dir, country)
    except Exception:
        pass

if __name__ == "__main__":
    main()
