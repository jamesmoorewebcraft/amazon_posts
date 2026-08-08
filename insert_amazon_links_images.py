import os
import re
import csv
import json
import time
import math
import atexit
import random
import logging
import logging.handlers
import sys
import uuid
import logging
from typing import Any, Optional, Iterable
from pathlib import Path
from contextlib import contextmanager
from functools import wraps
from bs4 import BeautifulSoup, NavigableString
from html import escape
from amazon_creatorsapi import AmazonCreatorsApi, Country
import unicodedata
from rapidfuzz import fuzz
from datetime import datetime, timezone
from string import Template
from urllib.parse import urlencode
from internal_links import (
    DEEPSEEK_API_KEY_FILE,
    DEEPSEEK_MODEL,
    _local_semantic_embed,
    ensure_internal_link_slots,
    replace_internal_link_slots,
    load_site_index,
    log_deepseek_usage,
    upsert_site_index_entry,
    _get_deepseek_client,             # shared DeepSeek client helper
    _extract_keywords_from_html,     # reuse keyword extractor from internal_links
)


# -----------------------------
# Small string helpers
# -----------------------------

def first_in_list(haystack: str, needles: Iterable[str]) -> Optional[str]:
    """Return the first needle that appears as a substring of haystack (case-sensitive).

    Caller should lowercase haystack/needles if it wants case-insensitive matching.
    """
    if not haystack or not needles:
        return None
    for n in needles:
        if n and n in haystack:
            return n
    return None

_HEADING_PREFIX_RX = re.compile(r"^\s*(?:h[1-6]\s+|<\s*h[1-6][^>]*>\s*)", re.I)

def sanitize_top_pick_name(name: str | None) -> str:
    """
    Defensive cleanup for top-pick names coming from upstream extractors.
    Removes leading 'H1 ', 'H2 ' etc and any leading <h1 ...> tag fragments.
    """
    s = normalize_ws(name or "")
    if not s:
        return ""

    # Strip a leading heading marker like "H1 " / "h2 "
    s = _HEADING_PREFIX_RX.sub("", s).strip()

    # If any html slipped in, strip it
    if "<" in s and ">" in s:
        try:
            s = normalize_ws(strip_html(s))
        except Exception:
            pass

    return s.strip()

# =========================
# Paths & Config
# =========================
CREDENTIALS_FILE = Path("config/amazon_credentials.txt")
KEYWORD_FILE = Path("config/current_keyword.csv")

def _resolve_category_config_file() -> Path:
    """Resolve a shared category database without hard-coding a machine path."""
    env_path = (os.getenv("CATEGORY_CONFIG_FILE") or "").strip()
    if env_path:
        return Path(env_path).expanduser()

    pointer_file = Path("config/category_config_location.txt")
    if pointer_file.exists():
        configured = pointer_file.read_text(encoding="utf-8-sig").strip()
        if configured:
            configured_path = Path(configured).expanduser()
            if not configured_path.is_absolute():
                configured_path = pointer_file.parent / configured_path
            return configured_path.resolve()

    return Path("config/category_config.json")


CATEGORY_CONFIG_FILE = _resolve_category_config_file()
CATEGORY_ALIASES_FILE = Path(
    os.getenv("CATEGORY_ALIASES_FILE")
    or CATEGORY_CONFIG_FILE.with_name("category_aliases.json")
)  # optional

# === Internal-linking / embeddings ===
SITE_INDEX_FILE = Path("output/site_index.json")  # global index of existing posts

MAX_SEARCH_RESULTS = 10
MAX_ITEM_IMAGES_DEFAULT = 3  # visual cap for "catalog"/item sections

# Reject weak Amazon thumbnails that will blur when displayed in-post
MIN_IMAGE_WIDTH = 450
ALLOW_MEDIUM_FALLBACK = False
AMAZON_IMAGE_TARGET_SIZES = [1500, 1200, 1000, 800, 679, 500]

SIMILAR_ITEM_LABEL = "Similar on Amazon:"

PRICE_DISCLAIMER_TEXT = (
    "Product prices and availability are accurate as of the date/time indicated and are subject to change. "
    "Any price and availability information displayed on amazon.co.uk or amazon.com at the time of "
    "purchase will apply to the purchase of this product."
)


ANCHOR_TOKEN_RX = re.compile(r"__ANCHOR__(.*?)__HERE__", re.S)



# =========================
# Logging (Console + per-job files) with custom TRACE
# =========================
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

TRACE_LEVEL_NUM = 5
if not hasattr(logging, "TRACE"):
    logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")

def exact_model_token_match(query: str, title: str) -> str | None:
    """Return the first model token from query that appears in the title (normalized), else None.
    Normalization removes spaces and hyphens and lowercases.
    Examples: 'LV-H132' should match 'LV H132' or 'lvh132'.
    """
    q_tokens = extract_model_tokens(query or "")
    if not q_tokens:
        return None
    title_norm = re.sub(r"[\s\-_/]+", "", (title or "").lower())
    for tok in q_tokens:
        tok_norm = re.sub(r"[\s\-_/]+", "", tok.lower())
        if tok_norm and tok_norm in title_norm:
            return tok
    return None

def trace(self, message, *args, **kws):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kws)
logging.Logger.trace = trace  # type: ignore

def _iso_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

RUN_ID = os.getenv("RUN_ID") or uuid.uuid4().hex[:12]
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LEVEL = getattr(logging, LOG_LEVEL, logging.INFO)

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": _iso_now(),
            "level": record.levelname,
            "run_id": getattr(record, "run_id", RUN_ID),
            "trace_id": getattr(record, "trace_id", None),
            "step": getattr(record, "step", None),
            "msg": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        if hasattr(record, "extra_json") and isinstance(record.extra_json, dict):
            payload.update(record.extra_json)
        return json.dumps(payload, ensure_ascii=False)

class ContextAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        extra.setdefault("run_id", RUN_ID)
        kwargs["extra"] = extra
        return msg, kwargs
    def trace(self, msg, *args, **kwargs):
        kwargs.setdefault("extra", {})
        kwargs["extra"].setdefault("run_id", RUN_ID)
        if self.logger.isEnabledFor(TRACE_LEVEL_NUM):
            msg, kwargs = self.process(msg, kwargs)
            self.logger._log(TRACE_LEVEL_NUM, msg, args, **kwargs)

class EnsureFields(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_id"):
            record.run_id = RUN_ID
        if not hasattr(record, "step"):
            record.step = ""
        if not hasattr(record, "trace_id"):
            record.trace_id = ""
        if not hasattr(record, "extra_json"):
            record.extra_json = {}
        return True

root_logger = logging.getLogger()
root_logger.setLevel(TRACE_LEVEL_NUM)

if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
    console = logging.StreamHandler()
    console.setLevel(LEVEL)
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    console.addFilter(EnsureFields())
    root_logger.addHandler(console)


logger = logging.getLogger("amazon_image_extractor")
logger.setLevel(TRACE_LEVEL_NUM)
logger.propagate = True
log = ContextAdapter(logger, {"run_id": RUN_ID})

def with_step(step_name):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            trace_id = kwargs.pop("trace_id", None)
            extra = {"step": step_name, "trace_id": trace_id}
            t0 = time.perf_counter()
            log.debug(f"\u2192 Start: {step_name}", extra=extra)
            try:
                result = fn(*args, **kwargs)
                dt = (time.perf_counter() - t0) * 1000
                log.debug(f"Ã¢â€ Â End: {step_name} ({dt:.1f} ms)", extra=extra)
                return result
            except Exception as e:
                dt = (time.perf_counter() - t0) * 1000
                log.exception(f"Ã¢Å“â€“ Error in {step_name} after {dt:.1f} ms: {e}", extra=extra)
                raise
        return wrapper
    return decorator

@contextmanager
def time_block(step_name, trace_id=None, payload: dict | None = None):
    extra = {"step": step_name, "trace_id": trace_id, "extra_json": payload or {}}
    t0 = time.perf_counter()
    log.debug(f"\u2192 Start: {step_name}", extra=extra)
    try:
        yield
        dt = (time.perf_counter() - t0) * 1000
        log.debug(f"Ã¢â€ Â End: {step_name} ({dt:.1f} ms)", extra=extra)
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        log.exception(f"Ã¢Å“â€“ Error in {step_name} after {dt:.1f} ms: {e}", extra=extra)
        raise

# =========================
# Small helpers
# =========================
PRICE_FALLBACK_TEXT = "See Price"

def _display_or_see_price(value) -> str:
    """
    Return a safe display string for a price badge.
    Never returns blank/whitespace.
    """
    if value is None:
        return PRICE_FALLBACK_TEXT
    s = str(value).strip()
    return s if s else PRICE_FALLBACK_TEXT

def sanitize_top_pick_name(name: str) -> str:
    s = (name or "").strip()

    # strip leaked heading tokens like "H1 "
    s = re.sub(r"(?i)^\s*(?:h[1-6]\s+)+", "", s).strip()

    # strip common wrapper phrase we see in audits
    s = re.sub(r"(?i)^\s*additional\s+details\s+about\s+(?:the\s+)?", "", s).strip()

    return s

def log_kv(logger: logging.Logger, level: str, msg: str, **fields) -> None:
    """
    Log a message with structured fields via `extra`.

    IMPORTANT:
    Do NOT do: logger.info("...", trace_id=..., asin=...)
    because logging.Logger._log will raise: unexpected keyword argument 'trace_id'.

    Use log_kv(log, "info", "...", trace_id=..., asin=..., ...)
    """
    trace_id = fields.pop("trace_id", "")
    step = fields.pop("step", "")
    extra = {"trace_id": trace_id, "step": step, "extra_json": fields}
    fn = getattr(logger, level.lower(), logger.info)
    fn(msg, extra=extra)


def _safe_dump(obj, max_len: int = 8000) -> Optional[str]:
    """Best-effort JSON-ish dump for debugging (truncated)."""
    if obj is None:
        return None
    try:
        if hasattr(obj, "model_dump"):
            data = obj.model_dump()
        elif hasattr(obj, "dict"):
            data = obj.dict()
        elif hasattr(obj, "to_dict"):
            data = obj.to_dict()
        else:
            data = getattr(obj, "__dict__", str(obj))
        s = json.dumps(data, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)

    if len(s) > max_len:
        s = s[:max_len] + "..."
    return s


def _disclaimer_id_for_asin(asin: str, context: str | None = None) -> str:
    base = re.sub(r"[^A-Za-z0-9_-]+", "", asin or "")
    if not context:
        return f"price-disc-{base}"
    ctx = re.sub(r"[^A-Za-z0-9_-]+", "", str(context))
    return f"price-disc-{base}-{ctx[:24]}"



def _slugify(text: str) -> str:
    txt = re.sub(r"<[^>]+>", "", (text or "")).strip().lower()
    txt = re.sub(r"^\d+(\.\d+)?\s+", "", txt)
    txt = re.sub(r"[^\w\s-]", "", txt)
    txt = re.sub(r"\s+", "-", txt)
    return txt[:80].strip("-") or "section"

    
def strip_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

def normalize_quotes(s):
    return s.replace('Ã¢â‚¬Â³', '"').replace('Ã¢â‚¬Â', '"').replace('Ã¢â‚¬Å“', '"').replace("Ã¢â‚¬Ëœ", "'").replace("â€™", "'")
    
def normalize_ws(s: str) -> str:
    return " ".join((s or "").split())
    
FAILURE_REASON_FILE_NAME = "insert_amazon_failure_reason.txt"

class AmazonEligibilityError(RuntimeError):
    """Raised when Amazon accepts credentials but denies API eligibility."""


def _is_amazon_eligibility_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    markers = (
        "associatenoteligible",
        "does not currently meet the eligibility requirements",
        "accessdeniedexception",
    )
    return any(marker in msg for marker in markers)


def _amazon_eligibility_message(exc: Exception) -> str:
    return (
        "Amazon Creators API denied this account for Catalog/searchItems: "
        f"{exc}. OAuth credentials were accepted, but Amazon returned AssociateNotEligible. "
        "Check that the Creator credential, affiliate tag, marketplace/country, and API product are enabled for this exact account."
    )

def _write_failure_reason(output_dir: Path, reason: str) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / FAILURE_REASON_FILE_NAME).write_text(
            (reason or "").strip() + "\n",
            encoding="utf-8"
        )
    except Exception:
        pass


def _clear_failure_reason(output_dir: Path) -> None:
    try:
        p = output_dir / FAILURE_REASON_FILE_NAME
        if p.exists():
            p.unlink()
    except Exception:
        pass


def _count_products_with_images(product_data: list[dict]) -> int:
    count = 0
    for pd in (product_data or []):
        if str((pd or {}).get("img_url") or "").strip():
            count += 1
    return count

def normalize_terms(terms):
    """Normalize a set/list of terms used for simple string matching.

    Returns a list of normalized strings:
      - lowercased
      - trimmed
      - internal whitespace collapsed

    Accepts either an iterable of terms or a single string.
    """
    if not terms:
        return []
    if isinstance(terms, str):
        terms = [terms]
    out = []
    for t in terms:
        if not t:
            continue
        try:
            s = str(t)
        except Exception:
            continue
        s = normalize_ws(s.strip().lower())
        if s:
            out.append(s)
    return out

    
_BEST_PREFIX_RX = re.compile(r"^\s*best\s+", flags=re.I)

def normalize_title_for_matching(title: str, cfg: dict | None = None) -> str:
    """
    Produce a 'core' title suitable for matching/scoring.

    - Removes trailing separators.
    - Prefers the portion before common separators (bullet/features often follow).
    - Normalizes whitespace.
    """
    t = normalize_ws(title or "")
    if not t:
        return ""

    # Remove dangling separators like " -", " Ã¢â‚¬â€", ":" at the end
    try:
        t = _TITLE_TRAIL_RX.sub("", t).strip()
    except Exception:
        pass

    # Prefer "core" segment before feature bullets/noise
    for sep in (" - ", " Ã¢â‚¬â€œ ", " Ã¢â‚¬â€ ", " | ", " : "):
        if sep in t:
            t = t.split(sep, 1)[0].strip()
            break

    return normalize_ws(t)


def strip_best_prefix(name: str) -> str:
    # Remove leading "Best " only (case-insensitive), keep rest intact.
    return _BEST_PREFIX_RX.sub("", (name or "").strip())

    


def normalize_name(name):
    return unicodedata.normalize("NFKD", normalize_quotes(name)).encode('ascii', 'ignore').decode('utf-8').strip().lower()

def contains_any(text: str, words: list[str] | set[str]) -> bool:
    tl = (text or "").lower()
    return any(w.lower() in tl for w in (words or []))

def first_whole_term_match(text: str, terms: Iterable[str]) -> Optional[str]:
    """Return the first complete word/phrase match, never a substring in a word.

    Separators between phrase words may be whitespace, hyphens, underscores, or
    slashes. This is intended for exclusion terms, where a short term such as
    ``tag`` must match "luggage tag" but must not match "Patagonia".
    """
    haystack = (text or "").lower()
    for term in terms or []:
        raw = str(term or "").strip().lower()
        tokens = re.findall(r"[a-z0-9]+", raw)
        if not tokens:
            continue
        pattern = r"(?<![a-z0-9])" + r"[\s\-_/]+".join(
            re.escape(token) for token in tokens
        ) + r"(?![a-z0-9])"
        if re.search(pattern, haystack, flags=re.I):
            return raw
    return None


def contains_whole_term(text: str, terms: Iterable[str]) -> bool:
    """Return whether any term occurs as a complete word or phrase."""
    return first_whole_term_match(text, terms) is not None


def contains_all_whole_terms(text: str, terms: Iterable[str]) -> bool:
    """Return whether every supplied term occurs as a complete word/phrase."""
    cleaned = [str(term).strip() for term in (terms or []) if str(term).strip()]
    return bool(cleaned) and all(contains_whole_term(text, [term]) for term in cleaned)
    
# replace your existing pattern with this:
_TITLE_TRAIL_RX = re.compile(r"\s*(?:Ã¢â‚¬â€œ|Ã¢â‚¬â€|-|:)\s*$", flags=re.UNICODE)


def _find_asin_for_name(product_data: list[dict], name: str) -> str | None:
    key = normalize_name(name)
    for pd in product_data:
        if normalize_name(pd.get("name","")) == key and pd.get("asin"):
            return pd["asin"]
    return None

def shorten_title(title: str, max_len: int = 60) -> str:
    t = normalize_ws(title or "")
    if len(t) <= max_len:
        return t
    # avoid mid-word cut
    cut = t[:max_len].rsplit(" ", 1)[0].strip()
    return (cut or t[:max_len]).strip() + "..."

def truncate_tooltip(s: str, max_len: int = 160) -> str:
    """
    Truncate tooltip text to max_len characters.
    If truncated, end with '...' (not the unicode ellipsis).
    Avoid cutting in the middle of a word when possible.
    """
    t = normalize_ws(s or "")
    if len(t) <= max_len:
        return t

    hard = max_len - 3  # room for "..."
    if hard <= 0:
        return "..."

    cut = t[:hard]
    # prefer cutting at last space to avoid mid-word cut
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0].strip()

    if not cut:
        cut = t[:hard].strip()

    return cut + "..."

    
def normalize_similar_note(note: str) -> str:
    """
    Normalize substitute note text so it's shorter and avoids nested parentheses wording.
    """
    n = normalize_ws(note or "")

    # collapse older phrasings to the new label
    n = re.sub(r"(?i)^similar item\s*\(closest match on amazon\)\s*:?\s*$", "Similar on Amazon", n)
    n = re.sub(r"(?i)^similar item on amazon\s*:?\s*$", "Similar on Amazon", n)
    n = re.sub(r"(?i)^closest match on amazon\s*:?\s*$", "Similar on Amazon", n)

    return n or "Similar on Amazon"



def is_substitute_name(extracted: str, amazon_title: str, cfg: dict, threshold: int = 82) -> bool:
    # Use the same strict identity rule as candidate selection so shared model,
    # capacity, or family words cannot disguise a different product variant.
    identity_ok, _identity_details = requested_title_identity_match(extracted, amazon_title, cfg)
    return not identity_ok
    a_raw = normalize_ws(extracted or "")
    b_raw = normalize_ws(amazon_title or "")
    a = normalize_name(a_raw)
    b = normalize_name(b_raw)

    if not a or not b:
        return True

    # âœ… If extracted is contained in Amazon title, do NOT treat as substitute
    if a in b:
        return False

    # NEW: ignore generic tokens that often differ in Amazon titles
    def get_generic_tokens(cfg: dict) -> set[str]:
        toks = set()
        for k in ("generic_adjectives", "generic_tails", "include_keywords"):
            for s in (cfg.get(k) or []):
                if isinstance(s, str) and s.strip():
                    toks.update(re.findall(r"[A-Za-z0-9]+", s.lower()))
        # also ignore very common glue words
        toks.update({"and", "with", "for", "the", "a", "an"})
        return toks

    GENERIC = get_generic_tokens(cfg)
    a_tokens = [t for t in a.split() if len(t) >= 3 and t not in GENERIC]

    # If we removed everything, fall back to original token list
    if not a_tokens:
        a_tokens = [t for t in a.split() if len(t) >= 3]

    if a_tokens and sum(1 for t in a_tokens if t in b) / len(a_tokens) >= 0.80:
        return False

    if has_model_overlap(a_raw, b_raw) or same_model_family(a_raw, b_raw):
        return False

    return fuzz.token_sort_ratio(a, b) < threshold



def get_display_title(pd: dict, *, fallback: str = "") -> str:
    """
    Prefer the Amazon title (pd['label']) for anything user-facing.
    Fall back to extracted name (pd['name']), then to provided fallback.
    """
    if not isinstance(pd, dict):
        return normalize_ws(fallback)
    return normalize_ws(pd.get("label") or pd.get("name") or fallback)

def get_extracted_title(pd: dict, *, fallback: str = "") -> str:
    """
    The extracted name from the article (what we matched/replaced).
    """
    if not isinstance(pd, dict):
        return normalize_ws(fallback)
    return normalize_ws(pd.get("name") or fallback)

def get_short_display_title(pd: dict, max_len: int = 60, *, fallback: str = "") -> str:
    return shorten_title(get_display_title(pd, fallback=fallback), max_len=max_len)

_ANCHOR_SLOT_RX = re.compile(r"__ANCHOR__(.*?)__HERE__", flags=re.S)

def unwrap_anchor_slots(html: str) -> str:
    if not html:
        return html
    # Replace __ANCHOR__X__HERE__ -> X
    return _ANCHOR_SLOT_RX.sub(r"\1", html)

def refresh_missing_prices_second_pass(
    api,
    products: list[dict],
    *,
    base_url: str,
    tag: str,
    trace_id: str | None = None,
    cooldown_sec: float = 8.0,
    batch_size: int = 10,
    mini_passes: int = 1,                 # <-- NEW: set to 2 for two mini-passes
    mini_pass_cooldown_sec: float = 6.0,  # <-- NEW: cooldown between mini-passes
):
    """
    Second-pass to maximize prices: retry GetItems for ASINs whose price is missing.
    Updates products in-place.

    Tiny improvement:
      - `mini_passes` lets you do 2 small batched passes over the remaining-missing ASINs,
        with a cooldown between passes. This mimics "rerun again later" without hammering.
    """
    def _is_blank_price(p: dict) -> bool:
        return p.get("asin") and not str(p.get("price") or "").strip()

    missing = [p for p in products if _is_blank_price(p)]
    if not missing:
        return

    log.info(
        f"Ã°Å¸â€Â Second-pass price refresh for {len(missing)} ASIN(s) after cooldown",
        extra={"trace_id": trace_id, "extra_json": {"count": len(missing), "cooldown_sec": cooldown_sec}},
    )
    time.sleep(cooldown_sec)

    for mp in range(max(1, int(mini_passes))):
        # Recompute missing each mini-pass (because some will be filled)
        missing = [p for p in products if _is_blank_price(p)]
        if not missing:
            if log:
                log.info(
                    "âœ… All missing prices filled before completing mini-passes",
                    extra={"trace_id": trace_id, "extra_json": {"mini_pass": mp + 1}},
                )
            return

        by_asin: dict[str, list[dict]] = {}
        for p in missing:
            by_asin.setdefault(str(p["asin"]), []).append(p)

        asins = list(by_asin.keys())
        try:
            log.info(
                f"Ã°Å¸â€Â Mini-pass {mp+1}/{mini_passes}: refreshing {len(asins)} ASIN(s)",
                extra={"trace_id": trace_id, "extra_json": {"mini_pass": mp + 1, "asins_count": len(asins)}},
            )
        except Exception:
            pass

        for i in range(0, len(asins), batch_size):
            chunk = asins[i : i + batch_size]
            res = safe_get_items(
                api,
                chunk,
                trace_id=trace_id,
                retries=2,
                empty_items_retries=2,
            )

            items = getattr(res, "items", None) or []
            for it in items:
                asin2 = getattr(it, "asin", None) or getattr(it, "ASIN", None)
                if not asin2:
                    continue
                asin2 = str(asin2)

                price2 = _safe_get_display_price(it, fallback=None)
                if price2 and str(price2).strip():
                    for p in by_asin.get(asin2, []):
                        p["price"] = price2
                        # keep any existing timestamp logic you already use
                        if getattr(it, "price_ts", None):
                            p["price_ts"] = getattr(it, "price_ts")

        # Cooldown between mini-passes (only if we have another pass to do)
        if mp < mini_passes - 1:
            still_missing = sum(1 for p in products if _is_blank_price(p))
            if still_missing:
                delay = float(mini_pass_cooldown_sec)
                try:
                    log.info(
                        f"Ã¢ÂÂ¸Ã¯Â¸Â Mini-pass cooldown before next refresh: {delay:.1f}s",
                        extra={"trace_id": trace_id, "extra_json": {"still_missing": still_missing}},
                    )
                except Exception:
                    pass
                time.sleep(delay)



# =========================
# CTA engine
# =========================
from random import Random

CTA_CONFIG_DEFAULT = {
    "enable_emojis": False,
    "inline_trailer_cap": 2,
    "button_cta_cap": 999,
    "image_cta_cap": 999,
    "use_weighted_rotation": False,
    
    # NEW: turn off Ã¢â‚¬Å“Ã¢â‚¬â€œ see todayâ€™s deal \u2192 / check latest price \u2192Ã¢â‚¬Â
    "show_inline_trailer": False,
}

CTA_POOLS = {
    "inline_trailer": [
        "check latest price \u2192",
        "see todayâ€™s deal \u2192",
        "view on Amazon \u2192",
        "check price & reviews \u2192",
    ],
    "image": [
        "Check latest price on Amazon",
        "See todayâ€™s deal",
        "View on Amazon",
        "Check price & reviews",
    ],
    "button_primary": [
        "âœ… Check latest price on Amazon",
        "ðŸ”¥ See todayâ€™s deal on Amazon",
        "View on Amazon",
        "Check price & reviews on Amazon",
    ],
    "sticky": [
        "âœ… View latest price on Amazon",
        "ðŸ”¥ Check todayâ€™s deal on Amazon",
        "View on Amazon",
        "Check price on Amazon",
    ]
}

CTA_WEIGHTS = {
    "inline_trailer": [0.35, 0.30, 0.20, 0.15],
    "image":          [0.35, 0.30, 0.20, 0.15],
    "button_primary": [0.40, 0.30, 0.15, 0.15],
    "sticky":         [0.40, 0.30, 0.15, 0.15],
}

CTA_STRINGS = {
    "image":           "View on Amazon",
    "sticky":          "Check Price on Amazon",
}

INLINE_TRAILER_POOL = [
    "check latest price \u2192",
    "see todayâ€™s deal \u2192",
]

def _seeded_rng():
    return Random(int(RUN_ID[:8], 16))
    
# --- Inline trailer budget (cap how many "check latest price \u2192" trailers we output per post) ---
_INLINE_TRAILER_USED = 0

def _inline_trailer_budget(action: str, cfg: dict | None = None) -> bool:
    """
    Enforce a per-post cap on how many inline trailer CTAs we emit.

    action:
      - "can"     -> just check budget
      - "consume" -> check and increment if allowed
      - "reset"   -> reset counter (useful in tests)
    """
    global _INLINE_TRAILER_USED
    cfg = cfg or CTA_CONFIG_DEFAULT
    cap = int(cfg.get("inline_trailer_cap", CTA_CONFIG_DEFAULT["inline_trailer_cap"]))

    if action == "reset":
        _INLINE_TRAILER_USED = 0
        return True

    if cap <= 0:
        return False

    if action == "can":
        return _INLINE_TRAILER_USED < cap

    if action == "consume":
        if _INLINE_TRAILER_USED >= cap:
            return False
        _INLINE_TRAILER_USED += 1
        return True

    # Unknown action -> safest: don't emit
    return False


def pick_cta_label(kind: str, cfg: dict | None = None) -> str:
    cfg = cfg or CTA_CONFIG_DEFAULT
    pool = CTA_POOLS.get(kind, [])
    if not pool:
        return "View on Amazon"
    rng = _seeded_rng()
    if cfg.get("use_weighted_rotation") and kind in CTA_WEIGHTS:
        weights = CTA_WEIGHTS[kind]
        if len(weights) != len(pool) or sum(weights) <= 0:
            weights = [1/len(pool)] * len(pool)
        r = rng.random()
        c, idx = 0.0, 0
        for i, w in enumerate(weights):
            c += w
            if r <= c:
                idx = i; break
        label = pool[idx]
    else:
        label = rng.choice(pool)
    if not cfg.get("enable_emojis", True):
        label = label.replace("âœ… ", "").replace("ðŸ”¥ ", "")
    return label
    
def get_cta_cfg(cfg: dict) -> dict:
    # merge defaults + per-category overrides
    out = dict(CTA_CONFIG_DEFAULT)
    out.update(cfg.get("cta") or {})
    return out


# =========================
# Config loading
# =========================
def read_keyword_from_file():
    """
    Read the current job row from config/current_keyword.csv.

    Supports either:
      - headerless rows: keyword,country,site,category
      - headered CSV with columns: keyword,country,site,category

    Returns:
        (keyword, country, site, category)
    """
    try:
        with KEYWORD_FILE.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))

        rows = [[(c or "").strip() for c in row] for row in rows if any((c or "").strip() for c in row)]
        if not rows:
            return "", "", "", ""

        first = rows[0]
        normalized = [c.lower() for c in first]
        header_fields = {"keyword", "country", "site", "category"}
        has_header = len(first) >= 2 and any(c in header_fields for c in normalized)

        if has_header:
            if len(rows) < 2:
                return "", "", "", ""
            values = rows[1]
            data = {normalized[i]: (values[i] if i < len(values) else "") for i in range(len(normalized))}
            keyword = (data.get("keyword") or "").strip()
            country = (data.get("country") or "").strip().upper()
            site = (data.get("site") or "").strip()
            category = (data.get("category") or "").strip()
        else:
            cols = first + [""] * max(0, 4 - len(first))
            keyword = cols[0].strip()
            country = cols[1].strip().upper() if len(cols) > 1 else ""
            site = cols[2].strip() if len(cols) > 2 else ""
            category = cols[3].strip() if len(cols) > 3 else ""

        return keyword, country, site, category
    except Exception as e:
        log.error(f"Error reading keyword file: {e}", extra={"step":"config"})
    return "", "", "", ""

def load_deepseek_api_key():
    """Ensure a DeepSeek API key is available to the shared client."""
    if os.getenv("DEEPSEEK_API_KEY"):
        log.debug("Using DEEPSEEK_API_KEY from environment", extra={"step": "config"})
        return

    try:
        api_key = DEEPSEEK_API_KEY_FILE.read_text(encoding="utf-8").strip()
        if not api_key:
            raise ValueError("DeepSeek API key file is empty")
        os.environ["DEEPSEEK_API_KEY"] = api_key
        log.debug("Loaded DeepSeek API key from file", extra={"step": "config"})
    except FileNotFoundError:
        log.error(
            f"DeepSeek API key file not found: {DEEPSEEK_API_KEY_FILE}",
            extra={"step": "config"},
        )
        raise
    except Exception as e:
        log.error(f"Error loading DeepSeek API key: {e}", extra={"step": "config"})
        raise


def load_amazon_credentials():
    credentials = {}
    try:
        for line in CREDENTIALS_FILE.read_text().splitlines():
            if '=' in line:
                key, value = line.strip().split('=', 1)
                credentials[key.strip()] = value.strip()
        redacted_keys = {
            k: "***" if any(x in k.upper() for x in ("SECRET", "KEY", "CREDENTIAL")) else v
            for k, v in credentials.items()
        }
        log.debug("Loaded Amazon credentials", extra={"step":"config","extra_json":{"keys":sorted(redacted_keys.keys())}})
    except FileNotFoundError:
        log.error(f"Amazon credentials file not found: {CREDENTIALS_FILE}", extra={"step":"config"})
        raise
    except Exception as e:
        log.error(f"Error loading Amazon credentials: {e}", extra={"step":"config"})
        raise
    return credentials

def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Failed to parse JSON", extra={"step":"config","extra_json":{"path":str(path),"error":str(e)}})
        return {}

def deep_merge(base: dict, *overrides: dict) -> dict:
    out = json.loads(json.dumps(base))
    for src in overrides:
        for k, v in (src or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
    return out

_ACTIVE_CATEGORY_CONFIG: dict = {}


def configure_runtime_category(cfg: dict | None) -> None:
    global _ACTIVE_CATEGORY_CONFIG
    _ACTIVE_CATEGORY_CONFIG = dict(cfg or {})


def _identity_word_aliases(cfg: dict | None = None) -> dict[str, str]:
    active = cfg if isinstance(cfg, dict) else _ACTIVE_CATEGORY_CONFIG
    return {
        str(key).casefold(): str(value).casefold()
        for key, value in (active.get("identity_word_aliases") or {}).items()
        if str(key).strip() and str(value).strip()
    }


def _builtin_base_config():
    # Safety/editorial defaults must remain active even when
    # category_config_location.txt points to a shared legacy config.
    return {
        "section_images": {
            "required_sections": ["purpose", "verdict"],
            "allow_close_substitute": True,
            "substitute_caption_template": "Similar model shown: {amazon_title}",
            "max_substitute_images_per_post": 2,
        },
        "substitute_links": {
            "enabled": True,
            "scope": "top_pick_only",
            "max_per_post": 1,
        },
    }

def load_category_db() -> dict:
    return _load_json(CATEGORY_CONFIG_FILE)

def _patterns_match(keyword_lower: str, patterns):
    for pat in patterns:
        if isinstance(pat, str) and pat.startswith('re:'):
            try:
                if re.search(pat[3:], keyword_lower, flags=re.I):
                    return True
            except re.error:
                continue
        else:
            if str(pat).lower() in keyword_lower:
                return True
    return False

def resolve_category_from_aliases(keyword: str) -> str | None:
    aliases = _load_json(CATEGORY_ALIASES_FILE)
    if not aliases:
        return None
    kw = keyword.strip().lower()
    for category, patterns in aliases.items():
        if _patterns_match(kw, patterns):
            return category
    return None

def infer_category_from_keywords(keyword: str, db: dict) -> str | None:
    kw = keyword.strip().lower()
    candidates = [c for c in db.keys() if c.lower() != "default"]
    if not candidates:
        return None
    best, best_score = None, 0
    for cat in candidates:
        cfg = deep_merge(_builtin_base_config(), db.get("default", {}), db.get(cat, {}))
        score = 0
        for term in (cfg.get("include_keywords") or []):
            if term.lower() in kw:
                score += 1
        if score > best_score:
            best, best_score = cat, score
    return best if best_score > 0 else None

def build_config_for_category(category_key: str, db: dict) -> dict:
    return deep_merge(_builtin_base_config(), db.get("default", {}), db.get(category_key, {}))

def resolve_category(keyword: str, db: dict, explicit_category: str | None = None) -> str:
    explicit = (explicit_category or "").strip()
    if explicit:
        for key in db.keys():
            if key.lower() == explicit.lower():
                return key
        return explicit
    cat = resolve_category_from_aliases(keyword) or infer_category_from_keywords(keyword, db)
    if cat:
        return cat
    for key in db.keys():
        if key.lower() == keyword.strip().lower():
            return key
    return "default"

# =========================
# DeepSeek extraction
# =========================
@with_step("deepseek.extract_product_names")
def extract_product_names(text):
    prompt = (
        "From the following blog content, identify and extract the *most specific and complete* product names mentioned. "
        "Return product names only Ã¢â‚¬â€ do NOT include ranking/award prefixes like 'Best', 'Top', 'Winner', etc. "
        "For each product, provide its full, distinguishing name as it would appear on a retail site. "
        "Return a JSON array of strings, where each string is a unique product name. No explanation.\n\n" + text
    )
    try:
        client = _get_deepseek_client()
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3,
            extra_body={"thinking": {"type": "disabled"}},
        )
        log_deepseek_usage(
            response,
            label="insert_amazon:extract_product_names",
            requested_model=DEEPSEEK_MODEL,
        )
        content = response.choices[0].message.content
        json_str = re.search(r"\[.*?\]", content, re.DOTALL)
        if json_str:
            names = json.loads(json_str.group(0))
            log.info(
                f"DeepSeek extracted {(names)} {len(names)} candidate names",
                extra={"step": "deepseek.extract_product_names"},
            )
            return names
        log.error(
            "DeepSeek returned non-JSON content",
            extra={"step": "deepseek.extract_product_names"},
        )
        return []
    except Exception as e:
        log.error(f"DeepSeek extraction error: {e}", extra={"step":"deepseek.extract_product_names"})
        return []



# =========================
_EDITORIAL_PRODUCT_SUFFIX_RX = re.compile(
    r"(?i)\s*[:\-â€“â€”]+\s*(?:"
    r"performance\s+over\s+time|long[-\s]+term\s+performance|"
    r"test(?:ing)?\s+results?|hands[-\s]+on\s+results?|"
    r"key\s+specs?|specifications?|review\s+summary|"
    r"price\s+comparison|alternatives?|final\s+verdict"
    r")\s*$"
)
_EDITORIAL_PRODUCT_SUFFIX_IN_TEXT_RX = re.compile(
    r"(?i)\s*[:\-â€“â€”]+\s*(?:"
    r"performance\s+over\s+time|long[-\s]+term\s+performance|"
    r"test(?:ing)?\s+results?|hands[-\s]+on\s+results?|"
    r"key\s+specs?|specifications?|review\s+summary|"
    r"price\s+comparison|alternatives?|final\s+verdict"
    r")"
)


def clean_extracted_product_candidate(name: str) -> str:
    """Remove editorial wrappers without changing genuine model text."""
    s = normalize_ws(name or "").strip(" \t\r\n-â€“â€”:;,.|*")
    s = re.sub(r"(?i)^(?:best|top|winner|also\s+great)\s+(?:overall\s+)?[:\-]?\s*", "", s)
    s = _EDITORIAL_PRODUCT_SUFFIX_RX.sub("", s)
    return normalize_ws(s).strip(" \t\r\n-â€“â€”:;,.")


def normalize_editorial_product_suffixes_in_text(text: str) -> str:
    """Remove known editorial suffixes wherever a generated product name appears."""
    return _EDITORIAL_PRODUCT_SUFFIX_IN_TEXT_RX.sub("", text or "")


def _product_candidate_identity_key(name: str) -> tuple[str, ...]:
    """Order-insensitive identity key used only to collapse obvious aliases."""
    tokens = re.findall(r"[a-z0-9]+", normalize_name(clean_extracted_product_candidate(name)))
    aliases = _identity_word_aliases()
    normalized = [aliases.get(token, token) for token in tokens if token not in {"the", "a", "an"}]
    return tuple(sorted(normalized))


def extract_explicit_product_names(html: str) -> list[str]:
    """Extract deterministic product names from explicit HTML product fields/tables."""
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[str] = []

    for node in soup.select("[data-product]"):
        value = clean_extracted_product_candidate(node.get("data-product") or "")
        if value:
            candidates.append(value)

    for table in soup.select("table"):
        headers = [normalize_ws(th.get_text(" ", strip=True)).lower() for th in table.select("thead th")]
        if not headers or headers[0] not in {"product", "product name", "model", "item", "name"}:
            continue
        for row in table.select("tbody tr"):
            cells = row.find_all("td", recursive=False)
            if not cells:
                continue
            value = clean_extracted_product_candidate(cells[0].get_text(" ", strip=True))
            if 2 <= len(value.split()) <= 16 and len(value) <= 180:
                candidates.append(value)
    return candidates


def merge_product_name_candidates(*candidate_lists) -> list[str]:
    """Clean and de-duplicate LLM and deterministic product-name candidates."""
    result: list[str] = []
    seen: set[tuple[str, ...]] = set()
    for values in candidate_lists:
        for raw in values or []:
            if not isinstance(raw, str):
                continue
            name = clean_extracted_product_candidate(raw)
            if not name or len(name.split()) < 2:
                continue
            key = _product_candidate_identity_key(name)
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(name)
    return result

# Slug helpers
# =========================
def build_post_slug(keyword: str, country: str, metadata: dict | None = None) -> str:
    """
   Final post slug logic for site_index.json.

    Rules:
    - If keyword ends with 'review' or 'reviews', put the country BEFORE that word.
        e.g. "manta sleep mask review" (UK) -> "manta-sleep-mask-uk-review"
             "10kg carry on bag reviews" (UK) -> "10kg-carry-on-bag-uk-reviews"
    - Otherwise, append the country at the end.
        e.g. "manta sleep mask" (UK) -> "manta-sleep-mask-uk"
             "manta sleep mask review guide" (UK) -> "manta-sleep-mask-review-guide-uk"

    If metadata["slug"] is already provided, we trust and reuse it.
    """

    # 1) Prefer explicit slug from the generatorâ€™s payload if present
    if metadata and isinstance(metadata, dict) and metadata.get("slug"):
        return str(metadata["slug"]).strip()

    # 2) Build based on keyword + country according to the rules above
    kw = (keyword or "").strip().lower()
    c = (country or "").strip().lower()

    # Normalise: non-alphanumeric \u2192 space, then split into words
    kw = re.sub(r"[^a-z0-9]+", " ", kw)
    words = kw.split()

    # If no usable keyword, fall back to just the country (or "post")
    if not words:
        return c or "post"

    last = words[-1]

    if last in ("review", "reviews"):
        # Insert country *before* 'review'/'reviews'
        base = words[:-1]
        if c:
            parts = base + [c, last]
        else:
            parts = words[:]  # no country, just the original words
    else:
        # Country simply appended at the end
        parts = words + ([c] if c else [])

    slug = "-".join(parts)
    return slug or "post"


def _inject_links_into_section(section_html: str, section_title: str, links: list[dict]) -> str:
    if not links:
        return section_html
    bits = []
    for e in links:
        bits.append(_build_inline_link(e.get("url",""), e.get("title","")))
    related_html = (
        '<div class="related-inline" style="margin:.75em 0 0; font-size:.95em;">'
        'Related: ' + " Ã‚Â· ".join(bits) +
        '</div>'
    )
    paras = list(re.finditer(r'(?is)<p\b[^>]*>.*?</p>', section_html or ""))
    if paras:
        last = paras[-1]
        # add blank line after the Related block before whatever comes next (e.g., the next header)
        return section_html[:last.end()] + related_html + '\n\n' + section_html[last.end():]
    # if no <p> found, still ensure the blank line after the Related block
    return (section_html or "") + related_html + '\n\n'


# Reuse existing _aff_url if it's already defined
def _aff_url(base_url: str, asin: str | None, tag: str | None) -> str:
    if not asin or not isinstance(asin, str) or len(asin.strip()) < 8:
        return ""
    u = base_url.rstrip("/") + "/dp/" + asin.strip()
    return u + ("?" + urlencode({"tag": (tag or "").strip()})) if tag else u

# Case-insensitive class matchers
_PRODUCT_LINK_RX = re.compile(r"(?i)\bproduct-link\b")
_PRICE_LINK_RX   = re.compile(r"(?i)\bamazon-price-link\b")


# Matches 'check-price' class regardless of case, and works when class_ is a list
_CHECK_PRICE_CLASS_RX = re.compile(r"(?i)\bcheck-price\b")


# --- add near other regex helpers ---
_BEST_OVERALL_RX = re.compile(r"\bBest\s+Overall\b", re.I)


def fix_quick_verdict_links(
    html: str,
    top_pick_name: str | None,
    product_data: list[dict],
    base_url: str,
    tag: str,
) -> str:
    """
    Quick Verdict rewrites (robust):
      - Ensures "Best Overall: <Product>" is correctly linked (normal case)
      - Substitute case: product name becomes bold plain text (NOT clickable)
      - Inserts "Similar on Amazon: <short title>" before CTA (substitute case)
      - Rewires QV CTA button(s) to affiliate href + sets tooltip/title (160 chars)
      - âœ… Ensures CTA has data-aff="1" so any JS/CSS gating recognizes it
      - Prevents 'Element has no parent' errors by only inserting siblings safely
      - Idempotent: safe to run multiple times
    """
    if not html:
        return html

    soup = BeautifulSoup(html, "html.parser")

    # -----------------------------
    # Locate Quick Verdict container
    # -----------------------------
    qv = (
        soup.select_one(".quick-verdict-box")
        or soup.select_one(".quick-verdict")
        or soup.select_one("#quick-verdict")
        or soup.find(attrs={"class": re.compile(r"quick[-_ ]?verdict", re.I)})
        or soup.find(attrs={"id": re.compile(r"quick[-_ ]?verdict", re.I)})
    )
    if not qv:
        return str(soup)

    # -----------------------------
    # Resolve product data for top pick
    # -----------------------------
    resolved_pd = None
    resolved_asin = None

    if top_pick_name and isinstance(product_data, list):
        tp_norm = normalize_name(top_pick_name)
        for pd in product_data:
            if normalize_name(pd.get("name", "")) == tp_norm and pd.get("asin"):
                resolved_pd = pd
                resolved_asin = pd.get("asin")
                break

    href = _aff_url(base_url, resolved_asin, tag) if resolved_asin else ""
    is_sub = bool(resolved_pd and (resolved_pd.get("is_substitute") or resolved_pd.get("substitute_for")))

    top_pick_title_full = (
        get_display_title(resolved_pd, fallback=(top_pick_name or ""))
        if resolved_pd else
        (top_pick_name or "")
    )

    def _tooltip_160(txt: str) -> str:
        return truncate_tooltip(normalize_ws(txt or ""), 160)

    # -----------------------------
    # Helper: safely insert an arrow after a node (tag or NavigableString)
    # -----------------------------
    def _safe_insert_arrow_after(node, arrow=" \u2192"):
        if node is None:
            return
        parent = getattr(node, "parent", None)
        if parent is None:
            return

        nxt = node.next_sibling

        # If the next sibling already contains an arrow, do nothing
        if isinstance(nxt, NavigableString) and "\u2192" in str(nxt):
            return
        try:
            if getattr(nxt, "name", None) in {"span", "i"}:
                cls = " ".join(nxt.get("class", [])).lower()
                if "arrow" in cls or "dash" in cls:
                    # not exactly the arrow, but often used for separators
                    # still allow arrow insertion, but don't duplicate actual arrows
                    pass
        except Exception:
            pass

        # Insert arrow as text
        if hasattr(node, "insert_after"):
            node.insert_after(NavigableString(arrow))
        else:
            # NavigableString case: insert after via parent contents
            node.replace_with(NavigableString(str(node) + arrow))

    # -----------------------------
    # (0) Undo accidental "Similar on Amazon" strong from earlier runs
    # -----------------------------
    strong = qv.find("strong")
    if strong:
        if re.search(r"\bSimilar\s+on\s+Amazon\b", strong.get_text(" ", strip=True), re.I):
            strong.string = "Best Overall:"


    # Prefer the explicit QV top-pick link
    pl = qv.find("a", class_=lambda c: c and "qv-top-pick-link" in c.split())

    # Fallback if older markup exists
    if pl is None:
        pl = qv.find(
            lambda t: (
                getattr(t, "name", None) == "a"
                and t.get("class")
                and any(_PRODUCT_LINK_RX.search(cls or "") for cls in (t.get("class") or []))
                and not any(_PRICE_LINK_RX.search(cls or "") for cls in (t.get("class") or []))
                and not any(_CHECK_PRICE_CLASS_RX.search(cls or "") for cls in (t.get("class") or []))
            )
        )


    qv_created_summary_template = False

    def _qv_short_pick_name(txt: str) -> str:
        txt = normalize_ws(txt or "")
        txt = re.split(r"\s*,\s*", txt, maxsplit=1)[0]
        return shorten_title(txt, max_len=60)

    def _rebuild_summary_quick_verdict(summary_text: str, existing_pl=None):
        pick_link = existing_pl or soup.new_tag("a", href=href or "#")
        pick_link["class"] = ["qv-top-pick-link"]
        pick_link.clear()
        pick_link.string = _qv_short_pick_name(top_pick_name)

        best_label = soup.new_tag("strong")
        best_label.string = "\U0001F3C6 Best Overall:"

        summary = soup.new_tag("span")
        summary["class"] = ["qv-verdict-summary"]
        summary.string = " " + normalize_ws(summary_text) if summary_text else ""

        cta_wrap = soup.new_tag("span")
        cta_wrap["class"] = ["quick-verdict-cta"]

        cta = soup.new_tag("a", href=href or "#")
        cta["class"] = ["check-price"]
        cta.string = "\U0001F525 See Today\u2019s Price on Amazon"
        cta_wrap.append(cta)

        qv.clear()
        qv.append(best_label)
        qv.append(NavigableString(" "))
        qv.append(pick_link)
        qv.append(summary)
        qv.append(NavigableString(" "))
        qv.append(cta_wrap)
        return pick_link


    # If the source report had no top pick, there may be a Best Overall label
    # but no product node for the fallback name to replace. Create that node so
    # the normal Quick Verdict wiring below can populate href/title/price attrs.
    if pl is None and top_pick_name:
        best_overall_label = qv.find(
            lambda t: (
                getattr(t, "name", None) in {"strong", "b"}
                and _BEST_OVERALL_RX.search(t.get_text(" ", strip=True) or "")
            )
        )
        quick_verdict_label = qv.find(
            lambda t: (
                getattr(t, "name", None) in {"strong", "b"}
                and re.search(r"\bQuick\s+Verdict\b", t.get_text(" ", strip=True) or "", re.I)
            )
        )
        if best_overall_label:
            nxt = best_overall_label.next_sibling
            while isinstance(nxt, NavigableString) and not normalize_ws(str(nxt)):
                remove_me = nxt
                nxt = nxt.next_sibling
                remove_me.extract()

            if isinstance(nxt, NavigableString):
                text = str(nxt)
                arrow_match = re.search(r"\u2192|&rarr;", text)
                if arrow_match:
                    tail = text[arrow_match.end():].strip()
                    if tail:
                        nxt.replace_with(NavigableString(" " + tail))
                    else:
                        remove_me = nxt
                        nxt = nxt.next_sibling
                        remove_me.extract()
                elif normalize_ws(text):
                    remove_me = nxt
                    nxt = nxt.next_sibling
                    remove_me.extract()

            pl = soup.new_tag("a", href=href or "#")
            pl["class"] = ["qv-top-pick-link"]
            pl.string = _qv_short_pick_name(top_pick_name)
            best_overall_label.insert_after(pl)
            best_overall_label.insert_after(NavigableString(" "))
        elif quick_verdict_label:
            summary_text = normalize_ws(qv.get_text(" ", strip=True))
            summary_text = re.sub(r"^\s*Quick\s+Verdict\s*:?\s*", "", summary_text, flags=re.I)

            pl = _rebuild_summary_quick_verdict(summary_text)
            qv_created_summary_template = True

    if pl is not None and top_pick_name and qv.select_one(".qv-top-pick-label"):
        summary_text = normalize_ws(qv.get_text(" ", strip=True))
        summary_text = re.sub(r"^\s*Quick\s+Verdict\s*:?\s*", "", summary_text, flags=re.I)
        summary_text = re.split(r"\bTop\s+pick\s*:", summary_text, maxsplit=1, flags=re.I)[0].strip()
        pl = _rebuild_summary_quick_verdict(summary_text, existing_pl=pl)
        qv_created_summary_template = True

    if pl and top_pick_name:
        visible_txt = _qv_short_pick_name(top_pick_name) if qv_created_summary_template else normalize_ws(top_pick_name)
        after_node = pl  # the node we will insert arrow after

        if is_sub:
            # Replace with bold text (not clickable)
            b = soup.new_tag("strong")
            b.string = visible_txt
            pl.replace_with(b)
            after_node = b
        else:
            if href:
                # Force <a> with correct attributes
                if pl.name != "a":
                    a = soup.new_tag("a", href=href)
                    a["class"] = list(pl.get("class") or [])
                    a.string = visible_txt
                    pl.replace_with(a)
                    pl = a

                pl["href"] = href
                pl["target"] = "_blank"
                pl["rel"] = ["nofollow", "noopener", "sponsored"]

                # ensure aff-inline + qv-top-pick-link
                cur = list(pl.get("class") or [])
                cur_lc = [c.lower() for c in cur]
                if "aff-inline" not in cur_lc:
                    cur.append("aff-inline")
                if "qv-top-pick-link" not in cur:
                    cur.append("qv-top-pick-link")
                pl["class"] = cur

                pl["data-aff"] = "1"
                tt = _tooltip_160(top_pick_title_full or visible_txt)
                if tt:
                    pl["title"] = tt
                pl["aria-label"] = "Open Amazon product page for " + (top_pick_title_full or visible_txt)

                cls_str = " ".join(pl.get("class") or []).lower()
                if _PRICE_LINK_RX.search(cls_str) or _CHECK_PRICE_CLASS_RX.search(cls_str) or "amazon-price-link" in cls_str:
                    # Never clear the CTA/price link
                    pass
                else:
                    pl.clear()
                    pl.string = visible_txt

                after_node = pl
            else:
                # No href: replace with plain text
                txt_node = NavigableString(visible_txt)
                pl.replace_with(txt_node)
                after_node = txt_node

        # Ensure only one arrow after (remove any immediate duplicate arrow text)
        try:
            nxt = after_node.next_sibling
            if isinstance(nxt, NavigableString) and "\u2192" in str(nxt):
                # keep one; remove extras if multiple
                # (simple: collapse to single arrow)
                nxt.replace_with(NavigableString(" \u2192"))
        except Exception:
            pass

        _safe_insert_arrow_after(after_node, arrow=" \u2192")

    # -----------------------------
    # (A) Remove existing similar blocks (idempotent)
    # -----------------------------
    for node in qv.select(".qv-similar-item"):
        node.decompose()

    for sp in qv.find_all("span"):
        t = sp.get_text(" ", strip=True)
        if re.search(r"Similar item\s*\(closest match on Amazon\)\s*:", t, re.I):
            sp.decompose()

    # -----------------------------
    # (B) Insert "Similar on Amazon: ..." (substitute case)
    # -----------------------------
    alt_title_full = ""
    alt_title_short = ""

    if is_sub and resolved_pd:
        alt_title_full = get_display_title(resolved_pd, fallback="")
        alt_title_short = get_short_display_title(resolved_pd, max_len=60, fallback=alt_title_full)
        alt_tt = _tooltip_160(alt_title_full or alt_title_short)

        cta_container = qv.select_one(".quick-verdict-cta") or qv.select_one(".qv-cta")

        sim_wrap = soup.new_tag("div")
        sim_wrap["class"] = ["qv-similar-item"]

        label = soup.new_tag("span")
        label.string = SIMILAR_ITEM_LABEL + " "
        sim_wrap.append(label)

        if href:
            a = soup.new_tag("a", href=href)
            a["target"] = "_blank"
            a["rel"] = ["nofollow", "noopener", "sponsored"]
            a["class"] = ["aff-inline", "qv-similar-link"]
            a["data-aff"] = "1"
            if alt_tt:
                a["title"] = alt_tt
            a["aria-label"] = "Open Amazon product page for " + (alt_title_full or alt_title_short or "")
            a.string = alt_title_short or alt_title_full or "View on Amazon"
            sim_wrap.append(a)
        else:
            sim_wrap.append(NavigableString(alt_title_short or alt_title_full or ""))

        if cta_container:
            cta_container.insert_before(sim_wrap)
        else:
            qv.append(sim_wrap)


    # -----------------------------
    # (C) Rewire CTA links (IMPORTANT: set data-aff="1")
    # -----------------------------
    cta_links = []
    for a in qv.find_all("a"):
        classes = a.get("class") or []
        cls_str = " ".join(classes).lower() if isinstance(classes, list) else str(classes).lower()

        # Match your CTA link classes
        if _PRICE_LINK_RX.search(cls_str) or _CHECK_PRICE_CLASS_RX.search(cls_str) or "cta" in cls_str:
            cta_links.append(a)

    # Fallback: first link inside CTA container
    if not cta_links:
        wrap = qv.select_one(".quick-verdict-cta") or qv.select_one(".qv-cta") or qv
        first_a = wrap.find(
            lambda t: (
                getattr(t, "name", None) == "a"
                and "qv-top-pick-link" not in (t.get("class") or [])
                and "qv-similar-link" not in (t.get("class") or [])
                and "aff-price-link" not in (t.get("class") or [])
            )
        )
        if first_a:
            cta_links.append(first_a)

    cta_tooltip_source = (alt_title_full if (is_sub and alt_title_full) else top_pick_title_full) or ""
    cta_tt = _tooltip_160(cta_tooltip_source)

    # âœ… Price source for QV CTA (top pick or substitute item)
    #
    # IMPORTANT: Creators API can occasionally return no offer price for an ASIN
    # (or you may only have a CTA href but no resolved_pd). In those cases we
    # still insert a price *placeholder* (Ã¢â‚¬â€) so front-end price scripts/CSS have
    # a stable target (matching create_affiliate_link()).
    qv_price_display_default = (resolved_pd or {}).get("price")
    qv_price_ts_default = (resolved_pd or {}).get("price_ts")

    def _extract_asin_from_href(u: str) -> str | None:
        if not u:
            return None
        m = (
            re.search(r"/dp/([A-Z0-9]{10})", u, re.I)
            or re.search(r"/gp/product/([A-Z0-9]{10})", u, re.I)
            or re.search(r"(?:[?&]asin=)([A-Z0-9]{10})", u, re.I)
        )
        return m.group(1).upper() if m else None

    def _pd_for_asin(a: str | None) -> dict | None:
        if not a or not isinstance(product_data, list):
            return None
        a0 = a.strip().upper()
        return next(
            (pd for pd in product_data if str(pd.get("asin") or "").strip().upper() == a0),
            None,
        )

    for a in cta_links:
        # Pick the best available href:
        # 1) our computed affiliate href (if present)
        # 2) otherwise the CTA's existing href
        existing_href = a.get("href") or ""
        effective_href = href or existing_href

        # Resolve an ASIN even if we couldn't match top_pick_name (e.g. substitute CTA)
        effective_asin = resolved_asin or _extract_asin_from_href(effective_href)
        effective_pd = resolved_pd or _pd_for_asin(effective_asin)

        # Prefer price for the effective ASIN; otherwise fall back to defaults
        qv_price_display = (effective_pd or {}).get("price") or qv_price_display_default
        qv_price_ts = (effective_pd or {}).get("price_ts") or qv_price_ts_default

        # One-line debug log: confirms which path we used on a bad run
        log.warning(
            "QV CTA path=%s asin=%s price=%r",
            "resolved" if href else "cta",
            effective_asin,
            qv_price_display,
        )

        if cta_tt:
            a["title"] = cta_tt

        # Always ensure affiliate/CTA attributes, even if href wasn't resolved from top_pick_name
        if effective_asin:
            aff = _aff_url(base_url, effective_asin, tag)
            if aff:
                a["href"] = aff

        a["target"] = "_blank"
        rel = set((a.get("rel") or [])) | {"nofollow", "noopener", "sponsored"}
        a["rel"] = sorted(rel)

        # âœ… activation marker
        a["data-aff"] = "1"

        cur_classes = list(a.get("class") or [])
        if "aff-inline" not in [c.lower() for c in cur_classes]:
            cur_classes.append("aff-inline")
        a["class"] = cur_classes

        a["aria-label"] = (
            "Open Amazon product page for a similar item (top pick not currently available)"
            if is_sub
            else "Open Amazon product page"
        )
        if is_sub:
            a.clear()
            a.append(NavigableString("See Similar Model on Amazon"))

        # âœ… Re-insert/refresh the price badge next to the CTA (idempotent)
        parent = a.parent
        if parent is not None:
            # remove any existing price/meta/disclaimer in the same CTA container
            for sp in parent.find_all("span", class_=lambda c: c and "aff-price" in c.split()):
                sp.decompose()
            for sp in parent.find_all("span", class_=lambda c: c and "aff-price-meta" in c.split()):
                sp.decompose()
            for sp in parent.find_all("span", class_=lambda c: c and "price-disclaimer" in c.split()):
                sp.decompose()

            # also remove any existing Details link in this CTA container (belt + braces)
            for al in parent.find_all("a", class_=lambda c: c and "price-disclaimer-toggle" in c.split()):
                al.decompose()

            
            # Always insert a price span when we have an ASIN (even if price is missing \u2192 placeholder)
            if effective_asin:
                shown_price = _display_or_see_price(qv_price_display)  # should return "See Price" for missing

                # --- 0) Remove arrow from the CTA link text to avoid double arrows ---
                # Your CTA <a> often contains "See Today's Price on Amazon \u2192"
                # We keep ONE arrow (the arrow span), so strip any trailing arrow character from the CTA anchor text.
                try:
                    # Only change visible text; don't touch attributes
                    txt = a.get_text(strip=False)
                    if txt and "\u2192" in txt:
                        # remove a trailing arrow (and surrounding spaces)
                        new_txt = re.sub(r"\s*\u2192\s*$", "", txt)
                        if new_txt != txt:
                            a.clear()
                            a.append(NavigableString(new_txt.strip()))
                except Exception:
                    pass
                # ---------------------------------------------------------------------

                # arrow (NOT clickable now)
                arrow = soup.new_tag("span")
                arrow["class"] = ["aff-price-arrow"]
                arrow["aria-hidden"] = "true"
                arrow.string = " \u2192"

                # price span (this will be clickable)
                sp = soup.new_tag("span")
                sp["class"] = ["aff-price"]
                sp["data-asin"] = str(effective_asin)
                if qv_price_ts:
                    sp["data-price-ts"] = str(qv_price_ts)
                sp.string = str(shown_price)

                # If fallback "See Price", make it black (and optionally bold)
                # This keeps it readable even though it sits inside a link.
                if str(shown_price).strip().lower() == "see price":
                    sp["class"].append("aff-price-fallback")
                    # optional: inline style if you don't want to add CSS
                    sp["style"] = "color:#000; font-weight:600;"

                # details + disclaimer (context-safe to avoid collisions with body)
                disc_id = _disclaimer_id_for_asin(str(effective_asin), context="quick-verdict")

                meta = soup.new_tag("span")
                meta["class"] = ["aff-price-meta"]
                meta["style"] = "font-size: 0.9rem;"
                meta.append(NavigableString(" ("))

                details = soup.new_tag("a")
                details["class"] = ["price-disclaimer-toggle"]
                details["href"] = f"#{disc_id}"
                details["aria-controls"] = disc_id
                details["aria-expanded"] = "false"
                details.string = "Details"
                meta.append(details)
                meta.append(NavigableString(")"))

                disc = soup.new_tag("span")
                disc["id"] = disc_id
                disc["class"] = ["price-disclaimer"]
                disc["hidden"] = ""
                disc.string = PRICE_DISCLAIMER_TEXT

                # --- NEW: make ONLY the price clickable (arrow stays outside link) ---
                price_link = soup.new_tag("a")
                price_link["class"] = ["aff-inline", "aff-price-link"]
                price_link["data-aff"] = "1"

                # Reuse the same URL/attrs as the existing Quick Verdict CTA link
                price_link["href"] = a.get("href", "")
                price_link["target"] = a.get("target", "_blank")
                price_link["rel"] = a.get("rel", ["nofollow", "noopener", "sponsored"])
                if a.get("title"):
                    price_link["title"] = a.get("title")
                if a.get("aria-label"):
                    price_link["aria-label"] = a.get("aria-label")

                # Put ONLY the price span inside the clickable link
                price_link.append(sp)
                # --------------------------------------------------------------------

                # Insert in the correct order:
                # CTA link \u2192 space \u2192 arrow \u2192 clickable price \u2192 meta \u2192 disclaimer
                a.insert_after(disc)
                a.insert_after(meta)
                a.insert_after(price_link)
                a.insert_after(arrow)
                a.insert_after(NavigableString(" "))



    return str(soup)



# =========================
# Amazon helpers & selection
# =========================
MODEL_TOKEN_RE = re.compile(
    r"\b((?:[A-Z]{1,6}\d{1,6}[A-Z]{0,3})|"
    r"(?:\d+(?:\.\d+)?(?:ML|L|KG|G|MM|CM|M|IN|GB|TB|W|V|HZ)))\b", re.I
)

def extract_model_tokens(s: str) -> set[str]:
    return {t.upper() for t in MODEL_TOKEN_RE.findall(s or "")}

def has_model_overlap(query: str, title: str) -> bool:
    q = extract_model_tokens(query); t = extract_model_tokens(title)
    if not q or not t: return False
    if q & t: return True
    for qt in q:
        for tt in t:
            if tt.startswith(qt) or qt.startswith(tt):
                return True
    return False

def _get_browse_node(it) -> str:
    """Return all display_name(s) joined, not just the first, to improve matching."""
    try:
        if it and it.browse_node_info and it.browse_node_info.browse_nodes:
            names = []
            for bn in it.browse_node_info.browse_nodes:
                if getattr(bn, "display_name", None):
                    names.append(bn.display_name.strip())
            return " > ".join(names)
    except Exception:
        pass
    return ""
    
def log_json(logger, level: str, msg: str, trace_id: str | None = None, **extra_json):
    getattr(logger, level)(
        msg,
        extra={
            "trace_id": trace_id,
            "extra_json": extra_json,
        },
    )



def _extract_primary_image_url_simple(item) -> str | None:
    try:
        p = getattr(getattr(item.images, "primary", None), "large", None) \
            or getattr(getattr(item.images, "primary", None), "medium", None) \
            or getattr(getattr(item.images, "primary", None), "small", None)
        return getattr(p, "url", None)
    except Exception:
        return None

def infer_amazon_image_dimensions_from_url(url: str) -> tuple[Optional[int], Optional[int], Optional[str]]:
    """Best-effort parse of Amazon size markers like _SL500_, _SX679_, _SY879_."""
    if not url:
        return (None, None, None)
    s = str(url)
    m_sl = re.search(r"_SL(\d+)_", s, re.I)
    m_sx = re.search(r"_SX(\d+)_", s, re.I)
    m_sy = re.search(r"_SY(\d+)_", s, re.I)

    width = int(m_sx.group(1)) if m_sx else None
    height = int(m_sy.group(1)) if m_sy else None

    if m_sl:
        side = int(m_sl.group(1))
        if width is None:
            width = side
        if height is None:
            height = side
        return (width, height, m_sl.group(0))

    if width is not None or height is not None:
        token = (m_sx.group(0) if m_sx else "") + (m_sy.group(0) if m_sy else "")
        return (width, height, token or None)

    return (None, None, None)


def get_image_dimensions(url: str, timeout: float = 10.0) -> tuple[Optional[int], Optional[int], str]:
    """
    Fetch an image URL and return pixel dimensions as (width, height, status).

    status values:
      - measured: dimensions were read from the image bytes
      - inferred_from_url: dimensions were inferred from Amazon URL markers
      - unknown: dimensions could not be determined
    """
    if not url:
        return (None, None, "unknown")
    try:
        import requests
        from io import BytesIO
        from PIL import Image

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Referer": "https://www.amazon.co.uk/",
        }
        resp = requests.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()
        with Image.open(BytesIO(resp.content)) as im:
            w, h = im.size
            return (w, h, "measured")
    except Exception as e:
        iw, ih, token = infer_amazon_image_dimensions_from_url(url)
        if iw is not None or ih is not None:
            log.info(
                "Using Amazon URL size marker fallback",
                extra={
                    "step": "amazon.image_dimensions",
                    "extra_json": {"url": url, "width": iw, "height": ih, "token": token, "error": str(e)},
                },
            )
            return (iw, ih, "inferred_from_url")
        log.warning(
            "Could not determine image dimensions",
            extra={"step": "amazon.image_dimensions", "extra_json": {"url": url, "error": str(e)}},
        )
        return (None, None, "unknown")


def generate_amazon_size_variants(url: str) -> list[tuple[int, str]]:
    """Generate Amazon image URL variants from _SL/_SX/_SY markers, largest first."""
    if not url:
        return []
    variants: list[tuple[int, str]] = []
    seen: set[str] = set()
    for s in AMAZON_IMAGE_TARGET_SIZES:
        new_url = re.sub(r"_S(?:L|X|Y)\d+_", f"_SL{s}_", url)
        if new_url == url and f"_SL{s}_" not in url:
            continue
        if new_url not in seen:
            seen.add(new_url)
            variants.append((s, new_url))
    if url not in seen:
        iw, ih, _ = infer_amazon_image_dimensions_from_url(url)
        declared = max(iw or 0, ih or 0, 0)
        variants.append((declared, url))
    variants.sort(key=lambda t: t[0], reverse=True)
    return variants


def estimate_image_content_ratio(url: str, timeout: float = 10.0) -> Optional[float]:
    """Estimate how much of an image contains non-background content (higher = less padded)."""
    if not url:
        return None
    try:
        import requests
        from io import BytesIO
        from PIL import Image, ImageChops

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Referer": "https://www.amazon.co.uk/",
        }
        resp = requests.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()
        with Image.open(BytesIO(resp.content)) as im:
            rgb = im.convert("RGB")
            bg = Image.new("RGB", rgb.size, rgb.getpixel((0, 0)))
            diff = ImageChops.difference(rgb, bg)
            bbox = diff.getbbox()
            if not bbox:
                return 0.0
            left, top, right, bottom = bbox
            content_area = max(0, right - left) * max(0, bottom - top)
            total_area = rgb.size[0] * rgb.size[1]
            if total_area <= 0:
                return None
            return max(0.0, min(1.0, content_area / total_area))
    except Exception:
        return None


def resolve_best_amazon_image_candidate(url: str, min_width: int = MIN_IMAGE_WIDTH) -> tuple[Optional[str], Optional[int], Optional[int], str, Optional[float]]:
    """Try Amazon size variants and return the best candidate without letting image probing block the pipeline."""
    best = None
    fallback = None
    for declared_size, test_url in generate_amazon_size_variants(url):
        try:
            w, h, dim_status = get_image_dimensions(test_url)
        except Exception as e:
            log.warning(
                "Image dimension probe failed for candidate",
                extra={"step": "amazon.image_selector", "extra_json": {"url": test_url, "error": str(e)}},
            )
            w, h, dim_status = (None, None, "unknown")

        if fallback is None:
            fallback = {
                "url": test_url,
                "width": w,
                "height": h,
                "dimension_status": dim_status,
                "content_ratio": None,
                "declared_size": declared_size or 0,
            }

        if w is None or w < min_width:
            continue

        try:
            content_ratio = estimate_image_content_ratio(test_url)
        except Exception as e:
            log.info(
                "Image content-ratio probe failed; continuing without ratio",
                extra={"step": "amazon.image_selector", "extra_json": {"url": test_url, "error": str(e)}},
            )
            content_ratio = None

        candidate = {
            "url": test_url,
            "width": w,
            "height": h,
            "dimension_status": dim_status,
            "content_ratio": content_ratio if content_ratio is not None else 0.0,
            "declared_size": declared_size or 0,
        }
        if best is None or (
            candidate["width"],
            1 if candidate["dimension_status"] == "measured" else 0,
            candidate["content_ratio"],
            candidate["height"] or 0,
            candidate["declared_size"],
        ) > (
            best["width"],
            1 if best["dimension_status"] == "measured" else 0,
            best["content_ratio"],
            best["height"] or 0,
            best["declared_size"],
        ):
            best = candidate

    if best is not None:
        return (best["url"], best["width"], best["height"], best["dimension_status"], best["content_ratio"])

    try:
        w, h, dim_status = get_image_dimensions(url)
    except Exception as e:
        log.warning(
            "Primary image dimension probe failed",
            extra={"step": "amazon.image_selector", "extra_json": {"url": url, "error": str(e)}},
        )
        w, h, dim_status = (None, None, "unknown")

    if w is not None and w >= min_width:
        try:
            content_ratio = estimate_image_content_ratio(url)
        except Exception:
            content_ratio = None
        return (url, w, h, dim_status, content_ratio)

    fallback_url = (fallback or {}).get("url") or url
    fallback_w = w if w is not None else (fallback or {}).get("width")
    fallback_h = h if h is not None else (fallback or {}).get("height")
    fallback_status = dim_status if dim_status != "unknown" else ((fallback or {}).get("dimension_status") or "unknown")
    return (fallback_url, fallback_w, fallback_h, fallback_status, None)

def _count_variants(item) -> int:
    try:
        variants = getattr(item.images, "variants", None) or []
        return len(variants)
    except Exception:
        return 0
def _safe_str(x, maxlen: int = 240):
    try:
        s = str(x)
        return s if len(s) <= maxlen else (s[:maxlen] + "...")
    except Exception:
        return None

def _dump_obj(obj, maxlen: int = 2000):
    """
    Best-effort dump for SDK objects (often pydantic models).
    Keeps logs safe-ish by truncating.
    """
    if obj is None:
        return None
    try:
        # pydantic v2
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        # pydantic v1
        if hasattr(obj, "dict"):
            return obj.dict()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return _safe_str(obj, maxlen=maxlen)
    except Exception:
        return _safe_str(obj, maxlen=maxlen)

def _extract_offer_summary(item) -> dict:
    offers = getattr(item, "offers", None)
    listings = getattr(offers, "listings", None) if offers else None
    l0 = listings[0] if listings else None

    price = getattr(l0, "price", None) if l0 else None
    avail = getattr(l0, "availability", None) if l0 else None
    cond = getattr(l0, "condition", None) if l0 else None
    merchant = getattr(l0, "merchant_info", None) if l0 else None
    delivery = getattr(l0, "delivery_info", None) if l0 else None

    return {
        "has_offers": offers is not None,
        "listings_count": len(listings) if listings else 0,
        "price_display_amount": getattr(price, "display_amount", None) if price else None,
        "price_amount": getattr(price, "amount", None) if price else None,
        "price_currency": getattr(price, "currency", None) if price else None,
        "availability_message": getattr(avail, "message", None) if avail else None,
        "availability_type": getattr(avail, "type", None) if avail else None,
        "condition_value": getattr(cond, "value", None) if cond else None,
        "merchant_name": getattr(merchant, "name", None) if merchant else None,
        "is_prime_eligible": getattr(delivery, "is_prime_eligible", None) if delivery else None,
    }


def _format_candidate_line(idx: int, item) -> str:
    asin = getattr(item, "asin", "") or ""
    try:
        title = item.item_info.title.display_value if (item.item_info and item.item_info.title) else ""
    except Exception:
        title = ""
    bn = _get_browse_node(item)
    img = _extract_primary_image_url_simple(item) or "n/a"
    vcount = _count_variants(item)

    offer = _extract_offer_summary(item)
    return (
        f"[search] candidate #{idx} asin={asin}; "
        f"title='{title[:140]}'; "
        f"browse_node='{bn}'; "
        f"primary_image='{img}'; variants={vcount}; "
        f"price={offer.get('price_display_amount')}; "
        f"avail={offer.get('availability_message')}; "
        f"prime={offer.get('is_prime_eligible')}"
    )


# -----------------------------
# PA-API resources we need for prices/images
# -----------------------------
_PAAPI_RESOURCES = [
    # Title / brand / basic info
    "ItemInfo.Title",
    "ItemInfo.ByLineInfo",
    "ItemInfo.Features",
    # Images
    "Images.Primary.Large",
    "Images.Primary.Medium",
    "Images.Primary.Small",
    "Images.Variants.Large",
    "Images.Variants.Medium",
    "Images.Variants.Small",
    # Offers / prices
    "Offers.Listings.Price",
    "Offers.Listings.Availability.Message",
    "Offers.Listings.Availability.Type",
    "Offers.Listings.DeliveryInfo.IsPrimeEligible",
    "Offers.Summaries.LowestPrice",
]

def _api_search_items(api, *, keywords: str, item_count: int | None = None, item_page: int | None = None, search_index: str | None = None):
    """
    Call api.search_items in a way that works across Creators API variants.

    We *prefer* not to pass `resources` if the AmazonApi instance is already
    configured with default resources (some forks do this). We also try to pass
    `item_count`, `item_page` and `search_index` when supported, falling back gracefully if
    the underlying SDK version doesn't accept them.
    """
    base_kwargs = {"keywords": keywords}
    if item_count is not None:
        base_kwargs["item_count"] = int(item_count)
    if item_page is not None:
        base_kwargs["item_page"] = int(item_page)
    if search_index:
        base_kwargs["search_index"] = search_index

    # Try progressive fallbacks to avoid TypeError explosions across SDK versions.
    attempts = [
        (dict(base_kwargs), False),  # no resources
        (dict(base_kwargs), True),   # with resources
    ]

    last_err: Exception | None = None
    for kwargs, include_resources in attempts:
        try:
            if include_resources:
                return api.search_items(**kwargs, resources=_PAAPI_RESOURCES)
            return api.search_items(**kwargs)
        except TypeError as e:
            last_err = e
            msg = str(e).lower()

            # If the SDK doesn't accept item_count/search_index, retry without them.
            if ("unexpected keyword argument" in msg) and ("item_count" in msg or "item_page" in msg or "search_index" in msg):
                kwargs.pop("item_count", None)
                kwargs.pop("item_page", None)
                kwargs.pop("search_index", None)
                try:
                    if include_resources:
                        return api.search_items(**kwargs, resources=_PAAPI_RESOURCES)
                    return api.search_items(**kwargs)
                except TypeError as e2:
                    last_err = e2
                    continue

            # If it *requires* resources, we'll handle that in the next attempt.
            if "missing" in msg and "resources" in msg:
                continue

            # If it complains about multiple values, surface a clearer error.
            if "multiple values" in msg and "resources" in msg:
                raise TypeError(
                    "search_items() received 'resources' twice. "
                    "Remove resources=... from the call site and configure resources on the AmazonApi instance."
                ) from e
        except Exception as e:
            if _is_amazon_eligibility_error(e):
                raise AmazonEligibilityError(_amazon_eligibility_message(e)) from e
            last_err = e
            break

    if last_err:
        raise last_err
    return None


def _api_get_items(api, items: list[str]):
    """Call the underlying Amazon API client's GetItems method.

    PA-API wrappers differ a bit:
      - Some expect `get_items(items=[...])`
      - Some accept `get_items([...])`
      - Some require/accept a `resources` argument
      - Some store resources on the client and will ERROR if you also pass `resources`
        ("multiple values for keyword argument 'resources'").

    Strategy:
      1) Try WITHOUT resources first (positional + keyword).
      2) Only try WITH resources if the error suggests resources are required AND the
         client doesn't already appear to have resources configured.
    """
    if not items:
        raise ValueError("No items/ASINs provided to GetItems")

    last_err: Exception | None = None

    # 1) Prefer calls WITHOUT resources (avoids 'multiple values for resources' issues).
    try:
        return api.get_items(items)
    except TypeError as e:
        last_err = e

    try:
        return api.get_items(items=items)
    except TypeError as e:
        last_err = e

    # 2) If resources are required, try again with resourcesÃ¢â‚¬â€but ONLY if the client
    # doesn't already seem to carry resources internally.
    err_msg = (str(last_err) if last_err else "").lower()
    needs_resources = ("resources" in err_msg) and ("required" in err_msg or "missing" in err_msg)
    has_client_resources = bool(getattr(api, "resources", None) or getattr(api, "_resources", None))

    if needs_resources and not has_client_resources:
        try:
            return api.get_items(items, resources=_PAAPI_RESOURCES)
        except TypeError as e:
            last_err = e
        try:
            return api.get_items(items=items, resources=_PAAPI_RESOURCES)
        except TypeError as e:
            last_err = e

    raise TypeError(f"GetItems call failed. Last error: {last_err}") from last_err
def _get_status_code(exc: Exception):
    # Some exceptions expose a direct status_code
    if hasattr(exc, "status_code"):
        return getattr(exc, "status_code")

    # Some attach a response object
    resp = getattr(exc, "response", None)
    if resp is not None and hasattr(resp, "status_code"):
        return resp.status_code

    return None


def _is_retryable_exception(exc: Exception) -> bool:
    code = _get_status_code(exc)
    if code in (429, 500, 502, 503, 504):
        return True

    msg = str(exc).lower()
    retry_markers = (
        "429",
        "too many request",
        "rate limit",
        "throttle",
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "temporarily unavailable",
    )
    return any(m in msg for m in retry_markers)



def safe_search_items(
    api,
    keywords,
    retries=5,
    trace_id=None,
    item_count: int | None = None,
    item_page: int | None = None,
    search_index: str | None = None,
):
    for attempt in range(retries):
        try:
            log.info(
                f"Ã°Å¸â€Â Amazon search attempt {attempt + 1} for: '{keywords}'",
                extra={"trace_id": trace_id}
            )

            results = _api_search_items(
                api,
                keywords=keywords,
                item_count=item_count,
                item_page=item_page,
                search_index=search_index,
            )

            items = results.items or [] if results else []
            if not items:
                log.info(
                    f"Ã¢â€žÂ¹Ã¯Â¸Â No items found for: '{keywords}'",
                    extra={"trace_id": trace_id, "extra_json": {"query": keywords}},
                )
                return results

            count = len(items)
            log.info(
                f"âœ… Items returned for: '{keywords}' Ã¢Å¾Å“ {count} items",
                extra={"trace_id": trace_id}
            )
            for idx, it in enumerate(items, start=1):
                log.info(_format_candidate_line(idx, it), extra={"trace_id": trace_id})
            return results

        except Exception as e:
            if isinstance(e, AmazonEligibilityError) or _is_amazon_eligibility_error(e):
                raise e if isinstance(e, AmazonEligibilityError) else AmazonEligibilityError(_amazon_eligibility_message(e))

            retryable = _is_retryable_exception(e)

            if retryable and attempt < retries - 1:
                # much slower backoff for Amazon throttling / 429
                status_code = _get_status_code(e)
                if status_code == 429 or "rate limit" in str(e).lower() or "throttle" in str(e).lower():
                    delay = min(60.0, (2 ** attempt) * 5.0 + random.uniform(0.5, 2.0))
                else:
                    delay = min(20.0, (2 ** attempt) * 2.0 + random.uniform(0.2, 1.0))

                log.warning(
                    "Ã¢ÂÂ³ Retryable search error",
                    extra={
                        "trace_id": trace_id,
                        "extra_json": {
                            "attempt": attempt + 1,
                            "delay_sec": round(delay, 2),
                            "error": type(e).__name__,
                            "message": str(e),
                        },
                    },
                )
                time.sleep(delay)
                continue

            log.error(
                f"Ã°Å¸Å¡Â¨ Non-retryable error: {e}",
                extra={"trace_id": trace_id}
            )
            break

    return None


@with_step("amazon.safe_get_items")



def safe_get_items(
    api,
    asins: list[str],
    retries: int = 2,
    trace_id: str | None = None,
    empty_items_retries: int = 2,
    empty_base_delay: float = 2.0,
):
    """Fetch full item details (incl. images) for the selected ASIN(s)."""
    empty_attempt = 0

    for attempt in range(retries):
        try:
            log.info(
                f"Ã°Å¸â€Å½ Amazon GetItems attempt {attempt+1} for: {asins}",
                extra={"trace_id": trace_id, "extra_json": {"asins": asins}},
            )

            res = _api_get_items(api, asins)
            items = getattr(res, "items", None) or []

            # Retry on HTTP 200 but empty items (intermittent catalog/offer gaps)
            if not items:
                if empty_attempt < empty_items_retries:
                    empty_attempt += 1

                    # Gentle exponential backoff + small jitter (avoid hammering)
                    delay = (empty_base_delay * (2 ** (empty_attempt - 1))) + random.uniform(0, 0.8)
                    delay = min(delay, 12.0)

                    log.warning(
                        "Ã¢â€žÂ¹Ã¯Â¸Â GetItems returned 0 items; retrying after short backoff",
                        extra={
                            "trace_id": trace_id,
                            "extra_json": {
                                "asins": asins,
                                "empty_attempt": empty_attempt,
                                "empty_items_retries": empty_items_retries,
                                "delay_sec": round(delay, 2),
                            },
                        },
                    )
                    time.sleep(delay)
                    continue

                log.info(
                    "Ã¢â€žÂ¹Ã¯Â¸Â GetItems returned 0 items (giving up after empty retries; treating as not found)",
                    extra={
                        "trace_id": trace_id,
                        "extra_json": {
                            "asins": asins,
                            "empty_attempts_used": empty_attempt,
                            "empty_items_retries": empty_items_retries,
                        },
                    },
                )
                return res  # keep existing caller behavior

            log.info(
                f"âœ… GetItems returned {len(items)} item(s)",
                extra={"trace_id": trace_id, "extra_json": {"count": len(items)}},
            )
            return res

        except Exception as e:
            if _is_retryable_exception(e) and attempt < retries - 1:
                delay = 5 ** attempt + random.uniform(0, 1.5)
                log.warning(
                    "Ã¢ÂÂ³ Retryable GetItems error",
                    extra={
                        "trace_id": trace_id,
                        "extra_json": {
                            "attempt": attempt + 1,
                            "delay_sec": round(delay, 2),
                            "error": type(e).__name__,
                            "message": str(e),
                        },
                    },
                )
                time.sleep(delay)
                continue

            log.error(f"Ã°Å¸Å¡Â¨ Non-retryable GetItems error: {e}", extra={"trace_id": trace_id})
            break

    return None





def get_brand_stopwords(cfg: dict) -> set[str]:
    words = (cfg.get("brand_stopwords") or [])
    return {w.strip().lower() for w in words if isinstance(w, str)}

def extract_brand_terms_from_query(query: str, stopwords: set[str] | None = None) -> list[str]:
    stopwords = stopwords or set()
    q = normalize_ws(query)
    words = re.findall(r"[A-Za-z][A-Za-z\-]+", q)
    brand_candidates = [w for w in words if w.lower() not in stopwords]
    if not brand_candidates:
        return []
    model_tokens = list(extract_model_tokens(q))
    if model_tokens:
        first_model = min((q.upper().find(mt) for mt in model_tokens if q.upper().find(mt) != -1), default=-1)
        if first_model != -1:
            prefix = []
            pos = 0
            for w in words:
                idx = q.find(w, pos); pos = idx + len(w) if idx != -1 else pos
                if idx != -1 and idx < first_model and w.lower() not in stopwords:
                    prefix.append(w)
            if prefix:
                return prefix[:2]
    return brand_candidates[:2]

def get_title_required_terms(cfg: dict) -> set[str]:
    explicit = cfg.get("title_required_terms")
    if explicit:
        return {t.strip().lower() for t in explicit if isinstance(t, str) and t.strip()}
    inferred = set()
    for kw in (cfg.get("include_keywords") or []):
        if not isinstance(kw, str):
            continue
        toks = re.findall(r"[A-Za-z]+", kw.lower())
        if len(toks) >= 1:
            inferred.add(toks[-1])
    return inferred

def _has_usable_image(item) -> tuple[bool, str | None]:
    try:
        if item.images and getattr(item.images, "primary", None):
            p = item.images.primary
            for sz in ("large", "medium", "small"):
                u = getattr(getattr(p, sz, None), "url", None)
                if u:
                    return True, f"primary.{sz}"
        if item.images and getattr(item.images, "variants", None):
            for vidx, v in enumerate(item.images.variants or []):
                for sz in ("large", "medium", "small"):
                    u = getattr(getattr(v, sz, None), "url", None)
                    if u:
                        return True, f"variant[{vidx}].{sz}"
    except Exception:
        pass
    return False, None

# --- Amazon helpers & selection ---

MODEL_TOKEN_RE = re.compile(
    r"\b((?:[A-Z]{1,6}\d{1,6}[A-Z]{0,3})|"
    r"(?:\d+(?:\.\d+)?(?:ML|L|KG|G|MM|CM|M|IN|GB|TB|W|V|HZ)))\b", re.I
)

def extract_model_tokens(s: str) -> set[str]:
    # Improve extraction for hyphenated models like LV-H132 -> LVH132
    cleaned = (s or "").replace("-", "")
    return {t.upper() for t in MODEL_TOKEN_RE.findall(cleaned)}

def _split_model_token(tok: str) -> tuple[str, str, str]:
    """
    Returns (letters_prefix, digits, suffix_letters)
    Example: 'AC2889X' -> ('AC', '2889', 'X')
    """
    m = re.match(r"^([A-Z]+)(\d+)([A-Z]*)$", (tok or "").upper())
    if not m:
        return ("", "", "")
    return (m.group(1), m.group(2), m.group(3))

def same_model_family(query: str, title: str) -> bool:
    qtokens = extract_model_tokens(query)
    ttokens = extract_model_tokens(title)
    if not qtokens or not ttokens:
        return False

    for qt in qtokens:
        qL, qD, _ = _split_model_token(qt)
        if not qL or len(qD) < 2:
            continue
        for tt in ttokens:
            tL, tD, _ = _split_model_token(tt)
            if not tL or len(tD) < 2:
                continue

            # Require same letters prefix AND first 2 digits match
            if qL == tL and qD[:2] == tD[:2]:
                return True

    return False


def near_model_match(query: str, title: str, cfg: dict | None = None) -> bool:
    """Return True if query/title model tokens are *close enough* to be considered a near match.

    This is meant as a pragmatic fallback when an exact model token isn't available on Amazon
    (e.g. LV-H132 vs LV-H128).

    Rules:
      - Extract model tokens from both strings.
      - A near match exists if:
          * the letter-prefix matches (e.g. LVH) AND
          * both have numeric parts AND
          * abs(num_a - num_b) <= cfg['near_model_max_numeric_distance'] (default 10)

    You can disable by setting cfg['allow_near_model_match']=false.
    """
    if not query or not title:
        return False
    if cfg and cfg.get("allow_near_model_match") is False:
        return False

    max_dist = 10
    if cfg:
        try:
            max_dist = int(cfg.get("near_model_max_numeric_distance", 10))
        except Exception:
            max_dist = 10

    q_tokens = extract_model_tokens(query)
    t_tokens = extract_model_tokens(title)
    if not q_tokens or not t_tokens:
        return False

    for qt in q_tokens:
        # _split_model_token returns (prefix, number, raw_prefix). We only need
        # the normalized prefix + numeric portion here.
        q_pref, q_num_s, _q_raw = _split_model_token(qt)
        if not q_pref or not q_num_s:
            continue
        try:
            q_num = int(q_num_s)
        except Exception:
            continue
        for tt in t_tokens:
            t_pref, t_num_s, _t_raw = _split_model_token(tt)
            if not t_pref or not t_num_s:
                continue
            try:
                t_num = int(t_num_s)
            except Exception:
                continue
            if q_pref == t_pref and abs(q_num - t_num) <= max_dist:
                return True
    return False

def _letters_prefix(token: str) -> str:
    m = re.match(r"[A-Za-z]+", token or "")
    return (m.group(0) if m else "").upper()

def get_deny_browse_nodes(cfg: dict) -> set[str]:
    return {b.strip().lower() for b in (cfg.get("deny_browse_nodes") or []) if isinstance(b, str)}

def _accessory_signal_hits(
    title_lc: str,
    browse_node_lc: str = "",
    cfg: dict | None = None,
) -> list[str]:
    text = f" {normalize_ws(title_lc or '').lower()} "
    node = f" {normalize_ws(browse_node_lc or '').lower()} "
    hits: list[str] = []

    active = cfg if isinstance(cfg, dict) else _ACTIVE_CATEGORY_CONFIG
    for term in (active.get("accessory_title_signals") or []):
        t = term.strip().lower()
        if not t:
            continue
        if t.endswith(" "):
            if t in text:
                hits.append(t.strip())
        elif re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", text):
            hits.append(t)

    for term in (active.get("accessory_node_signals") or []):
        t = term.strip().lower()
        if t and re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", node):
            hits.append(f"node:{t}")

    return sorted(set(hits))

def _looks_like_accessory_listing(
    title_lc: str,
    browse_node_lc: str = "",
    cfg: dict | None = None,
) -> bool:
    hits = _accessory_signal_hits(title_lc, browse_node_lc, cfg)
    if not hits:
        return False

    # A true vacuum can mention a HEPA filter or brush in its feature list. Accessory
    # listings usually combine those terms with "for", "compatible", packs, or parts nodes.
    strong_hits = {
        h for h in hits
        if h.startswith("node:")
        or h in {
            "replacement", "replace", "replaces", "refill", "cartridge",
            "compatible", "compatible with", "spare",
            "kit", "repair", "pack", "set", "pcs", "pieces", "count",
            "accessory", "accessories", "dust bag", "dust bags",
        }
    }
    if strong_hits:
        return True

    title = normalize_ws(title_lc or "").lower()
    active = cfg if isinstance(cfg, dict) else _ACTIVE_CATEGORY_CONFIG
    primary_terms = active.get("primary_product_terms") or active.get("include_keywords") or []
    return not any(str(term).casefold() in title for term in primary_terms)

def substitute_specialization_compatible(
    requested_name: str,
    candidate_title: str,
    cfg: dict | None = None,
    browse_node: str = "",
) -> tuple[bool, dict]:
    """Reject a substitute that changes the product's fundamental use case.

    Rules are category-neutral and configurable. A rule only blocks on a strong
    specialization phrase in the candidate (for example, ``backpack cooler``),
    and only when neither the requested name nor configured search context asks
    for that specialization.
    """
    cfg = cfg or {}
    rules = cfg.get("substitute_specialization_rules") or []
    configured_context = " ".join(
        str(value)
        for key in ("include_keywords", "search_query_suffix", "variant_suffix", "product_focus_queries")
        for value in ((cfg.get(key) or []) if isinstance(cfg.get(key), list) else [cfg.get(key)])
        if value
    )
    requested_context = normalize_name(f"{requested_name} {configured_context}")
    candidate_context = normalize_name(f"{candidate_title} {browse_node}")

    for rule in rules:
        strong = {normalize_name(x) for x in (rule.get("strong") or []) if str(x).strip()}
        context = {normalize_name(x) for x in (rule.get("context") or []) if str(x).strip()}
        candidate_hits = sorted(term for term in strong if contains_whole_term(candidate_context, {term}))
        requested_hits = sorted(term for term in context if contains_whole_term(requested_context, {term}))
        if candidate_hits and not requested_hits:
            return False, {"rule": rule.get("name", "specialization"), "candidate_hits": candidate_hits}
    return True, {}


def _candidate_baseline_ok(
    title_raw: str,
    title_lc: str,
    browse_node_lc: str,
    cfg: dict,
    query_name: str,
    is_rank1: bool,
) -> tuple[bool, list[str]]:
    """
    Baseline gating for Amazon candidates.

    Consolidated fixes:
      1) Option B for title_required_terms:
         - DO NOT hard-reject on missing required terms; treat as soft signal (handled in scoring).
      2) Deny-browse-node check only on *category* segments (avoid promo buckets like "Jewelry, Luggage, Watches").
      3) Exclude-in-title deny terms are treated as accessory indicators only when the title does NOT look like luggage.
      4) Keep allowed browse-node gating + rank/model-family gating (tunable below).
    """
    reasons: list[str] = []

    # Feature flags (category-config driven)
    # NOTE: defaults preserve historical behavior unless you explicitly set these in config.
    hard_gate_browse_nodes = bool(cfg.get("hard_gate_browse_nodes", True))
    brand_required_if_present = bool(cfg.get("brand_required_if_present", False))

    # --- Config normalization ---
    deny_terms_strict = {
        t.strip().lower()
        for t in (cfg.get("exclude_in_title_strict") or cfg.get("exclude_in_title") or [])
        if isinstance(t, str) and t.strip()
    }
    deny_terms_soft = {
        t.strip().lower()
        for t in (cfg.get("exclude_in_title_soft") or [])
        if isinstance(t, str) and t.strip()
    }

    deny_nodes = {
        b.strip().lower()
        for b in (cfg.get("deny_browse_nodes") or [])
        if isinstance(b, str) and b.strip()
    }

    include_nodes = [
        bn.strip().lower()
        for bn in (cfg.get("include_browse_nodes") or [])
        if isinstance(bn, str) and bn.strip()
    ]

    include_keywords = [
        k.strip().lower()
        for k in (cfg.get("include_keywords") or [])
        if isinstance(k, str) and k.strip()
    ]

    required_terms = get_title_required_terms(cfg)  # expected already lowercased in your helper

    # --- Core title for matching ---
    core_title_raw = normalize_title_for_matching(title_raw or "", cfg)
    core_title_lc = core_title_raw.lower()

    # --- Browse-node parsing ---
    bn = (browse_node_lc or "").lower()
    segments = [s.strip() for s in bn.split(">") if s.strip()]
    head = " > ".join(segments[:2])  # only top category segments; avoids promo buckets deep in the path

    # Allowed-node confidence
    node_ok = (not include_nodes) or any(n in bn for n in include_nodes)
    if include_nodes and not node_ok:
        # May be soft-only depending on `hard_gate_browse_nodes`
        reasons.append("not_in_allowed_browse_nodes")

    # Deny-node check (use head only)
    if any(d in head for d in deny_nodes):
        reasons.append("deny_browse_node")

    accessory_gate_enabled = not bool(cfg.get("disable_accessory_listing_gate", False))
    specialization_ok, specialization_evidence = substitute_specialization_compatible(
        query_name, core_title_raw, cfg, browse_node_lc
    )
    if not specialization_ok:
        reasons.append("unrequested_specialization")
    accessory_hits = _accessory_signal_hits(core_title_lc, browse_node_lc, cfg)
    if accessory_gate_enabled and _looks_like_accessory_listing(core_title_lc, browse_node_lc, cfg):
        reasons.append("accessory_listing")

    # --- Exclude terms in title (hard reject) ---
    # Terms in `exclude_in_title` are always treated as disqualifying, regardless of category.
    # This prevents selecting filters/replacement parts/accessories when the intent is to match
    # the primary product.
    if deny_terms_strict and contains_whole_term(core_title_lc, deny_terms_strict):
        # Some strict terms (notably "filter") can legitimately appear in primary-product titles.
        # Treat them as disqualifying ONLY when the listing shows accessory/parts signals.
        deny_hit = first_whole_term_match(core_title_lc, deny_terms_strict) or ""
        allowed_terms = set(normalize_terms(cfg.get("allow_terms_in_title", [])))
        allowed_terms.update(normalize_terms(
            cfg.get("strict_title_terms_allowed_without_accessory_signals", [])
        ))
        is_allowable = deny_hit in allowed_terms

        if not is_allowable:
            reasons.append("deny_term_in_title")
        else:
            looks_like_parts_category = contains_any(
                browse_node_lc,
                cfg.get("accessory_node_signals") or [],
            )
            if accessory_hits or looks_like_parts_category:
                reasons.append("deny_term_in_title")
            else:
                reasons.append("deny_term_allowed")


    # Soft exclude terms (score penalty only)
    if deny_terms_soft and contains_whole_term(core_title_lc, deny_terms_soft):
        reasons.append("soft_deny_term_in_title")

    # --- Option B: title_required_terms are NOT a baseline hard requirement --- are NOT a baseline hard requirement ---
    # We keep a soft marker for debugging/scoring insights but never gate on it here.
    if required_terms and not any(rt in core_title_lc for rt in required_terms):
        reasons.append("missing_required_terms_soft")

    # --- Brand gating (optional, config-controlled) ---
    # If the query includes a brand (e.g., "Levoit") we can force brand-consistency.
    # This is useful in categories where Amazon search returns many compatibles/similar models.
    if brand_required_if_present:
        stopwords = get_brand_stopwords(cfg)
        brand_terms = [b.lower() for b in extract_brand_terms_from_query(query_name, stopwords)]
        if brand_terms and not contains_all_whole_terms(core_title_lc, brand_terms):
            reasons.append("brand_mismatch")

    # --- Model overlap gating (keep as you had it; you can relax later if needed) ---
    mo = has_model_overlap(query_name, core_title_raw)
    same_family = same_model_family(query_name, core_title_raw)

    # Demoted from hard gate: keep as a soft marker only
    near_m = near_model_match(query_name, core_title_raw, cfg)
    if not is_rank1 and not (mo or same_family or near_m):
        reasons.append("no_model_or_family_overlap_soft")

    hard_blocks = {
        "unrequested_specialization",
        "deny_term_in_title",
        "deny_browse_node",
        "accessory_listing",
        # optionally gate on browse-node membership
        *( {"not_in_allowed_browse_nodes"} if hard_gate_browse_nodes else set() ),
        # optionally gate on brand mismatch
        *( {"brand_mismatch"} if brand_required_if_present else set() ),
        # intentionally NOT gating on model overlap anymore
    }
    ok = not any(r in hard_blocks for r in reasons)
    return ok, reasons



def _score_candidate(
    idx: int,
    title: str,
    browse_node_lc: str,
    cfg: dict,
    query_name: str,
    has_img: bool,
    img_slot: str | None,
    is_rank1: bool,
) -> tuple[int, dict]:
    """
    Score an Amazon candidate item.

    - Option B behavior for title_required_terms:
        * If browse node looks confidently in allowed luggage nodes, missing required terms
          is not a hard negative.
        * If browse node is NOT confidently allowed, required terms matter more (safety net).

    - Demoted no_model_or_family_overlap from hard gate to scoring:
        * If neither model-overlap nor family match, apply a mild penalty (optional)
          instead of filtering the candidate out in baseline.
    """
    tl_full = (title or "").lower()

    # Prefer scoring against the "core" title before feature bullets/noise
    core_title_lc = (tl_full.split(" - ", 1)[0] if tl_full else "")

    # Config
    stopwords = get_brand_stopwords(cfg)
    brand_terms = [b.lower() for b in extract_brand_terms_from_query(query_name, stopwords)]
    include_nodes = [
        bn.strip().lower()
        for bn in (cfg.get("include_browse_nodes") or [])
        if isinstance(bn, str) and bn.strip()
    ]
    required_terms = get_title_required_terms(cfg)  # already normalized/lower in your helper
    deny_terms_strict = {
        t.strip().lower()
        for t in (cfg.get("exclude_in_title_strict") or cfg.get("exclude_in_title") or [])
        if isinstance(t, str) and t.strip()
    }
    deny_terms_soft = {
        t.strip().lower()
        for t in (cfg.get("exclude_in_title_soft") or [])
        if isinstance(t, str) and t.strip()
    }

    # Node confidence (Option B)
    node_ok = (not include_nodes) or any(n in (browse_node_lc or "") for n in include_nodes)

    # Signals
    req_hit = bool(required_terms and any(rt in core_title_lc for rt in required_terms))
    deny_hit = contains_whole_term(core_title_lc, deny_terms_strict)
    soft_deny_hit = contains_whole_term(core_title_lc, deny_terms_soft)
    accessory_hits = _accessory_signal_hits(tl_full, browse_node_lc, cfg)
    accessory_gate_enabled = not bool(cfg.get("disable_accessory_listing_gate", False))
    accessory_listing = accessory_gate_enabled and _looks_like_accessory_listing(tl_full, browse_node_lc, cfg)
    brand_hit = contains_all_whole_terms(tl_full, brand_terms) if brand_terms else False
    exact_tok = exact_model_token_match(query_name, title)
    exact = bool(exact_tok)
    mo = has_model_overlap(query_name, title)
    fam = same_model_family(query_name, title)
    near_m = near_model_match(query_name, title, cfg)

    score = 0
    why: dict = {}

    # 1) Rank bonus (still matters, but not everything)
    base_rank_bonus = max(0, 60 - (idx * 5))
    if is_rank1:
        base_rank_bonus += 15
    score += base_rank_bonus
    why["rank_bonus"] = base_rank_bonus
    why["is_rank1"] = bool(is_rank1)

    # 2) Required terms (Option B weighting)
    if required_terms:
        why["required_terms_hit"] = req_hit
        why["required_terms_node_ok"] = node_ok

        if req_hit:
            bonus = 22 if node_ok else 45
            score += bonus
            why["required_terms_bonus"] = bonus
        else:
            if not node_ok:
                score -= 25
                why["missing_required_terms_penalty"] = 25

    # 3) Exact model token / overlap / family (scoring)
    why["exact_model_token"] = exact_tok
    if exact:
        score += 80
        why["exact_model_bonus"] = 80
        why["model_overlap"] = True
        why["same_family"] = bool(fam)
    elif mo:
        score += 40
        why["model_overlap_bonus"] = 40
        why["model_overlap"] = True
        why["same_family"] = bool(fam)
    elif fam:
        score += 28
        why["same_family_bonus"] = 28
        why["model_overlap"] = False
        why["same_family"] = True
    elif near_m:
        bonus = int(cfg.get("near_model_bonus", 16))
        score += bonus
        why["near_model_bonus"] = bonus
        why["model_overlap"] = False
        why["near_model"] = True
    else:
        # Mild penalty to push weak matches down, but keep them eligible
        score -= 10
        why["no_model_or_family_penalty"] = 10
        why["model_overlap"] = False
        why["same_family"] = False

    # 3b) Demoted overlap rule: mild penalty if neither model-overlap nor family match
    if not (mo or fam or near_m):
        score -= 18
        why["no_model_or_family_overlap_penalty"] = 18


    # 4) Brand match (bonus + mismatch penalty)
    if brand_terms:
        if brand_hit:
            score += 20
            why["brand_match"] = True
        else:
            score -= 70
            why["brand_match"] = False
            why["brand_mismatch_penalty"] = 70

    # 5) Browse node bonus (small but useful)
    if include_nodes:
        in_allowed = any(n in (browse_node_lc or "") for n in include_nodes)
        if in_allowed:
            score += 15
        why["allowed_node"] = bool(in_allowed)

    # 6) Deny terms (hard negative)
    if deny_hit:
        score -= 80
        why["deny_term_penalty"] = 80

    # 6a) Accessory/parts listings should not beat plausible primary products just
    # because they contain the exact model token.
    if accessory_listing:
        score -= 140
        why["accessory_listing"] = True
        why["accessory_penalty"] = 140
        why["accessory_hits"] = accessory_hits[:6]

    # 6b) Soft deny terms (penalty but not disqualifying)
    if soft_deny_hit:
        score -= 20
        why["soft_deny_term_penalty"] = 20
    # 7) Image availability
    if has_img:
        score += 25
        why["has_image"] = True
        why["img_slot"] = img_slot
    else:
        score -= 35
        why["has_image"] = False
        why["no_image_penalty"] = 35

    return score, why



def select_purifier_strict(results_items, query_name: str, cfg: dict, trace_id: str | None = None, return_score: bool = False):
    def _none():
        # Always return a consistent shape for callers that expect a tuple.
        return (None, -10_000) if return_score else None
    if not results_items:
        return _none()

    raw_candidates = []
    for it in results_items:
        title = it.item_info.title.display_value if (it.item_info and it.item_info.title) else ""
        tl = (title or "").lower()
        bn = (_get_browse_node(it) or "").lower()
        raw_candidates.append((title, tl, bn, it))

    candidates = []
    for idx, (title, tl, bn, it) in enumerate(raw_candidates):
        ok, reasons = _candidate_baseline_ok(
            title, tl, bn, cfg, query_name, is_rank1=(idx == 0)
        )

        asin = getattr(it, "asin", "") or ""

        if not ok:
            log.info(
                "[selector] baseline REJECT asin=%s node=%r query=%r is_rank1=%s title=%r",
                asin,
                bn,
                query_name,
                (idx == 0),
                title,
                extra={
                    "trace_id": trace_id,
                    "step": "selector",
                    "extra_json": {
                        "asin": asin,
                        "title": title,
                        "node": bn,
                        "reasons": reasons,
                        "rank": idx + 1,
                        "is_rank1": (idx == 0),
                    },
                },
            )
            continue

        candidates.append((title, tl, bn, it))

        log.info(
            "[selector] baseline ACCEPT asin=%s node=%r query=%r is_rank1=%s title=%r",
            asin,
            bn,
            query_name,
            (idx == 0),
            title,
            extra={
                "trace_id": trace_id,
                "step": "selector",
                "extra_json": {
                    "asin": asin,
                    "title": title,
                    "tl": tl,
                    "node": bn,
                    "reasons": reasons,  # keep if you want; usually empty/soft-only on accept
                    "query_name": query_name,
                    "rank": idx + 1,
                    "is_rank1": (idx == 0),
                },
            },
        )

    if not candidates:
        log.warning(
            "[selector] No eligible candidates after baseline gating",
            extra={
                "trace_id": trace_id,
                "step": "selector",
                "extra_json": {"query": query_name, "raw_count": len(raw_candidates)},
            },
        )
        return _none()

    include_nodes = [
        bn.strip().lower()
        for bn in (cfg.get("include_browse_nodes") or [])
        if isinstance(bn, str) and bn.strip()
    ]
    if include_nodes:
        in_bucket = [c for c in candidates if any(n in (c[2] or "") for n in include_nodes)]
        if in_bucket:
            log.info(
                "[selector] node preference applied",
                extra={
                    "trace_id": trace_id,
                    "step": "selector",
                    "extra_json": {
                        "preferred_nodes": include_nodes,
                        "kept": len(in_bucket),
                        "dropped": len(candidates) - len(in_bucket),
                    },
                },
            )
            candidates = in_bucket

    stopwords = get_brand_stopwords(cfg)
    brand_terms = [b.lower() for b in extract_brand_terms_from_query(query_name, stopwords)]
    if brand_terms:
        same_brand = [c for c in candidates if contains_all_whole_terms(c[1], brand_terms)]
        same_brand_primary = [
            c for c in same_brand
            if (cfg.get("disable_accessory_listing_gate", False) or not _looks_like_accessory_listing(c[1], c[2], cfg))
        ]
        if same_brand_primary:
            log.info(
                "[selector] brand preference applied",
                extra={
                    "trace_id": trace_id,
                    "step": "selector",
                    "extra_json": {
                        "brand_terms": brand_terms,
                        "kept": len(same_brand_primary),
                        "dropped": len(candidates) - len(same_brand_primary),
                    },
                },
            )
            candidates = same_brand_primary
        elif same_brand:
            log.info(
                "[selector] brand preference skipped; same-brand matches look like accessories",
                extra={
                    "trace_id": trace_id,
                    "step": "selector",
                    "extra_json": {
                        "brand_terms": brand_terms,
                        "same_brand": len(same_brand),
                        "kept": len(candidates),
                    },
                },
            )

    scoreboard = []
    best = None
    best_score = -10_000

    for idx, (title, tl, bn, it) in enumerate(candidates):
        is_rank1 = (idx == 0)
        has_img, img_slot = _has_usable_image(it)
        score, why = _score_candidate(
            idx, title, bn, cfg, query_name, has_img, img_slot, is_rank1=is_rank1
        )

        asin = getattr(it, "asin", "")
        scoreboard.append(
            {
                "rank": idx + 1,
                "asin": asin,
                "title": title[:120],
                "node": bn,
                "score": score,
                "why": {
                    k: v
                    for k, v in why.items()
                    if k
                    in (
                        "required_terms",
                        "model_overlap",
                        "same_family",
                        "brand_match",
                        "allowed_node",
                        "deny_term_penalty",
                        "accessory_listing",
                        "accessory_penalty",
                        "has_image",
                    )
                },
            }
        )

        if score > best_score:
            best = it
            best_score = score

    topn = sorted(scoreboard, key=lambda r: r["score"], reverse=True)[: min(5, len(scoreboard))]
    if topn:
        log_lines = [
            (
                f"    #{r['rank']:>2} score={r['score']:>3} asin={r['asin']} node='{r['node']}' "
                f"req={r['why'].get('required_terms')} mo={r['why'].get('model_overlap')} fam={r['why'].get('same_family')} "
                f"brand={r['why'].get('brand_match')} node_ok={r['why'].get('allowed_node')} "
                f"deny_pen={r['why'].get('deny_term_penalty')} acc_pen={r['why'].get('accessory_penalty')} "
                f"has_img={r['why'].get('has_image')}"
            )
            for r in topn
        ]
        log.info(
            "[selector] scored candidates (top {})\n{}".format(len(topn), "\n".join(log_lines)),
            extra={"trace_id": trace_id, "step": "selector"},
        )

    try:
        winner_rank = next(r["rank"] for r in scoreboard if r["score"] == best_score)
    except StopIteration:
        winner_rank = None

    if candidates:
        title1, _, _, it1 = candidates[0]
        log.info(
            "[selector] FINAL selection",
            extra={
                "trace_id": trace_id,
                "step": "selector",
                "extra_json": {
                    "winner_rank": winner_rank,
                    "winner_asin": getattr(best, "asin", ""),
                    "amazon_rank1_asin_after_filters": getattr(it1, "asin", ""),
                    "explanation": (
                        "Baseline-deny enforced; preferred nodes="
                        f"{include_nodes or []}; brand_terms={brand_terms or []}; "
                        "scored remaining by model/brand/node/image with reduced rank bias."
                    ),
                },
            },
        )
    else:
        log.info(
            "[selector] FINAL selection Ã¢â‚¬â€ no candidates",
            extra={"trace_id": trace_id, "step": "selector"},
        )

    # Optional absolute floor: if the best match is still poor, return _none()
    # so callers can retry with a different query variant.
    min_accept_score = int(cfg.get("min_accept_score", -10_000))
    if best is not None and best_score < min_accept_score:
        log.warning(
            "[selector] Best score below min_accept_score; rejecting",
            extra={
                "trace_id": trace_id,
                "step": "selector",
                "extra_json": {
                    "query": query_name,
                    "best_score": best_score,
                    "min_accept_score": min_accept_score,
                    "best_asin": getattr(best, "asin", ""),
                },
            },
        )
        return _none()

    return (best, best_score) if return_score else best

def _closest_product_with_image(product_data: list[dict], target_name: str, cfg: Optional[dict] = None):
    """
    Returns (best_pd, score) for the product whose name is closest to target_name,
    but only among items that actually have an image URL.
    """
    cfg = cfg or {}
    target_norm = normalize_name(target_name)

    # Substitution controls (config-driven)
    min_sim = int(cfg.get("substitute_min_similarity", 0))
    require_brand = bool(cfg.get("substitute_require_brand", False))
    require_model_overlap = bool(cfg.get("substitute_require_model_overlap", False))

    # Extract brand terms from the target (e.g., "Levoit") to avoid cross-brand substitutes
    stopwords = get_brand_stopwords(cfg)
    target_brand_terms = [b.lower() for b in extract_brand_terms_from_query(target_name, stopwords)]
    target_model_tokens = extract_model_tokens(target_name)

    best, best_score = None, -1
    for pd in product_data or []:
        if not pd.get("img_url"):
            continue

        cand_name = pd.get("name", "") or ""
        cand_norm = normalize_name(cand_name)
        score = fuzz.token_sort_ratio(cand_norm, target_norm)

        # Brand constraint (optional)
        if require_brand and target_brand_terms:
            if not contains_all_whole_terms((cand_name or "").lower(), target_brand_terms):
                continue

        # Model-token overlap constraint (optional)
        if require_model_overlap and target_model_tokens:
            if not (extract_model_tokens(cand_name) & target_model_tokens):
                continue

        if score >= min_sim and score > best_score:
            best, best_score = pd, score

    return best, best_score
    
def _safe_get_display_price(item, log=None, trace_id=None) -> Optional[str]:
    """
    Safely extract a human display price if present.
    Accepts optional log + trace_id so callers can pass them without crashing.
    """
    try:
        offers = getattr(item, "offers", None)
        listings = getattr(offers, "listings", None) if offers else None
        l0 = listings[0] if listings else None
        p0 = getattr(l0, "price", None) if l0 else None
        if p0 and getattr(p0, "display_amount", None):
            return p0.display_amount
    except Exception as e:
        if log:
            try:
                log.warning(
                    "safe_get_display_price failed",
                    extra={"trace_id": trace_id, "extra_json": {"err": str(e)}},
                )
            except Exception:
                pass
    return None





def _safe_dump(obj, max_len=6000):
    """Best-effort dump for PA-API SDK objects."""
    if obj is None:
        return None

    try:
        # 1) Pydantic v2
        if hasattr(obj, "model_dump"):
            payload = obj.model_dump()

        # 2) Many OpenAPI/Swagger SDKs
        elif hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            payload = obj.to_dict()

        elif hasattr(obj, "as_dict") and callable(getattr(obj, "as_dict")):
            payload = obj.as_dict()

        elif hasattr(obj, "to_str") and callable(getattr(obj, "to_str")):
            # already a string representation
            s = obj.to_str()
            return s[:max_len] + "...[truncated]" if len(s) > max_len else s

        # 3) Common internal stores
        elif hasattr(obj, "_data_store"):
            payload = getattr(obj, "_data_store")

        elif hasattr(obj, "data_store"):
            payload = getattr(obj, "data_store")

        # 4) Fallback to public __dict__
        elif hasattr(obj, "__dict__"):
            payload = {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}

        else:
            payload = str(obj)

        s = json.dumps(payload, default=str, ensure_ascii=False)
        return s[:max_len] + "...[truncated]" if len(s) > max_len else s

    except Exception as e:
        return f"<unserializable {type(obj).__name__}: {e}>"





def create_amazon_api(creds: dict, country: str):
    """Creators API client factory.

    Expects credential_id / credential_secret plus Associates tag.
    Credentials are loaded from config/amazon_credentials.txt via load_amazon_credentials().
    """
    country = (country or "").upper().strip()
    version = (creds.get("CREATORS_API_VERSION") or "2.2").strip()

    def _make(cred_id: str, cred_secret: str, tag: str, ctry: str):
        country_map = {
            "US": Country.US,
            "UK": Country.UK,
            "CA": Country.CA,
        }
        if ctry not in country_map:
            raise ValueError(f"Unsupported country: {ctry}")
        return AmazonCreatorsApi(
            credential_id=cred_id,
            credential_secret=cred_secret,
            version=version,
            tag=tag,
            country=country_map[ctry],
        )

    if country == "US":
        required = ("USA_CREDENTIAL_ID", "USA_CREDENTIAL_SECRET", "USA_TAG")
        if not all(creds.get(k) for k in required):
            raise ValueError("Missing US Creators API credentials: " + ", ".join(required))
        api = _make(creds["USA_CREDENTIAL_ID"], creds["USA_CREDENTIAL_SECRET"], creds["USA_TAG"], "US")
        return api, creds["USA_TAG"], "https://www.amazon.com"

    if country == "UK":
        required = ("UK_CREDENTIAL_ID", "UK_CREDENTIAL_SECRET", "UK_TAG")
        if not all(creds.get(k) for k in required):
            raise ValueError("Missing UK Creators API credentials: " + ", ".join(required))
        api = _make(creds["UK_CREDENTIAL_ID"], creds["UK_CREDENTIAL_SECRET"], creds["UK_TAG"], "UK")
        return api, creds["UK_TAG"], "https://www.amazon.co.uk"

    if country == "CA":
        required = ("CA_CREDENTIAL_ID", "CA_CREDENTIAL_SECRET", "CA_TAG")
        if not all(creds.get(k) for k in required):
            raise ValueError("Missing CA Creators API credentials: " + ", ".join(required))
        api = _make(creds["CA_CREDENTIAL_ID"], creds["CA_CREDENTIAL_SECRET"], creds["CA_TAG"], "CA")
        return api, creds["CA_TAG"], "https://www.amazon.ca"

    raise ValueError(f"Unsupported country: {country}")



# =========================
# Link/Image injection
# =========================

def create_affiliate_link(
    text,
    asin,
    base_url,
    tag,
    context: str | None = None,
    is_recommended=False,
    add_trailer=True,
    cta_cfg: dict | None = None,
    display_text: str | None = None,
    is_substitute: bool = False,
    substitute_note: str | None = None,
    strong_wrap: bool = True,
    price_display: str | None = None,
    price_ts: str | None = None,
    full_title: str | None = None,
):
    """
    Build an affiliate inline link, optionally with:
      - substitute styling/labels
      - a visible price placeholder span next to the link
      - optional inline trailer
      - optional bolding of the *whole unit* (link + price) once

    IMPORTANT:
      - Details/disclaimer IDs MUST be unique per section. This function now ensures that by
        resolving a default context when one is not provided.
    """

    cta_cfg = cta_cfg or CTA_CONFIG_DEFAULT

    shown = normalize_ws(display_text or text)
    if not asin:
        return escape(shown)

    # --- NEW: Resolve a safe default context to avoid duplicate disclaimer IDs ---
    # Priority:
    # 1) caller-provided context
    # 2) if clearly "quick verdict" style (recommended/top pick/cta), use quick-verdict
    # 3) otherwise, use body
    resolved_context = context
    if not resolved_context:
        # Heuristic: quick verdict blocks typically set is_recommended True,
        # or may use a cta_cfg marker if you add one later.
        if is_recommended or cta_cfg.get("context_hint") == "quick-verdict":
            resolved_context = "quick-verdict"
        else:
            resolved_context = "body"
    # ---------------------------------------------------------------------------

    label_inside = escape(shown, quote=True)
    href = f"{base_url}/dp/{asin}?tag={tag}"

    # Compute tooltip text (FULL Amazon title preferred) and truncate to 160 chars for ALL links
    tooltip_source = normalize_ws(full_title or shown)
    tooltip_txt = truncate_tooltip(tooltip_source, 160)
    title_attr = f' title="{escape(tooltip_txt, quote=True)}"' if tooltip_txt else ""

    # Substitute vs normal attributes/classes
    if is_substitute:
        note_txt = substitute_note or "Substitute item Ã¢â‚¬â€ original not currently available"
        aria = f' aria-label="{escape(note_txt, quote=True)}: {escape(tooltip_txt, quote=True)}"'
        extra_class = " aff-substitute"
        data_sub = ' data-sub="1"'
    else:
        aria = f' aria-label="Open Amazon product page for {escape(shown, quote=True)}"'
        extra_class = ""
        data_sub = ""

    # 1) Product title link
    a_tag = (
        f'<a class="aff-inline{extra_class}" data-aff="1"{data_sub} '
        f'href="{href}" target="_blank" rel="sponsored noopener nofollow"{aria}{title_attr}>'
        f'{label_inside}</a>'
    )

    # 2) Price + "Details" + hidden disclaimer
    shown_price = _display_or_see_price(price_display)  # should return "See Price" when missing

    # IMPORTANT: use RESOLVED context to avoid duplicate IDs across sections
    disc_id = _disclaimer_id_for_asin(str(asin), context=resolved_context)

    arrow_html = '<span class="aff-price-arrow" aria-hidden="true"> \u2192</span>'

    price_span = (
        f'<span class="aff-price" data-asin="{escape(str(asin), quote=True)}"'
        f' data-price-ts="{escape(str(price_ts or ""), quote=True)}">'
        f'{escape(str(shown_price))}'
        f'</span>'
    )

    # Make arrow + price clickable (same Amazon href as the title link)
    price_link = (
        f'<a class="aff-inline aff-price-link{extra_class}" data-aff="1"{data_sub} '
        f'href="{href}" target="_blank" rel="sponsored noopener nofollow"'
        f'{aria}{title_attr}>'
        f'{arrow_html}{price_span}'
        f'</a>'
    )

    price_html = (
        f'{price_link}'
        f'<span class="aff-price-meta"> ('
        f'<a class="price-disclaimer-toggle" href="#{disc_id}" '
        f'aria-controls="{disc_id}" aria-expanded="false">Details</a>'
        f')</span>'
        f'<span id="{disc_id}" class="price-disclaimer" hidden>'
        f'{escape(PRICE_DISCLAIMER_TEXT)}'
        f'</span>'
    )

    # 3) Optional trailer (REMOVED by default)
    trailer_html = ""
    if (
        add_trailer
        and cta_cfg.get("show_inline_trailer", False)
        and _inline_trailer_budget("consume", cta_cfg)
    ):
        trailer = random.choice(INLINE_TRAILER_POOL)
        trailer_html = (
            f'&nbsp;<span class="aff-inline-trailer">Ã¢â‚¬â€œ '
            f'<a href="{href}" target="_blank" rel="sponsored noopener nofollow">{escape(trailer)}</a>'
            f'</span>'
        )

    # 4) Bold the WHOLE unit once (link + price)
    unit = a_tag + price_html
    if strong_wrap:
        unit = f"<strong>{unit}</strong>"

    return unit + trailer_html




def _infer_suffix_from_keywords(cfg: dict) -> str:
    nouns = []
    for kw in (cfg.get("include_keywords") or []):
        if not isinstance(kw, str): continue
        toks = re.findall(r"[A-Za-z]+", kw.lower())
        if toks:
            nouns.append(toks[-1])
    return max(nouns, key=len) if nouns else ""



def _product_form_groups(cfg: dict) -> dict[str, list[str]]:
    configured = cfg.get("product_form_groups")
    if isinstance(configured, dict) and configured:
        return {
            str(group): [str(term) for term in terms if str(term).strip()]
            for group, terms in configured.items()
            if isinstance(terms, list)
        }
    return _DEFAULT_PRODUCT_FORM_GROUPS


def _detect_product_form(text: str, cfg: dict) -> tuple[str, str] | None:
    normalized = normalize_name(text or "")
    matches = []
    for group, phrases in _product_form_groups(cfg).items():
        for phrase in phrases:
            normalized_phrase = normalize_name(phrase)
            if normalized_phrase and contains_whole_term(normalized, {normalized_phrase}):
                matches.append((len(normalized_phrase), group, normalized_phrase))
    if not matches:
        return None
    _, group, phrase = max(matches)
    return group, phrase


def _suffix_is_product_form_compatible(name: str, suffix: str, cfg: dict) -> bool:
    requested_form = _detect_product_form(name, cfg)
    suffix_form = _detect_product_form(suffix, cfg)
    return not requested_form or not suffix_form or requested_form[0] == suffix_form[0]


def build_similar_fallback_queries(name: str, cfg: dict) -> list[str]:
    """Create one or two concise, non-contradictory substitute searches."""
    model_tokens = sorted(extract_model_tokens(name), key=len, reverse=True)
    model = model_tokens[0] if model_tokens else ""
    requested_form = _detect_product_form(name, cfg)

    suffixes = cfg.get("variant_suffix") or []
    if isinstance(suffixes, str):
        suffixes = [suffixes]
    inferred = next((str(s).strip() for s in suffixes if str(s).strip()), "")
    if not inferred:
        inferred = _infer_suffix_from_keywords(cfg)

    form_phrase = requested_form[1] if requested_form else inferred
    queries = []
    if model and form_phrase:
        queries.append(f"{model} {form_phrase}")

    audience = ""
    if re.search(r"(?i)\b(?:women|women's|womens|female)\b", name):
        audience = "women"
    elif re.search(r"(?i)\b(?:men|men's|mens|male)\b", name):
        audience = "men"
    elif re.search(r"(?i)\b(?:kids?|children|child|toddler)\b", name):
        audience = "kids"
    if audience and model and form_phrase:
        queries.append(f"{audience} {model} {form_phrase}")

    if not queries:
        queries.append(name)

    limit = max(1, int(cfg.get("amazon_fallback_query_limit", 2)))
    result = []
    seen = set()
    for query in queries:
        key = normalize_ws(query).lower()
        if key and key not in seen:
            seen.add(key)
            result.append(normalize_ws(query))
    return result[:limit]


def generate_query_variants(original: str, cfg: dict) -> list[str]:
    s = normalize_ws(original)
    variants = [s]

    # Removing an editorial/audience parenthetical is the highest-value retry.
    concise = normalize_ws(re.sub(r"\s*\([^)]{1,40}\)\s*", " ", s))
    if concise and concise.lower() != s.lower():
        variants.append(concise)

    suffixes = cfg.get("variant_suffix")
    if isinstance(suffixes, str):
        suffixes = [suffixes]
    if not suffixes:
        inferred = _infer_suffix_from_keywords(cfg)
        suffixes = [inferred] if inferred else []

    # At most one full-name/category retry, and never append a conflicting form
    # such as "backpack" to an explicitly named duffel or tote.
    for suffix in suffixes:
        suffix = normalize_ws(str(suffix))
        if suffix and _suffix_is_product_form_compatible(s, suffix, cfg):
            if not contains_whole_term(normalize_name(s), {normalize_name(suffix)}):
                variants.append(f"{s} {suffix}")
            break

    seen, out = set(), []
    for v in variants:
        v = v.strip()
        if v and v not in seen:
            seen.add(v); out.append(v)
    limit = max(1, int(cfg.get("amazon_exact_query_limit", 2)))
    return out[:limit]

def build_extracted_status_rows(
    filtered_names: list[str],
    selection_records: list[dict],
    unmatched: list[dict],
) -> list[list[str]]:
    """Return one transparent Amazon outcome row for every extracted name."""
    records_by_name = {
        row.get("name"): row for row in selection_records if row.get("name")
    }
    unmatched_by_name = {
        row.get("name"): row for row in unmatched if row.get("name")
    }
    rows = []
    for name in filtered_names:
        record = records_by_name.get(name)
        missing = unmatched_by_name.get(name, {})
        rows.append([
            name,
            (record or missing).get("match_status", "unmatched"),
            (record or {}).get("asin", ""),
            (record or {}).get("amazon_title") or (record or {}).get("label", ""),
            (record or {}).get("duplicate_of", ""),
            missing.get("reason", ""),
        ])
    return rows


def build_image_selection_rows(
    product_data: list[dict],
    cfg: dict | None = None,
) -> list[list[str]]:
    """Build the image audit, including explicitly disclosed close-model fallbacks."""
    cfg = cfg or {}
    image_policy = cfg.get("section_images") or {}
    allow_close_substitute = bool(image_policy.get("allow_close_substitute", True))
    rows = []
    for pd in product_data:
        image_eligible = bool(
            pd.get("img_url")
            and (
                not pd.get("is_substitute")
                or allow_close_substitute
            )
        )
        rows.append([
            pd.get("name", ""), pd.get("match_status", ""), pd.get("asin", ""),
            pd.get("amazon_title") or pd.get("label", ""),
            "yes" if image_eligible else "no",
            pd.get("img_url", ""), pd.get("img_size", ""),
            pd.get("img_width", ""), pd.get("img_height", ""),
            pd.get("img_dimension_status", ""), pd.get("img_content_ratio", ""),
        ])
    return rows



def section_classifier(section_patterns: dict):
    compiled = {sec: [re.compile(p, re.I) for p in pats] for sec, pats in (section_patterns or {}).items()}
    def classify(h_text: str) -> str:
        t = (h_text or "").strip().lower()
        for sec, regs in compiled.items():
            for rx in regs:
                if rx.search(t):
                    return sec
        return "general"
    return classify
    
def remove_trailers_in_tables(html: str) -> str:
    if not html:
        return html
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table"):
        for span in table.select("span.aff-inline-trailer"):
            span.decompose()  # drop the "Ã¢â‚¬â€œ check latest price \u2192" trailers
    return str(soup)
    
# put this near remove_trailers_in_tables (top level, not nested)
def remove_links_in_tables(html: str) -> str:
    if not html:
        return html
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table"):
        for a in table.find_all("a"):
            a.unwrap()  # drop the link, keep inner content
    return str(soup)

def format_image_html(
    asin: str,
    img_url: str,
    title: str,
    base_url: str,
    tag: str,
    css_class: str = "float-image",
    cta_text: str | None = None,
    show_cta: bool = True,
    cta_cfg: dict | None = None,
    caption_text: str | None = None,
) -> str:
    """
    Clickable affiliate image ONLY.
    (No under-image CTA button. Sticky CTA is handled elsewhere and remains.)
    """
    if not asin or not img_url:
        return ""

    href = _aff_url(base_url, asin, tag)
    if not href:
        return ""

    full = normalize_ws(title or "")
    tooltip = truncate_tooltip(full, 160)

    safe_title_full = escape(full, quote=True)
    safe_title_tooltip = escape(tooltip, quote=True)


    parts: list[str] = []
    parts.append(
        f'<figure class="{escape(css_class, quote=True)}" data-asin="{escape(asin, quote=True)}">'
    )

    # âœ… Keep clickable image
    parts.append(
        f'<a class="aff-img-link" href="{href}" target="_blank" rel="sponsored noopener nofollow" '
        f'aria-label="Open Amazon product page for {safe_title_full}">'
        f'<img src="{escape(img_url, quote=True)}" alt="{safe_title_full}" title="{safe_title_tooltip}" loading="lazy" />'
        f"</a>"
    )


    # Ã¢ÂÅ’ Remove the under-image "View on Amazon" button by not rendering it at all.
    # (Ignore show_cta/cta_text intentionally.)

    if caption_text:
        parts.append(
            f'<figcaption class="aff-image-disclosure">{escape(normalize_ws(caption_text))}</figcaption>'
        )
    parts.append("</figure>")
    return "\n".join(parts)

import re
from bs4 import BeautifulSoup

def drop_empty_pros_cons_columns(html: str) -> str:
    if not html:
        return html

    soup = BeautifulSoup(html, "html.parser")

    def is_effectively_empty(cell) -> bool:
        """True if cell has no real text and no meaningful elements."""
        if cell is None:
            return True

        # Remove empty lists (common when generator outputs <ul></ul>)
        for ul in cell.find_all(["ul", "ol"]):
            # If list has no non-empty <li>, drop it
            lis = [li.get_text(" ", strip=True) for li in ul.find_all("li")]
            if not any(t for t in lis):
                ul.decompose()

        text = cell.get_text(" ", strip=True)
        # Treat &nbsp; etc. as empty after stripping
        text = re.sub(r"\u00A0+", "", text).strip()
        return text == ""

    # Prefer tables that are explicitly "comparison-table"
    tables = soup.select("table.comparison-table") or soup.find_all("table")

    for table in tables:
        # Only act on tables inside the comparison wrapper OR with the class
        in_wrap = table.find_parent(class_=re.compile(r"\bcomparison-table-wrap\b")) is not None
        is_cmp  = "comparison-table" in (table.get("class") or [])
        if not (in_wrap or is_cmp):
            continue

        rows = table.find_all("tr")
        if not rows:
            continue

        # Find header row (first row containing th)
        header_row = None
        for r in rows:
            if r.find("th"):
                header_row = r
                break
        if not header_row:
            continue

        headers = header_row.find_all(["th", "td"], recursive=False)
        if not headers:
            continue

        header_texts = [h.get_text(" ", strip=True).lower() for h in headers]

        # Identify Pros/Cons indices (case-insensitive exact match)
        target_idxs = []
        for i, txt in enumerate(header_texts):
            if txt in ("pros", "cons"):
                target_idxs.append(i)

        if not target_idxs:
            continue

        # Decide which of those columns are fully empty across all data rows
        drop_idxs = []
        for idx in target_idxs:
            col_has_content = False
            for r in rows:
                # Skip header rows
                if r == header_row:
                    continue
                cells = r.find_all(["td", "th"], recursive=False)
                if idx >= len(cells):
                    continue
                if not is_effectively_empty(cells[idx]):
                    col_has_content = True
                    break
            if not col_has_content:
                drop_idxs.append(idx)

        if not drop_idxs:
            continue

        # Remove from rightmost to leftmost so indices don't shift
        for idx in sorted(drop_idxs, reverse=True):
            # Remove header cell
            hdr_cells = header_row.find_all(["th", "td"], recursive=False)
            if idx < len(hdr_cells):
                hdr_cells[idx].decompose()

            # Remove each row's cell at that index (if present)
            for r in rows:
                if r == header_row:
                    continue
                cells = r.find_all(["td", "th"], recursive=False)
                if idx < len(cells):
                    cells[idx].decompose()

    return str(soup)



def inject_links_and_images(content, product_data, country, base_url, tag, top_pick_name=None, cfg: dict | None = None):
    cfg = cfg or {}
    cta_cfg = get_cta_cfg(cfg) or CTA_CONFIG_DEFAULT
    TOP_PICK_MAX_LINKS = int(cfg.get("top_pick_max_links", 3))
    MAX_SECTION_LINKS = cfg.get("max_section_links", {"catalog": 3, "purpose": 1, "verdict": 1, "general": 1})
    section_patterns = cfg.get("section_patterns", {
        "purpose": ["purpose of the review"],
        "catalog": ["other popular", "other .* models"],
        "verdict": ["final verdict", "value for money"],
        "general": [".*"]
    })
    classify_section = section_classifier(section_patterns)
    ALIAS_FUZZ_THRESHOLD = 86
    MAX_ITEM_IMAGES = MAX_ITEM_IMAGES_DEFAULT

    top_pick_extra_budget = 1
    
    DEBUG_SUBS = bool(cfg.get("debug_substitutes", True))# default ON while debugging

    def _dbg(msg: str, **fields):
        if not DEBUG_SUBS:
            return
        log.info(
            msg,
            extra={
                "step": "inject.substitute_debug",
                "extra_json": fields,
            },
        )



    def build_tolerant_phrase_regex(phrase: str) -> re.Pattern:
        tokens = re.findall(r"\w+", normalize_quotes(phrase))
        if not tokens:
            return re.compile(r"(?!x)x")
        if len(tokens) < 2:
            return re.compile(rf"(?<!\w){re.escape(tokens[0])}(?!\w)", re.IGNORECASE | re.UNICODE)
        between = r"[^\w]{0,3}"
        pattern = r"(?<!\w)" + between.join(map(re.escape, tokens)) + r"(?!\w)"
        return re.compile(pattern, re.IGNORECASE | re.UNICODE)

    def section_type_from_heading(h_text: str) -> str:
        heading = normalize_ws(h_text or "").casefold()
        # Category-neutral semantic fallbacks cover modern outlines whose labels
        # differ from older configured templates.
        if re.search(r"\bwho\s+is\s+it\s+for\b|\bwho\s+should\s+(?:buy|use)\b", heading):
            return "purpose"
        if re.search(r"\bvalue\s+for\s+money\b|\bfinal\s+verdict\b", heading):
            return "verdict"
        sec = classify_section(h_text or "")
        return "item" if sec == "catalog" else sec

    content = normalize_quotes(content)

    # Build a set of "generic" tokens so we don't create alias links like "air purifier".
    # This reuses the same config-driven idea as substitute filtering, but applies it to alias generation.
    GENERIC_TOKENS: set[str] = set()
    for k in ("generic_adjectives", "generic_tails", "include_keywords"):
        for s in (cfg.get(k) or []):
            if isinstance(s, str) and s.strip():
                GENERIC_TOKENS.update(re.findall(r"[A-Za-z0-9]+", s.lower()))
    # common glue words
    GENERIC_TOKENS.update({"and", "or", "with", "for", "the", "a", "an", "of", "to", "in", "on"})

    # Normalize include_keywords phrases too (so we can block exact matches like "air purifier")
    _include_kw_norm = set()
    for s in (cfg.get("include_keywords") or []):
        if isinstance(s, str) and s.strip():
            _include_kw_norm.add(normalize_name(s))

    alias_map = {}
    product_name_to_data = {}
    for pd in product_data:
        full_norm = normalize_name(pd["name"])
        pd["normalized_name"] = full_norm
        product_name_to_data[full_norm] = pd
        toks = full_norm.split()
        for i in range(len(toks)):
            for j in range(i + 2, len(toks) + 1):
                alias = " ".join(toks[i:j])

                # Skip aliases that are purely generic/category phrases (e.g., "air purifier")
                alias_toks = alias.split()
                meaningful = [t for t in alias_toks if len(t) >= 3 and t.lower() not in GENERIC_TOKENS]
                if not meaningful:
                    continue
                if alias in _include_kw_norm:
                    continue

                if alias not in product_name_to_data and alias not in alias_map:
                    alias_map[alias] = full_norm

    sorted_full_names = sorted(product_name_to_data.keys(), key=len, reverse=True)
    sorted_aliases = sorted(alias_map.keys(), key=len, reverse=True)
    product_asin_to_data = {pd["asin"]: pd for pd in product_data}
    product_phrase_patterns = {
        name_key: build_tolerant_phrase_regex(product_name_to_data[name_key]["name"])
        for name_key in sorted_full_names
    }

    def _is_crowded_product_text(txt_for_matching: str, current_name_key: str | None = None) -> bool:
        """
        Product roundups often start with one sentence that lists every reviewed model.
        Prefer later, model-specific paragraphs for affiliate links when they exist.
        """
        if not txt_for_matching:
            return False

        mentioned: set[str] = set()
        for name_key, phrase_rx in product_phrase_patterns.items():
            try:
                if phrase_rx.search(txt_for_matching):
                    mentioned.add(name_key)
            except Exception:
                continue
            if len(mentioned) >= 2:
                return True

        return False

    def _has_affiliate_link(node) -> bool:
        if node is None:
            return False
        if getattr(node, "name", None) == "a":
            classes = node.get("class") or []
            return any(c in {"aff-inline", "aff-price-link", "aff-substitute"} for c in classes)
        if getattr(node, "name", None):
            return bool(node.select_one("a.aff-inline, a.aff-price-link, span.aff-sub-note"))
        return False

    def _is_paragraph_boundary(node) -> bool:
        if node is None:
            return True
        if isinstance(node, NavigableString):
            return "\n\n" in str(node)
        return getattr(node, "name", None) in {
            "h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "ul", "ol",
            "table", "figure", "div", "blockquote", "hr", "br"
        }

    def _text_cluster_already_has_affiliate_link(node, txt: str | None = None, match_start: int | None = None) -> bool:
        """
        Generated sections are often raw HTML fragments without <p> tags. Treat text
        between blank lines/block tags as a paragraph-like cluster and allow only one
        affiliate insertion inside it.
        """
        parent = getattr(node, "parent", None)
        if parent is None:
            return False

        block_parent = node.find_parent(["p", "li"])
        if block_parent is not None:
            return _has_affiliate_link(block_parent)

        txt = str(node) if txt is None else txt
        match_start = 0 if match_start is None else max(0, match_start)
        before_match = txt[:match_start]
        after_match = txt[match_start:]

        if "\n\n" not in before_match:
            cur = node.previous_sibling
            while cur is not None and not _is_paragraph_boundary(cur):
                if _has_affiliate_link(cur):
                    return True
                cur = cur.previous_sibling

        if "\n\n" not in after_match:
            cur = node.next_sibling
            while cur is not None and not _is_paragraph_boundary(cur):
                if _has_affiliate_link(cur):
                    return True
                cur = cur.next_sibling

        return False

    recommended = None
    if top_pick_name:
        recommended = product_name_to_data.get(normalize_name(top_pick_name))
    if not recommended and product_data:
        recommended = max(product_data, key=lambda x: x.get('score', 0))
    if not recommended:
        return content

    recommended_asin = recommended["asin"]
    recommended_img_url = recommended["img_url"]
    recommended_display = get_display_title(recommended)
    recommended_key = recommended["normalized_name"]

    global_link_counts: dict[str, int] = {}

    substitute_policy = cfg.get("substitute_links") or {}
    image_policy = cfg.get("section_images") or {}
    required_image_sections = {
        str(value).strip().lower()
        for value in (image_policy.get("required_sections") or ["purpose", "verdict"])
        if str(value).strip()
    }
    allow_close_substitute_image = bool(
        image_policy.get("allow_close_substitute", True)
    )
    max_substitute_images = max(
        0, int(image_policy.get("max_substitute_images_per_post", 2))
    )
    substitute_images_used = 0

    def _substitute_link_allowed(pd: dict, section_type: str) -> bool:
        if not pd.get("is_substitute"):
            return True
        if substitute_policy.get("enabled", True) is False:
            return False

        scope = str(substitute_policy.get("scope", "top_pick_only")).strip().lower()
        is_primary_name = (
            normalize_name(pd.get("name", "")) == recommended_key
        )
        if scope in {"none", "disabled"}:
            return False
        if scope == "top_pick_only" and not is_primary_name:
            return False
        if scope == "catalog_only" and section_type != "item":
            return False
        if scope == "top_pick_or_catalog" and not (
            is_primary_name or section_type == "item"
        ):
            return False
        if scope not in {
            "top_pick_only",
            "catalog_only",
            "top_pick_or_catalog",
            "all_unavailable",
        }:
            return False

        max_per_post = max(0, int(substitute_policy.get("max_per_post", 1)))
        used = sum(
            global_link_counts.get(name_key, 0)
            for name_key, product in product_name_to_data.items()
            if product.get("is_substitute")
        )
        if used >= max_per_post:
            return False

        # A substitute must have passed the existing product-form and
        # specialization gates during Amazon selection. This stage controls
        # editorial scope: secondary unavailable models remain plain text.
        return bool(pd.get("asin"))


    parts = re.split(r"(<h[2-4]>.*?</h[2-4]>)", content)
    output = []
    seen_headings = set()

    def count_existing_text_links_to_products(soup_section):
        count = 0
        top_pick_count = 0
        seen_local = set()
        for a in soup_section.select('a[href]'):
            # ignore any link that is located inside a table
            if a.find_parent('table') is not None:
                continue

            if not a.get_text(strip=True):
                continue
            href = a['href']
            for full_key, pdx in product_name_to_data.items():
                if f"/dp/{pdx['asin']}" in href:
                    key = (full_key, href, a.get_text(strip=True))
                    if key in seen_local:
                        continue
                    seen_local.add(key)
                    count += 1
                    global_link_counts[full_key] = global_link_counts.get(full_key, 0) + 1
                    if full_key == recommended_key:
                        top_pick_count += 1
        return count, top_pick_count
    

    _LEADING_PUNCT_RX = re.compile(r"^(\s*[\)\],.;:]+(?:\s+)?)")

    def _pull_leading_punct_into_wrapper(wrapper):
        # walk forward a few siblings to find the first text node with punctuation
        nxt = wrapper.next_sibling

        # skip whitespace-only nodes
        while isinstance(nxt, NavigableString) and not str(nxt).strip():
            nxt = nxt.next_sibling

        # if next is a tag, try to pull punctuation from its first text node (common with <br>, <p>, etc.)
        candidates = []
        if nxt is not None:
            candidates.append(nxt)
            # also check one more sibling just in case
            if getattr(nxt, "next_sibling", None) is not None:
                candidates.append(nxt.next_sibling)

        for cand in candidates:
            # Case A: direct text node
            if isinstance(cand, NavigableString):
                s = str(cand)
                m = _LEADING_PUNCT_RX.match(s)
                if not m:
                    continue

                punct = m.group(1)
                rest = s[len(punct):]
                wrapper.append(NavigableString(punct))
                if rest == "":
                    cand.extract()
                else:
                    cand.replace_with(NavigableString(rest))
                return

            # Case B: tag that contains a leading text node
            if getattr(cand, "name", None):
                # find first text node inside cand
                txt_node = next((t for t in cand.contents if isinstance(t, NavigableString)), None)
                if txt_node is None:
                    continue

                s = str(txt_node)
                m = _LEADING_PUNCT_RX.match(s)
                if not m:
                    continue

                punct = m.group(1)
                rest = s[len(punct):]
                wrapper.append(NavigableString(punct))
                if rest == "":
                    txt_node.extract()
                else:
                    txt_node.replace_with(NavigableString(rest))
                return

    # --- helper: convert <figure> to inline <span> for inline-run contexts (avoids wpautop gaps)
    _FIG_OPEN_RX = re.compile(r"<\s*figure(\s[^>]*)?>", re.I)
    _FIG_CLOSE_RX = re.compile(r"</\s*figure\s*>", re.I)

    def _figures_to_inline_spans(html: str) -> str:
        """
        Convert <figure ...>...</figure> to <span ...>...</span>.
        This prevents WP/wpautop/themes from treating the injected image as a block break mid-sentence.
        """
        if not html:
            return html
        html = _FIG_OPEN_RX.sub(r"<span\1>", html)
        html = _FIG_CLOSE_RX.sub(r"</span>", html)
        return html

       

    def _insert_images_near_first_aff_link(processed_html: str, img_fragments: list[str]) -> str:
        """
        Insert each image fragment close to its *matching* affiliate TEXT link (same ASIN),
        never inside tables.

        Key behavior:
          - Prefer a.aff-inline (product name) and a.aff-price-link for placement.
          - Do NOT treat a.aff-img-link as the target (that causes the "image near intro" bug).
          - If we must wrap an inline run, only pull a SMALL lead-in (e.g. "The ", "Lastly, the ").
          - Pull leading punctuation like ", " or ") " into the wrapper so it doesn't strand outside.
        """
        if not processed_html or not img_fragments:
            return processed_html

        soup = BeautifulSoup(processed_html, "html.parser")

        def _cleanup_broken_md_bold_markers(_soup: BeautifulSoup) -> None:
            """Remove stray markdown ** that can get split by injected image/link blocks.

            In the wild this shows up as a literal line like: "The **" before our affiliate block.
            """

            # 0) Remove any standalone "**" nodes anywhere.
            for t in list(_soup.find_all(string=True)):
                if isinstance(t, NavigableString) and t.strip() == "**":
                    t.extract()

            def _prev_text_node(node):
                cur = node
                while cur is not None:
                    cur = cur.previous_sibling
                    if cur is None:
                        break
                    if isinstance(cur, NavigableString):
                        if cur.strip() == "":
                            continue
                        return cur
                    if getattr(cur, "name", None) in ("figure", "span", "div"):
                        continue
                return None

            def _next_text_node(node):
                cur = node
                while cur is not None:
                    cur = cur.next_sibling
                    if cur is None:
                        break
                    if isinstance(cur, NavigableString):
                        if cur.strip() == "":
                            continue
                        return cur
                    if getattr(cur, "name", None) in ("figure", "span", "div"):
                        continue
                return None

            # 1) Remove "**" split around the affiliate <strong> block (even if a figure sits between).
            for _strong in _soup.find_all("strong"):
                if not _strong.find("a", class_=lambda c: c and "aff-inline" in c):
                    continue

                _prev_txt = _prev_text_node(_strong)
                if isinstance(_prev_txt, NavigableString):
                    _s = str(_prev_txt)
                    _new = re.sub(r"\*\*\s*$", "", _s)
                    if _new != _s:
                        _prev_txt.replace_with(NavigableString(_new))

                _next_txt = _next_text_node(_strong)
                if isinstance(_next_txt, NavigableString):
                    _s = str(_next_txt)
                    _new = re.sub(r"^\s*\*\*\s*", " ", _s)
                    if _new != _s:
                        _next_txt.replace_with(NavigableString(_new))

            # 2) paragraph/list-item text ends with "**"
            for _p in _soup.find_all(["p", "li"]):
                if not _p.contents:
                    continue
                _last = _p.contents[-1]
                if isinstance(_last, NavigableString):
                    _s = str(_last)
                    _new = re.sub(r"\*\*\s*$", "", _s)
                    if _new != _s:
                        _last.replace_with(NavigableString(_new))

            # 3) final safety net
            for t in list(_soup.find_all(string=True)):
                if isinstance(t, NavigableString) and "**" in str(t):
                    t.replace_with(NavigableString(str(t).replace("**", "")))

        # ---- Build asin -> fragment map from img_fragments
        asin_to_frag: dict[str, str] = {}
        for frag in img_fragments:
            try:
                fs = BeautifulSoup(frag, "html.parser")
                asin_tag = fs.find(attrs={"data-asin": True})
                asin = (asin_tag.get("data-asin") or "").strip().upper() if asin_tag else ""
                if asin:
                    asin_to_frag[asin] = frag
            except Exception:
                continue

        # ---------- helpers ----------
        inline_names = {"a", "strong", "span", "em", "b", "i", "small"}
        # small lead-in text only; prevents pulling whole paragraphs into wrapper
        def _is_small_leadin_text(s: str) -> bool:
            t = (s or "")
            # ignore pure whitespace/newlines
            if not t.strip():
                return True
            # don't pull back across sentence boundaries
            if any(p in t for p in (".", "!", "?", "\n")):
                return False
            # keep only short lead-ins like "The " / "Lastly, the "
            return len(t.strip()) <= 24

        def _extract_asin_from_href(u: str) -> str | None:
            if not u:
                return None
            m = (
                re.search(r"/dp/([A-Z0-9]{10})", u, re.I)
                or re.search(r"/gp/product/([A-Z0-9]{10})", u, re.I)
                or re.search(r"(?:[?&]asin=)([A-Z0-9]{10})", u, re.I)
            )
            return m.group(1).upper() if m else None

        # Prefer TEXT links for the ASIN. Only fall back to aff-img-link if needed.
        def find_target_for_asin(asin: str):
            asin = (asin or "").strip().upper()
            if not asin:
                return None

            # 1) a.aff-inline (product name link) with matching ASIN
            for a in soup.select("a.aff-inline[href]"):
                if a.find_parent("table") is not None:
                    continue
                if _extract_asin_from_href(a.get("href", "")) == asin:
                    return a

            # 2) a.aff-price-link (price link) with matching ASIN
            for a in soup.select("a.aff-price-link[href]"):
                if a.find_parent("table") is not None:
                    continue
                if _extract_asin_from_href(a.get("href", "")) == asin:
                    return a

            # 3) price span data-asin -> climb to <a> if any
            price_span = soup.find(
                lambda t: getattr(t, "name", None) == "span"
                and (t.get("data-asin") or "").strip().upper() == asin
                and t.find_parent("table") is None
            )
            if price_span:
                return price_span.find_parent("a") or price_span

            # 4) LAST resort: aff-img-link (but this is what caused your separation before)
            for a in soup.select("a.aff-img-link[href]"):
                if a.find_parent("table") is not None:
                    continue
                if _extract_asin_from_href(a.get("href", "")) == asin:
                    return a

            return None

        # ---------------- fallback (no asins) ----------------
        if not asin_to_frag:
            # Old behavior: insert block near first affiliate link outside tables.
            first_aff = None
            for a in soup.select("a.aff-inline, a.aff-price-link, a.aff-img-link"):
                if a.find_parent("table") is None:
                    first_aff = a
                    break

            img_block_html = _image_row(img_fragments) if len(img_fragments) >= 3 else "".join(img_fragments)
            img_block = BeautifulSoup(img_block_html, "html.parser")

            if not first_aff:
                soup.insert(0, img_block)
                _cleanup_broken_md_bold_markers(soup)
                return _figures_to_inline_spans(str(soup))

            host = first_aff.find_parent(["p", "li"])
            if host:
                host.insert_before(img_block)
                _cleanup_broken_md_bold_markers(soup)
                return _figures_to_inline_spans(str(soup))

            wrapper = soup.new_tag("span")
            wrapper["class"] = ["aff-inline-run"]

            # wrap a small inline run around the first link
            anchor = first_aff
            start_node = anchor
            prev = start_node.previous_sibling
            while prev is not None and (
                isinstance(prev, NavigableString) or getattr(prev, "name", None) in inline_names
            ):
                if isinstance(prev, NavigableString) and not _is_small_leadin_text(str(prev)):
                    break
                start_node = prev
                prev = prev.previous_sibling

            start_node.insert_before(wrapper)

            node = start_node
            while node is not None:
                nxt = node.next_sibling
                if getattr(node, "name", None) and node.name not in inline_names:
                    break
                wrapper.append(node.extract())
                node = nxt

            _pull_leading_punct_into_wrapper(wrapper)

            wrapper.insert(0, img_block)
            _cleanup_broken_md_bold_markers(soup)
            return _figures_to_inline_spans(str(soup))

        # ---------------- ASIN-matched placement ----------------
        placed_any = False

        for asin, frag in asin_to_frag.items():
            target = find_target_for_asin(asin)
            if not target:
                continue

            img_node = BeautifulSoup(frag, "html.parser")

            # If target already sits inside a wrapper, prepend image there.
            existing_run = target.find_parent(class_=lambda c: isinstance(c, list) and "aff-inline-run" in c)
            if existing_run:
                existing_run.insert(0, img_node)
                placed_any = True
                continue

            # If target is inside a paragraph/list item, place image right before that paragraph.
            host = target.find_parent(["p", "li"])
            if host:
                host.insert_before(img_node)
                placed_any = True
                continue

            # Otherwise wrap a SMALL inline run that includes the target link text.
            wrapper = soup.new_tag("span")
            wrapper["class"] = ["aff-inline-run"]

            # Find an anchor that has siblings (climb up if necessary)
            anchor = target
            while anchor is not None:
                prev = anchor.previous_sibling
                while isinstance(prev, NavigableString) and not str(prev).strip():
                    prev = prev.previous_sibling
                if prev is not None:
                    break
                parent = getattr(anchor, "parent", None)
                if parent is None or parent is soup:
                    break
                anchor = parent

            # Now find start of run, but only pull back short lead-in text (no whole sentences)
            start_node = anchor
            prev = start_node.previous_sibling
            while prev is not None and (
                isinstance(prev, NavigableString) or getattr(prev, "name", None) in inline_names
            ):
                if isinstance(prev, NavigableString) and not _is_small_leadin_text(str(prev)):
                    break
                if isinstance(prev, NavigableString) and not prev.strip():
                    prev = prev.previous_sibling
                    continue
                start_node = prev
                prev = prev.previous_sibling

            start_node.insert_before(wrapper)

            node = start_node
            while node is not None:
                nxt = node.next_sibling
                if getattr(node, "name", None) and node.name not in inline_names:
                    break
                wrapper.append(node.extract())
                node = nxt

            _pull_leading_punct_into_wrapper(wrapper)

            # Put image at the start of the wrapper so it stays "with" the link(s)
            wrapper.insert(0, img_node)
            placed_any = True

        if placed_any:
            _cleanup_broken_md_bold_markers(soup)
            return _figures_to_inline_spans(str(soup))

        # If we couldn't place any by ASIN, prepend as a last resort.
        img_block_html = _image_row(img_fragments) if len(img_fragments) >= 3 else "".join(img_fragments)
        soup.insert(0, BeautifulSoup(img_block_html, "html.parser"))
        _cleanup_broken_md_bold_markers(soup)
        return _figures_to_inline_spans(str(soup))




    def _safe_inline_insert(txt: str, start: int, end: int, insert_html: str) -> str:
        left = txt[:start]
        right = txt[end:]

        # add a space if weâ€™re joining wordchars
        if left and left[-1].isalnum() and insert_html and not insert_html.startswith((" ", "\n", "\t")):
            insert_html = " " + insert_html
        if right and right[0].isalnum() and not insert_html.endswith((" ", "\n", "\t")):
            insert_html = insert_html + " "

        return left + insert_html + right

            
    def _replace_match_with_optional_paragraph_note(
        node,
        txt: str,
        start: int,
        end: int,
        replacement_html: str,
        paragraph_end_html: str | None = None,
    ) -> None:
        """Replace a match and place substitute disclosure after the paragraph."""
        new_txt = _safe_inline_insert(txt, start, end, replacement_html)
        if not paragraph_end_html:
            node.replace_with(BeautifulSoup(new_txt, "html.parser"))
            return

        host = node.find_parent(["p", "li"])
        if host is not None:
            node.replace_with(BeautifulSoup(new_txt, "html.parser"))
            host.append(NavigableString(" "))
            fragment = BeautifulSoup(paragraph_end_html, "html.parser")
            for child in list(fragment.contents):
                host.append(child.extract())
            return

        # Some generated content is a paragraph-like raw text fragment rather
        # than a valid <p>. Append before its trailing whitespace in that case.
        trailing = re.search(r"\s*$", new_txt)
        insert_at = trailing.start() if trailing else len(new_txt)
        new_txt = (
            new_txt[:insert_at].rstrip()
            + " "
            + paragraph_end_html
            + new_txt[insert_at:]
        )
        node.replace_with(BeautifulSoup(new_txt, "html.parser"))


    def _reorder_section_images_text_tables(processed_html: str) -> str:
        """
        For the 1.1 section:
          - keep affiliate images at the top
          - keep all text (and links) in the middle
          - move any comparison table(s) to the bottom
        """
        if not processed_html:
            return processed_html
        soup = BeautifulSoup(processed_html, "html.parser")

        def _cleanup_broken_md_bold_markers(_soup: BeautifulSoup) -> None:
            """Remove stray markdown ** that can get split by injected image/link blocks.

            Root cause in your examples:
              - Original text had Markdown bold like: "The **Product Name** ..."
              - We replace "Product Name" with an affiliate <strong>...</strong> block
              - The remaining "**" can be separated by an injected <figure>/<span> and/or wpautop,
                leaving visible "The **" on its own line.
            """

            # 0) Remove any standalone "**" nodes anywhere (very safe).
            for t in list(_soup.find_all(string=True)):
                if isinstance(t, NavigableString) and t.strip() == "**":
                    t.extract()

            # Helper: find nearest non-empty text node before/after, skipping tags like figure/span.
            def _prev_text_node(node):
                cur = node
                while cur is not None:
                    cur = cur.previous_sibling
                    if cur is None:
                        break
                    if isinstance(cur, NavigableString):
                        if cur.strip() == "":
                            continue
                        return cur
                    # skip over common injected blocks
                    if getattr(cur, "name", None) in ("figure", "span", "div"):
                        continue
                return None

            def _next_text_node(node):
                cur = node
                while cur is not None:
                    cur = cur.next_sibling
                    if cur is None:
                        break
                    if isinstance(cur, NavigableString):
                        if cur.strip() == "":
                            continue
                        return cur
                    if getattr(cur, "name", None) in ("figure", "span", "div"):
                        continue
                return None

            # 1) Remove "**" split around the affiliate <strong> block (even if a figure sits between).
            for _strong in _soup.find_all("strong"):
                if not _strong.find("a", class_=lambda c: c and "aff-inline" in c):
                    continue

                _prev_txt = _prev_text_node(_strong)
                if isinstance(_prev_txt, NavigableString):
                    _s = str(_prev_txt)
                    _new = re.sub(r"\*\*\s*$", "", _s)
                    if _new != _s:
                        _prev_txt.replace_with(NavigableString(_new))

                _next_txt = _next_text_node(_strong)
                if isinstance(_next_txt, NavigableString):
                    _s = str(_next_txt)
                    _new = re.sub(r"^\s*\*\*\s*", " ", _s)
                    if _new != _s:
                        _next_txt.replace_with(NavigableString(_new))

            # 2) Paragraph/list-item text ends with "**" just before an affiliate block.
            for _p in _soup.find_all(["p", "li"]):
                if not _p.contents:
                    continue
                _last = _p.contents[-1]
                if isinstance(_last, NavigableString):
                    _s = str(_last)
                    _new = re.sub(r"\*\*\s*$", "", _s)
                    if _new != _s:
                        _last.replace_with(NavigableString(_new))

            # 3) As a final safety net, strip any remaining "**" tokens.
            # This generator outputs HTML; Markdown bold markers should not be needed at this stage.
            for t in list(_soup.find_all(string=True)):
                if isinstance(t, NavigableString) and "**" in str(t):
                    t.replace_with(NavigableString(str(t).replace("**", "")))

        # ---- 1) Extract table WRAPPERS first (preferred), else tables directly
        extracted_tables = []

        # capture wrappers like <div class="comparison-table-wrap">...</div>
        wraps = soup.find_all(
            lambda t: getattr(t, "name", None)
            and isinstance(t.get("class"), list)
            and any("comparison-table-wrap" == c for c in t.get("class"))
        )
        for w in wraps:
            extracted_tables.append(w.extract())

        # capture any remaining tables (not already extracted via wrapper)
        for tbl in soup.find_all("table"):
            extracted_tables.append(tbl.extract())

        # ---- 2) Extract our injected image figures (or any affiliate image blocks)
        extracted_imgs = []

        # Primary: your injected markup uses <figure class="float-image"> with <a class="aff-img-link">
        for fig in soup.find_all("figure"):
            cls = " ".join(fig.get("class") or []).lower()
            if "float-image" in cls or fig.select_one("a.aff-img-link"):
                extracted_imgs.append(fig.extract())

        # Fallback: in case older markup put clickable images outside <figure>
        # (keep this conservative so we don't hoover random content images)
        for a in soup.select("a.aff-img-link"):
            host = a.find_parent("figure") or a
            if host and host not in extracted_imgs:
                extracted_imgs.append(host.extract())

        # ---- 3) Rebuild: images first, then remaining content, then tables
        rebuilt = BeautifulSoup("", "html.parser")

        if extracted_imgs:
            # if 3+ images, wrap into the same row style used elsewhere
            img_html = _image_row([str(x) for x in extracted_imgs]) if len(extracted_imgs) >= 3 else "".join(str(x) for x in extracted_imgs)
            rebuilt.append(BeautifulSoup(img_html, "html.parser"))

        # remaining (text + links, etc.)
        for node in list(soup.contents):
            rebuilt.append(node.extract())

        # tables last
        for t in extracted_tables:
            rebuilt.append(t)

        return str(rebuilt)

    
    # helper: wrap any image fragments in a row and force subsequent text below
    def _image_row(html_snippets: list[str]) -> str:
        if not html_snippets:
            return ""
        return (
            '<div class="image-row" style="display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start;">'
            + "".join(html_snippets) +
            '</div><div style="clear:both"></div>'  # safety even if older float CSS applies
        )
       
    for i, chunk in enumerate(parts):
        if re.match(r"<h[2-4]>", chunk or ""):
            h_text = BeautifulSoup(chunk, "html.parser").get_text(strip=True)
            if h_text.lower() in seen_headings:
                continue
            seen_headings.add(h_text.lower())
            output.append(chunk)
            continue

        # Preserve content before the first H2/H3/H4 (Quick Verdict block lives here)
        if i == 0:
            output.append(chunk)
            continue

        prev_heading_text = BeautifulSoup(parts[i - 1], "html.parser").get_text(strip=True)
        sec_type = section_type_from_heading(prev_heading_text)
        json_key = "catalog" if sec_type == "item" else sec_type
        sec_cap = int(MAX_SECTION_LINKS.get(json_key, 1))

        section_soup = BeautifulSoup(chunk, "html.parser")

        catalog_cleanup_patterns = cfg.get("catalog_list_removal_heading_patterns") or []
        if any(
            re.search(str(pattern), prev_heading_text, re.I)
            for pattern in catalog_cleanup_patterns
            if str(pattern).strip()
        ):
            for _list in section_soup.find_all(["ul", "ol"]):
                _list.decompose()

        existing_text_links, existing_top_pick_links = count_existing_text_links_to_products(section_soup)

        if sec_type not in ("purpose", "verdict") and existing_top_pick_links > 0:
            top_pick_extra_budget = max(0, top_pick_extra_budget - existing_top_pick_links)

        remaining = max(0, sec_cap - existing_text_links)

        if remaining == 0:
            processed_html = str(section_soup)
        else:
            placeholders = {}
            for idx_a, a in enumerate(section_soup.find_all("a")):
                ph = f"__A_PLACEHOLDER_{idx_a}__"
                placeholders[ph] = str(a)
                a.replace_with(ph)

            temp_html = str(section_soup)
            temp_soup = BeautifulSoup(temp_html, "html.parser")

            section_linked_asins: list[str] = []

            def _add_linked_asin(asin_val: str):
                if asin_val and asin_val not in section_linked_asins:
                    section_linked_asins.append(asin_val)

            def _top_pick_allowed_here(pd):
                nonlocal top_pick_extra_budget
                if pd["asin"] != recommended_asin:
                    return True
                if sec_type in ("purpose", "verdict"):
                    return True
                return top_pick_extra_budget > 0

            def _consume_top_pick_budget_if_needed(pd):
                nonlocal top_pick_extra_budget
                if pd["asin"] == recommended_asin and sec_type not in ("purpose", "verdict"):
                    if top_pick_extra_budget > 0:
                        top_pick_extra_budget -= 1

            def try_link_in_text_nodes(soup, pd, section_type, recommended_asin, base_url, tag):
                nonlocal remaining
                if remaining <= 0:
                    return False
                if len(pd["name"].split()) < 2:
                    return False

                is_top = (pd["normalized_name"] == recommended_key)
                if not _substitute_link_allowed(pd, section_type):
                    _dbg(
                        "SKIP: substitute outside configured editorial scope",
                        name=pd.get("name"),
                        section_type=section_type,
                        scope=substitute_policy.get("scope", "top_pick_only"),
                    )
                    return False
                if is_top:
                    if not _top_pick_allowed_here(pd):
                        return False
                    cap = TOP_PICK_MAX_LINKS if sec_type in ("purpose", "verdict") else 1
                else:
                    cap = 1


                if global_link_counts.get(pd["normalized_name"], 0) >= cap:
                    _dbg(
                        "SKIP: cap reached",
                        name=pd.get("name"),
                        normalized=pd.get("normalized_name"),
                        cap=cap,
                        used=global_link_counts.get(pd["normalized_name"], 0),
                        section_type=section_type,
                        is_substitute=bool(pd.get("is_substitute")),
                    )
                    return False


                pattern = build_tolerant_phrase_regex(pd["name"])
                for node in soup.find_all(string=True):
                    # never touch nodes that are already links or are inside tables
                    if node.parent.name in ['a', 'script', 'style'] or node.find_parent('a'):
                        continue
                    if node.find_parent('table') is not None:
                        continue

                    txt = str(node)

                    # âœ… If the text contains __ANCHOR__...__HERE__, match against the inner text
                    txt_for_matching = ANCHOR_TOKEN_RX.sub(r"\1", txt)

                    m = pattern.search(txt_for_matching)
                    if not m:
                        continue

                    if _text_cluster_already_has_affiliate_link(node, txt, m.start()):
                        continue

                    if _is_crowded_product_text(txt_for_matching, pd.get("normalized_name")):
                        later_clean_match = False
                        seen_current_node = False
                        for later_node in soup.find_all(string=True):
                            if later_node is node:
                                seen_current_node = True
                                continue
                            if not seen_current_node:
                                continue
                            if later_node.parent.name in ['a', 'script', 'style'] or later_node.find_parent('a'):
                                continue
                            if later_node.find_parent('table') is not None:
                                continue
                            later_txt = str(later_node)
                            later_txt_for_matching = ANCHOR_TOKEN_RX.sub(r"\1", later_txt)
                            if _is_crowded_product_text(later_txt_for_matching, pd.get("normalized_name")):
                                continue
                            if pattern.search(later_txt_for_matching):
                                later_clean_match = True
                                break
                        if later_clean_match:
                            continue


                    matched = m.group(0)

                    paragraph_end_html = None
                    # inject link (we already know it is NOT in a table)
                    try:
                        _dbg(
                            "MATCH found in text node",
                            section_type=section_type,
                            matched=matched,
                            product_name=pd.get("name"),
                            asin=pd.get("asin"),
                            is_substitute=bool(pd.get("is_substitute")),
                            is_top=is_top,
                        )

                        if pd.get("is_substitute"):
                            note = "Similar on Amazon"                           
                            _dbg(
                                "SUBSTITUTE branch taken (try_link_in_text_nodes)",
                                extracted=matched,
                                asin=pd.get("asin"),
                                display=get_short_display_title(pd, 60, fallback=matched),
                            )

                            sub_link = create_affiliate_link(
                                get_short_display_title(pd, 60, fallback=matched),
                                pd["asin"],
                                base_url,
                                tag,
                                add_trailer=False,
                                cta_cfg=cta_cfg,
                                display_text=get_short_display_title(pd, 60, fallback=matched),
                                is_substitute=True,
                                substitute_note=note,
                                strong_wrap=False,
                                price_display=pd.get("price"),
                                full_title=get_display_title(pd, fallback=matched),  # Ã°Å¸â€˜Ë† NEW (full Amazon title tooltip)
                            )

                            # Keep the reviewed product unlinked; disclose the substitute at paragraph end.
                            link_html = escape(matched)
                            paragraph_end_html = (
                                '<span class="aff-sub-note">('
                                + 'Similar on Amazon: '
                                + sub_link
                                + ')</span>'
                            )
                        else:
                            link_html = create_affiliate_link(
                                matched, pd["asin"], base_url, tag,
                                context=section_type,
                                is_recommended=is_top,
                                add_trailer=True,
                                cta_cfg=cta_cfg,
                                display_text=matched,
                                is_substitute=False,
                                price_display=pd.get("price"),
                                price_ts=pd.get("price_ts"),
                                full_title=get_display_title(pd, fallback=matched),  # âœ… add
                            )


                    except Exception as e:
                        # Fail safe: keep original text if link building fails
                        log.exception(
                            "Affiliate link build failed (leaving text unlinked)",
                            extra={"step": "inject.link_build_error",
                                   "extra_json": {"error": str(e), "matched": matched, "asin": pd.get("asin"), "name": pd.get("name")}}
                        )
                        link_html = escape(matched)


                    _replace_match_with_optional_paragraph_note(
                        node,
                        txt,
                        m.start(),
                        m.end(),
                        link_html,
                        paragraph_end_html,
                    )
                    global_link_counts[pd["normalized_name"]] = global_link_counts.get(pd["normalized_name"], 0) + 1
                    _add_linked_asin(pd["asin"])
                    remaining -= 1
                    _consume_top_pick_budget_if_needed(pd)
                    return True
                return False


            made_change = True
            while made_change and remaining > 0:
                made_change = False
                for name_key in sorted_full_names:
                    pd = product_name_to_data[name_key]
                    if try_link_in_text_nodes(temp_soup, pd, sec_type, recommended_asin, base_url, tag):
                        made_change = True
                        if remaining <= 0:
                            break

            if remaining > 0 and sec_type in ("item", "purpose", "verdict"):
                for alias in sorted_aliases:
                    if remaining <= 0:
                        break
                    full_key = alias_map[alias]
                    pd = product_name_to_data[full_key]
                    if len(pd["name"].split()) < 2:
                        continue

                    is_top = (pd["normalized_name"] == recommended_key)
                    if not _substitute_link_allowed(pd, sec_type):
                        continue
                    if is_top:
                        if not _top_pick_allowed_here(pd):
                            continue
                        cap = TOP_PICK_MAX_LINKS if sec_type in ("purpose", "verdict") else 1
                    else:
                        cap = 1

                    if global_link_counts.get(full_key, 0) >= cap:
                        continue
                    if fuzz.partial_ratio(alias, temp_soup.get_text(" ").lower()) < ALIAS_FUZZ_THRESHOLD:
                        continue

                    alias_pattern = build_tolerant_phrase_regex(alias)
                    placed = False
                    for node in temp_soup.find_all(string=True):
                        if node.parent.name in ['a', 'script', 'style'] or node.find_parent('a'):
                            continue
                        # inside the alias loop, before using the match:
                        if node.find_parent('table') is not None:
                            continue

                        txt = str(node)

                        # âœ… If the text contains __ANCHOR__...__HERE__, match against the inner text
                        txt_for_matching = ANCHOR_TOKEN_RX.sub(r"\1", txt)

                        m = alias_pattern.search(txt_for_matching)
                        if not m:
                            continue

                        if _text_cluster_already_has_affiliate_link(node, txt, m.start()):
                            continue

                        if _is_crowded_product_text(txt_for_matching, full_key):
                            later_clean_match = False
                            seen_current_node = False
                            for later_node in temp_soup.find_all(string=True):
                                if later_node is node:
                                    seen_current_node = True
                                    continue
                                if not seen_current_node:
                                    continue
                                if later_node.parent.name in ['a', 'script', 'style'] or later_node.find_parent('a'):
                                    continue
                                if later_node.find_parent('table') is not None:
                                    continue
                                later_txt = str(later_node)
                                later_txt_for_matching = ANCHOR_TOKEN_RX.sub(r"\1", later_txt)
                                if _is_crowded_product_text(later_txt_for_matching, full_key):
                                    continue
                                if alias_pattern.search(later_txt_for_matching):
                                    later_clean_match = True
                                    break
                            if later_clean_match:
                                continue

                        matched = m.group(0)

                        amazon_title = get_display_title(pd, fallback=matched)
                        short_title = get_short_display_title(pd, 60, fallback=matched)

                        paragraph_end_html = None

                        try:
                            _dbg(
                                "ALIAS match found",
                                section_type=sec_type,
                                matched=matched,
                                product_name=pd.get("name"),
                                asin=pd.get("asin"),
                                is_substitute=bool(pd.get("is_substitute")),
                                is_top=is_top,
                                alias=alias,
                                full_key=full_key,
                            )

                            if pd.get("is_substitute"):
                                note = SIMILAR_ITEM_LABEL

                                _dbg(
                                    "SUBSTITUTE branch taken (try_link_in_text_nodes)",
                                    extracted=matched,
                                    asin=pd.get("asin"),
                                    display=get_short_display_title(pd, 60, fallback=matched),
                                )

                                sub_link = create_affiliate_link(
                                    get_short_display_title(pd, 60, fallback=matched),
                                    pd["asin"],
                                    base_url,
                                    tag,
                                    add_trailer=False,
                                    cta_cfg=cta_cfg,
                                    display_text=get_short_display_title(pd, 60, fallback=matched),
                                    is_substitute=True,
                                    substitute_note=note.rstrip(":"),
                                    strong_wrap=False,
                                    price_display=pd.get("price"),
                                    price_ts=pd.get("price_ts"),
                                    full_title=get_display_title(pd, fallback=matched),  # Ã°Å¸â€˜Ë† NEW
                                )


                                link_html = escape(matched)
                                paragraph_end_html = (
                                    '<span class="aff-sub-note">('
                                    + f'{escape(note)} '
                                    + sub_link
                                    + ')</span>'
                                )


                            else:
                                link_html = create_affiliate_link(
                                    matched,
                                    pd["asin"],
                                    base_url,
                                    tag,
                                    context=sec_type,
                                    is_recommended=is_top,
                                    add_trailer=True,
                                    cta_cfg=cta_cfg,
                                    display_text=matched,
                                    is_substitute=False,
                                    price_display=pd.get("price"),
                                    price_ts=pd.get("price_ts"),
                                    full_title=get_display_title(pd, fallback=matched),  # âœ… add
                                )


                        except Exception as e:
                            log.exception(
                                "Affiliate link build failed (alias loop)",
                                extra={"step": "inject.link_build_error",
                                       "extra_json": {"error": str(e), "matched": matched, "asin": pd.get("asin"), "name": pd.get("name"), "alias": alias}}
                            )
                            link_html = escape(matched)



                        _replace_match_with_optional_paragraph_note(
                            node,
                            txt,
                            m.start(),
                            m.end(),
                            link_html,
                            paragraph_end_html,
                        )
                        global_link_counts[full_key] = global_link_counts.get(full_key, 0) + 1
                        _add_linked_asin(pd["asin"])
                        remaining -= 1
                        _consume_top_pick_budget_if_needed(pd)
                        placed = True
                        break
                    if placed and remaining <= 0:
                        break

            if sec_type == "verdict" and remaining > 0:
                asin = recommended_asin
                if recommended.get("is_substitute"):
                    # Never make a substitute Amazon title look like the reviewed
                    # product or lead the final verdict with that substituted link.
                    for link in list(temp_soup.select(f'a[href*="/dp/{asin}"]')):
                        link.unwrap()
                    reviewed_name = top_pick_name or recommended.get("name") or "the reviewed product"
                    similar_title = get_short_display_title(
                        recommended, 70, fallback=recommended_display
                    )
                    href = _aff_url(base_url, asin, tag)
                    disclosure = temp_soup.new_tag("div")
                    disclosure["class"] = ["affiliate-substitute-disclosure", "verdict-substitute"]
                    disclosure.append(
                        NavigableString(
                            f"{reviewed_name} remains the reviewed model. "
                            "If that exact version is not currently available on Amazon, "
                            "the closest model we found is "
                        )
                    )
                    if href:
                        link = temp_soup.new_tag("a", href=href)
                        link["class"] = ["aff-inline", "aff-substitute"]
                        link["target"] = "_blank"
                        link["rel"] = ["sponsored", "noopener", "nofollow"]
                        link["data-aff"] = "1"
                        link["aria-label"] = f"See similar model {similar_title} on Amazon"
                        link.string = similar_title
                        disclosure.append(link)
                        disclosure.append(NavigableString(". See this similar model on Amazon."))
                    else:
                        disclosure.append(NavigableString(similar_title + "."))
                    temp_soup.append(disclosure)
                    remaining -= 1
                    global_link_counts[recommended_key] = global_link_counts.get(recommended_key, 0) + 1
                    _add_linked_asin(asin)
                else:
                    exists = any(a for a in temp_soup.select(f'a[href*="/dp/{asin}"]') if a.get_text(strip=True))
                    if not exists:
                        phrase = build_tolerant_phrase_regex(recommended_display)
                        inserted = False
                        for node in temp_soup.find_all(string=True):
                            if node.parent.name in ['a', 'script', 'style'] or node.find_parent('a'):
                                continue
                            txt = str(node)
                            m = phrase.search(txt)
                            if m:
                                matched = m.group(0)
                                link_html = create_affiliate_link(
                                    matched, asin, base_url, tag,
                                    context=sec_type, is_recommended=True
                                )
                                node.replace_with(BeautifulSoup(txt[:m.start()] + link_html + txt[m.end():], "html.parser"))
                                inserted = True
                                remaining -= 1
                                global_link_counts[recommended_key] = global_link_counts.get(recommended_key, 0) + 1
                                _add_linked_asin(asin)
                                break
                        if not inserted and remaining > 0:
                            host = temp_soup.find(["p", "li"])
                            generated = create_affiliate_link(
                                recommended_display, asin, base_url, tag,
                                context=sec_type, is_recommended=True
                            )
                            if host:
                                host.insert(0, BeautifulSoup(generated, "html.parser"))
                            else:
                                temp_soup.insert(0, BeautifulSoup(generated, "html.parser"))
                            remaining -= 1
                            global_link_counts[recommended_key] = global_link_counts.get(recommended_key, 0) + 1
                            _add_linked_asin(asin)

            if sec_type == "purpose" and recommended.get("is_substitute"):
                for old_note in list(temp_soup.select(".aff-sub-note")):
                    old_note.decompose()
                reviewed_name = top_pick_name or recommended.get("name") or "the reviewed product"
                similar_title = get_short_display_title(
                    recommended, 70, fallback=recommended_display
                )
                href = _aff_url(base_url, recommended_asin, tag)
                disclosure = temp_soup.new_tag("div")
                disclosure["class"] = [
                    "affiliate-substitute-disclosure",
                    "purpose-substitute",
                ]
                disclosure.append(
                    NavigableString(
                        f"{reviewed_name} remains the reviewed model. "
                        "If that exact version is not currently available on Amazon, "
                        "the closest model we found is "
                    )
                )
                if href:
                    link = temp_soup.new_tag("a", href=href)
                    link["class"] = ["aff-inline", "aff-substitute"]
                    link["target"] = "_blank"
                    link["rel"] = ["sponsored", "noopener", "nofollow"]
                    link["data-aff"] = "1"
                    link["aria-label"] = f"See similar model {similar_title} on Amazon"
                    link.string = similar_title
                    disclosure.append(link)
                    disclosure.append(
                        NavigableString(". See this similar model on Amazon.")
                    )
                else:
                    disclosure.append(NavigableString(similar_title + "."))
                temp_soup.append(disclosure)
                _add_linked_asin(recommended_asin)

            processed_html = str(temp_soup)
            for ph, html_a in placeholders.items():
                processed_html = processed_html.replace(ph, html_a)

            section_soup_after = BeautifulSoup(processed_html, "html.parser")
            existing_img_srcs = {img.get('src') for img in section_soup_after.find_all("img") if img.get('src')}
            img_fragments = []

            if sec_type == "purpose":
                substitute_image_ok = (
                    recommended.get("is_substitute")
                    and allow_close_substitute_image
                    and substitute_images_used < max_substitute_images
                )
                if (
                    recommended_img_url
                    and (not recommended.get("is_substitute") or substitute_image_ok)
                    and recommended_img_url not in existing_img_srcs
                ):
                    display_for_img = get_display_title(recommended)
                    caption = None
                    if recommended.get("is_substitute"):
                        template = str(image_policy.get(
                            "substitute_caption_template",
                            "Similar model shown: {amazon_title}",
                        ))
                        caption = template.format(
                            amazon_title=display_for_img,
                            reviewed_product=top_pick_name or recommended.get("name", ""),
                        )
                        substitute_images_used += 1
                    img_fragments.append(
                        format_image_html(
                            recommended_asin,
                            recommended_img_url,
                            display_for_img,
                            base_url,
                            tag,
                            "float-image",
                            cta_text=None,
                            show_cta=True,
                            cta_cfg=cta_cfg,
                            caption_text=caption,
                        )
                    )


            elif sec_type == "item":
                imgs = []
                for idx_asin, asin in enumerate(section_linked_asins):
                    pd = product_asin_to_data.get(asin)
                    if not pd or pd.get("is_substitute") or not pd.get("img_url") or pd["img_url"] in existing_img_srcs:
                        continue
                    if len(imgs) >= MAX_ITEM_IMAGES:
                        break
                    show_cta = (idx_asin == 0)
                    display_for_img = get_display_title(pd)   # Ã°Å¸â€˜Ë† prefer Amazon title
                    imgs.append(
                        format_image_html(
                            pd["asin"],
                            pd["img_url"],
                            display_for_img,   # Ã°Å¸â€˜Ë† correct hover/alt text
                            base_url,
                            tag,
                            "float-image",
                            cta_text=None,
                            show_cta=show_cta,
                            cta_cfg=cta_cfg
                        )
                    )
                if imgs:
                    # IMPORTANT: keep each image as its own fragment so we can count them
                    img_fragments.extend(imgs)


            elif sec_type == "verdict":
                substitute_image_ok = (
                    recommended.get("is_substitute")
                    and allow_close_substitute_image
                    and substitute_images_used < max_substitute_images
                )
                if (
                    recommended_img_url
                    and (not recommended.get("is_substitute") or substitute_image_ok)
                    and recommended_img_url not in existing_img_srcs
                ):
                    display_for_img = get_display_title(recommended)
                    caption = None
                    if recommended.get("is_substitute"):
                        template = str(image_policy.get(
                            "substitute_caption_template",
                            "Similar model shown: {amazon_title}",
                        ))
                        caption = template.format(
                            amazon_title=display_for_img,
                            reviewed_product=top_pick_name or recommended.get("name", ""),
                        )
                        substitute_images_used += 1
                    img_fragments.append(
                        format_image_html(
                            recommended_asin,
                            recommended_img_url,
                            display_for_img,
                            base_url,
                            tag,
                            "float-image",
                            cta_text=None,
                            show_cta=True,
                            cta_cfg=cta_cfg,
                            caption_text=caption,
                        )
                    )


            else:
                if section_linked_asins:
                    asin_one = section_linked_asins[0]
                    pd = product_asin_to_data.get(asin_one)
                    if pd and not pd.get("is_substitute") and pd.get("img_url") and pd["img_url"] not in existing_img_srcs:
                        display_for_img = get_display_title(pd)   # Ã°Å¸â€˜Ë† prefer Amazon title
                        img_fragments.append( 
                            format_image_html(
                                pd["asin"],
                                pd["img_url"],
                                display_for_img,   # Ã°Å¸â€˜Ë† correct hover/alt text
                                base_url,
                                tag,
                                "float-image",
                                cta_text=None,
                                show_cta=True,
                                cta_cfg=cta_cfg
                            )
                        )


            is_section_1_1 = bool(re.search(r'^\s*1\.1\b', prev_heading_text, re.I))

            if img_fragments:
                if is_section_1_1:
                    # 1.1: images always at top as a row (links/text below)
                    img_block_html = (
                        _image_row(img_fragments) if len(img_fragments) >= 3 else "".join(img_fragments)
                    )
                    processed_html = img_block_html + processed_html
                elif sec_type in required_image_sections:
                    # Required anchor sections must receive the image even when
                    # the exact product has no link target in their prose.
                    processed_html = "".join(img_fragments) + processed_html
                else:
                    # other sections: keep image close to its hyperlink
                    processed_html = _insert_images_near_first_aff_link(processed_html, img_fragments)



            # 1.1: enforce "images top, tables bottom"
            if is_section_1_1:
                processed_html = _reorder_section_images_text_tables(processed_html)



        output.append(processed_html)

    final_content = "".join(output)

    if recommended_asin and recommended_display:
        is_sticky_substitute = bool(recommended.get("is_substitute"))
        sticky_label = (
            "See Similar Model on Amazon"
            if is_sticky_substitute
            else CTA_STRINGS["sticky"]
        )
        sticky_aria = (
            f"See similar model {recommended_display} on Amazon"
            if is_sticky_substitute
            else f"Check {recommended_display} price on Amazon"
        )
        sticky_cta_html = f'''
        <aside id="sticky-cta" aria-label="Floating Amazon CTA">
            <a class="aff-btn aff-sticky" href="{base_url}/dp/{recommended_asin}?tag={tag}" target="_blank" rel="sponsored noopener nofollow" role="button" aria-label="{escape(sticky_aria)}">
                {escape(sticky_label)}
            </a>
        </aside>
        '''
        final_content = final_content + sticky_cta_html

    # Always sanitize tables, regardless of sticky CTA
    final_content = remove_trailers_in_tables(final_content)
    final_content = remove_links_in_tables(final_content)
    final_content = drop_empty_pros_cons_columns(final_content)
    final_content = re.sub(r"</\s*>", "", final_content)
    

    return final_content



def fix_check_price_placeholders(html: str, asin: str | None, base_url: str, tag: str) -> str:
    if not html or not asin:
        return html
    href = _aff_url(base_url, asin, tag)
    if not href:
        return html

    soup = BeautifulSoup(html, "html.parser")

    # Find all <a> tags that have a class matching /check-price/i
    for a in soup.find_all("a", class_=lambda c: (
        isinstance(c, list)  and any(_CHECK_PRICE_CLASS_RX.search(x or "") for x in c)
    ) or (
        isinstance(c, str)   and _CHECK_PRICE_CLASS_RX.search(c)
    )):
        if not a.get("href"):  # only fill missing hrefs
            a["href"] = href
            a["target"] = "_blank"
            a["rel"] = ["sponsored", "noopener", "nofollow"]
            a["aria-label"] = "Open Amazon product page"
            # keep existing classes; just add 'aff-inline' once
            cur = a.get("class") or []
            if "aff-inline" not in cur:
                a["class"] = cur + ["aff-inline"]

    return str(soup)


# =========================
# Step 2 rewritten wrapper
# =========================

def apply_search_query_suffix(query: str, cfg: dict) -> str:
    """Append an optional config-driven suffix to the Amazon search query.

    This helps disambiguate products vs accessories, e.g.:
      'Levoit LV-H132' + 'air purifier' -> 'Levoit LV-H132 air purifier'

    Config:
      search_query_suffix: string | list[string]
    """
    if not query:
        return query

    suffix = cfg.get("search_query_suffix", "")
    if isinstance(suffix, (list, tuple)):
        suffix = " ".join([str(s).strip() for s in suffix if str(s).strip()])
    suffix = str(suffix).strip()

    if not suffix:
        return query

    q_l = query.lower()
    s_l = suffix.lower()
    # Avoid double-appending if already present
    if s_l in q_l:
        return query

    return f"{query} {suffix}".strip()



def _amazon_item_title(item) -> str:
    try:
        return normalize_ws(item.item_info.title.display_value or "")
    except Exception:
        return ""


def _identity_tokens(text: str) -> list[str]:
    """Return comparable identity tokens while preserving model IDs and units."""
    s = normalize_name(text or "")
    s = re.sub(r"(?<=[a-z0-9])[-_/](?=[a-z0-9])", "", s)
    aliases = _identity_word_aliases()
    return [aliases.get(token, token) for token in re.findall(r"[a-z0-9]+", s)]


def review_keyword_identity_tokens(text: str) -> set[str]:
    """Model/size tokens that affiliate availability must never retarget."""
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
            r"\bbudget|[£$€])\s*$",
            prefix,
        ):
            continue
        tokens.add(token)
    return tokens


def reviewed_product_matches_keyword(product: str, keyword: str, cfg: dict) -> bool:
    controls = cfg.get("review_identity") or {}
    if controls.get("allow_product_retargeting", False):
        return True
    if controls.get("lock_keyword_model_tokens", True) is False:
        return True
    required = review_keyword_identity_tokens(keyword)
    return not required or required.issubset(review_keyword_identity_tokens(product))


def requested_title_identity_match(requested_name: str, title: str, cfg: dict) -> tuple[bool, dict]:
    """Require every meaningful requested identity token in the Amazon title."""
    query_tokens = _identity_tokens(requested_name)
    title_tokens = set(_identity_tokens(title))
    if not query_tokens or not title_tokens:
        return False, {"reason": "missing_tokens", "title": title}

    ignored = {
        "a", "an", "and", "for", "in", "of", "on", "the", "to", "with",
        "best", "review", "reviews", "reviewed", "top",
    }
    # Optional and deliberately separate from category configuration: callers
    # may declare editorial words, but product/category/type words remain part
    # of identity so variants cannot collapse into one another.
    for key in ("identity_ignored_terms",):
        for value in (cfg.get(key) or []):
            if isinstance(value, str):
                ignored.update(_identity_tokens(value))

    required = [
        token for token in query_tokens
        if token not in ignored and (len(token) >= 2 or any(ch.isdigit() for ch in token))
    ]
    overlap = sorted(set(required) & title_tokens)
    missing = sorted(set(required) - title_tokens)
    exact = bool(required and not missing)
    return exact, {
        "title": title,
        "required_tokens": required,
        "identity_overlap": overlap,
        "identity_missing": missing,
    }


def amazon_item_matches_requested_product(item, requested_name: str, cfg: dict) -> tuple[bool, dict]:
    """Require complete requested product identity, not merely a shared model/spec."""
    return requested_title_identity_match(requested_name, _amazon_item_title(item), cfg)


def step2_search_and_extract_for_name(api, name, cfg, trace_id, excluded_asins=None):
    """
    Two-stage Amazon strategy for images (PA-API compatible):

      1) SearchItems paged (item_count/item_page are 1..10) to build a larger candidate pool.
      2) Apply light gating + scoring to pick the best candidate.
      3) If the best score is below a threshold, run product-focused fallback queries.
      4) Once we have a plausible ASIN, optionally confirm/upgrade with GetItems and choose best image URL.

    Always returns an 8-tuple:
    (chosen_item | None,
     image_url | None,
     image_size_tag | None,
     image_width | None,
     image_height | None,
     image_dimension_status | str,
     image_content_ratio | None,
     used_similar_fallback | bool).
    """
    excluded_asins = {str(asin).strip().upper() for asin in (excluded_asins or []) if str(asin).strip()}
    pool_size = int(cfg.get("amazon_search_pool_size", cfg.get("search_pool_size", 30)))
    min_score = int(cfg.get("amazon_min_score_threshold", cfg.get("min_score_threshold", 25)))
    search_index = cfg.get("amazon_search_index")  # optional; e.g., "HomeAndKitchen"

    def _search_items_paged(query: str):
        """
        Creators API constrains item_count/item_page to integers between 1 and 10.
        Page through results to build a larger candidate pool.
        """
        desired = max(1, int(pool_size))
        pages = max(1, math.ceil(desired / 10))
        pages = min(pages, 10)  # PA-API item_page constraint
        all_items = []
        last_res = None
        remaining = desired

        for page in range(1, pages + 1):
            count = min(10, remaining)
            remaining -= count
            try:
                res = safe_search_items(
                    api,
                    query,
                    trace_id=trace_id,
                    item_count=count,
                    item_page=page,
                    search_index=search_index,
                )
            except AmazonEligibilityError:
                raise
            except Exception:
                res = None

            if res and getattr(res, "items", None):
                last_res = res
                all_items.extend(res.items)

            if remaining <= 0:
                break

        return last_res, all_items

    def _run_search(query: str):
        # Callers supply the complete query. Exact-name searches must not have a
        # generic category suffix silently appended.
        res, items = _search_items_paged(query)
        if not items:
            return None, None, None

        best_item, best_score = select_purifier_strict(
            items, name, cfg, trace_id=trace_id, return_score=True
        )
        return res, best_item, best_score

    # Stage 1: exact product name first, then concise product variants.
    exact_queries = generate_query_variants(name, cfg)
    if not exact_queries or exact_queries[0].strip().lower() != name.strip().lower():
        exact_queries.insert(0, name)

    chosen_item = None
    best_score = None
    used_similar_fallback = False
    any_results = False
    queried = set()
    best_similar_item = None
    best_similar_score = -10_000
    similar_min_score = int(cfg.get("amazon_similar_min_score", min_score))

    rejected_asin_counts = {}
    repeat_candidate_limit = max(1, int(cfg.get("amazon_repeat_candidate_stop", 2)))
    def _remember_similar(candidate, candidate_score):
        nonlocal best_similar_item, best_similar_score
        if candidate is None:
            return False
        compatible, compatibility_evidence = substitute_specialization_compatible(
            name, _amazon_item_title(candidate), cfg
        )
        candidate_asin = str(getattr(candidate, "asin", "") or "").strip().upper()
        if candidate_asin and candidate_asin in excluded_asins:
            log.info(
                "Rejected Similar on Amazon candidate because ASIN is already assigned",
                extra={
                    "trace_id": trace_id,
                    "extra_json": {
                        "name": name,
                        "amazon_title": _amazon_item_title(candidate),
                        "asin": candidate_asin,
                    },
                },
            )
            return False
        if not compatible:
            log.info(
                "Rejected semantically incompatible Similar on Amazon candidate",
                extra={
                    "trace_id": trace_id,
                    "extra_json": {
                        "name": name,
                        "amazon_title": _amazon_item_title(candidate),
                        **compatibility_evidence,
                    },
                },
            )
            return False
        score_value = candidate_score if candidate_score is not None else -10_000
        if score_value >= similar_min_score and score_value > best_similar_score:
            best_similar_item = candidate
            best_similar_score = score_value
            return True
        return False

    for idx, query in enumerate(exact_queries):
        query = normalize_ws(query)
        query_key = query.lower()
        if not query or query_key in queried:
            continue
        queried.add(query_key)
        if idx:
            log.info(
                "Retrying with concise product query",
                extra={"trace_id": trace_id, "extra_json": {"variant_query": query}},
            )
        res, candidate, candidate_score = _run_search(query)
        any_results = any_results or bool(res)
        if not candidate:
            continue
        identity_ok, identity_evidence = amazon_item_matches_requested_product(
            candidate, name, cfg
        )
        if identity_ok:
            chosen_item, best_score = candidate, candidate_score
            log.info(
                "Exact Amazon product identity confirmed",
                extra={"trace_id": trace_id, "extra_json": identity_evidence},
            )
            break
        retained_as_similar = _remember_similar(candidate, candidate_score)
        log.info(
            (
                "Candidate retained only as Similar on Amazon"
                if retained_as_similar else
                "Candidate rejected as exact and not retained as fallback"
            ),
            extra={
                "trace_id": trace_id,
                "extra_json": {
                    **identity_evidence,
                    "score": candidate_score,
                    "query": query,
                },
            },
        )
        rejected_asin = str(getattr(candidate, "asin", "") or "").strip().upper()
        if rejected_asin:
            rejected_asin_counts[rejected_asin] = rejected_asin_counts.get(rejected_asin, 0) + 1
            if rejected_asin_counts[rejected_asin] >= repeat_candidate_limit:
                log.info(
                    "Stopping exact retries after repeated rejected ASIN",
                    extra={
                        "trace_id": trace_id,
                        "extra_json": {
                            "name": name,
                            "asin": rejected_asin,
                            "repeat_count": rejected_asin_counts[rejected_asin],
                        },
                    },
                )
                break

    # Stage 2: retain the best acceptable category result as an explicitly labelled
    # substitute when the requested brand/model is not available on Amazon.
    if not chosen_item:
        fallback_queries = cfg.get("product_focus_queries") or build_similar_fallback_queries(name, cfg)
        if isinstance(fallback_queries, str):
            fallback_queries = [fallback_queries]
        fallback_limit = max(1, int(cfg.get("amazon_fallback_query_limit", 2)))
        fallback_queries = list(fallback_queries)[:fallback_limit]

        for query in fallback_queries:
            query = normalize_ws(str(query))
            query_key = query.lower()
            if not query or query_key in queried:
                continue
            queried.add(query_key)
            log.info(
                "Searching for a Similar on Amazon fallback",
                extra={"trace_id": trace_id, "extra_json": {"fallback_query": query}},
            )
            res, candidate, candidate_score = _run_search(query)
            any_results = any_results or bool(res)
            if not candidate:
                continue
            identity_ok, identity_evidence = amazon_item_matches_requested_product(
                candidate, name, cfg
            )
            if identity_ok:
                chosen_item, best_score = candidate, candidate_score
                log.info(
                    "Exact Amazon product identity confirmed during fallback search",
                    extra={"trace_id": trace_id, "extra_json": identity_evidence},
                )
                break
            _remember_similar(candidate, candidate_score)

        if not chosen_item and best_similar_item is not None:
            chosen_item = best_similar_item
            best_score = best_similar_score
            used_similar_fallback = True
            log.info(
                "Using explicitly labelled Similar on Amazon fallback",
                extra={
                    "trace_id": trace_id,
                    "extra_json": {
                        "name": name,
                        "amazon_title": _amazon_item_title(chosen_item),
                        "score": best_score,
                    },
                },
            )
    if not chosen_item:
        reason = "No acceptable product found" if any_results else "No Amazon match found"
        log.warning(
            "Name unmatched",
            extra={"trace_id": trace_id, "extra_json": {"name": name, "reason": reason}},
        )
        return (
            None,   # chosen_item
            None,   # selected_img_url
            None,   # selected_img_size
            None,   # selected_img_width
            None,   # selected_img_height
            "unknown",  # selected_img_dimension_status
            None,   # selected_img_content_ratio
            False,  # used_similar_fallback
        )

    # --- Stage 3: confirm/upgrade via GetItems for better images ---
    asin = getattr(chosen_item, "asin", None)
    if asin and bool(cfg.get("use_get_items", True)):
        gi = safe_get_items(api, [asin], trace_id=trace_id)
        gi_items = getattr(gi, "items", None) or []
        if gi_items:
            chosen_item = gi_items[0]

    # --- Image selection (primary large \u2192 medium \u2192 small; else variants) ---
    img_info = {}
    try:
        p = chosen_item.images.primary if chosen_item.images and chosen_item.images.primary else None
        if p:
            for size in ("large", "medium", "small"):
                url = getattr(getattr(p, size, None), "url", None)
                if url:
                    img_info.setdefault("primary", {})[size] = url

        vs = chosen_item.images.variants if chosen_item.images and getattr(chosen_item.images, "variants", None) else []
        img_info["variants"] = []
        for v in vs:
            v_urls = {}
            for size in ("large", "medium", "small"):
                url = getattr(getattr(v, size, None), "url", None)
                if url:
                    v_urls[size] = url
            if v_urls:
                img_info["variants"].append(v_urls)
    except Exception as e:
        log.error(
            f"Error reading images for item: {e}",
            extra={"trace_id": trace_id, "extra_json": {"asin": getattr(chosen_item, 'asin', None)}},
        )

    selected_img_url = None
    selected_img_size = None
    selected_img_width = None
    selected_img_height = None
    selected_img_dimension_status = "unknown"
    selected_img_content_ratio = None

    primary = img_info.get("primary") or {}
    candidates = []

    def _consider_candidate(label, url, variant_index=None):
        if not url:
            return
        try:
            best_url, w, h, dim_status, content_ratio = resolve_best_amazon_image_candidate(url, min_width=MIN_IMAGE_WIDTH)
        except Exception as e:
            log.warning(
                "[selector] Candidate probe failed; falling back to original URL",
                extra={"trace_id": trace_id, "extra_json": {"label": label, "url": url, "asin": getattr(chosen_item, "asin", None), "error": str(e)}},
            )
            best_url, w, h, dim_status, content_ratio = (url, None, None, "unknown", None)
        meta = {
            "asin": getattr(chosen_item, "asin", None),
            "width": w,
            "height": h,
            "dimension_status": dim_status,
            "content_ratio": content_ratio,
            "min_width": MIN_IMAGE_WIDTH,
            "original_url": url,
            "selected_url": best_url,
        }
        if variant_index is not None:
            meta["variant_index"] = variant_index

        selected_url = best_url or url

        if w is None:
            log.info(
                f"[selector] Dimensions unknown for {label}; keeping best-effort image candidate",
                extra={"trace_id": trace_id, "extra_json": meta},
            )
            candidates.append({
                "url": selected_url,
                "size": label,
                "width": 0,
                "height": h,
                "dimension_status": dim_status,
                "content_ratio": content_ratio or 0,
                "best_effort": True,
            })
            return

        if w >= MIN_IMAGE_WIDTH:
            log.info(
                "[selector] candidate evaluated",
                extra={
                    "trace_id": trace_id,
                    "extra_json": {
                        "label": label,
                        "width": w,
                        "height": h,
                        "content_ratio": content_ratio,
                        "url": selected_url,
                    },
                },
            )
            candidates.append({
                "url": selected_url,
                "size": label,
                "width": w,
                "height": h,
                "dimension_status": dim_status,
                "content_ratio": content_ratio or 0,
                "best_effort": False,
            })
            if selected_url != url:
                log.info(
                    "[selector] Upgraded Amazon image URL to a larger size-marker candidate",
                    extra={"trace_id": trace_id, "extra_json": meta},
                )
        else:
            log.info(
                f"[selector] Image below min width; keeping as best-effort fallback for {label}",
                extra={"trace_id": trace_id, "extra_json": meta},
            )
            candidates.append({
                "url": selected_url,
                "size": label,
                "width": w,
                "height": h,
                "dimension_status": dim_status,
                "content_ratio": content_ratio or 0,
                "best_effort": True,
            })

    _consider_candidate("primary.large", primary.get("large"))

    if ALLOW_MEDIUM_FALLBACK and primary.get("medium"):
        _consider_candidate("primary.medium", primary.get("medium"))

    if img_info.get("variants"):
        for vidx, v in enumerate(img_info["variants"]):
            _consider_candidate(f"variant[{vidx}].large", v.get("large"), variant_index=vidx)

    if candidates:
        best = max(candidates, key=lambda c: (
            0 if c.get("best_effort") else 1,
            c["width"],
            c.get("content_ratio", 0),
            1 if c.get("dimension_status") == "measured" else 0,
            c.get("height") or 0
        ))
        selected_img_url = best["url"]
        selected_img_size = best["size"]
        selected_img_width = best["width"]
        selected_img_height = best["height"]
        selected_img_dimension_status = best.get("dimension_status", "unknown")
        selected_img_content_ratio = best.get("content_ratio")

        log.info(
            "[selector] FINAL choice with content ratio",
            extra={
                "trace_id": trace_id,
                "extra_json": {
                    "selected_width": selected_img_width,
                    "selected_height": selected_img_height,
                    "selected_ratio": best.get("content_ratio"),
                    "selected_size": selected_img_size,
                    "dimension_status": selected_img_dimension_status,
                    "candidate_count": len(candidates),
                },
            },
        )

        log.info(
            "[selector] Selected sharpest acceptable image candidate",
            extra={
                "trace_id": trace_id,
                "extra_json": {
                    "asin": getattr(chosen_item, "asin", None),
                    "selected_img_size": selected_img_size,
                    "selected_width": best["width"],
                    "selected_height": best["height"],
                    "dimension_status": selected_img_dimension_status,
                    "content_ratio": selected_img_content_ratio,
                    "candidate_count": len(candidates),
                },
            },
        )
    else:
        log.warning(
            "[selector] Rejected item image: no acceptable large image found",
            extra={"trace_id": trace_id, "extra_json": {"asin": getattr(chosen_item, "asin", None), "min_width": MIN_IMAGE_WIDTH}},
        )

    asin_log = getattr(chosen_item, "asin", "unknown")
    title_log = (chosen_item.item_info.title.display_value if (chosen_item.item_info and chosen_item.item_info.title) else name)
    node_log = _get_browse_node(chosen_item) or "unknown"

    log.info(
        "[selector] FINAL image choice: " f"asin={asin_log}; title='{title_log}'; browse_node='{node_log}'; image='{(selected_img_url or 'n/a')}'",
        extra={"trace_id": trace_id, "extra_json": {
            "best_score": best_score,
            "min_score": min_score,
            "selected_width": selected_img_width,
            "selected_height": selected_img_height,
            "dimension_status": selected_img_dimension_status,
            "content_ratio": selected_img_content_ratio,
        }},
    )

    return (
        chosen_item,
        selected_img_url,
        selected_img_size,
        selected_img_width,
        selected_img_height,
        selected_img_dimension_status,
        selected_img_content_ratio,
        used_similar_fallback,
    )


# =========================
# Main
# =========================
def shutdown_amazon_api(api):
    """Best-effort cleanup for Creators API SDK pools (avoids WinError 6 on exit on Windows)."""
    try:
        client = getattr(api, "api_client", None) or getattr(api, "_api_client", None)
        if client is None:
            return
        pool = getattr(client, "pool", None)
        if pool is None:
            return
        try:
            pool.close()
        except Exception:
            pass
        try:
            pool.join()
        except Exception:
            pass
    except Exception:
        return


def load_top_pick(safe_keyword, country):
    safe_keyword_country = f"{safe_keyword}_{country}"
    top_pick_file = Path(f"output/{safe_keyword_country}/product_names_{country}.json")
    if not top_pick_file.exists():
        log.warning("Missing top pick JSON file", extra={"step":"inject","extra_json":{"path":str(top_pick_file)}})
        return None
    try:
        data = json.loads(top_pick_file.read_text(encoding='utf-8'))
        raw = data.get("top_pick", "")
        clean = sanitize_top_pick_name(raw)
        if raw and raw != clean:
            log.warning(
                "Sanitized top pick (removed heading/html noise)",
                extra={"step": "inject", "extra_json": {"raw": raw, "clean": clean}}
            )
        return clean
    except Exception as e:
        log.error(f"Failed to load top pick JSON: {e}", extra={"step":"inject"})
        return None

# --- Heading rewrite (unchanged behavior, with 1.0 preserved) -------------
HEADING_PROMPT_TPL = Template("""
Rewrite a numbered outline of <h2>/<h3> headings for a product review.

SPECIAL RULE Ã¢â‚¬â€ DO NOT TOUCH 1.0
- The first heading **1.0 Who Is It For?** must remain exactly as in the ORIGINAL OUTLINE.
- Do not personalize, shorten, or otherwise modify 1.0Ã¢â‚¬â€even if "1.0" is included in $rewrite_focus_numbers.

GOAL
- Make titles concise and specific to THIS post using the Source content.
- Keep the sense of each original heading, but remove generic prefaces and trailing add-ons.

WHAT TO REWRITE
- Rewrite every number in this focus list: $rewrite_focus_numbers using concrete details from the Source.
- Every focused title must differ materially from its original title. Do not merely prepend the product or brand name.
- Prefer the specific decision covered by the section, such as the verified feature, limitation, fit, use, material, control, or maintenance task.
- For numbers outside the focus list, keep the original base topic without any dash/colon suffix.
- Reminder: Ignore the above if the number is 1.0Ã¢â‚¬â€copy it exactly.

CONSTRAINTS
1) Output MUST preserve numbering exactly as in ORIGINAL OUTLINE. One line per heading.
2) Titles must be a single short phrase (Ã¢â€°Â¤ 70 chars).
3) Do NOT add dash/colon suffixes (no ' Ã¢â‚¬â€ ', ' Ã¢â‚¬â€œ ', ' - ', or ' : ' as separators). 
   Preserve internal hyphens inside compound terms from the Source (e.g., keep 'Anti-Theft' as-is)
4) Do NOT invent terms; only use names found in Source.
5) Canonical facts override conflicting Source wording and apply only to their exact product.
6) Do not transfer a related model's fact, feature, number, size or generation to the reviewed product.
7) Use a number or compatibility claim only when it is supported for the primary product by a
   high-confidence canonical fact that does not require attribution.
8) Use grammatical number agreement: words such as twin, dual, two or pair require a plural noun when they describe multiple items.
9) **For heading 1.0: output the original text verbatim.**

CANONICAL PRODUCT FACTS:
$canonical_facts

CONTEXT
- Primary category: "$primary_category"
- Brand hint (optional): "$brand_hint"
- Source: $blog_source_name

<<<CONTENT_START>>>
$content_excerpt
<<<CONTENT_END>>>


ORIGINAL OUTLINE:
$original_outline
""")

_numline_rx = re.compile(r"^\s*(\d+\.\d+)\s*[:\-Ã¢â‚¬â€œÃ¢â‚¬â€]?\s+(.*\S)\s*$", re.UNICODE)


def _extract_outline_from_html(html: str):
    soup = BeautifulSoup(html, "html.parser")
    items = []
    for tag in soup.find_all(["h2", "h3"]):
        txt = (tag.get_text() or "").strip()
        m = _numline_rx.match(txt)
        items.append({
            "tag": tag.name,
            "number": m.group(1) if m else None,
            "title": m.group(2).strip() if m else txt,
            "node": tag,
        })
    return items

def _parse_rewritten_lines(text: str):
    out = []
    for line in (text or "").splitlines():
        m = _numline_rx.match(line.strip())
        if m:
            out.append((m.group(1), m.group(2)))
    return out

def _apply_outline_by_number(html: str, mapping: dict[str, str]) -> str:
    """
    Rewrite <h2>/<h3> headings by matching their leading number (e.g., 3.0),
    regardless of nested markup. Preserves a leading span.quick-verdict-inline.
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(["h2", "h3"]):
        # Visible text including nested elements
        full_txt = " ".join(tag.stripped_strings)
        m = re.match(r"^\s*(\d+\.\d+)\s+(.*?)\s*$", full_txt)
        if not m:
            continue

        num = m.group(1)
        if num not in mapping:
            continue

        new_title = mapping[num]

        # Preserve a leading badge if itâ€™s the first child
        leading_badge = None
        first_child = next((c for c in tag.contents if hasattr(c, "name")), None)
        if (
            first_child is not None
            and first_child.name == "span"
            and "quick-verdict-inline" in (first_child.get("class") or [])
        ):
            leading_badge = first_child.extract()

        tag.clear()
        if leading_badge:
            tag.append(leading_badge)
            tag.append(" ")
        tag.append(f"{num} {new_title}")

    return str(soup)


def _apply_outline_by_order(html: str, rewritten_pairs: list[tuple[str, str]]) -> str:
    soup = BeautifulSoup(html, "html.parser")

    h2_titles = [t for (n, t) in rewritten_pairs if n.endswith(".0")]
    h3_titles = [t for (n, t) in rewritten_pairs if not n.endswith(".0")]

    for tag in soup.find_all(["h2", "h3"]):
        if tag.name == "h2" and h2_titles:
            new_text = h2_titles.pop(0)
        elif tag.name == "h3" and h3_titles:
            new_text = h3_titles.pop(0)
        else:
            continue

        leading_badge = None
        first_child = next((c for c in tag.contents if hasattr(c, "name")), None)
        if (
            first_child is not None
            and first_child.name == "span"
            and "quick-verdict-inline" in (first_child.get("class") or [])
        ):
            leading_badge = first_child.extract()

        tag.clear()
        if leading_badge:
            tag.append(leading_badge)
            tag.append(" ")
        tag.append(new_text)

    return str(soup)

def _read_excerpt(p: Path, max_chars: int = 12000) -> str:
    try:
        txt = p.read_text(encoding="utf-8")
    except Exception:
        return ""
    return re.sub(r"\s+", " ", txt).strip()[:max_chars]

def sanitize_heading_title(t: str) -> str:
    # strip only spaced separators like " Ã¢â‚¬â€ extra", " - extra", " : extra"
    t = re.sub(r"\s+[Ã¢â‚¬â€œÃ¢â‚¬â€-]\s+.*$", "", t)   # drop spaced dash suffixes
    t = re.sub(r"\s+:\s+.*$", "", t)       # drop spaced colon suffixes
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _brand_hint_from_name(name: str) -> str:
    m = re.match(r"[A-Za-z]+", name or "")
    return m.group(0) if m else ""
    
def _force_intro_heading_to_who_is_this_for(html: str) -> str:
    """
    Find the first H2/H3 whose text contains 'Purpose of the Review' (with or without a '1.0 ' prefix)
    and replace its title with 'Who is this for?' (unnumbered). If a leading 'quick-verdict-inline'
    span exists, preserve it before the new title. Includes a raw-HTML fallback.
    """
    soup = BeautifulSoup(html, "html.parser")

    def _looks_like_purpose(text: str) -> bool:
        t = (text or "").strip()
        return bool(re.search(r'(?:\b1\.0\s+)?Purpose of the Review\b', t, re.I))

    for tag in soup.find_all(["h2", "h3"]):
        # Full text (includes span text)
        full_txt = " ".join(tag.stripped_strings)
        if not _looks_like_purpose(full_txt):
            continue

        # Preserve the quick verdict span if present
        lead_span = tag.find("span", class_="quick-verdict-inline")
        if lead_span:
            lead_span.extract()  # detach so we can reinsert

        # Rebuild the heading, unnumbered
        tag.clear()
        if lead_span:
            tag.append(lead_span)
        tag.append("Who is this for?")
        return str(soup)

    # Fallback: if not found via parsing (very unlikely), do a careful HTML replace
    html_txt = str(soup)
    html_txt = re.sub(
        r'(<h[23][^>]*>[^<]*?)(?:\d+\.\d+\s+)?Purpose of the Review([^<]*?</h[23]>)',
        r'\1Who is this for?\2',
        html_txt,
        flags=re.I | re.S
    )
    return html_txt




def flatten_thin_parent_sections(
    html: str,
    max_intro_words: int = 45,
    preserve_first_h2: bool = True,
) -> tuple[str, int]:
    """Promote H3 children when their H2 is only a thin container."""
    soup = BeautifulSoup(html or "", "html.parser")
    h2_nodes = list(soup.find_all("h2"))
    flattened = 0

    for h2_index, parent in enumerate(h2_nodes):
        if parent.parent is None or (preserve_first_h2 and h2_index == 0):
            continue

        section_nodes = []
        for sibling in list(parent.next_siblings):
            if getattr(sibling, "name", None) == "h2":
                break
            section_nodes.append(sibling)

        children = [
            node for node in section_nodes
            if getattr(node, "name", None) == "h3"
        ]
        if not children:
            continue

        first_child_index = section_nodes.index(children[0])
        lead_nodes = section_nodes[:first_child_index]
        lead_text = " ".join(
            node.get_text(" ", strip=True)
            if hasattr(node, "get_text") else str(node).strip()
            for node in lead_nodes
        )
        lead_words = re.findall(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)?", lead_text)
        if len(lead_words) > max(0, int(max_intro_words)):
            continue

        # Put the short lead after the first promoted heading so it remains in
        # the correct document section rather than falling under the previous H2.
        anchor = children[0]
        for node in lead_nodes:
            extracted = node.extract()
            anchor.insert_after(extracted)
            anchor = extracted

        for child in children:
            child.name = "h2"
        parent.decompose()
        flattened += 1

    return str(soup), flattened


def _primary_canonical_record(
    canonical_profile: dict | None,
    primary_product: str | None = None,
) -> dict:
    profile = canonical_profile or {}
    target = normalize_ws(primary_product or profile.get("primary_product") or "")
    for item in profile.get("products") or []:
        if not isinstance(item, dict):
            continue
        name = normalize_ws(str(item.get("name") or ""))
        if name.casefold() == target.casefold():
            return item
    return {}


def _primary_canonical_corpus(
    canonical_profile: dict | None,
    primary_product: str | None = None,
) -> str:
    record = _primary_canonical_record(canonical_profile, primary_product)
    values = [str(record.get("name") or "")]
    for attribute, fact in (record.get("facts") or {}).items():
        if not isinstance(fact, dict):
            continue
        values.extend([
            str(attribute),
            str(fact.get("canonical_value") or ""),
            str(fact.get("safe_wording") or ""),
            str(fact.get("evidence_excerpt") or ""),
        ])
    return normalize_ws(" ".join(values)).casefold()


def _heading_has_unsupported_primary_claim(
    title: str,
    canonical_profile: dict | None,
    structure_controls: dict | None,
) -> tuple[bool, str]:
    guard = ((structure_controls or {}).get("heading_claim_guard") or {})
    if guard.get("enabled", True) is False or not canonical_profile:
        return False, ""
    corpus = _primary_canonical_corpus(canonical_profile)
    for group in guard.get("risk_groups") or []:
        if not isinstance(group, dict):
            continue
        patterns = [str(value) for value in (group.get("patterns") or []) if str(value)]
        if not any(re.search(pattern, title or "", re.I) for pattern in patterns):
            continue
        support_terms = [
            str(value).casefold() for value in (group.get("support_terms") or [])
            if str(value).strip()
        ]
        if support_terms and any(term in corpus for term in support_terms):
            continue
        return True, str(group.get("name") or "unsupported_heading_claim")
    return False, ""


def _smooth_source_attribution(text: str) -> str:
    pattern = re.compile(
        r"\b((?:one|a)\s+(?:source|reviewer|user|owner|tester)\s+"
        r"(?:reports|notes|states|observes|found)):\s+(the|this|it)\b",
        re.I,
    )
    return pattern.sub(lambda match: f"{match.group(1)} that {match.group(2).lower()}", text)


def _apply_configured_regex_rewrites(
    text: str,
    rules: list,
) -> tuple[str, int]:
    """Apply category-neutral configured prose rewrites and return a count."""
    updated = str(text or "")
    total = 0
    for rule in rules or []:
        if not isinstance(rule, dict):
            continue
        pattern = str(rule.get("pattern") or "")
        replacement = str(rule.get("replacement") or "")
        if not pattern:
            continue
        try:
            updated, count = re.subn(pattern, replacement, updated, flags=re.I)
        except re.error:
            continue
        total += count
    return updated, total

def _editorial_topic_tokens(value: str, controls: dict, product_words: set[str]) -> set[str]:
    value = str(value or "")
    # Compound phrases may contain a heading word while referring to another
    # topic: "water bottle pockets" is storage, not weather protection.
    for phrase in controls.get("section_alignment_ignored_phrases", []):
        phrase = normalize_ws(str(phrase))
        if phrase:
            value = re.sub(rf"(?<!\w){re.escape(phrase)}(?!\w)", " ", value, flags=re.I)
    stop_words = {
        str(item).casefold()
        for item in controls.get(
            "section_alignment_stop_words",
            ["and", "or", "the", "a", "an", "for", "with", "of", "to", "in", "on", "its", "this", "that", "product", "model", "review", "features", "performance", "design"],
        )
    }
    result = set()
    for word in re.findall(r"[a-z0-9]+", value.casefold()):
        if word in stop_words or word in product_words or len(word) < 3:
            continue
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith("s") and len(word) > 4:
            word = word[:-1]
        result.add(word)
    return result


def _final_qa_remove_repeated_canonical_facts(
    soup: BeautifulSoup,
    canonical_profile: dict | None,
    primary_name: str,
    controls: dict,
) -> int:
    """Keep the best-located occurrence of repeated single-source facts."""
    record = _primary_canonical_record(canonical_profile, primary_name)
    if not record:
        return 0
    default_limit = max(1, int(controls.get("single_source_fact_max_occurrences", 1)))
    configured_limits = controls.get("canonical_fact_occurrence_limits") or {}
    topic_terms = controls.get("canonical_fact_topic_terms") or {}
    sentence_rx = re.compile(r"(?P<space>\s*)(?P<body>[^.!?\n]+)(?P<punct>[.!?]?)")
    removed = 0

    for attribute, fact in (record.get("facts") or {}).items():
        if not isinstance(fact, dict) or not fact.get("requires_attribution"):
            continue
        safe = normalize_ws(str(fact.get("safe_wording") or ""))
        if not safe:
            continue
        marker_text = " ".join([
            str(fact.get("canonical_value") or ""),
            safe,
            str(fact.get("evidence_excerpt") or ""),
        ])
        fact_markers = _final_qa_measurement_markers(marker_text)
        safe_key = re.sub(r"[^a-z0-9]+", " ", safe.casefold()).strip()
        safe_core = re.sub(
            r"^(?:one|a)\s+(?:source|reviewer|user|owner|tester)\s+"
            r"(?:reports?|notes?|states?|observes?|found)\s*(?:that|:)?\s*",
            "",
            safe_key,
        ).strip()
        occurrences = []
        for node in list(soup.find_all(string=True)):
            if not isinstance(node, NavigableString):
                continue
            if getattr(node.parent, "name", "") in {
                "script", "style", "noscript", "table", "td", "th"
            }:
                continue
            original = str(node)
            for match in sentence_rx.finditer(original):
                body = normalize_ws(match.group("body"))
                key = re.sub(r"[^a-z0-9]+", " ", body.casefold()).strip()
                if not key:
                    continue
                attributed = bool(re.search(
                    r"\b(?:one|a|the)\s+(?:source|reviewer|user|owner|tester)\b|"
                    r"\b(?:manufacturer|listing)\s+(?:reports|states|lists|gives)\b",
                    body,
                    re.I,
                ))
                sentence_markers = _final_qa_measurement_markers(body)
                safe_wording_match = key == safe_key or key.endswith(safe_core)
                measurement_match = bool(fact_markers & sentence_markers)
                if not attributed or not (safe_wording_match or measurement_match):
                    continue
                heading = node.parent.find_previous(["h2", "h3"])
                heading_text = normalize_ws(heading.get_text(" ", strip=True)) if heading else ""
                raw_terms = [
                    str(term)
                    for term in topic_terms.get(str(attribute), [])
                    if str(term).strip()
                ]
                raw_terms.append(str(attribute).replace("_", " "))
                terms = _editorial_topic_tokens(
                    " ".join(raw_terms), controls, set()
                )
                heading_tokens = _editorial_topic_tokens(
                    heading_text, controls, set()
                )
                occurrences.append((
                    len(heading_tokens & terms), node, match.start(), match.end(), original
                ))

        limit = max(1, int(configured_limits.get(str(attribute), default_limit)))
        if len(occurrences) <= limit:
            continue
        keep = {
            (id(item[1]), item[2])
            for item in sorted(occurrences, key=lambda item: item[0], reverse=True)[:limit]
        }
        by_node = {}
        for item in occurrences:
            by_node.setdefault(id(item[1]), {"node": item[1], "original": item[4], "ranges": []})
            if (id(item[1]), item[2]) not in keep:
                by_node[id(item[1])]["ranges"].append((item[2], item[3]))
                removed += 1
        for data in by_node.values():
            if not data["ranges"]:
                continue
            updated = data["original"]
            for start, end in sorted(data["ranges"], reverse=True):
                updated = updated[:start] + " " + updated[end:]
            data["node"].replace_with(NavigableString(re.sub(r"\s{2,}", " ", updated)))

    for paragraph in list(soup.find_all("p")):
        if not normalize_ws(paragraph.get_text(" ", strip=True)) and not paragraph.find(
            ["a", "img", "figure"]
        ):
            paragraph.decompose()
    return removed


def _final_qa_remove_cross_model_advice(
    soup: BeautifulSoup,
    canonical_profile: dict | None,
    primary_name: str,
    controls: dict,
) -> int:
    """Stop a related model's uncertainty becoming primary-product advice."""
    names = [
        normalize_ws(str(item.get("name") or ""))
        for item in (canonical_profile or {}).get("products") or []
        if isinstance(item, dict) and normalize_ws(str(item.get("name") or ""))
    ]
    other_names = [name for name in names if name.casefold() != primary_name.casefold()]
    if not primary_name or not other_names:
        return 0
    uncertainty_rx = re.compile(str(controls.get(
        "related_model_uncertainty_pattern",
        r"\b(?:uncertain|uncertainty|doubt|questioned|whether|may not|might not)\b",
    )), re.I)
    advice_rx = re.compile(str(controls.get(
        "related_model_advice_pattern",
        r"\b(?:buyers?|owners?|users?)\b[^.!?]{0,100}\b(?:should|advisable|recommend)|"
        r"\b(?:testing|checking|confirming|trying)\b[^.!?]{0,100}\b(?:before (?:buying|purchase)|advisable|recommended)",
    )), re.I)
    sentence_rx = re.compile(r"[^.!?]+[.!?]?")
    removed = 0
    for paragraph in list(soup.find_all("p")):
        if paragraph.find(True):
            continue
        original = str(paragraph.string or "")
        sentences = [m.group(0) for m in sentence_rx.finditer(original) if normalize_ws(m.group(0))]
        discard = set()
        related_uncertainty = False
        for index, sentence in enumerate(sentences):
            folded = sentence.casefold()
            mentions_other = any(name.casefold() in folded for name in other_names)
            if mentions_other and uncertainty_rx.search(sentence):
                discard.add(index)
                related_uncertainty = True
                continue
            if related_uncertainty and not mentions_other:
                if advice_rx.search(sentence):
                    discard.add(index)
                related_uncertainty = False
        if discard:
            paragraph.string.replace_with(" ".join(
                normalize_ws(sentence)
                for index, sentence in enumerate(sentences)
                if index not in discard
            ))
            removed += len(discard)
    return removed


def _final_qa_measurement_markers(value: str) -> set[str]:
    """Return boundary-safe normalized numeric measurements from prose."""
    markers = set()
    patterns = [
        r"(?<!\w)\d+(?:\.\d+)?\s*(?:-|\u2013|\u2014)?\s*(?:%|[a-zA-Z]{1,12})(?!\w)",
        r"(?<!\w)\d+\s*['\u2019\u2032]\s*\d+(?:\s*[\"\u201d\u2033])?(?!\w)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, str(value or ""), re.I):
            marker = re.sub(r"[\s\-\u2013\u2014]+", "", match.group(0).casefold())
            if marker:
                markers.add(marker)
    return markers


def _final_qa_required_attribution(
    soup: BeautifulSoup,
    canonical_profile: dict | None,
    primary_name: str,
    source_fragment_rules: list,
) -> int:
    """Replace unattributed single-source measurements with approved wording."""
    record = _primary_canonical_record(canonical_profile, primary_name)
    if not record:
        return 0
    attributed_rx = re.compile(
        r"\b(?:according to|one|a|the)\s+(?:source|reviewer|user|owner|tester)\b|"
        r"\b(?:manufacturer|listing)\s+(?:reports|states|lists|gives)\b|"
        r"\breported\b",
        re.I,
    )
    fact_rules = []
    for _attribute, fact in (record.get("facts") or {}).items():
        if not isinstance(fact, dict) or not fact.get("requires_attribution"):
            continue
        safe = normalize_ws(str(fact.get("safe_wording") or ""))
        marker_text = " ".join([
            str(fact.get("canonical_value") or ""),
            safe,
            str(fact.get("evidence_excerpt") or ""),
        ])
        markers = _final_qa_measurement_markers(marker_text)
        if not markers:
            continue
        natural_safe, _ = _apply_configured_regex_rewrites(safe, source_fragment_rules)
        fact_rules.append((markers, natural_safe or safe))
    if not fact_rules:
        return 0

    profile_names = [
        normalize_ws(str(item.get("name") or ""))
        for item in (canonical_profile or {}).get("products") or []
        if isinstance(item, dict) and normalize_ws(str(item.get("name") or ""))
    ]
    primary_key = primary_name.casefold()
    changed = 0
    sentence_rx = re.compile(r"(?P<space>\s*)(?P<body>[^.!?\n]+)(?P<punct>[.!?]?)")

    for node in list(soup.find_all(string=True)):
        if not isinstance(node, NavigableString):
            continue
        parent_name = getattr(node.parent, "name", "")
        if parent_name in {"script", "style", "noscript", "table", "td", "th"}:
            continue
        original = str(node)

        def replace_sentence(match):
            nonlocal changed
            body = normalize_ws(match.group("body"))
            if not body or match.group("punct") == "?" or attributed_rx.search(body):
                return match.group(0)
            body_folded = body.casefold()
            mentioned_other = any(
                name.casefold() != primary_key and name.casefold() in body_folded
                for name in profile_names
            )
            if mentioned_other and primary_key not in body_folded:
                return match.group(0)
            sentence_markers = _final_qa_measurement_markers(body)
            for markers, safe in fact_rules:
                if not (markers & sentence_markers):
                    continue
                punctuation = "" if safe.endswith((".", "!", "?")) else "."
                changed += 1
                return f"{match.group('space')}{safe}{punctuation}"
            return match.group(0)

        updated = sentence_rx.sub(replace_sentence, original)
        if updated != original:
            node.replace_with(NavigableString(updated))

    for cell in soup.find_all("td"):
        cell_text = normalize_ws(cell.get_text(" ", strip=True))
        if not cell_text or attributed_rx.search(cell_text):
            continue
        row_text = normalize_ws(cell.parent.get_text(" ", strip=True)).casefold()
        mentioned_other = any(
            name.casefold() != primary_key and name.casefold() in row_text
            for name in profile_names
        )
        if mentioned_other and primary_key not in row_text:
            continue
        markers = _final_qa_measurement_markers(cell_text)
        if any(fact_markers & markers for fact_markers, _safe in fact_rules):
            cell.clear()
            cell.append(NavigableString(f"Reported: {cell_text}"))
            changed += 1
    return changed


def _final_qa_remove_adjacent_near_duplicates(
    soup: BeautifulSoup,
    controls: dict,
) -> int:
    """Remove raw adjacent source fragments duplicated by natural prose."""
    threshold = float(controls.get("near_duplicate_sentence_containment", 0.55))
    minimum_shared = max(2, int(controls.get("near_duplicate_sentence_min_shared_tokens", 3)))
    raw_prefixes = [
        str(value) for value in controls.get(
            "near_duplicate_raw_prefixes",
            [
                r"^(?:one|a)\s+(?:source|reviewer|user|owner|tester)\s+\w+:\s*",
                r"^(?:made from|includes|features|contains|weighs|measures)\b",
            ],
        )
        if str(value).strip()
    ]
    token_stops = {
        "one", "source", "reviewer", "user", "owner", "tester", "reports",
        "reported", "notes", "states", "says", "the", "a", "an", "it",
        "this", "that", "is", "was", "are", "were", "may", "can", "for",
        "of", "to", "in", "on", "with", "and", "or", "but", "not",
    }

    def tokens(sentence: str) -> set[str]:
        result = set()
        for token in re.findall(r"[a-z0-9]+", sentence.casefold()):
            if token in token_stops or len(token) < 3:
                continue
            if token.endswith("ies") and len(token) > 4:
                token = token[:-3] + "y"
            elif token.endswith("s") and len(token) > 4:
                token = token[:-1]
            result.add(token)
        return result

    removed = 0
    sentence_rx = re.compile(r"[^.!?]+[.!?]?")
    for paragraph in list(soup.find_all("p")):
        if paragraph.find(True):
            continue
        original = str(paragraph.string or "")
        sentences = [match.group(0) for match in sentence_rx.finditer(original) if normalize_ws(match.group(0))]
        if len(sentences) < 2:
            continue
        keep = [True] * len(sentences)
        for index in range(len(sentences) - 1):
            left = normalize_ws(sentences[index])
            right = normalize_ws(sentences[index + 1])
            left_tokens = tokens(left)
            right_tokens = tokens(right)
            shared = left_tokens & right_tokens
            containment = len(shared) / max(1, min(len(left_tokens), len(right_tokens)))
            left_is_raw = any(re.search(pattern, left, re.I) for pattern in raw_prefixes)
            if left_is_raw and len(shared) >= minimum_shared and containment >= threshold:
                keep[index] = False
                removed += 1
        if not all(keep):
            revised = " ".join(
                normalize_ws(sentence) for sentence, retain in zip(sentences, keep) if retain
            )
            paragraph.string.replace_with(revised)
    return removed


def _final_qa_remove_adjacent_attribute_repetition(
    soup: BeautifulSoup,
    controls: dict,
) -> int:
    """Remove an adjacent sentence that restates the same configured attribute."""
    topic_map = controls.get("canonical_fact_topic_terms") or {}
    attributes = [
        str(value)
        for value in controls.get(
            "adjacent_attribute_dedupe_attributes",
            ["ventilation"],
        )
        if str(value).strip()
    ]
    minimum_terms = max(
        1, int(controls.get("adjacent_attribute_dedupe_min_terms", 2))
    )
    minimum_shared = max(
        2, int(controls.get("adjacent_attribute_dedupe_min_shared_tokens", 3))
    )
    threshold = float(
        controls.get("adjacent_attribute_dedupe_containment", 0.35)
    )
    stops = {
        "the", "a", "an", "and", "or", "but", "however", "so", "it",
        "this", "that", "your", "its", "is", "are", "was", "were", "be",
        "can", "could", "may", "might", "during", "for", "to", "of",
        "in", "on", "with", "without", "against", "from",
    }

    def tokens(value: str) -> set[str]:
        return {
            token[:-1] if token.endswith("s") and len(token) > 4 else token
            for token in re.findall(r"[a-z0-9]+", value.casefold())
            if len(token) >= 3 and token not in stops
        }

    def term_hits(value: str, attribute: str) -> int:
        folded = value.casefold()
        terms = [
            str(term).casefold()
            for term in topic_map.get(attribute, [])
            if str(term).strip()
        ]
        return sum(
            1 for term in terms
            if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", folded)
        )

    removed = 0
    sentence_rx = re.compile(r"[^.!?]+[.!?]?")
    for paragraph in list(soup.find_all("p")):
        if paragraph.find(True):
            continue
        original = str(paragraph.string or "")
        sentences = [
            match.group(0)
            for match in sentence_rx.finditer(original)
            if normalize_ws(match.group(0))
        ]
        if len(sentences) < 2:
            continue
        keep = [True] * len(sentences)
        for index in range(len(sentences) - 1):
            left = normalize_ws(sentences[index])
            right = normalize_ws(sentences[index + 1])
            matching_attribute = next(
                (
                    attribute for attribute in attributes
                    if term_hits(left, attribute) >= minimum_terms
                    and term_hits(right, attribute) >= minimum_terms
                ),
                "",
            )
            if not matching_attribute:
                continue
            left_tokens = tokens(left)
            right_tokens = tokens(right)
            shared = left_tokens & right_tokens
            containment = len(shared) / max(
                1, min(len(left_tokens), len(right_tokens))
            )
            if len(shared) < minimum_shared or containment < threshold:
                continue
            # Retain the more informative sentence; on a tie retain the latter
            # because it generally explains the preceding summary.
            remove_index = index if len(left_tokens) <= len(right_tokens) else index + 1
            keep[remove_index] = False
            removed += 1
        if not all(keep):
            paragraph.string.replace_with(" ".join(
                normalize_ws(sentence)
                for sentence, retain in zip(sentences, keep)
                if retain
            ))
    return removed


def _final_qa_scope_ambiguous_measurements(
    soup: BeautifulSoup,
    canonical_profile: dict | None,
    primary_name: str,
) -> int:
    """Name a measurement's owner when nearby prose is about another model."""
    products = [
        item for item in (canonical_profile or {}).get("products") or []
        if isinstance(item, dict) and normalize_ws(str(item.get("name") or ""))
    ]
    if len(products) < 2:
        return 0
    marker_owners = {}
    for product in products:
        owner = normalize_ws(str(product.get("name") or ""))
        for fact in (product.get("facts") or {}).values():
            if not isinstance(fact, dict):
                continue
            marker_text = " ".join([
                str(fact.get("canonical_value") or ""),
                str(fact.get("safe_wording") or ""),
                str(fact.get("evidence_excerpt") or ""),
            ])
            for marker in _final_qa_measurement_markers(marker_text):
                marker_owners.setdefault(marker, set()).add(owner)
    unique_owners = {
        marker: next(iter(owners))
        for marker, owners in marker_owners.items()
        if len(owners) == 1
    }
    if not unique_owners:
        return 0

    active_product = primary_name
    changed = 0
    sentence_rx = re.compile(r"(?P<space>\s*)(?P<body>[^.!?\n]+)(?P<punct>[.!?]?)")
    for node in list(soup.find_all(string=True)):
        if not isinstance(node, NavigableString):
            continue
        if getattr(node.parent, "name", "") in {
            "script", "style", "noscript", "table", "td", "th"
        }:
            continue
        original = str(node)

        def replace_sentence(match):
            nonlocal active_product, changed
            body = normalize_ws(match.group("body"))
            if not body:
                return match.group(0)
            body_normalized = normalize_name(body)
            mentioned = [
                normalize_ws(str(product.get("name") or ""))
                for product in products
                if normalize_name(str(product.get("name") or "")) in body_normalized
            ]
            if mentioned:
                active_product = mentioned[0]
            owners = {
                unique_owners[marker]
                for marker in _final_qa_measurement_markers(body)
                if marker in unique_owners
            }
            if len(owners) != 1:
                return match.group(0)
            owner = next(iter(owners))
            if (
                normalize_name(owner)
                == normalize_name(active_product)
                or any(
                    normalize_name(owner)
                    == normalize_name(name)
                    for name in mentioned
                )
            ):
                return match.group(0)
            referent_rx = re.compile(
                r"^((?:one|a|the)\s+(?:source|reviewer|user|owner|tester)\s+"
                r"(?:reports?|notes?|states?|observes?|found)\s*(?:that|:)?)\s+"
                r"(?:the\s+(?:pack|product|model)|it)\b",
                re.I,
            )
            referent_match = referent_rx.search(body)
            count = 0
            scoped = body
            if referent_match:
                prefix = referent_match.group(1).rstrip()
                if not re.search(r"(?:\bthat|:)$", prefix, re.I):
                    prefix += " that"
                scoped = prefix + " " + owner + body[referent_match.end():]
                count = 1
            if not count:
                scoped, count = re.subn(
                    r"^(?:the\s+(?:pack|product|model)|it)\b",
                    owner,
                    body,
                    count=1,
                    flags=re.I,
                )
            if not count:
                return match.group(0)
            active_product = owner
            changed += 1
            punctuation = match.group("punct")
            return f"{match.group('space')}{scoped}{punctuation}"

        updated = sentence_rx.sub(replace_sentence, original)
        if updated != original:
            node.replace_with(NavigableString(updated))
    return changed


def _final_qa_qualify_observational_table_values(
    soup: BeautifulSoup,
    canonical_profile: dict | None,
    controls: dict,
) -> int:
    """Present single-source load observations as observations, not ratings."""
    attributes = {
        str(value).casefold()
        for value in controls.get(
            "observational_table_attributes",
            ["load_support"],
        )
        if str(value).strip()
    }
    rules = []
    for product in (canonical_profile or {}).get("products") or []:
        if not isinstance(product, dict):
            continue
        for attribute, fact in (product.get("facts") or {}).items():
            if (
                str(attribute).casefold() not in attributes
                or not isinstance(fact, dict)
                or not fact.get("requires_attribution")
            ):
                continue
            canonical = normalize_ws(str(fact.get("canonical_value") or ""))
            markers = _final_qa_measurement_markers(" ".join([
                canonical,
                str(fact.get("safe_wording") or ""),
                str(fact.get("evidence_excerpt") or ""),
            ]))
            display = re.sub(
                r"^(?:up to|about|around)\s+",
                "",
                canonical,
                flags=re.I,
            )
            if markers and display:
                rules.append((markers, display))
    changed = 0
    for cell in soup.find_all("td"):
        value = normalize_ws(cell.get_text(" ", strip=True))
        markers = _final_qa_measurement_markers(value)
        for fact_markers, display in rules:
            if not (markers & fact_markers):
                continue
            replacement = f"Reported manageable around {display}"
            if value.casefold() == replacement.casefold():
                break
            cell.clear()
            cell.append(NavigableString(replacement))
            changed += 1
            break
    return changed


def _final_qa_qualify_collective_material_claims(
    soup: BeautifulSoup,
    canonical_profile: dict | None,
) -> int:
    """Qualify collective material claims when model specifications differ."""
    values = {
        normalize_ws(str((fact or {}).get("canonical_value") or "")).casefold()
        for product in (canonical_profile or {}).get("products") or []
        if isinstance(product, dict)
        for attribute, fact in (product.get("facts") or {}).items()
        if str(attribute).casefold() in {"material", "materials", "construction"}
        and normalize_ws(str((fact or {}).get("canonical_value") or ""))
    }
    if len(values) < 2:
        return 0
    collective_rx = re.compile(
        r"\b(?:all\s+(?:three|\d+)|both|the\s+(?:three|\d+)\s+models?)\b",
        re.I,
    )
    material_rx = re.compile(
        r"\b(?:material|fabric|construction|shell|coating|laminate|ripstop)\b",
        re.I,
    )
    changed = 0
    sentence_rx = re.compile(r"(?P<body>[^.!?\n]+)(?P<punct>[.!?]?)")
    for node in list(soup.find_all(string=True)):
        if not isinstance(node, NavigableString):
            continue
        if getattr(node.parent, "name", "") in {
            "script", "style", "noscript", "table", "td", "th"
        }:
            continue
        original = str(node)

        def qualify(match):
            nonlocal changed
            body = match.group("body")
            if (
                not collective_rx.search(body)
                or not material_rx.search(body)
                or re.search(r"specifications?\s+(?:vary|differ)", body, re.I)
            ):
                return match.group(0)
            changed += 1
            return (
                body.rstrip(" ,;")
                + ", although material specifications vary by model"
                + (match.group("punct") or ".")
            )

        updated = sentence_rx.sub(qualify, original)
        if updated != original:
            node.replace_with(NavigableString(updated))
    return changed


def _final_qa_replace_vague_table_values(soup: BeautifulSoup, controls: dict) -> int:
    vague_values = {
        normalize_ws(str(value)).casefold()
        for value in controls.get(
            "vague_table_values",
            ["full load", "normal load", "standard load", "typical load", "varies"],
        )
        if normalize_ws(str(value))
    }
    replacement = str(controls.get("vague_table_value_replacement", "Not confirmed"))
    changed = 0
    for cell in soup.find_all("td"):
        value = normalize_ws(cell.get_text(" ", strip=True))
        if value.casefold() not in vague_values:
            continue
        cell.clear()
        cell.append(NavigableString(replacement))
        changed += 1
    return changed

def apply_final_generic_editorial_controls(
    html: str,
    cfg: dict,
    canonical_profile: dict | None = None,
    primary_product: str | None = None,
) -> tuple[str, dict]:
    """Apply evidence-safe trust controls after links and heading rewrites."""
    controls = cfg.get("final_trust_controls") or {}
    report = {
        "enabled": bool(controls.get("enabled", True)),
        "hands_on_rewrites": 0,
        "attribution_rewrites": 0,
        "price_rewrites": 0,
        "source_fragment_rewrites": 0,
        "required_attribution_repairs": 0,
        "repeated_canonical_facts_removed": 0,
        "cross_model_inferences_removed": 0,
        "ambiguous_measurements_scoped": 0,
        "claim_strength_rewrites": 0,
        "adjacent_attribute_repetitions_removed": 0,
        "near_duplicate_sentences_removed": 0,
        "vague_table_values_replaced": 0,
        "observational_table_values_qualified": 0,
        "collective_material_claims_qualified": 0,
        "restrained_style_rewrites": 0,
        "raw_feature_fragments_removed": 0,
        "raw_feature_fragments_rewritten": 0,
        "malformed_source_rewrites": 0,
        "unsupported_comparative_rewrites": 0,
        "unsupported_security_benefits_removed": 0,
        "misplaced_summaries_removed": 0,
        "verdict_drawbacks_added": 0,
    }
    if controls.get("enabled", True) is False:
        return html, report

    soup = BeautifulSoup(html or "", "html.parser")
    record = _primary_canonical_record(canonical_profile, primary_product)
    primary_name = normalize_ws(
        primary_product
        or (canonical_profile or {}).get("primary_product")
        or record.get("name")
        or ""
    )

    trust_rules = controls.get("unverified_hands_on_rewrites") or [
        {
            "pattern": r"\bwe have tested its durability and features\s+to see\b",
            "replacement": "We examined its durability, features, and real-world owner feedback to see",
        },
        {
            "pattern": r"\bwe have tested\s+(.+?)\s+to see\b",
            "replacement": r"We examined \1 and real-world owner feedback to see",
        },
        {"pattern": r"\bwe tested\b", "replacement": "we assessed"},
        {"pattern": r"\bour testing\b", "replacement": "the available evidence"},
    ]
    hands_on_verified = bool(controls.get("hands_on_testing_verified", False))

    price_corpus = " ".join(
        " ".join([
            str(attribute),
            str((fact or {}).get("canonical_value") or ""),
            str((fact or {}).get("safe_wording") or ""),
        ])
        for attribute, fact in (record.get("facts") or {}).items()
        if isinstance(fact, dict)
        and re.search(r"price|cost|value|budget|premium", str(attribute), re.I)
    ).casefold()
    budget_supported = any(
        str(term).casefold() in price_corpus
        for term in controls.get("budget_support_terms", ["budget", "low-cost", "affordable", "inexpensive"])
    )
    premium_supported = any(
        str(term).casefold() in price_corpus
        for term in controls.get("premium_support_terms", ["premium", "higher upfront", "expensive", "high price"])
    )
    price_replacement = str(
        controls.get(
            "premium_price_replacement" if premium_supported else "neutral_price_replacement",
            "while its durability helps justify the higher upfront cost"
            if premium_supported
            else "with durability central to its long-term value",
        )
    )
    price_phrases = [
        str(value) for value in controls.get(
            "unsupported_budget_phrases",
            ["without breaking the bank", "won't break the bank", "will not break the bank"],
        )
        if str(value).strip()
    ]
    source_fragment_rules = controls.get("source_fragment_rewrites") or [
        {
            "pattern": r"\b((?:one|a)\s+(?:source|reviewer|user|owner|tester))\s+"
                       r"(?:reports|notes|states):\s+weighs\s+([^.!?]+)",
            "replacement": r"\1 lists its weight as \2",
        },
        {
            "pattern": r"\b((?:one|a)\s+(?:source|reviewer|user|owner|tester))\s+"
                       r"(?:reports|notes|states):\s+measures\s+([^.!?]+)",
            "replacement": r"\1 lists its dimensions as \2",
        },
    ]
    malformed_source_rules = controls.get("malformed_source_language_rewrites") or []
    restrained_style_rules = controls.get("restrained_style_rewrites") or []
    unsupported_comparative_rules = controls.get("unsupported_comparative_rewrites") or []
    claim_strength_rules = controls.get("claim_strength_rewrites") or []
    security_absence_markers = [
        str(value).casefold() for value in controls.get(
            "security_absence_markers",
            ["no anti-theft", "no dedicated security", "no security features", "pockets remain exposed", "pockets sit exposed"],
        )
        if str(value).strip()
    ]
    unsupported_security_rules = controls.get("unsupported_security_benefit_rewrites") or []

    for node in list(soup.find_all(string=True)):
        if not isinstance(node, NavigableString):
            continue
        if getattr(node.parent, "name", "") in {"script", "style", "noscript"}:
            continue
        original = str(node)
        updated = original
        if not hands_on_verified:
            for rule in trust_rules:
                pattern = str((rule or {}).get("pattern") or "")
                replacement = str((rule or {}).get("replacement") or "")
                if not pattern:
                    continue
                updated, count = re.subn(pattern, replacement, updated, flags=re.I)
                report["hands_on_rewrites"] += count
        updated, count = _apply_configured_regex_rewrites(
            updated, source_fragment_rules
        )
        report["source_fragment_rewrites"] += count
        smoothed = _smooth_source_attribution(updated)
        if smoothed != updated:
            report["attribution_rewrites"] += 1
            updated = smoothed
        if not budget_supported:
            for phrase in price_phrases:
                updated, count = re.subn(
                    re.escape(phrase), price_replacement, updated, flags=re.I
                )
                report["price_rewrites"] += count
        updated, count = _apply_configured_regex_rewrites(
            updated, malformed_source_rules
        )
        report["malformed_source_rewrites"] += count
        updated, count = _apply_configured_regex_rewrites(
            updated, restrained_style_rules
        )
        report["restrained_style_rewrites"] += count
        updated, count = _apply_configured_regex_rewrites(
            updated, unsupported_comparative_rules
        )
        report["unsupported_comparative_rewrites"] += count
        updated, count = _apply_configured_regex_rewrites(
            updated, claim_strength_rules
        )
        report["claim_strength_rewrites"] += count
        parent_text = normalize_ws(
            node.parent.get_text(" ", strip=True)
            if getattr(node, "parent", None) is not None
            and hasattr(node.parent, "get_text") else ""
        ).casefold()
        if (
            getattr(getattr(node, "parent", None), "name", "") == "p"
            and any(marker in parent_text for marker in security_absence_markers)
        ):
            updated, count = _apply_configured_regex_rewrites(
                updated, unsupported_security_rules
            )
            report["unsupported_security_benefits_removed"] += count
        if updated != original:
            node.replace_with(NavigableString(updated))

    report["ambiguous_measurements_scoped"] += (
        _final_qa_scope_ambiguous_measurements(
            soup, canonical_profile, primary_name
        )
    )
    report["required_attribution_repairs"] += _final_qa_required_attribution(
        soup,
        canonical_profile,
        primary_name,
        source_fragment_rules,
    )
    report["repeated_canonical_facts_removed"] += (
        _final_qa_remove_repeated_canonical_facts(
            soup, canonical_profile, primary_name, controls
        )
    )
    report["cross_model_inferences_removed"] += _final_qa_remove_cross_model_advice(
        soup, canonical_profile, primary_name, controls
    )
    report["near_duplicate_sentences_removed"] += (
        _final_qa_remove_adjacent_near_duplicates(soup, controls)
    )
    report["adjacent_attribute_repetitions_removed"] += (
        _final_qa_remove_adjacent_attribute_repetition(soup, controls)
    )
    report["vague_table_values_replaced"] += _final_qa_replace_vague_table_values(
        soup, controls
    )
    report["observational_table_values_qualified"] += (
        _final_qa_qualify_observational_table_values(
            soup, canonical_profile, controls
        )
    )
    report["collective_material_claims_qualified"] += (
        _final_qa_qualify_collective_material_claims(
            soup, canonical_profile
        )
    )
    if controls.get("clean_subjectless_fact_fragments", True):
        fragment_verbs = [
            re.escape(str(value).strip()) for value in controls.get(
                "subjectless_fact_fragment_verbs",
                ["includes", "features", "contains", "made from", "weighs", "measures"],
            )
            if str(value).strip()
        ]
        if fragment_verbs:
            fragment_templates = {
                "includes": "It includes {body}{punctuation}",
                "features": "It features {body}{punctuation}",
                "contains": "It contains {body}{punctuation}",
                "made from": "It is made from {body}{punctuation}",
                "weighs": "It weighs {body}{punctuation}",
                "measures": "It measures {body}{punctuation}",
            }
            fragment_templates.update({
                str(key).casefold(): str(value)
                for key, value in (
                    controls.get("subjectless_fact_fragment_templates") or {}
                ).items()
                if str(key).strip() and str(value).strip()
            })
            fragment_rx = re.compile(
                r"(^|(?<=[.!?])\s+)(" + "|".join(fragment_verbs) +
                r")\s+([^.!?]{12,260})([.!?])",
                re.I,
            )
            repetition_ratio = float(
                controls.get("subjectless_fragment_repetition_ratio", 0.6)
            )
            minimum_repeated_items = max(
                1, int(controls.get("subjectless_fragment_min_repeated_items", 2))
            )
            for node in list(soup.find_all(string=True)):
                if not isinstance(node, NavigableString):
                    continue
                if getattr(node.parent, "name", "") in {"script", "style", "noscript"}:
                    continue
                original = str(node)
                document_text = normalize_ws(soup.get_text(" ", strip=True)).casefold()

                def replace_fragment(match):
                    full_sentence = normalize_ws(match.group(0))
                    verb = match.group(2)
                    body = normalize_ws(match.group(3))
                    punctuation = match.group(4)
                    other_text = document_text.replace(full_sentence.casefold(), " ", 1)
                    items = [
                        normalize_ws(value).casefold()
                        for value in re.split(r"\s*(?:,|;|\band\b)\s*", body, flags=re.I)
                        if normalize_ws(value)
                    ]
                    repeated = sum(
                        1 for item in items
                        if len(re.findall(r"[a-z0-9]+", item)) >= 2
                        and item in other_text
                    )
                    fragment_minimum = (
                        1 if verb.casefold() in {"made from", "weighs", "measures"}
                        else minimum_repeated_items
                    )
                    required = max(
                        fragment_minimum,
                        int(len(items) * repetition_ratio + 0.999),
                    )
                    if items and repeated >= required:
                        report["raw_feature_fragments_removed"] += 1
                        return match.group(1)
                    report["raw_feature_fragments_rewritten"] += 1
                    template = fragment_templates.get(
                        verb.casefold(), "It {verb} {body}{punctuation}"
                    )
                    rewritten = template.format(
                        verb=verb.casefold(), body=body, punctuation=punctuation
                    )
                    return f"{match.group(1)}{rewritten}"

                updated = fragment_rx.sub(replace_fragment, original)
                if updated != original:
                    node.replace_with(NavigableString(updated))
            for paragraph in list(soup.find_all("p")):
                if not normalize_ws(paragraph.get_text(" ", strip=True)) and not paragraph.find(
                    ["a", "img", "figure"]
                ):
                    paragraph.decompose()

    if controls.get("remove_misplaced_section_summaries", True):
        product_words = set(re.findall(r"[a-z0-9]+", primary_name.casefold()))
        for heading in list(soup.find_all("h2")):
            heading_text = normalize_ws(heading.get_text(" ", strip=True))
            heading_tokens = _editorial_topic_tokens(heading_text, controls, product_words)
            if len(heading_tokens) < int(controls.get("section_alignment_min_heading_tokens", 2)):
                continue
            paragraphs = []
            node = heading.find_next_sibling()
            while node is not None and getattr(node, "name", None) != "h2":
                if getattr(node, "name", None) == "p" and normalize_ws(node.get_text(" ", strip=True)):
                    paragraphs.append(node)
                    if len(paragraphs) >= 2:
                        break
                node = node.find_next_sibling()
            if len(paragraphs) < 2:
                continue
            lead_tokens = _editorial_topic_tokens(
                paragraphs[0].get_text(" ", strip=True), controls, product_words
            )
            next_tokens = _editorial_topic_tokens(
                paragraphs[1].get_text(" ", strip=True), controls, product_words
            )
            prefix_min = max(
                5, int(controls.get("section_alignment_prefix_min_length", 6))
            )
            def matched_heading_topics(candidate_tokens: set[str]) -> int:
                matched = set()
                for heading_token in heading_tokens:
                    if heading_token in candidate_tokens:
                        matched.add(heading_token)
                        continue
                    if any(
                        len(heading_token) >= prefix_min
                        and len(candidate) >= prefix_min
                        and heading_token[:prefix_min] == candidate[:prefix_min]
                        for candidate in candidate_tokens
                    ):
                        matched.add(heading_token)
                return len(matched)

            minimum_shared = (
                1 if len(heading_tokens) < 3
                else max(2, int(controls.get("section_alignment_min_shared_tokens", 2)))
            )
            minimum_ratio = float(
                controls.get("section_alignment_min_topic_coverage", 0.34)
            )
            required_matches = min(
                len(heading_tokens),
                max(minimum_shared, int(math.ceil(len(heading_tokens) * minimum_ratio))),
            )
            lead_matches = matched_heading_topics(lead_tokens) >= required_matches
            next_matches = matched_heading_topics(next_tokens) >= required_matches
            if not lead_matches and next_matches:
                paragraphs[0].decompose()
                report["misplaced_summaries_removed"] += 1

    if controls.get("require_verdict_drawback", True) and record:
        verdict_patterns = controls.get(
            "verdict_heading_patterns", ["value for money", "final verdict", "verdict"]
        )
        drawback_markers = [
            str(value).casefold() for value in controls.get(
                "drawback_markers",
                [" no ", " not ", "without", "lack", "limited", "strain", "less suitable"],
            )
        ]
        priority = [
            str(value).casefold() for value in controls.get(
                "verdict_drawback_attribute_priority",
                ["ventilation", "load_support", "comfort", "limitations", "security", "water_protection"],
            )
        ]
        facts = record.get("facts") or {}
        ordered_facts = sorted(
            facts.items(),
            key=lambda item: (
                priority.index(str(item[0]).casefold())
                if str(item[0]).casefold() in priority else len(priority)
            ),
        )
        candidate = ""
        for _attribute, fact in ordered_facts:
            safe = normalize_ws(str((fact or {}).get("safe_wording") or ""))
            padded = f" {safe.casefold()} "
            if safe and any(
                re.search(
                    rf"(?<!\w){re.escape(marker.strip())}(?!\w)",
                    padded,
                    re.I,
                )
                for marker in drawback_markers
                if marker.strip()
            ):
                candidate = _smooth_source_attribution(safe)
                break
        if candidate:
            for heading in soup.find_all("h2"):
                heading_text = normalize_ws(heading.get_text(" ", strip=True)).casefold()
                if not any(str(pattern).casefold() in heading_text for pattern in verdict_patterns):
                    continue
                paragraphs = []
                node = heading.find_next_sibling()
                while node is not None and getattr(node, "name", None) != "h2":
                    if getattr(node, "name", None) == "p":
                        paragraphs.append(node)
                    node = node.find_next_sibling()
                if not paragraphs:
                    break
                verdict_text = f" {normalize_ws(' '.join(p.get_text(' ', strip=True) for p in paragraphs)).casefold()} "
                if any(
                    re.search(
                        rf"(?<!\w){re.escape(marker.strip())}(?!\w)",
                        verdict_text,
                        re.I,
                    )
                    for marker in drawback_markers
                    if marker.strip()
                ):
                    break
                if candidate.casefold() not in verdict_text:
                    punctuation = "" if candidate.endswith((".", "!", "?")) else "."
                    paragraphs[-1].append(NavigableString(f" {candidate}{punctuation}"))
                    report["verdict_drawbacks_added"] += 1
                break

    return str(soup), report

def _canonical_heading_numeric_tokens(canonical_profile: dict | None) -> set[str]:
    """Numbers allowed in rewritten titles for the exact primary product."""
    profile = canonical_profile or {}
    primary = str(profile.get("primary_product") or "").strip()
    products = profile.get("products") or []
    record = next(
        (
            item for item in products
            if isinstance(item, dict)
            and str(item.get("name") or "").strip().casefold() == primary.casefold()
        ),
        None,
    )
    number_rx = re.compile(r"\b\d+(?:\.\d+)?\s*(?:[a-z]{1,8}|%)?\b", re.I)

    def collect(value: str) -> set[str]:
        prepared = re.sub(
            r"(?<=\d)[\s-]+(?=[a-z])",
            "",
            (value or "").casefold(),
        )
        return {
            re.sub(r"\s+", "", match.group(0)).casefold()
            for match in number_rx.finditer(prepared)
        }

    allowed = collect(primary)
    for fact in ((record or {}).get("facts") or {}).values():
        if not isinstance(fact, dict):
            continue
        if str(fact.get("confidence") or "").casefold() != "high":
            continue
        if fact.get("requires_attribution"):
            continue
        allowed.update(collect(str(fact.get("canonical_value") or "")))
        allowed.update(collect(str(fact.get("safe_wording") or "")))
    return allowed


def rewrite_headings_in_file(final_html_path: Path,
                             blog_source_path: Path,
                             primary_category: str | None = None,
                             out_suffix: str = "_h2h3_rewritten",
                             brand_hint: str | None = None,
                             rewrite_focus_numbers: list[str] | None = None,
                             structure_controls: dict | None = None,
                             canonical_profile: dict | None = None) -> Path:
    """
    Rewrites <h2>/<h3> titles using model output while preserving numbering and order.

    Rules:
      - Preserve numbering exactly (e.g., "1.0 ..." stays "1.0 ...").
      - Only titles change; numbers and heading levels are untouched.
      - For numbers in rewrite_focus_numbers, use model's rewritten titles.
      - For other numbers, keep the original base topic (sanitized).
      - If headings in HTML lack numbers, apply by order: H2s consume *.0, H3s consume the rest.
      - Never introduce ":" or "Ã¢â‚¬â€œÃ¢â‚¬â€-" in titles; cap length to 60 chars.
    """
    try:
        html = final_html_path.read_text(encoding="utf-8")
    except Exception as e:
        log.error(f"Failed to read HTML for heading rewrite: {e}", extra={"step": "rewrite_headings"})
        return final_html_path

    # Parse current headings (document order)
    headings = _extract_outline_from_html(html)
    numbered_in_html = any(h["number"] for h in headings)
    available_numbers = [h["number"] for h in headings if h.get("number")]
    if rewrite_focus_numbers is None or "*" in rewrite_focus_numbers:
        effective_focus_numbers = [
            number for number in available_numbers if number != "1.0"
        ]
    else:
        effective_focus_numbers = [
            number for number in rewrite_focus_numbers
            if number in available_numbers and number != "1.0"
        ]

    log.info("Heading rewrite mode",
             extra={"step": "rewrite_headings",
                    "extra_json": {"numbered_in_html": bool(numbered_in_html),
                                   "src": blog_source_path.name}})

    # Build ORIGINAL_OUTLINE for the prompt (use numbers if present, else fallback)
    original_outline = "\n".join(
        f"{h['number']} {h['title']}" for h in headings if h.get("number")
    )
    if not original_outline.strip():
        # Nothing to rewrite safely
        log.info("No outline available for rewrite; leaving headings unchanged.",
                 extra={"step": "rewrite_headings"})
        return final_html_path


    content_excerpt = _read_excerpt(blog_source_path, max_chars=12000)

    # Prepare the prompt (reuse your template and constraints)
    prompt = HEADING_PROMPT_TPL.safe_substitute(
        primary_category=primary_category,
        blog_source_name=blog_source_path.name,
        content_excerpt=content_excerpt,
        original_outline=original_outline,
        brand_hint=(brand_hint or ""),
        rewrite_focus_numbers=", ".join(effective_focus_numbers),
        canonical_facts=json.dumps(canonical_profile or {}, ensure_ascii=False, indent=2)[:14000],
    )


    # Call DeepSeek through the shared client
    try:
        client = _get_deepseek_client()
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
            extra_body={"thinking": {"type": "disabled"}},
        )
        log_deepseek_usage(
            resp,
            label="insert_amazon:rewrite_headings",
            requested_model=DEEPSEEK_MODEL,
        )
        rewritten_text = resp.choices[0].message.content.strip()
    except Exception as e:
        log.error(
            f"DeepSeek heading rewrite failed: {e}",
            extra={"step": "rewrite_headings"},
        )
        return final_html_path

    # Strip code fences and parse "N.N title" lines
    rewritten_text = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", rewritten_text)
    rewritten_text = re.sub(r"\n```$", "", rewritten_text)
    model_pairs = _parse_rewritten_lines(rewritten_text)
    if not model_pairs:
        log.warning("Model returned no parseable heading lines; leaving unchanged.",
                    extra={"step": "rewrite_headings"})
        return final_html_path

    # Build original pairs (used to keep base topic for NON-focus headings)
    orig_pairs = _parse_rewritten_lines(original_outline)
    if not orig_pairs:
        # As a fallback, synthesize from current parsed headings if numbered
        if numbered_in_html:
            orig_pairs = [(h["number"], h["title"]) for h in headings if h.get("number")]

    # If still empty, bail
    if not orig_pairs:
        log.info("No original pairs to align against; leaving headings unchanged.",
                 extra={"step": "rewrite_headings"})
        return final_html_path

    # Sanitize & cap length helper
    def _sanitize_cap(t: str) -> str:
        t = sanitize_heading_title(t or "")
        if len(t) > 60:
            t = (t[:60]).rstrip()
        return t

    focus_set = set(effective_focus_numbers)
    model_map = {n: _sanitize_cap(t) for (n, t) in model_pairs}
    base_map  = {n: _sanitize_cap(t) for (n, t) in orig_pairs}

    # Reject numeric details that are not licensed for the exact primary product.
    # Falling back to the existing heading is safer than publishing a related
    # model's size, compatibility or legacy specification as a heading.
    if canonical_profile:
        allowed_numbers = _canonical_heading_numeric_tokens(canonical_profile)
        number_rx = re.compile(r"\b\d+(?:\.\d+)?\s*(?:[a-z]{1,8}|%)?\b", re.I)
        for number, title in list(model_map.items()):
            prepared_title = re.sub(
                r"(?<=\d)[\s-]+(?=[a-z])",
                "",
                (title or "").casefold(),
            )
            used = {
                re.sub(r"\s+", "", match.group(0)).casefold()
                for match in number_rx.finditer(prepared_title)
            }
            unsupported = sorted(used - allowed_numbers)
            if unsupported:
                log.warning(
                    "Rejected rewritten heading with unsupported numeric fact",
                    extra={
                        "step": "rewrite_headings",
                        "extra_json": {
                            "number": number,
                            "title": title,
                            "unsupported_numbers": unsupported,
                        },
                    },
                )
                model_map[number] = base_map.get(number, title)

    # Reject rewritten headings that make a risky factual claim unsupported by
    # the exact primary-product profile. Fall back to the original base topic.
    if canonical_profile:
        for number, title in list(model_map.items()):
            unsupported, claim_group = _heading_has_unsupported_primary_claim(
                title,
                canonical_profile,
                structure_controls,
            )
            if unsupported:
                log.warning(
                    "Rejected rewritten heading with unsupported product claim",
                    extra={
                        "step": "rewrite_headings",
                        "extra_json": {
                            "number": number,
                            "title": title,
                            "claim_group": claim_group,
                        },
                    },
                )
                model_map[number] = base_map.get(number, title)

    # Build final (number -> title) mapping in original order to ensure stability
    final_pairs = []
    for num, base_title in orig_pairs:
        if num in focus_set and num in model_map:
            final_pairs.append((num, model_map[num]))
        else:
            # Keep the base topic (sanitized), even if model suggested something else
            final_pairs.append((num, base_map.get(num, base_title)))
            
    # Ã°Å¸â€˜â€¡ ADD THIS so it's always available (even if you log it or fall back)
    mapping = {n: t for (n, t) in final_pairs}

    # Apply to HTML
    try:
        if numbered_in_html:
            before_sample = re.findall(r"<h[23][^>]*>.*?</h[23]>", html, flags=re.S)[:3]
            new_html = _apply_outline_by_number(html, mapping)
            after_sample = re.findall(r"<h[23][^>]*>.*?</h[23]>", new_html, flags=re.S)[:3]
            log.debug("Headings before/after sample",
                      extra={"step": "rewrite_headings",
                             "extra_json": {"before": before_sample, "after": after_sample}})
        else:
            new_html = _apply_outline_by_order(html, final_pairs)


        controls = structure_controls or {}
        flattened_count = 0
        if controls.get("flatten_thin_parent_sections", True):
            new_html, flattened_count = flatten_thin_parent_sections(
                new_html,
                max_intro_words=int(controls.get("thin_parent_max_words", 45)),
                preserve_first_h2=bool(controls.get("preserve_first_h2", True)),
            )

        # Save in place
        final_html_path.write_text(new_html, encoding="utf-8")
        log.info("Headings rewritten and saved",
                 extra={"step": "rewrite_headings",
                        "extra_json": {"path": str(final_html_path),
                                       "h2": sum(1 for n, _ in final_pairs if n.endswith(".0")),
                                       "h3": sum(1 for n, _ in final_pairs if not n.endswith(".0")),
                                       "flattened_thin_h2": flattened_count}})
        return final_html_path
    except Exception as e:
        log.error(f"Failed to apply rewritten headings: {e}", extra={"step": "rewrite_headings"})
        return final_html_path


def main():
    # Ã¢â€â‚¬Ã¢â€â‚¬ Bootstrap & logging Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
    with time_block("bootstrap"):
        keyword, country, site, category = read_keyword_from_file()
        log.info("Current job", extra={"extra_json": {"keyword": keyword, "country": country, "site": site, "category": category}})
        if not keyword or country not in ["US", "UK", "CA"]:
            log.error("Invalid keyword or country. Please check config/current_keyword.csv.")
            return

        safe_keyword = keyword.replace(" ", "_")
        safe_keyword_country = f"{safe_keyword}_{country}"

        JOB_LOG_DIR = Path(os.getenv("LOG_DIR", "logs")) / safe_keyword_country
        JOB_LOG_DIR.mkdir(parents=True, exist_ok=True)

        # rotate per-run file handlers
        for h in list(root_logger.handlers):
            if isinstance(h, logging.handlers.RotatingFileHandler):
                root_logger.removeHandler(h)
                h.close()

        ts = _iso_now().replace(":", "-")
        txt_handler = logging.handlers.RotatingFileHandler(
            JOB_LOG_DIR / f"{ts}_{safe_keyword_country}_run_{RUN_ID}.txt",
            maxBytes=5_000_000, backupCount=5, encoding="utf-8"
        )
        txt_handler.setLevel(TRACE_LEVEL_NUM)
        txt_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | step=%(step)s trace=%(trace_id)s | %(message)s"))
        txt_handler.addFilter(EnsureFields())
        root_logger.addHandler(txt_handler)

        jsonl_handler = logging.handlers.RotatingFileHandler(
            JOB_LOG_DIR / f"{ts}_{safe_keyword_country}_run_{RUN_ID}.jsonl",
            maxBytes=5_000_000, backupCount=5, encoding="utf-8"
        )
        jsonl_handler.setLevel(TRACE_LEVEL_NUM)
        jsonl_handler.setFormatter(JsonFormatter())
        jsonl_handler.addFilter(EnsureFields())
        root_logger.addHandler(jsonl_handler)

        log.info("Per-job logging configured",
                 extra={"extra_json": {
                     "txt_log": str(getattr(txt_handler, "baseFilename", "")),
                     "jsonl_log": str(getattr(jsonl_handler, "baseFilename", "")),
                     "job_log_dir": str(JOB_LOG_DIR)
                 }})

    # Ã¢â€â‚¬Ã¢â€â‚¬ Config & credentials Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
    db = load_category_db()
    resolved_category = resolve_category(keyword, db, explicit_category=category)
    cfg = build_config_for_category(resolved_category, db)
    configure_runtime_category(cfg)
    log.info("Using category config", extra={"extra_json": {"category": resolved_category}})
    log.debug("Category config details",
              extra={"extra_json": {
                  "exclude_in_title": cfg.get("exclude_in_title"),
                  "include_browse_nodes": cfg.get("include_browse_nodes"),
                  "include_keywords": cfg.get("include_keywords"),
                  "title_required_terms": cfg.get("title_required_terms"),
                  "deny_browse_nodes": cfg.get("deny_browse_nodes"),
                  "variant_suffix": cfg.get("variant_suffix"),
              }})

    try:
        load_deepseek_api_key()
        creds = load_amazon_credentials()
    except Exception:
        log.critical("Failed to load API keys or credentials. Exiting.")
        return

    try:
        api, tag, base_url = create_amazon_api(creds, country)
        atexit.register(shutdown_amazon_api, api)
    except Exception as e:
        log.error(str(e))
        return

        _clear_failure_reason(output_dir)
    # Ã¢â€â‚¬Ã¢â€â‚¬ Paths Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
    output_dir = Path(f"output/{safe_keyword_country}")
    output_dir.mkdir(parents=True, exist_ok=True)

    blog_file = output_dir / f"generated_blog_content_{country}.txt"
    output_final_blog_file = output_dir / f"processed_blog_final_updated_{country}.txt"
    downloaded_images_csv_file = output_dir / f"downloaded_image_urls_{country}.csv"
    unmatched_products_csv_file = output_dir / f"unmatched_product_names_{country}.csv"
    extracted_names_csv_file = output_dir / f"extracted_product_names_{country}_2.csv"

    # Prefer NEW structured payload produced by generator
    payload_file = output_dir / f"generated_post_payload_{country}.json"
    metadata: dict = {}
    outline: list[dict] = []
    content = ""

    if payload_file.exists():
        try:
            payload = json.loads(payload_file.read_text(encoding="utf-8"))
            content  = payload.get("body_html") or ""
            metadata = payload.get("metadata") or {}
            outline  = payload.get("outline") or []
            log.info("Loaded structured payload for internal linking", extra={"step": "internal_links"})
        except Exception as e:
            log.error(f"Failed to parse payload: {e}", extra={"step": "internal_links"})

    # Fallback to legacy .txt **only if still empty**
    if not content:
        if not blog_file.exists():
            log.error(f"Missing blog file: {blog_file}")
            return
        content = blog_file.read_text(encoding="utf-8")

    if not content.strip():
        log.warning("Blog content is empty.")
        output_final_blog_file.write_text("", encoding="utf-8")
        return

    # Ã¢â€â‚¬Ã¢â€â‚¬ Internal links: ensure & consume slots BEFORE affiliate injection Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
    with time_block("step0.internal_links"):
        try:
            if "<!-- INTERNAL_LINK_SLOT:" not in content:
                content = ensure_internal_link_slots(content)
                log.info("Inserted default INTERNAL_LINK_SLOT markers", extra={"step": "internal_links"})

            site_index = load_site_index()
            if site_index and "<!-- INTERNAL_LINK_SLOT:" in content:
                # --- ensure metadata identifies THIS post before internal linking ---
                meta = dict(metadata or {})

                post_slug = build_post_slug(keyword, country, meta)
                meta.setdefault("slug", post_slug)
                meta.setdefault("url", f"/posts/{post_slug}")
                meta.setdefault("canonical_url", f"/posts/{post_slug}")  # optional, but helpful

                content = replace_internal_link_slots(content, meta, outline or [], site_index)

                log.info("Inserted internal links from site_index.json", extra={"step": "internal_links"})
            else:
                log.info("No internal link slots found or empty site index; skipping", extra={"step": "internal_links"})
        except Exception as e:
            log.error(f"Internal link step failed: {e}", extra={"step": "internal_links"})



    # Ã¢â€â‚¬Ã¢â€â‚¬ Step 1: Extract product names from final text (post-internal-links) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
    with time_block("step1.extract_names"):
        content = unwrap_anchor_slots(content)
        content = normalize_editorial_product_suffixes_in_text(content)
        explicit_product_names = extract_explicit_product_names(content)
        llm_product_names = extract_product_names(content)
        # Explicit HTML fields come first so the article's primary spelling wins
        # when an LLM returns an equivalent reordered alias.
        product_names = merge_product_name_candidates(explicit_product_names, llm_product_names)
        log.info("Merged deterministic and LLM product names", extra={"step": "step1.extract_names", "extra_json": {"explicit": len(explicit_product_names), "llm": len(llm_product_names), "merged": len(product_names)}})
        if not product_names:
            log.warning(
                "No product names extracted. Writing content unchanged."
            )

            # NEW: still upsert this post into site_index.json using the current content
            try:
                # Build slug from keyword + country (use your existing helper if you have one)
                post_slug = build_post_slug(keyword, country, metadata)  # or _keyword_slug_with_country(...)

                # Canonical URL for this post
                url = f"/posts/{post_slug}"

                # Seed keywords from metadata keywords/tags if available
                seed_terms = (
                    metadata.get("keywords")
                    or metadata.get("tags")
                    or [keyword]
                )

                # Auto-extract additional keywords from the current HTML
                # (use helper from internal_links: _extract_keywords_from_html(html, k=...))
                keywords = _extract_keywords_from_html(content, k=50)


                # Text for embedding (for similarity-based internal links)
                post_text_for_embed = " ".join(
                    [
                        metadata.get("title", ""),
                        metadata.get("summary", ""),
                        " ".join(seed_terms),
                        strip_html(content),
                    ]
                )[:12000]

                vec = _local_semantic_embed(post_text_for_embed)

                # Metadata passed to upsert (no embedding here)
                metadata_for_index = {
                    "url": url,
                    "slug": post_slug,
                    "title": metadata.get("title") or keyword.title(),
                    "keywords": keywords,
                }

                # Ã°Å¸â€˜â€¡ Use upsert (no duplicate entries for same url/slug)
                upsert_site_index_entry(metadata_for_index, vec)

            except Exception as e:
                log.error(
                    f"Could not upsert into site index (no-products path): {e}",
                    extra={"step": "internal_links"},
                )

            # Existing behaviour: write content unchanged and return
            output_final_blog_file.write_text(content, encoding="utf-8")
            return


        seen_names = set()
        deduped_product_names = []
        for name in product_names:
            cleaned = strip_best_prefix(name)
            norm = cleaned.strip().lower()
            if not norm:
                continue
            if norm not in seen_names:
                seen_names.add(norm)
                deduped_product_names.append(cleaned.strip())
                
    def passes_first_word_rule(name: str, keyword: str, cfg: dict) -> bool:
        if not cfg.get("require_keyword_first_word"):
            return True
        first = (name.split()[:1] or [""])[0].lower()
        # If you mean Ã¢â‚¬Å“first word must be one of the category keywordsÃ¢â‚¬Â
        allowed = set()
        for s in (cfg.get("include_keywords") or []):
            if isinstance(s, str) and s.strip():
                allowed.add(s.split()[0].lower())
        return first in allowed


    filtered_product_names = [
        n for n in deduped_product_names
        if len(n.split()) >= 2 and passes_first_word_rule(n, keyword, cfg)
    ]

    skipped_names = [n for n in deduped_product_names if len(n.split()) < 2]
    if skipped_names:
        logging.info(f"Ã°Å¸Å¡Â« Skipped one-word product names: {skipped_names}")

    # Ã¢â€â‚¬Ã¢â€â‚¬ Step 2: Amazon search & selection Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
    log.info("Step 2: Searching Amazon for product details.", extra={"step": "step2"})
    product_data = []
    unmatched = []
    selection_records = []

    for name in filtered_product_names:
        trace_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{RUN_ID}:{name}").hex[:12]
        with time_block("step2.search_and_extract", trace_id=trace_id, payload={"query": name}):
            try:
                (
                    chosen_item,
                    selected_img_url,
                    selected_img_size,
                    selected_img_width,
                    selected_img_height,
                    selected_img_dimension_status,
                    selected_img_content_ratio,
                    used_similar_fallback,
                ) = step2_search_and_extract_for_name(
                    api, name, cfg, trace_id,
                    excluded_asins={
                        pd.get("asin") for pd in product_data if pd.get("asin")
                    },
                )
            except AmazonEligibilityError as e:
                reason = str(e)
                log.error(reason, extra={"step": "step2.amazon_eligibility", "trace_id": trace_id})
                output_final_blog_file.write_text(content, encoding="utf-8")
                (output_dir / "amazon_api_not_eligible.txt").write_text(reason + "\n", encoding="utf-8")
                _write_failure_reason(output_dir, reason)
                sys.exit(5)
            if not chosen_item:
                unmatched.append({"name": name, "reason": "No acceptable product found", "match_status": "unmatched"})
                time.sleep(random.uniform(0.5, 1.5))
                continue

            label = (chosen_item.item_info.title.display_value
                     if (chosen_item.item_info and chosen_item.item_info.title) else name)

            # Explicit fallback state guarantees the existing Similar on Amazon label,
            # while the older heuristic remains a defence for unusual title formats.
            flag = bool(used_similar_fallback or is_substitute_name(name, label, cfg))

            asin = getattr(chosen_item, "asin", None)
            price_display = _safe_get_display_price(chosen_item, log=log, trace_id=trace_id)


            # If price is blank on the first pass, DO NOT do a per-item GetItems retry here.
            # Rationale: per-item retries increase throttling and are one of the reasons
            # prices appear on a second full run. We batch-refresh missing prices later
            # using refresh_missing_prices_second_pass(...).
            if price_display is None or str(price_display).strip() == "":
                try:
                    log.warning(
                        "Blank price extracted; deferring to batched second-pass price refresh",
                        extra={
                            "step": "step2.price_blank_deferred",
                            "trace_id": trace_id,
                            "extra_json": {"asin": asin, "name": name[:140]},
                        },
                    )
                except Exception:
                    pass


                gi = safe_get_items(api, [asin], retries=2, trace_id=trace_id)
                gi_items = getattr(gi, "items", None) if gi else None
                gi_item0 = gi_items[0] if gi_items else None
                if gi_item0:
                    price_display2 = _safe_get_display_price(gi_item0, log=log, trace_id=trace_id)
                    if price_display2 and str(price_display2).strip():
                        price_display = price_display2
                        chosen_item = gi_item0  # keep richer object for later extraction
                        try:
                            log.info(
                                "Price retry succeeded",
                                extra={
                                    "step": "step2.price_retry_ok",
                                    "trace_id": trace_id,
                                    "extra_json": {"asin": asin, "price": price_display},
                                },
                            )
                        except Exception:
                            pass
                    else:
                        try:
                            log.warning(
                                "Price retry still blank",
                                extra={
                                    "step": "step2.price_retry_still_blank",
                                    "trace_id": trace_id,
                                    "extra_json": {"asin": asin},
                                },
                            )
                        except Exception:
                            pass


            # --- PRICE/OFFERS DEBUG (raw dumps + listing[0] detail) ---
            try:
                offers = getattr(chosen_item, "offers", None)
                listings = getattr(offers, "listings", None) if offers else None
                summaries = getattr(offers, "summaries", None) if offers else None

                l0 = listings[0] if listings else None
                p0 = getattr(l0, "price", None) if l0 else None

                s0 = summaries[0] if summaries else None
                lp0 = getattr(s0, "lowest_price", None) if s0 else None

                # Raw dumps (same pattern you wanted)
                log_kv(
                    log, "info", "Price offers raw dump",
                    trace_id=trace_id,
                    asin=asin,
                    offers_is_none=(offers is None),
                    listings_len=(len(listings) if listings else 0),
                    summaries_len=(len(summaries) if summaries else 0),
                    offers_dump=_safe_dump(offers),
                    listings_dump=_safe_dump(listings),
                    listing0_dump=_safe_dump(l0),
                    listing0_price_dump=_safe_dump(p0),
                    summaries_dump=_safe_dump(summaries),
                    summary0_dump=_safe_dump(s0),
                    summary0_lowest_price_dump=_safe_dump(lp0),
                )
            except Exception as e:
                log.warning(
                    "Price offers raw dump failed",
                    extra={"trace_id": trace_id, "extra_json": {"asin": asin, "err": str(e)}},
                )



            # Availability (keep your existing method, but ensure it cannot crash logging)
            availability_message = None
            availability_type = None
            try:
                offers = getattr(chosen_item, "offers", None)
                listings = getattr(offers, "listings", None) if offers else None
                if listings:
                    availability = getattr(listings[0], "availability", None)
                    if availability:
                        availability_message = getattr(availability, "message", None)
                        availability_type = getattr(availability, "type", None)
            except Exception:
                pass

            # Listing[0] detail (structured, wonâ€™t throw unexpected keyword arg)
            try:
                offers = getattr(chosen_item, "offers", None)
                listings = getattr(offers, "listings", None) if offers else None
                l0 = listings[0] if listings else None
                p0 = getattr(l0, "price", None) if l0 else None

                log_kv(
                    log, "info", "Price listing[0] detail",
                    trace_id=trace_id,
                    asin=asin,
                    listing_price_is_none=(p0 is None),
                    listing_price_display_amount=getattr(p0, "display_amount", None) if p0 else None,
                    listing_price_amount=getattr(p0, "amount", None) if p0 else None,
                    listing_price_currency=getattr(p0, "currency", None) if p0 else None,
                    availability_message=availability_message,
                    availability_type=availability_type,
                )
            except Exception as e:
                log.warning(
                    "Price listing[0] detail failed",
                    extra={"trace_id": trace_id, "extra_json": {"asin": asin, "err": str(e)}},
                )
            # --- END PRICE/OFFERS DEBUG ---



            # existing line
            log_kv(
                log, "info", "Price extracted",
                trace_id=trace_id,
                asin=asin,
                price=price_display,
            )          

            price_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            product_data.append({
                "name": name,
                "asin": getattr(chosen_item, "asin", None),
                "img_url": selected_img_url,
                "img_size": selected_img_size,
                "img_width": selected_img_width,
                "img_height": selected_img_height,
                "img_dimension_status": selected_img_dimension_status,
                "img_content_ratio": selected_img_content_ratio,
                "label": label,
                "score": 1,
                "is_substitute": flag,
                "price": price_display,
                "match_status": "similar" if flag else "exact",
                "amazon_title": label,
                "duplicate_of": "",
                "price_ts": price_ts,
            })



            if not flag:
                log.info("Not flagged substitute",
                         extra={"step": "step2.substitute_flag",
                                "trace_id": trace_id,
                                "extra_json": {"extracted": name, "amazon_title": label[:200]}})
            selection_records.append(product_data[-1])


            # (a) DEBUG: log when we flag a substitute
            if cfg.get("debug_substitutes") and product_data[-1].get("is_substitute"):
                log.warning(
                    "Marked as substitute (title mismatch heuristic)",
                    extra={
                        "step": "step2.substitute_flag",
                        "trace_id": trace_id,
                        "extra_json": {
                            "extracted": name,
                            "amazon_title": label[:200],
                            "asin": product_data[-1].get("asin"),
                        },
                    },
                )
          

            log.info("Selection complete",
                     extra={"trace_id": trace_id, "extra_json": {
                         "asin": getattr(chosen_item, "asin", None),
                         "label": label[:140],
                         "has_image": bool(selected_img_url)
                     }})
            time.sleep(random.uniform(0.5, 1.5))


    # Deduplicate by ASIN
    seen_asins = set()
    unique_data = []
    asin_to_name = {}
    for pd in product_data:
        asin = pd["asin"]
        if asin not in seen_asins:
            seen_asins.add(asin); unique_data.append(pd); asin_to_name[asin] = pd["name"]
        else:
            pd["match_status"] = "duplicate_asin"
            pd["duplicate_of"] = asin_to_name[asin]
            log.warning("Duplicate ASIN skipped", extra={"extra_json": {"asin": asin, "duplicate_of": asin_to_name[asin]}})
    product_data = unique_data
    
    # Second pass: refresh missing prices after a short cooldown (only if needed)
    missing = [p for p in product_data if p.get("asin") and not str(p.get("price") or "").strip()]
    if missing:
        refresh_missing_prices_second_pass(api, product_data, base_url=base_url, tag=tag, trace_id=trace_id,  mini_passes=2, mini_pass_cooldown_sec=6.0,)

    if not product_data:
        reason = "No valid Amazon product data (no matches returned after retries / throttling)."
        logging.error(reason)

        # keep artifacts for debugging
        output_final_blog_file.write_text(content, encoding="utf-8")
        if unmatched:
            with unmatched_products_csv_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Product Name", "Reason"])
                for u in unmatched:
                    writer.writerow([u["name"], u["reason"]])

        (output_dir / "no_amazon_matches.txt").write_text(
            "No Amazon matches were found for this keyword.\n",
            encoding="utf-8"
        )
        _write_failure_reason(output_dir, reason)
        sys.exit(2)

    # CSVs (downloaded images + extracted names + unmatched)
    with downloaded_images_csv_file.open("w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Product Name (Extracted)", "ASIN", "Image URL", "Image Size Source", "Width", "Height", "Dimension Status", "Content Ratio"])
        for pd in product_data:
            writer.writerow([pd["name"], pd["asin"], pd.get("img_url", ""), pd.get("img_size", ""), pd.get("img_width", ""), pd.get("img_height", ""), pd.get("img_dimension_status", ""), pd.get("img_content_ratio", "")])

    valid_names = set(pd["name"] for pd in product_data)
    with extracted_names_csv_file.open("w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["Product Name"])
        for name in filtered_product_names:
            if True:  # Legacy one-column write; replaced below by the status-rich schema.
                writer.writerow([name])
            else:
                log.warning(f"Ã°Å¸â€”â€˜Ã¯Â¸Â Skipped product name '{name}' Ã¢â‚¬â€ duplicate ASIN removed")
    # Rewrite the legacy CSV artifacts with transparent, status-rich schemas.
    # The first column remains backward compatible for existing consumers.
    with downloaded_images_csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Product Name (Extracted)", "Match Status", "ASIN", "Amazon Title",
            "Image Eligible for Post", "Image URL", "Image Size Source", "Width",
            "Height", "Dimension Status", "Content Ratio",
        ])
        for row in build_image_selection_rows(product_data, cfg):
            writer.writerow(row)

    with extracted_names_csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Product Name", "Amazon Status", "ASIN", "Amazon Title",
            "Duplicate Of", "Reason",
        ])
        for row in build_extracted_status_rows(filtered_product_names, selection_records, unmatched):
            writer.writerow(row)


    if unmatched:
        with unmatched_products_csv_file.open("w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f); writer.writerow(["Product Name", "Reason"])
            for u in unmatched: writer.writerow([u["name"], u["reason"]])

    # Ã¢â€â‚¬Ã¢â€â‚¬ Step 3: Inject Amazon links & images (AFTER internal links) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
    logging.info("Step 3: Injecting links and images.")
    top_pick_name = load_top_pick(safe_keyword, country)
    top_pick_name = sanitize_top_pick_name(top_pick_name or "")

    if not top_pick_name:
        # Option A fallback: pick something reasonable so we can continue injecting links/images.
        fallback_name = None

        # Prefer an actual Amazon-resolved product name if available
        if product_data:
            try:
                fallback_name = (product_data[0] or {}).get("name")
            except Exception:
                fallback_name = None

        # Otherwise fall back to the first extracted product name
        if not fallback_name and filtered_product_names:
            fallback_name = filtered_product_names[0]

        if fallback_name:
            top_pick_name = fallback_name
            logging.warning(
                "Top pick missing in product_names json; falling back to '%s'.",
                top_pick_name
            )
            try:
                (output_dir / "no_top_pick_fallback.txt").write_text(
                    f"Top pick missing; fell back to: {top_pick_name}\n",
                    encoding="utf-8"
                )
            except Exception:
                pass
        else:
            logging.error("No top pick set and no fallback candidates available; aborting post.")
            try:
                (output_dir / "no_top_pick.txt").write_text(
                    "Top pick missing and no fallback available Ã¢â‚¬â€ aborting insert_amazon_links_images.py.\n",
                    encoding="utf-8"
                )
            except Exception:
                pass
            sys.exit(3)

    if (
        not reviewed_product_matches_keyword(top_pick_name, keyword, cfg)
        and (cfg.get("review_identity") or {}).get("block_on_mismatch", True)
    ):
        reason = (
            "Reviewed-product identity mismatch: "
            f"keyword '{keyword}' cannot be retargeted to '{top_pick_name}'."
        )
        logging.error(reason)
        (output_dir / "abandon_post.txt").write_text(reason + "\n", encoding="utf-8")
        _write_failure_reason(output_dir, reason)
        sys.exit(4)

    top_pick_normalized = normalize_name(top_pick_name)
    available_names = {normalize_name(pd["name"]) for pd in product_data}

    if top_pick_normalized not in available_names:
        # Top pick not found among Amazon matches Ã¢â‚¬â€ use the *closest* product that has an image.
        substitute_pd, score = _closest_product_with_image(product_data, top_pick_name, cfg)
        if substitute_pd:
            logging.warning(
                "Top pick '%s' not found in Amazon results; using substitute '%s' "
                "(similarity=%s) for image/CTA wiring.",
                top_pick_name, substitute_pd["name"], score
            )
            # Create a synthetic alias row so downstream code can look up the top pick by name.
            product_data.append({
                "name": top_pick_name,
                "asin": substitute_pd.get("asin"),
                "img_url": substitute_pd.get("img_url"),
                "img_size": substitute_pd.get("img_size"),
                "img_width": substitute_pd.get("img_width"),
                "img_height": substitute_pd.get("img_height"),
                "img_dimension_status": substitute_pd.get("img_dimension_status"),
                "img_content_ratio": substitute_pd.get("img_content_ratio"),
                "label": substitute_pd.get("label", substitute_pd.get("name", top_pick_name)),
                "score": substitute_pd.get("score", 0),
                "is_substitute": True,
                "substitute_for": top_pick_name,   # <-- MUST be here
                "price": substitute_pd.get("price"),  # <-- also add this so QV can show Ã‚Â£69.99 immediately
            })


        else:
            logging.error(
                "Top pick '%s' not found and no suitable substitute with an image was available. "
                "Proceeding without a dedicated top-pick image.",
                top_pick_name
            )
    else:
        # Top pick exists in product_data but it might have no image Ã¢â‚¬â€ borrow from the closest match with an image.
        try:
            top_pd = next(pd for pd in product_data if normalize_name(pd["name"]) == top_pick_normalized)
            if not top_pd.get("img_url"):
                substitute_pd, score = _closest_product_with_image(product_data, top_pick_name, cfg)
                if substitute_pd and substitute_pd is not top_pd:
                    logging.warning(
                        "Top pick '%s' found but has no image; borrowing image from '%s' "
                        "(similarity=%s).",
                        top_pick_name, substitute_pd["name"], score
                    )
                    top_pd["img_url"]  = substitute_pd.get("img_url")
                    top_pd["img_size"] = substitute_pd.get("img_size")
                    top_pd["img_width"] = substitute_pd.get("img_width")
                    top_pd["img_height"] = substitute_pd.get("img_height")
                    top_pd["img_dimension_status"] = substitute_pd.get("img_dimension_status")
                    top_pd["img_content_ratio"] = substitute_pd.get("img_content_ratio")
        except StopIteration:
            pass
            
    image_count = _count_products_with_images(product_data)

    if image_count <= 0:
        reason = "No usable Amazon images were found for any matched product."
        logging.error(reason)

        try:
            (output_dir / "no_images.txt").write_text(reason + "\n", encoding="utf-8")
        except Exception:
            pass

        _write_failure_reason(output_dir, reason)
        sys.exit(4)

    content_for_injection = unwrap_anchor_slots(content)

    final_processed_content = inject_links_and_images(
        content_for_injection, product_data, country, base_url, tag, top_pick_name, cfg
    )

    top_pick_asin = _find_asin_for_name(product_data, top_pick_name)
    if not top_pick_asin:
        log.warning("Top pick ASIN not found; skipping placeholder fix",
                    extra={"step":"inject","extra_json":{"top_pick_name": top_pick_name}})
    else:
        final_processed_content = fix_check_price_placeholders(
            final_processed_content, asin=top_pick_asin, base_url=base_url, tag=tag
        )
        
    # NEW: link Quick Verdict product name + CTA to the TOP PICK
    final_processed_content = fix_quick_verdict_links(
        final_processed_content,
        top_pick_name=top_pick_name,
        product_data=product_data,
        base_url=base_url,
        tag=tag,
    )


    # Ã¢â€â‚¬Ã¢â€â‚¬ Step 4: Save processed blog Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
    logging.info("Step 4: Saving processed blog.")
    output_final_blog_file.write_text(final_processed_content, encoding='utf-8')
    logging.info(f"âœ… Final processed blog saved to: {output_final_blog_file}")

    # Ã¢â€â‚¬Ã¢â€â‚¬ Step 5: Append this post to site index for future internal links Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
    try:
        # Build final slug for this post
        post_slug = build_post_slug(keyword, country, metadata)

        # Compute embedding from content (or summary)
        embedding_source = " ".join([
            metadata.get("title", ""),
            metadata.get("summary", ""),
            " ".join(metadata.get("keywords", [])),
            normalize_ws(strip_html(content)),
        ])
        embedding_vec = _local_semantic_embed(embedding_source)

        # Word count fallback if generator didn't supply it
        word_count = metadata.get("word_count")
        if not isinstance(word_count, int):
            word_count = len(strip_html(content).split())

        # Published date: prefer metadata, else "today" UTC
        published_at = metadata.get("published_at")
        if not published_at:
            published_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Build metadata record (NO embedding here Ã¢â‚¬â€œ that is passed separately)
        metadata_for_index = {
            "url": f"/posts/{post_slug}",
            "slug": post_slug,
            "title": metadata.get("title") or keyword,
            "summary": metadata.get("summary", ""),
            "tags": metadata.get("keywords", []),
            "keywords": metadata.get("keywords", []),
            "categories": metadata.get("categories", []),
            "published_at": published_at,
            "word_count": word_count,
            "anchor_candidates": metadata.get("anchor_candidates", []),
        }

        # Upsert into site_index.json (update existing entry with same url/slug or append new)
        upsert_site_index_entry(metadata_for_index, embedding_vec)
        log.info(
            "Upserted post into site_index.json",
            extra={"step": "internal_links", "extra_json": {"slug": post_slug}}
        )

    except Exception as e:
        log.error(f"Failed to update site_index.json: {e}", extra={"step": "internal_links"})



    # Ã¢â€â‚¬Ã¢â€â‚¬ Step 6: Optional heading rewrite (post-save, in-place) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
    blog_source_path = blog_file
    brand_hint = _brand_hint_from_name(top_pick_name)
    canonical_profile = {}
    canonical_profile_path = output_dir / f"canonical_product_profile_{country}.json"
    try:
        if canonical_profile_path.exists():
            canonical_profile = json.loads(canonical_profile_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning(
            f"Could not load canonical profile for heading rewrite: {exc}",
            extra={"step": "rewrite_headings"},
        )

    _ = rewrite_headings_in_file(
        final_html_path=output_final_blog_file,
        blog_source_path=blog_source_path,
        primary_category=resolved_category,
        brand_hint=brand_hint,
        rewrite_focus_numbers=None,
        structure_controls=cfg.get("heading_structure") or {},
        canonical_profile=canonical_profile,
    )
        
    # NEW: enforce unnumbered "Who is this for?" for the first section
    try:
        txt = output_final_blog_file.read_text(encoding="utf-8")
        txt = _force_intro_heading_to_who_is_this_for(txt)
        output_final_blog_file.write_text(txt, encoding="utf-8")
        log.info("Intro heading normalized to 'Who is this for?' without numbering.",
                 extra={"step": "rewrite_headings"})
    except Exception as e:
        log.error(f"Failed to normalize intro heading: {e}", extra={"step": "rewrite_headings"})

    # Ã¢â€â‚¬Ã¢â€â‚¬ Step 7: Country-specific affiliate tag tweak (CA only) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
    # Final deterministic trust controls run after heading rewriting/flattening,
    # because those stages can otherwise create a new heading/summary mismatch.
    try:
        txt = output_final_blog_file.read_text(encoding="utf-8")
        txt, trust_report = apply_final_generic_editorial_controls(
            txt,
            cfg,
            canonical_profile=canonical_profile,
            primary_product=top_pick_name,
        )
        output_final_blog_file.write_text(txt, encoding="utf-8")
        log.info(
            "Applied final generic editorial controls",
            extra={"step": "final_trust_controls", "extra_json": trust_report},
        )
    except Exception as exc:
        log.error(
            f"Failed to apply final generic editorial controls: {exc}",
            extra={"step": "final_trust_controls"},
        )

    tag_rewrite = (cfg.get("affiliate_tag_rewrites") or {}).get(country) or {}
    old_tag = str(tag_rewrite.get("from") or "").strip()
    new_tag = str(tag_rewrite.get("to") or "").strip()
    if old_tag and new_tag:
        try:
            txt = output_final_blog_file.read_text(encoding="utf-8")
            new_txt = txt.replace(old_tag, new_tag)
            if new_txt != txt:
                output_final_blog_file.write_text(new_txt, encoding="utf-8")
                logging.info(
                    "Rewrote configured affiliate tag for %s in %s",
                    country,
                    output_final_blog_file,
                )
        except Exception as e:
            logging.error(
                "Failed to apply configured affiliate tag rewrite for %s: %s",
                country,
                e,
            )



if __name__ == "__main__":
    main()




