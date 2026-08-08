# scrape_urls.py — anchored paths + CLI + sanitized output folder

import os
import re
import sys
import time
import json
import random
import gc
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from urllib.parse import urlparse

from shared_company_names import load_company_names

import pandas as pd
import requests
from rich.console import Console
from rich.logging import RichHandler

from bing_scraper import BingScraper
from url_utils import normalize_url_any  # shared helper


# ────────────────────────────────────────────────────────────────────────────────
# Base-dir anchoring
# ────────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
def p(*parts: str) -> str:
    return str(BASE_DIR.joinpath(*parts))

# ────────────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────────────
console = Console()
log = logging.getLogger("scraper")
filtered_log = logging.getLogger("filtered")

log.setLevel(logging.DEBUG)
filtered_log.setLevel(logging.DEBUG)
log.propagate = False
filtered_log.propagate = False

# UTF-8 safe Console handler detection
def _supports_utf8(stream) -> bool:
    enc = getattr(stream, "encoding", "") or ""
    return enc.lower().replace("-", "") == "utf8"

def _make_console_handler(verbose: bool) -> logging.Handler:
    level = logging.DEBUG if verbose else logging.INFO

    if _supports_utf8(sys.stdout):
        rich_console = Console(markup=True, stderr=False)
        rh = RichHandler(console=rich_console, rich_tracebacks=True,
                         markup=True, show_time=False)
        rh.setLevel(level)
        return rh

    else:
        class SafeStreamHandler(logging.StreamHandler):
            def emit(self, record):
                try:
                    msg = self.format(record)
                    enc = getattr(self.stream, "encoding", "ascii")
                    try:
                        msg.encode(enc)
                    except UnicodeEncodeError:
                        msg = msg.encode("ascii", "ignore").decode("ascii")
                    self.stream.write(msg + self.terminator)
                    self.flush()
                except Exception:
                    self.handleError(record)

        ch = SafeStreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        return ch

def setup_logging(log_dir: str, keyword: str, verbose: bool):
    safe_keyword = (keyword or "bootstrap").replace(" ", "_")
    target_dir = os.path.join(log_dir, safe_keyword)
    os.makedirs(target_dir, exist_ok=True)

    log_file = os.path.join(
        target_dir, f"scrape_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log"
    )

    log.handlers.clear()
    filtered_log.handlers.clear()

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    log.addHandler(fh)

    rh = _make_console_handler(verbose)
    log.addHandler(rh)

    filtered_file = os.path.join(target_dir, "filtered_out_urls.log")
    ff = logging.FileHandler(filtered_file, encoding="utf-8")
    ff.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    filtered_log.addHandler(ff)

    msg = f"Using log file: {Path(log_file).resolve()}"
    if isinstance(rh, RichHandler):
        rh.console.print(
            f"[bold green]{msg.split(':',1)[0]}:[/bold green] {msg.split(':',1)[1].strip()}"
        )
    else:
        print(msg)

# ────────────────────────────────────────────────────────────────────────────────
# Config management
# ────────────────────────────────────────────────────────────────────────────────

COUNTRY_SETTINGS = {
    "US": {"mkt": "en-US", "cc": "US"},
    "UK": {"mkt": "en-GB", "cc": "GB"},
    "CA": {"mkt": "en-CA", "cc": "CA"},
}

def load_config(path=p("config/results_settings.json")):
    defaults = {
        "min_raw_urls_to_accept": 30,
        "max_results_to_save": 25,
        "max_results_to_scrape": 75,
        "min_score_to_allow_multiple_per_domain": 8,
        "min_required_urls_to_proceed": 6,
        "min_required_urls_to_consider_success": 10,
        "site_names_to_strip": load_company_names(path),
    }

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if isinstance(cfg, dict):
            if not cfg.get("site_names_to_strip") and cfg.get("company_names"):
                cfg["site_names_to_strip"] = cfg["company_names"]
            defaults.update(cfg)
            log.debug(f"Loaded config: {defaults}")
    except Exception as e:
        log.warning(f"⚠️ Failed to load config from {path}: {e} — using defaults.")

    # basic validation / clamping
    try:
        defaults["max_results_to_save"] = int(defaults.get("max_results_to_save", 25))
        defaults["max_results_to_scrape"] = int(defaults.get("max_results_to_scrape", 75))

        if defaults["max_results_to_save"] < 1:
            defaults["max_results_to_save"] = 1

        if defaults["max_results_to_scrape"] < defaults["max_results_to_save"]:
            defaults["max_results_to_scrape"] = defaults["max_results_to_save"]

        defaults["max_results_to_scrape"] = min(defaults["max_results_to_scrape"], 500)

    except Exception:
        log.warning("⚠️ Invalid numeric config values; falling back to safe defaults.")
        defaults["max_results_to_save"] = 25
        defaults["max_results_to_scrape"] = 75

    return defaults


def read_current_keyword(path=p("config/current_keyword.csv")):
    import csv
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            row = next(reader, None)

        if not row or len(row) < 4:
            raise ValueError(
                f"Malformed row in {path!s}: {row!r} "
                f"(expected: keyword,country,site,category)"
            )

        keyword = row[0].strip()
        country = row[1].strip().upper()
        site = row[2].strip()
        category = row[3].strip().lower()

        return keyword, country, site, category

    except Exception as e:
        log.error(f"❌ Could not read current_keyword.csv: {e}")
        return None, None, None, None
def load_category_config(category: str):
    """
    Load category settings from config/category_config.json.

    Tries several key variants, e.g.:
        'air_purifiers'  -> 'air purifiers' (and lowercase versions)
    """
    json_path = p("config", "category_config.json")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            all_cfg = json.load(f)

        if not isinstance(all_cfg, dict):
            log.warning(
                f"⚠️ Category config at {json_path} is not a dict; "
                f"got {type(all_cfg).__name__}. Using empty config."
            )
            return {}

        raw = (category or "").strip()

        # Generate a small list of likely keys
        candidates = []

        # as-is
        if raw:
            candidates.append(raw)
            candidates.append(raw.lower())

            # underscores -> spaces
            unders_to_spaces = raw.replace("_", " ")
            candidates.append(unders_to_spaces)
            candidates.append(unders_to_spaces.lower())

            # spaces -> underscores
            spaces_to_unders = raw.replace(" ", "_")
            candidates.append(spaces_to_unders)
            candidates.append(spaces_to_unders.lower())

        # Pick the first that exists
        for key in candidates:
            if key in all_cfg:
                return all_cfg[key]

        log.warning(
            f"⚠️ No config found in {json_path} for category '{category}'. "
            f"Tried keys: {', '.join(candidates) or '[none]'}"
        )
        return {}

    except Exception as e:
        log.warning(f"⚠️ Failed to load category config from {json_path}: {e}")
        return {}



# ────────────────────────────────────────────────────────────────────────────────
# Keyword sanitizing
# ────────────────────────────────────────────────────────────────────────────────
DOMAIN_RE = re.compile(
    r"\b(?:https?://)?(?:www\.)?([a-z0-9-]+(?:\.[a-z0-9-]+)+)\b",
    re.IGNORECASE
)
TLD_RE = re.compile(
    r"\b([a-z0-9-]+)\.(com|co\.uk|co|net|org|io|uk|us|ca|de|fr|it|es|au|nl|se|no|fi|ru|cn|jp|kr)\b",
    re.IGNORECASE
)

def sanitize_keyword_for_folder(keyword: str, site_names_to_strip=None) -> str:
    if not isinstance(keyword, str) or not keyword.strip():
        return ""

    site_names_to_strip = set((site_names_to_strip or []))
    k = keyword

    k = re.sub(r"https?://\S+", " ", k, flags=re.IGNORECASE)
    k = DOMAIN_RE.sub(" ", k)
    k = TLD_RE.sub(" ", k)

    tokens = re.split(r"\s+", k)
    kept = []
    for t in tokens:
        if not t:
            continue
        tl = re.sub(r"[^a-z0-9\-]", "", t.lower())
        if tl in site_names_to_strip:
            continue
        kept.append(t)

    return " ".join(kept).strip()


# ────────────────────────────────────────────────────────────────────────────────
# Filters / validators
# ────────────────────────────────────────────────────────────────────────────────
def is_blocked_domain(domain, blocklist_domains=None, blocklist_suffixes=None):
    if not domain:
        return True
    domain = domain.lower().strip().lstrip("www.")

    blocklist_domains = blocklist_domains or set()
    blocklist_suffixes = blocklist_suffixes or ()

    if domain in blocklist_domains:
        return True

    for suffix in blocklist_suffixes:
        if domain.endswith(suffix):
            return True

    return False

AMAZON_PRODUCT_RE = re.compile(
    r"^https://www\.amazon\.(com|co\.uk|ca|de|fr|it|es)"
    r"(?:/[^/]+)?/(?:dp|gp/product)/[A-Z0-9]{10}",
    re.IGNORECASE,
)

def is_valid_amazon_product_url(url: str) -> bool:
    return AMAZON_PRODUCT_RE.match(url) is not None

def should_remove(url, title=""):
    if not url or not isinstance(url, str) or url.strip() == "":
        filtered_log.debug(
            f"Removed: {url} | Reason: empty_url | Title: \"{title}\""
        )
        return True, "empty_url"

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        if is_blocked_domain(domain, {"localhost", "0.0.0.0", ""}):
            filtered_log.debug(
                f"Removed: {url} | Reason: blocked_domain | Title: \"{title}\""
            )
            return True, "blocked_domain"

        if "amazon." in domain:
            if "adsystem" in domain or "assoc-amazon" in domain:
                filtered_log.debug(
                    f"Removed: {url} | Reason: amazon_ad_url | Title: \"{title}\""
                )
                return True, "amazon_ad_url"

            if not is_valid_amazon_product_url(url):
                filtered_log.debug(
                    f"Removed: {url} | Reason: not_amazon_product_url | Title: \"{title}\""
                )
                return True, "not_amazon_product_url"

        if url.startswith(("data:", "file:", "javascript:")):
            filtered_log.debug(
                f"Removed: {url} | Reason: non_http_url | Title: \"{title}\""
            )
            return True, "non_http_url"

    except Exception:
        filtered_log.debug(
            f"Removed: {url} | Reason: url_parse_error | Title: \"{title}\""
        )
        return True, "url_parse_error"

    return False, ""

# Text helpers
def clean_text(text):
    return re.sub(r"[^a-z0-9\s]", " ", str(text).lower())


def normalize_match_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def collapse_match_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def build_keyword_matcher(keyword: str) -> dict:
    spaced = normalize_match_text(keyword)
    collapsed = collapse_match_text(keyword)
    tokens = [t for t in spaced.split() if t]

    joined_ngrams = set()
    for n in range(2, len(tokens) + 1):
        for i in range(0, len(tokens) - n + 1):
            joined_ngrams.add("".join(tokens[i:i + n]))

    return {
        "raw": keyword,
        "spaced": spaced,
        "collapsed": collapsed,
        "tokens": tokens,
        "joined_ngrams": joined_ngrams,
    }


def keyword_matches_text(text: str, matcher: dict) -> bool:
    if not matcher:
        return False

    spaced_text = normalize_match_text(text)
    collapsed_text = collapse_match_text(text)

    if matcher.get("spaced") and matcher["spaced"] in spaced_text:
        return True

    if matcher.get("collapsed") and matcher["collapsed"] in collapsed_text:
        return True

    if any(ng in collapsed_text for ng in matcher.get("joined_ngrams", set())):
        return True

    sig_tokens = [t for t in matcher.get("tokens", []) if len(t) >= 3]
    if sig_tokens and all(re.search(rf"\b{re.escape(t)}\b", spaced_text) for t in sig_tokens):
        return True

    return False


def first_word(s: str) -> str:
    s = (s or "").strip().lower()
    return s.split(maxsplit=1)[0] if s else ""


def contains_any_word(haystack: str, words) -> bool:
    return any(re.search(rf"\b{re.escape(w)}\b", haystack) for w in words)

def contains_any(haystack: str, terms) -> bool:
    return any(t in haystack for t in terms)

# Global cross-category exclusions
HARD_EXCLUDE_TERMS = {
    "cart", "checkout", "basket", "add to cart", "buy now", "sold by",
    "order", "return policy",
}
SOFT_EXCLUDE_TERMS = {
    "price", "prices", "discount", "collection", "deal", "deals",
    "promotion", "clearance", "catalogue", "range", "delivery",
    "our products", "shipping",
}
EXCLUDED_DOMAINS = {
    "ebay.co.uk", "amazon.com", "luggageadvisor.uk", "realreviews.io", "alpinetrek.co.uk",
    "uk.trustpilot.com", "trustpilot.com", "ie.trustpilot.com",
    "nz.trustpilot.com", "selfridges.com", "au.trustpilot.com",
    "bestproductsreviews.co.uk", "tog24.com", "reviewmeta.com",
    "wiki.org.uk", "scam-detector.com", "tripadvisor.co.uk",
    "youtube.com", "johnlewis.com", "trustedreviews.com",
}
TITLE_PRIORITY_TERMS = [
    "review", "reviews", "best", "comparison", "compare", "vs", "top",
    "tested", "test", "overview", "breakdown", "roundup", "guide",
    "tips", "insights", "analysis",
]

# Default review item terms (can be overridden by category_config.json)
REVIEW_ITEM_TERMS = set()

# Whether the first word of the keyword must appear in the result text.
# This can be overridden per-category via category_config.json
REQUIRE_KEYWORD_FIRST_WORD = True



LOCALE_EXCLUDED_DOMAINS = {"baidu.com", "zhidao.baidu.com", "zhihu.com"}
LOCALE_EXCLUDED_TLDS = {".cn", ".ru"}

def passes_relevance_gate(title_raw: str,
                          description_raw: str,
                          path_raw: str,
                          keyword_matcher: dict) -> bool:
    """
    Review-preserving relevance gate.

    Keep review-style pages as the default target:
      - review/editorial intent is still required
      - then either:
          a) category item terms are present, OR
          b) there is a strong normalized keyword match

    This keeps pages like:
      - "Arc'teryx packing cubes review"
      - "Best packing cubes including Arc’teryx"
    while still rejecting most product/category/store pages that have no
    review/editorial intent.
    """
    title_clean = clean_text(title_raw)
    description_clean = clean_text(description_raw)
    path_clean = clean_text(path_raw)

    combined_clean = f"{title_clean} {description_clean} {path_clean}".strip()
    combined_raw = f"{title_raw} {description_raw} {path_raw}".strip()

    has_priority = (
        contains_any_word(title_clean, TITLE_PRIORITY_TERMS)
        or contains_any_word(combined_clean, TITLE_PRIORITY_TERMS)
    )

    has_item = (
        contains_any_word(title_clean, REVIEW_ITEM_TERMS)
        or contains_any_word(path_clean, REVIEW_ITEM_TERMS)
        or contains_any_word(combined_clean, REVIEW_ITEM_TERMS)
    )

    if not REVIEW_ITEM_TERMS:
        has_item = True

    strong_keyword_match = False
    if keyword_matcher:
        strong_keyword_match = (
            keyword_matches_text(title_raw, keyword_matcher)
            or keyword_matches_text(description_raw, keyword_matcher)
            or keyword_matches_text(path_raw, keyword_matcher)
            or keyword_matches_text(combined_raw, keyword_matcher)
        )

    if not has_priority:
        return False

    if has_item:
        return True

    if strong_keyword_match:
        return True

    return False

# ────────────────────────────────────────────────────────────────────────────────
# Filter results
# ────────────────────────────────────────────────────────────────────────────────
def filter_results(results, country, keyword_matcher):
    filtered_out, final_results = [], []

    for r in results:
        raw_url = r.get("url", "")
        decoded_url = normalize_url_any(raw_url)

        if not decoded_url or not isinstance(decoded_url, str):
            continue

        if not decoded_url.startswith(("http://", "https://")):
            continue

        # Hard removals
        flag, reason = should_remove(decoded_url, r.get("title", ""))
        if flag:
            filtered_out.append({
                "url_raw": raw_url,
                "decoded_url": decoded_url,
                "reason": reason,
                "title": r.get("title", ""),
                "description": r.get("description", ""),
                "country": country,
            })
            continue

        decoded_url = decoded_url.rstrip("=")
        parsed = urlparse(decoded_url)
        domain = parsed.netloc.lower()
        domain_wo = domain.lstrip("www.")
        path = parsed.path or "/"

        title_raw = r.get("title", "")
        description_raw = r.get("description", "")

        title_clean = clean_text(title_raw)
        description_clean = clean_text(description_raw)
        path_clean = clean_text(path)

        combined_clean = f"{title_clean} {description_clean} {path_clean}"

        # Hard domain exclusions
        if domain_wo in EXCLUDED_DOMAINS:
            filtered_out.append({
                "url_raw": raw_url,
                "decoded_url": decoded_url,
                "reason": "excluded_domain",
                "title": title_raw,
                "description": description_raw,
                "country": country,
            })
            continue

        # Locale-based exclusions
        if country in {"US", "UK", "CA"}:
            if domain_wo in LOCALE_EXCLUDED_DOMAINS:
                filtered_log.debug(
                    f"Removed {decoded_url} for locale_excluded_domain"
                )
                filtered_out.append({
                    "url_raw": raw_url,
                    "decoded_url": decoded_url,
                    "reason": "locale_excluded_domain",
                    "title": title_raw,
                    "description": description_raw,
                    "country": country,
                })
                continue

            if any(domain_wo.endswith(tld) for tld in LOCALE_EXCLUDED_TLDS):
                filtered_out.append({
                    "url_raw": raw_url,
                    "decoded_url": decoded_url,
                    "reason": "locale_excluded_tld",
                    "title": title_raw,
                    "description": description_raw,
                    "country": country,
                })
                continue

        # Keyword in domain heuristic
        domain_collapsed = collapse_match_text(domain_wo)
        keyword_collapsed = (keyword_matcher or {}).get("collapsed", "")
        if keyword_collapsed and keyword_collapsed in domain_collapsed and not re.search(
            r"/(review|blog|news|story|stories|test|insight|compare|guides?)/",
            path, re.I
        ):
            filtered_out.append({
                "url_raw": raw_url,
                "decoded_url": decoded_url,
                "reason": "keyword_in_domain",
                "title": title_raw,
                "description": description_raw,
                "country": country,
            })
            continue

        # Amazon relevance check
        is_amz = is_valid_amazon_product_url(decoded_url)

        if is_amz:
            amazon_text = f"{decoded_url} {title_raw} {description_raw} {path}".strip()
            amazon_text_clean = clean_text(amazon_text)

            has_keyword_match = keyword_matches_text(amazon_text, keyword_matcher)
            has_category_term = bool(REVIEW_ITEM_TERMS) and contains_any(amazon_text_clean, REVIEW_ITEM_TERMS)

            if not (has_keyword_match or has_category_term):
                filtered_out.append({
                    "url_raw": raw_url,
                    "decoded_url": decoded_url,
                    "reason": "amazon_irrelevant_product",
                    "title": title_raw,
                    "description": description_raw,
                    "country": country,
                })
                continue

        # Hard exclude terms
        if not is_amz and contains_any(combined_clean, HARD_EXCLUDE_TERMS):
            filtered_out.append({
                "url_raw": raw_url,
                "decoded_url": decoded_url,
                "reason": "hard_excluded_term",
                "title": title_raw,
                "description": description_raw,
                "country": country,
            })
            continue

        # Store-like domain filtering
        looks_like_store = (
            any(s in domain_wo for s in ["store", "shop"])
            or re.search(
                r"/(product|cart|checkout|category|collection|catalog|basket)/",
                path, re.I
            )
        )

        if not is_amz and looks_like_store and contains_any(combined_clean, SOFT_EXCLUDE_TERMS):
            filtered_out.append({
                "url_raw": raw_url,
                "decoded_url": decoded_url,
                "reason": "soft_excluded_term",
                "title": title_raw,
                "description": description_raw,
                "country": country,
            })
            continue

        # Relevance gate
        if not is_amz and not passes_relevance_gate(
            title_raw,
            description_raw,
            path,
            keyword_matcher
        ):
            filtered_out.append({
                "url_raw": raw_url,
                "decoded_url": decoded_url,
                "reason": "fails_relevance_gate",
                "title": title_raw,
                "description": description_raw,
                "country": country,
            })
            continue

        r["url"] = decoded_url
        final_results.append(r)

    return final_results, filtered_out


# ────────────────────────────────────────────────────────────────────────────────
# Scraping and telemetry helpers
# ────────────────────────────────────────────────────────────────────────────────

def dump_debug_artifacts(scraper, note=""):
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_dir = p("logs", "debug")
        os.makedirs(debug_dir, exist_ok=True)

        driver = getattr(scraper, "driver", None) or getattr(scraper, "browser", None)
        if driver:
            try:
                driver.save_screenshot(os.path.join(debug_dir, f"bing_fail_{ts}.png"))
                log.debug("📸 Saved screenshot.")
            except Exception:
                pass
            try:
                src = driver.page_source or ""
                with open(os.path.join(debug_dir, f"bing_fail_{ts}.html.txt"),
                          "w", encoding="utf-8") as f:
                    f.write(src[:4000])
                log.debug("🧾 Saved HTML snippet.")
            except Exception:
                pass
    except Exception as e:
        log.debug(f"Failed to dump debug artifacts: {e}")

def prewarm_bing_if_possible(scraper, country_code: str):
    try:
        mkt = COUNTRY_SETTINGS.get(country_code, {}).get("mkt", "en-GB")
        driver = getattr(scraper, "driver", None) or getattr(scraper, "browser", None)
        if not driver:
            return

        driver.get(f"https://www.bing.com/?mkt={mkt}&setlang={mkt}")
        time.sleep(1.5)

        try:
            driver.add_cookie({"name": "_EDGE_S", "value": f"mkt={mkt}&ui={mkt}",
                               "domain": ".bing.com", "path": "/"})
            driver.add_cookie({"name": "_SS", "value": "SID=00",
                               "domain": ".bing.com", "path": "/"})
        except Exception:
            pass

    except Exception as e:
        log.debug(f"Prewarm skipped: {e}")

def try_rotate_identity(scraper, country):
    """
    Best-effort identity rotation.
    If the scraper doesn't support it, do nothing.
    """
    try:
        if hasattr(scraper, "rotate_identity") and callable(scraper.rotate_identity):
            scraper.rotate_identity()
            return

        # fallback: full restart (heavy but safe)
        if hasattr(scraper, "quit") and hasattr(scraper, "setup_browser"):
            scraper.quit()
            time.sleep(1.0)
            scraper.setup_browser(country, headless=False)
    except Exception:
        pass



def sleep_with_backoff(base_delay: float, attempt: int, cap: float = 60.0) -> None:
    """Exponential backoff with jitter (caps at `cap`)."""
    try:
        a = max(1, int(attempt))
        delay = min(float(cap), float(base_delay) * (2 ** (a - 1)))
        delay *= random.uniform(0.85, 1.25)  # jitter
        time.sleep(max(0.0, delay))
    except Exception:
        # best-effort sleep only
        try:
            time.sleep(float(base_delay))
        except Exception:
            pass


def scrape_with_retries(
    scraper,
    keyword,
    country,
    max_attempts=4,
    delay=6,
    max_scrape=75,
    min_results_to_accept=10,
    force_plain_query: bool = False,
):
    """
    Fetch up to `max_scrape` raw URLs.

    SUFFIX MODE (force_plain_query=False):
        - Uses "keyword + country" in the query string (we build it ourselves)
        - Allows the scraper to ALSO apply its country clause (site:.uk language:en) for tighter targeting
        - Stops early once we reach min_results_to_accept

    PLAIN MODE (force_plain_query=True):
        - Uses keyword only (no country suffix)
        - Explicitly DISABLES country clause so it is truly plain
        - Stops early once we reach min_results_to_accept
    """

    def _dedupe_by_url(rows):
        seen = set()
        out = []
        for r in rows or []:
            u = (r or {}).get("url")
            if not u:
                continue
            if u in seen:
                continue
            seen.add(u)
            out.append(r)
        return out

    def _run_scrape(add_country_suffix: bool):
        """
        Compatibility wrapper around BingScraper.

        - If add_country_suffix=True:
            query = "keyword COUNTRY" (e.g. "samsung air purifier review UK")
            use_country_clause = True  (adds site:.uk language:en inside bing_scraper)
        - If add_country_suffix=False:
            query = "keyword"
            use_country_clause = False (TRULY plain)
        """
        query = f"{keyword} {country}" if add_country_suffix else keyword
        use_country_clause = True if add_country_suffix else False

        log.info(
            f"QUERY_MODE={'SUFFIX' if add_country_suffix else 'PLAIN'} | "
            f"query={query!r} | target_country={country} | use_country_clause={use_country_clause}"
        )

        if not hasattr(scraper, "scrape_multiple_keywords"):
            raise AttributeError("BingScraper missing scrape_multiple_keywords()")

        # Prefer calling with the new flag if available, otherwise fall back gracefully.
        try:
            return scraper.scrape_multiple_keywords(
                [query],
                target_country=country,
                num_results=max_scrape,
                should_remove=should_remove,
                add_country_suffix=False,       # we already built the query ourselves
                use_country_clause=use_country_clause,
            )
        except TypeError:
            # Older BingScraper without use_country_clause
            return scraper.scrape_multiple_keywords(
                [query],
                target_country=country,
                num_results=max_scrape,
                should_remove=should_remove,
                add_country_suffix=False,
            )

    # ─────────────────────────────────────────────────────────────────────
    # PLAIN MODE
    # ─────────────────────────────────────────────────────────────────────
    if force_plain_query:
        log.info(
            f"✅ Entered PLAIN retry mode | max_attempts={max_attempts} | "
            f"min_results_to_accept={min_results_to_accept}"
        )

        best_results = []
        best_count = 0

        for attempt in range(1, max_attempts + 1):
            log.info(
                f"🔁 [PLAIN] Attempt {attempt}/{max_attempts} | keyword={keyword!r} | "
                f"target_country={country} | target={max_scrape}"
            )

            try:
                results = _run_scrape(add_country_suffix=False) or []
                combined = _dedupe_by_url(results)
                count = len(combined)

                if count > best_count:
                    best_count = count
                    best_results = combined

                if count:
                    log.info(f"📥 Raw results fetched (plain): {count}")
                    if count >= min_results_to_accept:
                        log.info(
                            f"🎯 Target reached in PLAIN mode: {count} >= {min_results_to_accept}. "
                            f"Stopping retries."
                        )
                        return combined[:max_scrape]
                else:
                    log.warning("⚠️ Plain-keyword scrape returned empty/None")

            except Exception as e:
                log.warning(f"⚠️ Plain attempt {attempt} failed: {e}")
                dump_debug_artifacts(scraper, note=f"plain_only_attempt_{attempt}")

            if attempt < max_attempts:
                log.debug(f"⏳ Sleeping with backoff before next attempt.")
                sleep_with_backoff(delay, attempt, cap=60.0)
                try:
                    scraper.rotate_identity()
                except Exception:
                    pass

        return best_results[:max_scrape] if best_results else []

    # ─────────────────────────────────────────────────────────────────────
    # SUFFIX MODE
    # ─────────────────────────────────────────────────────────────────────
    best_results = []
    best_count = 0
    prev_best = None
    plateau_delta = 3

    for attempt in range(1, max_attempts + 1):
        log.info(
            f"🔁 [SUFFIX] Attempt {attempt}/{max_attempts} | keyword={keyword!r} | "
            f"suffix={country} | target={max_scrape}"
        )

        try:
            results = _run_scrape(add_country_suffix=True) or []
            combined = _dedupe_by_url(results)
            count = len(combined)

            log.debug(
                f"[SUFFIX] attempt={attempt} | count={count} | best_count={best_count} | prev_best={prev_best}"
            )

            if count > best_count:
                best_count = count
                best_results = combined

            if count:
                log.info(f"📥 Raw results fetched (suffix): {count}")
                if count >= min_results_to_accept:
                    log.info(
                        f"🎯 Target reached in SUFFIX mode: {count} >= {min_results_to_accept}. "
                        f"Stopping retries."
                    )
                    return combined[:max_scrape]
            else:
                log.warning("⚠️ Scraper returned empty/None")

            # SAFE early-exit: only when we have a real “low ceiling” plateau (not 0–1 result blockage)
            if 1 < best_count < min_results_to_accept:
                plateauing = (prev_best is not None and best_count <= (prev_best + plateau_delta))
                # only after at least 3 attempts
                if attempt >= 3 and plateauing:
                    log.warning(
                        f"🛑 Early-exit: suffix query appears low-ceiling/plateaued "
                        f"(best {best_count}/{min_results_to_accept}, attempt {attempt}). "
                        f"Let orchestrator try plain keyword."
                    )
                    return best_results[:max_scrape]

        except Exception as e:
            log.warning(f"⚠️ Suffix attempt {attempt} failed: {e}")
            dump_debug_artifacts(scraper, note=f"suffix_attempt_{attempt}")

        prev_best = best_count

        if attempt < max_attempts:
            log.debug(f"⏳ Sleeping with backoff before next attempt.")
            sleep_with_backoff(delay, attempt, cap=60.0)
            try:
                scraper.rotate_identity()
            except Exception:
                pass

    return best_results[:max_scrape] if best_results else []



# ────────────────────────────────────────────────────────────────────────────────
# IO helpers
# ────────────────────────────────────────────────────────────────────────────────

def write_results(data, output_path):
    if not data:
        log.warning("⚠️ No data to write.")
        return

    df = pd.DataFrame(data)
    for col in ["url", "title", "description", "country"]:
        if col not in df.columns:
            df[col] = ""

    before = len(df)
    df = df[df["url"].notnull() & (df["url"].astype(str).str.strip() != "")]
    after = len(df)
    log.info(f"Filtered out {before - after} rows with empty URLs.")

    df = df.drop_duplicates(subset=["url", "country"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)

    log.info(f"📂 Saved {len(df)} records to {output_path}")

def deduplicate_excel_file(filepath):
    try:
        df = pd.read_excel(filepath)
        df = df.drop_duplicates(subset=["url", "title"])

        if "title" in df.columns:
            blank_titles = df["title"].astype(str).str.strip() == ""
            df = df[~(df.duplicated(subset=["url"]) & blank_titles)]

        df.to_excel(filepath, index=False)
        log.info(f"Deduplicated and saved cleaned data to {filepath}")

    except Exception as e:
        log.error(f"❌ Error during deduplication: {e}")

def log_filter_summary(raw_results, kept_results, filtered_out,
                       country, keyword, keyword_first_word):

    raw_count = len(raw_results) if isinstance(raw_results, list) else 0
    kept_count = len(kept_results)
    dropped_count = len(filtered_out)

    log.info("── Decision Trace: Filter Summary ──")
    log.info(
        f"🌐 Country={country} | Keyword='{keyword}' | "
        f"first_word='{keyword_first_word}'"
    )
    log.info(f"📥 Raw fetched: {raw_count}  |  ✅ Kept: {kept_count}  |  🗑️ Dropped: {dropped_count}")

    reason_hist = Counter([f.get("reason", "unknown") for f in filtered_out])
    if reason_hist:
        log.info("🧮 Drop reasons: " + ", ".join([f"{r}:{c}" for r, c in reason_hist.most_common(10)]))

    kept_domains = Counter([urlparse(r["url"]).netloc.lower() for r in kept_results])
    if kept_domains:
        log.info("🏷️ Top kept domains: " + ", ".join([f"{d}({c})" for d, c in kept_domains.most_common(10)]))

# ────────────────────────────────────────────────────────────────────────────────
# Safe shutdown
# ────────────────────────────────────────────────────────────────────────────────

def safe_quit_scraper(scraper):
    try:
        if scraper is not None and hasattr(scraper, "quit"):
            try:
                scraper.quit()
            except OSError as e:
                if getattr(e, "winerror", None) == 6:
                    log.debug("Ignoring WinError 6 during driver quit.")
                else:
                    raise
            except Exception as e:
                log.debug(f"Ignoring driver quit error: {e}")

        for attr in ("driver", "browser"):
            if hasattr(scraper, attr):
                try:
                    setattr(scraper, attr, None)
                except Exception:
                    pass

    finally:
        try:
            gc.collect()
        except Exception:
            pass

# ────────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ────────────────────────────────────────────────────────────────────────────────

def main():
    start_ts = time.time()

    parser = argparse.ArgumentParser(description="Scrape URLs with robust logging.")
    parser.add_argument("--force", action="store_true", help="Ignore existing results and scrape anyway.")
    parser.add_argument("--verbose", action="store_true", help="Verbose console logging.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if insufficient results.")
    parser.add_argument("--keyword", type=str, help="Override keyword (otherwise read from current_keyword.csv).")
    parser.add_argument("--country", type=str, choices=["US","UK","CA"], help="Override country.")
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--delay", type=int, default=6)
    parser.add_argument("--max-scrape", type=int, help="Override max_results_to_scrape (URLs to fetch from Bing).")
    args = parser.parse_args()
    
    force_plain = os.environ.get("ORCH_FORCE_PLAIN_QUERY", "").strip().lower() in {"1", "true", "yes"}


    # Temporary bootstrap logging
    setup_logging(p("logs"), "bootstrap", args.verbose)

    # Load general config
    config = load_config()
    max_save = config["max_results_to_save"]
    max_scrape = args.max_scrape if args.max_scrape is not None else config["max_results_to_scrape"]
    min_required_to_proceed = max(6, int(config.get("min_required_urls_to_proceed", 6)))
    min_required_success = max(min_required_to_proceed, int(config.get("min_required_urls_to_consider_success", 10)))
    site_names_to_strip = set(n.lower() for n in config.get("site_names_to_strip", []))

    # Resolve keyword/country/category
    kw = args.keyword if args.keyword else None
    country = args.country if args.country else None
    site = None
    category = None

    if not kw or not country:
        file_kw, file_country, file_site, file_category = read_current_keyword()
        kw = kw or file_kw
        country = country or file_country
        site = file_site
        category = file_category

    # ✅ Normalize/strip to avoid whitespace bugs
    kw = (kw or "").strip()
    country = (country or "").strip().upper()
    category = (category or "").strip().lower()

    if not kw or not country or not category:
        status = "FATAL_CONFIG"
        log.error("❌ Missing keyword, country, or category; cannot proceed.")
        summarize_and_exit(status, start_ts, 2 if args.strict else 0)

    # Re-init logging with keyword
    safe_keyword_country = f"{re.sub(r'\\s+', '_', kw).strip()}_{country}"
    setup_logging(p("logs"), safe_keyword_country, args.verbose)



    # Load category config AFTER logging is reinitialised
    category_cfg = load_category_config(category)

    global REVIEW_ITEM_TERMS, REQUIRE_KEYWORD_FIRST_WORD

    loaded_terms = set(category_cfg.get("include_keywords", []))
    if loaded_terms:
        REVIEW_ITEM_TERMS = loaded_terms

    # NEW: allow category to relax the keyword-first-word requirement
    REQUIRE_KEYWORD_FIRST_WORD = category_cfg.get("require_keyword_first_word", True)

    log.info(f"Category detected: '{category}'")
    log.info(f"REVIEW_ITEM_TERMS loaded from category config: {sorted(REVIEW_ITEM_TERMS)}")
    log.info(f"REQUIRE_KEYWORD_FIRST_WORD = {REQUIRE_KEYWORD_FIRST_WORD}")



    # Build output folder name
    sanitized_kw = sanitize_keyword_for_folder(kw, site_names_to_strip=site_names_to_strip)
    if not sanitized_kw:
        log.warning("Keyword sanitized to empty; falling back to raw keyword.")
        sanitized_kw = kw

    safe_keyword_sanitized = re.sub(r"\s+", "_", sanitized_kw.strip())
    output_folder_name = f"{safe_keyword_sanitized}_{country}"

    log.info(
        f"▶️ Starting scrape | keyword='{kw}' (sanitized='{sanitized_kw}') | country={country}"
    )
    log.info(
        f"Config: max_save={max_save}, max_scrape={max_scrape}, "
        f"min_score_to_allow_multiple_per_domain={config['min_score_to_allow_multiple_per_domain']}, "
        f"min_required_to_proceed={min_required_to_proceed}, "
        f"min_required_success={min_required_success}"
    )

    # Paths
    output_dir = p("output", output_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    final_output_file = p("output", output_folder_name, f"input_urls_{country}.xlsx")
    raw_file = p("output", output_folder_name, f"input_urls_{country}_raw.xlsx")
    filtered_out_path = p("output", output_folder_name, f"filtered_out_urls_{country}.xlsx")
    proceed_flag = p("output", output_folder_name, "proceed_with_post.txt")

    # Check existing results
    skipped_due_to_existing = False
    if os.path.exists(final_output_file) and not args.force:
        try:
            existing_df = pd.read_excel(final_output_file)

            if not existing_df.empty and "url" in existing_df.columns and existing_df["url"].notna().any():
                log.info("📄 Existing results found — skipping scrape (use --force to override).")
                skipped_due_to_existing = True

                try:
                    with open(proceed_flag, "w", encoding="utf-8") as f:
                        f.write("YES")
                    log.info("🟩 proceed_with_post.txt written (YES).")
                except Exception as e:
                    log.warning(f"⚠️ Could not write proceed flag: {e}")

        except Exception as e:
            log.warning(f"⚠️ Could not read existing results: {e}")

    raw_results = []
    filtered_results = []
    filtered_out = []
    amazon = []
    non_amazon = []
    combined = []

    bing_scraper = None

    try:
        if not skipped_due_to_existing:
            bing_scraper = BingScraper(COUNTRY_SETTINGS)
            bing_scraper.setup_browser(country)

            prewarm_bing_if_possible(bing_scraper, country)

            # Decide mode ONLY from env flag (orchestrator controls this)
            force_plain = os.environ.get("ORCH_FORCE_PLAIN_QUERY", "").strip().lower() in {"1", "true", "yes"}

            # We always "aim" for max_scrape raw URLs (e.g. 75) in both modes.
            # Acceptance floors are applied later in the sufficiency check.
            PLAIN_MIN_RAW_SUCCESS = 20  # used later, keep here for clarity
            min_accept = max_scrape     # target raw URLs to try to fetch

            # Ensure plain fallback always runs 4 attempts (even if CLI passed something else)
            effective_max_attempts = 4 if force_plain else args.max_attempts

            log.info(
                f"🔧 ORCH_FORCE_PLAIN_QUERY={os.environ.get('ORCH_FORCE_PLAIN_QUERY','')} "
                f"| force_plain={force_plain} | args.max_attempts={args.max_attempts} "
                f"| effective_max_attempts={effective_max_attempts} | max_scrape={max_scrape} "
                f"| min_accept={min_accept}"
            )

            log.info(
                f"🔧 scrape_with_retries settings | force_plain={force_plain} "
                f"| max_attempts={effective_max_attempts} | delay={args.delay} "
                f"| max_scrape={max_scrape} | min_results_to_accept={min_accept}"
            )

            raw_results = scrape_with_retries(
                bing_scraper, kw, country,
                max_attempts=effective_max_attempts,
                delay=args.delay,
                max_scrape=max_scrape,
                min_results_to_accept=min_accept,
                force_plain_query=force_plain,
            )


            if not isinstance(raw_results, list):
                log.error("🛑 scrape_with_retries returned invalid format — forcing empty list.")
                raw_results = []

            keyword_matcher = build_keyword_matcher(kw)
            filtered_results, filtered_out = filter_results(raw_results, country, keyword_matcher)

            log_filter_summary(raw_results, filtered_results, filtered_out,
                               country, kw, kw.lower().split()[0])

            # Score & sort
            for r in filtered_results:
                url = r["url"]
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
                path = parsed.path.lower()

                title = str(r.get("title", ""))
                description = str(r.get("description", ""))
                title_clean = clean_text(title)
                description_clean = clean_text(description)

                score = 5

                if contains_any_word(title_clean, TITLE_PRIORITY_TERMS):
                    score += 3
                elif contains_any_word(description_clean, TITLE_PRIORITY_TERMS):
                    score += 2

                if contains_any_word(title_clean, REVIEW_ITEM_TERMS):
                    score += 2

                if any(d in domain for d in ["blogspot", "wordpress", "medium", "substack", "tumblr"]):
                    score += 2

                if "blog" in domain or "blog" in path or "review" in path:
                    score += 1

                r["match_score"] = score

            filtered_results.sort(key=lambda x: x.get("match_score", 0), reverse=True)

            # Save diagnostics
            write_results(raw_results, raw_file)
            pd.DataFrame(filtered_out).to_excel(filtered_out_path, index=False)

            log.info(f"🧾 Saved filtered-out diagnostics → {filtered_out_path}")

            # Select final results
            amazon = [
                r for r in filtered_results if AMAZON_PRODUCT_RE.match(r["url"])
            ]
            non_amazon = [
                r for r in filtered_results if r["url"] not in {a["url"] for a in amazon}
            ]

            log.info(
                f"📌 Candidates: total={len(filtered_results)} | "
                f"amazon={len(amazon)} | non_amazon={len(non_amazon)}"
            )

            combined = amazon[:6]
            seen = set(r["url"] for r in combined)
            domain_seen = defaultdict(int)

            for r in combined:
                domain_seen[urlparse(r["url"]).netloc.lower()] += 1

            threshold = config["min_score_to_allow_multiple_per_domain"]

            for r in non_amazon:
                if len(combined) >= max_save:
                    break

                domain = urlparse(r["url"]).netloc.lower()

                if domain_seen[domain] >= 1 and r.get("match_score", 0) < threshold:
                    log.debug(
                        f"↩️ Skip {r['url']} — domain cap reached (score {r.get('match_score',0)}<{threshold})"
                    )
                    continue

                if r["url"] in seen:
                    continue

                combined.append(r)
                seen.add(r["url"])
                domain_seen[domain] += 1

 
            # ---- Final counts (compute before deciding whether to write final file) ----
            final_count = len(combined)
            raw_fetched = len(raw_results) if isinstance(raw_results, list) else 0

            # Proceed/write decisions are based on final usable URLs,
            # i.e. the rows that would actually be written to input_urls_{country}.xlsx
            met_final_sufficient = final_count >= min_required_to_proceed

            log.info("── Sufficiency Check ──")
            log.info(
                f"Raw fetched: {raw_fetched} | "
                f"Final kept: {final_count} "
                f"(proceed≥{min_required_to_proceed}, success≥{min_required_success})"
            )

            # ✅ Write the FINAL input_urls_{country}.xlsx when enough final usable URLs exist.
            if met_final_sufficient:
                write_results(combined[:14], final_output_file)
                deduplicate_excel_file(final_output_file)
                log.info(f"✅ Wrote FINAL results → {final_output_file}")
            else:
                log.warning(
                    f"🟨 Skipping FINAL file write ({Path(final_output_file).name}) because "
                    f"final_count={final_count} (need ≥{min_required_to_proceed}). "
                    f"Raw/diagnostics were still saved."
                )

            # ✅ Proceed flag: allow pipeline to continue when enough final usable URLs exist.
            if met_final_sufficient:
                try:
                    with open(proceed_flag, "w", encoding="utf-8") as f:
                        f.write("YES")
                    log.info("🟩 proceed_with_post.txt written (YES).")
                except Exception as e:
                    log.warning(f"⚠️ Could not write proceed flag: {e}")
            else:
                log.warning(
                    f"❌ Not enough final usable URLs ({final_count} < {min_required_to_proceed}); "
                    f"proceed flag not written."
                )



    except Exception as e:
        log.exception(f"Unhandled error in main flow: {e}")

    finally:
        safe_quit_scraper(bing_scraper)

    # Exit status
    raw_fetched = len(raw_results) if isinstance(raw_results, list) else 0

    # Safety fallback in case an exception happened before sufficiency vars were set
    try:
        met_final_sufficient
    except NameError:
        met_final_sufficient = len(combined) >= min_required_to_proceed

    status = (
        "SKIPPED_EXISTING" if skipped_due_to_existing else
        "SUCCESS" if met_final_sufficient and len(combined) >= min_required_success else
        "MARGINAL" if met_final_sufficient else
        "INSUFFICIENT" if filtered_results or raw_results else
        "NO_RESULTS"
    )

    exit_code = (
        0 if status in {"SUCCESS", "MARGINAL", "SKIPPED_EXISTING"}
        else (1 if args.strict else 0)
    )

    summarize_and_exit(status, start_ts, exit_code)

def summarize_and_exit(status: str, start_ts: float, exit_code: int):
    elapsed = time.time() - start_ts
    log.info(f"🏁 Run finished with status: {status} | elapsed={elapsed:.1f}s")

    for h in log.handlers + filtered_log.handlers:
        try:
            h.flush()
        except Exception:
            pass

    sys.exit(exit_code)

# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
