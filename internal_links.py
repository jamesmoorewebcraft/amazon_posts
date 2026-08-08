# internal_links.py

from __future__ import annotations

import os
import re
import json
import hashlib
import logging
from pathlib import Path
from html import escape
from datetime import datetime, timezone

from openai import OpenAI as DeepSeekClient
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

# === DeepSeek / internal-linking configuration ===
DEEPSEEK_API_KEY_FILE = Path("config/deepseek_api_key.txt")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_PRO_MODEL = "deepseek-v4-pro"
DEEPSEEK_FLASH_MODEL = "deepseek-v4-flash"
# Flash is the default for routine generation; quality-critical calls opt into Pro.
DEEPSEEK_MODEL = DEEPSEEK_FLASH_MODEL

# USD per 1M tokens, from DeepSeek's pricing page on 2026-08-01.
# Update these values if DeepSeek changes its published prices.
DEEPSEEK_PRICING_USD_PER_MILLION = {
    "deepseek-v4-pro": {
        "cache_hit_input": 0.003625,
        "cache_miss_input": 0.435,
        "output": 0.87,
    },
    "deepseek-v4-flash": {
        "cache_hit_input": 0.0028,
        "cache_miss_input": 0.14,
        "output": 0.28,
    },
}
DEEPSEEK_USAGE_LOG_FILE = Path("logs/deepseek_usage.jsonl")
_deepseek_usage_totals = {"requests": 0, "estimated_cost_usd": 0.0}

# DeepSeek does not expose an embeddings endpoint. Internal-link vectors are
# generated locally so this project has no remaining dependency on OpenAI's API.
SITE_INDEX_FILE = Path("output/site_index.json")  # global index of existing posts
LOCAL_EMBEDDING_DIM = 768

# sensible caps
INTERNAL_LINK_MAX_TOTAL = 3          # overall per post
INTERNAL_LINK_SIM_THRESHOLD = 0.50   # cosine similarity cut-off (raised from 0.10)
INTERNAL_RELATED_FOOTER_MAX = 4      # (kept for future use)

# ------------------------------------------------------------
# DeepSeek client helper (via DeepSeek's OpenAI-compatible endpoint)
# ------------------------------------------------------------
_client: DeepSeekClient | None = None


def _get_deepseek_client() -> DeepSeekClient:
    """Lazily create and reuse a single DeepSeek client."""
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key and DEEPSEEK_API_KEY_FILE.exists():
        api_key = DEEPSEEK_API_KEY_FILE.read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError(
            "DeepSeek client not initialised. Set DEEPSEEK_API_KEY or create "
            "config/deepseek_api_key.txt."
        )

    _client = DeepSeekClient(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    return _client


def _usage_field(usage, name: str):
    """Read an SDK usage field from either an object or dictionary."""
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage.get(name)
    return getattr(usage, name, None)


def _log_deepseek_usage_impl(response, label: str, requested_model: str | None = None) -> dict:
    """Build and write one DeepSeek token-usage and cost record."""
    usage = getattr(response, "usage", None)
    model = str(getattr(response, "model", None) or requested_model or "unknown")
    pricing_model = model.split("[", 1)[0]

    prompt_tokens = _usage_field(usage, "prompt_tokens")
    cache_hit_tokens = _usage_field(usage, "prompt_cache_hit_tokens")
    cache_miss_tokens = _usage_field(usage, "prompt_cache_miss_tokens")
    output_tokens = _usage_field(usage, "completion_tokens")

    prompt_tokens = int(prompt_tokens or 0)
    cache_hit_tokens = int(cache_hit_tokens or 0)
    if cache_miss_tokens is None:
        cache_miss_tokens = max(prompt_tokens - cache_hit_tokens, 0)
    cache_miss_tokens = int(cache_miss_tokens or 0)
    output_tokens = int(output_tokens or 0)

    rates = DEEPSEEK_PRICING_USD_PER_MILLION.get(pricing_model)
    estimated_cost_usd = None
    if rates:
        estimated_cost_usd = (
            cache_hit_tokens * rates["cache_hit_input"]
            + cache_miss_tokens * rates["cache_miss_input"]
            + output_tokens * rates["output"]
        ) / 1_000_000

    _deepseek_usage_totals["requests"] += 1
    if estimated_cost_usd is not None:
        _deepseek_usage_totals["estimated_cost_usd"] += estimated_cost_usd

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "process_id": os.getpid(),
        "label": str(label or "unlabelled"),
        "model": model,
        "prompt_tokens": prompt_tokens,
        "cache_hit_tokens": cache_hit_tokens,
        "cache_miss_tokens": cache_miss_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": (
            round(estimated_cost_usd, 10) if estimated_cost_usd is not None else None
        ),
        "process_request_number": _deepseek_usage_totals["requests"],
        "process_estimated_cost_usd": round(
            _deepseek_usage_totals["estimated_cost_usd"], 10
        ),
    }

    log.info(
        "[DEEPSEEK_USAGE] label=%s | model=%s | cache_hit_tokens=%d | "
        "cache_miss_tokens=%d | output_tokens=%d | estimated_cost_usd=%s | "
        "process_requests=%d | process_cost_usd=%.10f",
        record["label"],
        model,
        cache_hit_tokens,
        cache_miss_tokens,
        output_tokens,
        (
            f"{estimated_cost_usd:.10f}"
            if estimated_cost_usd is not None
            else "unknown_model_price"
        ),
        record["process_request_number"],
        record["process_estimated_cost_usd"],
    )

    try:
        DEEPSEEK_USAGE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with DEEPSEEK_USAGE_LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        log.warning("Could not append DeepSeek usage record: %s", exc)

    return record


def log_deepseek_usage(response, label: str, requested_model: str | None = None) -> dict:
    """Log one response without ever disrupting the generation pipeline."""
    try:
        return _log_deepseek_usage_impl(response, label, requested_model)
    except Exception as exc:
        try:
            log.warning("DeepSeek usage accounting failed for %s: %s", label, exc)
        except Exception:
            pass
        return {
            "label": str(label or "unlabelled"),
            "model": str(requested_model or "unknown"),
            "estimated_cost_usd": None,
            "logging_error": str(exc),
        }


# ------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------
def _slugify(text: str) -> str:
    txt = re.sub(r"<[^>]+>", "", text).strip().lower()
    txt = re.sub(r"^\d+(\.\d+)?\s+", "", txt)  # drop "1.0 " prefix if present
    txt = re.sub(r"[^\w\s-]", "", txt)
    txt = re.sub(r"\s+", "-", txt)
    return txt[:80].strip("-") or "section"


def _first_n_paragraph_spans(html: str, n: int = 2):
    """Return a list of (start, end) indices for the first N <p>...</p> occurrences."""
    spans = []
    for m in re.finditer(r"(?is)<p\b[^>]*>[\s\S]*?</p>", html or ""):
        spans.append(m.span())
        if len(spans) >= n:
            break
    return spans


def normalize_ws(s: str) -> str:
    return " ".join((s or "").split())
    
# Add near your other helpers
def _truncate_on_word(s: str, max_chars: int = 80) -> str:
    s = normalize_ws(s or "")
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars].rstrip()
    # back up to last space to avoid "Tra"
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    # if we cut too aggressively (single long word), fall back to hard cut
    if not cut:
        cut = s[:max_chars].rstrip()
    return cut + "…"



# ------------------------------------------------------------
# Lightweight keyword extractor (fixes missing _extract_keywords_from_html)
# ------------------------------------------------------------
def _extract_keywords_from_html(html: str, k: int = 10) -> list[str]:
    """
    Extract top keywords from an HTML fragment.

    This replaces the missing helper that caused:
      name '_extract_keywords_from_html' is not defined
    """
    text = BeautifulSoup(html or "", "html.parser").get_text(" ")
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())

    stop = {
        "the", "a", "an", "and", "or", "for", "to", "in", "on", "with", "by",
        "of", "is", "are", "was", "were", "be", "being", "been",
        "it", "its", "this", "that", "these", "those",
        "as", "at", "from", "into", "over", "under", "up", "down",
        "you", "your", "our", "their", "there", "here",
    }

    counts: dict[str, int] = {}
    for w in words:
        if w in stop:
            continue
        counts[w] = counts.get(w, 0) + 1

    return [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


# ------------------------------------------------------------
# Slot insertion (generator side & fallback)
# ------------------------------------------------------------
def insert_internal_link_slots(html: str) -> str:
    """
    Injects HTML comments marking where internal links should go.
    - <!-- INTERNAL_LINK_SLOT:intro -->
    - <!-- INTERNAL_LINK_SLOT:after-h2:<slug> -->
    """
    if not html:
        return html or ""

    # 1) Intro slot (after first or second paragraph overall)
    intro_inserted = False
    spans = _first_n_paragraph_spans(html, n=2)
    if spans:
        # Prefer after the second paragraph if present & substantial
        idx = -1
        if len(spans) >= 2:
            p_html = html[spans[1][0]:spans[1][1]]
            if len(re.sub(r"<[^>]+>", "", p_html).strip()) >= 120:
                idx = spans[1][1]
        if idx == -1:
            idx = spans[0][1]
        html = html[:idx] + "\n<!-- INTERNAL_LINK_SLOT:intro -->\n" + html[idx:]
        intro_inserted = True  # noqa: F841

    # 2) After each H2: add one slot after the first substantial paragraph in that section,
    #    BUT skip the first main H2 section and skip the verdict section.
    try:
        rebuilt: list[str] = []
        cursor = 0
        first_h2 = True

        for m in re.finditer(r'(?is)(<h2\b[^>]*>[\s\S]*?</h2>)', html):
            start, end = m.span()
            rebuilt.append(html[cursor:start])
            h2_block = html[start:end]

            title_text = re.sub(r'(?is)</?h2\b[^>]*>', '', h2_block).strip()
            title_plain = re.sub(r"<[^>]+>", "", title_text).strip()
            title_lc = title_plain.lower()

            is_verdict_section = bool(re.search(r"(value\s*for\s*money|final\s*verdict)", title_lc))
            is_intro_section   = bool(re.search(r"\bwho\s+is\s+this\s+for\??\b", title_lc))

            next_m = re.search(r'(?is)(<h2\b[^>]*>[\s\S]*?</h2>)', html[end:])
            body_end = end + (next_m.start() if next_m else len(html) - end)
            body = html[end:body_end]

            # ✅ skip slot insertion for the first H2 section (and for Who-is-this-for)
            if first_h2 or is_intro_section or is_verdict_section:
                rebuilt.append(h2_block + body)
                cursor = body_end
                first_h2 = False
                continue

            # (existing paragraph selection + slot insertion)
            chosen_end = None
            for p in re.finditer(r'(?is)<p\b[^>]*>[\s\S]*?</p>', body):
                plain = re.sub(r"<[^>]+>", " ", p.group(0)).strip()
                if len(plain) >= 120:
                    chosen_end = p.end()
                    break
            if chosen_end is None:
                m_p = re.search(r'(?is)<p\b[^>]*>[\s\S]*?</p>', body)
                if m_p:
                    chosen_end = m_p.end()

            if chosen_end is not None:
                slug = _slugify(title_plain)
                body = (
                    body[:chosen_end]
                    + f'\n<!-- INTERNAL_LINK_SLOT:after-h2:{slug} -->\n'
                    + body[chosen_end:]
                )

            rebuilt.append(h2_block + body)
            cursor = body_end
            first_h2 = False

        rebuilt.append(html[cursor:])
        html = "".join(rebuilt)
    except Exception:
        pass


    # (no pre-conclusion slot anymore)
    return html


def ensure_internal_link_slots(html: str) -> str:
    """
    If the generator didn’t add INTERNAL_LINK_SLOT comments, insert a minimal set:
    - intro: after the first <p> (or top of doc)
    """
    if not html:
        return ""
    if "<!-- INTERNAL_LINK_SLOT:" in html:
        return html

    m = re.search(r'(?is)<p\b[^>]*>[\s\S]*?</p>', html)
    if m:
        html = html[:m.end()] + "\n<!-- INTERNAL_LINK_SLOT:intro -->\n" + html[m.end():]
    else:
        html = "<!-- INTERNAL_LINK_SLOT:intro -->\n" + html

    # No pre-conclusion fallback anymore
    return html


# ------------------------------------------------------------
# Section split/join (used by some flows; kept generic)
# ------------------------------------------------------------
def _split_h2_sections(html: str):
    parts = re.split(r'(?is)(<h2\b[^>]*>[\s\S]*?</h2>)', html or "")
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
            if current is None:
                current = {"title": "Preamble", "body": part}
            else:
                current["body"] += part
    if current:
        sections.append(current)
    return sections


def _join_h2_sections(sections: list[dict]) -> str:
    out = []
    for sec in sections:
        if sec["title"] != "Preamble":
            out.append(f"<h2>{sec['title']}</h2>")
        out.append(sec["body"])
    return "".join(out)


# ------------------------------------------------------------
# Embeddings + site index
# ------------------------------------------------------------
def _cosine(u, v):
    import math

    if not u or not v or len(u) != len(v):
        return 0.0
    su = sum(x * x for x in u)
    sv = sum(y * y for y in v)
    if su <= 0 or sv <= 0:
        return 0.0
    dot = sum(x * y for x, y in zip(u, v))
    return dot / (su**0.5 * sv**0.5)


def _local_semantic_embed(text: str) -> list[float]:
    """Create a deterministic local feature-hashing vector for link similarity.

    DeepSeek's API currently has no embeddings endpoint. Combining word and
    adjacent-word features retains useful topical matching without requiring a
    second API provider.
    """
    txt = normalize_ws(text)[:8000]
    tokens = re.findall(r"[a-z0-9]+", txt.lower())
    if not tokens:
        return []

    vector = [0.0] * LOCAL_EMBEDDING_DIM
    features = [(token, 1.0) for token in tokens]
    features.extend(
        (f"{left}_{right}", 1.5)
        for left, right in zip(tokens, tokens[1:])
    )
    for feature, weight in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % LOCAL_EMBEDDING_DIM
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign * weight

    magnitude = sum(value * value for value in vector) ** 0.5
    if magnitude:
        vector = [value / magnitude for value in vector]
    log.info("Local semantic vector length for new post: %d", len(vector))
    return vector


def _load_site_index() -> list[dict]:
    if not SITE_INDEX_FILE.exists():
        return []
    try:
        data = json.loads(SITE_INDEX_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        # ensure each entry has minimal required fields
        sanitized = []
        for it in data:
            if isinstance(it, dict) and it.get("url") and it.get("title"):
                sanitized.append(it)
        return sanitized
    except Exception as e:
        log.warning(
            "Failed to read site_index.json",
            extra={"step": "internal_links", "extra_json": {"error": str(e)}},
        )
        return []


def load_site_index() -> list[dict]:
    """Public wrapper used by other scripts."""
    return _load_site_index()


# ------------------------------------------------------------
# UPSERT into site_index (avoid duplicates on updates)
# ------------------------------------------------------------
def upsert_site_index_entry(metadata: dict, embedding: list[float]) -> None:
    """
    Insert or update a site_index entry for this post.

    If an entry exists with the same URL and/or slug, update it instead of
    appending a new one. This prevents duplicate entries when posts are updated.

    Matching rules:
    - Normalized URL equality (ignoring trailing slash)
    - OR slug equality (normalized, without leading/trailing '/')
    """
    site_index = _load_site_index()

    url = (
        metadata.get("url")
        or metadata.get("canonical_url")
        or metadata.get("permalink")
        or ""
    ).strip()
    url_norm = url.rstrip("/")

    slug_meta = (metadata.get("slug") or "").strip().strip("/")
    title = (metadata.get("title") or "").strip()
    slug = slug_meta or _slugify(title)

    existing_idx = None
    for i, it in enumerate(site_index):
        it_url = (it.get("url") or "").strip().rstrip("/")
        it_slug = (it.get("slug") or "").strip().strip("/")

        same_url = bool(url_norm and it_url and url_norm == it_url)
        same_slug = bool(slug and it_slug and slug == it_slug)

        if same_url or same_slug:
            existing_idx = i
            break

    new_entry = {
        "url": url or (site_index[existing_idx].get("url") if existing_idx is not None else ""),
        "slug": slug,
        "title": title,
        "embedding": embedding or [],
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }

    if existing_idx is not None:
        existing = site_index[existing_idx]
        existing.update(new_entry)
        site_index[existing_idx] = existing
        log.info(
            "Updated existing site_index entry at index %d for url=%r slug=%r",
            existing_idx,
            new_entry["url"],
            new_entry["slug"],
        )
    else:
        site_index.append(new_entry)
        log.info(
            "Appended new site_index entry for url=%r slug=%r",
            new_entry["url"],
            new_entry["slug"],
        )

    try:
        SITE_INDEX_FILE.write_text(
            json.dumps(site_index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("site_index.json written successfully (len=%d)", len(site_index))
    except Exception as e:
        log.error(
            "Failed to write site_index.json",
            extra={"step": "internal_links", "extra_json": {"error": str(e)}},
        )


# ------------------------------------------------------------
# Anchor picking + inline link construction
# ------------------------------------------------------------
# strip trailing “ – ” / “ — ” / “ - ” / “ : ” at end of titles
_TITLE_TRAIL_RX = re.compile(r"\s*(?:–|—|-|:)\s*$", flags=re.UNICODE)


def _sanitize_link_title(s: str) -> str:
    return _TITLE_TRAIL_RX.sub("", (s or "").strip())


def _pick_anchor_in_para(
    para_html: str,
    candidates: list[str],
    fallback_title: str,
) -> tuple[str, str]:
    if not para_html:
        return para_html, ""

    # still compute plain text in case we want it later
    _ = BeautifulSoup(para_html, "html.parser").get_text(" ")

    for cand in (candidates or []):
        if not cand or len(cand.split()) > 6:
            continue

        # Build regex for the candidate phrase
        rx = re.compile(rf"(?<!\w){re.escape(cand)}(?!\w)", re.I)

        # Try to find it directly in the HTML (simpler + avoids text/HTML mismatch)
        if rx.search(para_html):
            marked = re.sub(rx, f"__ANCHOR__{cand}__HERE__", para_html, count=1)
            return marked, cand

    # No good anchor found – fall back to using the title as anchor text
    fallback = _sanitize_link_title(fallback_title) if fallback_title else ""
    return para_html, (_truncate_on_word(fallback, 80) if fallback else "")



def _build_inline_link(url: str, title: str) -> str:
    u = escape(url, quote=True)
    t = escape(title or url, quote=True)
    # internal links: no target, no nofollow. Keep it clean.
    return f'<a class="internal" href="{u}" aria-label="{t}">{t}</a>'

def _norm_url(u: str) -> str:
    return (u or "").strip().rstrip("/")


def _last_path_segment(u: str) -> str:
    u = _norm_url(u)
    if not u:
        return ""
    return u.split("/")[-1].strip("/")


def _is_self_candidate(
    candidate: dict,
    *,
    current_url: str,
    current_slug_meta: str,
    current_slug_from_title: str,
    current_title_lc: str,
) -> bool:
    """
    Strong self/duplicate guard that does NOT rely on embeddings.

    Returns True if `candidate` appears to be the current post.
    """
    cand_url = _norm_url(candidate.get("url") or "")
    cand_slug = (candidate.get("slug") or "").strip().strip("/")
    cand_title_lc = (candidate.get("title") or "").strip().lower()

    cur_url = _norm_url(current_url)
    cur_slugs = {s for s in [current_slug_meta, current_slug_from_title] if s}
    cand_last = _last_path_segment(cand_url)

    # URL exact match (handles trailing slashes)
    if cur_url and cand_url and cand_url == cur_url:
        return True

    # Slug equality
    if cand_slug and cand_slug in cur_slugs:
        return True

    # URL ends with slug (works even if candidate slug field missing)
    if cand_url and cur_slugs:
        for s in cur_slugs:
            if cand_url.endswith("/" + s) or cand_last == s:
                return True

    # Title exact match (fallback)
    if current_title_lc and cand_title_lc and cand_title_lc == current_title_lc:
        return True

    return False


def _resolve_slot_in_html(
    html: str,
    slot_key: str,
    queue: list[dict],
    metadata: dict,
    inserted_so_far: int,
    used_sections: set[str],
    current_url: str,              # 👈 NEW
    current_slug_meta: str,        # 👈 NEW
    current_slug_from_title: str,  # 👈 NEW
    current_title_lc: str,         # 👈 NEW
) -> str:
    """
    Resolve a single INTERNAL_LINK_SLOT marker into a single internal link.

    Guarantees:
    - Never inserts more than INTERNAL_LINK_MAX_TOTAL overall (caller tracks inserted_so_far).
    - Inserts at most ONE internal link per section (intro counts as its own section;
      after-h2:<slug> counts per slug).
    - Never appends an internal link into a paragraph that already contains an internal link.
    """
    if f"INTERNAL_LINK_SLOT:{slot_key}" not in html:
        return html
    if inserted_so_far >= INTERNAL_LINK_MAX_TOTAL:
        return html

    # Locate the marker
    m = re.search(rf"<!--\s*INTERNAL_LINK_SLOT:{re.escape(slot_key)}\s*-->", html)
    if not m:
        return html

    # Derive a stable "section id" for the one-link-per-section rule
    section_id = slot_key
    if slot_key.startswith("after-h2:"):
        section_id = slot_key.split("after-h2:", 1)[1].strip() or slot_key
    elif slot_key == "intro":
        section_id = "intro"

    # If this section already got an internal link, remove this marker and skip
    if section_id in used_sections:
        return re.sub(
            rf"<!--\s*INTERNAL_LINK_SLOT:{re.escape(slot_key)}\s*-->\s*",
            "",
            html,
            count=1,
        )

    pos = m.end()
    after = html[pos:]

    # Find the next paragraph after the slot
    p = re.search(r'(?is)<p\b[^>]*>[\s\S]*?</p>', after)
    if not p:
        # No paragraph after the slot – just remove the marker
        return html.replace(m.group(0), "")

    para_html = after[p.start():p.end()]

    # Never stack multiple internal links into the same paragraph
    if 'class="internal"' in para_html:
        return html.replace(m.group(0), "")

    # Pick the next suitable target from the queue
    target = None
    while queue and not target:
        it = queue.pop(0)
        candidate_url = (it.get("url") or "").strip()
        candidate_title = (it.get("title") or "").strip()

        log.debug(
            "Slot %s: checking queue candidate url=%r title=%r",
            slot_key,
            candidate_url,
            candidate_title,
        )

        # ✅ HARD GUARD: never link to the current post (even if ranking let it through)
        if _is_self_candidate(
            it,
            current_url=current_url,
            current_slug_meta=current_slug_meta,
            current_slug_from_title=current_slug_from_title,
            current_title_lc=current_title_lc,
        ):
            log.debug(
                "Slot %s: skipping SELF candidate url=%r slug=%r title=%r",
                slot_key,
                candidate_url,
                (it.get("slug") or ""),
                candidate_title,
            )
            continue

        # Avoid re-using a URL already present in the HTML
        if candidate_url and candidate_url not in html:
            target = it
            log.info(
                "Slot %s: using internal link target url=%r title=%r",
                slot_key,
                candidate_url,
                candidate_title,
            )
        else:
            log.debug(
                "Slot %s: skipping candidate url=%r (missing or already in HTML)",
                slot_key,
                candidate_url,
            )


    if not target:
        # No suitable target – just remove the slot comment
        return html.replace(m.group(0), "")

    # ---- Anchor selection (optional) ----
    para_html_marked, anchor_text = _pick_anchor_in_para(
        para_html,
        metadata.get("anchor_candidates", []),
        target.get("title", ""),
    )
    anchor_text = _sanitize_link_title(anchor_text)

    # We will keep behavior predictable: if we can’t reliably place a natural anchor,
    # we append ONE link at the end of the paragraph.
    full_title = _sanitize_link_title(target.get("title", "")) or "Related article"
    visible_title = _truncate_on_word(full_title, 80)

    u = escape(target["url"], quote=True)
    aria = escape(full_title, quote=True)
    text = escape(visible_title)

    link_html = f' (<a class="internal" href="{u}" aria-label="{aria}">{text}</a>)'


    # Append before closing </p> if present
    if re.search(r"(?is)</p>\s*$", para_html_marked):
        new_para = re.sub(r"(?is)</p>\s*$", link_html + "</p>", para_html_marked, count=1)
    else:
        new_para = para_html_marked + link_html

    # Mark this section as having received its one allowed internal link
    used_sections.add(section_id)

    # Stitch back together
    new_after = after[:p.start()] + new_para + after[p.end():]

    # Remove the marker (we replace by reconstructing around it)
    # Note: marker stays in html[:pos], but html[:pos] already includes marker end.
    # We simply drop it by removing m.group(0) from the prefix slice.
    prefix = html[:m.start()]  # everything before the marker
    return prefix + html[m.end():pos] + new_after




# ------------------------------------------------------------
# Slot replacement (link resolution) API
# ------------------------------------------------------------
def _replace_internal_link_slots(
    html: str,
    metadata: dict,
    outline: list[dict],
    site_index: list[dict],
) -> str:
    """
    Resolve <!-- INTERNAL_LINK_SLOT:* --> markers using site_index similarities.

    Guarantees / Rules:
    - At most INTERNAL_LINK_MAX_TOTAL internal links per post.
    - At most ONE internal link per section:
        - "intro" counts as its own section.
        - each "after-h2:<slug>" counts as a section.
    - Never append an internal link into a paragraph that already contains an internal link.
    - Skip first main H2 section (idx == 0 in outline), skip "who is this for"/purpose-ish, and skip verdict.
    - Never link to the current post itself (URL/slug + near-identical embeddings + slug prefix).
    """
    if not html or not site_index:
        return html

    log.info("Internal linking: site_index length=%d", len(site_index))

    # --- Prepare new post embedding once (robust keyword handling) ---
    kw = metadata.get("keywords") or []
    if isinstance(kw, str):
        kw = [kw]
    elif not isinstance(kw, list):
        kw = [str(kw)]
    kw = [str(t).strip() for t in kw if str(t).strip()]

    new_text_for_embed = " ".join(
        [
            (metadata.get("title") or "").strip(),
            (metadata.get("summary") or "").strip(),
            " ".join(kw),
            normalize_ws(BeautifulSoup(html, "html.parser").get_text(" ")),
        ]
    )
    new_vec = _local_semantic_embed(new_text_for_embed)

    # --- Identity of THIS post (avoid self-links / near-duplicates) ---
    current_title = (metadata.get("title") or "").strip()
    current_title_lc = current_title.lower()
    current_url = (
        metadata.get("url")
        or metadata.get("canonical_url")
        or metadata.get("permalink")
        or ""
    ).strip()
    current_slug_meta = (metadata.get("slug") or "").strip().strip("/")
    current_slug_from_title = _slugify(current_title) if current_title else ""

    log.info(
        "Internal linking current post: title=%r url=%r slug_meta=%r slug_from_title=%r",
        current_title,
        current_url,
        current_slug_meta,
        current_slug_from_title,
    )

    # --- Rank candidates by cosine, excluding self ---
    ranked: list[tuple[float, dict]] = []

    for it in site_index:
        vec = it.get("embedding") or []
        if not vec:
            continue

        url = (it.get("url") or "").strip()
        candidate_slug = (it.get("slug") or "").strip().strip("/")
        candidate_title_lc = (it.get("title") or "").strip().lower()
        sim = _cosine(new_vec, vec)

        log.debug(
            "Internal-link candidate raw: url=%r slug=%r title=%r sim=%.3f",
            url,
            candidate_slug,
            it.get("title"),
            sim,
        )

        # ── Self / near-duplicate detection ───────────────────────────────
        same_url = bool(current_url and url and url.rstrip("/") == current_url.rstrip("/"))

        same_slug_meta = bool(current_slug_meta and candidate_slug and candidate_slug == current_slug_meta)
        same_slug_title = bool(current_slug_from_title and candidate_slug and candidate_slug == current_slug_from_title)

        same_title = bool(current_title_lc and candidate_title_lc and candidate_title_lc == current_title_lc)

        slug_in_url_match = False
        if url:
            u = url.rstrip("/")
            for s in (current_slug_meta, current_slug_from_title):
                if not s:
                    continue
                if u.endswith("/" + s) or u.endswith(s):
                    slug_in_url_match = True
                    break

        candidate_slug_from_url = url.rstrip("/").split("/")[-1] if url else ""
        slug_prefix_match = False
        for s in (current_slug_meta, current_slug_from_title):
            if not s or not candidate_slug_from_url:
                continue
            if candidate_slug_from_url.startswith(s[:18]) or s.startswith(candidate_slug_from_url[:18]):
                slug_prefix_match = True
                break

        almost_identical = sim >= 0.96

        log.debug(
            "Self-check for %r: same_url=%s same_slug_meta=%s same_slug_title=%s "
            "same_title=%s slug_in_url_match=%s slug_prefix_match=%s almost_identical=%s",
            url,
            same_url,
            same_slug_meta,
            same_slug_title,
            same_title,
            slug_in_url_match,
            slug_prefix_match,
            almost_identical,
        )

        if (
            same_url
            or same_slug_meta
            or same_slug_title
            or same_title
            or slug_in_url_match
            or slug_prefix_match
            or almost_identical
        ):
            log.debug(
                "Internal-link: SKIP self/near-duplicate candidate url=%r slug=%r sim=%.3f",
                url,
                candidate_slug,
                sim,
            )
            continue
        # ──────────────────────────────────────────────────────────────────

        if sim >= INTERNAL_LINK_SIM_THRESHOLD:
            ranked.append((sim, it))

    ranked.sort(reverse=True, key=lambda x: x[0])
    queue = [it for _, it in ranked]

    # --- Enforce overall + per-section caps ---
    total_inserted = 0
    used_sections: set[str] = set()  # 👈 one-link-per-section

    # ── Intro slot ───────────────────────────────────────────────────────
    html = _resolve_slot_in_html(
        html, "intro", queue, metadata, total_inserted, used_sections,
        current_url, current_slug_meta, current_slug_from_title, current_title_lc
    )
    total_inserted = html.count('class="internal"')
    if total_inserted >= INTERNAL_LINK_MAX_TOTAL:
        html = re.sub(r"<!--\s*INTERNAL_LINK_SLOT:[^>]*-->\s*", "", html)
        return html

    # ── Per-H2 slots (skip first outline section + intro/purpose + verdict) ──
    for idx, sec in enumerate(outline or []):
        title = (sec.get("title") or "").strip()
        title_lc = title.lower()

        # Skip first outline H2 section
        if idx == 0:
            continue

        # Skip intro/purpose style sections
        if "purpose of the review" in title_lc or "who is this for" in title_lc:
            continue

        # Skip verdict section
        if "value for money" in title_lc or "final verdict" in title_lc:
            continue

        slug = sec.get("slug") or _slugify(title)
        slot_key = f"after-h2:{slug}"

        # Resolve ONE link for this section at most (guarded inside _resolve_slot_in_html)
        html = _resolve_slot_in_html(
            html, slot_key, queue, metadata, total_inserted, used_sections,
            current_url, current_slug_meta, current_slug_from_title, current_title_lc
        )


        total_inserted = html.count('class="internal"')
        if total_inserted >= INTERNAL_LINK_MAX_TOTAL:
            break

    # Remove any leftover slot comments (unfilled slots or skipped sections)
    html = re.sub(r"<!--\s*INTERNAL_LINK_SLOT:[^>]*-->\s*", "", html)
    return html



def replace_internal_link_slots(
    html: str,
    metadata: dict,
    outline: list[dict],
    site_index: list[dict],
) -> str:
    """
    Public API: resolve all INTERNAL_LINK_SLOT comments into inline internal links.
    """
    return _replace_internal_link_slots(html, metadata, outline, site_index)
