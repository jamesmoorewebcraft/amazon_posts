import csv
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import date, datetime

KEYWORDS_FILE = "input/keywords.csv"
CURRENT_KEYWORD_FILE = "config/current_keyword.csv"

SCRIPTS = [
    "scrape_urls.py",
    "generate_headings.py",
    "amend_product_names.py",
    "get_response_from_openai.py",
    "insert_amazon_links_images.py",
    "final_tidy_up.py",
    "title_description.py",
    "upload_wordpress.py",
]

CURRENT_MASTER_LOG = None

RESULTS_CFG = "config/results_settings.json"

# ── URL insufficiency retry tracking ──────────────────────────────────────────
URL_RETRY_STATE_FILE = os.path.join("logs", "url_retry_state.json")

# Max scrape attempts per keyword within the same batch run
MAX_URL_RETRIES = int(os.environ.get("ORCH_MAX_URL_RETRIES", "3"))

# Base sleep between immediate re-attempts within the same batch run
URL_RETRY_SLEEP_BASE_SEC = int(os.environ.get("ORCH_URL_RETRY_SLEEP_BASE_SEC", "20"))

# Optional comma-separated list of scripts to skip, e.g.
# ORCH_SKIP_SCRIPTS=insert_amazon_links_images.py,final_tidy_up.py,title_description.py,upload_wordpress.py
SKIP_SCRIPTS = {
    s.strip() for s in os.environ.get("ORCH_SKIP_SCRIPTS", "").split(",") if s.strip()
}

TOP_PICK_REPORT = os.path.join("output", "top_pick_report.csv")

INSERT_AMAZON_FAILURE_REASON_FILE = "insert_amazon_failure_reason.txt"

DOMAIN_RE = re.compile(r"\b(?:https?://)?(?:www\.)?([a-z0-9-]+(?:\.[a-z0-9-]+)+)\b", re.IGNORECASE)
TLD_RE = re.compile(
    r"\b([a-z0-9-]+)\.(com|co\.uk|co|net|org|io|uk|us|ca|de|fr|it|es|au|nl|se|no|fi|ru|cn|jp|kr)\b",
    re.IGNORECASE,
)

# Make console writes more tolerant on Windows cp1252 consoles
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Retry-state helpers for "Not enough URLs" ────────────────────────────────
def _retry_key(keyword: str, country: str, site: str, category: str) -> str:
    return "|".join([
        (keyword or "").strip(),
        (country or "").strip().upper(),
        (site or "").strip(),
        (category or "").strip(),
    ])


def _load_retry_state() -> dict:
    os.makedirs(os.path.dirname(URL_RETRY_STATE_FILE), exist_ok=True)
    try:
        with open(URL_RETRY_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _save_retry_state(state: dict) -> None:
    os.makedirs(os.path.dirname(URL_RETRY_STATE_FILE), exist_ok=True)
    tmp = URL_RETRY_STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, URL_RETRY_STATE_FILE)


def get_url_retry_count(keyword: str, country: str, site: str, category: str) -> int:
    state = _load_retry_state()
    k = _retry_key(keyword, country, site, category)
    entry = state.get(k) or {}
    try:
        return int(entry.get("count", 0))
    except Exception:
        return 0


def bump_url_retry(keyword: str, country: str, site: str, category: str, reason: str = "Not enough URLs") -> int:
    state = _load_retry_state()
    k = _retry_key(keyword, country, site, category)
    entry = state.get(k) or {}
    n = int(entry.get("count", 0)) + 1
    entry["count"] = n
    entry["last_reason"] = reason
    entry["last_ts"] = datetime.now().isoformat(timespec="seconds")
    state[k] = entry
    _save_retry_state(state)
    return n


def reset_url_retry(keyword: str, country: str, site: str, category: str) -> None:
    state = _load_retry_state()
    k = _retry_key(keyword, country, site, category)
    if k in state:
        state.pop(k, None)
        _save_retry_state(state)


# ── Config / output helpers ───────────────────────────────────────────────────
def load_results_config():
    defaults = {
        "site_names_to_strip": [
            "amazon", "ebay", "youtube", "walmart", "bestbuy", "target", "argos",
            "currys", "newegg", "aliexpress", "alibaba", "johnlewis"
        ]
    }
    try:
        with open(RESULTS_CFG, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            if isinstance(cfg, dict):
                defaults.update(cfg)
    except Exception:
        pass
    return defaults


def sanitize_keyword_for_folder(keyword: str, site_names_to_strip=None) -> str:
    if not isinstance(keyword, str) or not keyword.strip():
        return ""

    site_names_to_strip = set(site_names_to_strip or [])
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


def make_output_dir(keyword: str, country: str) -> str:
    cfg = load_results_config()
    sanitized = sanitize_keyword_for_folder(keyword, cfg.get("site_names_to_strip", [])) or keyword
    safe_keyword_sanitized = re.sub(r"\s+", "_", sanitized.strip())
    output_folder_name = f"{safe_keyword_sanitized}_{country}"
    return os.path.join("output", output_folder_name)


# ── CSV helpers ────────────────────────────────────────────────────────────────
def read_keywords():
    """
    Read keywords from CSV.

    Supported formats:
      - keyword,country,site,category
      - keyword,country,site
      - keyword,country

    Always returns list of 4-tuples:
      (keyword, country, site, category)
    """
    keywords = []
    try:
        with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            _header = next(reader, None)

            for row in reader:
                if not row:
                    continue

                if len(row) >= 4:
                    kw = row[0].strip()
                    country = row[1].strip().upper()
                    site = row[2].strip()
                    category = row[3].strip()
                    keywords.append((kw, country, site, category))

                elif len(row) >= 3:
                    kw = row[0].strip()
                    country = row[1].strip().upper()
                    site = row[2].strip()
                    category = ""
                    keywords.append((kw, country, site, category))

                elif len(row) >= 2:
                    kw = row[0].strip()
                    country = row[1].strip().upper()
                    site = "https://luggageadvisor.uk"
                    category = ""
                    keywords.append((kw, country, site, category))

    except Exception as e:
        print(f"[ERROR] Could not read {KEYWORDS_FILE}: {e}")

    return keywords


def write_current_keyword(keyword, country, site, category):
    try:
        os.makedirs(os.path.dirname(CURRENT_KEYWORD_FILE), exist_ok=True)
        with open(CURRENT_KEYWORD_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([keyword, country, site, category])
    except Exception as e:
        print(f"[ERROR] Failed to write current keyword: {e}")


def remove_keyword_from_csv_atomic(target_keyword, target_country, target_site, target_category):
    """
    Remove a processed keyword row from keywords.csv.

    - For 4-column rows, match on keyword, country, site, category.
    - For 3-column rows, match on keyword, country, site (ignoring category).
    - For 2-column rows, match on keyword, country.
    """
    try:
        with open(KEYWORDS_FILE, "r", encoding="utf-8", newline="") as f_in:
            reader = csv.reader(f_in)
            header = next(reader, None)

            remaining = []
            for row in reader:
                if len(row) >= 4:
                    kw = row[0].strip()
                    co = row[1].strip().upper()
                    si = row[2].strip()
                    cat = row[3].strip()
                    if kw == target_keyword and co == target_country and si == target_site and cat == target_category:
                        continue
                    remaining.append(row)

                elif len(row) >= 3:
                    kw = row[0].strip()
                    co = row[1].strip().upper()
                    si = row[2].strip()
                    if kw == target_keyword and co == target_country and si == target_site and not target_category:
                        continue
                    remaining.append(row)

                elif len(row) >= 2:
                    kw = row[0].strip()
                    co = row[1].strip().upper()
                    if kw == target_keyword and co == target_country:
                        continue
                    remaining.append(row)

        csv_dir = os.path.dirname(KEYWORDS_FILE) or "."
        fd, tmp_path = tempfile.mkstemp(prefix="kw_", suffix=".csv", dir=csv_dir, text=True)
        os.close(fd)

        with open(tmp_path, "w", encoding="utf-8", newline="") as f_out:
            w = csv.writer(f_out)
            if header:
                w.writerow(header)
            w.writerows(remaining)

        os.replace(tmp_path, KEYWORDS_FILE)
        log(f"🗑️ Removed keyword: {target_keyword} ({target_country}) [{target_site}] [{target_category}]")
        return True

    except Exception as e:
        log(f"[ERROR] Failed to remove keyword from CSV: {e}")
        return False


# ── Logging / reports ─────────────────────────────────────────────────────────
def run_script(script, timeout_sec=900, log_dir=None, env_overrides=None):
    os.makedirs(log_dir or "logs", exist_ok=True)

    script_file = shlex.split(script)[0]
    script_name_for_file = os.path.splitext(os.path.basename(script_file))[0]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    per_script_log = os.path.join(log_dir or "logs", f"{script_name_for_file}_{timestamp}.log")

    log(f"➡️ Running {script}...")
    t0 = time.monotonic()
    cmd = [sys.executable, "-u", *shlex.split(script)]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})

    with open(per_script_log, "w", encoding="utf-8") as lf:
        lf.write(f"[{datetime.now().isoformat()}] CMD: {' '.join(cmd)}\n")
        lf.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        try:
            for line in proc.stdout:
                try:
                    print(line, end="")
                except UnicodeEncodeError:
                    sys.stdout.write(
                        line.encode(sys.stdout.encoding or "utf-8", "replace").decode(
                            sys.stdout.encoding or "utf-8"
                        )
                    )

                lf.write(line)

                global CURRENT_MASTER_LOG
                if CURRENT_MASTER_LOG:
                    try:
                        with open(CURRENT_MASTER_LOG, "a", encoding="utf-8") as mf:
                            mf.write(line)
                    except Exception:
                        pass

            rc = proc.wait(timeout=timeout_sec)

        except subprocess.TimeoutExpired:
            proc.kill()
            msg = f"❌ {script} timed out after {timeout_sec}s"
            log(msg)
            lf.write(msg + "\n")
            return False

        finally:
            if proc.stdout:
                proc.stdout.close()

    dt = time.monotonic() - t0
    if rc == 0:
        log(f"✅ {script} completed in {dt:.1f}s\n")
        return True
    else:
        log(f"❌ {script} failed with return code {rc} after {dt:.1f}s\n")
        return False


def log_skipped_keyword(keyword, country, site, category, reason):
    log_file = os.path.join("logs", "skipped_keywords.csv")
    os.makedirs("logs", exist_ok=True)
    is_new = not os.path.exists(log_file)

    try:
        with open(log_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["Keyword", "Country", "Site", "Category", "Reason"])
            writer.writerow([keyword, country, site, category, reason])
    except Exception as e:
        print(f"[ERROR] Could not log skipped keyword: {e}")
        
def read_insert_amazon_failure_reason(output_dir: str) -> str:
    """
    Read the specific failure reason emitted by insert_amazon_links_images.py, if present.
    """
    try:
        path = os.path.join(output_dir, INSERT_AMAZON_FAILURE_REASON_FILE)
        if os.path.exists(path):
            return open(path, "r", encoding="utf-8").read().strip()
    except Exception:
        pass
    return ""


def upsert_top_pick_report_row(
    keyword,
    country,
    site,
    category,
    *,
    processed_date=None,
    status="",
    selected_top_pick="",
    reason="",
    report_path=TOP_PICK_REPORT,
):
    """
    Upsert one row per keyword/country/site/category into output/top_pick_report.csv.
    """
    report_dir = os.path.dirname(report_path) or "."
    os.makedirs(report_dir, exist_ok=True)

    processed_date = (processed_date or date.today().isoformat()).strip()
    keyword = (keyword or "").strip()
    country = (country or "").strip().upper()
    site = (site or "").strip()
    category = (category or "").strip()
    status = (status or "").strip()
    selected_top_pick = (selected_top_pick or "").strip()
    reason = (reason or "").strip()

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
                            "status": status or normalized.get("status", ""),
                            "selected_top_pick": (
                                selected_top_pick if selected_top_pick
                                else normalized.get("selected_top_pick", "")
                            ),
                            "reason": reason,
                        })
                        found = True

                    rows.append(normalized)

        except Exception as e:
            log(f"[WARN] Could not read existing top-pick report '{report_path}': {e}")
            rows = []

    if not found:
        rows.append({
            "processed_date": processed_date,
            "keyword": keyword,
            "country": country,
            "site": site,
            "category": category,
            "status": status,
            "selected_top_pick": selected_top_pick,
            "reason": reason,
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

    return report_path


# ── Main workflow helpers ─────────────────────────────────────────────────────
def _prepare_context(keyword, country, site, category):
    """
    Compute output_dir/log_dir/master log + env for a keyword
    and write current_keyword.csv.
    """
    global CURRENT_MASTER_LOG

    write_current_keyword(keyword, country, site, category)

    output_dir = make_output_dir(keyword, country)
    log_dir = os.path.join("logs", f"{os.path.basename(output_dir)}")
    os.makedirs(log_dir, exist_ok=True)

    master_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    CURRENT_MASTER_LOG = os.path.join(log_dir, f"master_{master_stamp}.log")

    try:
        with open(CURRENT_MASTER_LOG, "a", encoding="utf-8") as mf:
            mf.write(
                f"[{datetime.now().isoformat()}] START "
                f"keyword='{keyword}' country='{country}' site='{site}' category='{category}'\n"
            )
            mf.write(
                f"[{datetime.now().isoformat()}] output_dir='{output_dir}' log_dir='{log_dir}'\n"
            )
    except Exception:
        pass

    env_common = {
        "CURRENT_KEYWORD_FILE": CURRENT_KEYWORD_FILE,
        "ORCH_KEYWORD": keyword,
        "ORCH_COUNTRY": country,
        "ORCH_SITE": site,
        "ORCH_CATEGORY": category,
        "ORCH_OUTPUT_DIR": output_dir,
    }

    flag_file = os.path.join(output_dir, "proceed_with_post.txt")
    no_product_flag = os.path.join(output_dir, "no_products.txt")
    abandon_file = os.path.join(output_dir, "abandon_post.txt")

    return output_dir, log_dir, env_common, flag_file, no_product_flag, abandon_file


def _close_master(note: str = ""):
    global CURRENT_MASTER_LOG
    try:
        if CURRENT_MASTER_LOG:
            with open(CURRENT_MASTER_LOG, "a", encoding="utf-8") as mf:
                mf.write(f"[{datetime.now().isoformat()}] END {note}\n")
    except Exception:
        pass


def _run_pipeline_after_scrape(
    keyword,
    country,
    site,
    category,
    output_dir,
    log_dir,
    env_common,
    no_product_flag,
    abandon_file,
):
    """
    Run the remaining scripts after scrape_urls.py has succeeded
    (proceed_with_post.txt exists).
    """
    if not run_script("generate_headings.py", log_dir=log_dir, env_overrides=env_common):
        reason = "generate_headings.py failed"
        log_skipped_keyword(keyword, country, site, category, reason)
        upsert_top_pick_report_row(keyword, country, site, category, status="failed", reason=reason)
        remove_keyword_from_csv_atomic(keyword, country, site, category)
        _close_master(f"keyword='{keyword}' (generate_headings failed)")
        return

    if not run_script("amend_product_names.py", log_dir=log_dir, env_overrides=env_common):
        reason = "amend_product_names.py failed"
        log_skipped_keyword(keyword, country, site, category, reason)
        upsert_top_pick_report_row(keyword, country, site, category, status="failed", reason=reason)
        remove_keyword_from_csv_atomic(keyword, country, site, category)
        _close_master(f"keyword='{keyword}' (amend_product_names failed)")
        return

    if os.path.exists(no_product_flag):
        reason = "No product names extracted"
        log(f"🚫 No product names extracted for '{keyword}' ({country}). Skipping.\n")
        log_skipped_keyword(keyword, country, site, category, reason)
        upsert_top_pick_report_row(keyword, country, site, category, status="failed", reason=reason)
        remove_keyword_from_csv_atomic(keyword, country, site, category)
        _close_master(f"keyword='{keyword}' (no products)")
        return

    pipeline_failed = False

    for script in SCRIPTS[3:]:
        if script in SKIP_SCRIPTS:
            log(f"⏭️ Skipping {script} due to ORCH_SKIP_SCRIPTS")
            continue

        success = run_script(script, log_dir=log_dir, env_overrides=env_common)

        if script == "get_response_from_openai.py":
            if os.path.exists(abandon_file):
                try:
                    reason = open(abandon_file, "r", encoding="utf-8").read().strip()
                except Exception:
                    reason = "Post abandoned (see abandon_post.txt)."
                reason = reason or "Post abandoned"

                log(f"🚫 Abandoning post for '{keyword}' ({country}).\n{reason}\n")
                log_skipped_keyword(keyword, country, site, category, "Post abandoned")
                upsert_top_pick_report_row(
                    keyword,
                    country,
                    site,
                    category,
                    status="failed",
                    reason=reason
                )
                remove_keyword_from_csv_atomic(keyword, country, site, category)
                _close_master(f"keyword='{keyword}' (abandoned)")
                pipeline_failed = True
                break

        if script == "insert_amazon_links_images.py" and not success:
            reason = read_insert_amazon_failure_reason(output_dir) or "No valid Amazon product data"

            log(
                f"🚫 Skipping remaining steps for '{keyword}' ({country}) "
                f"due to failed Amazon product injection.\nReason: {reason}\n"
            )
            log_skipped_keyword(keyword, country, site, category, reason)
            upsert_top_pick_report_row(
                keyword,
                country,
                site,
                category,
                status="failed",
                reason=reason
            )
            remove_keyword_from_csv_atomic(keyword, country, site, category)
            _close_master(f"keyword='{keyword}' (amazon injection failed)")
            pipeline_failed = True
            break

        if not success:
            reason = f"{script} failed"
            log_skipped_keyword(keyword, country, site, category, reason)
            upsert_top_pick_report_row(keyword, country, site, category, status="failed", reason=reason)
            remove_keyword_from_csv_atomic(keyword, country, site, category)
            _close_master(f"keyword='{keyword}' ({script} failed)")
            pipeline_failed = True
            break

    if pipeline_failed:
        return

    upsert_top_pick_report_row(keyword, country, site, category, status="success", reason="")

    log(f"⏭️ Post-processing for: {keyword} ({country}) @ {site} [{category}]")

    try:
        ok = remove_keyword_from_csv_atomic(keyword, country, site, category)
        if ok:
            log(
                f"🗑️ Removed completed keyword: {keyword} ({country}) "
                f"@ {site} [{category}] from keywords.csv"
            )
        else:
            log(
                f"[WARN] Could not atomically remove completed keyword for: "
                f"{keyword} ({country}) @ {site} [{category}]"
            )
    except Exception as e:
        log(f"[ERROR] Failed to remove keyword from CSV: {e}")

    log(f"✅ Finished processing: {keyword} ({country}) @ {site} [{category}]\n")
    _close_master(f"keyword='{keyword}'")
    time.sleep(1)


def _run_scrape_only(
    keyword,
    country,
    site,
    category,
    output_dir,
    log_dir,
    env_common,
    flag_file,
    force_plain: bool = False,
) -> str:
    """
    Run scrape_urls.py once.

    Returns:
      - 'ok'              : scrape succeeded and proceed_with_post.txt exists
      - 'insufficient'    : scrape ran, but not enough URLs yet, and more attempts remain
      - 'retry_exhausted' : scrape ran, but max attempts have now been used up
      - 'script_failed'   : scrape_urls.py process itself failed
    """
    prior = get_url_retry_count(keyword, country, site, category)

    if prior >= MAX_URL_RETRIES:
        reason = f"Not enough URLs (retry {prior}/{MAX_URL_RETRIES})"
        log(
            f"🧹 Retry budget already exhausted for '{keyword}' ({country}) "
            f"@ {site} [{category}] — no further scrape attempt."
        )
        upsert_top_pick_report_row(
            keyword,
            country,
            site,
            category,
            status="failed",
            reason=reason
        )
        return "retry_exhausted"

    env_attempt = dict(env_common)
    env_attempt["ORCH_SCRAPE_ATTEMPT"] = str(prior + 1)

    if force_plain:
        env_attempt["ORCH_FORCE_PLAIN_QUERY"] = "1"
    else:
        env_attempt.pop("ORCH_FORCE_PLAIN_QUERY", None)

    try:
        if os.path.exists(flag_file):
            os.remove(flag_file)
    except Exception:
        pass

    mode = "PLAIN" if force_plain else "SUFFIX"
    log(f"🧪 scrape_urls.py ({mode}) attempt {prior + 1}/{MAX_URL_RETRIES} for '{keyword}' ({country})")

    ok = run_script("scrape_urls.py", log_dir=log_dir, env_overrides=env_attempt)
    if not ok:
        upsert_top_pick_report_row(
            keyword,
            country,
            site,
            category,
            status="failed",
            reason="scrape_urls.py failed"
        )
        return "script_failed"

    if os.path.exists(flag_file):
        reset_url_retry(keyword, country, site, category)
        return "ok"

    n = bump_url_retry(keyword, country, site, category, reason="Not enough URLs")
    remaining = max(0, MAX_URL_RETRIES - n)
    reason = f"Not enough URLs (retry {n}/{MAX_URL_RETRIES})"

    log(
        f"⚠️ Not enough URLs for '{keyword}' ({country}) — "
        f"attempt {n}/{MAX_URL_RETRIES}. Remaining budget: {remaining}."
    )
    log_skipped_keyword(keyword, country, site, category, reason)

    if n >= MAX_URL_RETRIES:
        upsert_top_pick_report_row(
            keyword,
            country,
            site,
            category,
            status="failed",
            reason=reason
        )
        log(
            f"🧹 Reached MAX_URL_RETRIES={MAX_URL_RETRIES} for '{keyword}' ({country}) "
            f"@ {site} [{category}]"
        )
        return "retry_exhausted"

    upsert_top_pick_report_row(
        keyword,
        country,
        site,
        category,
        status="deferred",
        reason=reason
    )
    return "insufficient"


def _attempt_scrape_until_done(
    keyword,
    country,
    site,
    category,
    output_dir,
    log_dir,
    env_common,
    flag_file,
    max_attempts_this_phase=None,
    sleep_between_attempts=True,
):
    """
    Try scrape_urls.py for this phase only.

    The retry count is persisted in logs/url_retry_state.json, so a first pass can
    make one attempt and a later final pass can use the remaining budget.

    Attempt strategy:
      - Retry count 0 / first ever attempt: normal SUFFIX mode
      - Retry count >= 1: PLAIN mode

    Returns:
      - "ok"
      - "script_failed"
      - "retry_exhausted"
      - "insufficient"
    """
    max_attempts_this_phase = MAX_URL_RETRIES if max_attempts_this_phase is None else int(max_attempts_this_phase)
    max_attempts_this_phase = max(0, max_attempts_this_phase)

    if max_attempts_this_phase <= 0:
        return "retry_exhausted"

    for phase_attempt_no in range(1, max_attempts_this_phase + 1):
        prior = get_url_retry_count(keyword, country, site, category)
        force_plain = prior >= 1

        status = _run_scrape_only(
            keyword,
            country,
            site,
            category,
            output_dir,
            log_dir,
            env_common,
            flag_file,
            force_plain=force_plain,
        )

        if status == "ok":
            return "ok"

        if status == "script_failed":
            return "script_failed"

        if status == "retry_exhausted":
            return "retry_exhausted"

        if phase_attempt_no < max_attempts_this_phase:
            if sleep_between_attempts:
                sleep_for = max(1, URL_RETRY_SLEEP_BASE_SEC * phase_attempt_no)
                log(
                    f"🕘 Waiting {sleep_for}s before retry "
                    f"{phase_attempt_no + 1}/{max_attempts_this_phase} for '{keyword}' ({country})"
                )
                time.sleep(sleep_for)
        else:
            return "insufficient"

    return "insufficient"


# ── Main ───────────────────────────────────────────────────────────────────────
def _handle_scrape_status(
    scrape_status,
    keyword,
    country,
    site,
    category,
    output_dir,
    log_dir,
    env_common,
    no_product_flag,
    abandon_file,
):
    """
    Finish a keyword after a scrape phase.

    Returns True if the keyword is finished, False if it should remain queued
    for the final URL retry pass.
    """
    if scrape_status == "ok":
        _run_pipeline_after_scrape(
            keyword,
            country,
            site,
            category,
            output_dir,
            log_dir,
            env_common,
            no_product_flag,
            abandon_file,
        )
        return True

    if scrape_status == "script_failed":
        log_skipped_keyword(keyword, country, site, category, "scrape_urls.py failed")
        remove_keyword_from_csv_atomic(keyword, country, site, category)
        _close_master(f"keyword='{keyword}' (scrape script failed)")
        return True

    retry_count = get_url_retry_count(keyword, country, site, category)
    reason = f"Not enough URLs (retry {retry_count}/{MAX_URL_RETRIES})"

    if scrape_status == "retry_exhausted" or retry_count >= MAX_URL_RETRIES:
        upsert_top_pick_report_row(
            keyword,
            country,
            site,
            category,
            status="failed",
            reason=reason,
        )
        log(
            f"⚠️ Giving up on '{keyword}' ({country}) @ {site} [{category}] "
            f"after {retry_count}/{MAX_URL_RETRIES} scrape attempts.\n"
        )
        _close_master(f"keyword='{keyword}' (retry exhausted)")
        return True

    upsert_top_pick_report_row(
        keyword,
        country,
        site,
        category,
        status="deferred",
        reason=reason,
    )
    _close_master(f"keyword='{keyword}' (deferred for final retry pass)")
    return False


def main():
    all_keywords = read_keywords()
    if not all_keywords:
        log("No keywords found. Exiting.")
        return

    log(
        f"🚀 Starting batch. Keywords={len(all_keywords)} | "
        f"MAX_URL_RETRIES={MAX_URL_RETRIES} | "
        f"Retry strategy=one initial scrape, then final pass for URL-insufficient keywords | "
        f"Skipped scripts={sorted(SKIP_SCRIPTS) if SKIP_SCRIPTS else 'None'}"
    )

    deferred_keywords = []

    for idx, (keyword, country, site, category) in enumerate(all_keywords, start=1):
        log(f"\n🔵 [{idx}/{len(all_keywords)}] Processing: {keyword} ({country}) @ {site} [{category}]")

        output_dir, log_dir, env_common, flag_file, no_product_flag, abandon_file = _prepare_context(
            keyword, country, site, category
        )

        upsert_top_pick_report_row(
            keyword,
            country,
            site,
            category,
            status="started",
            selected_top_pick="",
            reason="",
        )

        scrape_status = _attempt_scrape_until_done(
            keyword,
            country,
            site,
            category,
            output_dir,
            log_dir,
            env_common,
            flag_file,
            max_attempts_this_phase=1,
            sleep_between_attempts=False,
        )

        finished = _handle_scrape_status(
            scrape_status,
            keyword,
            country,
            site,
            category,
            output_dir,
            log_dir,
            env_common,
            no_product_flag,
            abandon_file,
        )

        if not finished:
            deferred_keywords.append((keyword, country, site, category))
            log(f"↩️ Deferred URL retry until final pass: {keyword} ({country})")

    if deferred_keywords:
        log(f"\n🔁 Starting final URL retry pass for {len(deferred_keywords)} deferred keyword(s).")

    for idx, (keyword, country, site, category) in enumerate(deferred_keywords, start=1):
        retry_count = get_url_retry_count(keyword, country, site, category)
        remaining_budget = max(0, MAX_URL_RETRIES - retry_count)

        output_dir, log_dir, env_common, flag_file, no_product_flag, abandon_file = _prepare_context(
            keyword, country, site, category
        )

        if remaining_budget <= 0:
            scrape_status = "retry_exhausted"
        else:
            log(
                f"\n🟣 [retry {idx}/{len(deferred_keywords)}] Reprocessing: "
                f"{keyword} ({country}) @ {site} [{category}] | "
                f"remaining attempts={remaining_budget}"
            )
            scrape_status = _attempt_scrape_until_done(
                keyword,
                country,
                site,
                category,
                output_dir,
                log_dir,
                env_common,
                flag_file,
                max_attempts_this_phase=remaining_budget,
                sleep_between_attempts=True,
            )

        _handle_scrape_status(
            scrape_status,
            keyword,
            country,
            site,
            category,
            output_dir,
            log_dir,
            env_common,
            no_product_flag,
            abandon_file,
        )

    log("🏁 All keywords processed.")


if __name__ == "__main__":
    main()