import os
import re
import csv
import traceback
import requests
import logging
from datetime import datetime
from requests.auth import HTTPBasicAuth
from rich.console import Console
from rich.logging import RichHandler
from utils.config_reader import read_keyword_and_country, read_all_keywords
from utils.path_helpers import get_paths
from pathlib import Path
from urllib.parse import urlparse
import json
import argparse
import time
import shutil


console = Console()
log = logging.getLogger("scraper")
log.setLevel(logging.DEBUG)

# Will be set dynamically in main()
BASE_URL = None

# --------------------------------------------------------------------------- #
#                              LIGHTWEIGHT LOGGER                             #
# --------------------------------------------------------------------------- #
def log_ts(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
#                               LOGGING HELPERS                               #
# --------------------------------------------------------------------------- #



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hreflang-sync", choices=["none", "post", "full"], default="post")
    return p.parse_args()

def setup_logging(log_dir: str, keyword: str, country: str) -> None:
    def sanitize(s: str) -> str:
        import re
        return re.sub(r"[^A-Za-z0-9_-]+", "_", s or "")

    safe_keyword = sanitize(keyword.replace(" ", "_"))
    safe_country = sanitize(country.split(",")[0])
    safe_keyword_country = f"{safe_keyword}_{safe_country}"
    keyword_log_dir = os.path.join(log_dir, safe_keyword_country)
    os.makedirs(keyword_log_dir, exist_ok=True)
    log_file = os.path.join(keyword_log_dir, f"scrape_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    log.handlers.clear()
    log.addHandler(file_handler)
    log.addHandler(RichHandler(console=console, rich_tracebacks=True, markup=True))


def log_error(log_file: str, message: str) -> None:
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {message}\n")


# --------------------------------------------------------------------------- #
#                        CATEGORY / SITE CONFIG HELPERS                        #
# --------------------------------------------------------------------------- #

def resolve_config_path(filename: str, preferred_dir: Path | None = None, allow_missing: bool = False) -> Path | None:
    candidates: list[Path] = []
    if preferred_dir:
        candidates.append(Path(preferred_dir) / filename)

    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent

    candidates += [
        cwd / "config" / filename,
        script_dir / "config" / filename,
        cwd / filename,
        script_dir / filename,
    ]

    for p in candidates:
        if p.exists():
            return p

    if allow_missing:
        return None

    raise FileNotFoundError(
        "Could not locate configuration file. Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def load_category_config() -> dict:
    cfg_path = Path("config") / "category_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing required config file: {cfg_path}")
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to load {cfg_path}: {e}")


def detect_category(keyword: str, category_config: dict) -> str:
    """
    Determine which category to use for this keyword.

    Strategy:
    - First, try to match the category key name inside the keyword
      (e.g. keyword contains "luggage" -> category "luggage").
    - Then, try matching based on include_keywords in each category config.
    - If there is only one category defined and nothing matches, use that.
    - Otherwise, raise an error instead of falling back to a 'default'.
    """
    kw = (keyword or "").lower()
    if not category_config:
        raise RuntimeError("category_config.json has no categories defined.")

    items = list(category_config.items())

    # 1) Try to match category key name in the keyword string
    for cat, cfg in items:
        if cat.lower() in kw:
            return cat

    # 2) Try include_keywords for each category
    for cat, cfg in items:
        includes = [str(s).lower() for s in cfg.get("include_keywords", [])]
        if includes and any(tok in kw for tok in includes):
            return cat

    # 3) If there is only one category, assume that one
    if len(items) == 1:
        only_cat = items[0][0]
        log.info(
            f"No explicit category match for keyword '{keyword}'. "
            f"Using the only defined category: '{only_cat}'."
        )
        return only_cat

    # 4) Multiple categories and no match -> hard error
    raise RuntimeError(
        f"Could not detect category for keyword '{keyword}'. "
        f"Available categories: {', '.join(category_config.keys())}"
    )


def get_target_site_for_category(category: str, category_config: dict) -> str:
    """
    Return the target_site for the resolved category.

    There is no global default or hard-coded fallback. If the category or its
    target_site is missing, we fail fast.
    """
    cfg = category_config.get(category)
    if not isinstance(cfg, dict):
        raise RuntimeError(
            f"No configuration found for category '{category}' "
            f"in category_config.json. Available: {list(category_config.keys())}"
        )

    site = (cfg.get("target_site") or "").strip()
    if not site:
        raise RuntimeError(
            f"Category '{category}' in category_config.json does not define 'target_site'."
        )

    return site.rstrip("/")


def hostname_from_url(url: str) -> str:
    return urlparse(url).netloc


def _normalize_site_url(site: str) -> str:
    s = (site or "").strip()
    if not s:
        return s
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9+.\-]*://', s):
        s = "https://" + s
    return s.rstrip("/")


def _host_variants(site_url: str) -> list[str]:
    host = (urlparse(site_url).netloc or site_url).strip().lower()
    no_www = host[4:] if host.startswith("www.") else host
    with_www = f"www.{no_www}"
    return [host, no_www, with_www] if host != no_www else [host, with_www]


def read_keyword_country_site_from_current():
    path = Path("config/current_keyword.csv")
    if not path.exists():
        kw, co = read_keyword_and_country()
        return kw, (co or "").strip().upper(), None, None

    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            row = next(csv.reader(f))
        keyword = (row[0] if len(row) > 0 else "").strip()
        country_cell = (row[1] if len(row) > 1 else "").strip()
        site_cell = (row[2] if len(row) > 2 else "").strip()
        category_cell = (row[3] if len(row) > 3 else "").strip()

        if "," in country_cell and not site_cell:
            parts = [p.strip() for p in country_cell.split(",", 1)]
            country_cell = parts[0]
            site_cell = parts[1] if len(parts) > 1 else ""

        country_clean = country_cell.split(",")[0].strip().upper()
        site_norm = _normalize_site_url(site_cell) if site_cell else None
        category_key = underscore_slug(category_cell) if category_cell else None

        return keyword, country_clean, site_norm, category_key
    except Exception as e:
        log.warning(f"Could not read site/country/category cleanly from {path}: {e}")
        kw, co = read_keyword_and_country()
        return kw, (co or "").strip().upper(), None, None


def underscore_slug(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')


# --------------------------------------------------------------------------- #
#                          WORDPRESS CREDENTIALS                              #
# --------------------------------------------------------------------------- #

def read_wordpress_credentials_for_site(base_url: str):
    site_url = _normalize_site_url(base_url)
    hosts = _host_variants(site_url)
    config_dir = Path("config")

    direct_candidates = [config_dir / f"wordpress_credentials_{h}.txt" for h in hosts]

    all_creds = list(config_dir.glob("wordpress_credentials_*.txt"))
    fuzzy_candidates = []
    if not any(p.exists() for p in direct_candidates):
        for p in all_creds:
            name_host = p.stem.replace("wordpress_credentials_", "").lower()
            if any(name_host.endswith(h) or h.endswith(name_host) for h in hosts):
                fuzzy_candidates.append(p)

    for p in direct_candidates + fuzzy_candidates:
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    user = f.readline().strip()
                    password = f.readline().strip()
                if not user or not password:
                    raise RuntimeError(f"Credentials file missing user/password: {p}")
                log.info(f"Loaded WordPress credentials from {p}")
                return user, password
            except Exception as e:
                raise RuntimeError(f"Failed to read WordPress credentials at {p}: {e}")

    available = [c.name for c in all_creds]
    raise RuntimeError(
        "Missing WordPress credentials file.\n"
        f"Tried hosts: {hosts}\n"
        "Expected one of:\n  "
        + "\n  ".join(f"config/wordpress_credentials_{h}.txt" for h in hosts)
        + f"\nAvailable: {available}"
    )


# --------------------------------------------------------------------------- #
#                            CONTENT / CSV HELPERS                            #
# --------------------------------------------------------------------------- #

def read_first_post_from_csv(file_path: str, log_file: str):
    """
    Read the first post (title + slug) from a hash-delimited CSV.
    Hreflang fields have been removed.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter="#")
            next(reader)  # skip header
            for row in reader:
                if len(row) >= 3:
                    title, slug = row[0].strip(), row[2].strip()
                    if not title or not slug:
                        log_error(log_file, f"Empty title or slug in CSV '{file_path}': {row}")
                        return None
                    print(f"Preparing post: Title='{title}', Slug='{slug}'")
                    return {"title": title, "slug": slug}
        return None
    except Exception as e:
        log_error(log_file, f"Error reading CSV '{file_path}': {e}")
        raise


def read_blog_content(file_path: str, log_file: str):
    try:
        content = Path(file_path).read_text(encoding="utf-8")
        if not content.strip():
            log_error(log_file, f"Blog content is empty: {file_path}")
            return None
        return content
    except Exception as e:
        log_error(log_file, f"Error reading blog content: {e}")
        raise


# --------------------------------------------------------------------------- #
#                    WORDPRESS POST / META MANAGEMENT                         #
# --------------------------------------------------------------------------- #

def get_post_id_by_slug(slug: str, user: str, password: str, log_file: str):
    try:
        r = requests.get(
            f"{BASE_URL}/wp-json/wp/v2/posts",
            params={"slug": slug},
            auth=(user, password),
        )
        if r.status_code == 200:
            posts = r.json()
            return posts[0]["id"] if posts else None
        else:
            log_error(log_file, f"Failed to fetch post for slug '{slug}': {r.status_code} - {r.text}")
            return None
    except Exception:
        log_error(log_file, f"Error checking post existence:\n{traceback.format_exc()}")
        return None


def upsert_published_post_by_slug(slug, title, content, user, password, category_id, log_file):
    # 1) Find existing post by slug
    post_id = get_post_id_by_slug(slug, user, password, log_file)

    # 2) If exists -> overwrite it (and keep published)
    if post_id:
        ok = update_wordpress_post(
            post_id=post_id,
            title=title,
            slug=slug,
            content=content,
            category_id=category_id,
            user=user,
            password=password,
            log_file=log_file,
        )
        if not ok:
            raise RuntimeError(f"Update failed for existing post ID {post_id}")
        return post_id, slug

    # 3) If not exists -> create a NEW published post (recommended fallback)
    post_payload = {
        "title": title,
        "slug": slug,
        "categories": [category_id],
        "content": content,
        "status": "publish",
    }

    log.info(f"No existing post for slug '{slug}'. Creating NEW published post.")

    try:
        create_r = requests.post(
            f"{BASE_URL}/wp-json/wp/v2/posts",
            json=post_payload,
            auth=HTTPBasicAuth(user, password),
        )
    except Exception:
        raise RuntimeError(f"HTTP error during post creation:\n{traceback.format_exc()}")

    if create_r.status_code not in [200, 201]:
        raise RuntimeError(f"Failed to create post: {create_r.status_code} - {create_r.text}")

    post_id = create_r.json()["id"]
    log.info(f"Created new published post ID {post_id} with slug '{slug}'")
    return post_id, slug



def update_wordpress_post(post_id, title, slug, content, category_id, user, password, log_file):
    try:
        update_data = {
            "title": title,
            "slug": slug,
            "content": content,
            "categories": [category_id],
            "status": "publish",   # <-- IMPORTANT: keep it published
        }

        print(f"Updating existing WordPress post ID {post_id} (publishing)...")
        r = requests.post(
            f"{BASE_URL}/wp-json/wp/v2/posts/{post_id}",
            json=update_data,
            auth=HTTPBasicAuth(user, password),
        )

        if r.status_code == 200:
            print(f"Updated post ID {post_id} successfully (status=publish).")
            return True
        else:
            log_error(log_file, f"Failed to update post ID {post_id}: {r.status_code} - {r.text}")
            return False

    except Exception:
        log_error(log_file, f"HTTP error during update:\n{traceback.format_exc()}")
        return False


def category_key_to_wp_slug(category_key: str) -> str:
    """
    Convert a config category key like 'sleeping_masks' or 'luggage'
    into a WordPress-style slug like 'sleeping-masks' or 'luggage'.
    """
    key = (category_key or "").strip().lower()
    # Replace underscores with hyphens and collapse any non-alphanumerics.
    key = key.replace("_", "-")
    key = re.sub(r"[^a-z0-9\-]+", "-", key)
    return key.strip("-")


def get_wp_category_id_by_slug(slug: str, user: str, password: str) -> int | None:
    """
    Look up a WordPress category ID by its slug via the WP REST API.
    Returns the ID if found, else None.
    """
    try:
        r = requests.get(
            f"{BASE_URL}/wp-json/wp/v2/categories",
            params={"slug": slug},
            auth=HTTPBasicAuth(user, password),
        )
        if r.status_code == 200:
            cats = r.json()
            if cats:
                cat_id = cats[0]["id"]
                log.info(f"Resolved WP category slug '{slug}' to ID {cat_id}")
                return cat_id
            else:
                log.error(f"No WordPress category found for slug '{slug}'")
                return None
        else:
            log.error(
                f"Failed to fetch category for slug '{slug}': "
                f"{r.status_code} - {r.text}"
            )
            return None
    except Exception:
        log.error(
            f"Error fetching category for slug '{slug}':\n{traceback.format_exc()}"
        )
        return None

# --------------------------------------------------------------------------- #
#                       CREATE-OR-GET POST (main flow)                        #
# --------------------------------------------------------------------------- #

def create_new_draft_post_always(
    slug, title, content, user, password, category_id, status="draft"
):
    """
    Always create a NEW WordPress post as a draft.
    Never updates existing posts.

    If the provided slug already exists, we generate a unique one by appending -2, -3, ...
    Returns: (post_id, final_slug)
    """

    # Helper: check if a slug already exists
    def slug_exists(s: str) -> bool:
        r = requests.get(
            f"{BASE_URL}/wp-json/wp/v2/posts",
            params={"slug": s},
            auth=HTTPBasicAuth(user, password),
        )
        if r.status_code != 200:
            raise RuntimeError(
                f"Failed to check slug '{s}': {r.status_code} - {r.text}"
            )
        return bool(r.json())

    # Pick a unique slug
    base_slug = (slug or "").strip()
    if not base_slug:
        raise ValueError("Slug is empty; cannot create post.")

    final_slug = base_slug
    if slug_exists(final_slug):
        i = 2
        while True:
            candidate = f"{base_slug}-{i}"
            if not slug_exists(candidate):
                final_slug = candidate
                break
            i += 1

        log.info(f"Slug '{base_slug}' exists; using new slug '{final_slug}'")

    post_payload = {
        "title": title,
        "slug": final_slug,
        "categories": [category_id],
        "content": content,
        "status": status,  # draft by default
    }

    log.info(f"Creating NEW post (draft) with slug '{final_slug}'")

    try:
        create_r = requests.post(
            f"{BASE_URL}/wp-json/wp/v2/posts",
            json=post_payload,
            auth=HTTPBasicAuth(user, password),
        )
    except Exception:
        raise RuntimeError(
            f"HTTP error during post creation:\n{traceback.format_exc()}"
        )

    if create_r.status_code not in [200, 201]:
        raise RuntimeError(
            f"Failed to create post: {create_r.status_code} - {create_r.text}"
        )

    post_id = create_r.json()["id"]
    log.info(f"Created new post ID {post_id} with slug '{final_slug}'")
    return post_id, final_slug


# --------------------------------------------------------------------------- #
#                              HREFLANG HELPERS                               #
# --------------------------------------------------------------------------- #

def _hreflang_registry_path() -> Path:
    return Path("output") / "hreflang_registry.json"

def _load_registry_robust(path: Path) -> dict:
    """
    Load the canonical hreflang registry. Supports both:
    - a normal single JSON object
    - legacy malformed files where entry blocks are still present but the
      outer JSON object has been corrupted
    """
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    merged: dict[str, dict] = {}
    entry_pattern = re.compile(r'"(?P<key>(?:[^"\\]|\\.)+)"\s*:\s*(?P<obj>\{[^{}]*\})', re.S)

    for match in entry_pattern.finditer(text):
        try:
            keyword = json.loads('"' + match.group('key') + '"')
            regions = json.loads(match.group('obj'))
        except Exception:
            continue
        if not isinstance(regions, dict):
            continue

        slot = merged.setdefault(keyword, {})
        for region, url in regions.items():
            region_key = str(region).strip().upper()
            if region_key == 'GB':
                region_key = 'UK'
            if region_key in {'UK', 'US', 'CA'} and url:
                slot[region_key] = str(url).strip()

    return merged

def _save_registry(path: Path, registry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        bak = path.with_suffix(path.suffix + '.bak')
        try:
            shutil.copy2(path, bak)
        except Exception:
            pass
    path.write_text(json.dumps(registry, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding='utf-8')

def _merge_hreflang_registry_entry(keyword: str, country: str, url: str) -> dict:
    path = _hreflang_registry_path()
    registry = _load_registry_robust(path)
    slot = registry.setdefault(keyword, {})
    c = (country or '').strip().upper()
    if c == 'GB':
        c = 'UK'
    if c not in {'UK', 'US', 'CA'}:
        raise ValueError(f'Unsupported hreflang country: {country}')
    slot[c] = url
    _save_registry(path, registry)
    return registry

def update_post_meta(post_id: int, field: str, value: str, user: str, password: str):
    url = f"{BASE_URL}/wp-json/wp/v2/posts/{post_id}"
    r = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={"meta": {field: value}},
        auth=HTTPBasicAuth(user, password),
    )
    if r.status_code not in [200, 201]:
        log.warning(f"Failed to update {field} on post {post_id}: {r.status_code} - {r.text}")

def get_post_id_by_url(post_url: str, user: str, password: str):
    slug = post_url.rstrip('/').rsplit('/', 1)[-1]
    return get_post_id_by_slug(slug, user, password, str((Path('logs') / 'wp_errors.log').resolve()))

def update_hreflang_for_post_and_siblings(keyword: str, country: str, slug: str, post_id: int, user: str, password: str):
    t0 = time.monotonic()
    current_url = f"{BASE_URL.rstrip('/')}/{slug.strip('/')}"

    registry = _merge_hreflang_registry_entry(keyword, country, current_url)
    region_links = registry.get(keyword, {}) or {}

    field_map = {
        'UK': 'hreflang_uk',
        'US': 'hreflang_us',
        'CA': 'hreflang_ca',
    }

    # Update current post with the full merged hreflang set.
    for region, field in field_map.items():
        value = region_links.get(region, '')
        update_post_meta(post_id, field, value, user, password)

    # Update sibling posts for the same keyword so they now point to the new/updated region too.
    new_country = country.strip().upper().replace('GB', 'UK')
    new_field = field_map.get(new_country)
    if new_field:
        for region, sibling_url in region_links.items():
            if region == new_country or not sibling_url:
                continue
            try:
                sibling_post_id = get_post_id_by_url(sibling_url, user, password)
                if sibling_post_id:
                    update_post_meta(sibling_post_id, new_field, current_url, user, password)
            except Exception as e:
                log.warning(f"Could not update sibling post {sibling_url}: {e}")

    log_ts(f"hreflang(post) finished in {time.monotonic() - t0:.1f}s")


# --------------------------------------------------------------------------- #
#                                    MAIN                                     #
# --------------------------------------------------------------------------- #

def main():
    global BASE_URL
    args = parse_args()

    # ------------------------------------------------------------
    # 1) Read keyword / country / optional site from CSV
    # ------------------------------------------------------------
    keyword, country_raw, site_from_csv, category_from_csv = read_keyword_country_site_from_current()
    if not keyword:
        print("No keyword found in current_keyword.csv or config.")
        return

    setup_logging("logs", keyword, country_raw)
    log.info(f"Starting publish/update for '{keyword}' ({country_raw})")

    country = country_raw.split(",")[0].strip().upper()
    kw_slug = re.sub(r"[^a-z0-9]+", "_", keyword.lower()).strip("_")
    safe_keyword_country = f"{kw_slug}_{country}"

    # Build canonical paths for this run
    paths = get_paths(safe_keyword_country)
    output_dir = paths["output_dir"]

    # ------------------------------------------------------------
    # 2) Load category config & detect logical category for keyword
    # ------------------------------------------------------------
    try:
        category_config = load_category_config()
    except Exception as e:
        log.error(f"Failed to load category_config.json: {e}")
        return

    if category_from_csv:
        if category_from_csv not in category_config:
            log.error(
                f"Category '{category_from_csv}' from current_keyword.csv is not configured. "
                f"Available categories: {', '.join(category_config.keys())}"
            )
            return
        category = category_from_csv
        log.info(f"Using logical category '{category}' from current_keyword.csv")
    else:
        try:
            category = detect_category(keyword, category_config)
            log.info(f"Detected logical category '{category}' for keyword '{keyword}'")
        except Exception as e:
            log.error(f"Could not detect category for keyword '{keyword}': {e}")
            return

    # ------------------------------------------------------------
    # 3) Decide target site (BASE_URL)
    #    - Prefer site from CSV
    #    - Otherwise use category_config[target_site]
    # ------------------------------------------------------------
    if site_from_csv:
        BASE_URL = site_from_csv
        log.info(f"Using site from CSV: {BASE_URL}")
    else:
        try:
            BASE_URL = get_target_site_for_category(category, category_config)
            log.info(
                f"No site in CSV. Detected category '{category}' -> "
                f"BASE_URL set to {BASE_URL}"
            )
        except Exception as e:
            log.error(f"Failed to resolve target site for category '{category}': {e}")
            return

    # ------------------------------------------------------------
    # 4) Resolve input files for title/meta/slug and content
    # ------------------------------------------------------------
    title_file = output_dir / f"title_description_{country}.txt"
    blog_file  = output_dir / f"processed_blog_final_{country}.txt"

    # 4a) Load title / meta_description / slug from the hash-delimited TXT
    try:
        if not title_file.exists():
            log.error(f"Title/meta file not found: {title_file.resolve()}")
            try:
                contents = [p.name for p in output_dir.iterdir()]
                log.info(f"Contents of {output_dir}: {contents}")
            except Exception:
                pass
            return

        with open(title_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="#")
            header = next(reader, None)
            if not header:
                log.error(f"Empty or missing header in {title_file.resolve()}")
                return

            header = [h.strip().strip('"').lower() for h in header]

            def col_idx(name: str) -> int:
                try:
                    return header.index(name)
                except ValueError:
                    return -1

            i_title = col_idx("title")
            i_meta  = col_idx("meta_description")
            i_slug  = col_idx("slug")

            missing = [
                n for n, i in [
                    ("title", i_title),
                    ("meta_description", i_meta),
                    ("slug", i_slug),
                ] if i < 0
            ]
            if missing:
                log.error(
                    f"Missing required columns {missing} in {title_file.resolve()}. "
                    f"Headers: {header}"
                )
                return

            row = None
            for r in reader:
                if r and any(cell.strip().strip('"') for cell in r):
                    row = r
                    break

            if not row:
                log.error(f"No data rows found in {title_file.resolve()}")
                return

            title = row[i_title].strip().strip('"')
            meta_description = row[i_meta].strip().strip('"')
            slug = row[i_slug].strip().strip('"')

            if not title or not slug:
                log.error(
                    f"Empty title/slug in first data row of {title_file.resolve()}: {row}"
                )
                return

    except Exception as e:
        log.error(f"Failed to load post details from {title_file}: {e}")
        return

    # 4b) Load blog content (fallback to meta_description if empty/error)
    try:
        content = Path(blog_file).read_text(encoding="utf-8").strip()
        # Clean any stray placeholder tags that might slip through
        content = content.replace("&lt;/&gt;", "").replace("</>", "")
        if not content:
            log.warning(
                f"Blog content empty in {blog_file.resolve()}, "
                "falling back to meta_description."
            )
            content = meta_description
    except Exception as e:
        log.warning(
            f"Error reading blog content from {blog_file.resolve()}: {e}. "
            "Falling back to meta_description."
        )
        content = meta_description

    # ------------------------------------------------------------
    # 5) Resolve WP credentials & category ID for this category
    # ------------------------------------------------------------
    try:
        user, password = read_wordpress_credentials_for_site(BASE_URL)
    except Exception as e:
        log.error(f"Failed to read WordPress credentials: {e}")
        return

    wp_category_slug = category_key_to_wp_slug(category)
    wp_category_id = get_wp_category_id_by_slug(wp_category_slug, user, password)
    if not wp_category_id:
        log.error(
            f"Cannot continue: could not find WordPress category for slug "
            f"'{wp_category_slug}' (derived from config key '{category}')."
        )
        return

    # ------------------------------------------------------------
    # 6) Overwrite existing published post (by slug) or create new
    # ------------------------------------------------------------
    try:
        post_id, final_slug = upsert_published_post_by_slug(
            slug=slug,
            title=title,
            content=content,
            user=user,
            password=password,
            category_id=wp_category_id,
            log_file=str((Path("logs") / "wp_errors.log").resolve()),  # or your existing log file path
        )
        log.info(f"Upsert complete. Post ID {post_id}, slug '{final_slug}', status=publish")

    except Exception as e:
        log.error(f"Failed to publish/update post: {e}")
        return

    if args.hreflang_sync == "none":
        log_ts("Skipping hreflang updates (--hreflang-sync=none).")
        return

    try:
        update_hreflang_for_post_and_siblings(keyword, country, final_slug, post_id, user, password)
    except Exception as e:
        log.error(f"Failed post-level hreflang update: {e}")

    if args.hreflang_sync == "full":
        try:
            try:
                from utils.hreflang_sync import sync_all_hreflangs
            except Exception:
                from hreflang_sync import sync_all_hreflangs
            sync_all_hreflangs()
        except Exception as e:
            log.error(f"Full hreflang sync failed: {e}")


if __name__ == "__main__":
    main()
