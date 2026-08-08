import os
import pandas as pd
from openai import OpenAI as DeepSeekClient
from internal_links import log_deepseek_usage
import logging
import re
import csv
from pathlib import Path
from urllib.parse import urlparse
from textwrap import shorten
from datetime import datetime

# ------------------ Config ------------------
DEEPSEEK_API_KEY_FILE = Path("config/deepseek_api_key.txt")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
KEYWORD_FILE = Path("config/current_keyword.csv")
WP_API_URL = None  # built from 'site' in current_keyword.csv

# ------------------ Logging ------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FORMAT = '[%(levelname)s] %(asctime)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(
    level=logging.DEBUG,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[
        logging.FileHandler(LOG_DIR / "log_output.txt", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.debug("Logger initialised.")
logging.debug(f"Working directory: {Path.cwd()}")
logging.debug(f"Python file location: {Path(__file__).resolve()}")

# ------------------ DeepSeek client ------------------
_client: DeepSeekClient | None = None  # lazy singleton


def _get_deepseek_client() -> DeepSeekClient:
    """Lazily create and reuse a single DeepSeek client."""
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        try:
            if not DEEPSEEK_API_KEY_FILE.exists():
                logging.error(
                    f"DeepSeek key file not found: {DEEPSEEK_API_KEY_FILE.resolve()}"
                )
                raise FileNotFoundError(DEEPSEEK_API_KEY_FILE)
            api_key = DEEPSEEK_API_KEY_FILE.read_text(encoding="utf-8").strip()
            if not api_key:
                raise ValueError("DeepSeek API key file is empty.")
            os.environ["DEEPSEEK_API_KEY"] = api_key
            logging.info("DeepSeek API key loaded from file.")
        except Exception as e:
            logging.exception(f"Failed to load DeepSeek API key: {e}")
            raise SystemExit(1)

    try:
        _client = DeepSeekClient(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        logging.debug("DeepSeek client initialised successfully.")
    except Exception as e:
        logging.exception(f"Failed to initialise DeepSeek client: {e}")
        raise SystemExit(1)

    return _client


# ------------------ Helpers ------------------
def underscore_slug(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')


def read_keyword_from_file():
    """
    Reads config/current_keyword.csv and returns (keyword, country, site).
    Expects 3 columns: keyword,country,site.
    """
    try:
        if not KEYWORD_FILE.exists():
            raise FileNotFoundError(KEYWORD_FILE)
        with KEYWORD_FILE.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            row = next(reader, None)
            logging.debug(f"current_keyword.csv row: {row}")
            if not row or len(row) < 2:
                logging.error("current_keyword.csv must contain at least: keyword,country")
                return "", "", ""
            kw = row[0].strip()
            country = row[1].strip().upper()
            site = row[2].strip() if len(row) >= 3 else ""
            if not site:
                logging.error("current_keyword.csv missing 'site' (3rd column).")
                return "", "", ""
            site = site.rstrip("/")
            return kw, country, site
    except Exception as e:
        logging.exception(f"Failed to read current keyword file: {e}")
        return "", "", ""


def credentials_path_for_site(site: str) -> Path:
    parsed = urlparse(site)
    host = (parsed.netloc or parsed.path).strip("/")
    path = Path("config") / f"wordpress_credentials_{host}.txt"
    logging.debug(f"Resolved WP credentials path: {path}")
    return path


def read_wordpress_credentials(creds_path: Path):
    try:
        if not creds_path.exists():
            candidates = list(Path("config").glob("wordpress_credentials_*.txt"))
            logging.error(f"WordPress credentials file not found: {creds_path.resolve()}")
            logging.info(f"Available credential files: {[c.name for c in candidates]}")
            raise FileNotFoundError(creds_path)
        with creds_path.open('r', encoding='utf-8') as file:
            user = file.readline().strip()
            password = file.readline().strip()
        if not user or not password:
            raise ValueError(f"Credentials file missing user/password: {creds_path}")
        logging.info(f"Loaded WordPress credentials from {creds_path}")
        return user, password
    except Exception as e:
        logging.exception(f"Failed to read WordPress credentials: {e}")
        raise SystemExit(1)


def create_slug(keyword, country_code):
    keyword = keyword.strip().lower()
    country_code = country_code.strip().lower()
    match = re.search(r'\b(reviews?|review)\b$', keyword)
    base = re.sub(r'\b(reviews?|review)\b$', '', keyword).strip() if match else keyword
    base = re.sub(r'[^\w\s-]', '', base)
    base = re.sub(r'\s+', '-', base)
    slug = f"{base}-{country_code}-reviews" if match else f"{base}-{country_code}"
    logging.debug(f"create_slug -> {slug}")
    return slug


def generate_title_meta(blog_text, keyword):
    year = datetime.utcnow().year  # kept in case you want to use it later
    prompt = f"""
You are an expert SEO copywriter.

Task: Using the keyword "{keyword}", create:
- Title (≤60 chars): START with the keyword, add a concrete benefit/value angle.
- Meta Description (≤150 chars): include the keyword once; clearly state value + urgency with a soft CTA (e.g., "see why", "compare now"); avoid quotes and emojis.

Output format (EXACTLY two lines, no extra text/markdown):
Title: <title here>
Meta Description: <meta here>

Context:
{blog_text}
""".strip()

    try:
        logging.info("Calling DeepSeek for title/meta generation...")
        client = _get_deepseek_client()
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert SEO content writer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            extra_body={"thinking": {"type": "disabled"}},
        )
        log_deepseek_usage(
            response,
            label="title_description:title_and_meta",
            requested_model=DEEPSEEK_MODEL,
        )
        content = response.choices[0].message.content.strip()
        logging.debug(
            f"DeepSeek raw response (trimmed): "
            f"{shorten(content, width=400, placeholder='…')}"
        )
        title, meta = "", ""
        for line in content.splitlines():
            line = line.replace("**", "").strip()
            if line.lower().startswith("title:"):
                title = line.split(":", 1)[1].strip()
            elif line.lower().startswith("meta description:"):
                meta = line.split(":", 1)[1].strip()
        logging.info(
            f"Generated title/meta. Title len: {len(title)}, Meta len: {len(meta)}"
        )
        return title, meta
    except Exception as e:
        logging.exception(f"Error generating title/meta: {e}")
        return "", ""


def normalize_site(site: str) -> str:
    site = (site or "").strip()
    if not site:
        return site
    # Add scheme if missing (http/https/etc.)
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9+.\-]*://', site):
        site = "https://" + site
    return site.rstrip("/")


def main():
    try:
        keyword, country, site = read_keyword_from_file()
        logging.info(
            f"current_keyword -> keyword='{keyword}', country='{country}', site='{site}'"
        )

        if not keyword or not country:
            logging.error("Invalid or missing keyword or country.")
            return
        if country not in ['US', 'UK', 'CA']:
            logging.error(
                f"Unsupported country '{country}'. Expected one of ['US','UK','CA']."
            )
            return

        # Normalize site (ensures scheme present) and build WP API endpoint
        site = normalize_site(site)
        logging.info(f"Normalized site: {site}")

        global WP_API_URL
        WP_API_URL = f"{site}/wp-json/wp/v2/posts"
        logging.info(f"WordPress API endpoint: {WP_API_URL}")

        # Load site-specific credentials (kept in case you want to use them later)
        creds_path = credentials_path_for_site(site)
        WP_USERNAME, WP_APP_PASSWORD = read_wordpress_credentials(creds_path)

        # -------- Locate blog content (supports subfolder or flat layout) --------
        keyword_slug = underscore_slug(keyword)  # e.g., 'swiss_gear_luggage_reviews'
        output_root = Path("output")

        candidate_sub = (
            output_root
            / f"{keyword_slug}_{country}"
            / f"processed_blog_final_{country}.txt"
        )
        candidate_flat = (
            output_root / f"processed_blog_final_{country}_{keyword_slug}.txt"
        )

        if candidate_sub.exists():
            input_file = candidate_sub.resolve()
            out_dir = candidate_sub.parent
            logging.info(f"Using blog content (subfolder): {input_file}")
        elif candidate_flat.exists():
            input_file = candidate_flat.resolve()
            out_dir = output_root
            logging.info(f"Using blog content (flat): {input_file}")
        else:
            logging.error("Missing blog content file.")
            logging.error(f"Tried (subfolder): {candidate_sub.resolve()}")
            logging.error(f"Tried (flat):      {candidate_flat.resolve()}")
            matches = list(output_root.rglob(f"processed_blog_final_{country}*.txt"))
            if matches:
                logging.info(
                    f"Found similar files: {[str(m) for m in matches[:5]]}"
                )
            return

        blog_text = Path(input_file).read_text(encoding='utf-8').strip()
        if not blog_text:
            logging.warning(f"Blog content is empty: {input_file}")
            return

        # Generate fields
        slug = create_slug(keyword, country)
        title, meta = generate_title_meta(blog_text, keyword)
        if not title or not meta:
            logging.error("Failed to generate title/meta (empty result).")
            return

        # -------- Save TXT (with 'site' column, no hreflang columns) --------
        out_dir.mkdir(parents=True, exist_ok=True)
        if input_file == candidate_sub.resolve():
            output_file = out_dir / f"title_description_{country}.txt"
        else:
            output_file = out_dir / f"title_description_{country}_{keyword_slug}.txt"

        df = pd.DataFrame(
            [[title, meta, slug, country, site]],
            columns=['title', 'meta_description', 'slug', 'country', 'site'],
        )
        df.to_csv(
            output_file,
            index=False,
            sep="#",
            header=True,
            quoting=csv.QUOTE_ALL,
        )
        logging.info(
            f"Saved TXT: {output_file.resolve()} with header {list(df.columns)}"
        )

        logging.info("title_description.py completed successfully.")

    except Exception as e:
        logging.exception(f"Fatal error in main(): {e}")


if __name__ == "__main__":
    main()
