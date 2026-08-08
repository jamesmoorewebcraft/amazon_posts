import logging
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
# from fake_useragent import UserAgent  # optional; not needed now
from playwright.sync_api import sync_playwright
from tqdm import tqdm
import os
import time
import random
import csv
import re

from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# ── Global session ──
SESSION = None  # built in main()


# ── Setup logging ──
def setup_logging(keyword, country, site=None):
    # NOTE: folder name excludes 'site' by request
    safe_keyword = _sanitize_for_path(keyword)
    safe_keyword_country = f"{safe_keyword}_{country}"

    log_dir = os.path.join("logs", safe_keyword_country)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "generate_headings.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)



# ── Build a retrying session (optionally with proxies) ──
def build_session():
    s = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504, 522, 524),
        allowed_methods=frozenset(['GET', 'HEAD'])
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=50)
    s.mount("http://", adapter)
    s.mount("https://", adapter)

    # Stable, “normal” browser-like headers
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Upgrade-Insecure-Requests": "1",
    })
    return s

def _sanitize_for_path(s: str) -> str:
    s = s.strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]", "_", s)
    return s


# ── Read keyword ──
def read_current_keyword():
    """
    Reads config/current_keyword.csv and returns (keyword, country, site)
    - Supports 2 or 3 columns.
    - Uses CSV reader to handle quoted commas.
    """
    with open("config/current_keyword.csv", "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all(not (c or "").strip() for c in row):
                continue
            parts = [ (c or "").strip() for c in row ]
            if len(parts) >= 3:
                keyword, country, site = parts[0], parts[1], parts[2]
                return keyword.strip(), country.strip().upper(), site.strip()
            elif len(parts) == 2:
                keyword, country = parts[0], parts[1]
                return keyword.strip(), country.strip().upper(), None
            else:
                raise ValueError(f"Unexpected format in current_keyword.csv; got columns: {parts}")
    raise FileNotFoundError("current_keyword.csv appears to be empty.")




# ── Fallback requests fetch (now via shared SESSION + retries) ──
def fetch_with_requests(url):
    global SESSION
    try:
        # If you still want to occasionally rotate UA:
        # headers = {"User-Agent": UserAgent().random}
        # return SESSION.get(url, headers=headers, timeout=20).text
        resp = SESSION.get(url, timeout=20)
        # Do NOT raise_for_status(); retries already handled transient 5xx/429
        return resp.text
    except Exception as e:
        logging.warning(f"Requests attempt failed for {url}: {e}")
        return None


# ── JSON parser (unused in current script but retained) ──
def parse_json_for_content(data_json):
    results = []

    def walk(node):
        if isinstance(node, dict):
            heading = None
            text = None
            for k, v in node.items():
                kl = k.lower()
                if isinstance(v, (dict, list)):
                    walk(v)
                elif kl in ['headline', 'title', 'name'] and isinstance(v, str) and len(v.strip()) > 3:
                    heading = v.strip()
                elif kl in ['reviewbody', 'articlebody', 'description'] and isinstance(v, str) and len(v.strip()) > 20:
                    text = v.strip()
            if heading or text:
                results.append({
                    "level": "H2" if heading else "P",
                    "heading": heading or "Content",
                    "text": text or ""
                })
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data_json)
    return results


# ── New JS-rendered extraction logic ──
def extract_headings_and_content_safe(url, browser):
    page = None
    try:
        context = browser.new_context(
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/115.0.0.0 Safari/537.36")
        )
        page = context.new_page()
        page.goto(url, timeout=60000)
        page.wait_for_load_state('networkidle')
        page.wait_for_timeout(5000)  # Wait for dynamic content

        content_blocks = page.evaluate("""
        () => {
            try {
                const results = [];
                const headings = document.querySelectorAll("h1, h2, h3, h4, h5, h6");

                headings.forEach(h => {
                    const headingText = h.innerText.trim();
                    if (!headingText) return;

                    let container = h.closest('div') || h.parentElement;
                    if (!container) return;

                    const paragraphs = container.querySelectorAll('p, li');
                    const paraTexts = Array.from(paragraphs)
                        .map(p => p.innerText.trim())
                        .filter(p => p.length > 30);

                    results.push({
                        level: h.tagName,
                        heading: headingText,
                        text: paraTexts.join("\\n\\n")
                    });
                });

                return results;
            } catch (e) {
                return [];
            }
        }
        """)

        page.close()

        final_results = [
            (item['level'], item['heading'], item['text'])
            for item in content_blocks if item['heading'] or item['text']
        ]

        return url, final_results

    except Exception as e:
        logging.warning(f"Playwright container extract failed for {url}: {e}")
        try:
            if page:
                page.close()
        except:
            pass

    return url, []


# ── Fallback BeautifulSoup parser ──
def extract_from_html(content):
    soup = BeautifulSoup(content, 'html.parser')
    headings_with_content = []
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

    for i, heading in enumerate(headings):
        level = heading.name.upper()
        heading_text = heading.get_text(strip=True)
        content_text = extract_content_under_heading(heading, headings[i+1:] if i+1 < len(headings) else [])
        headings_with_content.append((level, heading_text, content_text))

    return headings_with_content


def extract_content_under_heading(start_heading, remaining_headings):
    content_blocks = []
    next_heading_tag_names = [h.name for h in remaining_headings]

    for sibling in start_heading.find_next_siblings():
        if sibling.name and sibling.name.lower() in next_heading_tag_names:
            break
        if sibling.name in ['div', 'section', 'article']:
            paras = sibling.find_all(['p', 'li'])
            for para in paras:
                text = para.get_text(strip=True)
                if text:
                    content_blocks.append(text)
        elif sibling.name in ['ul', 'ol']:
            items = sibling.find_all('li')
            content_blocks.extend([li.get_text(strip=True) for li in items if li.get_text(strip=True)])
        else:
            text = sibling.get_text(strip=True)
            if text:
                content_blocks.append(text)

    return "\n".join(content_blocks).strip()


# ── Process URLs ──
def process_keyword(keyword, country_code, site=None):
    # NOTE: output directory excludes 'site' by request
    safe_keyword = _sanitize_for_path(keyword)
    safe_keyword_country = f"{safe_keyword}_{country_code}"

    output_dir = os.path.join("output", safe_keyword_country)
    os.makedirs(output_dir, exist_ok=True)

    input_file = os.path.join(output_dir, f"input_urls_{country_code}.xlsx")
    headings_file = os.path.join(output_dir, f"headings_{country_code}.txt")
    content_file = os.path.join(output_dir, f"content_{country_code}.txt")
    failed_file = os.path.join(output_dir, f"failed_urls_{country_code}.txt")

    MAX_FILE_SIZE = 550 * 1024  # 550 KB
    stop_processing = False

    for file in [headings_file, content_file]:
        Path(file).write_text("")

    # ... (rest of your existing function body remains unchanged)

    print(f"[OK] Finished for {country_code}, Keyword: {keyword}"
          f"{f', Site: {site}' if site else ''} : {headings_file}, {content_file}")



    MAX_FILE_SIZE = 550 * 1024  # 550 KB
    stop_processing = False

    for file in [headings_file, content_file]:
        Path(file).write_text("")

    try:
        data = pd.read_excel(input_file)
        if 'url' not in data.columns:
            raise ValueError(f"Missing 'url' column in {input_file}")
        if 'match_score' not in data.columns:
            raise ValueError("Missing 'match_score' column in input file.")
        urls = data[data['match_score'] >= 5]['url'].dropna().drop_duplicates().tolist()
    except Exception as e:
        logging.error(f"Error reading {input_file}: {e}")
        return

    processed_urls = set()
    if Path(headings_file).exists():
        with open(headings_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("URL: "):
                    processed_urls.add(line.strip().replace("URL: ", ""))

    failed_urls = []

    with sync_playwright() as p:
        # Try system Chrome first; falls back to bundled if not available
        browser = p.chromium.launch(channel="chrome", headless=True)


        for url in tqdm(urls, desc=f"Processing {country_code}, Keyword: {keyword}"):
            if stop_processing:
                logging.info("Stopping further processing due to file size limit.")
                break

            if url in processed_urls:
                logging.info(f"Skipping already processed URL: {url}")
                continue

            if "amazon." in url:
                html = fetch_with_requests(url)
                if not html:
                    time.sleep(0.5 + random.random())  # light jitter before one quick retry
                    html = fetch_with_requests(url)
                headings_and_text = extract_from_html(html) if html else []
                url_result = url
            else:
                url_result, headings_and_text = extract_headings_and_content_safe(url, browser)

            if not headings_and_text:
                fallback_html = fetch_with_requests(url)
                if fallback_html:
                    headings_and_text = extract_from_html(fallback_html)

            if not headings_and_text:
                failed_urls.append(url)
                logging.warning(f"Failed to extract content from {url}")
                continue

            # Write headings file (not size constrained)
            with open(headings_file, 'a', encoding='utf-8') as hf:
                hf.write(f"\nURL: {url}\n")
                for level, heading, _ in headings_and_text:
                    hf.write(f"{level}: {heading}\n")

            # Prepare content string
            content_str = f"\nURL: {url}\n"
            for level, heading, text in headings_and_text:
                content_str += f"{level} {heading}\nText: {text}\n\n"

            encoded_content = content_str.encode('utf-8')
            content_size = len(encoded_content)
            current_size = os.path.getsize(content_file) if os.path.exists(content_file) else 0

            if current_size + content_size > MAX_FILE_SIZE:
                remaining_space = MAX_FILE_SIZE - current_size
                if remaining_space > 0:
                    trimmed_content = encoded_content[:remaining_space]
                    trimmed_str = trimmed_content.decode('utf-8', errors='ignore')
                    with open(content_file, 'a', encoding='utf-8') as cf:
                        cf.write(trimmed_str)
                    logging.info(f"Trimmed content from URL {url} to fit size limit.")
                else:
                    logging.info(f"Skipping writing content from URL {url} due to size limit.")
                stop_processing = True
                break
            else:
                with open(content_file, 'a', encoding='utf-8') as cf:
                    cf.write(content_str)

        browser.close()

    if failed_urls:
        with open(failed_file, 'w', encoding='utf-8') as ff:
            ff.write("\n".join(failed_urls))

    print(f"[OK] Finished for {country_code}, Keyword: {keyword} : {headings_file}, {content_file}")


# ── Main ──
def main():
    keyword, country, site = read_current_keyword()
    global log
    log = setup_logging(keyword, country, site)


    global SESSION
    SESSION = build_session()


    process_keyword(keyword, country, site)




if __name__ == "__main__":
    main()
