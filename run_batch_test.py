import subprocess
import csv
import os
import time

KEYWORDS_FILE = "input/keywords.csv"
CURRENT_KEYWORD_FILE = "config/current_keyword.csv"
DEFAULT_SITE = "https://luggageadvisor.uk"

SCRIPTS = [
    # "scrape_urls.py",
    # "generate_headings.py",
    # "amend_product_names.py",
    # "get_response_from_openai.py",
    "insert_amazon_links_images.py",
    "final_tidy_up.py",
    "title_description.py",
    "upload_wordpress.py"


 ]


def read_keywords():
    """
    Read keywords from CSV.

    New format: keyword,country,site,category
    Backwards compatible with:
      - keyword,country,site
      - keyword,country

    Always returns list of 4-tuples: (keyword, country, site, category)
    """
    keywords = []
    try:
        with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            _ = next(reader, None)  # skip header if present

            for row in reader:
                if not row:
                    continue

                # New format: keyword,country,site,category
                if len(row) >= 4:
                    kw = row[0].strip()
                    country = row[1].strip().upper()
                    site = (row[2].strip() or DEFAULT_SITE)
                    category = row[3].strip()
                    keywords.append((kw, country, site, category))

                # Old: keyword,country,site
                elif len(row) >= 3:
                    kw = row[0].strip()
                    country = row[1].strip().upper()
                    site = (row[2].strip() or DEFAULT_SITE)
                    category = ""  # no category provided
                    keywords.append((kw, country, site, category))

                # Very old: keyword,country
                elif len(row) >= 2:
                    kw = row[0].strip()
                    country = row[1].strip().upper()
                    site = DEFAULT_SITE
                    category = ""
                    keywords.append((kw, country, site, category))

    except Exception as e:
        print(f"[ERROR] Could not read {KEYWORDS_FILE}: {e}")
    return keywords


def write_current_keyword(keyword, country, site, category):
    """
    Write current keyword context to CURRENT_KEYWORD_FILE.

    Format: keyword,country,site,category
    """
    try:
        os.makedirs(os.path.dirname(CURRENT_KEYWORD_FILE), exist_ok=True)
        with open(CURRENT_KEYWORD_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([keyword, country, site, category])
    except Exception as e:
        print(f"[ERROR] Failed to write current keyword: {e}")


def run_script(script):
    print(f"➡️ Running {script}...")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ {script} completed.\n")
    else:
        print(f"❌ {script} failed.")
        print(f"--- STDERR ---\n{result.stderr}\n")


def main():
    all_keywords = read_keywords()
    if not all_keywords:
        print("No keywords found. Exiting.")
        return

    for idx, (keyword, country, site, category) in enumerate(all_keywords, start=1):
        print(f"\n🔵 [{idx}/{len(all_keywords)}] Processing: {keyword} ({country}) @ {site} [{category}]")
        write_current_keyword(keyword, country, site, category)

        for script in SCRIPTS:
            run_script(script)

        print(f"✅ Finished processing: {keyword} ({country}) @ {site} [{category}]\n")
        time.sleep(1)  # optional delay between keywords

    print("🏁 All keywords processed.")


if __name__ == "__main__":
    main()
