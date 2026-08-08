# bing_scraper.py — proxy+headless handling, resilient consent, SERP guards,
# token-preserving pagination (Next → pager href → deterministic URL)


import os
import re
import time
import json
import random
import logging
import shutil
import subprocess
import tempfile
import urllib.parse
from contextlib import contextmanager
from datetime import datetime
from urllib.parse import urlparse, parse_qs, unquote

if os.name == "nt":
    import msvcrt
else:
    import fcntl

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    InvalidSessionIdException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

from url_utils import normalize_url_any

COUNTRY_CANON = {"US", "UK", "CA"}

def normalize_country(c: str) -> str:
    if not c:
        return "UK"
    c = c.strip().upper()
    c = c.split(",")[0]
    c = re.sub(r"[^A-Z]", "", c)
    return c if c in COUNTRY_CANON else "UK"

log = logging.getLogger("scraper")

# Default number of results per Bing SERP page we request.
# Keep as int; convert to str only when building URLs.
DEFAULT_COUNT = 20

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edg/124.0.2478.67",
]


def _parse_chrome_major(version_text: str):
    match = re.search(r"\b(\d+)\.\d+\.\d+\.\d+\b", version_text or "")
    if not match:
        match = re.search(r"\b(\d{2,3})\b", version_text or "")
    return int(match.group(1)) if match else None


def _detect_chrome_major_version():
    """Return the installed Chrome major version when it can be detected."""
    for env_name in ("CHROME_MAJOR_VERSION", "UC_CHROME_VERSION_MAIN"):
        major = _parse_chrome_major(os.getenv(env_name, ""))
        if major:
            return major

    if os.name == "nt":
        try:
            import winreg

            registry_locations = [
                (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Google\Chrome\BLBeacon"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Google\Chrome\BLBeacon"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Google\Chrome\BLBeacon"),
            ]
            for hive, key_path in registry_locations:
                try:
                    with winreg.OpenKey(hive, key_path) as key:
                        version, _ = winreg.QueryValueEx(key, "version")
                    major = _parse_chrome_major(str(version))
                    if major:
                        return major
                except OSError:
                    continue
        except Exception:
            pass

    candidate_paths = [
        shutil.which("chrome"),
        shutil.which("chrome.exe"),
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
    ]
    if os.name == "nt":
        candidate_paths.extend(
            [
                os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
                os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
                os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
            ]
        )

    seen = set()
    for path in candidate_paths:
        if not path or path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        try:
            output = subprocess.check_output(
                [path, "--version"],
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5,
            )
            major = _parse_chrome_major(output)
            if major:
                return major
        except Exception:
            continue

    return None




UC_STARTUP_LOCK_NAME = "uc_chrome_startup.lock"
UC_STARTUP_LOCK_TIMEOUT = 90
UC_STARTUP_RETRIES = 3


@contextmanager
def _file_lock(lock_path: str, timeout: float = UC_STARTUP_LOCK_TIMEOUT):
    """
    Cross-process lock used to serialize undetected_chromedriver startup.

    Why:
        uc.Chrome(...) may patch/copy/use shared temp artifacts during launch.
        If two Python processes do that at the same time on Windows, one can fail
        with a "file is in use" style error even though Chrome itself supports
        multiple concurrent browser processes.

    Scope:
        Hold this lock only around uc.Chrome(...) creation, not for the whole
        scraping session, so both scrapers can still run in parallel after
        startup completes.
    """
    lock_dir = os.path.dirname(lock_path) or tempfile.gettempdir()
    os.makedirs(lock_dir, exist_ok=True)

    with open(lock_path, "a+b") as fh:
        start = time.time()
        locked = False

        while True:
            try:
                fh.seek(0)
                if os.name == "nt":
                    if os.path.getsize(lock_path) < 1:
                        fh.write(b"0")
                        fh.flush()
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                locked = True
                break

            except OSError:
                if (time.time() - start) >= float(timeout):
                    raise TimeoutError(f"Timed out waiting for Chrome startup lock: {lock_path}")
                time.sleep(0.25)

        try:
            yield
        finally:
            if locked:
                try:
                    fh.seek(0)
                    if os.name == "nt":
                        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass


def _chrome_startup_lock_path() -> str:
    return os.path.join(tempfile.gettempdir(), UC_STARTUP_LOCK_NAME)


def _safe_screenshot(driver, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        driver.save_screenshot(path)
        log.debug(f"📸 Saved screenshot: {path}")
    except Exception as e:
        log.debug(f"Screenshot failed: {e}")


def _dump_html(driver, path):
    try:
        html = driver.page_source or ""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        log.debug(f"🧾 Saved HTML: {path}")
    except Exception as e:
        log.debug(f"HTML dump failed: {e}")


class BingScraper:
    def __init__(self, country_settings):
        self.country_settings = country_settings
        self.driver = None
        self.current_country = None
        self._profile_dir = None
        self._headless = True
        self._last_first_h2 = None
        self._serp_tokens = {}   # snapshot of stable SERP tokens
        self._active_terms = set()

        # ── navigation pacing / backoff (no proxy) ──────────────────────────
        self._min_nav_delay = 0.9   # base delay between navigations
        self._max_backoff = 8.0     # cap for adaptive backoff
        self._backoff = 0.0
        self._last_nav_ts = 0.0

    def _nav_sleep(self, extra: float = 0.0) -> None:
        """Rate-limit navigations with jitter and adaptive backoff."""
        try:
            now = time.time()
            since = now - (self._last_nav_ts or 0.0)
            target = (self._min_nav_delay + float(extra) + float(self._backoff))
            wait = max(0.0, target - since)
            wait += random.uniform(0.05, 0.45)  # jitter
            if wait > 0:
                time.sleep(wait)
        finally:
            self._last_nav_ts = time.time()

    def _adjust_backoff_after_nav(self) -> None:
        """Increase backoff when challenged; decay it on success."""
        try:
            challenged = self._detect_challenge()
        except Exception:
            challenged = False

        if challenged:
            self._backoff = min(self._max_backoff, (self._backoff * 1.6) + 1.0)
        else:
            self._backoff = max(0.0, (self._backoff * 0.75) - 0.15)

    def _get(self, url: str, extra_delay: float = 0.0) -> None:
        """Rate-limited driver.get wrapper."""
        self._nav_sleep(extra=extra_delay)
        self.driver.get(url)
        try:
            time.sleep(random.uniform(0.1, 0.35))
        except Exception:
            pass
        self._adjust_backoff_after_nav()

# ── init helpers ───────────────────────────────────────────────────────────
    def _normalize_country(self, c: str) -> str:
        if not c:
            return "UK"
        c = c.strip().upper().split(",")[0]
        c = re.sub(r"[^A-Z]", "", c)
        return c if c in self.country_settings else "UK"

    def _derive_lang_from_mkt(self, mkt: str) -> str:
        return mkt if mkt else "en-GB"

    def _prime_bing_locale(self, country: str):
        """Set mkt/lang/cookies and Accept-Language so results stick to en-*"""
        try:
            c = normalize_country(country or self.current_country)
            if c == "UK":
                mkt, cc = "en-GB", "GB"
            elif c == "US":
                mkt, cc = "en-US", "US"
            elif c == "CA":
                mkt, cc = "en-CA", "CA"
            else:
                mkt, cc = "en-GB", "GB"

            self._get(f"https://www.bing.com/setprefs?mkt={mkt}&setlang={mkt}&cc={cc}")
            self._enforce_single_tab()          # ✅ add
            time.sleep(0.6)


            for ck in [
                {"name": "_EDGE_S", "value": f"mkt={mkt}&ui={mkt}", "domain": ".bing.com", "path": "/"},
                {"name": "_SS", "value": "SID=00", "domain": ".bing.com", "path": "/"},
            ]:
                try:
                    self.driver.add_cookie(ck)
                except Exception:
                    pass

            try:
                self.driver.execute_cdp_cmd("Network.enable", {})
                self.driver.execute_cdp_cmd("Network.setExtraHTTPHeaders", {
                    "headers": {"Accept-Language": mkt}
                })
            except Exception:
                pass
        except Exception:
            pass

    def setup_browser(self, country, headless=True):
        self.current_country = normalize_country(country)
        self._headless = headless
        settings = self.country_settings.get(self.current_country, {})
        mkt = settings.get("mkt", "en-GB")
        lang = self._derive_lang_from_mkt(mkt)

        if self.driver is not None:
            try:
                self.quit()
            except Exception:
                pass

        self._profile_dir = tempfile.mkdtemp(prefix="uc_profile_")

        def build_options():
            options = uc.ChromeOptions()
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument(f"--lang={lang}")
            options.add_argument(f"--user-agent={random.choice(USER_AGENTS)}")
            options.add_argument(f"--user-data-dir={self._profile_dir}")
            options.add_argument("--disable-notifications")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--no-first-run")
            options.add_argument("--no-default-browser-check")
            options.add_argument("--disable-background-networking")
            options.add_argument("--disable-features=OptimizationGuideModelDownloading")

            if headless:
                options.add_argument("--headless=new")

            return options

        startup_lock = _chrome_startup_lock_path()
        last_error = None
        chrome_major_version = _detect_chrome_major_version()
        if chrome_major_version:
            log.info(f"Detected Chrome major version {chrome_major_version}; using matching ChromeDriver")

        for attempt in range(1, UC_STARTUP_RETRIES + 1):
            try:
                with _file_lock(startup_lock, timeout=UC_STARTUP_LOCK_TIMEOUT):
                    log.info(
                        f"🧭 Starting Chrome (attempt {attempt}/{UC_STARTUP_RETRIES}) "
                        f"with isolated profile {self._profile_dir}"
                    )
                    chrome_kwargs = {
                        "options": build_options(),
                        "use_subprocess": True,
                    }
                    if chrome_major_version:
                        chrome_kwargs["version_main"] = chrome_major_version
                    self.driver = uc.Chrome(**chrome_kwargs)

                self.driver.set_page_load_timeout(45)
                log.info(f"🧭 Stealth browser started for {country}")
                break

            except Exception as e:
                last_error = e
                detected_from_error = _parse_chrome_major(str(e))
                if detected_from_error and detected_from_error != chrome_major_version:
                    chrome_major_version = detected_from_error
                    log.info(
                        f"Retrying Chrome startup with detected browser major version {chrome_major_version}"
                    )
                log.warning(
                    f"⚠️ Chrome startup failed on attempt {attempt}/{UC_STARTUP_RETRIES}: {e}"
                )

                try:
                    if self.driver is not None:
                        self.driver.quit()
                except Exception:
                    pass
                finally:
                    self.driver = None

                if attempt >= UC_STARTUP_RETRIES:
                    raise

                time.sleep(min(6.0, 1.5 * attempt))

        if self.driver is None:
            raise RuntimeError(f"Failed to start Chrome driver: {last_error}")

        # Prime Bing locale early
        self._prime_bing_locale(country)

        # Prewarm Bing (best-effort)
        try:
            prewarm_url = f"https://www.bing.com/?mkt={mkt}&setlang={mkt}"
            self._get(prewarm_url)
            self._enforce_single_tab()
            self._post_nav_humanize()

            try:
                self.driver.add_cookie({"name": "_EDGE_S", "value": f"mkt={mkt}&ui={mkt}", "domain": ".bing.com", "path": "/"})
                self.driver.add_cookie({"name": "_SS", "value": "SID=00", "domain": ".bing.com", "path": "/"})
            except Exception:
                pass
        except Exception as e:
            log.debug(f"Prewarm skipped: {e}")

    def quit(self):
        try:
            if self.driver:
                self.driver.quit()
        finally:
            self.driver = None
            if self._profile_dir and os.path.isdir(self._profile_dir):
                try:
                    shutil.rmtree(self._profile_dir, ignore_errors=True)
                except Exception:
                    pass
            self._profile_dir = None
            log.info("🛑 Browser session ended")

    @staticmethod
    def _is_dead_session_error(exc: Exception) -> bool:
        """Return True when Chrome/ChromeDriver has permanently lost the session."""
        if isinstance(exc, InvalidSessionIdException):
            return True
        if not isinstance(exc, WebDriverException):
            return False

        message = str(exc).lower()
        dead_session_markers = (
            "invalid session id",
            "session deleted because of page crash",
            "chrome not reachable",
            "disconnected: not connected to devtools",
            "disconnected: unable to receive message from renderer",
            "target window already closed",
            "no such window",
        )
        return any(marker in message for marker in dead_session_markers)

    def _restart_dead_session(self, country) -> None:
        """Discard an unusable driver and start a clean browser session."""
        log.warning("♻️ Chrome session was lost; starting a fresh browser session.")
        try:
            self.quit()
        except Exception:
            # A dead ChromeDriver commonly raises again during quit().
            self.driver = None

        self.setup_browser(country, headless=self._headless)
    # ── query helpers ──────────────────────────────────────────────────────────
    def enhance_keyword(self, keyword, country: str = None):
        c = normalize_country(country or self.current_country)
        if c == "UK":
            enhanced = f"{keyword} UK"
        elif c == "CA":
            enhanced = f"{keyword} Canada"
        elif c == "US":
            enhanced = f"{keyword} USA"
        else:
            enhanced = keyword
        log.info(f"🔎 Enhanced query → {enhanced!r} (country={c})")
        return enhanced

    def _add_unstick_params(self, url: str) -> str:
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}sp=-1"

    def _force_submit_query(self, query: str) -> None:
        """Type the query into the Bing search box and submit, to override silent rewrites."""
        try:
            box = WebDriverWait(self.driver, 8).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#sb_form_q"))
            )
            current = (box.get_attribute("value") or "").strip()
            if not current or current.lower() != query.lower():
                box.clear()
                box.send_keys(query)
                box.send_keys(Keys.ENTER)
        except Exception:
            pass

    def _country_query_clauses(self, country: str) -> str:
        c = normalize_country(country or self.current_country)
        if c == "UK":
            return "site:.uk language:en"
        if c == "US":
            return "site:.com language:en"
        if c == "CA":
            return "site:.ca language:en"
        return "language:en"

    # ── humanize / anti-bot helpers ───────────────────────────────────────────
    def _post_nav_humanize(self):
        """Small, randomized actions that reduce bot-likeness."""
        try:
            time.sleep(random.uniform(0.35, 0.9))
            y = random.randint(120, 320)
            self.driver.execute_script(f"window.scrollTo(0, {y});")
            time.sleep(random.uniform(0.15, 0.4))
            self.driver.execute_script("window.scrollTo(0, 0);")
        except Exception:
            pass
            
    def _enforce_single_tab(self):
        """Close any extra tabs/windows and switch back to the first handle."""
        try:
            handles = self.driver.window_handles
            if not handles or len(handles) <= 1:
                return
            main = handles[0]

            # Close everything except the first tab
            for h in handles[1:]:
                try:
                    self.driver.switch_to.window(h)
                    self.driver.close()
                except Exception:
                    pass

            self.driver.switch_to.window(main)
            log.debug(f"🧭 Closed extra tabs; current URL: {self.driver.current_url}")
        except Exception:
            pass


    # ── consent & interstitials ───────────────────────────────────────────────
    def _try_click(self, by, sel, timeout=3):
        try:
            el = WebDriverWait(self.driver, timeout).until(EC.element_to_be_clickable((by, sel)))
            tag = (el.tag_name or "").lower()
            role = (el.get_attribute("role") or "").lower()
            if tag == "a" and role != "button":
                return False
            # avoid clicking inside SERP containers
            try:
                p = el
                while p is not None:
                    pid = (p.get_attribute("id") or "").lower()
                    if pid in ("b_results", "b_content"):
                        return False
                    p = p.find_element(By.XPATH, "..")
            except Exception:
                pass
            el.click()
            return True
        except Exception:
            return False

    def _try_all_consent_selectors(self):
        XPATHS = [
            "//button[contains(., 'Accept')]",
            "//button[contains(., 'Agree')]",
            "//input[@value='Accept' or @value='Agree']",
            "//a[@role='button' and (contains(., 'Accept') or contains(., 'Agree'))]",
            "//*[@data-privacy='accept']",
        ]
        CSS = [
            "#bnp_btn_accept",
            "button[aria-label*='Accept' i]",
            "button[aria-label*='Agree' i]",
            "button#bnp_btn_accept",
            ".bnp_btn_accept",
        ]
        for xp in XPATHS:
            if self._try_click(By.XPATH, xp, timeout=2):
                return True
        for cs in CSS:
            if self._try_click(By.CSS_SELECTOR, cs, timeout=2):
                return True

        # iframe variants
        try:
            iframes = self.driver.find_elements(By.TAG_NAME, "iframe")
            for frame in iframes:
                src = (frame.get_attribute("src") or "").lower()
                if any(k in src for k in ["consent", "privacy", "cookie"]):
                    self.driver.switch_to.frame(frame)
                    for xp in XPATHS:
                        if self._try_click(By.XPATH, xp, timeout=2):
                            self.driver.switch_to.default_content()
                            return True
                    for cs in CSS:
                        if self._try_click(By.CSS_SELECTOR, cs, timeout=2):
                            self.driver.switch_to.default_content()
                            return True
                    self.driver.switch_to.default_content()
        except Exception:
            try:
                self.driver.switch_to.default_content()
            except Exception:
                pass

        # JS last resort (safe)
        try:
            clicked = self.driver.execute_script("""
                function isInSerp(el){
                    for (let n=el; n; n=n.parentElement){
                        if (n.id === 'b_results' || n.id === 'b_content') return true;
                    }
                    return false;
                }
                const texts = ['accept','agree'];
                const nodes = Array.from(document.querySelectorAll(
                    'button, input[type=button], input[type=submit], [role=button]'
                ));
                for (const el of nodes){
                    if (isInSerp(el)) continue;
                    if (el.offsetParent === null) continue;
                    const t = (el.innerText || el.value || '').toLowerCase();
                    if (texts.some(x => t.includes(x))) { el.click(); return true; }
                }
                return false;
            """)
            if clicked:
                return True
        except Exception:
            pass
        return False

    def _force_remove_consent(self):
        try:
            self.driver.execute_script("""
                // Remove only known cookie/consent containers. Don't nuke all fixed nodes.
                const sels = [
                  '#bnp_container', '#bnp_bg', '#bnp_btn_accept', '.bnp_btn_accept',
                  '.bnp_desc_left', '.bnp_btn_wrapper',
                  '[id*="consent"]', '[class*="consent"]',
                  '[id*="cookie"]',  '[class*="cookie"]'
                ];
                sels.forEach(sel => {
                  document.querySelectorAll(sel).forEach(el => el.remove());
                });

                // Some sites put the dialog in an iframe
                document.querySelectorAll('iframe').forEach(f => {
                  try {
                    const src = (f.getAttribute('src') || '').toLowerCase();
                    if (src.includes('consent') || src.includes('cookie')) f.remove();
                  } catch(e) {}
                });
            """)
            log.debug("🍪 Cookie/consent elements forcefully removed via JS")
        except Exception as e:
            log.debug(f"⚠️ JS removal failed: {e}")

    def dismiss_cookie_banner(self):
        try:
            self.driver.execute_script("window.scrollTo(0, 120);")
        except Exception:
            pass
        if self._try_all_consent_selectors():
            log.debug("🍪 Cookie banner dismissed via selector click")
            return
        log.debug("⚠️ Consent click selectors failed — attempting JS removal")
        self._force_remove_consent()
        time.sleep(0.4)

    # ── human verification detection & recovery ───────────────────────────────
    def _detect_challenge(self) -> bool:
        """Detect Bing/Edge interstitials like 'Please solve the challenge'."""
        try:
            html = (self.driver.page_source or "").lower()
            needles = [
                "please solve the challenge", "verify you are human",
                "are you a robot", "unusual traffic", "to continue, please"
            ]
            if any(n in html for n in needles):
                return True
            sels = [
                "#b_captcha", ".b_captcha",
                "iframe[src*='captcha']", "iframe[title*='challenge']",
                "div[aria-label*='verification']",
                "div[role='dialog'][aria-label*='challenge']",
            ]
            for s in sels:
                try:
                    el = self.driver.find_element(By.CSS_SELECTOR, s)
                    if el and el.is_displayed():
                        return True
                except Exception:
                    pass
        except Exception:
            pass
        return False

    def _maybe_solve_simple_challenge(self, timeout=10) -> bool:
        """
        Try only 'safe' interactions (e.g., a visible checkbox). If it escalates
        to a visual puzzle, we return False and let the caller rotate identity.
        """
        end = time.time() + timeout
        while time.time() < end:
            try:
                # checkbox in iframe
                frames = self.driver.find_elements(By.TAG_NAME, "iframe")
                for fr in frames:
                    try:
                        title = (fr.get_attribute("title") or "").lower()
                        src   = (fr.get_attribute("src") or "").lower()
                        if any(k in (title + " " + src) for k in ["challenge", "captcha", "verify"]):
                            self.driver.switch_to.frame(fr)
                            for sel in ["input[type='checkbox']", "div[role='checkbox']", "label[for]"]:
                                try:
                                    el = self.driver.find_element(By.CSS_SELECTOR, sel)
                                    if el.is_displayed():
                                        try: el.click()
                                        except Exception:
                                            self.driver.execute_script("arguments[0].click();", el)
                                        time.sleep(1.0)
                                        self.driver.switch_to.default_content()
                                        return not self._detect_challenge()
                                except Exception:
                                    pass
                            self.driver.switch_to.default_content()
                    except Exception:
                        try: self.driver.switch_to.default_content()
                        except Exception: pass

                # non-iframe simple button
                try:
                    el = self.driver.find_element(By.XPATH, "//button[contains(., 'human') or contains(., 'Continue')]")
                    if el.is_displayed():
                        try: el.click()
                        except Exception:
                            self.driver.execute_script("arguments[0].click();", el)
                        time.sleep(1.0)
                        return not self._detect_challenge()
                except Exception:
                    pass
            except Exception:
                pass
            time.sleep(0.4)
        return False

    def _recover_from_challenge(self, country: str, query: str, resume_url: str = "") -> bool:
        """Rotate identity, re-prime Bing, and resume from a specific URL if provided."""
        log.warning("🔄 Rotating identity after challenge...")
        try:
            self.quit()
            time.sleep(1.0)

            # reopen headed to reduce bot pressure
            self.setup_browser(country, headless=False)

            try:
                w = random.randint(1280, 1680); h = random.randint(800, 1000)
                self.driver.set_window_size(w, h)
            except Exception:
                pass

            self._prime_bing_locale(country)

            # If we know the URL we were trying to load, go straight there
            if resume_url:
                self._get(resume_url)
                self._enforce_single_tab() 
                self._post_nav_humanize()
                self.dismiss_cookie_banner()
                if self._detect_challenge():
                    return False
                return True

            # Fallback: reload Bing and submit query
            self._get("https://www.bing.com/")
            self._enforce_single_tab() 
            self._post_nav_humanize()
            self.dismiss_cookie_banner()

            self._force_submit_query(query)
            self._post_nav_humanize()

            return not self._detect_challenge()
        except Exception as e:
            log.warning(f"⚠️ Challenge recovery failed: {e}")
            return False


    # ── safety nets / page helpers ─────────────────────────────────────────────
    def _ensure_on_bing(self, expected_url: str):
        try:
            u = urlparse(self.driver.current_url)
            on_bing = u.netloc.endswith("bing.com") and (u.path in ("/search", "/"))
            if not on_bing:
                log.warning(f"⚠️ Navigated off Bing to {self.driver.current_url}; restoring SERP")
                self.driver.back()
                time.sleep(0.6)
                u2 = urlparse(self.driver.current_url)
                if not (u2.netloc.endswith("bing.com") and (u2.path in ("/search", "/"))):
                    self._get(expected_url)
                    self._enforce_single_tab()
                    self._post_nav_humanize()
        except Exception:
            pass

            

    def _is_bing_home(self) -> bool:
        try:
            u = urlparse(self.driver.current_url)
            if not u.netloc.endswith("bing.com"):
                return False
            if u.path != "/":
                return False
            html = (self.driver.page_source or "").lower()
            return ('id="b_results"' not in html) and ('class="b_algo"' not in html)
        except Exception:
            return False

    def _ensure_serp_or_resubmit(self, query: str, mkt: str, cc: str) -> None:
        try:
            u = urlparse(self.driver.current_url)
            on_search = u.netloc.endswith("bing.com") and u.path == "/search"
            if on_search:
                return

            if self._is_bing_home() or (u.netloc.endswith("bing.com") and u.path == "/"):
                log.warning("⚠️ Landed on Bing homepage; reloading /search and resubmitting query")
                base = "https://www.bing.com/search"
                params = {
                    "q": query,
                    "mkt": mkt,
                    "setmkt": mkt,
                    "setlang": mkt,
                    "cc": cc,
                    "count": str(DEFAULT_COUNT),
                }
                self._get(f"{base}?{urllib.parse.urlencode(params)}")
                self._enforce_single_tab()
                self._post_nav_humanize()
                time.sleep(0.6)
                self.dismiss_cookie_banner()
                self._force_submit_query(query)
        except Exception:
            pass


    def _is_valid_serp(self, query_first_word: str) -> bool:
        try:
            html = (self.driver.page_source or "").lower()
            has_container = ('id="b_results"' in html) or ('id="b_content"' in html)
            has_algo = ('class="b_algo"' in html)
            return has_container and has_algo and (query_first_word.lower() in html)
        except Exception:
            return False

    def _wait_query_reflected(self, query: str, timeout: float = 12.0) -> bool:
        """
        Wait until the current query appears to be reflected in the page:
        - Search box (#sb_form_q) value matches or contains the query, OR
        - At least one organic card contains any term from the query (title/snippet).
        """
        end = time.time() + timeout
        q = (query or "").strip().lower()
        terms = [t for t in re.split(r"\s+", q) if t and len(t) > 2]

        while time.time() < end:
            try:
                try:
                    box = self.driver.find_element(By.CSS_SELECTOR, "#sb_form_q")
                    val = (box.get_attribute("value") or "").strip().lower()
                    if val and (q in val or val in q):
                        return True
                except Exception:
                    pass

                cards = self.extract_result_cards()
                if cards:
                    for c in cards:
                        try:
                            h2 = c.find_element(By.CSS_SELECTOR, "h2").text.lower()
                        except Exception:
                            h2 = ""
                        snip = ""
                        for sel in [".b_caption p", ".b_snippet", "p"]:
                            try:
                                for el in c.find_elements(By.CSS_SELECTOR, sel):
                                    t = (el.text or "").strip().lower()
                                    if t:
                                        snip = t
                                        break
                                if snip:
                                    break
                            except Exception:
                                pass
                        blob = (h2 + " " + snip).strip()
                        if blob and any(t in blob for t in terms):
                            return True
            except Exception:
                pass
            time.sleep(0.3)
        return False

    def _current_page_number(self) -> int:
        try:
            el = self.driver.find_element(By.CSS_SELECTOR, 'a[aria-current="page"]')
            t = (el.text or "").strip()
            if t.isdigit():
                return int(t)
        except Exception:
            pass

        for sel in ["span.sb_pagS", "li.sb_pagS", "a.sb_pagS", "strong.sb_pagS", "li.b_pag a.sb_pagS"]:
            try:
                el = self.driver.find_element(By.CSS_SELECTOR, sel)
                t = (el.text or "").strip()
                if t.isdigit():
                    return int(t)
            except Exception:
                pass

        # Fallback: derive from URL `first` AND `count` (count-aware pagination)
        try:
            u = urlparse(self.driver.current_url)
            q = parse_qs(u.query)

            first_s = (q.get("first", [""])[0] or "").strip()
            count_s = (q.get("count", [""])[0] or "").strip()

            first = int(first_s) if first_s.isdigit() else None
            count = int(count_s) if count_s.isdigit() else None

            if not count or count <= 0:
                count = DEFAULT_COUNT


            if first and first >= 1:
                return ((first - 1) // count) + 1
        except Exception:
            pass

        return 1



    # ── SERP token snapshot & URL builder ─────────────────────────────────────
    def _snapshot_serp_tokens(self):
        """
        Capture query tokens Bing expects to persist across pages.

        Fix:
          - Keep additional tokens seen on Bing SERPs (cvid, FORM, pq, sc, qs, sk, lq, FPIG, etc.)
          - This improves deterministic pagination when 'Next' click fails.
        """
        try:
            u = urlparse(self.driver.current_url)
            q = parse_qs(u.query)

            # Tokens that Bing frequently uses to keep pagination stable
            keep = {
                "q", "mkt", "setlang", "setmkt", "cc", "safeSearch", "count",
                "cvid", "form", "FORM",
                "pq", "sc", "qs", "sk", "lq", "sp",
                "FPIG", "filt", "filters", "ensearch", "qft", "toWww"
            }

            tokens = {}
            for k, v in q.items():
                if k in keep and v:
                    tokens[k] = v[-1]

            if "count" not in tokens:
                tokens["count"] = str(DEFAULT_COUNT)


            # 'first' changes each page; don't keep it
            tokens.pop("first", None)

            self._serp_tokens = tokens
            log.debug(f"🧭 Snapshotted SERP tokens: {self._serp_tokens}")
        except Exception as e:
            log.debug(f"Token snapshot failed: {e}")
            self._serp_tokens = {}

    def _build_page_url(self, base_url: str, page_idx: int) -> str:
        try:
            base = "https://www.bing.com/search"
            snap = dict(self._serp_tokens or {})
            u = urlparse(base_url)
            q = parse_qs(u.query)

            mkt_val = snap.get("mkt", q.get("mkt", ["en-GB"])[0])
            setlang_val = snap.get("setlang", q.get("setlang", [mkt_val])[0])

            params = {
                "q": snap.get("q", q.get("q", [""])[0]),
                "mkt": mkt_val,
                "setmkt": mkt_val,
                "setlang": setlang_val,
                "cc": snap.get("cc", q.get("cc", ["GB"])[0]),
                "safeSearch": snap.get("safeSearch", q.get("safeSearch", ["moderate"])[0]),
                "count": snap.get("count", q.get("count", [str(DEFAULT_COUNT)])[0]),
                "ensearch": "1",
                "qft": "+filterui:language-en",
                "toWww": "1",
                "FORM": "PERE",
                "sp": str(page_idx),
            }

            # ✅ count-aware page step (if count=20, first increments by 20)
            count_str = params.get("count", str(DEFAULT_COUNT))
            try:
                count = int(str(count_str).strip())
                if count <= 0:
                    count = 10
            except Exception:
                count = 10

            params["first"] = str((page_idx - 1) * count + 1)
            return f"{base}?{urllib.parse.urlencode(params)}"

        except Exception:
            u = urlparse(base_url)
            q = parse_qs(u.query)

            # Minimal fallback: still count-aware if URL has count
            try:
                count = int((q.get("count", [str(DEFAULT_COUNT)])[0] or str(DEFAULT_COUNT)).strip())
                if count <= 0:
                    count = 10
            except Exception:
                count = 10

            q["first"] = [str((page_idx - 1) * count + 1)]
            q.setdefault("count", [str(count)])

            return f"https://www.bing.com/search?{urllib.parse.urlencode({k: v[-1] for k, v in q.items()})}"

    # ── pager / infinite-scroll helpers ────────────────────────────────────────
    def _get_pager_href(self, page_idx: int) -> str:
        sels = [f'a[aria-label="Page {page_idx}"]', f'a[href][data-page="{page_idx}"]']
        try:
            containers = self.driver.find_elements(By.CSS_SELECTOR, "nav[role='navigation'], .sb_pag, #b_results .b_pag")
        except Exception:
            containers = []
        for cont in containers:
            for sel in sels:
                try:
                    el = cont.find_element(By.CSS_SELECTOR, sel)
                    href = (el.get_attribute("href") or "").strip()
                    if href:
                        return href
                except Exception:
                    pass
            try:
                for a in cont.find_elements(By.CSS_SELECTOR, "a[href]"):
                    if (a.text or "").strip() == str(page_idx):
                        href = (a.get_attribute("href") or "").strip()
                        if href:
                            return href
            except Exception:
                pass
        try:
            for a in self.driver.find_elements(By.CSS_SELECTOR, "a[href]"):
                if (a.get_attribute("aria-label") or "").strip() == f"Page {page_idx}":
                    href = (a.get_attribute("href") or "").strip()
                    if href:
                        return href
        except Exception:
            pass
        return ""

    def _goto_page_via_pager(self, page_idx: int) -> bool:
        href = self._get_pager_href(page_idx)
        if not href:
            return False
        try:
            self._get(href)
            self._enforce_single_tab()          # ✅ add
            self._post_nav_humanize()
            time.sleep(random.uniform(1.1, 1.9))
            self.dismiss_cookie_banner()
            self._ensure_on_bing(href)
            return True
        except Exception:
            return False

    def _find_more_results_button(self):
        sels = [
            "a[aria-label='More results']",
            "a[role='button'][aria-label^='More results']",
            "a[role='button'][class*='more']",
            "a[role='button'][class*='moreresult']",
            ".b_pag a[role='button']",
        ]
        for s in sels:
            try:
                el = self.driver.find_element(By.CSS_SELECTOR, s)
                if el and el.is_displayed():
                    return el
            except Exception:
                pass
        try:
            return self.driver.find_element(By.XPATH, "//a[@role='button' and contains(., 'More results')]")
        except Exception:
            return None

    def _force_scroll_to_bottom(self):
        try:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        except Exception:
            pass

    def _click_more_results_once(self, timeout=10) -> bool:
        try:
            before = len(self.driver.find_elements(By.CSS_SELECTOR, "li.b_algo, div#b_content li.b_algo, div.b_algo"))
            self._force_scroll_to_bottom()
            btn = self._find_more_results_button()
            if not btn:
                return False
            try:
                self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
                time.sleep(0.2)
                try:
                    btn.click()
                except Exception:
                    self.driver.execute_script("arguments[0].click();", btn)
            except Exception:
                return False
            end = time.time() + timeout
            while time.time() < end:
                time.sleep(0.4)
                self.dismiss_cookie_banner()
                after = len(self.driver.find_elements(By.CSS_SELECTOR, "li.b_algo, div#b_content li.b_algo, div.b_algo"))
                if after > before:
                    log.debug(f"🧩 Infinite-scroll: cards {before} → {after}")
                    return True
            return False
        except Exception:
            return False

    def _advance_infinite_scroll_pages(self, target_page_idx: int) -> bool:
        desired = max(0, target_page_idx - 1)
        progressed = False
        for _ in range(desired):
            if not self._click_more_results_once():
                break
            progressed = True
        return progressed

    # ── parsing ────────────────────────────────────────────────────────────────
    def _parse_cards_from_html(self, html):
        soup = BeautifulSoup(html, "html.parser")
        blocks = soup.select("li.b_algo, div#b_content li.b_algo, div.b_algo")
        results = []
        for b in blocks:
            a = b.select_one("h2 > a") or b.select_one("a[href]")
            if not a:
                continue
            raw_url = (a.get("href") or "").strip()
            if not raw_url:
                continue
            url = normalize_url_any(raw_url) or normalize_url_any(unquote(raw_url)) or raw_url
            if (not url) or self.is_low_value_url(url):
                continue
            title = (a.get_text(" ", strip=True) or "").strip()
            desc = ""
            cap = b.select_one(".b_caption p") or b.select_one(".b_snippet") or b.select_one("p")
            if cap:
                desc = cap.get_text(" ", strip=True)

            # relevance guard client-side
            blob = (title + " " + desc).lower()
            if getattr(self, "_active_terms", None) and not any(t in blob for t in self._active_terms):
                continue

            results.append({"url": url, "url_raw": raw_url, "title": title, "description": desc})
        return results

    def extract_result_cards(self):
        try:
            containers = []
            for sel in ["#b_results", "#b_content #b_results"]:
                containers.extend(self.driver.find_elements(By.CSS_SELECTOR, sel))

            cards = []
            for cont in containers:
                for el in cont.find_elements(By.CSS_SELECTOR, "li.b_algo, div.b_algo"):
                    try:
                        if not el.is_displayed():
                            continue
                        outer = el.get_attribute("outerHTML") or ""
                        if "b_ad" in outer or 'aria-label="Ads"' in outer:
                            continue
                        cards.append(el)
                    except Exception:
                        continue
            return cards
        except Exception:
            return []

    def fetch_meta_description(self, url):
        try:
            headers = {"User-Agent": random.choice(USER_AGENTS)}
            r = requests.get(url, headers=headers, timeout=6)
            if r.ok:
                m = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']', r.text, re.I)
                if m:
                    return BeautifulSoup(m.group(1), "html.parser").get_text(" ", strip=True)
        except Exception as e:
            log.debug(f"⚠️ Meta description fallback failed for {url}: {e}")
        return ""

    def get_card_data(self, card):
        try:
            # Prefer the main organic link
            try:
                a_tag = card.find_element(By.CSS_SELECTOR, "h2 > a")
            except Exception:
                links = card.find_elements(By.CSS_SELECTOR, "a[href]")
                a_tag = links[0] if links else None
            if not a_tag:
                return None

            raw_url = (a_tag.get_attribute("href") or "").strip()
            if not raw_url:
                return None

            normalized = normalize_url_any(raw_url) or normalize_url_any(unquote(raw_url)) or raw_url
            if (not normalized) or normalized.startswith("javascript:") or normalized == "#":
                return None

            title = (a_tag.text or "").strip()

            # Try common snippet locations
            description = ""
            for sel in [".b_caption p", ".b_snippet", "p"]:
                try:
                    els = card.find_elements(By.CSS_SELECTOR, sel)
                    for el in els:
                        t = (el.text or "").strip()
                        if t:
                            description = t
                            break
                    if description:
                        break
                except Exception:
                    pass

            if not description:
                description = self.fetch_meta_description(normalized)

            # Relevance guard
            text_blob = (title + " " + description).lower()
            if getattr(self, "_active_terms", None):
                blob = (title + " " + description + " " + normalized).lower()
                if not any(term in blob for term in self._active_terms):
                    return None


            if self.is_low_value_url(normalized):
                return None

            return {
                "url": normalized,
                "url_raw": raw_url,
                "title": title,
                "description": description,
            }
        except Exception as e:
            log.debug(f"❌ Failed to extract card data: {e}")
            return None

    def get_next_button(self):
        for sel in ["a.sb_pagN", "a[aria-label='Next page']", "a[title='Next page']"]:
            try:
                btn = self.driver.find_element(By.CSS_SELECTOR, sel)
                if btn and btn.is_displayed():
                    return btn
            except Exception:
                pass
        try:
            btn = self.driver.find_element(By.XPATH, "//a[contains(., 'Next')]")
            if btn and btn.is_displayed():
                return btn
        except Exception:
            pass
        return None
        
    
    def _click_next_page(self, expected_next_page: int, primary_query: str, timeout: float = 12.0) -> bool:
        """
        Preferred pagination:
          - Click the visible 'Next' control (best token preservation)
          - Fallback: navigate directly to Next button href (often more reliable than click)
        """
        def _get_first_param(u: str) -> int:
            try:
                q = parse_qs(urlparse(u).query)
                v = q.get("first", [""])
                return int(v[0]) if v and str(v[0]).isdigit() else -1
            except Exception:
                return -1

        try:
            before_url = self.driver.current_url
        except Exception:
            before_url = ""

        before_first = _get_first_param(before_url)
        try:
            before_page = self._current_page_number()
        except Exception:
            before_page = -1

        nxt = self.get_next_button()
        if not nxt:
            return False

        # reject disabled next
        try:
            aria_disabled = (nxt.get_attribute("aria-disabled") or "").strip().lower()
            if aria_disabled in ("true", "1"):
                return False
            cls = (nxt.get_attribute("class") or "").lower()
            if "disabled" in cls:
                return False
        except Exception:
            pass

        # Capture href now (may be tokenized)
        try:
            next_href = (nxt.get_attribute("href") or "").strip()
        except Exception:
            next_href = ""

        # Click attempt first
        try:
            self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", nxt)
            time.sleep(0.2)
            try:
                nxt.click()
            except Exception:
                self.driver.execute_script("arguments[0].click();", nxt)

            self._enforce_single_tab()
            time.sleep(0.4)
            self._post_nav_humanize()
            self.dismiss_cookie_banner()
        except Exception:
            pass

        # Wait for advancement
        end = time.time() + timeout
        while time.time() < end:
            self._enforce_single_tab()
            if self._detect_challenge():
                if not self._maybe_solve_simple_challenge(timeout=8):
                    break

            try:
                after_url = self.driver.current_url
            except Exception:
                after_url = ""

            after_first = _get_first_param(after_url)
            try:
                after_page = self._current_page_number()
            except Exception:
                after_page = -1

            if before_first != -1 and after_first != -1 and after_first > before_first:
                self._wait_query_reflected(primary_query, timeout=8.0)
                return True

            if after_page >= expected_next_page and after_page != before_page:
                self._wait_query_reflected(primary_query, timeout=8.0)
                return True

            time.sleep(0.25)

        # Fallback: navigate to href if we have one
        if next_href:
            try:
                self._get(next_href)
                self._enforce_single_tab()
                self._post_nav_humanize()
                time.sleep(random.uniform(1.0, 1.6))
                self.dismiss_cookie_banner()
                self._ensure_on_bing(next_href)

                if self._detect_challenge():
                    if not self._maybe_solve_simple_challenge(timeout=8):
                        return False

                # verify advance
                try:
                    final_url = self.driver.current_url
                except Exception:
                    final_url = ""
                final_first = _get_first_param(final_url)
                final_page = self._current_page_number()

                if before_first != -1 and final_first != -1 and final_first > before_first:
                    self._wait_query_reflected(primary_query, timeout=8.0)
                    return True

                if final_page >= expected_next_page and final_page != before_page:
                    self._wait_query_reflected(primary_query, timeout=8.0)
                    return True
            except Exception:
                pass

        return False




    def is_low_value_url(self, url):
        if not url:
            return True
        u = url.lower()

        # Bing internals & verticals
        if (
            u.startswith("https://www.bing.com") or u.startswith("https://bing.com") or
            u.startswith("http://www.bing.com") or u.startswith("http://bing.com") or
            "/images/search?" in u or "/videos/search?" in u or "/maps" in u
        ):
            return True

        try:
            host = urlparse(u).netloc
        except Exception:
            host = ""

        noisy_hosts = {
            "baidu.com", "zhidao.baidu.com", "jingyan.baidu.com", "zhihu.com",
            "sogou.com", "so.com", "360.cn", "weibo.cn", "weibo.com",
            "naver.com", "yandex.ru"
        }
        if host.endswith(".cn") or any(h in host for h in noisy_hosts):
            return True

        return False

    # ── core scrape ────────────────────────────────────────────────────────────
    def scrape_keyword(
        self,
        keyword,
        country,
        num_results=75,
        max_pages=30,
        use_country_clause: bool = True,
        _session_retries: int = 1,
    ):
        log.info(f"\nStarting scrape for keyword '{keyword}' in {country}")
        
        aborted_due_to_challenge = False

        # Cache query terms (keep 3+ char words) for relevance filter
        base_query = keyword
        self._active_terms = {t for t in re.findall(r"[a-zA-Z]{3,}", base_query.lower())}

        # Build queries
        # - base_query is exactly what the caller passed (may already include "UK", etc.)
        # - use_country_clause controls whether we append "site:.uk language:en" style clauses
        country_clause = self._country_query_clauses(country) if use_country_clause else ""
        primary_query = f"{base_query} {country_clause}".strip() if country_clause else base_query.strip()
        simple_query = base_query.strip()  # fallback if challenged


        all_results, seen_urls = [], set()

        try:
            settings = self.country_settings.get(self.current_country, {})
            mkt = settings.get("mkt", "en-GB")
            cc = settings.get("cc", "GB")
            log.info(f"🌍 Bing market: mkt={mkt}, cc={cc}, current_country={self.current_country}")

            base = "https://www.bing.com/search"
            params = {
                "q": primary_query,
                "mkt": mkt, "setmkt": mkt, "setlang": mkt, "cc": cc,
                "safeSearch": "moderate",
                "count": str(DEFAULT_COUNT),
                "ensearch": "1",
                "qft": "+filterui:language-en",
                "toWww": "1",
                "FORM": "QBLH"
            }
            start_url = f"{base}?{urllib.parse.urlencode(params)}"

            # Load page 1
            self._get(start_url)
            self._enforce_single_tab()          # ✅ add
            self._post_nav_humanize()
            time.sleep(1.0)
            self.dismiss_cookie_banner()
            self._ensure_on_bing(start_url)

            # Challenge handling: try checkbox; else switch to simpler query; else rotate identity
            if self._detect_challenge():
                if not self._maybe_solve_simple_challenge(timeout=10):
                    # try with simpler query
                    self._force_submit_query(simple_query)
                    self._post_nav_humanize()
                    if self._detect_challenge() and not self._maybe_solve_simple_challenge(timeout=10):
                        if not self._recover_from_challenge(country, simple_query):
                            return []

            # Force-submit whatever is in the box or primary
            self._force_submit_query(self._current_box_value_or(primary_query))

            # Wait for SERP container
            try:
                WebDriverWait(self.driver, 20).until(
                    EC.any_of(
                        EC.presence_of_element_located((By.ID, "b_results")),
                        EC.presence_of_element_located((By.ID, "b_content")),
                    )
                )
            except TimeoutException:
                html = self.driver.page_source or ""
                parsed = self._parse_cards_from_html(html)
                if parsed:
                    log.debug("⚗️ Parsed results from HTML fallback after timeout.")
                    for r in parsed:
                        if r["url"] in seen_urls or self.is_low_value_url(r["url"]):
                            continue
                        seen_urls.add(r["url"])
                        r["country"] = country
                        all_results.append(r)
                    if all_results:
                        log.info(f"✅ Retrieved {len(all_results)} results via HTML fallback.")
                        return all_results
                raise

            time.sleep(random.uniform(1.0, 2.0))
            os.makedirs("screenshots", exist_ok=True)

            # 5) Wait until the current query is reflected in the visible results
            if not self._wait_query_reflected(primary_query, timeout=12.0):
                log.debug("⚠️ Query not reflected yet; giving page a nudge and re-checking.")
                try:
                    self.driver.execute_script("window.scrollTo(0, 250);")
                    time.sleep(0.6)
                except Exception:
                    pass
                self._wait_query_reflected(primary_query, timeout=6.0)

            # Snapshot tokens only after the query is reflected
            self._snapshot_serp_tokens()

            page = 1
            q_first = (base_query.split() or [""])[0]

            # >>> NEW: track consecutive empty pages
            empty_pages_streak = 0

            while page <= max_pages:
                log.debug(f"📄 Page {page} URL: {self.driver.current_url}")
                self.dismiss_cookie_banner()
                self._ensure_on_bing(start_url)
                self._ensure_serp_or_resubmit(primary_query, mkt, cc)


                # Challenge mid-loop
                if self._detect_challenge():
                    if not self._maybe_solve_simple_challenge(timeout=8):
                        if not self._recover_from_challenge(country, simple_query, resume_url=self.driver.current_url):
                            aborted_due_to_challenge = True
                            break


                if not self._is_valid_serp(q_first):
                    log.warning("⚠️ SERP integrity check failed; attempting recovery to start_url")
                    self._get(start_url)
                    self._enforce_single_tab()
                    self._post_nav_humanize()
                    time.sleep(1.0)
                    self.dismiss_cookie_banner()
                    self._ensure_on_bing(start_url)
                    self._ensure_serp_or_resubmit(primary_query, mkt, cc)
                    if not self._is_valid_serp(q_first):
                        log.error("🛑 Could not recover SERP; aborting this page.")
                        break

                log.debug(f"🔢 Bing shows current page: {self._current_page_number()}")

                cards = self.extract_result_cards()
                log.debug(f"🔍 Found {len(cards)} total cards")

                # track first H2 to detect sticking
                try:
                    first_h2 = ""
                    if cards:
                        try:
                            h2a = cards[0].find_element(By.CSS_SELECTOR, "h2 > a")
                            first_h2 = (h2a.text or "").strip()
                        except Exception:
                            pass
                    if first_h2:
                        if self._last_first_h2 == first_h2 and page > 1:
                            log.warning("⚠️ First result unchanged from previous page; Bing likely stuck — relying on de-dup.")
                        self._last_first_h2 = first_h2
                except Exception:
                    pass

                if not cards:
                    html = self.driver.page_source or ""
                    parsed = self._parse_cards_from_html(html)
                    if parsed:
                        new_added = 0
                        for r in parsed:
                            if r["url"] in seen_urls or self.is_low_value_url(r["url"]):
                                continue
                            seen_urls.add(r["url"])
                            r["country"] = country
                            all_results.append(r)
                            new_added += 1
                        log.info(f"✅ Page {page}: {new_added} results via HTML fallback, total {len(seen_urls)}")
                        # >>> NEW: update empty streak based on new_added
                        if new_added == 0:
                            empty_pages_streak += 1
                        else:
                            empty_pages_streak = 0
                    else:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        _dump_html(self.driver, f"screenshots/empty_page_{country}_{page}_{ts}.html")
                        _safe_screenshot(self.driver, f"screenshots/empty_page_{country}_{page}_{ts}.png")
                        log.warning(f"⚠️ No cards found — saved HTML and screenshot for page {page}")
                        # >>> NEW: count truly empty page
                        empty_pages_streak += 1

                    # >>> NEW: stop if we have 3 empty pages in a row
                    if empty_pages_streak >= 3:
                        log.info("🛑 Stopping: reached 3 successive pages with zero new results.")
                        break
                else:
                    new_results = []
                    for card in cards:
                        data = self.get_card_data(card)
                        if not data:
                            continue
                        if data["url"] in seen_urls:
                            continue
                        seen_urls.add(data["url"])
                        data["country"] = country
                        new_results.append(data)
                        log.debug(f"✅ Collected: {data['url']}")
                    all_results.extend(new_results)
                    log.info(f"✅ Page {page}: {len(new_results)} new results, total so far: {len(seen_urls)}")

                    # >>> NEW: update/reset empty streak based on this page
                    if len(new_results) == 0:
                        empty_pages_streak += 1
                    else:
                        empty_pages_streak = 0

                    # >>> NEW: early stop if streak hit 3 after this page
                    if empty_pages_streak >= 3:
                        log.info("🛑 Stopping: reached 3 successive pages with zero new results.")
                        break

                if len(seen_urls) >= num_results:
                    log.info(f"🎯 Target of {num_results} unique results reached.")
                    return all_results

                # Hybrid pagination (preferred: click Next to preserve Bing tokens)
                next_page = page + 1
                if next_page > max_pages:
                    break

                # A) Click “Next” (preferred: preserves ALL Bing tokens)
                if self._click_next_page(next_page, primary_query):
                    cur = self._current_page_number()
                    log.debug(f"🔢 After clicking Next, Bing shows current page: {cur}")
                    page = next_page
                    continue


                # B) Pager href (still token-preserving)
                before_first = -1
                try:
                    q0 = parse_qs(urlparse(self.driver.current_url).query)
                    v0 = q0.get("first", [""])
                    before_first = int(v0[0]) if v0 and str(v0[0]).isdigit() else -1
                except Exception:
                    pass

                if self._goto_page_via_pager(next_page):
                    self.dismiss_cookie_banner()
                    self._ensure_on_bing(self.driver.current_url)

                    if self._detect_challenge():
                        if not self._maybe_solve_simple_challenge(timeout=8):
                            if not self._recover_from_challenge(country, simple_query):
                                aborted_due_to_challenge = True
                                break


                    self._post_nav_humanize()
                    self._wait_query_reflected(primary_query, timeout=8.0)

                    cur = self._current_page_number()
                    log.debug(f"🔢 After pager href, Bing shows current page: {cur}")

                    after_first = -1
                    try:
                        q1 = parse_qs(urlparse(self.driver.current_url).query)
                        v1 = q1.get("first", [""])
                        after_first = int(v1[0]) if v1 and str(v1[0]).isdigit() else -1
                    except Exception:
                        pass

                    if (before_first != -1 and after_first != -1 and after_first > before_first) or (cur >= next_page):
                        page = next_page
                        continue


                # C) Deterministic URL (last resort)
                next_url = self._build_page_url(self.driver.current_url, next_page)
                log.debug(f"➡️ Navigating (tokenized fallback) to page {next_page}: {next_url}")
                self._get(next_url)
                self._enforce_single_tab()          # ✅ add
                self._post_nav_humanize()
                time.sleep(random.uniform(1.1, 1.9))
                self.dismiss_cookie_banner()
                self._ensure_on_bing(next_url)
                if self._detect_challenge():
                    if not self._maybe_solve_simple_challenge(timeout=8):
                        if not self._recover_from_challenge(country, simple_query, resume_url=next_url):
                            aborted_due_to_challenge = True
                            break

                self._wait_query_reflected(primary_query, timeout=8.0)


                # If stuck, nudge once
                cur = self._current_page_number()
                if cur >= next_page:
                    try:
                        cards_now = self.extract_result_cards()
                        first_h2_now = ""
                        if cards_now:
                            try:
                                h2a = cards_now[0].find_element(By.CSS_SELECTOR, "h2 > a")
                                first_h2_now = (h2a.text or "").strip()
                            except Exception:
                                pass
                        if self._last_first_h2 and first_h2_now and self._last_first_h2 == first_h2_now:
                            alt_url = self._add_unstick_params(next_url)
                            log.debug(f"🔧 Unstick retry: {alt_url}")
                            self._get(alt_url)
                            self._enforce_single_tab()          # ✅ add
                            self._post_nav_humanize()
                            time.sleep(random.uniform(1.0, 1.6))
                            self.dismiss_cookie_banner()
                            self._ensure_on_bing(alt_url)
                            self._wait_query_reflected(primary_query, timeout=8.0)
                    except Exception:
                        pass

                cur = self._current_page_number()
                log.debug(f"🔢 Bing shows current page after all fallbacks: {cur}")

                if cur < next_page:
                    log.warning(f"⚠️ Could not advance to page {next_page} (still on {cur}). Stopping pagination to avoid looping on same SERP.")
                    break

                page = next_page


        except Exception as e:
            if self._is_dead_session_error(e) and _session_retries > 0:
                log.warning(
                    "⚠️ WebDriver session ended unexpectedly while scraping "
                    f"{keyword!r}; restarting Chrome and retrying once."
                )
                try:
                    self._restart_dead_session(country)
                except Exception as restart_error:
                    log.warning(f"⚠️ Could not restart Chrome after session loss: {restart_error}")
                    return all_results

                return self.scrape_keyword(
                    keyword,
                    country,
                    num_results=num_results,
                    max_pages=max_pages,
                    use_country_clause=use_country_clause,
                    _session_retries=_session_retries - 1,
                )

            log.warning(f"⚠️ Scraping failed: {e}")
            _safe_screenshot(self.driver, "screenshots/fatal_scrape_error.png")
            _dump_html(self.driver, "screenshots/fatal_scrape_error.html")
            log.debug("Saved fatal artifacts for inspection.")

        log.warning(f"⚠️ Finished with {len(seen_urls)} unique results.")
        
        # ✅ If we bailed due to bot pressure and got very few results, force a retry upstream
        if aborted_due_to_challenge and len(seen_urls) < 10:   # or pass in a threshold
            return []
        return all_results


    # ── public API ─────────────────────────────────────────────────────────────
    def scrape_multiple_keywords(
        self,
        keywords,
        target_country,
        num_results=75,
        should_remove=None,
        add_country_suffix: bool = True,
        use_country_clause: bool | None = None,
    ):
        """
        add_country_suffix:
            - True: enhance_keyword() appends UK/USA/Canada token
            - False: keyword is used verbatim

        use_country_clause:
            - True: append site/language clause in scrape_keyword (e.g. "site:.uk language:en")
            - False: do not append clause (true "plain query" behavior)
            - None: default behaviour:
                * if add_country_suffix True  -> use_country_clause True
                * if add_country_suffix False -> use_country_clause False
        """
        if use_country_clause is None:
            use_country_clause = True if add_country_suffix else False

        results = []
        for keyword in keywords:
            c = normalize_country(target_country or self.current_country)

            # Caller can force "plain keyword" (no appended country token)
            enhanced_kw = self.enhance_keyword(keyword, country=c) if add_country_suffix else keyword

            log.info(f"🔍 Searching: {enhanced_kw!r} | target_country={c}")
            try:
                new_results = self.scrape_keyword(enhanced_kw, target_country, num_results=num_results)
                if not new_results:
                    log.warning(f"⚠️ No results scraped from Bing for keyword: '{enhanced_kw}'")
                    return []


                log.info(f"📥 Fetched {len(new_results)} raw results for keyword: '{enhanced_kw}'")

                seen = set()
                deduped = []
                for r in new_results:
                    u = r.get("url")
                    if not u or u in seen:
                        continue
                    seen.add(u)
                    deduped.append(r)

                log.info(f"✅ Retained {len(deduped)} unique URL(s) after duplicate filtering")
                results.extend(deduped[:num_results])

            except Exception as e:
                log.warning(f"⚠️ Error scraping '{enhanced_kw}': {e}")
                return []

        return results



    # ── quick fetch ────────────────────────────────────────────────────────────
    def fetch_page_text(self, url):
        try:
            headers = {"User-Agent": random.choice(USER_AGENTS)}
            r = requests.get(url, headers=headers, timeout=7)
            if not r.ok:
                return ""
            soup = BeautifulSoup(r.text, "html.parser")
            for s in soup(["script", "style", "noscript"]):
                s.decompose()
            return soup.get_text(separator="\n", strip=True)
        except Exception as e:
            log.warning(f"⚠️ Failed to fetch or parse content from {url}: {e}")
            return ""

    # helper to read current box text
    def _current_box_value_or(self, fallback: str) -> str:
        try:
            box = self.driver.find_element(By.CSS_SELECTOR, "#sb_form_q")
            val = (box.get_attribute("value") or "").strip()
            return val or fallback
        except Exception:
            return fallback
