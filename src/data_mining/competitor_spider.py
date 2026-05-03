"""
competitor_spider.py — multi-page web crawler for technical domain content.

Improvements over the original:
  A. Multi-page crawling (follows internal links up to max_depth / max_pages)
  B. Full sentence extraction via trafilatura (with BeautifulSoup fallback)
  C. Language detection — filters to Italian-only sentences
  D. Schema.org / JSON-LD structured data extraction
  E. Rate limiting and per-domain page cap
  F. robots.txt respect
"""
import json
import logging
import re
import time
from collections import deque
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

try:
    import trafilatura
    _HAS_TRAFILATURA = True
except ImportError:
    _HAS_TRAFILATURA = False

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "ita_Latn": [" il ", " lo ", " la ", " gli ", " le ", " di ", " del ", " della ",
                 " per ", " con ", " una ", " che ", " in ", " è ", " sono ", " viene "],
    "eng_Latn": [" the ", " and ", " of ", " for ", " with ", " from ", " this ",
                 " that ", " is ", " are ", " to ", " a "],
    "fra_Latn": [" le ", " la ", " les ", " des ", " pour ", " avec ", " dans ",
                 " une ", " que ", " est ", " sont "],
    "spa_Latn": [" el ", " la ", " los ", " las ", " para ", " con ", " una ",
                 " que ", " del ", " en ", " es ", " son "],
    "deu_Latn": [" die ", " der ", " das ", " und ", " für ", " mit ", " ist ",
                 " ein ", " eine ", " von ", " zu "],
}

_NOISE_TERMS = {
    "privacy", "cookie", "cookies", "contatti", "contact", "contacts",
    "azienda", "company", "news", "blog", "login", "area riservata",
    "newsletter", "catalogo", "catalogue", "download", "scarica",
    "home", "menu", "search", "cerca", "seguici", "facebook",
    "instagram", "linkedin", "youtube", "termini", "condizioni",
    "copyright", "credits", "lavora con noi", "chi siamo", "about",
    "sitemap", "404", "error",
}

_DOMAIN_KEYWORDS = {
    "casson", "coibent", "monobloc", "serrament", "finestr", "foro",
    "telaio", "controtelaio", "avvolg", "tapparella", "frangisole",
    "zanzar", "guarnizion", "sigill", "isol", "termic", "acustic",
    "tenuta", "profil", "soglia", "bancale", "sottobancale",
    "posa", "giunto", "vapore", "membrana", "nastro", "schiuma",
    "oscurante", "facciata", "lucernar", "deventer", "presystem",
    "casement", "shutter", "roller", "blind", "seal", "gasket",
    "thermal", "acoustic", "weatherstrip", "profile", "frame",
    "fenêtre", "volet", "store", "joint", "isolation", "thermique",
    "ventana", "persiana", "sellado", "perfil", "aislamiento",
}


def _detect_lang(text: str) -> str:
    lowered = f" {text.lower()} "
    scores = {
        lang: sum(lowered.count(t) for t in tokens)
        for lang, tokens in _STOPWORDS.items()
    }
    best_lang, best_score = max(scores.items(), key=lambda x: x[1])
    return best_lang if best_score > 0 else "unknown"


def _is_domain_relevant(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in _DOMAIN_KEYWORDS)


def _clean_sentence(text: str) -> str | None:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 25 or len(text) > 600:
        return None
    if text.lower() in _NOISE_TERMS:
        return None
    if any(noise in text.lower() for noise in _NOISE_TERMS if len(noise) > 6):
        return None
    return text


class CompetitorSpider:

    def __init__(
        self,
        max_depth: int = 2,
        max_pages_per_site: int = 40,
        request_delay: float = 0.8,
        timeout: int = 12,
        respect_robots: bool = True,
    ):
        self.max_depth = max_depth
        self.max_pages_per_site = max_pages_per_site
        self.request_delay = request_delay
        self.timeout = timeout
        self.respect_robots = respect_robots
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; LEO-NMT-Spider/1.0; "
                "research crawler for technical translation data)"
            )
        }
        self._robots_cache: dict[str, RobotFileParser] = {}

    # ------------------------------------------------------------------ #
    # robots.txt                                                           #
    # ------------------------------------------------------------------ #

    def _can_fetch(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots_cache:
            rp = RobotFileParser()
            try:
                rp.set_url(f"{origin}/robots.txt")
                rp.read()
            except Exception:
                rp = None
            self._robots_cache[origin] = rp
        rp = self._robots_cache[origin]
        return rp is None or rp.can_fetch(self.headers["User-Agent"], url)

    # ------------------------------------------------------------------ #
    # HTTP fetch                                                           #
    # ------------------------------------------------------------------ #

    def _fetch(self, url: str) -> tuple[str, BeautifulSoup | None]:
        if not self._can_fetch(url):
            logger.debug(f"Blocked by robots.txt: {url}")
            return "", None
        try:
            resp = requests.get(url, headers=self.headers, timeout=self.timeout)
            if resp.status_code != 200:
                return "", None
            soup = BeautifulSoup(resp.text, "html.parser")
            return resp.text, soup
        except Exception as e:
            logger.debug(f"Fetch error {url}: {e}")
            return "", None

    # ------------------------------------------------------------------ #
    # Content extraction                                                   #
    # ------------------------------------------------------------------ #

    def _extract_sentences_trafilatura(self, raw_html: str, url: str) -> list[str]:
        try:
            text = trafilatura.extract(
                raw_html,
                url=url,
                include_tables=True,
                include_links=False,
                no_fallback=False,
            )
            if not text:
                return []
            sentences = re.split(r"(?<=[.!?])\s+", text)
            return [s for s in sentences if _clean_sentence(s)]
        except Exception:
            return []

    def _extract_sentences_bs4(self, soup: BeautifulSoup) -> list[str]:
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        sentences = []
        for tag in soup.find_all(["p", "li", "h2", "h3", "td", "dd"]):
            text = tag.get_text(" ", strip=True)
            cleaned = _clean_sentence(text)
            if cleaned:
                sentences.append(cleaned)
        return sentences

    def _extract_jsonld_sentences(self, soup: BeautifulSoup) -> list[str]:
        sentences = []
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
                items = data if isinstance(data, list) else [data]
                for item in items:
                    for field in ("description", "name", "headline", "text"):
                        val = item.get(field, "")
                        if isinstance(val, str):
                            cleaned = _clean_sentence(val)
                            if cleaned:
                                sentences.append(cleaned)
            except Exception:
                continue
        return sentences

    def _extract_terms_heuristic(self, soup: BeautifulSoup) -> list[str]:
        candidates = set()
        for tag in soup.find_all(["h1", "h2", "h3"]):
            text = tag.get_text().strip()
            if 3 < len(text) < 60:
                candidates.add(text.lower())
        for li in soup.find_all("li"):
            strong = li.find("strong")
            if strong:
                text = strong.get_text().strip()
                if 3 < len(text) < 50:
                    candidates.add(text.lower())
        return [t for t in candidates if t not in _NOISE_TERMS]

    # ------------------------------------------------------------------ #
    # Link discovery                                                       #
    # ------------------------------------------------------------------ #

    def _same_domain_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        base = urlparse(base_url)
        origin = f"{base.scheme}://{base.netloc}"
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
                continue
            full = urljoin(base_url, href)
            parsed = urlparse(full)
            if parsed.netloc != base.netloc:
                continue
            # Skip non-HTML resources
            if re.search(r"\.(pdf|jpg|jpeg|png|gif|svg|css|js|zip|doc|xls)$", parsed.path, re.I):
                continue
            # Drop query strings and fragments
            clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
            links.append(clean)
        return list(dict.fromkeys(links))  # preserve order, deduplicate

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def scrape_site(self, url: str) -> list[str]:
        """Crawl one site and return heuristic term candidates (legacy interface)."""
        _, soup = self._fetch(url)
        if soup is None:
            return []
        return self._extract_terms_heuristic(soup)

    def crawl_site(self, start_url: str) -> dict:
        """
        BFS crawl starting from start_url.

        Returns:
            {
                "sentences": list[str],   # full Italian sentences for direct translation
                "terms":     list[str],   # short term candidates for generate_dataset()
            }
        """
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(start_url.rstrip("/"), 0)])
        all_sentences: list[str] = []
        all_terms: list[str] = []
        pages_fetched = 0

        logger.info(f"Crawling {start_url} (depth≤{self.max_depth}, pages≤{self.max_pages_per_site})")

        while queue and pages_fetched < self.max_pages_per_site:
            url, depth = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            raw_html, soup = self._fetch(url)
            if soup is None:
                continue

            pages_fetched += 1
            time.sleep(self.request_delay)

            # --- sentence extraction ---
            if _HAS_TRAFILATURA:
                sents = self._extract_sentences_trafilatura(raw_html, url)
                if not sents:
                    sents = self._extract_sentences_bs4(soup)
            else:
                sents = self._extract_sentences_bs4(soup)

            sents += self._extract_jsonld_sentences(soup)

            # filter Italian + domain-relevant
            for s in sents:
                lang = _detect_lang(s)
                if lang in {"ita_Latn", "unknown"} and _is_domain_relevant(s):
                    all_sentences.append(s)

            # --- term extraction ---
            all_terms.extend(self._extract_terms_heuristic(soup))

            # --- link discovery ---
            if depth < self.max_depth:
                for link in self._same_domain_links(soup, url):
                    if link not in visited:
                        queue.append((link, depth + 1))

        logger.info(
            f"  {start_url}: {pages_fetched} pages → "
            f"{len(all_sentences)} sentences, {len(all_terms)} terms"
        )
        return {"sentences": all_sentences, "terms": all_terms}
