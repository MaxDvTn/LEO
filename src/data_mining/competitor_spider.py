"""
competitor_spider.py — multi-page web crawler for technical domain content.

  A. Multi-page crawling (BFS, max_depth / max_pages)
  B. Full sentence extraction via trafilatura (BeautifulSoup fallback)
  C. Language detection — keeps Italian-only sentences
  D. Schema.org / JSON-LD structured data extraction
  E. Rate limiting and per-domain page cap
  F. robots.txt respect
  G. Per-site crawl report (pages, extracted, lang_ok, domain_ok, terms)
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
from tqdm import tqdm

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

# Exact-match noise: short standalone strings that are pure navigation/UI.
# NOT used as substring — only matched against the full lowercased sentence.
_NOISE_EXACT = {
    "privacy", "cookie", "cookies", "contatti", "contact", "contacts",
    "azienda", "company", "news", "blog", "login", "area riservata",
    "newsletter", "catalogo", "catalogue", "download", "scarica",
    "home", "menu", "search", "cerca", "seguici", "facebook",
    "instagram", "linkedin", "youtube", "termini", "condizioni",
    "copyright", "credits", "lavora con noi", "chi siamo", "about",
    "sitemap", "404", "error",
}

_DOMAIN_KEYWORDS = {
    # Core Roverplastik product line
    "casson", "coibent", "monobloc", "presystem", "deventer",
    # Window & frame components
    "serrament", "infiss", "finestr", "lucernar", "facciata",
    "telaio", "controtelaio", "foro finestra", "davanzale",
    "soglia", "bancale", "sottobancale", "spalla",
    # Profiles & seals
    "profil", "guarnizion", "sigill", "tenuta", "nastro",
    "schiuma", "giunto", "impermeab",
    # Insulation & vapour
    "isol", "coibent", "membrana", "vapore", "cappotto",
    "intercapedin", "spessor",
    # Thermal & acoustic physics
    "termic", "acustic", "trasmittan", "ponte termic",
    "abbattim", "condensa", "muffa",
    # Shading & sun control
    "avvolg", "tapparella", "frangisole", "zanzar",
    "oscurante", "venezian", "persiana",
    # Installation & process
    "posa", "montaggio", "fissagg", "muratur", "rivestiment",
    "riquali", "ristruttur", "retrofit",
    # Glazing
    "vetr", "doppio vetro", "triplo vetro", "lastr",
    # Certification & performance
    "certific", "prestazion", "classe energet", "norma uni",
    "motorizzat",
    # Multilingual synonyms (for sites with mixed-language pages)
    "casement", "shutter", "roller", "blind", "seal", "gasket",
    "thermal", "acoustic", "weatherstrip", "profile", "frame",
    "fenêtre", "volet", "store", "joint", "isolation", "thermique",
    "ventana", "sellado", "perfil", "aislamiento",
}

_MD_BOLD_RE = re.compile(r"\*{1,2}(.+?)\*{1,2}")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")


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


def _normalize_term_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _clean_sentence(text: str) -> str | None:
    # Strip markdown formatting artifacts
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < 25 or len(text) > 600:
        return None

    # Exact match only — do NOT use substring; legitimate sentences may
    # contain words like "catalogo" or "newsletter" in a technical context.
    if text.lower().strip() in _NOISE_EXACT:
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
            logger.info("Blocked by robots.txt: %s", url)
            return "", None
        try:
            resp = requests.get(url, headers=self.headers, timeout=self.timeout)
            if resp.status_code != 200:
                logger.info("Non-200 response url=%s status=%d final_url=%s", url, resp.status_code, resp.url)
                return "", None
            soup = BeautifulSoup(resp.text, "html.parser")
            return resp.text, soup
        except Exception as e:
            logger.info("Fetch error url=%s error=%s", url, e)
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

        def _walk_jsonld(node):
            if isinstance(node, list):
                for item in node:
                    yield from _walk_jsonld(item)
            elif isinstance(node, dict):
                yield node
                graph = node.get("@graph")
                if graph is not None:
                    yield from _walk_jsonld(graph)

        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
                for item in _walk_jsonld(data):
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
        candidates: dict[str, str] = {}
        for tag in soup.find_all(["h1", "h2", "h3"]):
            text = tag.get_text().strip()
            if 3 < len(text) < 60:
                candidates.setdefault(_normalize_term_key(text), text)
        for li in soup.find_all("li"):
            strong = li.find("strong")
            if strong:
                text = strong.get_text().strip()
                if 3 < len(text) < 50:
                    candidates.setdefault(_normalize_term_key(text), text)
        return [
            term for key, term in candidates.items()
            if key not in _NOISE_EXACT
        ]

    # ------------------------------------------------------------------ #
    # Link discovery                                                       #
    # ------------------------------------------------------------------ #

    def _same_domain_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        base = urlparse(base_url)
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
                continue
            full = urljoin(base_url, href)
            parsed = urlparse(full)
            if parsed.netloc != base.netloc:
                continue
            if re.search(r"\.(pdf|jpg|jpeg|png|gif|svg|css|js|zip|doc|xls)$", parsed.path, re.I):
                continue
            clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
            links.append(clean)
        return list(dict.fromkeys(links))

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
                "sentences":    list[str],  # Italian domain sentences for translation
                "terms":        list[str],  # short term candidates for generate_dataset()
                "pages_fetched": int,
                "report": {                 # per-site diagnostic counters
                    "n_extracted":  int,    # sentences passing _clean_sentence
                    "n_lang_ok":    int,    # sentences detected as Italian
                    "n_domain_ok":  int,    # sentences also passing domain filter
                    "n_terms":      int,    # heuristic term candidates
                }
            }
        """
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(start_url.rstrip("/"), 0)])
        all_sentences: list[str] = []
        all_terms: list[str] = []
        pages_fetched = 0
        n_extracted = 0
        # sentences grouped by detected language, domain-filtered
        sentences_by_lang: dict[str, list[str]] = {}
        _TRACKED_LANGS = {"ita_Latn", "eng_Latn", "fra_Latn", "spa_Latn", "deu_Latn"}

        logger.info(f"Crawling {start_url} (depth≤{self.max_depth}, pages≤{self.max_pages_per_site})")

        site_label = urlparse(start_url).netloc.replace("www.", "")
        with tqdm(total=self.max_pages_per_site, desc=f"Crawl {site_label}", unit="page", leave=False) as pbar:
            while queue and pages_fetched < self.max_pages_per_site:
                url, depth = queue.popleft()
                if url in visited:
                    continue
                visited.add(url)

                raw_html, soup = self._fetch(url)
                if soup is None:
                    continue

                pages_fetched += 1
                pbar.update(1)
                n_it = len(sentences_by_lang.get("ita_Latn", []))
                pbar.set_postfix(it=n_it, terms=len(all_terms), refresh=False)
                time.sleep(self.request_delay)

                jsonld_sents = self._extract_jsonld_sentences(soup)

                if _HAS_TRAFILATURA:
                    sents = self._extract_sentences_trafilatura(raw_html, url)
                    if not sents:
                        sents = self._extract_sentences_bs4(soup)
                else:
                    sents = self._extract_sentences_bs4(soup)

                sents += jsonld_sents
                n_extracted += len(sents)

                for s in sents:
                    if not _is_domain_relevant(s):
                        continue
                    lang = _detect_lang(s)
                    if lang in _TRACKED_LANGS:
                        sentences_by_lang.setdefault(lang, []).append(s)

                all_terms.extend(self._extract_terms_heuristic(soup))

                if depth < self.max_depth:
                    for link in self._same_domain_links(soup, url):
                        if link not in visited:
                            queue.append((link, depth + 1))

                n_it = len(sentences_by_lang.get("ita_Latn", []))
                pbar.set_postfix(it=n_it, terms=len(all_terms), refresh=False)

        it_sentences = sentences_by_lang.get("ita_Latn", [])
        n_lang_ok = sum(len(v) for v in sentences_by_lang.values())
        report = {
            "n_extracted": n_extracted,
            "n_lang_ok": n_lang_ok,
            "n_domain_ok": len(it_sentences),
            "n_terms": len(all_terms),
            "by_lang": {lang: len(sents) for lang, sents in sentences_by_lang.items()},
        }
        logger.info(
            f"  {start_url}: {pages_fetched} pages → "
            f"{n_extracted} extracted, {n_lang_ok} domain-ok "
            f"({', '.join(f'{l}:{len(s)}' for l, s in sentences_by_lang.items())}), "
            f"{len(all_terms)} terms"
        )
        return {
            "sentences": it_sentences,           # Italian only — backward compat with cache
            "sentences_by_lang": sentences_by_lang,
            "terms": all_terms,
            "pages_fetched": pages_fetched,
            "report": report,
        }
