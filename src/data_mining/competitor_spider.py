import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
from tqdm import tqdm
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.common.config import conf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SPIDER")

class CompetitorSpider:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def fetch_text(self, url):
        """Scarica il testo pulito da una pagina web."""
        try:
            logger.info(f"🌐 Fetching: {url}")
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code != 200:
                logger.error(f"Failed to fetch {url}: {resp.status_code}")
                return ""
            
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            # Rimuoviamo script, stili e footer per pulire
            for trash in soup(['script', 'style', 'nav', 'footer']):
                trash.decompose()
                
            # Prendiamo solo i paragrafi e le liste
            text_blocks = [p.get_text().strip() for p in soup.find_all(['p', 'li', 'h2', 'h3'])]
            full_text = " ".join([t for t in text_blocks if len(t) > 20])
            return full_text[:4000] # Limitiamo a 4000 caratteri per non intasare la GPU
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return ""

    def scrape_site(self, url: str) -> list:
        """Fetches a URL and returns heuristic term candidates."""
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code != 200:
                logger.error(f"Failed to fetch {url}: {resp.status_code}")
                return []
            soup = BeautifulSoup(resp.content, 'html.parser')
            terms = self.extract_terms_heuristic(soup)
            logger.info(f"Found {len(terms)} terms at {url}")
            return terms
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return []

    def extract_terms_heuristic(self, soup):
        """Estrae candidati termini tecnici basandosi su tag HTML rilevanti (H1-H3, strong, li)."""
        candidates = set()
        
        # 1. Product titles usually in H1, H2, H3
        for tag in soup.find_all(['h1', 'h2', 'h3']):
            text = tag.get_text().strip()
            if 3 < len(text) < 50: # Length filter
                candidates.add(text.lower())
                
        # 2. Bold items in lists (often features)
        for li in soup.find_all('li'):
            strong = li.find('strong')
            if strong:
                text = strong.get_text().strip()
                if 3 < len(text) < 40:
                    candidates.add(text.lower())
                    
        return list(candidates)

def main():
    spider = CompetitorSpider()
    
    # Use URLs from configuration
    target_urls = conf.spider.target_urls
    
    all_terms = []
    
    print("🕸️ Inizio scansione web (Modalità Leggera - No GPU)...")
    for url in target_urls:
        try:
            # We fetch manually here to pass soup to heuristic
            logger.info(f"🌐 Fetching: {url}")
            resp = requests.get(url, headers=spider.headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, 'html.parser')
                terms = spider.extract_terms_heuristic(soup)
                print(f"   -> Trovati {len(terms)} candidati su {url}")
                all_terms.extend(terms)
        except Exception as e:
            logger.error(f"Failed {url}: {e}")
    
    unique_terms = sorted(list(set(all_terms)))
    print(f"\n📦 Totale termini grezzi trovati: {len(unique_terms)}")
    
    # Filter very common words or junk (simple heuristic)
    filtered_terms = [t for t in unique_terms if " " in t or len(t) > 6] # Keep compounds or long words
    
    if filtered_terms:
        df_terms = pd.DataFrame(filtered_terms, columns=["term"])
        term_path = PROJECT_ROOT / "data" / "glossary" / "scraped_candidates.csv"
        term_path.parent.mkdir(parents=True, exist_ok=True)
        df_terms.to_csv(term_path, index=False)
        print(f"✅ Candidati salvati: {term_path}")
        print("👉 Questi termini sono grezzi. Quando la GPU sarà libera, potremo filtrarli con l'AI.")

if __name__ == "__main__":
    main()