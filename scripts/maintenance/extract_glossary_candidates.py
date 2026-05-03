"""
extract_glossary_candidates.py — mina candidati per il glossario da due sorgenti:

  A. PDF corpus (n-gram frequency su frasi già estratte)
  B. Web spider cache (termini estratti dal BFS crawl)

Output: lista stampata + CSV in reports/glossary_candidates.csv
Usage:
    python scripts/maintenance/extract_glossary_candidates.py
    python scripts/maintenance/extract_glossary_candidates.py --top 80 --min-freq 2
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_mining.pdf_processor import PdfMiner
from src.synthesis.glossary_data import ROVER_GLOSSARY
from src.common.config import conf

# ── Domain filter (mirrors competitor_spider._DOMAIN_KEYWORDS) ──────────────
_DOMAIN_KW = {
    "casson", "coibent", "monobloc", "serrament", "finestr", "foro",
    "telaio", "controtelaio", "avvolg", "tapparella", "frangisole",
    "zanzar", "guarnizion", "sigill", "isol", "termic", "acustic",
    "tenuta", "profil", "soglia", "bancale", "sottobancale",
    "posa", "giunto", "vapore", "membrana", "nastro", "schiuma",
    "oscurante", "facciata", "lucernar", "deventer", "presystem",
    "riquali", "ristruttur", "retrofit", "impermeab", "condensa",
    "muffa", "trasmittan", "ponte termic", "efficien", "energetica",
    "certific", "norma", "classe", "prestazion", "fissagg", "montaggio",
    "poliuretan", "epdm", "silicone", "resina", "rivestiment",
}

# Italian stopwords (minimal, for n-gram cleaning)
_STOPWORDS = {
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
    "di", "del", "della", "dei", "degli", "delle", "da", "dal",
    "dalla", "dei", "in", "nel", "nella", "nei", "nelle",
    "a", "al", "alla", "ai", "alle", "con", "su", "per", "tra", "fra",
    "e", "ed", "o", "ma", "che", "come", "se", "quando", "dove",
    "non", "è", "sono", "viene", "si", "ha", "hanno", "deve",
    "può", "anche", "più", "molto", "questo", "questa", "questi",
    "queste", "tale", "tali", "ogni", "tutto", "tutti",
}

# Context inference rules (first match wins)
_CONTEXT_RULES = [
    ({"muffa", "condensa", "umidità", "degrado"}, "problem"),
    ({"riquali", "ristruttur", "retrofit", "non invasiv"}, "renovation"),
    ({"posa", "installaz", "montaggio", "fissagg", "cantiere"}, "installation"),
    ({"trasmittan", "ponte termic", "acustic", "abbattim", "rw", "uw"}, "physics"),
    ({"efficien", "energetic", "emissioni", "co2", "ecolog"}, "ecology"),
    ({"certific", "norma", "prestazion", "classe", "test", "verifica"}, "performance"),
    ({"guarnizion", "sigill", "tenuta", "nastro", "schiuma", "impermeab"}, "sealing"),
    ({"isol", "membrana", "vapore", "cappotto", "coibent"}, "insulation"),
    ({"avvolg", "tapparella", "oscurante", "venezian", "frangisole", "zanzar"}, "shading"),
    ({"profil", "telaio", "controtelaio", "finestr", "serrament"}, "window_frame"),
    ({"casson", "monobloc", "spalla", "bancale", "sottobancale", "soglia"}, "structure"),
    ({"poliuretan", "epdm", "silicone", "resina"}, "hardware"),
]


def _infer_context(term: str) -> str:
    low = term.lower()
    for keywords, ctx in _CONTEXT_RULES:
        if any(kw in low for kw in keywords):
            return ctx
    return "product"


def _is_domain_relevant(phrase: str) -> bool:
    low = phrase.lower()
    return any(kw in low for kw in _DOMAIN_KW)


def _clean_token(tok: str) -> str:
    return re.sub(r"[^\wàèéìòùÀÈÉÌÒÙ]", "", tok).lower()


def _extract_ngrams(sentences: list[str], min_n: int = 2, max_n: int = 4) -> Counter:
    counter: Counter = Counter()
    for sent in sentences:
        tokens = [_clean_token(t) for t in sent.split()]
        tokens = [t for t in tokens if t and t not in _STOPWORDS and len(t) > 2]
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i : i + n])
                if _is_domain_relevant(phrase):
                    counter[phrase] += 1
    return counter


def _normalize(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip().lower())


def _deduplicate_candidates(candidates: list[str], existing: set[str]) -> list[str]:
    seen: set[str] = set(existing)
    out = []
    for c in candidates:
        key = _normalize(c)
        if key not in seen and len(key) > 4:
            seen.add(key)
            out.append(c)
    return out


# ── Source A: PDF corpus ─────────────────────────────────────────────────────

def mine_pdfs(min_freq: int = 2) -> list[tuple[str, int]]:
    pdf_dir = conf.paths.data_raw_pdfs
    pdfs = list(pdf_dir.rglob("*.pdf"))
    if not pdfs:
        print(f"⚠️  No PDFs found in {pdf_dir}")
        return []

    print(f"📄 Mining {len(pdfs)} PDFs...")
    miner = PdfMiner(min_length=20)
    all_sentences: list[str] = []
    for pdf in pdfs:
        try:
            raw = miner.extract_text_from_pdf(pdf)
            all_sentences.extend(miner.clean_and_segment(raw))
        except Exception as e:
            print(f"   ⚠️  {pdf.name}: {e}")

    print(f"   {len(all_sentences)} sentences → extracting n-grams...")
    counter = _extract_ngrams(all_sentences)
    results = [(phrase, freq) for phrase, freq in counter.most_common() if freq >= min_freq]
    print(f"   {len(results)} candidates (freq ≥ {min_freq})")
    return results


# ── Source C: web spider cache ───────────────────────────────────────────────

def mine_web_cache() -> list[str]:
    cache_path = conf.paths.data_synthetic / "checkpoints" / "competitor_crawl_cache.json"
    if not cache_path.exists():
        print("⚠️  Web crawl cache not found — run 'python scripts/leo.py data web-spider' first")
        return []

    cache = json.loads(cache_path.read_text())
    raw_terms = cache.get("terms", [])
    relevant = [t for t in raw_terms if _is_domain_relevant(t) and len(t.split()) >= 2]
    print(f"🌐 Web cache: {len(raw_terms)} raw terms → {len(relevant)} domain-relevant")
    return relevant


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract glossary term candidates")
    parser.add_argument("--top", type=int, default=60, help="Max candidates to show per source")
    parser.add_argument("--min-freq", type=int, default=2, help="Min PDF n-gram frequency")
    args = parser.parse_args()

    existing = {_normalize(item["term"]) for item in ROVER_GLOSSARY}
    print(f"\n📚 Existing glossary: {len(existing)} terms\n{'─' * 60}")

    # --- Source A: PDF ---
    print("\n[A] PDF corpus candidates")
    pdf_raw = mine_pdfs(min_freq=args.min_freq)
    pdf_candidates = _deduplicate_candidates(
        [phrase for phrase, _ in pdf_raw], existing
    )[: args.top]

    # --- Source C: Web cache ---
    print("\n[C] Web spider cache candidates")
    web_raw = mine_web_cache()
    web_candidates = _deduplicate_candidates(web_raw, existing | {_normalize(c) for c in pdf_candidates})[: args.top]

    # --- Print results ---
    rows = []

    print(f"\n{'═' * 60}")
    print(f"  SOURCE A — PDF corpus  ({len(pdf_candidates)} candidates)")
    print(f"{'═' * 60}")
    freq_map = {phrase: freq for phrase, freq in pdf_raw}
    for term in pdf_candidates:
        freq = freq_map.get(term, 1)
        ctx = _infer_context(term)
        print(f"  [{freq:>3}×]  {term:<40}  → {ctx}")
        rows.append({"source": "pdf", "term": term, "freq": freq, "suggested_context": ctx})

    print(f"\n{'═' * 60}")
    print(f"  SOURCE C — Web spider cache  ({len(web_candidates)} candidates)")
    print(f"{'═' * 60}")
    for term in web_candidates:
        ctx = _infer_context(term)
        print(f"          {term:<40}  → {ctx}")
        rows.append({"source": "web", "term": term, "freq": 1, "suggested_context": ctx})

    # --- Save CSV ---
    import csv
    out_path = PROJECT_ROOT / "reports" / "glossary_candidates.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "term", "freq", "suggested_context"])
        writer.writeheader()
        writer.writerows(rows)

    total = len(pdf_candidates) + len(web_candidates)
    print(f"\n✅ {total} candidates saved → {out_path}")
    print("   Review the CSV, then add approved terms to src/synthesis/glossary_data.py")


if __name__ == "__main__":
    main()
