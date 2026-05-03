"""
extract_glossary_candidates.py — mina candidati per il glossario da due sorgenti:

  A. PDF corpus (n-gram frequency su frasi già estratte)
  C. Web spider cache (termini estratti dal BFS crawl)

Con --llm aggiunge due step Gemini:
  1. Filtraggio: scarta i falsi positivi estratti statisticamente
  2. Espansione: suggerisce termini mancanti per categorie scoperte

Output: lista stampata + CSV in reports/glossary_candidates.csv
Usage:
    python scripts/maintenance/extract_glossary_candidates.py
    python scripts/maintenance/extract_glossary_candidates.py --llm
    python scripts/maintenance/extract_glossary_candidates.py --llm --top 80 --min-freq 2
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

# ── Domain filter ────────────────────────────────────────────────────────────
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

_STOPWORDS = {
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
    "di", "del", "della", "dei", "degli", "delle", "da", "dal",
    "dalla", "in", "nel", "nella", "nei", "nelle",
    "a", "al", "alla", "ai", "alle", "con", "su", "per", "tra", "fra",
    "e", "ed", "o", "ma", "che", "come", "se", "quando", "dove",
    "non", "è", "sono", "viene", "si", "ha", "hanno", "deve",
    "può", "anche", "più", "molto", "questo", "questa", "questi",
    "queste", "tale", "tali", "ogni", "tutto", "tutti",
}

_VALID_CONTEXTS = {
    "structure", "sealing", "window_frame", "insulation", "shading",
    "construction", "installation", "hardware", "physics", "ecology",
    "performance", "problem", "renovation", "action", "product", "solution",
}

_CONTEXT_RULES = [
    ({"muffa", "condensa", "umidità", "degrado"}, "problem"),
    ({"riquali", "ristruttur", "retrofit", "non invasiv"}, "renovation"),
    ({"posa", "installaz", "montaggio", "fissagg", "cantiere"}, "installation"),
    ({"trasmittan", "ponte termic", "acustic", "abbattim"}, "physics"),
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
    return any(kw in phrase.lower() for kw in _DOMAIN_KW)


def _clean_token(tok: str) -> str:
    return re.sub(r"[^\wàèéìòùÀÈÉÌÒÙ]", "", tok).lower()


def _normalize(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip().lower())


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


def _deduplicate_candidates(candidates: list[str], existing: set[str]) -> list[str]:
    seen: set[str] = set(existing)
    out = []
    for c in candidates:
        key = _normalize(c)
        if key not in seen and len(key) > 4:
            seen.add(key)
            out.append(c)
    return out


def _parse_json_list(raw: str) -> list[dict]:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw.strip(), flags=re.M)
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        result = json.loads(cleaned[start : end + 1])
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


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


# ── LLM step 1: filter false positives ──────────────────────────────────────

_FILTER_SYSTEM = (
    "Sei un esperto tecnico nel settore serramenti, cassonetti coibentati e posa in opera di finestre in Italia."
)

_FILTER_USER = """\
Ti fornisco una lista di candidati estratti automaticamente da documenti tecnici con analisi statistica.
Valuta quali sono termini tecnici genuini e pertinenti per il settore cassonetti/serramenti/isolamento.

ESCLUDI:
- Termini troppo generici o ambigui fuori contesto (es. "sistema", "soluzione generale")
- Frammenti di testo non terminologici (es. "viene installato", "deve essere")
- Termini quasi-duplicati di altri già nella lista (tieni il più specifico)

Per ogni termine VALIDO restituisci un oggetto JSON con:
  "term": il termine (correggi maiuscole/minuscole se necessario),
  "context": uno tra {contexts},
  "reason": una parola che spiega perché è valido (es. "componente", "materiale", "norma", "processo")

Candidati:
{candidates}

Rispondi SOLO con un array JSON valido, nessun altro testo."""

_EXPAND_SYSTEM = _FILTER_SYSTEM

_EXPAND_USER = """\
Ecco il glossario tecnico attuale (termini già coperti — NON ripetere nessuno di questi):
{existing}

Identifica categorie SCOPERTE o SOTTORAPPRESENTATE e suggerisci esattamente 30 termini tecnici aggiuntivi.
Priorità a:
- Materiali specifici: poliuretano espanso, EPDM, PVC, acciaio zincato, polistirene, lana di roccia...
- Normative e certificazioni: UNI EN 14351, classi permeabilità all'aria, Uw, Rw, classe energetica...
- Varianti di prodotto Roverplastik non ancora coperte: monoblocco con zanzariera, cassonetto a taglio termico...
- Tecniche di posa specifiche: schiuma poliuretanica bicomponente, nastro butilico, tassello chimico...
- Fisica del vapore e condensazione: punto di rugiada, pressione parziale del vapore, permeanza...

Per ogni termine:
  "term": in italiano, 2-5 parole,
  "context": uno tra {contexts},
  "reason": perché manca e perché è importante

Rispondi SOLO con un array JSON valido di esattamente 30 elementi."""


def _llm_filter(candidates: list[dict], chat_fn, batch_size: int = 30) -> list[dict]:
    """Filter candidates through Gemini in batches. Returns validated list."""
    contexts_str = ", ".join(sorted(_VALID_CONTEXTS))
    validated: list[dict] = []
    batches = [candidates[i : i + batch_size] for i in range(0, len(candidates), batch_size)]

    print(f"\n🤖 [Gemini] Filtering {len(candidates)} candidates in {len(batches)} batch(es)...")
    for i, batch in enumerate(batches, 1):
        terms_list = "\n".join(f"- {c['term']}" for c in batch)
        prompt = _FILTER_USER.format(candidates=terms_list, contexts=contexts_str)
        try:
            raw = chat_fn(_FILTER_SYSTEM, prompt, max_tokens=2000)
            items = _parse_json_list(raw)
            for item in items:
                term = str(item.get("term", "")).strip()
                ctx = item.get("context", "product")
                if ctx not in _VALID_CONTEXTS:
                    ctx = _infer_context(term)
                reason = str(item.get("reason", "")).strip()
                if term:
                    validated.append({"term": term, "context": ctx, "reason": reason})
            print(f"   Batch {i}/{len(batches)}: {len(batch)} in → {len(items)} valid")
        except Exception as e:
            print(f"   ⚠️  Batch {i} failed: {e} — keeping unfiltered")
            validated.extend(batch)

    return validated


def _llm_expand(existing_glossary: list[dict], validated_candidates: list[dict], chat_fn) -> list[dict]:
    """Ask Gemini to suggest terms for uncovered categories."""
    contexts_str = ", ".join(sorted(_VALID_CONTEXTS))
    all_known = existing_glossary + validated_candidates
    existing_lines = "\n".join(f"- {item['term']} ({item.get('context', '?')})" for item in all_known)
    prompt = _EXPAND_USER.format(existing=existing_lines, contexts=contexts_str)

    print(f"\n🤖 [Gemini] Expanding glossary — suggesting new terms for uncovered categories...")
    try:
        raw = chat_fn(_EXPAND_SYSTEM, prompt, max_tokens=3000)
        items = _parse_json_list(raw)
        result = []
        for item in items:
            term = str(item.get("term", "")).strip()
            ctx = item.get("context", "product")
            if ctx not in _VALID_CONTEXTS:
                ctx = _infer_context(term)
            reason = str(item.get("reason", "")).strip()
            if term:
                result.append({"term": term, "context": ctx, "reason": reason})
        print(f"   {len(result)} new terms suggested")
        return result
    except Exception as e:
        print(f"   ⚠️  Expansion failed: {e}")
        return []


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract glossary term candidates")
    parser.add_argument("--top", type=int, default=60, help="Max candidates per source before LLM filter")
    parser.add_argument("--min-freq", type=int, default=2, help="Min PDF n-gram frequency")
    parser.add_argument("--llm", action="store_true", help="Enable Gemini filter + expansion (requires GEMINI_API_KEY)")
    args = parser.parse_args()

    existing_terms = list(ROVER_GLOSSARY)
    existing_keys = {_normalize(item["term"]) for item in existing_terms}
    print(f"\n📚 Existing glossary: {len(existing_keys)} terms\n{'─' * 60}")

    # ── Source A: PDF ──────────────────────────────────────────────────────
    print("\n[A] PDF corpus candidates")
    pdf_raw = mine_pdfs(min_freq=args.min_freq)
    freq_map = {phrase: freq for phrase, freq in pdf_raw}
    pdf_candidates_raw = _deduplicate_candidates(
        [phrase for phrase, _ in pdf_raw], existing_keys
    )[: args.top]
    pdf_candidates = [
        {"term": t, "context": _infer_context(t), "freq": freq_map.get(t, 1), "reason": ""}
        for t in pdf_candidates_raw
    ]

    # ── Source C: Web cache ────────────────────────────────────────────────
    print("\n[C] Web spider cache candidates")
    web_raw = mine_web_cache()
    web_candidates_raw = _deduplicate_candidates(
        web_raw, existing_keys | {_normalize(c["term"]) for c in pdf_candidates}
    )[: args.top]
    web_candidates = [
        {"term": t, "context": _infer_context(t), "freq": 1, "reason": ""}
        for t in web_candidates_raw
    ]

    all_candidates = pdf_candidates + web_candidates
    llm_expanded: list[dict] = []

    # ── LLM steps ─────────────────────────────────────────────────────────
    if args.llm:
        from src.synthesis.cloud_generator import CloudGenerator
        gen = CloudGenerator(model_id="google/gemini-2.5-flash")
        chat_fn = gen._chat_with_retry

        # Step 1: filter
        validated = _llm_filter(all_candidates, chat_fn)
        # Re-deduplicate after LLM may have normalised terms
        validated_keys = existing_keys.copy()
        deduped_validated = []
        for item in validated:
            key = _normalize(item["term"])
            if key not in validated_keys:
                validated_keys.add(key)
                deduped_validated.append(item)
        all_candidates = deduped_validated
        print(f"   After filter: {len(all_candidates)} candidates")

        # Step 2: expand
        llm_expanded = _llm_expand(existing_terms, all_candidates, chat_fn)
        # Deduplicate expansions against everything seen so far
        expansion_keys = validated_keys.copy()
        deduped_expanded = []
        for item in llm_expanded:
            key = _normalize(item["term"])
            if key not in expansion_keys:
                expansion_keys.add(key)
                deduped_expanded.append(item)
        llm_expanded = deduped_expanded

    # ── Print results ──────────────────────────────────────────────────────
    rows: list[dict] = []

    pdf_out = [c for c in all_candidates if c.get("freq", 1) > 1 or c["term"] in freq_map]
    web_out = [c for c in all_candidates if c not in pdf_out]

    print(f"\n{'═' * 65}")
    print(f"  SOURCE A — PDF corpus  ({len(pdf_out)} candidates)")
    print(f"{'═' * 65}")
    for c in pdf_out:
        tag = f"[{c.get('freq', 1):>3}×]"
        note = f"  ← {c['reason']}" if c.get("reason") else ""
        print(f"  {tag}  {c['term']:<42} → {c['context']}{note}")
        rows.append({"source": "pdf", **c})

    print(f"\n{'═' * 65}")
    print(f"  SOURCE C — Web spider cache  ({len(web_out)} candidates)")
    print(f"{'═' * 65}")
    for c in web_out:
        note = f"  ← {c['reason']}" if c.get("reason") else ""
        print(f"         {c['term']:<42} → {c['context']}{note}")
        rows.append({"source": "web", **c})

    if llm_expanded:
        print(f"\n{'═' * 65}")
        print(f"  GEMINI EXPANSION — {len(llm_expanded)} new terms for uncovered categories")
        print(f"{'═' * 65}")
        for c in llm_expanded:
            note = f"  ← {c['reason']}" if c.get("reason") else ""
            print(f"  [NEW]  {c['term']:<42} → {c['context']}{note}")
            rows.append({"source": "llm_expansion", **c})

    # ── Save CSV ───────────────────────────────────────────────────────────
    import csv
    out_path = PROJECT_ROOT / "reports" / "glossary_candidates.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source", "term", "freq", "context", "reason", "suggested_context"]
    for row in rows:
        row.setdefault("freq", 1)
        row.setdefault("reason", "")
        row["suggested_context"] = row.get("context", "product")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    print(f"\n✅ {total} candidates saved → {out_path}")
    print("   Review the CSV, delete unwanted rows, then add to src/synthesis/glossary_data.py")


if __name__ == "__main__":
    main()
