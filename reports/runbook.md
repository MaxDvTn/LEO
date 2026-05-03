# LEO — Runbook

Sequenza completa degli script da eseguire, nell'ordine corretto.

---

## 0. Setup

```bash
conda activate LEO
cd /home/mbosetti/LEO
```

---

## 1. Generazione dataset sintetico

### 1a. PDF mining — multi-modello, multi-direzionale

Traduce ~17k frasi IT/unknown → EN/FR/ES (+ reverse pair gratuito)
e ~15.9k frasi native EN/FR/ES → IT dai PDF tecnici.
Riprende automaticamente dal checkpoint se interrotto.

```bash
python scripts/data/run_generation_suite.py \
    --models ollama/mistral-small3.2 ollama/qwen2.5:32b google/gemini-2.5-flash ollama/gemma3:27b ollama/aya-expanse:8b \
    --dataset-kind pdf \
    --skip-benchmark \
    --skip-judge \
    --skip-ppl
```

Output: `data/synthetic/runs/rover_pdf_augmented__<model>__<timestamp>.csv`
Checkpoint: `data/synthetic/checkpoints/rover_pdf_augmented__<model>.csv`

### 1b. Web spider — frasi autentiche + cache termini

Il web spider crawla siti competitor/tecnici, traduce frasi autentiche e salva
termini tecnici grezzi nella crawl cache. Non genera più frasi sintetiche dai
termini: quella parte è stata spostata nel glossario.

```bash
LEO_GEN_MODEL_ID=google/gemini-2.5-flash python scripts/leo.py data web-spider
```

Output: `data/synthetic/runs/competitor_synthetic__<model>__<timestamp>.csv`
Active file: `data/synthetic/competitor_synthetic.csv`
Crawl cache: `data/synthetic/checkpoints/competitor_crawl_cache.json`
Translation checkpoint: `data/synthetic/checkpoints/competitor_synthetic__<model>__checkpoint.csv`
Log: `logs/web_spider__<model>__<timestamp>.log`

Per monitorare durante la run:

```bash
tail -f logs/web_spider__google_gemini-2.5-flash__<timestamp>.log
```

```bash
watch -n 5 'python -c "import pandas as pd; p=\"data/synthetic/checkpoints/competitor_synthetic__google_gemini-2.5-flash__checkpoint.csv\"; df=pd.read_csv(p); print(len(df)); print(df[\"prompt_version\"].value_counts())"'
```

Il CSV competitor viene pulito automaticamente al salvataggio. Per ripulire il
file attivo manualmente:

```bash
python scripts/leo.py maintenance clean-competitor
```

### 1c. Glossario — statico + termini web

Il glossario usa i termini statici in `src/synthesis/glossary_data.py` e aggiunge
i termini web puliti presenti in `competitor_crawl_cache.json`.

```bash
LEO_GEN_MODEL_ID=google/gemini-2.5-flash python scripts/leo.py data generate
```

Per rigenerare con più modelli:

```bash
python scripts/data/run_generation_suite.py \
    --models ollama/mistral-small3.2 ollama/gemma3:27b google/gemini-2.5-flash \
    --dataset-kind glossary \
    --skip-benchmark \
    --skip-judge \
    --skip-ppl
```

Output: `data/synthetic/runs/glossary_synthetic__<model>__<timestamp>.csv`

## 2. Build ensemble

Merge di tutti i run (glossario + PDF + competitor), deduplicazione esatta,
near-dedup semantico (cosine sim > 0.90), filtro qualità.

```bash
python scripts/leo.py data build-ensemble
```

Output: `data/synthetic/ensemble_training_set.csv`

---

## 3. Ricreazione test set gold

Crea il test set escludendo data leakage (source texts già nel training).
Da eseguire **dopo** il build-ensemble.

```bash
python scripts/leo.py data test-set
```

Output: `data/gold/test_set.csv`

---

## 4. Training

### Fresh start (nessun checkpoint esistente)

```bash
python scripts/leo.py train --fresh
```

### Ripresa da checkpoint

```bash
python scripts/leo.py train
```

Il trainer cerca automaticamente l'ultimo checkpoint in
`checkpoints_facebook_seamless-m4t-v2-large/`.

Configurazione attiva (`src/common/config.py`):
- Model: `facebook/seamless-m4t-v2-large`
- LoRA: r=32, α=64, target=`out_proj`
- Batch: 12 · Grad accum: 3 · LR: 5e-5
- Precision: bf16-mixed · Epochs: 20

---

## 5. Benchmark

Confronta il modello base vs LEO fine-tuned sul gold test set.
Richiede che `leo_hf_release/` sia stato esportato prima.

```bash
python scripts/leo.py benchmark
```

Output: WandB run con BLEU · chrF · METEOR per lingua, heatmap, tabella regressioni.

---

## 6. Inferenza

```bash
python scripts/leo.py infer \
    --text "Il cassonetto coibentato garantisce ottime prestazioni termiche." \
    --src-lang ita_Latn \
    --tgt-lang eng_Latn
```

Con checkpoint specifico:

```bash
python scripts/leo.py infer \
    --text "..." \
    --src-lang ita_Latn \
    --tgt-lang fra_Latn \
    --checkpoint checkpoints_facebook_seamless-m4t-v2-large/leo-epoch=05.ckpt
```

---

## Note

| Script | Riprende dal checkpoint? | Parallelismo |
|---|---|---|
| `run_generation_suite.py --dataset-kind pdf` | Sì (ogni 300 righe) | `generator.num_workers` |
| `leo.py data web-spider` | Sì (crawl cache + checkpoint traduzioni) | `generator.num_workers` |
| `run_generation_suite.py --dataset-kind glossary` | No; usa termini web già cacheati | `generator.num_workers` |
| `leo.py train` | Sì (Lightning checkpoint) | GPU |
| `leo.py data build-ensemble` | No (< 1 min) | CPU |
| `leo.py data test-set` | No (< 1 sec) | CPU |

**Modelli configurati per il PDF mining:**

| Modello | Workers | Note |
|---|---|---|
| `ollama/aya-expanse:8b` | 4 | più veloce |
| `ollama/mistral-small3.2` | 2 | migliore benchmark |
| `ollama/gemma3:27b` | 2 | 100% glossario coverage |
| `ollama/qwen2.5:32b` | 2 | top-2 benchmark |
| `google/gemini-2.5-flash` | 8 | cloud, molto veloce |
