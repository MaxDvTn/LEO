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

### 1b. Glossario — già eseguito il 2026-05-02

Se si vuole rigenerare con nuovi modelli:

```bash
python scripts/data/run_generation_suite.py \
    --models ollama/mistral-small3.2 ollama/qwen2.5:32b google/gemini-2.5-flash \
    --dataset-kind glossary \
    --skip-benchmark \
    --skip-judge \
    --skip-ppl
```

Output: `data/synthetic/runs/glossary_synthetic__<model>__<timestamp>.csv`

### 1c. Web spider (opzionale)

```bash
python scripts/leo.py data web-spider
```

```bash
LEO_GEN_MODEL_ID=google/gemini-2.5-flash python scripts/leo.py data web-spider 

Output: `data/synthetic/runs/competitor_synthetic__<model>__<timestamp>.csv`

---

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
| `run_generation_suite.py --dataset-kind glossary` | No | `generator.num_workers` |
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
