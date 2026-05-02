# LEO Synthetic Dataset Model Selection Report

Date: 2026-05-02

## Executive Summary

The current local runs produced complete and structurally clean glossary datasets for six Ollama models plus Gemini:

- `ollama/qwen2.5:32b`
- `ollama/aya-expanse:8b`
- `ollama/mistral-nemo`
- `ollama/mistral-small3.2`
- `ollama/gemma3:27b`
- `ollama/phi4`
- `google/gemini-2.5-flash`

Each model generated 138 rows: 46 Italian source sentences x 3 target languages. All datasets are balanced across English, French, and Spanish, and no exact duplicate translation rows were detected.

The strongest provisional models are:

1. `ollama/mistral-small3.2`
2. `ollama/gemma3:27b`
3. `ollama/qwen2.5:32b`

The best practical dataset direction is not to choose a single model immediately. The best next dataset should be a curated ensemble built from `mistral-small3.2`, `gemma3:27b`, and `qwen2.5:32b`, with `aya-expanse:8b` used selectively for multilingual coverage checks.

Gemini was re-run after the parser/thinking fix and now produces a valid dataset. Its benchmark score is mid-pack: structurally clean and useful as a cloud reference/diversity source, but not better than the top local models on the current benchmark.

## Evidence Base

Analyzed run:

`runs/generation_suite/20260502T080910178352Z`

Gemini follow-up run:

`runs/generation_suite/20260502T091526197919Z`

Generated datasets:

`data/synthetic/runs/glossary_synthetic__*.csv`

Important limitation: the benchmark used `data/gold/test_set.csv`, which currently has 607 rows and is still contaminated with synthetic/PDF-mined rows. It is not yet the new gold-only test set. Because `--benchmark-sample-size 20` was used, the benchmark evaluated only the first 20 rows, all `eng_Latn`. The benchmark is therefore useful as a relative smoke test, not as a final model-selection authority.

## Dataset Structural Quality

| Model | Rows | Unique Sources | Duplicates | Target Languages | Avg Source Words | Avg Target Words |
|---|---:|---:|---:|---|---:|---:|
| `aya-expanse:8b` | 138 | 46 | 0 | 46 EN / 46 FR / 46 ES | 23.7 | 24.9 |
| `gemma3:27b` | 138 | 46 | 0 | 46 EN / 46 FR / 46 ES | 30.5 | 32.1 |
| `mistral-nemo` | 138 | 46 | 0 | 46 EN / 46 FR / 46 ES | 18.8 | 19.6 |
| `mistral-small3.2` | 138 | 46 | 0 | 46 EN / 46 FR / 46 ES | 18.5 | 19.8 |
| `phi4` | 138 | 46 | 0 | 46 EN / 46 FR / 46 ES | 21.0 | 22.2 |
| `qwen2.5:32b` | 138 | 46 | 0 | 46 EN / 46 FR / 46 ES | 19.5 | 20.3 |
| `gemini-2.5-flash` | 138 | 46 | 0 | 46 EN / 46 FR / 46 ES | 25.7 | 27.7 |

No model showed obvious structural noise:

- identical source/target rows: 0
- leaked `IT:` / `EN:` / `FR:` / `ES:` markers: 0
- placeholder text: 0
- very short target rows: 0
- very long target rows: 0

This means the current parser and filters are working for both Ollama and Gemini outputs.

## Benchmark Results

Benchmark sample:

- 20 rows
- all English target rows
- old test set, not gold-only

| Rank | Model | BLEU | chrF | Valid Rows |
|---:|---|---:|---:|---:|
| 1 | `mistral-small3.2` | 0.3581 | 0.5643 | 20/20 |
| 2 | `gemma3:27b` | 0.3493 | 0.5576 | 20/20 |
| 3 | `qwen2.5:32b` | 0.3529 | 0.5575 | 20/20 |
| 4 | `aya-expanse:8b` | 0.3278 | 0.5457 | 20/20 |
| 5 | `gemini-2.5-flash` | 0.3287 | 0.5430 | 20/20 |
| 6 | `phi4` | 0.3244 | 0.5370 | 20/20 |
| 7 | `mistral-nemo` | 0.3226 | 0.5324 | 20/20 |

Interpretation:

- `mistral-small3.2` is the best benchmark performer on this sample.
- `gemma3:27b` and `qwen2.5:32b` are effectively tied on chrF.
- `gemini-2.5-flash` is valid after the fix but does not beat the best local models on this benchmark.
- The gap between the top three is small; final selection should depend on dataset quality and downstream fine-tuning results, not only these 20-row benchmark metrics.

## Qualitative Observations

`mistral-small3.2`

- Produces concise, instruction-following technical sentences.
- Best provisional benchmark result.
- Good candidate for the primary synthetic generator.

`gemma3:27b`

- Produces longer, more descriptive source sentences.
- Good technical richness, useful for expanding model exposure to complex sentence structure.
- Should be included, but watch for overlong or over-elaborate phrasing.

`qwen2.5:32b`

- Strong technical coverage and solid benchmark score.
- Some translations show less natural phrasing in French/Spanish samples, but it remains valuable for domain terminology coverage.

`aya-expanse:8b`

- Multilingual-first model.
- Structurally clean and efficient.
- Slightly weaker benchmark score, but useful as a diversity or multilingual sanity-check generator.

`gemini-2.5-flash`

- Now produces valid JSON-backed rows after disabling Gemini thinking budget through LiteLLM.
- Generates polished and fluent technical sentences, usually longer than Mistral/Qwen and shorter than Gemma.
- Benchmark score is close to `aya-expanse:8b`, above `phi4` and `mistral-nemo`, but below the top local trio.
- Useful as an external-cloud reference model and possible 10% ensemble component, not as the primary generator based on this run.
- LLM judge scores in the evaluation report are all zero because the judge did not return parseable score lines; those values are invalid for ranking.

`mistral-nemo`

- Efficient and clean, but weaker than `mistral-small3.2`.
- Useful if speed/cost matters, but not a top dataset-quality choice.

`phi4`

- Good structured output model.
- Benchmark and qualitative evidence do not justify making it a primary generator.
- Can remain a fallback for JSON/format reliability tests.

## Model Choice

Recommended primary set for the next synthetic dataset:

1. `ollama/mistral-small3.2`
2. `ollama/gemma3:27b`
3. `ollama/qwen2.5:32b`

Recommended secondary checks:

4. `ollama/aya-expanse:8b`
5. `google/gemini-2.5-flash`

Models to deprioritize:

- `ollama/mistral-nemo`
- `ollama/phi4`

Do not remove them permanently; they are still useful baselines, but they should not drive the definitive training dataset unless later evaluations contradict this run.

## Dataset Strategy

The best dataset should be an ensemble, not a single-model dump.

Recommended composition for the next training candidate:

- 40% `mistral-small3.2`
- 30% `gemma3:27b`
- 20% `qwen2.5:32b`
- 10% `aya-expanse:8b` or `gemini-2.5-flash`

Rationale:

- `mistral-small3.2` provides the strongest concise technical generation.
- `gemma3:27b` adds sentence complexity and richer context.
- `qwen2.5:32b` adds domain breadth and robust local generation.
- `aya-expanse:8b` adds multilingual bias and helps avoid English-centric generation behavior.
- Gemini adds a high-fluency external reference style and is now structurally clean.

## Required Next Steps

1. Regenerate the test set as gold-only.

Current `data/gold/test_set.csv` is still old and contaminated. The code has been fixed, but the file must be regenerated.

Run:

```bash
python -c "from src.pipelines.factory import DataFactory; DataFactory().create_test_set()"
```

Warning: current `rover_gold_dataset.csv` has only 5 rows, all English. A gold-only test set will be methodologically cleaner but too small for final selection. More human validation is needed.

2. Fix LLM judge parsing or disable judge for model selection.

The Gemini run generated a valid dataset, but `avg_fluency`, `avg_adequacy`, and `avg_terminology` are zero because the judge did not return parseable score lines. Treat those judge metrics as invalid until the judge parser is hardened.

3. Run a balanced benchmark.

The current benchmark sampled only English because it used `head(20)`. The benchmark script should sample per target language or use a larger/full set after test set cleanup.

4. Add a dataset scorer before training.

Before fine-tuning, score each generated row with:

- parse validity
- source/target length ratio
- term presence in Italian source
- target language detection
- LLM judge fluency/adequacy/terminology
- duplicate or near-duplicate source detection

5. Build a candidate ensemble CSV.

Create one training candidate from the recommended model mix. Keep provenance columns:

- `model_id`
- `term`
- `context`
- `doc_type`
- `prompt_version`
- `created_at`
- `raw_output`

6. Fine-tune NLLB on three candidate datasets.

Recommended experiments:

- A: `mistral-small3.2` only
- B: `mistral-small3.2 + gemma3:27b + qwen2.5:32b`
- C: ensemble plus Gemini/Aya

The final winner should be selected by downstream NLLB validation/test performance, not generator benchmark alone.

## Decision

Provisional winner: `ollama/mistral-small3.2`.

Best engineering choice for the next dataset: ensemble of `mistral-small3.2`, `gemma3:27b`, and `qwen2.5:32b`, with `gemini-2.5-flash` or `aya-expanse:8b` as a small diversity component.

Do not use the current benchmark as final proof. It is a useful smoke test, but final selection requires:

- regenerated gold-only test set,
- balanced multilingual benchmark,
- fixed judge scoring,
- and one actual fine-tuning comparison.
