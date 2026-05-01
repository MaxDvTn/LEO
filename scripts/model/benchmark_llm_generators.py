import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from torchmetrics.text import CHRFScore, SacreBLEUScore
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.synthesis.generator import get_generator


DEFAULT_MODELS = [
    "ollama/qwen2.5:32b",
]

LANG_LABELS = {
    "eng_Latn": "English",
    "fra_Latn": "French",
    "spa_Latn": "Spanish",
    "ita_Latn": "Italian",
}

LANG_PREFIXES = {
    "eng_Latn": "EN:",
    "fra_Latn": "FR:",
    "spa_Latn": "ES:",
    "ita_Latn": "IT:",
}


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def _score(preds, refs):
    if not preds:
        return {"bleu": 0.0, "chrf": 0.0}
    targets = [[ref] for ref in refs]
    return {
        "bleu": float(SacreBLEUScore()(preds, targets).item()),
        "chrf": float(CHRFScore()(preds, targets).item()),
    }


def _metrics_by_group(predictions: pd.DataFrame, group_col: str):
    metrics = {}
    for value, group in predictions.groupby(group_col):
        metrics[str(value)] = {
            **_score(group["prediction"].astype(str).tolist(), group["target_text"].astype(str).tolist()),
            "rows": int(len(group)),
        }
    return metrics


def _extract_prefixed_line(text: str, prefix: str) -> str:
    for line in str(text).splitlines():
        clean = line.strip()
        if clean.startswith(prefix):
            return clean[len(prefix):].strip()
    return str(text).strip()


def _translate_with_generator(generator, source_text: str, target_lang: str) -> str:
    row = generator.translate_text(source_text)
    key_by_lang = {
        "eng_Latn": "target_text_en",
        "fra_Latn": "target_text_fr",
        "spa_Latn": "target_text_es",
        "ita_Latn": "source_text",
    }
    return row.get(key_by_lang.get(target_lang, "target_text_en")) or ""


def _translate_custom_prompt(generator, source_text: str, source_lang: str, target_lang: str) -> str:
    if not hasattr(generator, "_chat"):
        raise NotImplementedError(
            f"{generator.__class__.__name__} has no _chat() method; use --use-generator-translate."
        )
    source_label = LANG_LABELS.get(source_lang, source_lang)
    target_label = LANG_LABELS.get(target_lang, target_lang)
    prefix = LANG_PREFIXES.get(target_lang, "TR:")
    system = "You are a professional translator for technical construction documentation."
    user = (
        f"Translate the following technical text from {source_label} to {target_label}.\n"
        "Preserve product names, codes, measurements, and technical terminology.\n"
        f"Return exactly one line in this format:\n{prefix} [translation]\n\n"
        f"Text: \"{source_text}\""
    )
    out = generator._chat(system, user, max_tokens=256)
    return _extract_prefixed_line(out, prefix)


def benchmark_model(model_id: str, test_df: pd.DataFrame, output_dir: Path, use_generator_translate: bool):
    print(f"\n📊 Benchmark LLM generator: {model_id}")
    generator = get_generator(model_id)

    rows = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=_safe_name(model_id)):
        try:
            if use_generator_translate:
                prediction = _translate_with_generator(
                    generator,
                    source_text=str(row["source_text"]),
                    target_lang=str(row["target_lang"]),
                )
            else:
                prediction = _translate_custom_prompt(
                    generator,
                    source_text=str(row["source_text"]),
                    source_lang=str(row["source_lang"]),
                    target_lang=str(row["target_lang"]),
                )
            error = ""
        except Exception as exc:
            prediction = ""
            error = str(exc)
            print(f"⚠️ row {idx}: {error}")

        rows.append(
            {
                "row_id": idx,
                "model": model_id,
                "source_text": row["source_text"],
                "target_text": row["target_text"],
                "prediction": prediction,
                "source_lang": row["source_lang"],
                "target_lang": row["target_lang"],
                "origin": row.get("origin", ""),
                "term_keyword": row.get("term_keyword", ""),
                "error": error,
            }
        )

    pred_df = pd.DataFrame(rows)
    predictions_path = output_dir / f"{_safe_name(model_id)}__predictions.csv"
    pred_df.to_csv(predictions_path, index=False)

    valid_df = pred_df[pred_df["error"].astype(str) == ""]
    summary = {
        "model": model_id,
        "rows": int(len(pred_df)),
        "valid_rows": int(len(valid_df)),
        "error_rows": int(len(pred_df) - len(valid_df)),
        **_score(valid_df["prediction"].astype(str).tolist(), valid_df["target_text"].astype(str).tolist()),
        "by_target_lang": _metrics_by_group(valid_df, "target_lang") if not valid_df.empty else {},
        "predictions_path": str(predictions_path),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark prompt-based LLM generators on LEO test set.")
    parser.add_argument("--test-set", type=Path, default=PROJECT_ROOT / "data" / "gold" / "test_set.csv")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "benchmarks" / "llm_generators")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional quick-test row limit.")
    parser.add_argument(
        "--use-generator-translate",
        action="store_true",
        help="Use generator.translate_text(), which always asks for EN/FR/ES. Default uses a per-row target-language prompt.",
    )
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_set).dropna(subset=["source_text", "target_text", "source_lang", "target_lang"])
    if args.sample_size is not None:
        test_df = test_df.head(args.sample_size)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for model_id in args.models:
        summaries.append(
            benchmark_model(
                model_id=model_id,
                test_df=test_df,
                output_dir=output_dir,
                use_generator_translate=args.use_generator_translate,
            )
        )

    summary_path = output_dir / "summary.json"
    summary_csv_path = output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    pd.DataFrame(
        [
            {
                "model": item["model"],
                "rows": item["rows"],
                "valid_rows": item["valid_rows"],
                "error_rows": item["error_rows"],
                "bleu": item["bleu"],
                "chrf": item["chrf"],
                "predictions_path": item["predictions_path"],
            }
            for item in summaries
        ]
    ).to_csv(summary_csv_path, index=False)

    print(f"\n✅ Benchmark summary: {summary_path}")
    print(f"✅ Benchmark table: {summary_csv_path}")


if __name__ == "__main__":
    main()
