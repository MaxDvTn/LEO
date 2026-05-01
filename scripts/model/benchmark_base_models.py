import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from torchmetrics.text import CHRFScore, SacreBLEUScore
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.training.dataset_module import is_seamless_model, normalize_lang_code


DEFAULT_MODELS = [
    "facebook/nllb-200-3.3B",
    "facebook/seamless-m4t-v2-large",
]


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def _load_processor(model_name: str):
    if is_seamless_model(model_name):
        return AutoProcessor.from_pretrained(model_name)
    return AutoTokenizer.from_pretrained(model_name)


def _load_model(model_name: str, load_in_4bit: bool):
    kwargs = {"device_map": "auto"}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16
    return AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs).eval()


def _generate(model, processor, model_name: str, source_text: str, source_lang: str, target_lang: str, max_new_tokens: int):
    source_lang = normalize_lang_code(source_lang, model_name)
    target_lang = normalize_lang_code(target_lang, model_name)

    if hasattr(processor, "src_lang"):
        processor.src_lang = source_lang
    if hasattr(processor, "tgt_lang"):
        processor.tgt_lang = target_lang

    input_kwargs = {"return_tensors": "pt", "truncation": True, "max_length": 256}
    if is_seamless_model(model_name):
        input_kwargs["src_lang"] = source_lang
        generation_kwargs = {"tgt_lang": target_lang}
    else:
        generation_kwargs = {"forced_bos_token_id": processor.convert_tokens_to_ids(target_lang)}

    inputs = processor(source_text, **input_kwargs).to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **generation_kwargs,
        )
    return processor.decode(generated[0].tolist(), skip_special_tokens=True)


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


def benchmark_model(model_name: str, test_df: pd.DataFrame, output_dir: Path, max_new_tokens: int, load_in_4bit: bool):
    print(f"\n📊 Benchmark base model: {model_name}")
    processor = _load_processor(model_name)
    model = _load_model(model_name, load_in_4bit=load_in_4bit)

    rows = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=_safe_name(model_name)):
        try:
            prediction = _generate(
                model=model,
                processor=processor,
                model_name=model_name,
                source_text=str(row["source_text"]),
                source_lang=str(row["source_lang"]),
                target_lang=str(row["target_lang"]),
                max_new_tokens=max_new_tokens,
            )
            error = ""
        except Exception as exc:
            prediction = ""
            error = str(exc)
            print(f"⚠️ row {idx}: {error}")

        rows.append(
            {
                "row_id": idx,
                "model": model_name,
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
    model_safe = _safe_name(model_name)
    predictions_path = output_dir / f"{model_safe}__predictions.csv"
    pred_df.to_csv(predictions_path, index=False)

    valid_df = pred_df[pred_df["error"].astype(str) == ""]
    summary = {
        "model": model_name,
        "rows": int(len(pred_df)),
        "valid_rows": int(len(valid_df)),
        "error_rows": int(len(pred_df) - len(valid_df)),
        **_score(valid_df["prediction"].astype(str).tolist(), valid_df["target_text"].astype(str).tolist()),
        "by_target_lang": _metrics_by_group(valid_df, "target_lang") if not valid_df.empty else {},
        "predictions_path": str(predictions_path),
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark non-finetuned base translation models on LEO test set.")
    parser.add_argument("--test-set", type=Path, default=PROJECT_ROOT / "data" / "gold" / "test_set.csv")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "benchmarks" / "base_models")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--sample-size", type=int, default=None, help="Optional quick-test row limit.")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantized loading.")
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_set).dropna(subset=["source_text", "target_text", "source_lang", "target_lang"])
    if args.sample_size is not None:
        test_df = test_df.head(args.sample_size)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for model_name in args.models:
        summaries.append(
            benchmark_model(
                model_name=model_name,
                test_df=test_df,
                output_dir=output_dir,
                max_new_tokens=args.max_new_tokens,
                load_in_4bit=not args.no_4bit,
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
