import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from scripts.data.compare_synthetic import compare
from scripts.model.benchmark_llm_generators import benchmark_model
from src.common.config import conf
from src.pipelines.factory import DataFactory


def _set_model(model_id: str):
    conf.gen.model_id = model_id


def _generate_dataset(model_id: str, dataset_kind: str):
    _set_model(model_id)
    factory = DataFactory()
    if dataset_kind == "glossary":
        return factory.run_glossary_gen()
    if dataset_kind == "web":
        return factory.run_web_spider()
    if dataset_kind == "pdf":
        return factory.run_pdf_mining()
    raise ValueError(f"Unsupported dataset kind: {dataset_kind}")


def _compare_to_text(old_path: Path, new_path: Path, sample_size: int) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        compare(old_path, new_path, sample_size=sample_size)
    return buffer.getvalue()


def _write_comparisons(generated: list, output_dir: Path, sample_size: int):
    comparison_dir = output_dir / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_paths = []

    for i, left in enumerate(generated):
        for right in generated[i + 1:]:
            left_path = Path(left["run_path"])
            right_path = Path(right["run_path"])
            out_path = comparison_dir / f"{left['safe_model']}__vs__{right['safe_model']}.txt"
            out_path.write_text(_compare_to_text(left_path, right_path, sample_size), encoding="utf-8")
            comparison_paths.append(str(out_path))
            print(f"📎 Comparison: {out_path}")

    return comparison_paths


def _safe_model(model_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model_id).strip("_")


def _run_benchmarks(models: list, test_set: Path, output_dir: Path, sample_size: int | None, use_generator_translate: bool):
    test_df = pd.read_csv(test_set).dropna(subset=["source_text", "target_text", "source_lang", "target_lang"])
    if sample_size is not None:
        test_df = test_df.head(sample_size)

    benchmark_dir = output_dir / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for model_id in models:
        summaries.append(
            benchmark_model(
                model_id=model_id,
                test_df=test_df,
                output_dir=benchmark_dir,
                use_generator_translate=use_generator_translate,
            )
        )

    summary_json = benchmark_dir / "summary.json"
    summary_csv = benchmark_dir / "summary.csv"
    summary_json.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
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
    ).to_csv(summary_csv, index=False)
    print(f"✅ Benchmark summary: {summary_json}")
    print(f"✅ Benchmark table: {summary_csv}")
    return {"summary_json": str(summary_json), "summary_csv": str(summary_csv)}


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic datasets with multiple models, compare them, and benchmark the same models."
    )
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--dataset-kind", choices=["glossary", "web", "pdf"], default="glossary")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "generation_suite")
    parser.add_argument("--test-set", type=Path, default=PROJECT_ROOT / "data" / "gold" / "test_set.csv")
    parser.add_argument("--compare-sample-size", type=int, default=5)
    parser.add_argument("--benchmark-sample-size", type=int, default=None, help="Row limit for benchmark (default: full test set).")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument(
        "--use-generator-translate",
        action="store_true",
        help="Benchmark with generator.translate_text() instead of per-row target-language prompts.",
    )
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for model_id in args.models:
        print(f"\n🧪 Dataset generation with {model_id}")
        canonical_path, run_path = _generate_dataset(model_id, args.dataset_kind)
        if not run_path:
            print(f"⚠️ No dataset generated for {model_id}")
            continue
        generated.append(
            {
                "model": model_id,
                "safe_model": _safe_model(model_id),
                "canonical_path": str(canonical_path),
                "run_path": str(run_path),
            }
        )

    comparison_paths = []
    if not args.skip_compare and len(generated) > 1:
        comparison_paths = _write_comparisons(generated, output_dir, args.compare_sample_size)

    benchmark = None
    if not args.skip_benchmark:
        benchmark = _run_benchmarks(
            models=args.models,
            test_set=args.test_set,
            output_dir=output_dir,
            sample_size=args.benchmark_sample_size,
            use_generator_translate=args.use_generator_translate,
        )

    manifest = {
        "run_id": run_id,
        "dataset_kind": args.dataset_kind,
        "models": args.models,
        "generated": generated,
        "comparisons": comparison_paths,
        "benchmark": benchmark,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ Suite manifest: {manifest_path}")


if __name__ == "__main__":
    main()
