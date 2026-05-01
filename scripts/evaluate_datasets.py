import argparse
from pathlib import Path
import pandas as pd
import json

from src.evaluation.stats import calculate_stats
from src.evaluation.llm_judge import evaluate_with_llm
from src.evaluation.perplexity import calculate_perplexity

def main():
    parser = argparse.ArgumentParser(description="Evaluate synthetic datasets")
    parser.add_argument("--data_dir", type=str, default="data/synthetic", help="Directory containing synthetic CSVs")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of samples to evaluate for LLM and Perplexity")
    parser.add_argument("--judge_model", type=str, default=None, help="Model ID for LLM Judge (e.g. ollama/qwen2.5:32b)")
    parser.add_argument("--nmt_baseline", type=str, default="facebook/nllb-200-distilled-600M", help="NMT model for perplexity scoring")
    parser.add_argument("--skip_judge", action="store_true", help="Skip the LLM-as-a-Judge evaluation (faster)")
    parser.add_argument("--skip_ppl", action="store_true", help="Skip the Perplexity evaluation (faster)")
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists() or not data_dir.is_dir():
        print(f"Directory {data_dir} does not exist.")
        return

    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return

    print(f"Found {len(csv_files)} datasets to evaluate in {data_dir}")
    
    results = {}

    for file_path in csv_files:
        print(f"\n{'='*50}\nEvaluating: {file_path.name}\n{'='*50}")
        try:
            df = pd.read_csv(file_path)
            
            # 1. Statistics & Coverage
            print("📊 Running Statistical Analysis...")
            stats_res = calculate_stats(df)
            
            # 2. LLM Judge
            judge_res = {}
            if not args.skip_judge:
                judge_res = evaluate_with_llm(df, sample_size=args.sample_size, model_id=args.judge_model)
                
            # 3. Perplexity
            ppl_res = {}
            if not args.skip_ppl:
                ppl_res = calculate_perplexity(df, sample_size=args.sample_size, model_name=args.nmt_baseline)
                
            results[file_path.name] = {
                **stats_res,
                **judge_res,
                **ppl_res
            }
            
        except Exception as e:
            print(f"❌ Error evaluating {file_path.name}: {e}")

    # Generate Report
    print("\n\n" + "#"*50)
    print("🏆 EVALUATION SUMMARY REPORT")
    print("#"*50)
    
    report_df = pd.DataFrame.from_dict(results, orient='index')
    
    # Print a nice table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(report_df.to_string())
    
    # Save to JSON
    out_path = data_dir / "evaluation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n✅ Detailed report saved to {out_path}")

if __name__ == "__main__":
    main()
