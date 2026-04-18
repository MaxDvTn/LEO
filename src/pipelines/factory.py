import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import torch
from torchmetrics.text import SacreBLEUScore, CHRFScore
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import wandb
import os

# Project Imports
from src.data_mining.pdf_processor import PdfMiner
from src.synthesis.generator import SyntheticGenerator
from src.synthesis.prompts import GENERATION_PROMPT_TEMPLATE
from src.synthesis.glossary_data import get_terms_list
from src.data_mining.competitor_spider import CompetitorSpider
from src.training.trainer_engine import TrainerEngine
from src.training.model_module import SeamlessFineTuner
from src.common.config import conf

class DataFactory:
    """
    Unified orchestrator for all data operations in the L.E.O. project.
    Consolidates PDF mining, Web spidering, Synthetic generation, and Data formatting.
    """

    def __init__(self):
        self.paths = conf.paths
        self.synthetic_dir = self.paths.data_synthetic
        self.gold_dir = self.paths.data_gold

    def _get_gold_sources(self):
        """Loads all source texts and terms currently in the gold dataset to avoid duplication."""
        gold_path = self.gold_dir / "rover_gold_dataset.csv"
        if not gold_path.exists():
            return set(), set()
        
        try:
            df = pd.read_csv(gold_path)
            # Collect existing source sentences
            known_sources = set()
            if 'source_text' in df.columns:
                known_sources = set(df['source_text'].dropna().astype(str).str.strip())
            
            # Collect existing specific terms
            known_terms = set()
            if 'term_keyword' in df.columns:
                known_terms = set(df['term_keyword'].dropna().astype(str).str.strip())
                
            return known_sources, known_terms
        except Exception as e:
            print(f"⚠️ Could not load gold dataset for filtering: {e}")
            return set(), set()

    def run_pdf_mining(self):
        """Extracts text from PDFs and translates it using AI."""
        print("\n🦁 [Phase: PDF Mining]")
        pdf_dir = self.paths.data_raw_pdfs
        pdf_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_files = list(pdf_dir.rglob("*.pdf"))
        if not pdf_files:
            print(f"⚠️  No PDFs found in {pdf_dir}")
            return

        miner = PdfMiner(min_length=20)
        generator = SyntheticGenerator()
        all_sentences = []
        
        for pdf_file in pdf_files:
            print(f"   📄 Reading: {pdf_file.name}...")
            raw_text = miner.extract_text_from_pdf(pdf_file)
            all_sentences.extend(miner.clean_and_segment(raw_text))

        unique_sentences = list(set(all_sentences))
        
        # --- ANTI-DUPLICATION FILTER ---
        known_sources, _ = self._get_gold_sources()
        original_count = len(unique_sentences)
        unique_sentences = [s for s in unique_sentences if s.strip() not in known_sources]
        skipped = original_count - len(unique_sentences)
        print(f"   📚 Unique sentences: {len(unique_sentences)} (Skipped {skipped} already validated)")
        # -------------------------------
        
        TRANSLATION_PROMPT = """[INST]
        Translate the following Italian technical sentence into English (EN), French (FR), and Spanish (ES).
        Sentence: "{text}"
        Format:
        EN: [Translation]
        FR: [Translation]
        ES: [Translation]
        [/INST]"""

        # Batch generation setup
        BATCH_SIZE = 8
        all_prompts = []
        for sent in unique_sentences:
            all_prompts.append(TRANSLATION_PROMPT.format(text=sent))

        total_batches = (len(all_prompts) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"   🚀 Starting Batched Inference (Batches: {total_batches})...")

        augmented_data = []

        for i in tqdm(range(0, len(all_prompts), BATCH_SIZE), desc="AI Translation"):
            batch_prompts = all_prompts[i : i + BATCH_SIZE]
            batch_sentences = unique_sentences[i : i + BATCH_SIZE]

            try:
                # Use pipeline batching
                outputs = generator.pipe(
                    batch_prompts,
                    max_new_tokens=200,
                    do_sample=False,
                    batch_size=BATCH_SIZE
                )

                for j, out in enumerate(outputs):
                    raw_text = out[0]['generated_text'] if isinstance(out, list) else out['generated_text']
                    sent = batch_sentences[j]
                    
                    parts = raw_text.split("[/INST]")[-1].strip().split('\n')
                    row = {"source_text": sent, "source_lang": "ita_Latn"}

                    mapping = {"EN:": ("eng_Latn", "target_text_en"), "FR:": ("fra_Latn", "target_text_fr"), "ES:": ("spa_Latn", "target_text_es")}
                    for line in parts:
                        for prefix, (lang_code, key) in mapping.items():
                            if line.startswith(prefix): row[key] = line.replace(prefix, "").strip()

                    # Append available translations
                    for lang_code, key in [("eng_Latn", "target_text_en"), ("fra_Latn", "target_text_fr"), ("spa_Latn", "target_text_es")]:
                        if key in row:
                            augmented_data.append({
                                "source_text": sent, "target_text": row[key],
                                "source_lang": "ita_Latn", "target_lang": lang_code, "origin": "pdf_mining"
                            })
            except Exception as e:
                print(f"❌ Batch Error: {e}")
                continue

        if augmented_data:
            out_path = self.synthetic_dir / "rover_pdf_augmented.csv"
            pd.DataFrame(augmented_data).to_csv(out_path, index=False)
            print(f"✅ Saved PDF data: {out_path}")

    def run_web_spider(self):
        """Scrapes competitor websites for terms and sentences."""
        print("\n🕷️ [Phase: Web Spidering]")
        spider = CompetitorSpider()
        # The spider internally uses conf.spider.target_urls and saves to synthetic dir
        # We can trigger it by calling its main logic if we refactor its main to a method, 
        # but for now we can just use its existing structure or call it as a subprocess if preferred.
        # Let's assume we want to call it directly.
        target_urls = conf.spider.target_urls
        all_terms = []
        for url in target_urls:
            terms = spider.scrape_site(url)
            all_terms.extend(terms)
        
        unique_terms = list(set(all_terms))
        
        # --- ANTI-DUPLICATION FILTER ---
        _, known_terms = self._get_gold_sources()
        original_count = len(unique_terms)
        unique_terms = [t for t in unique_terms if t.strip() not in known_terms]
        skipped = original_count - len(unique_terms)
        print(f"   🔍 Found {len(unique_terms)} unique terms. (Skipped {skipped} already validated)")
        # -------------------------------
        
        sentences = []
        for term in tqdm(unique_terms, desc="Generating from terms"):
            sentences.extend(spider.generate_sentences_from_term(term))
            
        if sentences:
            df = pd.DataFrame(sentences)
            out_path = self.synthetic_dir / "competitor_synthetic.csv"
            df.to_csv(out_path, index=False)
            print(f"✅ Saved Web data: {out_path}")

    def run_glossary_gen(self):
        """Generates synthetic data from the internal glossary."""
        print("\n🤖 [Phase: Glossary Generation]")
        generator = SyntheticGenerator()
        terms = get_terms_list()
        generated_data = []

        # Configuration for massive generation
        SAMPLES_PER_TERM = 80
        BATCH_SIZE = 8
        
        all_prompts = []
        all_terms_ref = []
        
        print(f"   🔨 Preparing prompts: {len(terms)} terms x {SAMPLES_PER_TERM} variations = {len(terms)*SAMPLES_PER_TERM} samples.")
        
        for term in terms:
            for _ in range(SAMPLES_PER_TERM):
                all_prompts.append(GENERATION_PROMPT_TEMPLATE.format(term=term))
                all_terms_ref.append(term)
        
        # Batch generation
        # We process in chunks to verify progress and memory safety
        total_batches = (len(all_prompts) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"   🚀 Starting Inference (Batches: {total_batches})...")
        
        # Using generator.pipe directly with batch_size
        # Note: transformers pipeline handles batching if we pass a KeyDataset or list
        # For simplicity and progress tracking, we chunk manually
        
        for i in tqdm(range(0, len(all_prompts), BATCH_SIZE), desc="Generating Batches"):
            batch_prompts = all_prompts[i : i + BATCH_SIZE]
            batch_terms = all_terms_ref[i : i + BATCH_SIZE]
            
            try:
                # Higher temperature for variety since we repeat prompts
                outputs = generator.pipe(
                    batch_prompts, 
                    max_new_tokens=250, 
                    do_sample=True, 
                    temperature=0.85, 
                    top_p=0.95,
                    batch_size=BATCH_SIZE
                )
                
                for j, out in enumerate(outputs):
                    # pipeline returns list of dicts, or list of list of dicts
                    # for text-generation it's usually [{'generated_text': '...'}]
                    text = out[0]['generated_text'] if isinstance(out, list) else out['generated_text']
                    term = batch_terms[j]
                    
                    parsed = generator.parse_output(text, term)
                    if parsed and parsed['source_text']:
                        for lang, key in [("eng_Latn", "target_text_en"), ("fra_Latn", "target_text_fr"), ("spa_Latn", "target_text_es")]:
                            if parsed.get(key):
                                generated_data.append({
                                    "source_text": parsed['source_text'], "target_text": parsed[key],
                                    "source_lang": "ita_Latn", "target_lang": lang, "term_keyword": term
                                })
            except Exception as e:
                print(f"❌ Batch Error: {e}")
                continue

        if generated_data:
            out_path = self.synthetic_dir / "rover_synthetic_multilingual.csv"
            df = pd.DataFrame(generated_data)
            df.to_csv(out_path, index=False)
            print(f"✅ Saved Glossary data: {out_path} ({len(df)} rows)")

    def create_test_set(self):
        """Creates a balanced test set from Gold and Synthetic data."""
        print("\n🧪 [Phase: Test Set Creation]")
        gold_path = self.gold_dir / "rover_gold_dataset.csv"
        
        dfs = []
        if gold_path.exists():
            dfs.append(pd.read_csv(gold_path))
        
        all_files = list(self.synthetic_dir.glob("*.csv"))
        for f in all_files:
            dfs.append(pd.read_csv(f))
            
        if not dfs: return
        source_df = pd.concat(dfs, ignore_index=True)

        test_rows = []
        for lang in source_df['target_lang'].unique():
            lang_df = source_df[source_df['target_lang'] == lang]
            sample_n = max(20, int(len(lang_df) * 0.01))
            if sample_n > 0:
                test_rows.append(lang_df.sample(n=sample_n, random_state=42))
        
        if test_rows:
            test_df = pd.concat(test_rows, ignore_index=True)
            out_path = self.gold_dir / "test_set.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            test_df.to_csv(out_path, index=False)
            print(f"✅ Created Test Set: {out_path} ({len(test_df)} rows)")

    def run_full_pipeline(self):
        """Runs all stages in sequence."""
        print("🚀 [STARTING FULL DATA PIPELINE]")
        self.run_pdf_mining()
        self.run_web_spider()
        self.run_glossary_gen()
        self.create_test_set()
        print("\n✨ [PIPELINE COMPLETED]")

class ModelFactory:
    """
    Unified orchestrator for all model operations in the L.E.O. project.
    Consolidates Training, Inference, and Benchmarking.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train(self):
        """Starts the training process using the TrainerEngine."""
        print("\n🏋️ [Phase: Training]")
        # Force log visibility
        os.environ['TRANSFORMERS_VERBOSITY'] = 'info'
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
        os.environ["WANDB_MODE"] = "online"
        os.environ["WANDB_SILENT"] = "false"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        engine = TrainerEngine()
        
        # Check for existing checkpoints to resume
        checkpoints = sorted(list(conf.paths.output_dir.glob("*.ckpt")))
        ckpt_path = None
        if checkpoints:
            # Prefer 'last.ckpt' if exists, else newest
            last_ckpt = conf.paths.output_dir / "last.ckpt"
            if last_ckpt.exists():
                ckpt_path = str(last_ckpt)
            else:
                ckpt_path = str(checkpoints[-1])
            print(f"🔄 Resuming training from: {ckpt_path}")
            
        engine.run(ckpt_path=ckpt_path)

    def translate(self, text, src_lang, tgt_lang, checkpoint_path=None):
        """Translates a single string using a specific checkpoint."""
        print(f"\n🔮 [Phase: Inference] {src_lang} -> {tgt_lang}")
        
        if checkpoint_path is None:
            # Try to find last checkpoint
            checkpoints = list(conf.paths.output_dir.glob("*.ckpt"))
            if not checkpoints:
                raise FileNotFoundError("❌ No checkpoints found in output dir.")
            checkpoint_path = str(checkpoints[-1])
            print(f"🔎 Using latest checkpoint: {checkpoint_path}")

        # Load model module
        model_module = SeamlessFineTuner.load_from_checkpoint(checkpoint_path)
        model_module.setup()
        model_module.eval()
        if torch.cuda.is_available(): model_module.cuda()
        
        processor = model_module.tokenizer
        model = model_module.model
        
        # SeamlessM4T map
        lang_map = {"ita_Latn": "ita", "eng_Latn": "eng", "fra_Latn": "fra", "spa_Latn": "spa"}
        seamless_src = lang_map.get(src_lang, src_lang)
        seamless_tgt = lang_map.get(tgt_lang, tgt_lang)

        # Process input
        inputs = processor(text, src_lang=seamless_src, return_tensors="pt").to(model_module.device)
        
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                tgt_lang=seamless_tgt,
                max_new_tokens=conf.model.max_target_length,
                early_stopping=True
            )

        result = processor.decode(generated_tokens[0].tolist(), skip_special_tokens=True)
        print(f"Input: {text}")
        print(f"Output: {result}")
        return result

    def run_benchmark(self):
        """Runs a full comparison benchmark between Base and LEO models on WandB."""
        print("\n📊 [Phase: Benchmarking]")
        
        wandb.init(
            project="LEO-Translation", 
            job_type="benchmark", 
            name="Final-Model-Evaluation",
            config=conf.model.__dict__
        )
        
        data_path = conf.paths.data_gold / "test_set.csv"
        if not data_path.exists():
            print(f"❌ Test set not found at {data_path}. Run DataFactory().create_test_set() first.")
            return
            
        test_set = pd.read_csv(data_path)
        
        # Load Base Model
        model_name = conf.model.model_name
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_name)
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config)
        
        # Base Model Evaluation Helper
        def eval_loop(model, prefix):
            preds, targets, sources = [], [], []
            lang_pairs = [] # To track language direction
            bleu, chrf = SacreBLEUScore(), CHRFScore()
            
            # Use NLTK for METEOR
            from nltk.translate import meteor_score
            import nltk
            
            lang_map = {"ita_Latn": "ita", "eng_Latn": "eng", "fra_Latn": "fra", "spa_Latn": "spa"}
            
            for _, row in tqdm(test_set.iterrows(), total=len(test_set), desc=f"Eval {prefix}"):
                seamless_src = lang_map.get(row['source_lang'], row['source_lang'])
                seamless_tgt = lang_map.get(row['target_lang'], row['target_lang'])
                
                inputs = processor(row['source_text'], src_lang=seamless_src, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    try:
                        gen = model.generate(**inputs, tgt_lang=seamless_tgt, max_new_tokens=128)
                    except Exception as e:
                        print(f"⚠️ Error in generation for row {row['source_text'][:50]}... : {e}")
                        import gc
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    
                decoded = str(processor.decode(gen[0].tolist(), skip_special_tokens=True))
                preds.append(decoded); targets.append([str(row['target_text'])]); sources.append(str(row['source_text']))
                lang_pairs.append(f"{seamless_src}->{seamless_tgt}")
                
                # Periodic memory cleanup
                if _ % 50 == 0:
                    import gc
                    torch.cuda.empty_cache()
                    gc.collect()
                
            # Compute global METEOR
            meteor_scores = [meteor_score.meteor_score([str(t[0]).split()], str(p).split()) for t, p in zip(targets, preds)]
            global_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
            
            # Compute per-language metrics
            lang_metrics = {}
            for lp in set(lang_pairs):
                lp_indices = [i for i, x in enumerate(lang_pairs) if x == lp]
                lp_preds = [preds[i] for i in lp_indices]
                lp_targets = [targets[i] for i in lp_indices]
                lp_meteor = sum([meteor_scores[i] for i in lp_indices]) / len(lp_indices)
                lang_metrics[lp] = {
                    "BLEU": bleu(lp_preds, lp_targets).item(),
                    "CHRF": chrf(lp_preds, lp_targets).item(),
                    "METEOR": lp_meteor
                }
                
            return {
                "BLEU": bleu(preds, targets).item(), 
                "CHRF": chrf(preds, targets).item(), 
                "METEOR": global_meteor,
                "Samples": list(zip(sources, [t[0] for t in targets], preds, lang_pairs)),
                "LangMetrics": lang_metrics
            }

        # Evaluate Base Model Before Adapter is Loaded
        print("Evaluating Base Model...")
        res_base = eval_loop(base_model, "BASE")
        
        # Load LEO (Adapter)
        adapter_path = conf.paths.output_dir / "leo_hf_release"
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found at {adapter_path}. Run scripts/export_to_hf.py first.")
        
        # Load the base model with PEFT adapters applied properly
        print("Evaluating LEO Model...")
        leo_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        leo_model.eval()
        
        # LEO Model is already evaluated directly from peft
        res_leo = eval_loop(leo_model, "LEO")
        
        # Log to WandB
        wandb.log({
            "baseline_bleu": res_base['BLEU'], 
            "leo_bleu": res_leo['BLEU'], 
            "improvement_bleu": res_leo['BLEU'] - res_base['BLEU'],
            "baseline_chrf": res_base['CHRF'],
            "leo_chrf": res_leo['CHRF'],
            "improvement_chrf": res_leo['CHRF'] - res_base['CHRF'],
            "baseline_meteor": res_base['METEOR'],
            "leo_meteor": res_leo['METEOR'],
            "improvement_meteor": res_leo['METEOR'] - res_base['METEOR']
        })
        
        # Log Correlation Matrix / Heatmap Data
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np

        metrics = ["BLEU", "CHRF", "METEOR"]
        directions = list(res_leo['LangMetrics'].keys())
        
        # Create a heatmap grid measuring language performance vs baseline differences
        heatmap_data = np.zeros((len(directions), len(metrics)))
        for i, direction in enumerate(directions):
            for j, metric in enumerate(metrics):
                heatmap_data[i, j] = res_leo["LangMetrics"][direction][metric] - res_base["LangMetrics"][direction][metric]
                
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, xticklabels=metrics, yticklabels=directions, cmap="RdYlGn", ax=ax)
        plt.title("LEO Improvement over Baseline per Language Pair")
        plt.tight_layout()
        wandb.log({"Language_Performance_Heatmap": wandb.Image(fig)})
        plt.close(fig)
        
        table = wandb.Table(columns=["Direction", "Source (IT)", "Reference (Human)", "Base Model", "LEO Model"])
        regression_table = wandb.Table(columns=["Direction", "Source (IT)", "Reference", "Base Model", "LEO Model", "Base CHRF", "LEO CHRF", "Diff"])
        
        chrf_scorer = CHRFScore()
        
        import string
        from collections import Counter
        
        def tokenize_and_clean(text):
            return [w.strip(string.punctuation).lower() for w in text.split() if w.strip(string.punctuation).lower()]

        missing_words = Counter()
        hallucinated_words = Counter()
        
        for i in range(len(test_set)):
            src_text = res_leo['Samples'][i][0]
            ref_text = res_leo['Samples'][i][1]
            base_pred = res_base['Samples'][i][2]
            leo_pred = res_leo['Samples'][i][2]
            direction = res_leo['Samples'][i][3]
            
            table.add_data(direction, src_text, ref_text, base_pred, leo_pred)
            
            # Record Token-Level Errors for LEO Predictions
            ref_words = set(tokenize_and_clean(ref_text))
            pred_words = set(tokenize_and_clean(leo_pred))
            
            missing_words.update(ref_words - pred_words) # Target words LEO forgot
            hallucinated_words.update(pred_words - ref_words) # Guessed words LEO invented
            
            # Compute sentence-level scores to find regressions
            base_score = chrf_scorer([base_pred], [[ref_text]]).item()
            leo_score = chrf_scorer([leo_pred], [[ref_text]]).item()
            
            diff = leo_score - base_score
            if diff < 0: # LEO is worse
                regression_table.add_data(direction, src_text, ref_text, base_pred, leo_pred, round(base_score, 4), round(leo_score, 4), round(diff, 4))
        
        # Plotting Top 15 Most Hallucinated / Missing Words
        fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Hallucinations
        hw_labels = [w for w, c in hallucinated_words.most_common(15)]
        hw_counts = [c for w, c in hallucinated_words.most_common(15)]
        sns.barplot(x=hw_counts, y=hw_labels, ax=axes[0], palette="Reds_d")
        axes[0].set_title("Top 15 Predicted Words Missing in Reference (False Positives)")
        axes[0].set_xlabel("Frequency")
        
        # Missed Words
        mw_labels = [w for w, c in missing_words.most_common(15)]
        mw_counts = [c for w, c in missing_words.most_common(15)]
        sns.barplot(x=mw_counts, y=mw_labels, ax=axes[1], palette="Blues_d")
        axes[1].set_title("Top 15 Reference Words Missing in Prediction (False Negatives)")
        axes[1].set_xlabel("Frequency")
        
        plt.tight_layout()
        wandb.log({
            "benchmark_comparison": table,
            "regressions_table": regression_table,
            "Word_Level_Errors_Analysis": wandb.Image(fig2)
        })
        plt.close(fig2)
        print("✅ Results logged to WandB!")
        wandb.finish()
