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
from src.training.model_module import NLLBFineTuner
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

    def run_pdf_mining(self):
        """Extracts text from PDFs and translates it using AI."""
        print("\n🦁 [Phase: PDF Mining]")
        pdf_dir = self.paths.data_raw_pdfs
        pdf_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
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
        print(f"   📚 Unique sentences: {len(unique_sentences)}")
        
        TRANSLATION_PROMPT = """[INST]
        Translate the following Italian technical sentence into English (EN), French (FR), and Spanish (ES).
        Sentence: "{text}"
        Format:
        EN: [Translation]
        FR: [Translation]
        ES: [Translation]
        [/INST]"""

        augmented_data = []
        for sent in tqdm(unique_sentences, desc="AI Translation"):
            try:
                raw_output = generator.pipe(TRANSLATION_PROMPT.format(text=sent), max_new_tokens=200, do_sample=False)[0]['generated_text']
                parts = raw_output.split("[/INST]")[-1].strip().split('\n')
                row = {"source_text": sent, "source_lang": "ita_Latn"}
                
                mapping = {"EN:": ("eng_Latn", "target_text_en"), "FR:": ("fra_Latn", "target_text_fr"), "ES:": ("ES:", "target_text_es")}
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
            except Exception: continue

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
        print(f"   🔍 Found {len(unique_terms)} unique terms.")
        
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
            sample_n = min(20, len(lang_df))
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
        model_module = NLLBFineTuner.load_from_checkpoint(checkpoint_path)
        model_module.setup()
        model_module.eval()
        if torch.cuda.is_available(): model_module.cuda()
        
        tokenizer = model_module.tokenizer
        model = model_module.model
        
        tokenizer.src_lang = src_lang
        inputs = tokenizer(text, return_tensors="pt").to(model_module.device)
        
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        if forced_bos_token_id is None:
            raise ValueError(f"Language code {tgt_lang} not found.")

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=conf.model.max_target_length,
                num_beams=5,
                early_stopping=True
            )

        result = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        print(f"Input: {text}")
        print(f"Output: {result}")
        return result

    def run_benchmark(self):
        """Runs a full comparison benchmark between Base and LEO models on WandB."""
        print("\n📊 [Phase: Benchmarking]")
        
        wandb.init(project="LEO-Translation", job_type="benchmark", name="Final-Model-Evaluation")
        
        data_path = conf.paths.data_gold / "test_set.csv"
        if not data_path.exists():
            print(f"❌ Test set not found at {data_path}. Run DataFactory().create_test_set() first.")
            return
            
        test_set = pd.read_csv(data_path)
        
        # Load Base
        model_name = "facebook/nllb-200-distilled-1.3B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config)
        
        # Load LEO (Adapter)
        adapter_path = conf.paths.output_dir / "final_adapter"
        if not adapter_path.exists():
            checkpoints = list(conf.paths.output_dir.glob("*.ckpt"))
            if not checkpoints: raise FileNotFoundError("No adapter/checkpoint found.")
            adapter_path = checkpoints[-1]
        
        # Evaluation Helper
        def eval_loop(model, prefix):
            preds, targets, sources = [], [], []
            bleu, chrf = SacreBLEUScore(), CHRFScore()
            for _, row in tqdm(test_set.iterrows(), total=len(test_set), desc=f"Eval {prefix}"):
                tokenizer.src_lang = row['source_lang']
                inputs = tokenizer(row['source_text'], return_tensors="pt").to(self.device)
                with torch.no_grad():
                    gen = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(row['target_lang']), max_new_tokens=128)
                decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
                preds.append(decoded); targets.append([row['target_text']]); sources.append(row['source_text'])
            return {"BLEU": bleu(preds, targets).item(), "CHRF": chrf(preds, targets).item(), "Samples": list(zip(sources, [t[0] for t in targets], preds))}

        res_base = eval_loop(base_model, "BASE")
        
        leo_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        leo_model.eval()
        res_leo = eval_loop(leo_model, "LEO")
        
        # Log to WandB
        wandb.log({"baseline_bleu": res_base['BLEU'], "leo_bleu": res_leo['BLEU'], "improvement_bleu": res_leo['BLEU'] - res_base['BLEU']})
        
        table = wandb.Table(columns=["Source (IT)", "Reference (Human)", "Base Model", "LEO Model"])
        for i in range(len(test_set)):
            table.add_data(res_leo['Samples'][i][0], res_leo['Samples'][i][1], res_base['Samples'][i][2], res_leo['Samples'][i][2])
        
        wandb.log({"benchmark_comparison": table})
        print("✅ Results logged to WandB!")
        wandb.finish()
