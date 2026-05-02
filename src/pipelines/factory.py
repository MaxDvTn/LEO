import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import torch
import re
import shutil
from datetime import datetime, timezone
from torchmetrics.text import SacreBLEUScore, CHRFScore
from transformers import AutoModelForSeq2SeqLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import wandb
import os

# Project Imports
from src.data_mining.pdf_processor import PdfMiner
from src.synthesis.generator import get_generator
from src.synthesis.glossary_data import get_terms_list
from src.data_mining.competitor_spider import CompetitorSpider
from src.training.trainer_engine import TrainerEngine
from src.training.model_module import SeamlessFineTuner
from src.training.dataset_module import is_seamless_model, normalize_lang_code
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

    def _safe_model_name(self):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", conf.gen.model_id).strip("_")

    def _is_relevant_web_term(self, term: str) -> bool:
        normalized = re.sub(r"\s+", " ", str(term).strip().lower())
        if not normalized:
            return False

        blacklist = {
            "privacy", "cookie", "cookies", "contatti", "contact", "contacts",
            "azienda", "company", "news", "blog", "login", "area riservata",
            "newsletter", "catalogo", "catalogue", "download", "scarica",
            "home", "menu", "search", "cerca", "seguici", "facebook",
            "instagram", "linkedin", "youtube", "termini", "condizioni",
            "copyright", "credits", "lavora con noi",
        }
        if normalized in blacklist:
            return False
        if any(part in blacklist for part in normalized.split()):
            return False

        domain_keywords = {
            "casson", "coibent", "monobloc", "serrament", "finestr", "foro",
            "telaio", "controtelaio", "avvolg", "tapparella", "frangisole",
            "zanzar", "guarnizion", "sigill", "isol", "termic", "acustic",
            "tenuta", "profil", "soglia", "bancale", "sottobancale",
            "posa", "giunto", "vapore", "membrana", "nastro", "schiuma",
            "oscurante", "facciata", "lucernar", "deventer", "presystem",
        }
        return any(keyword in normalized for keyword in domain_keywords)

    def _detect_source_lang(self, text: str) -> str:
        """Lightweight PDF language guard for the languages used in this project."""
        lowered = f" {str(text).lower()} "
        stopwords = {
            "ita_Latn": [" il ", " lo ", " la ", " gli ", " le ", " di ", " del ", " della ", " per ", " con ", " una ", " che ", " in "],
            "eng_Latn": [" the ", " and ", " of ", " for ", " with ", " from ", " this ", " that ", " is ", " are "],
            "fra_Latn": [" le ", " la ", " les ", " des ", " pour ", " avec ", " dans ", " une ", " que ", " est "],
            "spa_Latn": [" el ", " la ", " los ", " las ", " para ", " con ", " una ", " que ", " del ", " en "],
        }
        scores = {
            lang: sum(lowered.count(token) for token in tokens)
            for lang, tokens in stopwords.items()
        }
        best_lang, best_score = max(scores.items(), key=lambda item: item[1])
        return best_lang if best_score > 0 else "unknown"

    def _looks_bad_translation(self, source: str, target: str) -> bool:
        source_norm = re.sub(r"\s+", " ", str(source).strip().lower())
        target_norm = re.sub(r"\s+", " ", str(target).strip().lower())
        if not source_norm or not target_norm:
            return True
        if source_norm == target_norm:
            return True
        if any(marker in target_norm for marker in ["target_text_", "translation]", "[english", "[french", "[spanish"]):
            return True
        if re.search(r"\b(it|en|fr|es)\s*:", target_norm):
            return True

        source_words = source_norm.split()
        target_words = target_norm.split()
        if len(source_words) < 4 or len(target_words) < 3:
            return True

        ratio = len(target_words) / max(len(source_words), 1)
        return ratio < 0.35 or ratio > 2.8

    def _normalize_synthetic_df(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        required = ["source_text", "target_text", "source_lang", "target_lang"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{label} is missing required columns: {missing}")

        before = len(df)
        df = df.copy()
        if "model_id" not in df.columns:
            df["model_id"] = conf.gen.model_id
        if "prompt_version" not in df.columns:
            df["prompt_version"] = "unknown"
        if "created_at" not in df.columns:
            df["created_at"] = datetime.now(timezone.utc).isoformat()

        for col in required + [
            "origin",
            "term",
            "context",
            "doc_type",
            "model_id",
            "prompt_version",
            "created_at",
            "raw_output",
            "detected_source_lang",
        ]:
            if col in df.columns:
                df[col] = df[col].astype("string").str.strip()

        df = df.dropna(subset=required)
        for col in required:
            df = df[df[col].astype(str).str.len() > 0]

        bad_mask = df.apply(
            lambda row: self._looks_bad_translation(row["source_text"], row["target_text"]),
            axis=1,
        )
        df = df[~bad_mask]
        df = df.drop_duplicates(subset=required).reset_index(drop=True)
        removed = before - len(df)
        if removed:
            print(f"🧹 Removed {removed} invalid/duplicate rows from {label}")
        return df

    def _wide_to_long(self, df: pd.DataFrame, origin: str, source_lang: str = "ita_Latn") -> pd.DataFrame:
        lang_map = [
            ("target_text_en", "eng_Latn"),
            ("target_text_fr", "fra_Latn"),
            ("target_text_es", "spa_Latn"),
        ]
        rows = []
        created_at = datetime.now(timezone.utc).isoformat()
        metadata_cols = ["term", "context", "doc_type", "raw_output"]

        for _, row in df.iterrows():
            for col, tgt_lang in lang_map:
                value = row.get(col)
                if pd.notna(value) and str(value).strip():
                    row_source_lang = row.get("source_lang", source_lang)
                    if pd.isna(row_source_lang) or not str(row_source_lang).strip():
                        row_source_lang = source_lang
                    item = {
                        "source_text": row["source_text"],
                        "target_text": value,
                        "source_lang": row_source_lang,
                        "target_lang": tgt_lang,
                        "origin": origin,
                        "model_id": conf.gen.model_id,
                        "prompt_version": "json_v1",
                        "created_at": created_at,
                    }
                    for meta_col in metadata_cols:
                        if meta_col in row and pd.notna(row.get(meta_col)):
                            item[meta_col] = row.get(meta_col)
                    rows.append(item)
        return pd.DataFrame(rows)

    def _load_gold_dataset(self) -> pd.DataFrame | None:
        gold_path = self.gold_dir / "rover_gold_dataset.csv"
        if not gold_path.exists():
            print(f"⚠️  Gold dataset not found: {gold_path}")
            return None
        return pd.read_csv(gold_path)

    def _save_synthetic_dataset(self, df: pd.DataFrame, filename: str, label: str):
        """Save the active synthetic CSV and keep timestamped copies for comparison."""
        self.synthetic_dir.mkdir(parents=True, exist_ok=True)
        df = self._normalize_synthetic_df(df, label)
        if df.empty:
            print(f"⚠️  No valid rows for {label}, skipping save.")
            return None, None

        out_path = self.synthetic_dir / filename
        stem = out_path.stem
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

        archive_dir = self.synthetic_dir / "archive"
        runs_dir = self.synthetic_dir / "runs"
        archive_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            archived_path = archive_dir / f"{stem}__previous__{timestamp}.csv"
            shutil.copy2(out_path, archived_path)
            print(f"📦 Archived previous {label}: {archived_path}")

        run_path = runs_dir / f"{stem}__{self._safe_model_name()}__{timestamp}.csv"
        df.to_csv(run_path, index=False)
        df.to_csv(out_path, index=False)
        print(f"✅ Saved {label}: {out_path} ({len(df)} rows)")
        print(f"🧾 Versioned copy: {run_path}")
        return out_path, run_path

    def run_pdf_mining(self):
        """Extracts text from PDFs and translates it using AI."""
        print("\n🦁 [Phase: PDF Mining]")
        pdf_dir = self.paths.data_raw_pdfs
        pdf_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = list(pdf_dir.rglob("*.pdf"))
        if not pdf_files:
            print(f"⚠️  No PDFs found in {pdf_dir}")
            return None, None

        miner = PdfMiner(min_length=20)
        generator = get_generator()
        all_sentences = []

        for pdf_file in pdf_files:
            print(f"   📄 Reading: {pdf_file.name}...")
            raw_text = miner.extract_text_from_pdf(pdf_file)
            all_sentences.extend(miner.clean_and_segment(raw_text))

        unique_sentences = sorted(set(all_sentences))

        # --- ANTI-DUPLICATION FILTER ---
        known_sources, _ = self._get_gold_sources()
        original_count = len(unique_sentences)
        unique_sentences = [s for s in unique_sentences if s.strip() not in known_sources]
        skipped = original_count - len(unique_sentences)
        print(f"   📚 Unique sentences: {len(unique_sentences)} (Skipped {skipped} already validated)")
        # -------------------------------

        print(f"   🚀 Starting Translation ({len(unique_sentences)} sentences)...")
        augmented_data = []
        lang_map = [("eng_Latn", "target_text_en"), ("fra_Latn", "target_text_fr"), ("spa_Latn", "target_text_es")]
        skipped_non_italian = 0
        created_at = datetime.now(timezone.utc).isoformat()

        for sent in tqdm(unique_sentences, desc="AI Translation"):
            try:
                detected_lang = self._detect_source_lang(sent)
                if detected_lang not in {"ita_Latn", "unknown"}:
                    skipped_non_italian += 1
                    continue
                row = generator.translate_text(sent)
                for lang_code, key in lang_map:
                    if row.get(key):
                        augmented_data.append({
                            "source_text": sent,
                            "target_text": row[key],
                            "source_lang": "ita_Latn",
                            "detected_source_lang": detected_lang,
                            "target_lang": lang_code,
                            "origin": "pdf_mining",
                            "model_id": conf.gen.model_id,
                            "prompt_version": "translate_json_v1",
                            "created_at": created_at,
                        })
            except Exception as e:
                print(f"❌ Translation Error: {e}")
                continue
        if skipped_non_italian:
            print(f"   🌐 Skipped {skipped_non_italian} non-Italian PDF sentences")

        if augmented_data:
            return self._save_synthetic_dataset(
                pd.DataFrame(augmented_data),
                "rover_pdf_augmented.csv",
                "PDF data",
            )
        return None, None

    def run_web_spider(self):
        """Scrapes competitor websites for terms and generates synthetic sentences."""
        print("\n🕷️ [Phase: Web Spidering]")
        spider = CompetitorSpider()
        target_urls = conf.spider.target_urls
        all_terms = []
        for url in target_urls:
            all_terms.extend(spider.scrape_site(url))

        unique_terms = sorted(set(all_terms))
        pre_filter_count = len(unique_terms)
        unique_terms = [t for t in unique_terms if self._is_relevant_web_term(t)]
        filtered_out = pre_filter_count - len(unique_terms)

        # --- ANTI-DUPLICATION FILTER ---
        _, known_terms = self._get_gold_sources()
        original_count = len(unique_terms)
        unique_terms = [t for t in unique_terms if t.strip() not in known_terms]
        skipped = original_count - len(unique_terms)
        print(
            f"   🔍 Found {len(unique_terms)} relevant unique terms. "
            f"(Filtered {filtered_out} web-noise terms, skipped {skipped} already validated)"
        )
        # -------------------------------

        if not unique_terms:
            return None, None

        generator = get_generator()
        df = generator.generate_dataset(terms=unique_terms, num_variants=conf.gen.num_variants)
        if df.empty:
            return None, None

        long_df = self._wide_to_long(df, origin="web_spider")
        if not long_df.empty:
            return self._save_synthetic_dataset(
                long_df,
                "competitor_synthetic.csv",
                "Web data",
            )
        return None, None

    def create_test_set(self):
        """Creates a balanced test set from human-validated gold data only."""
        print("\n🧪 [Phase: Test Set Creation]")
        source_df = self._load_gold_dataset()
        if source_df is None or source_df.empty:
            return
        required = ["source_text", "target_text", "source_lang", "target_lang"]
        source_df = source_df.dropna(subset=required)

        test_rows = []
        for lang in source_df['target_lang'].unique():
            lang_df = source_df[source_df['target_lang'] == lang]
            # Bug A fix: cap sample_n to len(lang_df) so we never request more rows than available
            sample_n = min(max(20, int(len(lang_df) * 0.01)), len(lang_df))
            if sample_n > 0:
                test_rows.append(lang_df.sample(n=sample_n, random_state=42))

        if test_rows:
            test_df = pd.concat(test_rows, ignore_index=True)
            out_path = self.gold_dir / "test_set.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            test_df.to_csv(out_path, index=False)
            print(f"✅ Created gold-only Test Set: {out_path} ({len(test_df)} rows)")

    def _dedup_near_duplicates(self, df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
        """Remove source groups whose source_text is nearly identical to an already-kept source.

        The dataset is long-format (one Italian source × N target languages). Operates
        on unique source texts, then re-includes all target-language rows for kept sources.

        Uses sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) for semantic
        similarity when available; falls back to TF-IDF char n-grams with threshold=0.60.
        """
        if len(df) <= 1:
            return df

        source_df = df[["source_text"]].drop_duplicates().reset_index(drop=True)
        texts = source_df["source_text"].astype(str).tolist()
        if len(texts) <= 1:
            return df

        # --- similarity matrix ---
        sim_matrix = None
        effective_threshold = threshold
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
            model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            embeddings = model.encode(texts, show_progress_bar=False)
            sim_matrix = cos_sim(embeddings)
        except Exception:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity as cos_sim
                tfidf = TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=1
                ).fit_transform(texts)
                sim_matrix = cos_sim(tfidf)
                effective_threshold = 0.60  # char n-gram similarity scale differs
                logger.warning("sentence-transformers unavailable — using TF-IDF char n-gram fallback (threshold=0.60)")
            except Exception:
                logger.warning("No similarity backend available — skipping near-duplicate filter")
                return df

        kept: list[int] = []
        for i in range(len(texts)):
            if not kept:
                kept.append(i)
                continue
            max_sim = max(sim_matrix[i][j] for j in kept)
            if max_sim < effective_threshold:
                kept.append(i)

        removed = len(texts) - len(kept)
        if removed:
            print(f"🔁 Near-duplicate filter: removed {removed} similar source texts (threshold={effective_threshold})")

        kept_sources = set(source_df.iloc[kept]["source_text"].astype(str))
        return df[df["source_text"].astype(str).isin(kept_sources)].reset_index(drop=True)

    def build_ensemble(
        self,
        run_paths: list | None = None,
        weights: dict | None = None,
        dedup_threshold: float = 0.85,
    ) -> Path | None:
        """Build a training-ready ensemble CSV from multiple model run datasets.

        Args:
            run_paths: explicit list of CSV paths from runs/. If None, uses all
                       glossary_synthetic__*.csv files in synthetic/runs/.
            weights:   {model_id: fraction} for proportional sampling, e.g.
                       {"ollama/mistral-small3.2": 0.4, "ollama/gemma3:27b": 0.3, ...}.
                       If None, all rows from all files are included equally.
            dedup_threshold: cosine-similarity cutoff for near-duplicate removal.
        """
        print("\n🧩 [Phase: Build Ensemble Dataset]")
        runs_dir = self.synthetic_dir / "runs"
        if run_paths is None:
            run_paths = sorted(runs_dir.glob("glossary_synthetic__*.csv"))

        if not run_paths:
            print(f"⚠️  No run CSVs found in {runs_dir}")
            return None

        dfs = []
        for path in run_paths:
            try:
                df = pd.read_csv(path)
                dfs.append(df)
                model_label = df["model_id"].iloc[0] if "model_id" in df.columns else path.stem
                print(f"   📂 {path.name}: {len(df)} rows  [{model_label}]")
            except Exception as e:
                logger.warning(f"Could not load {path}: {e}")

        if not dfs:
            return None

        if weights:
            total = sum(len(d) for d in dfs)
            sampled = []
            for df in dfs:
                model = df["model_id"].iloc[0] if "model_id" in df.columns else "unknown"
                frac = weights.get(model, 1.0 / len(dfs))
                n = max(1, int(total * frac))
                sampled.append(df.sample(n=min(n, len(df)), random_state=42))
            combined = pd.concat(sampled, ignore_index=True)
        else:
            combined = pd.concat(dfs, ignore_index=True)

        print(f"\n   Combined: {len(combined)} rows from {len(dfs)} run files")

        # 1. Exact dedup on source+target pair
        before = len(combined)
        combined = combined.drop_duplicates(
            subset=["source_text", "target_text", "target_lang"]
        ).reset_index(drop=True)
        if len(combined) < before:
            print(f"   Exact duplicates removed: {before - len(combined)}")

        # 2. Cross-model near-duplicate filter on source_text
        combined = self._dedup_near_duplicates(combined, threshold=dedup_threshold)

        # 3. Final quality filter
        combined = self._normalize_synthetic_df(combined, "ensemble")

        out_path = self.synthetic_dir / "ensemble_training_set.csv"
        combined.to_csv(out_path, index=False)
        print(f"✅ Ensemble saved: {out_path} ({len(combined)} rows)")
        return out_path

    def run_glossary_gen(self):
        """Generates synthetic sentences from the domain glossary using the configured backend."""
        print("\n📖 [Phase: Glossary Generation]")
        generator = get_generator()

        # Keep all glossary terms, including already validated gold terms, so repeated
        # generation can build diverse contexts around high-value terminology.
        terms = get_terms_list(with_context=True)
        print(f"   📝 Generating for {len(terms)} terms (×{conf.gen.num_variants} variants)...")

        df = generator.generate_dataset(terms=terms, num_variants=conf.gen.num_variants)
        if not df.empty:
            long_df = self._wide_to_long(df, origin="glossary")
            if long_df.empty:
                print("⚠️  No valid translations in generated data, skipping save.")
                return None, None
            return self._save_synthetic_dataset(
                long_df,
                "glossary_synthetic.csv",
                "glossary data",
            )
        return None, None

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

        # Prefer end-of-epoch checkpoints over last.ckpt (which may be mid-epoch)
        ckpt_path = None
        epoch_ckpts = sorted(conf.paths.output_dir.glob("leo-*.ckpt"))
        last_ckpt = conf.paths.output_dir / "last.ckpt"
        if epoch_ckpts:
            ckpt_path = str(epoch_ckpts[-1])
            print(f"🔄 Resuming from epoch checkpoint: {ckpt_path}")
        elif last_ckpt.exists():
            ckpt_path = str(last_ckpt)
            print(f"⚠️  Resuming from last.ckpt (mid-epoch — results may vary): {ckpt_path}")

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

        model_name = conf.model.model_name
        src_lang = normalize_lang_code(src_lang, model_name)
        tgt_lang = normalize_lang_code(tgt_lang, model_name)
        if hasattr(processor, "src_lang"):
            processor.src_lang = src_lang
        if hasattr(processor, "tgt_lang"):
            processor.tgt_lang = tgt_lang

        # Process input
        input_kwargs = {"return_tensors": "pt"}
        if is_seamless_model(model_name):
            input_kwargs["src_lang"] = src_lang
        inputs = processor(text, **input_kwargs).to(model_module.device)
        generation_kwargs = {"tgt_lang": tgt_lang}
        if not is_seamless_model(model_name):
            generation_kwargs = {"forced_bos_token_id": processor.convert_tokens_to_ids(tgt_lang)}

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                max_new_tokens=conf.model.max_target_length,
                early_stopping=True,
                **generation_kwargs,
            )

        result = processor.decode(generated_tokens[0].tolist(), skip_special_tokens=True)
        print(f"Input: {text}")
        print(f"Output: {result}")
        return result

    def run_benchmark(self):
        """Runs a full comparison benchmark between Base and LEO models on WandB."""
        print("\n📊 [Phase: Benchmarking]")

        # Bug E fix: check data_path existence BEFORE calling wandb.init to avoid zombie runs
        data_path = conf.paths.data_gold / "test_set.csv"
        if not data_path.exists():
            print(f"❌ Test set not found at {data_path}. Run DataFactory().create_test_set() first.")
            return

        wandb.init(
            project="LEO-Translation",
            job_type="benchmark",
            name="Final-Model-Evaluation",
            config=conf.model.__dict__
        )

        test_set = pd.read_csv(data_path)

        # Load Base Model
        model_name = conf.model.model_name
        if is_seamless_model(model_name):
            processor = AutoProcessor.from_pretrained(model_name)
        else:
            processor = AutoTokenizer.from_pretrained(model_name)
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config)

        # Base Model Evaluation Helper
        def eval_loop(model, prefix):
            preds, targets, sources = [], [], []
            lang_pairs = [] # To track language direction
            # Bug C fix: removed shared bleu/chrf instances here; fresh instances are created per call below

            # Use NLTK for METEOR
            from nltk.translate import meteor_score
            import nltk

            # Bug F fix: use enumerate to get a reliable integer loop counter (_loop_idx)
            # instead of relying on the DataFrame index label (_) which may be non-zero-based
            for _loop_idx, (_, row) in enumerate(tqdm(test_set.iterrows(), total=len(test_set), desc=f"Eval {prefix}")):
                source_lang = normalize_lang_code(row['source_lang'], model_name)
                target_lang = normalize_lang_code(row['target_lang'], model_name)
                if hasattr(processor, "src_lang"):
                    processor.src_lang = source_lang
                if hasattr(processor, "tgt_lang"):
                    processor.tgt_lang = target_lang

                input_kwargs = {"return_tensors": "pt"}
                if is_seamless_model(model_name):
                    input_kwargs["src_lang"] = source_lang
                inputs = processor(row['source_text'], **input_kwargs).to(self.device)
                generation_kwargs = {"tgt_lang": target_lang}
                if not is_seamless_model(model_name):
                    generation_kwargs = {"forced_bos_token_id": processor.convert_tokens_to_ids(target_lang)}

                # Bug B fix: move decoded inside try/except and always append (even on failure)
                # so that preds/targets/sources/lang_pairs stay aligned
                with torch.no_grad():
                    try:
                        gen = model.generate(**inputs, max_new_tokens=128, **generation_kwargs)
                        decoded = str(processor.decode(gen[0].tolist(), skip_special_tokens=True))
                    except Exception as e:
                        print(f"⚠️ Error in generation for row {row['source_text'][:50]}... : {e}")
                        import gc
                        torch.cuda.empty_cache()
                        gc.collect()
                        decoded = ""

                preds.append(decoded)
                targets.append([str(row['target_text'])])
                sources.append(str(row['source_text']))
                lang_pairs.append(f"{source_lang}->{target_lang}")

                # Periodic memory cleanup — Bug F fix: use _loop_idx (integer counter)
                if _loop_idx % 50 == 0:
                    import gc
                    torch.cuda.empty_cache()
                    gc.collect()

            # Compute global METEOR
            meteor_scores = [meteor_score.meteor_score([str(t[0]).split()], str(p).split()) for t, p in zip(targets, preds)]
            global_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

            # Compute per-language metrics
            # Bug C fix: create fresh SacreBLEUScore/CHRFScore instances for each call
            lang_metrics = {}
            for lp in set(lang_pairs):
                lp_indices = [i for i, x in enumerate(lang_pairs) if x == lp]
                lp_preds = [preds[i] for i in lp_indices]
                lp_targets = [targets[i] for i in lp_indices]
                lp_meteor = sum([meteor_scores[i] for i in lp_indices]) / len(lp_indices)
                lang_metrics[lp] = {
                    "BLEU": SacreBLEUScore()(lp_preds, lp_targets).item(),
                    "CHRF": CHRFScore()(lp_preds, lp_targets).item(),
                    "METEOR": lp_meteor
                }

            return {
                # Bug C fix: fresh metric instances for global scores so per-lang calls don't pollute them
                "BLEU": SacreBLEUScore()(preds, targets).item(),
                "CHRF": CHRFScore()(preds, targets).item(),
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
            raise FileNotFoundError(f"Adapter not found at {adapter_path}. Run python scripts/hf.py export first.")

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
        # Bug D fix: only include directions present in BOTH models to avoid KeyError
        directions = [d for d in res_leo['LangMetrics'] if d in res_base['LangMetrics']]

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

        # Bug B fix: iterate over res_leo['Samples'] (not test_set) so the range matches actual predictions
        for i in range(len(res_leo['Samples'])):
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
        word_errors_df = pd.DataFrame(
            [
                {"type": "missing", "word": word, "count": count}
                for word, count in missing_words.most_common()
            ]
            + [
                {"type": "hallucinated", "word": word, "count": count}
                for word, count in hallucinated_words.most_common()
            ]
        )
        word_errors_path = conf.paths.project_root / "benchmarks" / "word_level_errors.csv"
        word_errors_path.parent.mkdir(parents=True, exist_ok=True)
        word_errors_df.to_csv(word_errors_path, index=False)
        print(f"✅ Word-level errors saved: {word_errors_path}")

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
            "word_level_errors": wandb.Table(dataframe=word_errors_df.head(100)),
            "Word_Level_Errors_Analysis": wandb.Image(fig2)
        })
        plt.close(fig2)
        print("✅ Results logged to WandB!")
        wandb.finish()
