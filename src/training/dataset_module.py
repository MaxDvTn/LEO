import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoProcessor, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import logging

# Import interno per la config
from src.common.config import conf

logger = logging.getLogger(__name__)

SEAMLESS_LANG_MAP = {
    "ita_Latn": "ita",
    "eng_Latn": "eng",
    "fra_Latn": "fra",
    "spa_Latn": "spa",
}


def is_seamless_model(model_name: str) -> bool:
    return "seamless" in model_name.lower()


def normalize_lang_code(lang: str, model_name: str) -> str:
    if is_seamless_model(model_name):
        return SEAMLESS_LANG_MAP.get(lang, lang)
    return lang

class LeoDataset(Dataset):
    """
    Custom Dataset for NLLB Multilingual.
    """
    def __init__(self, data: pd.DataFrame, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_name = conf.model.model_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        source_text = str(row['source_text'])
        target_text = str(row['target_text'])
        src_lang = row['source_lang']
        tgt_lang = row['target_lang']

        src_lang = normalize_lang_code(src_lang, self.model_name)
        tgt_lang = normalize_lang_code(tgt_lang, self.model_name)

        if hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = src_lang
        if hasattr(self.tokenizer, "tgt_lang"):
            self.tokenizer.tgt_lang = tgt_lang

        # 1. Process Input
        input_kwargs = {
            "text": source_text,
            "return_tensors": "pt",
            "truncation": True,
            "max_length": self.max_length,
        }
        if is_seamless_model(self.model_name):
            input_kwargs["src_lang"] = src_lang
        inputs = self.tokenizer(**input_kwargs)

        # 2. Process Target
        if is_seamless_model(self.model_name):
            labels = self.tokenizer(
                text=target_text,
                src_lang=tgt_lang,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            )
        else:
            labels = self.tokenizer(
                text_target=target_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels["input_ids"].squeeze(0),
            "target_lang": tgt_lang,
        }

class NMTDataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        target_langs = [item["target_lang"] for item in batch]
        
        # Pad sequences
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100) # -100 is ignored by CrossEntropyLoss
        
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
            "target_lang": target_langs,
        }

class NMTDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.config = conf.model
        self.paths = conf.paths
        
        self.train_dataset = None
        self.val_dataset = None
        self.tokenizer = None

    def setup(self, stage=None):
        if self.train_dataset is not None:
            return

        # 1. Load Tokenizer/Processor
        logger.info(f"Loading tokenizer/processor: {self.config.model_name}")
        if is_seamless_model(self.config.model_name):
            self.tokenizer = AutoProcessor.from_pretrained(self.config.model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # 2. Load and Combine Data
        dfs = []

        # Collect test-set source texts upfront so we can exclude them from training
        test_set_sources: set[str] = set()
        test_set_path = self.paths.data_gold / "test_set.csv"
        if test_set_path.exists():
            ts_df = pd.read_csv(test_set_path, usecols=lambda c: c == "source_text")
            test_set_sources = set(ts_df["source_text"].dropna().astype(str).str.strip())
            logger.info(f"Loaded {len(test_set_sources)} test-set source texts for leakage exclusion")

        # Check gold data — skip test_set.csv itself
        gold_files = [f for f in self.paths.data_gold.glob("*.csv") if f.name != "test_set.csv"]
        for f in gold_files:
            dfs.append(pd.read_csv(f))
            logger.info(f"Loaded gold data: {f.name}")

        # Check synthetic data. If an ensemble has been built, combine it with
        # the stable historical synthetic corpora and skip duplicated/noisy
        # top-level files.
        ensemble_path = self.paths.data_synthetic / "ensemble_training_set.csv"
        if ensemble_path.exists():
            preferred_synth_names = [
                "rover_synthetic_multilingual.csv",
                "competitor_synthetic.csv",
                "ensemble_training_set.csv",
            ]
            synth_files = [
                self.paths.data_synthetic / name
                for name in preferred_synth_names
                if (self.paths.data_synthetic / name).exists()
            ]
            logger.info(
                "Using curated synthetic training pool: "
                + ", ".join(f.name for f in synth_files)
            )
        else:
            synth_files = list(self.paths.data_synthetic.glob("*.csv"))
        for f in synth_files:
            dfs.append(pd.read_csv(f))
            logger.info(f"Loaded synthetic data: {f.name}")

        if not dfs:
            raise FileNotFoundError(f"No CSV files found in {self.paths.data_gold} or {self.paths.data_synthetic}")

        df = pd.concat(dfs, ignore_index=True)

        # Basic cleaning
        df = df.dropna(subset=['source_text', 'target_text', 'source_lang', 'target_lang'])

        # Exclude test-set rows to prevent data leakage into training
        if test_set_sources:
            before = len(df)
            df = df[~df["source_text"].astype(str).str.strip().isin(test_set_sources)]
            excluded = before - len(df)
            if excluded:
                logger.info(f"Excluded {excluded} training rows whose source_text appears in the test set")
        
        # 3. Split Train/Validation
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=self.config.seed)
        logger.info(f"Data Split -> Train: {len(train_df)} | Val: {len(val_df)}")

        # 4. Create Datasets
        self.train_dataset = LeoDataset(train_df, self.tokenizer, self.config.max_source_length)
        self.val_dataset = LeoDataset(val_df, self.tokenizer, self.config.max_source_length)

    def train_dataloader(self):
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = self.tokenizer.tokenizer.pad_token_id
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=NMTDataCollator(pad_id)
        )

    def val_dataloader(self):
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = self.tokenizer.tokenizer.pad_token_id
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=NMTDataCollator(pad_id)
        )
