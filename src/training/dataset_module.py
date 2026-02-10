import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

# Import interno per la config
from src.common.config import conf

logger = logging.getLogger(__name__)

class LeoDataset(Dataset):
    """
    Custom Dataset for NLLB Multilingual.
    """
    def __init__(self, data: pd.DataFrame, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        source_text = str(row['source_text'])
        target_text = str(row['target_text'])
        src_lang = row['source_lang']
        tgt_lang = row['target_lang']

        # 1. Set source language
        self.tokenizer.src_lang = src_lang

        # 2. Tokenize input
        inputs = self.tokenizer(
            source_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )

        # 3. Tokenize target
        self.tokenizer.tgt_lang = tgt_lang
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_text,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_tensors=None
            )

        # 4. Get forced_bos_token_id for target language
        # NLLB uses special tokens for languages. 
        # For example 'ita_Latn', 'eng_Latn', etc.
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels["input_ids"],
            "forced_bos_token_id": forced_bos_token_id
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

        # 1. Load Tokenizer
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # 2. Load and Combine Data
        dfs = []
        
        # Check gold data
        gold_files = list(self.paths.data_gold.glob("*.csv"))
        for f in gold_files:
            dfs.append(pd.read_csv(f))
            logger.info(f"Loaded gold data: {f.name}")
            
        # Check synthetic data
        synth_files = list(self.paths.data_synthetic.glob("*.csv"))
        for f in synth_files:
            dfs.append(pd.read_csv(f))
            logger.info(f"Loaded synthetic data: {f.name}")

        if not dfs:
            raise FileNotFoundError(f"No CSV files found in {self.paths.data_gold} or {self.paths.data_synthetic}")

        df = pd.concat(dfs, ignore_index=True)
        
        # Basic cleaning
        df = df.dropna(subset=['source_text', 'target_text', 'source_lang', 'target_lang'])
        
        # 3. Split Train/Validation
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=self.config.seed)
        logger.info(f"Data Split -> Train: {len(train_df)} | Val: {len(val_df)}")

        # 4. Create Datasets
        self.train_dataset = LeoDataset(train_df, self.tokenizer, self.config.max_source_length)
        self.val_dataset = LeoDataset(val_df, self.tokenizer, self.config.max_source_length)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=DataCollatorForSeq2Seq(self.tokenizer, padding=True)
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=DataCollatorForSeq2Seq(self.tokenizer, padding=True)
        )