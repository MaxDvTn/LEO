import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from torch.nn.utils.rnn import pad_sequence
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

        # SeamlessM4T uses linguistic tags like 'ita', 'eng', etc.
        # We need to map from NLLB tags if they are still in the CSVs
        lang_map = {
            "ita_Latn": "ita",
            "eng_Latn": "eng", 
            "fra_Latn": "fra",
            "spa_Latn": "spa"
        }
        
        seamless_src = lang_map.get(src_lang, src_lang)
        seamless_tgt = lang_map.get(tgt_lang, tgt_lang)

        # 1. Process Input
        inputs = self.tokenizer(
            text=source_text,
            src_lang=seamless_src,
            return_tensors="pt"
        )

        # 2. Process Target
        labels = self.tokenizer(
            text=target_text,
            src_lang=seamless_tgt, # For Seamless target tokenization
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels["input_ids"].squeeze(0)
        }

class SeamlessDataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        # Pad sequences
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100) # -100 is ignored by CrossEntropyLoss
        
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded
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

        # 1. Load Tokenizer/Processor for Seamless
        logger.info(f"Loading processor: {self.config.model_name}")
        from transformers import AutoProcessor
        self.tokenizer = AutoProcessor.from_pretrained(self.config.model_name)

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
        pad_id = self.tokenizer.tokenizer.pad_token_id
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=SeamlessDataCollator(pad_id)
        )

    def val_dataloader(self):
        pad_id = self.tokenizer.tokenizer.pad_token_id
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=SeamlessDataCollator(pad_id)
        )