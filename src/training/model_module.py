import torch
import pytorch_lightning as pl
from transformers import AutoModelForSeq2SeqLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import logging
import warnings
from torchmetrics.text import SacreBLEUScore, CHRFScore

# Import della configurazione centrale
from src.common.config import conf
from src.training.dataset_module import is_seamless_model, normalize_lang_code

logger = logging.getLogger(__name__)

class SeamlessFineTuner(pl.LightningModule):
    def __init__(self, 
                 model_name: str = None, 
                 learning_rate: float = None):
        super().__init__()
        self.save_hyperparameters()
        self.strict_loading = False # Ignore extra bitsandbytes keys
        
        self.model_name = model_name if model_name is not None else conf.model.model_name
        self.learning_rate = learning_rate if learning_rate is not None else conf.model.learning_rate
        self.tokenizer = None 
        # Metric
        self.bleu_metric = SacreBLEUScore()
        self.chrf_metric = CHRFScore()

    def setup(self, stage=None):
        """
        Carica il modello e applica LoRA. 
        Chiamato automaticamente da Lightning all'inizio del training.
        """
        if hasattr(self, "model"):
            return # Già caricato

        logger.info(f"Loading model: {self.model_name} in 4-bit...")

        # 1. Configurazione 4-bit (QLoRA)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # 2. Carica il Modello Base
        # Nota: Usiamo dtype=torch.bfloat16 per evitare la conversione da float32
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.bfloat16
        )
        
        # Prepara per il training k-bit (congela i pesi base)
        self.base_model = prepare_model_for_kbit_training(self.base_model)
        self.base_model.gradient_checkpointing_enable()

        # 3. Configurazione LoRA (Low-Rank Adaptation)
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=conf.model.lora_r,
            lora_alpha=conf.model.lora_alpha,
            lora_dropout=conf.model.lora_dropout,
            target_modules=conf.model.target_modules
        )

        # 4. Applica LoRA al modello
        self.model = get_peft_model(self.base_model, peft_config)
        self.model.print_trainable_parameters() 
        
        # Carichiamo anche tokenizer/processor
        if is_seamless_model(self.model_name):
            self.tokenizer = AutoProcessor.from_pretrained(self.model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _generation_kwargs(self, tgt_lang: str):
        tgt_lang = normalize_lang_code(tgt_lang, self.model_name)
        if is_seamless_model(self.model_name):
            return {"tgt_lang": tgt_lang}
        return {"forced_bos_token_id": self.tokenizer.convert_tokens_to_ids(tgt_lang)}

    def forward(self, input_ids, attention_mask, labels=None):
        # NLLB requires labels to calculate loss if wanted, or just forward
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        batch_size = batch["input_ids"].shape[0]
        loss = outputs.loss
        #batch_size = batch["input_ids"].size(0)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        # torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
            batch_size = batch["input_ids"].shape[0]
            #batch_size = batch["input_ids"].size(0)
            # 1. Loss Calculation (For EarlyStopping)
            outputs = self(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=batch_size)
            
            # 2. BLEU Calculation (Generation)
            # We generate for the whole batch
            if self.tokenizer:
                # Generate inputs
                src_ids = batch["input_ids"]
                tgt_ids = batch["labels"]
                
                # Generate translation
                # Note: We limit max_new_tokens for speed
                # In val_step we mostly translate to English, but theoretically the dataset is mixed targets
                target_langs = batch.get("target_lang") or ["eng_Latn"]
                tgt_lang = target_langs[0]
                gen_ids = self.model.generate(
                    input_ids=src_ids, 
                    max_new_tokens=64,
                    **self._generation_kwargs(tgt_lang),
                )
                preds = []
                targets = []
                
                for i in range(len(src_ids)):
                    # Decode prediction
                    pred_text = self.tokenizer.decode(gen_ids[i], skip_special_tokens=True)
                    preds.append(pred_text)
                    
                    # Decode target (handle -100 masking if present, though dataset_module usually handles it)
                    lbl_ids = tgt_ids[i]
                    lbl_ids = lbl_ids[lbl_ids != -100] 
                    tgt_text = self.tokenizer.decode(lbl_ids, skip_special_tokens=True)
                    targets.append([tgt_text]) # SacreBLEU expects list of references
                
                # Update Metric
                self.bleu_metric.update(preds, targets)
                self.chrf_metric.update(preds, targets)
                self.log("val_bleu", self.bleu_metric, on_epoch=True, prog_bar=True, batch_size=batch_size)
                self.log("val_chrf", self.chrf_metric, on_epoch=True, prog_bar=True, batch_size=batch_size)

                # 3. Log Examples (Only for first batch)
                if batch_idx == 0:
                    columns = ["Epoch", "Source (IT)", "Target (Human)", "Prediction (AI)"]
                    data = []
                    
                    # Log first 3 samples
                    limit = min(3, len(preds))
                    for i in range(limit):
                        src_text = self.tokenizer.decode(src_ids[i], skip_special_tokens=True)
                        data.append([self.current_epoch, src_text, targets[i][0], preds[i]])
                    
                    # Log to WandB
                    # Find WandB logger if available
                    wandb_logger = None
                    if isinstance(self.logger, list):
                        for l in self.logger:
                            if hasattr(l, "experiment") and hasattr(l.experiment, "log"):
                                wandb_logger = l
                                break
                    elif hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "log"):
                        wandb_logger = self.logger

                    if wandb_logger:
                        import wandb
                        wandb_logger.experiment.log({
                            "translation_examples": wandb.Table(columns=columns, data=data)
                        })
                    
                    print(f"\n🔍 [EPOCH {self.current_epoch}] Prediction: {data[0][3]}")

            return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
