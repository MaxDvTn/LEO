import pytorch_lightning as pl
import warnings
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from src.common.config import conf
from src.training.dataset_module import NMTDataModule
from src.training.model_module import SeamlessFineTuner

class TrainerEngine:
    def __init__(self):
        self.config = conf.model
        self.paths = conf.paths
        pl.seed_everything(self.config.seed)

    def run(self, ckpt_path=None):
        # DataModule
        dm = NMTDataModule() # Uses conf internally
        dm.setup()

        # Model
        model = SeamlessFineTuner() # Uses conf internally
        
        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.paths.output_dir,
            filename="nllb-finetuned-{epoch:02d}-{val_loss:.2f}",
            save_top_k=2,
            monitor="val_loss",
            mode="min",
            save_last=True
        )
        
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=3,
            mode="min"
        )

        lr_monitor = LearningRateMonitor(logging_interval='step')

        # Loggers
        loggers = []
        
        # WandB Logger (Primary)
        if conf.wandb.enabled:
            from pytorch_lightning.loggers import WandbLogger
            import wandb
            # Convert configs to dicts for logging
            from dataclasses import asdict
            hparams = {
                "model": asdict(conf.model),
                "generation": asdict(conf.gen)
            }
            
            wandb_logger = WandbLogger(
                project=conf.wandb.project,
                name=f"{conf.wandb.name}-{conf.model.model_name.split('/')[-1]}",
                mode=conf.wandb.mode,
                save_dir="wandb_logs",
                log_model=True, # Logs checkpoints as artifacts
                config=hparams
            )
            # Weights & Biases parameter watching disabled to save massive VRAM
            # wandb_logger.watch(model, log="all", log_freq=50)
            loggers.append(wandb_logger)
        
        warnings.filterwarnings("ignore", message=".*Found.*module.*eval mode.*")

        # Trainer
        trainer = pl.Trainer(
            max_epochs=self.config.max_epochs,
            accelerator="gpu",
            devices=1,
            precision=self.config.precision,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            check_val_every_n_epoch=1,
            val_check_interval=self.config.val_check_interval,
            logger=loggers,
            callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
            log_every_n_steps=self.config.log_every_n_steps,
        )

        print("Starting Training...")
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
        print(f"Training Complete. Best model path: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    trainer = TrainerEngine()
    trainer.run()
