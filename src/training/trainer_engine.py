import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from src.common.config import conf
from src.training.dataset_module import NMTDataModule
from src.training.model_module import NLLBFineTuner

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
        model = NLLBFineTuner() # Uses conf internally
        
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
            loggers.append(WandbLogger(
                project=conf.wandb.project,
                name=conf.wandb.name,
                mode=conf.wandb.mode,
                save_dir="wandb_logs"
            ))
        
        # TensorBoard fallback
        loggers.append(TensorBoardLogger("lightning_logs", name="nllb_finetune"))

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
            callbacks=[checkpoint_callback, lr_monitor],
            log_every_n_steps=self.config.log_every_n_steps,
        )

        print("Starting Training...")
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
        print(f"Training Complete. Best model path: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    trainer = TrainerEngine()
    trainer.run()
