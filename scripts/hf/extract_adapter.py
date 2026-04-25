import torch
from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.training.model_module import SeamlessFineTuner

from src.common.config import conf

def extract_adapter():
    checkpoints = list(conf.paths.output_dir.glob("*.ckpt"))
    if not checkpoints: 
        raise FileNotFoundError("No adapter/checkpoint found.")
    
    def get_val_loss(p):
        try:
            val_part = p.name.split("val_loss=")[1]
            return float(val_part.replace(".ckpt", "").split("-")[0])
        except:
            return 999.0

    best_ckpt = sorted(checkpoints, key=get_val_loss)[0]
    ckpt_path = str(best_ckpt)
    
    output_dir = conf.paths.output_dir / "leo_hf_release"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Loading FineTuner from checkpoint: {ckpt_path}")
    # Nota: carichiamo su CPU per sicurezza
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # Istanziamo il modello (senza pesi inizialmente)
    tuner = SeamlessFineTuner()
    tuner.setup() # Questo crea self.model (PeftModel)
    
    print("📥 Loading state_dict into tuner (strict=False because base is already loaded)...")
    # Carichiamo i pesi nel tuner. 
    # Poiché il base_model è già caricato da from_pretrained, ci interessano i pesi LoRA.
    tuner.load_state_dict(checkpoint["state_dict"], strict=False)
    
    print(f"💾 Saving adapter to: {output_dir}")
    tuner.model.save_pretrained(output_dir)
    print("✅ Adapter saved successfully!")

if __name__ == "__main__":
    extract_adapter()
