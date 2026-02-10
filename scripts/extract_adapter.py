import torch
from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.training.model_module import NLLBFineTuner

def extract_adapter():
    ckpt_path = "/home/mbosetti/LEO/checkpoints/nllb-finetuned-epoch=04-val_loss=0.47.ckpt"
    output_dir = "/home/mbosetti/LEO/checkpoints/final_adapter"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📦 Loading FineTuner from checkpoint: {ckpt_path}")
    # Nota: carichiamo su CPU per sicurezza
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # Istanziamo il modello (senza pesi inizialmente)
    tuner = NLLBFineTuner()
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
