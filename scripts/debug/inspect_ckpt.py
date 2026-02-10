import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.training.model_module import NLLBFineTuner

def inspect_checkpoint():
    ckpt_path = "/home/mbosetti/LEO/checkpoints/nllb-finetuned-epoch=02-val_loss=1.02-v1.ckpt"
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    state_dict = checkpoint["state_dict"]
    print(f"\nTotal keys in state_dict: {len(state_dict)}")
    print("First 10 keys:")
    for k in list(state_dict.keys())[:10]:
        print(f" - {k}")

    # Instantiate model to compare
    model = NLLBFineTuner()
    model.setup() # Initialize Peft
    
    model_keys = list(model.state_dict().keys())
    print(f"\nTotal keys in model: {len(model_keys)}")
    print("First 10 model keys:")
    for k in model_keys[:10]:
        print(f" - {k}")

if __name__ == "__main__":
    inspect_checkpoint()
