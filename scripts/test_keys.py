import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.common.config import conf

checkpoints = sorted(conf.paths.output_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
if not checkpoints:
    raise FileNotFoundError(f"No checkpoint found in {conf.paths.output_dir}")

checkpoint_path = checkpoints[0]
checkpoint = torch.load(checkpoint_path, map_location="cpu")
state_dict = checkpoint["state_dict"]

lora_keys = [k for k in state_dict.keys() if "lora" in k]
print(f"Total lora keys: {len(lora_keys)}")
for k in lora_keys[:10]:
    print(k)
