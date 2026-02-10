import torch
import sys

checkpoint_path = 'checkpoints/nllb-finetuned-epoch=05-val_loss=1.02.ckpt'
print(f"Loading {checkpoint_path}...")
ckpt = torch.load(checkpoint_path, map_location='cpu')
sd = ckpt['state_dict']

lora_keys = [k for k in sd if 'lora_' in k]
print(f"Total LoRA keys: {len(lora_keys)}")

prefix1 = "base_model."
prefix2 = "model.base_model."

keys1 = sorted([k for k in lora_keys if k.startswith(prefix1) and not k.startswith(prefix2)])
keys2 = sorted([k for k in lora_keys if k.startswith(prefix2)])

print(f"Keys starting with '{prefix1}' (but not '{prefix2}'): {len(keys1)}")
print(f"Keys starting with '{prefix2}': {len(keys2)}")

# Check if they are pairs
matched_pairs = 0
for k1 in keys1:
    # Try to construct k2
    # k1: base_model.model.encoder...
    # k2: model.base_model.model.model.encoder... 
    # Wait, let's look at the sample from last output:
    # k1: base_model.model.encoder.layers.0.self_attn.k_proj.lora_A.default.weight
    # k2: model.base_model.model.model.encoder.layers.8.self_attn.q_proj.lora_A.default.weight (Wait, this was layer 8)
    
    # Let's try matching by suffix
    suffix = ".".join(k1.split(".")[1:]) # everything after base_model.
    # Check if any k2 ends with something related
    for k2 in keys2:
        if k2.endswith(suffix):
            if torch.equal(sd[k1], sd[k2]):
                matched_pairs += 1
            break

print(f"Matched value pairs: {matched_pairs}")

# Identify targeted modules
modules = set()
for k in lora_keys:
    parts = k.split('.')
    # Find where 'lora_' is
    for i, p in enumerate(parts):
        if 'lora_' in p:
            modules.add(parts[i-1])
print(f"Targeted modules: {sorted(list(modules))}")
