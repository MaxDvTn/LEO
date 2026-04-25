import sys
from pathlib import Path
import re
import torch

# Setup Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from src.common.config import conf

def export_ckpt_to_peft(checkpoint_path, export_dir):
    """
    Loads a .ckpt file and saves only the PEFT adapters in Hugging Face format.
    """
    print(f"📦 Loading checkpoint from: {checkpoint_path}")
    
    model_name = conf.model.model_name
    
    # 1. Initialize Base Model (Must match training config)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    print(f"🧠 Initializing base model: {model_name}")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16
    )
    base_model = prepare_model_for_kbit_training(base_model)
    
    # 2. Apply LoRA (Must match training config)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=32,            
        lora_alpha=32,   
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(base_model, peft_config)
    
    # 3. Load State Dict
    print("📂 Loading weights from checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Lightning prefixes everything with "model." (if self.model exists) or "base_model" etc.
    # In our case, NLLBFineTuner has self.model (which is the peft model)
    state_dict = checkpoint["state_dict"]
    
    # Remove "model." prefix and keep only lora adapters
    print(f"🎯 Extracting LoRA adapters...")
    
    target_keys = set(model.state_dict().keys())
    checkpoint_lora_keys = [k for k in state_dict if "lora_" in k]
    print(f"🔍 Found {len(checkpoint_lora_keys)} LoRA keys in checkpoint.")
    
    target_lora_keys = [k for k in target_keys if "lora_" in k]
    print(f"🔍 Found {len(target_lora_keys)} LoRA keys in release model.")

    new_state_dict = {}
    matched_checkpoint_keys = set()
    matched_target_keys = set()
    
    for k, v in state_dict.items():
        if "lora_" in k:
            # Checkpoint keys: "base_model.text_encoder.layers.0.self_attn.k_proj.lora_A.default.weight"
            # Target HF PEFT keys: "base_model.model.text_encoder.layers.0.self_attn.k_proj.lora_A.default.weight"
            
            # Use suffix starting from "text_encoder" or "text_decoder"
            clean_key = k
            parts = clean_key.split(".")
            start_node = -1
            for i, p in enumerate(parts):
                if p in ["text_encoder", "text_decoder", "encoder", "decoder"]:
                    start_node = i
                    break
            
            if start_node != -1:
                suffix = ".".join(parts[start_node:])
                for tk in target_keys:
                    if tk.endswith(suffix):
                        if tk not in matched_target_keys:
                            new_state_dict[tk] = v
                            matched_target_keys.add(tk)
                        matched_checkpoint_keys.add(k)
                        break
    
    print(f"📊 Match Summary:")
    print(f"   - Total unique LoRA tensors expected: {len(target_lora_keys)}")
    print(f"   - Total unique LoRA tensors matched:  {len(matched_target_keys)}")
    print(f"   - Checkpoint keys consumed:           {len(matched_checkpoint_keys)} / {len(checkpoint_lora_keys)}")
    
    if len(matched_target_keys) < len(target_lora_keys):
        print("⚠️ SOME TARGET LORA KEYS WERE NOT MATCHED!")
        missing = [tk for tk in target_lora_keys if tk not in matched_target_keys]
        for m in missing[:5]:
            print(f"     - {m}")
    else:
        print("✅ 100% of target LoRA weights successfully matched from checkpoint!")

    if len(matched_checkpoint_keys) < len(checkpoint_lora_keys):
        unmatched_ckpt = len(checkpoint_lora_keys) - len(matched_checkpoint_keys)
        print(f"ℹ️ Note: {unmatched_ckpt} checkpoint keys were ignored (likely duplicates or unrelated modules).")

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"ℹ️ Load status: {msg}")
    print(f"ℹ️ Load status: {msg}")
    
    # 4. Save PEFT Format
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Saving PEFT adapters to: {export_path}")
    model.save_pretrained(export_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(export_path)
    
    print("\n✅ Export Completed!")


def main():
    # Find the best checkpoint automatically
    ckpt_dir = conf.paths.output_dir
    checkpoints = list(ckpt_dir.glob("*.ckpt"))
    
    if not checkpoints:
        print(f"❌ No checkpoints found in {ckpt_dir}")
        return

    def get_val_loss(path: Path):
        match = re.search(r"val_loss=([0-9]+(?:\.[0-9]+)?)", path.name)
        return float(match.group(1)) if match else None

    with_loss = [(ckpt, get_val_loss(ckpt)) for ckpt in checkpoints]
    scored = [(ckpt, loss) for ckpt, loss in with_loss if loss is not None]
    if scored:
        best_ckpt, best_loss = sorted(scored, key=lambda item: item[1])[0]
        print(f"🏆 Best checkpoint selected: {best_ckpt.name} (Loss: {best_loss})")
    else:
        best_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"🏆 No val_loss in filenames. Selected latest checkpoint: {best_ckpt.name}")

    export_dir = conf.paths.output_dir / "leo_hf_release"
    export_ckpt_to_peft(str(best_ckpt), export_dir)

if __name__ == "__main__":
    main()
