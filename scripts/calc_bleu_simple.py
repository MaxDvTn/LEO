import sys
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from torchmetrics.text import SacreBLEUScore

# Setup Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.training.model_module import NLLBFineTuner
from src.common.config import conf

def main():
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "nllb-finetuned-epoch=10-val_loss=0.52.ckpt"
    test_set_path = PROJECT_ROOT / "data" / "gold" / "test_set.csv"
    
    print(f"⚖️  Calculating BLEU for Checkpoint: {checkpoint_path.name}")
    print(f"    Test Set: {test_set_path}")
    
    # 1. Load Model
    print("    Loading Model (this may take a moment)...")
    
    # Manually instantiate and setup
    # We must match the hyperparameters if they were saved, but here we use defaults or config
    model_module = NLLBFineTuner()
    print("    Building model architecture...")
    model_module.setup() 
    
    # Load weights
    print(f"    Restoring weights from {checkpoint_path}...")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    # State dict keys might have 'model.' prefix or similar depending on how they were saved
    # But usually Lightning saves 'state_dict' key
    state_dict = checkpoint["state_dict"]
    
    # Handle PEFT/LoRA keys matching
    # Our model is 'self.model' which is the PeftModel
    # The state dict probably has keys like 'model.base_model.model.encoder...'
    # We load with strict=False to ignore standard missing keys if any, but we want to ensure adapters load
    keys = model_module.load_state_dict(state_dict, strict=False)
    print(f"    Missing keys: {len(keys.missing_keys)} | Unexpected keys: {len(keys.unexpected_keys)}")
    
    model_module.eval()
    if torch.cuda.is_available():
        model_module.cuda()
    
    tokenizer = model_module.tokenizer
    model = model_module.model
    
    # 2. Load Data
    test_df = pd.read_csv(test_set_path)
    print(f"    Loaded {len(test_df)} test samples.")
    
    # 3. Evaluation Loop
    preds = []
    targets = []
    
    bleu_metric = SacreBLEUScore()
    
    print("    Running Translation...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        src_text = row['source_text']
        tgt_text = row['target_text']
        src_lang = row['source_lang']
        tgt_lang = row['target_lang']
        
        tokenizer.src_lang = src_lang
        inputs = tokenizer(src_text, return_tensors="pt").to(model_module.device)
        
        forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        
        with torch.no_grad():
            gen_tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                max_new_tokens=128,
            )
        
        decoded_pred = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        
        preds.append(decoded_pred)
        targets.append([tgt_text]) # SacreBLEU expects list of lists for references
        
    # 4. Calculate Score
    print("\n📊 Results:")
    final_bleu = bleu_metric(preds, targets).item()
    print(f"    ✅ FINAL TEST BLEU: {final_bleu:.2f}")

    # Show a few examples
    print("\n    Examples:")
    for i in range(min(3, len(preds))):
        print(f"    IT: {test_df.iloc[i]['source_text']}")
        print(f"    GT: {targets[i][0]}")
        print(f"    PD: {preds[i]}")
        print("-" * 30)

if __name__ == "__main__":
    main()
