import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoProcessor, BitsAndBytesConfig
from src.common.config import conf
import pprint

def main():
    model_name = conf.model.model_name
    print(f"Loading Base model {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config)
    
    adapter_path = conf.paths.output_dir / "leo_hf_release"
    print(f"Loading LEO Adapter from {adapter_path}...")
    leo_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    leo_model.eval()

    text = "La guarnizione termoacustica è installata nel cassonetto."
    print("\nEncoding input:", text)
    inputs = processor(text, src_lang="ita", return_tensors="pt").to("cuda")

    with torch.no_grad():
        print("Disabling adapter for Base model inference...")
        leo_model.disable_adapter_layers()
        gen_base = leo_model.generate(**inputs, tgt_lang="eng", max_new_tokens=128, do_sample=False)
        
        print("Enabling adapter for LEO inference...")
        leo_model.enable_adapter_layers()
        gen_leo = leo_model.generate(**inputs, tgt_lang="eng", max_new_tokens=128, do_sample=False)

    out_base = processor.decode(gen_base[0].tolist(), skip_special_tokens=True)
    out_leo = processor.decode(gen_leo[0].tolist(), skip_special_tokens=True)
    
    print("\n--- RESULTS ---")
    print(f"BASE : {out_base}")
    print(f"LEO  : {out_leo}")
    print(f"Match: {out_base == out_leo}")
    
    for name, module in leo_model.named_modules():
        if "lora" in name.lower() and hasattr(module, "weight"):
            print(f"Found LoRA layer: {name}. Norm: {module.weight.norm().item()}")
            break

if __name__ == "__main__":
    main()
