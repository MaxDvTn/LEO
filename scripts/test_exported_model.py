import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "facebook/nllb-200-distilled-1.3B"
ADAPTER_PATH = "checkpoints/leo_hf_release"

print(f"🚀 Loading base model: {BASE_MODEL}")
model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="auto"
)

print(f"📂 Loading adapters from: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

test_sentences = [
    "Cassonetto coibentato per serramenti.",
    "Profilo in gomma per guarnizioni."
]

print("\n✨ Testing Translations:")
target_lang = "eng_Latn"
tokenizer.src_lang = "ita_Latn"

for text in test_sentences:
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
            max_new_tokens=128
        )
    translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print(f"🇮🇹 {text}")
    print(f"🇬🇧 {translation}\n")

print("✅ Headless test completed!")
