import torch
from transformers import AutoModelForSeq2SeqLM, AutoProcessor
from peft import PeftModel

BASE_MODEL = "facebook/seamless-m4t-v2-large"
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

processor = AutoProcessor.from_pretrained(BASE_MODEL)

test_sentences = [
    "Cassonetto coibentato per serramenti.",
    "Profilo in gomma per guarnizioni."
]

print("\n✨ Testing Translations:")
target_lang = "eng"
src_lang = "ita"

for text in test_sentences:
    inputs = processor(text, src_lang=src_lang, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            tgt_lang=target_lang,
            max_new_tokens=128
        )
    translation = processor.decode(generated_tokens[0].tolist()[0], skip_special_tokens=True)
    print(f"🇮🇹 {text}")
    print(f"🇬🇧 {translation}\n")

print("✅ Headless test completed!")
