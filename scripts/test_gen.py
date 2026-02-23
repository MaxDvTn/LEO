import sys
import torch
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/seamless-m4t-v2-large", torch_dtype=torch.bfloat16)

text = "Ciao come stai?"
inputs = processor(text, src_lang="ita", return_tensors="pt")
gen = model.generate(**inputs, tgt_lang="eng", max_new_tokens=20)

print(f"gen Type: {type(gen)}")
print(f"gen Shape: {gen.shape if hasattr(gen, 'shape') else 'no shape'}")
print(f"gen[0]: {gen[0]}")
print(f"gen[0].tolist(): {gen[0].tolist()}")

decoded_correct = processor.decode(gen[0].tolist(), skip_special_tokens=True)
print(f"\nCorrect Output: {decoded_correct}")

try:
    decoded_wrong = processor.decode(gen[0].tolist()[0], skip_special_tokens=True)
    print(f"\nWrong Output: {decoded_wrong}")
except Exception as e:
    print(f"\nWrong Output Error: {e}")
