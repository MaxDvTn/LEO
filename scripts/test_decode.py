from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
decoded = processor.decode([100, 200, 300], skip_special_tokens=True)
print("Decoded list:", decoded)
try:
    decoded2 = processor.decode(100, skip_special_tokens=True)
    print("Decoded int:", decoded2)
except Exception as e:
    print("Error decoding int:", e)
