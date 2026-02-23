import gradio as gr
import torch
import os
from transformers import AutoModelForSeq2SeqLM, AutoProcessor
from peft import PeftModel

# Configuration
BASE_MODEL = "facebook/seamless-m4t-v2-large"
# Use '.' for relative paths when deployed on Spaces, or env var for local testing
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "./")

# Load Processor
processor = AutoProcessor.from_pretrained(BASE_MODEL)

# Load Model with LoRA
print("🚀 Loading LEO Model...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
# Load custom LEO adapters
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

def translate(text, target_lang):
    """Translates Italian text to the selected target language."""
    if not text.strip():
        return ""
    
    # Mapping for UI labels to Seamless codes
    lang_map = {
        "English": "eng",
        "French": "fra",
        "Spanish": "spa"
    }
    tgt_code = lang_map.get(target_lang, "eng")
    
    inputs = processor(text, src_lang="ita", return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            tgt_lang=tgt_code,
            max_new_tokens=128
        )
    
    return processor.decode(generated_tokens[0].tolist()[0], skip_special_tokens=True)

# Build Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🦁 L.E.O. Translation Hub")
    gr.Markdown("### Roverplastik Neural Translation - Specialized NLLB-200")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="🇮🇹 Testo in Italiano",
                placeholder="Inserisci qui la frase tecnica...",
                lines=5
            )
            lang_selector = gr.Radio(
                choices=["English", "French", "Spanish"],
                value="English",
                label="🎯 Lingua di Destinazione"
            )
            btn = gr.Button("Traduci", variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(
                label="✨ Traduzione L.E.O.",
                lines=5,
                interactive=False
            )
            
    btn.click(
        fn=translate,
        inputs=[input_text, lang_selector],
        outputs=output_text
    )
    
    gr.Markdown("---")
    gr.Markdown("ℹ️ **Nota**: Questo modello è stato ottimizzato specificamente per la terminologia tecnica Roverplastik.")

if __name__ == "__main__":
    demo.launch()
