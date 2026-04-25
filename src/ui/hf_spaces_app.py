import gradio as gr
import torch
import os
from transformers import AutoModelForSeq2SeqLM, AutoProcessor
from peft import PeftModel
import pandas as pd
from datetime import datetime
from pathlib import Path

# Configuration
BASE_MODEL = "facebook/seamless-m4t-v2-large"
# Use '.' for relative paths when deployed on Spaces, or env var for local testing
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "./")
CORRECTIONS_FILE = os.getenv("CORRECTIONS_PATH", "/home/mbosetti/LEO/data/gold/human_corrections.csv")

# Global Placeholders
model = None
processor = None

def load_model():
    """Loads the model and processor lazily."""
    global model, processor
    if model is not None:
        return model, processor
    
    print("🚀 Loading LEO Processor...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    
    print("🚀 Loading LEO Model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    # Load custom LEO adapters
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    return model, processor

def translate(text, source_lang, target_lang):
    """Translates text from the selected source language to target language."""
    if not text.strip():
        return ""
    
    lang_map = {
        "Italian": "ita",
        "English": "eng",
        "French": "fra",
        "Spanish": "spa",
    }
    src_code = lang_map.get(source_lang, "ita")
    tgt_code = lang_map.get(target_lang, "eng")
    if src_code == tgt_code:
        return text
    
    model, processor = load_model()
    inputs = processor(text, src_lang=src_code, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            tgt_lang=tgt_code,
            max_new_tokens=128
        )
    
    return processor.decode(generated_tokens[0].tolist(), skip_special_tokens=True)

def save_correction(source, model_output, correction, source_lang, target_lang):
    """Saves the human correction to a CSV file for future retraining."""
    if not correction.strip() or not source.strip():
        return "⚠️ Errore: Inserisci sia il testo originale che la correzione."
    
    lang_map = {
        "Italian": "ita_Latn",
        "English": "eng_Latn",
        "French": "fra_Latn",
        "Spanish": "spa_Latn",
    }
    src_code = lang_map.get(source_lang, "ita_Latn")
    tgt_code = lang_map.get(target_lang, "eng_Latn")
    
    new_data = {
        "source_text": [source],
        "target_text": [correction],
        "source_lang": [src_code],
        "target_lang": [tgt_code],
        "model_output": [model_output],
        "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    }
    
    df = pd.DataFrame(new_data)
    file_path = Path(CORRECTIONS_FILE)
    
    # Append if exists, else create
    header = not file_path.exists()
    df.to_csv(file_path, mode='a', index=False, header=header)
    
    return f"✅ Correzione salvata in {file_path.name}!"

# Build Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🦁 L.E.O. Translation Hub")
    gr.Markdown("### Roverplastik Technical Translation - Seamless-M4T v2 (LoRA)")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="📝 Testo sorgente",
                placeholder="Inserisci qui la frase da tradurre...",
                lines=5
            )
            source_lang_selector = gr.Dropdown(
                choices=["Italian", "English", "French", "Spanish"],
                value="Italian",
                label="📥 Lingua di input"
            )
            target_lang_selector = gr.Dropdown(
                choices=["Italian", "English", "French", "Spanish"],
                value="English",
                label="📤 Lingua di output"
            )
            btn = gr.Button("Traduci", variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(
                label="✨ Traduzione L.E.O.",
                lines=5,
                interactive=False
            )
            correction_text = gr.Textbox(
                label="📝 Suggerisci Correzione (opzionale)",
                placeholder="Se la traduzione non è corretta, inserisci qui la versione corretta...",
                lines=5
            )
            save_btn = gr.Button("Salva Correzione", variant="secondary")
            status_msg = gr.Markdown("")
            
    btn.click(
        fn=translate,
        inputs=[input_text, source_lang_selector, target_lang_selector],
        outputs=output_text
    )
    
    save_btn.click(
        fn=save_correction,
        inputs=[input_text, output_text, correction_text, source_lang_selector, target_lang_selector],
        outputs=status_msg
    )
    
    gr.Markdown("---")
    gr.Markdown("ℹ️ **Nota**: Questa app usa Seamless-M4T v2 con adattatori LoRA ottimizzati per la terminologia tecnica Roverplastik.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
