import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.common.config import conf

def calculate_perplexity(df: pd.DataFrame, sample_size: int = 200, model_name: str = None) -> dict:
    """
    Calculates the average cross-entropy loss (perplexity) of the target sentences 
    given the source sentences using a baseline NMT model.
    A lower loss means the model naturally predicts this translation, 
    indicating good grammatical structure and alignment.
    """
    if df.empty or 'target_text' not in df.columns or 'source_text' not in df.columns:
        return {}

    n_samples = min(len(df), sample_size)
    sampled_df = df.sample(n=n_samples, random_state=42)

    m_name = model_name or conf.model.model_name
    print(f"   📉 Calculating Perplexity with {m_name} (Samples: {n_samples})...")

    # Load model in evaluation mode
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(m_name)
    # Load in bfloat16 to save memory and speed up inference
    model = AutoModelForSeq2SeqLM.from_pretrained(m_name, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    total_loss = 0.0
    valid_samples = 0

    with torch.no_grad():
        for _, row in tqdm(sampled_df.iterrows(), total=n_samples, desc="Perplexity"):
            source = str(row['source_text'])
            target = str(row['target_text'])
            src_lang = row.get('source_lang', 'ita_Latn')
            tgt_lang = row.get('target_lang', 'eng_Latn')

            # Tokenize source
            if "seamless" in m_name.lower():
                # Seamless uses AutoProcessor, this might need adjustment if using seamless
                continue # Skip seamless for now to keep it simple, or user can pass nllb
            else:
                tokenizer.src_lang = src_lang
                inputs = tokenizer(source, return_tensors="pt", truncation=True, max_length=128).to(device)
                tokenizer.tgt_lang = tgt_lang
                labels = tokenizer(text_target=target, return_tensors="pt", truncation=True, max_length=128).to(device)

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels["input_ids"],
            )
            loss = outputs.loss.item()
            
            total_loss += loss
            valid_samples += 1

    # Cleanup memory
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if valid_samples == 0:
        return {}

    avg_loss = total_loss / valid_samples
    
    import math
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float('inf')

    return {
        "avg_cross_entropy_loss": round(avg_loss, 4),
        "perplexity": round(ppl, 2),
        "ppl_samples_evaluated": valid_samples
    }
