from transformers import AutoTokenizer
import torch

def debug_tokenizer():
    model_name = "facebook/nllb-200-distilled-1.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer class: {tokenizer.__class__.__name__}")
    
    try:
        print(f"lang_code_to_id exists: {hasattr(tokenizer, 'lang_code_to_id')}")
        if hasattr(tokenizer, 'lang_code_to_id'):
            print(f"ita_Latn id: {tokenizer.lang_code_to_id.get('ita_Latn')}")
    except Exception as e:
        print(f"Error checking lang_code_to_id: {e}")

    # Alternative way: convert_tokens_to_ids
    try:
        token_id = tokenizer.convert_tokens_to_ids("ita_Latn")
        print(f"convert_tokens_to_ids('ita_Latn'): {token_id}")
    except Exception as e:
        print(f"Error with convert_tokens_to_ids: {e}")

if __name__ == "__main__":
    debug_tokenizer()
