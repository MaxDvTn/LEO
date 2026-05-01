# src/synthesis/generator.py
from src.synthesis.base import BaseGenerator
from src.common.config import conf


def get_generator(model_id: str = None) -> BaseGenerator:
    mid = model_id or conf.gen.model_id

    if mid.startswith("ollama/"):
        from src.synthesis.ollama_generator import OllamaGenerator
        return OllamaGenerator(model_id=mid)

    from src.synthesis.hf_generator import HFChatGenerator
    return HFChatGenerator(model_id=mid)


SyntheticGenerator = get_generator


if __name__ == "__main__":
    gen = get_generator()
    df = gen.generate_dataset(terms=["guarnizione termoacustica"])
    print(df.head())







# # src/synthesis/generator.py
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# import pandas as pd
# import logging
# from typing import List, Dict

# # Import interni
# from src.synthesis.glossary_data import get_terms_list
# from src.synthesis.prompts import GENERATION_PROMPT_TEMPLATE
# from src.common.config import conf

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class SyntheticGenerator:
#     def __init__(self, model_id: str = None):
#         if model_id is None:
#             model_id = conf.gen.model_id
        
#         self.model_id = model_id
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
#         logger.info(f"🚀 Loading Model {model_id} on {self.device}...")
        
#         # Load Tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)
#         self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
#         # Load Pipeline with 4-bit Quantization (BitsAndBytes)
#         self.pipe = pipeline(
#             "text-generation",
#             model=model_id,
#             tokenizer=self.tokenizer,
#             device_map="auto",
#             model_kwargs={"load_in_4bit": True}, # Ottimizzato per RTX 4090
#         )
#         logger.info("✅ Model Loaded Successfully.")

#     def parse_output(self, raw_text: str, term: str) -> Dict:
#         """Estrae IT, EN, FR, ES dall'output grezzo dell'LLM."""
#         try:
#             # Rimuove il prompt dall'output per pulizia
#             if "[/INST]" in raw_text:
#                 generated_part = raw_text.split("[/INST]")[-1].strip()
#             else:
#                 generated_part = raw_text.strip()
                
#             lines = generated_part.split('\n')
            
#             result = {
#                 "term": term,
#                 "source_text": None, # IT
#                 "target_text_en": None,
#                 "target_text_fr": None,
#                 "target_text_es": None,
#                 "raw_output": generated_part
#             }

#             for line in lines:
#                 clean_line = line.strip()
#                 if clean_line.startswith("IT:"):
#                     result["source_text"] = clean_line.replace("IT:", "").strip()
#                 elif clean_line.startswith("EN:"):
#                     result["target_text_en"] = clean_line.replace("EN:", "").strip()
#                 elif clean_line.startswith("FR:"):
#                     result["target_text_fr"] = clean_line.replace("FR:", "").strip()
#                 elif clean_line.startswith("ES:"):
#                     result["target_text_es"] = clean_line.replace("ES:", "").strip()
            
#             # Validazione base: deve esserci almeno l'Italiano
#             if result["source_text"]:
#                 return result
#             else:
#                 logger.warning(f"⚠️ Failed to parse IT sentence for term: {term}")
#                 return None

#         except Exception as e:
#             logger.error(f"❌ Parsing Error for {term}: {e}")
#             return None

#     def generate_dataset(self, terms: List[str] = None) -> pd.DataFrame:
#         if terms is None:
#             terms = get_terms_list()
        
#         logger.info(f"Starting generation for {len(terms)} terms...")
#         data = []

#         for i, term in enumerate(terms):
#             logger.info(f"[{i+1}/{len(terms)}] Generating for: {term}")
            
#             prompt = GENERATION_PROMPT_TEMPLATE.format(term=term)
            
#             sequences = self.pipe(
#                 prompt,
#                 do_sample=True,
#                 temperature=0.7,
#                 max_new_tokens=250,
#                 top_p=0.9
#             )
            
#             raw_output = sequences[0]['generated_text']
#             parsed_entry = self.parse_output(raw_output, term)
            
#             if parsed_entry:
#                 data.append(parsed_entry)
#                 # Stampa anteprima
#                 print(f"   🇮🇹 {parsed_entry['source_text'][:50]}...")
#                 print(f"   🇬🇧 {parsed_entry['target_text_en'][:50]}...")

#         df = pd.DataFrame(data)
#         logger.info(f"🎉 Generation Complete. Created {len(df)} samples.")
#         return df

# if __name__ == "__main__":
#     gen = SyntheticGenerator()
#     df = gen.generate_dataset(terms=["guarnizione termoacustica"])
#     print(df.head())