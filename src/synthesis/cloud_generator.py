# src/synthesis/cloud_generator.py
import logging
from pathlib import Path
import pandas as pd
from typing import List, Dict

try:
    from dotenv import load_dotenv
    import litellm
    litellm.drop_params = True
except ImportError:
    raise ImportError("Run: pip install litellm python-dotenv")

from src.synthesis.base import BaseGenerator
from src.synthesis.glossary_data import get_terms_list
from src.synthesis.prompts import (
    GENERATION_SYSTEM_PROMPT,
    GENERATION_USER_TEMPLATE,
    TRANSLATION_SYSTEM_PROMPT,
    TRANSLATION_USER_TEMPLATE,
)
from src.common.config import conf

logger = logging.getLogger(__name__)

# litellm uses "gemini/" for Google models, not "google/"
_PREFIX_REMAP = {"google/": "gemini/"}
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CloudGenerator(BaseGenerator):
    """
    Synthetic data generator backed by cloud LLM APIs via litellm.

    Supported model_id formats (set conf.gen.model_id or pass explicitly):
      openai/gpt-4o-mini          OPENAI_API_KEY
      anthropic/claude-3-haiku    ANTHROPIC_API_KEY
      google/gemini-2.5-flash     GEMINI_API_KEY
      deepseek/deepseek-chat      DEEPSEEK_API_KEY
    """

    def __init__(self, model_id: str = None):
        load_dotenv(_PROJECT_ROOT / ".env")
        raw = model_id or conf.gen.model_id
        self.model = self._remap(raw)
        logger.info(f"CloudGenerator ready — model: {self.model}")

    @staticmethod
    def _remap(model_id: str) -> str:
        for src, dst in _PREFIX_REMAP.items():
            if model_id.startswith(src):
                return dst + model_id[len(src):]
        return model_id

    def _chat(self, system: str, user: str, max_tokens: int = 400) -> str:
        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=conf.gen.temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def parse_output(self, generated_text: str, term: str) -> Dict | None:
        result = {
            "term": term,
            "source_text": None,
            "target_text_en": None,
            "target_text_fr": None,
            "target_text_es": None,
            "raw_output": generated_text,
        }
        for line in generated_text.split("\n"):
            clean = line.strip()
            if clean.startswith("IT:"):
                result["source_text"] = clean[3:].strip()
            elif clean.startswith("EN:"):
                result["target_text_en"] = clean[3:].strip()
            elif clean.startswith("FR:"):
                result["target_text_fr"] = clean[3:].strip()
            elif clean.startswith("ES:"):
                result["target_text_es"] = clean[3:].strip()
        if result["source_text"] and any([result["target_text_en"], result["target_text_fr"], result["target_text_es"]]):
            return result
        logger.warning(f"Failed to parse output for term: {term}")
        return None

    def translate_text(self, text: str) -> dict:
        out = self._chat(
            TRANSLATION_SYSTEM_PROMPT,
            TRANSLATION_USER_TEMPLATE.format(text=text),
            max_tokens=200,
        )
        result = {
            "source_text": text,
            "target_text_en": None,
            "target_text_fr": None,
            "target_text_es": None,
        }
        for line in out.split("\n"):
            clean = line.strip()
            if clean.startswith("EN:"):
                result["target_text_en"] = clean[3:].strip()
            elif clean.startswith("FR:"):
                result["target_text_fr"] = clean[3:].strip()
            elif clean.startswith("ES:"):
                result["target_text_es"] = clean[3:].strip()
        return result

    def generate_dataset(self, terms: List[str] = None) -> pd.DataFrame:
        terms = terms or get_terms_list()
        logger.info(f"Generating {len(terms)} samples with {self.model} ...")
        data = []
        for i, term in enumerate(terms):
            logger.info(f"[{i+1}/{len(terms)}] {term}")
            try:
                generated_text = self._chat(
                    GENERATION_SYSTEM_PROMPT,
                    GENERATION_USER_TEMPLATE.format(term=term),
                )
                entry = self.parse_output(generated_text, term)
                if entry:
                    data.append(entry)
                    print(f"  IT: {entry['source_text'][:60]}")
                    print(f"  EN: {(entry.get('target_text_en') or '')[:60]}")
            except Exception as e:
                logger.error(f"Error on term '{term}': {e}")
                continue
        df = pd.DataFrame(data)
        logger.info(f"Done. {len(df)} samples generated.")
        return df
