# src/synthesis/ollama_generator.py
import logging
import pandas as pd
from typing import List, Dict

try:
    import ollama
except ImportError:
    raise ImportError("Run: pip install ollama")

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


class OllamaGenerator(BaseGenerator):
    """
    Synthetic data generator backed by a local Ollama server.

    Model ID format in config: "ollama/<model_name>"
    Examples: "ollama/qwen2.5:32b", "ollama/qwen2.5:7b", "ollama/llama3.1:8b"

    Requires Ollama running locally:  ollama serve
    Pull the model first:             ollama pull qwen2.5:32b
    """

    def __init__(self, model_id: str = None):
        raw = model_id or conf.gen.model_id
        # Strip the "ollama/" prefix to get the Ollama model name
        self.model = raw.removeprefix("ollama/")
        self.base_url = conf.gen.ollama_base_url

        logger.info(f"OllamaGenerator ready — model: {self.model}, server: {self.base_url}")
        self._check_model()

    def _check_model(self):
        """Warn if the model is not pulled yet."""
        try:
            client = ollama.Client(host=self.base_url)
            available = [m.model for m in client.list().models]
            if self.model not in available:
                logger.warning(
                    f"Model '{self.model}' not found locally. "
                    f"Run: ollama pull {self.model}"
                )
        except Exception as e:
            logger.warning(f"Could not reach Ollama server at {self.base_url}: {e}")

    def _chat(self, system: str, user: str, max_tokens: int = 400) -> str:
        client = ollama.Client(host=self.base_url)
        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={
                "temperature": conf.gen.temperature,
                "top_p": conf.gen.top_p,
                "num_predict": max_tokens,
            },
        )
        return response.message.content

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
        client = ollama.Client(host=self.base_url)
        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                {"role": "user", "content": TRANSLATION_USER_TEMPLATE.format(text=text)},
            ],
            options={"temperature": 0.1, "num_predict": 200},
        )
        result = {
            "source_text": text,
            "target_text_en": None,
            "target_text_fr": None,
            "target_text_es": None,
        }
        for line in response.message.content.split("\n"):
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
        logger.info(f"Generating {len(terms)} samples with Ollama/{self.model} ...")
        data = []

        for i, term in enumerate(terms):
            logger.info(f"[{i+1}/{len(terms)}] {term}")
            try:
                generated_text = self._chat(
                    GENERATION_SYSTEM_PROMPT,
                    GENERATION_USER_TEMPLATE.format(term=term),
                    conf.gen.max_new_tokens,
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
