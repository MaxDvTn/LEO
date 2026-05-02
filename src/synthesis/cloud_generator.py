# src/synthesis/cloud_generator.py
import logging
import time
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
    DOC_TYPES,
    format_generation_prompt,
    TRANSLATION_SYSTEM_PROMPT,
    TRANSLATION_USER_TEMPLATE,
)
from src.synthesis.parsing import parse_translations
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

    def _chat_with_retry(self, system: str, user: str, max_tokens: int = 400, max_retries: int = 3) -> str:
        """Call _chat with exponential-backoff retry on transient failures."""
        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(max_retries):
            try:
                return self._chat(system, user, max_tokens)
            except Exception as e:
                last_exc = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
        raise last_exc

    def parse_output(self, generated_text: str, term: str) -> Dict | None:
        result = {
            "term": term,
            "source_text": None,
            "target_text_en": None,
            "target_text_fr": None,
            "target_text_es": None,
            "raw_output": generated_text,
        }
        result.update(parse_translations(generated_text, include_source=True))
        if result["source_text"] and any([result["target_text_en"], result["target_text_fr"], result["target_text_es"]]):
            return result
        logger.warning(f"Failed to parse output for term: {term}")
        return None

    def translate_text(self, text: str) -> dict:
        out = self._chat_with_retry(
            TRANSLATION_SYSTEM_PROMPT,
            TRANSLATION_USER_TEMPLATE.format(text=text),
            max_tokens=240,
        )
        result = {
            "source_text": text,
            "target_text_en": None,
            "target_text_fr": None,
            "target_text_es": None,
        }
        result.update(parse_translations(out, include_source=False))
        return result

    def generate_dataset(self, terms=None, num_variants: int = 1) -> pd.DataFrame:
        """Generate synthetic translation pairs.

        Args:
            terms: List of term strings or list of dicts with 'term'/'context' keys.
            num_variants: How many sentences to generate per term, each using a different
                          document-type prompt to increase dataset diversity.
        """
        if terms is None:
            terms = get_terms_list(with_context=True)

        # Normalise to list of dicts
        normalized = [
            t if isinstance(t, dict) else {"term": t, "context": "general"}
            for t in terms
        ]

        effective_variants = max(1, num_variants)
        logger.info(
            f"Generating {len(normalized)} terms × {effective_variants} variant(s) "
            f"with {self.model} ..."
        )
        data = []

        for i, term_dict in enumerate(normalized):
            term = term_dict["term"]
            context = term_dict.get("context", "general")
            logger.info(f"[{i + 1}/{len(normalized)}] {term}")

            for v in range(effective_variants):
                doc_type = DOC_TYPES[v % len(DOC_TYPES)]
                try:
                    prompt = format_generation_prompt(term, context=context, doc_type=doc_type)
                    generated_text = self._chat_with_retry(
                        GENERATION_SYSTEM_PROMPT, prompt, conf.gen.max_new_tokens
                    )
                    entry = self.parse_output(generated_text, term)
                    if entry:
                        entry["context"] = context
                        entry["doc_type"] = doc_type
                        data.append(entry)
                        print(f"  IT: {entry['source_text'][:60]}")
                        print(f"  EN: {(entry.get('target_text_en') or '')[:60]}")
                except Exception as e:
                    logger.error(f"Error on term '{term}' (variant {v + 1}): {e}")

        df = pd.DataFrame(data)
        logger.info(f"Done. {len(df)} samples generated.")
        return df
