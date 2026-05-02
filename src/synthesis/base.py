# src/synthesis/base.py
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

import pandas as pd

from src.synthesis.glossary_data import get_terms_list
from src.synthesis.prompts import (
    DOC_TYPES,
    GENERATION_SYSTEM_PROMPT,
    format_generation_prompt,
)
from src.synthesis.parsing import parse_translations
from src.common.config import conf

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    """Abstract base for synthetic data generators."""

    # ------------------------------------------------------------------ #
    # Abstract interface — subclasses implement only these two methods     #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _chat(self, system: str, user: str, max_tokens: int = 400) -> str:
        """Single LLM call. Raises on failure."""
        ...

    @abstractmethod
    def translate_text(self, text: str) -> dict:
        """Translate an Italian sentence to EN/FR/ES.
        Returns dict with source_text + target_text_en/fr/es keys."""
        ...

    # ------------------------------------------------------------------ #
    # Shared concrete methods                                              #
    # ------------------------------------------------------------------ #

    @property
    def num_workers(self) -> int:
        """Parallelism degree. Subclasses override for backend-specific defaults."""
        return max(1, conf.gen.num_workers)

    def _chat_with_retry(
        self, system: str, user: str, max_tokens: int = 400, max_retries: int = 3
    ) -> str:
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
        if result["source_text"] and any(
            [result["target_text_en"], result["target_text_fr"], result["target_text_es"]]
        ):
            return result
        logger.warning(f"Failed to parse output for term: {term}")
        return None

    def generate_dataset(self, terms=None, num_variants: int = 1) -> pd.DataFrame:
        """Generate synthetic translation pairs, optionally in parallel.

        Args:
            terms: List of term strings or list of dicts with 'term'/'context' keys.
                   If None, uses the full domain glossary.
            num_variants: How many sentences to generate per term, each using a
                          different document-type prompt for diversity.
        """
        if terms is None:
            terms = get_terms_list(with_context=True)

        normalized = [
            t if isinstance(t, dict) else {"term": t, "context": "general"}
            for t in terms
        ]

        effective_variants = max(1, num_variants)
        num_workers = self.num_workers

        # Flatten to individual (term_dict, variant_index) tasks
        tasks = [
            (term_dict, v)
            for term_dict in normalized
            for v in range(effective_variants)
        ]

        logger.info(
            f"Generating {len(normalized)} terms × {effective_variants} variant(s) "
            f"= {len(tasks)} tasks, workers={num_workers}"
        )

        def _run_task(term_dict: dict, v: int) -> Dict | None:
            term = term_dict["term"]
            context = term_dict.get("context", "general")
            doc_type = DOC_TYPES[v % len(DOC_TYPES)]
            prompt = format_generation_prompt(term, context=context, doc_type=doc_type)
            generated_text = self._chat_with_retry(
                GENERATION_SYSTEM_PROMPT, prompt, conf.gen.max_new_tokens
            )
            entry = self.parse_output(generated_text, term)
            if entry:
                entry["context"] = context
                entry["doc_type"] = doc_type
            return entry

        data = []
        completed = 0
        total = len(tasks)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_term = {
                executor.submit(_run_task, term_dict, v): term_dict["term"]
                for term_dict, v in tasks
            }
            for future in as_completed(future_to_term):
                term = future_to_term[future]
                completed += 1
                try:
                    entry = future.result()
                    if entry:
                        data.append(entry)
                        print(
                            f"  [{completed}/{total}] {term}: "
                            f"{entry['source_text'][:60]}"
                        )
                    else:
                        logger.warning(f"  [{completed}/{total}] {term}: no valid output")
                except Exception as e:
                    logger.error(f"  [{completed}/{total}] {term}: error — {e}")

        df = pd.DataFrame(data)
        logger.info(f"Done. {len(df)} samples generated.")
        return df
