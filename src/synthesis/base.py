# src/synthesis/base.py
from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class BaseGenerator(ABC):
    """Abstract base for synthetic data generators."""

    @abstractmethod
    def generate_dataset(self, terms=None, num_variants: int = 1) -> pd.DataFrame:
        """Generate a DataFrame with columns: term, source_text, target_text_en/fr/es, raw_output.

        Args:
            terms: List of term strings or list of dicts with 'term' and 'context' keys.
                   If None, uses the full domain glossary.
            num_variants: Number of sentences to generate per term, each using a different
                          document-type prompt (manual, datasheet, catalog, …).
        """
        ...

    def translate_text(self, text: str) -> dict:
        """Translate an Italian sentence to EN/FR/ES. Returns dict with source_text + target_text_en/fr/es."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement translate_text()")
