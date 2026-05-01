# src/synthesis/base.py
from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class BaseGenerator(ABC):
    """Abstract base for synthetic data generators."""

    @abstractmethod
    def generate_dataset(self, terms: List[str] = None) -> pd.DataFrame:
        """Generate a DataFrame with columns: term, source_text, target_text_en/fr/es, raw_output."""
        ...

    def translate_text(self, text: str) -> dict:
        """Translate an Italian sentence to EN/FR/ES. Returns dict with source_text + target_text_en/fr/es."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement translate_text()")