# src/synthesis/ollama_generator.py
import logging
import re

try:
    import ollama
except ImportError:
    raise ImportError("Run: pip install ollama")

from src.synthesis.base import BaseGenerator
from src.synthesis.prompts import TRANSLATION_SYSTEM_PROMPT, TRANSLATION_USER_TEMPLATE
from src.synthesis.parsing import parse_translations
from src.common.config import conf

logger = logging.getLogger(__name__)


def _workers_for_model(model_name: str) -> int:
    """Infer safe parallel-worker count from model size tag (e.g. 'qwen2.5:32b' → 2)."""
    m = re.search(r":(\d+)b", model_name.lower())
    if m:
        size = int(m.group(1))
        if size >= 20:
            return 2
        if size >= 12:
            return 3
    return conf.gen.num_workers  # fall back to config for untagged / small models


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

        self._num_workers = _workers_for_model(self.model)
        logger.info(
            f"OllamaGenerator ready — model: {self.model}, "
            f"server: {self.base_url}, workers: {self._num_workers}"
        )
        self._check_model()

    @property
    def num_workers(self) -> int:
        return self._num_workers

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

