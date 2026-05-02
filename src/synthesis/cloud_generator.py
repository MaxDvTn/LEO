# src/synthesis/cloud_generator.py
import logging
from pathlib import Path

try:
    from dotenv import load_dotenv
    import litellm
    litellm.drop_params = True
except ImportError:
    raise ImportError("Run: pip install litellm python-dotenv")

from src.synthesis.base import BaseGenerator
from src.synthesis.prompts import TRANSLATION_SYSTEM_PROMPT, TRANSLATION_USER_TEMPLATE
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
        logger.info(f"CloudGenerator ready — model: {self.model}, workers: {conf.gen.cloud_num_workers}")

    @property
    def num_workers(self) -> int:
        return max(1, conf.gen.cloud_num_workers)

    @staticmethod
    def _remap(model_id: str) -> str:
        for src, dst in _PREFIX_REMAP.items():
            if model_id.startswith(src):
                return dst + model_id[len(src):]
        return model_id

    def _chat(self, system: str, user: str, max_tokens: int = 400) -> str:
        kwargs = {}
        if self.model.startswith("gemini/gemini-2.5-flash"):
            kwargs["thinking"] = {"type": "disabled", "budget_tokens": 0}

        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=conf.gen.temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.choices[0].message.content

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
