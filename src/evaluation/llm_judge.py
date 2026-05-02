import json
import logging
import re
from pathlib import Path

import litellm
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from src.common.config import conf

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _litellm_model_id(model_id: str) -> str:
    if model_id.startswith("google/"):
        return "gemini/" + model_id[len("google/"):]
    return model_id


JUDGE_PROMPT = """\
You are an expert linguist evaluating synthetic NMT training data (Italian → target language).
Score each criterion from 1 (worst) to 5 (best):

1. Fluency: grammatically correct and natural-sounding target sentence
2. Adequacy: meaning faithfully conveyed, no omissions or additions
3. Terminology: correct technical terms for windows, doors, building components

Source (Italian): "{source_text}"
Target: "{target_text}"

Return ONLY valid JSON, no other text:
{{"fluency": <1-5>, "adequacy": <1-5>, "terminology": <1-5>}}
"""

# Fallback: matches "Fluency: 4", "Fluency: **4**", "Fluency: [4]", "fluency : 4/5"
_SCORE_RE = {
    "fluency":     re.compile(r"fluency\s*[:\-]\s*\**\[?([1-5])\]?\**", re.I),
    "adequacy":    re.compile(r"adequacy\s*[:\-]\s*\**\[?([1-5])\]?\**", re.I),
    "terminology": re.compile(r"terminolog\w*\s*[:\-]\s*\**\[?([1-5])\]?\**", re.I),
}


def _parse_scores(content: str) -> dict[str, int] | None:
    """Try JSON first, then regex fallback. Returns dict or None on failure."""
    # Strip markdown fences if present
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", content.strip(), flags=re.M)

    # JSON path
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            scores = {}
            for key in ("fluency", "adequacy", "terminology"):
                val = parsed.get(key)
                if val is None:
                    val = parsed.get(key.capitalize())
                if val is not None and str(val).strip().isdigit():
                    scores[key] = max(1, min(5, int(val)))
            if len(scores) == 3:
                return scores
        except (json.JSONDecodeError, ValueError):
            pass

    # Regex fallback
    scores = {}
    for key, pattern in _SCORE_RE.items():
        m = pattern.search(content)
        if m:
            scores[key] = max(1, min(5, int(m.group(1))))
    return scores if len(scores) == 3 else None


def evaluate_with_llm(
    df: pd.DataFrame,
    sample_size: int = 100,
    model_id: str | None = None,
) -> dict:
    """Evaluate a random sample of translation pairs with an LLM judge.

    Returns average scores for fluency, adequacy, terminology, plus
    judge_samples_successful / judge_samples_attempted for visibility.
    """
    if df.empty or "source_text" not in df.columns or "target_text" not in df.columns:
        return {}

    n_samples = min(len(df), sample_size)
    sampled_df = df.sample(n=n_samples, random_state=42)

    load_dotenv(PROJECT_ROOT / ".env")
    mid = _litellm_model_id(model_id or conf.gen.model_id)

    call_kwargs: dict = {}
    if mid.startswith("ollama/"):
        call_kwargs["api_base"] = conf.gen.ollama_base_url
    if mid.startswith("gemini/gemini-2.5-flash"):
        call_kwargs["thinking"] = {"type": "disabled", "budget_tokens": 0}

    scores: dict[str, list[int]] = {"fluency": [], "adequacy": [], "terminology": []}
    attempted = 0

    print(f"   🤖 LLM Judge: {n_samples} samples with {mid}...")

    for _, row in tqdm(sampled_df.iterrows(), total=n_samples, desc="LLM Judge"):
        prompt = JUDGE_PROMPT.format(
            source_text=str(row["source_text"]),
            target_text=str(row["target_text"]),
        )
        attempted += 1
        try:
            response = litellm.completion(
                model=mid,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=80,
                **call_kwargs,
            )
            content = response.choices[0].message.content or ""
            parsed = _parse_scores(content)
            if parsed:
                for key in scores:
                    scores[key].append(parsed[key])
            else:
                logger.debug(f"Judge parse failed — raw response: {content!r}")
        except Exception as e:
            logger.warning(f"LLM judge error: {e}")

    successful = len(scores["fluency"])
    if successful == 0:
        logger.warning(
            f"Judge produced 0 parseable scores out of {attempted} attempts with {mid}. "
            "Check that the model is reachable and returns the expected JSON format."
        )

    def _avg(lst: list) -> float:
        return round(sum(lst) / len(lst), 2) if lst else 0.0

    return {
        "avg_fluency":               _avg(scores["fluency"]),
        "avg_adequacy":              _avg(scores["adequacy"]),
        "avg_terminology":           _avg(scores["terminology"]),
        "judge_samples_successful":  successful,
        "judge_samples_attempted":   attempted,
    }
