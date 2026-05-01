import logging
import pandas as pd
import litellm
import re
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from src.common.config import conf

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _litellm_model_id(model_id: str) -> str:
    if model_id.startswith("google/"):
        return "gemini/" + model_id[len("google/"):]
    return model_id

JUDGE_PROMPT = """You are an expert linguist and translator evaluating synthetic datasets for Neural Machine Translation fine-tuning.
You must evaluate the translation from Italian to the target language based on three criteria. Give a score from 1 to 5 for each.

Criteria:
1. Fluency: Is the target sentence grammatically correct and natural-sounding? (1=unreadable, 5=perfectly natural)
2. Adequacy: Does the target sentence accurately convey the meaning of the source sentence without omissions or additions? (1=completely different, 5=perfectly faithful)
3. Terminology: Does the target sentence correctly translate technical terms related to windows, doors, and building components? (1=completely wrong/generic, 5=perfect domain terminology)

Source (Italian): "{source_text}"
Target: "{target_text}"

Respond STRICTLY in the following format and nothing else:
Fluency: [score]
Adequacy: [score]
Terminology: [score]
"""

def evaluate_with_llm(df: pd.DataFrame, sample_size: int = 100, model_id: str = None) -> dict:
    """
    Evaluates a random sample of translation pairs using an LLM as a judge.
    Returns average scores for Fluency, Adequacy, and Terminology.
    """
    if df.empty or 'target_text' not in df.columns or 'source_text' not in df.columns:
        return {}

    # Sample rows
    n_samples = min(len(df), sample_size)
    sampled_df = df.sample(n=n_samples, random_state=42)

    load_dotenv(PROJECT_ROOT / ".env")
    mid = _litellm_model_id(model_id or conf.gen.model_id)
    
    # Configure litellm for ollama if needed
    kwargs = {}
    if mid.startswith("ollama/"):
        kwargs["api_base"] = conf.gen.ollama_base_url

    scores = {"fluency": [], "adequacy": [], "terminology": []}
    
    print(f"   🤖 Evaluating {n_samples} sentences with LLM Judge ({mid})...")

    for _, row in tqdm(sampled_df.iterrows(), total=n_samples, desc="LLM Judge"):
        source = str(row['source_text'])
        target = str(row['target_text'])
        
        prompt = JUDGE_PROMPT.format(source_text=source, target_text=target)
        
        try:
            response = litellm.completion(
                model=mid,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=80,
                **kwargs
            )
            content = response.choices[0].message.content
            
            # Parse the output
            f_match = re.search(r"Fluency:\s*([1-5])", content, re.IGNORECASE)
            a_match = re.search(r"Adequacy:\s*([1-5])", content, re.IGNORECASE)
            t_match = re.search(r"Terminology:\s*([1-5])", content, re.IGNORECASE)
            
            if f_match and a_match and t_match:
                scores["fluency"].append(int(f_match.group(1)))
                scores["adequacy"].append(int(a_match.group(1)))
                scores["terminology"].append(int(t_match.group(1)))

        except Exception as e:
            logger.warning(f"LLM judge error: {e}")

    avg_f = sum(scores["fluency"]) / len(scores["fluency"]) if scores["fluency"] else 0
    avg_a = sum(scores["adequacy"]) / len(scores["adequacy"]) if scores["adequacy"] else 0
    avg_t = sum(scores["terminology"]) / len(scores["terminology"]) if scores["terminology"] else 0

    return {
        "avg_fluency": round(avg_f, 2),
        "avg_adequacy": round(avg_a, 2),
        "avg_terminology": round(avg_t, 2),
        "judge_samples_successful": len(scores["fluency"])
    }
