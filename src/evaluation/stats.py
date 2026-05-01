import pandas as pd
import numpy as np
from src.synthesis.glossary_data import get_terms_list

def calculate_stats(df: pd.DataFrame) -> dict:
    """
    Computes statistical and coverage metrics for a dataset of translated sentences.
    Assumes df has 'source_text' and 'target_text' columns.
    """
    if df.empty or 'target_text' not in df.columns or 'source_text' not in df.columns:
        return {}

    source_texts = df['source_text'].dropna().astype(str).tolist()
    target_texts = df['target_text'].dropna().astype(str).tolist()

    if not target_texts:
        return {}

    # 1. Length Metrics
    source_lengths = [len(text.split()) for text in source_texts]
    target_lengths = [len(text.split()) for text in target_texts]

    avg_src_len = np.mean(source_lengths) if source_lengths else 0
    avg_tgt_len = np.mean(target_lengths) if target_lengths else 0
    src_len_var = np.var(source_lengths) if source_lengths else 0
    tgt_len_var = np.var(target_lengths) if target_lengths else 0

    # 2. Type-Token Ratio (TTR) for target
    all_target_words = []
    for text in target_texts:
        all_target_words.extend(text.lower().split())
    
    unique_words = set(all_target_words)
    total_words = len(all_target_words)
    ttr = len(unique_words) / total_words if total_words > 0 else 0

    # 3. Glossary Coverage
    # Check how many glossary terms appear in the Italian source sentences.
    terms = get_terms_list()
    joined_source_text = " ".join(source_texts).lower()

    covered_terms_count = 0
    for term in terms:
        if term.lower() in joined_source_text:
            covered_terms_count += 1
            
    coverage_percentage = (covered_terms_count / len(terms)) * 100 if terms else 0

    return {
        "avg_source_len": round(avg_src_len, 2),
        "avg_target_len": round(avg_tgt_len, 2),
        "var_source_len": round(src_len_var, 2),
        "var_target_len": round(tgt_len_var, 2),
        "ttr": round(ttr, 4),
        "glossary_coverage_pct": round(coverage_percentage, 2),
        "total_sentences": len(target_texts)
    }
