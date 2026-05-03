import re

import pandas as pd


_MD_BOLD_RE = re.compile(r"\*{1,2}(.+?)\*{1,2}")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_LEADING_NUMBER_RE = re.compile(r"^\s*\d+[.)]\s+")
_INLINE_NUMBER_RE = re.compile(r"\b\d+[.)]\s+(?=[A-ZÀ-Ü])")
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\ufeff]")
_GENERIC_WEB_RE = re.compile(
    r"spazi verdi|parchi pubblici|zone pedonali|mobilit[aà] sostenibile|Marco Ferretti",
    re.IGNORECASE,
)
_FORBIDDEN_TRANSLATION_RE = re.compile(
    r"\bwaste bin\b|\bdumpster\b|\btrash\b|\bgarbage\b|\bcowl\b",
    re.IGNORECASE,
)


def clean_competitor_text(value: object) -> object:
    if pd.isna(value):
        return value
    text = str(value)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = text.replace("\xa0", " ")
    text = _INLINE_NUMBER_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_competitor_term(value: object) -> object:
    if pd.isna(value):
        return value
    text = clean_competitor_text(value)
    text = _LEADING_NUMBER_RE.sub("", str(text)).strip()
    return text


def clean_competitor_df(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean competitor synthetic rows before training/ensemble use."""
    before = len(df)
    out = df.copy()

    for col in ("source_text", "target_text"):
        if col in out.columns:
            out[col] = out[col].map(clean_competitor_text)

    if "term" in out.columns:
        out["term"] = out["term"].map(clean_competitor_term)

    required = ["source_text", "target_text", "source_lang", "target_lang"]
    out = out.dropna(subset=[c for c in required if c in out.columns])
    for col in ("source_text", "target_text"):
        if col in out.columns:
            out = out[out[col].astype(str).str.len() > 0]

    generic_mask = out.apply(
        lambda row: bool(
            _GENERIC_WEB_RE.search(str(row.get("source_text", "")))
            or _GENERIC_WEB_RE.search(str(row.get("target_text", "")))
        ),
        axis=1,
    )
    forbidden_mask = out["target_text"].astype(str).str.contains(_FORBIDDEN_TRANSLATION_RE, na=False)

    removed_generic = int(generic_mask.sum())
    removed_forbidden = int((~generic_mask & forbidden_mask).sum())
    out = out[~generic_mask & ~forbidden_mask]

    dedup_subset = [c for c in required if c in out.columns]
    before_dedup = len(out)
    out = out.drop_duplicates(subset=dedup_subset).reset_index(drop=True)

    stats = {
        "before": before,
        "after": len(out),
        "removed_total": before - len(out),
        "removed_generic": removed_generic,
        "removed_forbidden": removed_forbidden,
        "removed_duplicates": before_dedup - len(out),
    }
    return out, stats
