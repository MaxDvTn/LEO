import re


_LINE_RE = re.compile(r"^\s*(?:[-*]\s*)?(IT|EN|FR|ES)\s*:\s*(.+?)\s*$", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"^\[.*\]$")


def clean_generated_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip().strip('"').strip("'").strip()
    if not cleaned or _PLACEHOLDER_RE.match(cleaned):
        return None
    return cleaned


def parse_prefixed_translations(text: str, *, include_source: bool) -> dict:
    result = {
        "source_text": None,
        "target_text_en": None,
        "target_text_fr": None,
        "target_text_es": None,
    }
    key_by_prefix = {
        "IT": "source_text",
        "EN": "target_text_en",
        "FR": "target_text_fr",
        "ES": "target_text_es",
    }

    for line in str(text).splitlines():
        match = _LINE_RE.match(line)
        if not match:
            continue
        prefix, value = match.groups()
        if prefix.upper() == "IT" and not include_source:
            continue
        cleaned = clean_generated_value(value)
        if cleaned:
            result[key_by_prefix[prefix.upper()]] = cleaned

    return result
