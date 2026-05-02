import json
import re


_LINE_RE = re.compile(r"^\s*(?:[-*]\s*)?(IT|EN|FR|ES)\s*:\s*(.+?)\s*$", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"^\[.*\]$")
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)

_JSON_KEYS = {
    "source_text": ("source_text", "italian", "it", "source", "source_sentence"),
    "target_text_en": ("target_text_en", "english", "en", "translation_en", "en_translation"),
    "target_text_fr": ("target_text_fr", "french", "fr", "translation_fr", "fr_translation"),
    "target_text_es": ("target_text_es", "spanish", "es", "translation_es", "es_translation"),
}


def clean_generated_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip().strip('"').strip("'").strip()
    if not cleaned or _PLACEHOLDER_RE.match(cleaned):
        return None
    return cleaned


def _empty_translation_result() -> dict:
    return {
        "source_text": None,
        "target_text_en": None,
        "target_text_fr": None,
        "target_text_es": None,
    }


def _extract_json_object(text: str) -> dict | None:
    cleaned = _FENCE_RE.sub("", str(text)).strip()
    candidates = [cleaned]

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(cleaned[start:end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
    return None


def _lookup_nested(parsed: dict, aliases: tuple[str, ...]) -> object | None:
    for alias in aliases:
        if alias in parsed:
            return parsed[alias]

    translations = parsed.get("translations")
    if isinstance(translations, dict):
        for alias in aliases:
            if alias in translations:
                return translations[alias]

    return None


def parse_json_translations(text: str, *, include_source: bool) -> dict:
    result = _empty_translation_result()
    parsed = _extract_json_object(text)
    if not parsed:
        return result

    for output_key, aliases in _JSON_KEYS.items():
        if output_key == "source_text" and not include_source:
            continue
        value = _lookup_nested(parsed, aliases)
        cleaned = clean_generated_value(value)
        if cleaned:
            result[output_key] = cleaned
    return result


def parse_prefixed_translations(text: str, *, include_source: bool) -> dict:
    result = _empty_translation_result()
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


def parse_translations(text: str, *, include_source: bool) -> dict:
    """Parse JSON first, then fall back to IT:/EN:/FR:/ES: lines."""
    result = parse_json_translations(text, include_source=include_source)
    if result["source_text"] or any(
        result[key] for key in ("target_text_en", "target_text_fr", "target_text_es")
    ):
        return result
    return parse_prefixed_translations(text, include_source=include_source)
