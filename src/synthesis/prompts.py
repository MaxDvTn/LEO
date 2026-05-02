# src/synthesis/prompts.py

GENERATION_SYSTEM_PROMPT = (
    'You are a senior technical writer for "Roverplastik", a leader in window technology and renovation.'
)

CONTEXT_DESCRIPTIONS = {
    "structure": "structural components and building assemblies",
    "sealing": "sealing materials and weatherproofing",
    "window_frame": "window frame and profile systems",
    "insulation": "thermal and acoustic insulation",
    "shading": "solar shading and blind systems",
    "construction": "construction processes and building techniques",
    "installation": "installation procedures and on-site work",
    "hardware": "hardware, fasteners, and mechanical components",
    "physics": "thermal and acoustic physics, energy performance",
    "ecology": "energy efficiency and environmental performance",
    "performance": "product performance testing and certification",
    "problem": "building defects, condensation, and moisture problems",
    "renovation": "renovation, refurbishment, and energy retrofit",
    "action": "installation actions and construction procedures",
    "product": "product specifications and catalog descriptions",
    "solution": "technical solutions and system integrations",
}

DOC_TYPES = [
    "an installation manual",
    "a technical datasheet",
    "a product catalog",
    "a technical specification document",
    "a building energy report",
]

_GENERATION_BODY = (
    'Write a technical sentence in Italian incorporating the term: "{term}".\n\n'
    "Rules:\n"
    "1. The sentence must be professional and realistic, written as part of {doc_type}.\n"
    "2. Domain context: {domain_context}.\n"
    "3. After the Italian sentence, translate it into technical English (EN), French (FR), and Spanish (ES).\n\n"
    "Return only valid JSON with these exact keys:\n"
    "{{\n"
    '  "source_text": "Italian sentence containing {term}",\n'
    '  "target_text_en": "English technical translation",\n'
    '  "target_text_fr": "French technical translation",\n'
    '  "target_text_es": "Spanish technical translation"\n'
    "}}"
)

# Legacy constant kept so existing callers that do GENERATION_USER_TEMPLATE.format(term=...) still work.
GENERATION_USER_TEMPLATE = (
    'Write a technical sentence in Italian incorporating the term: "{term}".\n\n'
    "Rules:\n"
    "1. The sentence must be professional, likely to appear in an installation manual or technical datasheet.\n"
    "2. The context is construction, energy efficiency, and window installation.\n"
    "3. After the Italian sentence, translate it into technical English (EN), French (FR), and Spanish (ES).\n\n"
    "Format exactly as follows:\n"
    'IT: [Italian Sentence containing "{term}"]\n'
    "EN: [English Technical Translation]\n"
    "FR: [French Technical Translation]\n"
    "ES: [Spanish Technical Translation]"
)


def format_generation_prompt(
    term: str,
    context: str = "general",
    doc_type: str = "an installation manual",
) -> str:
    """Build a generation prompt enriched with domain context and document type."""
    domain_context = CONTEXT_DESCRIPTIONS.get(context, "construction and window technology")
    return _GENERATION_BODY.format(term=term, doc_type=doc_type, domain_context=domain_context)


TRANSLATION_SYSTEM_PROMPT = "You are a professional translator for technical documentation."

TRANSLATION_USER_TEMPLATE = (
    'Translate the following Italian technical sentence into English (EN), French (FR), and Spanish (ES).\n\n'
    'Sentence: "{text}"\n\n'
    "Return only valid JSON with these exact keys:\n"
    "{{\n"
    '  "target_text_en": "English translation",\n'
    '  "target_text_fr": "French translation",\n'
    '  "target_text_es": "Spanish translation"\n'
    "}}"
)
