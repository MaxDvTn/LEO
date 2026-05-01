# src/synthesis/prompts.py

GENERATION_SYSTEM_PROMPT = (
    'You are a senior technical writer for "Roverplastik", a leader in window technology and renovation.'
)

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

TRANSLATION_SYSTEM_PROMPT = "You are a professional translator for technical documentation."

TRANSLATION_USER_TEMPLATE = (
    'Translate the following Italian technical sentence into English (EN), French (FR), and Spanish (ES).\n\n'
    'Sentence: "{text}"\n\n'
    "Format exactly as follows:\n"
    "EN: [English translation]\n"
    "FR: [French translation]\n"
    "ES: [Spanish translation]"
)
