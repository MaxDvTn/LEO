# src/synthesis/prompts.py

class Prompts:
    TRANSLATION_TEMPLATE = """
You are a professional translator for technical documentation.
Translate the following technical text from {src_lang} to {tgt_lang}.
Ensure the terminology is accurate and the tone is formal.

Source Text: "{text}"

Translation:
"""
    
    PARAPHRASE_TEMPLATE = """
Rewrite the following sentence to have the same meaning but different wording, suitable for data augmentation.

Original: "{text}"

Rewritten:
"""

    @staticmethod
    def format_translation(text: str, src: str, tgt: str) -> str:
        return Prompts.TRANSLATION_TEMPLATE.format(src_lang=src, tgt_lang=tgt, text=text)

GENERATION_PROMPT_TEMPLATE = """[INST]
You are a senior technical writer for "Roverplastik", a leader in window technology and renovation.
Your task is to write a technical sentence in Italian incorporating the term: "{term}".

Rules:
1. The sentence must be professional, likely to appear in an installation manual or technical datasheet.
2. The context is construction, energy efficiency, and window installation.
3. After the Italian sentence, translate it into technical English (EN), French (FR), and Spanish (ES).

Format exactly as follows:
IT: [Italian Sentence containing "{term}"]
EN: [English Technical Translation]
FR: [French Technical Translation]
ES: [Spanish Technical Translation]
[/INST]
"""
