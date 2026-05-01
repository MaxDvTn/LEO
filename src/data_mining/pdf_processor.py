import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize
import re
import logging

# Assicurati di aver scaricato il tokenizer di NLTK
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PDF_MINER")

class PdfMiner:
    def __init__(self, min_length=15):
        self.min_length = min_length

    def extract_text_from_pdf(self, pdf_path):
        """Estrae tutto il testo da un PDF."""
        full_text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                full_text += page.get_text() + " "
        return full_text

    def clean_and_segment(self, raw_text):
        """
        Pulisce il testo e lo divide in frasi di senso compiuto.
        Rimuove numeri di pagina, intestazioni strane, ecc.
        """
        # 1. Pulizia preliminare (Rimuove spazi multipli e newlines spezzate)
        text = re.sub(r'\s+', ' ', raw_text).strip()

        # 2. Segmentazione in frasi (uso NLTK che è smart)
        sentences = sent_tokenize(text, language='italian')

        valid_sentences = []
        for sent in sentences:
            sent = sent.strip()

            # --- FILTRI DI QUALITÀ ---
            # Deve essere lunga abbastanza
            if len(sent) < self.min_length:
                continue

            # Non deve iniziare con numeri strani (es. indici di sommario "1.2.3")
            if re.match(r'^[\d\.\s]+$', sent):
                continue

            # Deve contenere almeno una lettera (evita "-----------")
            if not re.search(r'[a-zA-Z]', sent):
                continue

            valid_sentences.append(sent)

        return list(dict.fromkeys(valid_sentences))  # Removes exact duplicates, preserving order.
