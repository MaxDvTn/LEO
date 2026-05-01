import torch
from sentence_transformers import SentenceTransformer, util
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ALIGNER")

class BitextAligner:
    def __init__(self, model_name="sentence-transformers/LaBSE"):
        """
        LaBSE è il modello standard di Google per trovare traduzioni 
        in mezzo al caos di testi non allineati.
        """
        logger.info(f"⏳ Loading Alignment Model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def align_sentences(self, src_sentences, tgt_sentences, threshold=0.75):
        """
        Confronta tutte le frasi IT con tutte le frasi EN e trova le coppie.
        threshold=0.75 significa "accetta solo se sei sicuro al 75%".
        """
        if not src_sentences or not tgt_sentences:
            return []
        logger.info(f"📐 Encoding {len(src_sentences)} source sentences...")
        src_embeddings = self.model.encode(src_sentences, convert_to_tensor=True, show_progress_bar=False)
        
        logger.info(f"📐 Encoding {len(tgt_sentences)} target sentences...")
        tgt_embeddings = self.model.encode(tgt_sentences, convert_to_tensor=True, show_progress_bar=False)

        logger.info("⚡ Computing similarity matrix (Heavy GPU task)...")
        # Calcola la similarità coseno tra TUTTI i vettori
        cosine_scores = util.cos_sim(src_embeddings, tgt_embeddings)

        aligned_pairs = []
        used_tgt_indices = set()

        # Per ogni frase sorgente, troviamo la migliore frase target
        # (Logica Greedy: prende la migliore e passa avanti)
        for i in range(len(src_sentences)):
            scores = cosine_scores[i]
            best_score_idx = torch.argmax(scores).item()
            best_score = scores[best_score_idx].item()

            if best_score >= threshold:
                if best_score_idx not in used_tgt_indices:
                    aligned_pairs.append({
                        "source_text": src_sentences[i],
                        "target_text": tgt_sentences[best_score_idx],
                        "score": best_score
                    })
                    used_tgt_indices.add(best_score_idx)
        
        return aligned_pairs