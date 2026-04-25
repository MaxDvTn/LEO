import sys
from pathlib import Path

# Aggiungi la root del progetto al path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_mining.aligner import BitextAligner

def main():
    # 1. Inizializza l'aligner (carica il modello LaBSE)
    aligner = BitextAligner()

    # 2. Definiamo due liste di frasi scompagnate (es. estratte da un manuale IT e uno EN)
    # Nota: l'ordine non deve essere necessariamente lo stesso!
    it_sentences = [
        "La guarnizione deve essere inserita nella battuta del telaio.",
        "Il cassonetto coibentato riduce i ponti termici.",
        "Il motore tubolare ha una coppia di 20 Nm.",
        "Questa è una frase inutile senza traduzione."
    ]

    en_sentences = [
        "The insulated box reduces thermal bridges.",
        "The tubular motor has a torque of 20 Nm.",
        "The gasket must be inserted into the frame rebate.",
        "Something about windows that doesn't match."
    ]

    print("\n📐 Inizio allineamento bi-testuale...")
    
    # 3. Eseguiamo l'allineamento
    # threshold=0.75 è robusto per traduzioni tecniche
    matches = aligner.align_sentences(it_sentences, en_sentences, threshold=0.7)

    # 4. Mostriamo i risultati
    print(f"\n✅ Trovate {len(matches)} corrispondenze:")
    for m in matches:
        print("-" * 30)
        print(f"🇮🇹 IT: {m['source_text']}")
        print(f"🇬🇧 EN: {m['target_text']}")
        print(f"📊 Score: {m['score']:.4f}")

    if not matches:
        print("❌ Nessuna corrispondenza trovata sopra la soglia.")

if __name__ == "__main__":
    main()
