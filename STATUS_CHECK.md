# 🦁 Project L.E.O. - Status Check & Advancement Report

## 📈 Stato Attuale del Progetto (Progress Report)

Il progetto **L.E.O.** si trova in una fase avanzata di sviluppo dell'infrastruttura di dati e training. Ecco il check degli avanzamenti basato sul piano originale e le implementazioni correnti:

### 1. 🏗️ Architettura Core (Monorepo)
- [x] **Configurazione Centralizzata:** `src/common/config.py` gestisce percorsi, parametri del modello e WandB.
- [x] **Motore di Training:** `src/training/trainer_engine.py` e `model_module.py` implementano QLoRA (4-bit) con salvataggio checkpoint.
- [x] **Logging Avanzato:** Integrazione completa con **WandB** per monitorare loss e tabelle di traduzione in tempo reale.
- [x] **Sicurezza:** Risolte le vulnerabilità `safetensors` (CVE-2025-32434) aggiornando PyTorch alla v2.9.1.

### 2. ⛏️ Data Mining & Synthesis (Pipeline Dati)
- [x] **Glossary-Based Gen:** Generazione sintetica `run_gen.py` per coprire i termini tecnici principali.
- [x] **PDF Extraction:** `run_pdf_mining.py` estrae testo dai cataloghi Roverplastik e usa l'AI per tradurli.
- [x] **Web Spider:** `competitor_spider.py` analizza i siti dei competitor, estrae termini tecnici e li trasforma in frasi.
- [x] **Alignment System:** Struttura per `BitextAligner` (LaBSE) pronta in `src/data_mining/aligner.py`.
- [x] **Aligner Demo Script:** Creato `scripts/run_alignment.py` per mostrare come allineare cataloghi IT/EN.

### 3. 🎨 User Interface (Validation Factory)
- [x] **Google OAuth2 Integration:** Sistema di login sicuro `src/ui/auth.py` limitato al dominio `@liceodavincitn.it`.
- [x] **Multi-CSV Support:** La WebApp legge automaticamente tutti i file sintetici generati.
- [x] **Validator Tracking:** Salvataggio automatico dell'ID dello studente (`validator_id`) e dello stato della traduzione (`ai_approved` vs `human_corrected`).
- [x] **Student Guide:** Documentazione `GUIDE_STUDENTS.md` creata in IT/EN.

### 4. 📊 Evaluation & Benchmarking
- [x] **Automated Suite:** `scripts/run_benchmark.py` calcola SacreBLEU e CHRF confrontando il modello Base con il modello LEO.
- [x] **Stable Evaluation:** Creato `test_set.csv` e lo script `scripts/create_test_set.py` per benchmark consistenti.
- [x] **WandB Benchmarking:** Risultati dei benchmark caricati automaticamente su WandB con tabelle comparative.

---

## 🛠️ Come usare l'Aligner (BitextAligner)

Se hai un testo in Italiano e la sua traduzione in Inglese estratti separatamente (es. da due PDF diversi o pagine web), puoi usare l'Aligner per trovare le coppie corrette:

1.  **Script di Demo:** Lancia `python scripts/run_alignment.py` per vedere un esempio in azione.
2.  **Esempio di Codice:**
    ```python
    from src.data_mining.aligner import BitextAligner
    aligner = BitextAligner()
    pairs = aligner.align_sentences(lista_it, lista_en, threshold=0.7)
    ```
3.  **Utilizzo Pratico:** Questo è perfetto per "pulire" i dati grezzi dei cataloghi prima di darli in pasto alla WebApp per la validazione finale.

---

## 🚀 Idee per Migliorare il Progetto (Recommendations)

Il framework è solido. Per rendere il progetto "eccellente" (da produzione), ecco cosa suggerisco:

### A. Dataset Quality & Cleaning
1.  **Deduplicazione Semantica:** Ora abbiamo molte frasi da PDF, Web e Glossary. Potresti implementare uno script che usa embeddings (SBERT) per rimuovere frasi troppo simili tra loro per evitare overfitting.
2.  **Back-Translation:** Per aumentare ancora il dataset, potresti prendere le frasi validate dagli studenti (IT -> EN) e ri-tradurle (EN -> IT) con un altro modello per creare nuove varianti.

### B. UI / Developer Experience (DX)
3.  **Dashboard Validator:** Nella WebApp, aggiungi una piccola classifica (Leaderboard) che mostra quante frasi ha validato ogni studente. Questo aumenta il coinvolgimento (Gamification).
4.  **Filtri Strategici:** Permetti agli studenti di filtrare per "Lingua" o "Sorgente" (es. solo frasi dai competitor) direttamente dalla UI.

### C. Training Improvements
5.  **Multi-Term Training:** Invece di tradurre solo frasi semplici, potresti forzare il modello a imparare "Glossary Constraints", incoraggiandolo a usare i termini esatti del PDF.
6.  **Full Adapter Merging:** Crea uno script finale che fa il merging dei pesi LoRA nel modello base per rendere l'inferenza più veloce e portabile in produzione (senza caricare l'adapter separatamente).

### D. Deployment
7.  **Dockerization:** Crea un `Dockerfile` per la WebApp. In questo modo sarà semplicissimo lanciarla su qualsiasi server (o sul server della scuola) senza preoccuparsi di installare manualmente tutte le dipendenze.
8.  **Automated Cron:** Imposta uno script che ogni notte scarica nuovi termini dai siti competitor e li mette in "attesa" per la validazione degli studenti.

---
**💡 Cosa fare ora?**
Ti consiglio di far iniziare gli studenti sulla **WebApp**. Il numero di frasi validate ("Gold Standard") è il fattore più importante per superare le prestazioni del modello base di Facebook.
