# Architettura della Generazione del Dataset Sintetico in LEO

## 1. Obiettivo della Pipeline

La pipeline di generazione del dataset sintetico di LEO ha lo scopo di creare dati paralleli per il fine-tuning di modelli di traduzione neurale nel dominio tecnico dei serramenti, dei cassonetti, delle guarnizioni, dei profili e dei materiali per edilizia.

Il sistema non dipende da un singolo modello generativo. Al contrario, può interrogare diversi modelli locali o cloud, salvare separatamente i dataset prodotti da ciascuno, confrontarli, filtrarli e costruire un ensemble finale da usare nel training.

L'idea principale è:

1. generare dati sintetici con modelli diversi;
2. conservare ogni generazione in modo versionato;
3. confrontare qualità, copertura e diversità;
4. costruire un dataset ensemble controllato;
5. usare solo il dataset controllato nel training finale.

## 2. Componenti Principali

La generazione è organizzata in tre livelli:

| Livello | Componente | Responsabilità |
|---|---|---|
| Orchestrazione | `scripts/data/run_generation_suite.py` | Esegue più modelli in sequenza, salva manifest, confronti, valutazioni e benchmark |
| Pipeline dati | `src/pipelines/factory.py` | Implementa glossary generation, web spidering, PDF mining, salvataggio e ensemble |
| Backend LLM | `src/synthesis/` | Seleziona e invoca Ollama, Gemini, OpenAI o altri provider LiteLLM |

Il punto di ingresso più completo è:

```bash
python scripts/data/run_generation_suite.py
```

Il punto di ingresso singolo per il modello configurato in `conf.gen.model_id` è:

```bash
python scripts/leo.py data generate
```

## 3. Selezione del Modello Generativo

Il modello usato per generare dati sintetici è definito da:

```python
conf.gen.model_id
```

La selezione del backend avviene tramite prefisso del model id.

| Prefisso | Backend | Esempio |
|---|---|---|
| `ollama/` | Ollama locale | `ollama/qwen2.5:32b` |
| `google/` | Gemini via LiteLLM | `google/gemini-2.5-flash` |
| `openai/` | OpenAI via LiteLLM | `openai/gpt-5-mini` |
| `anthropic/` | Anthropic via LiteLLM | `anthropic/claude-sonnet-4` |
| `deepseek/` | DeepSeek via LiteLLM | `deepseek/deepseek-chat` |

Il dispatch è implementato in:

```text
src/synthesis/generator.py
```

I backend concreti sono:

```text
src/synthesis/ollama_generator.py
src/synthesis/cloud_generator.py
```

Entrambi ereditano da:

```text
src/synthesis/base.py
```

La classe base contiene la logica comune:

- costruzione del prompt;
- retry in caso di errore;
- parsing dell'output;
- parallelizzazione;
- progress bar;
- normalizzazione del risultato.

I backend specifici implementano principalmente il metodo `_chat()`, cioè il modo concreto in cui viene inviata la richiesta al modello.

## 4. Parallelizzazione

La generazione usa `ThreadPoolExecutor` per inviare più richieste in parallelo.

Per Ollama, il numero di worker viene adattato in modo conservativo alla dimensione del modello:

| Categoria modello | Esempi | Worker |
|---|---|---|
| Modelli 20B+ | `qwen2.5:32b`, `gemma3:27b`, `mistral-small3.2` | 2 |
| Modelli 12B-14B | `mistral-nemo`, `phi4` | 3 |
| Modelli piccoli o non classificati | `aya-expanse:8b` | `conf.gen.num_workers` |
| Modelli cloud | Gemini, OpenAI, Anthropic | `conf.gen.cloud_num_workers` |

Per sfruttare il parallelismo lato Ollama, anche il servizio deve essere configurato con:

```ini
[Service]
Environment="OLLAMA_NUM_PARALLEL=4"
```

Dopo la modifica:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
systemctl show ollama --property=Environment
```

## 5. Tipi di Dataset Generabili

La suite supporta tre modalità principali:

```text
glossary
web
pdf
```

Queste modalità corrispondono a tre sorgenti diverse.

### 5.1 Glossary Dataset

Il dataset `glossary` parte da una lista curata di termini tecnici.

La sorgente è:

```text
src/synthesis/glossary_data.py
```

Per ogni termine, il modello generativo produce frasi tecniche e le relative traduzioni verso:

```text
eng_Latn
fra_Latn
spa_Latn
```

Il risultato viene normalizzato in formato long, cioè una riga per ogni coppia sorgente-target.

Schema tipico:

```text
source_text,target_text,source_lang,target_lang,origin,model_id,prompt_version,created_at,term,context
```

Questa è la modalità più controllata e più adatta per confrontare modelli generativi diversi.

### 5.2 Web Dataset

Il dataset `web` parte da siti competitor o da siti tecnici configurati in:

```text
SpiderConfig.target_urls
```

La pipeline:

1. scarica le pagine;
2. estrae candidati terminologici;
3. filtra rumore non tecnico, per esempio cookie, privacy, contatti;
4. invia i termini rimasti al generatore;
5. normalizza l'output nello stesso formato long del glossary dataset.

Il web dataset serve ad ampliare la copertura lessicale con termini reali usati da competitor o produttori del settore.

### 5.3 PDF Dataset

Il dataset `pdf` è la modalità più pesante.

La sorgente sono i PDF presenti in:

```text
data/raw/pdfs/
```

La pipeline:

1. estrae testo dai PDF;
2. segmenta il testo in frasi;
3. rimuove duplicati esatti;
4. esclude eventuali frasi già validate nel gold set;
5. rileva grossolanamente la lingua della frase;
6. traduce le frasi in base alla lingua rilevata.

Le frasi italiane o non riconosciute vengono tradotte verso:

```text
eng_Latn
fra_Latn
spa_Latn
```

Per queste frasi vengono anche creati reverse pairs, cioè coppie di ritorno verso l'italiano:

```text
eng_Latn -> ita_Latn
fra_Latn -> ita_Latn
spa_Latn -> ita_Latn
```

Le frasi native già in inglese, francese o spagnolo vengono invece tradotte direttamente verso:

```text
ita_Latn
```

Questa modalità può generare decine di migliaia di richieste LLM. Per questo deve essere eseguita separatamente e monitorata con attenzione.

## 6. Salvataggio dei Dataset

Ogni dataset generato viene salvato in modo riproducibile.

### 6.1 File Canonico

Il file canonico è quello più recente per una certa famiglia di dataset.

Esempi:

```text
data/synthetic/glossary_synthetic.csv
data/synthetic/competitor_synthetic.csv
data/synthetic/rover_pdf_augmented.csv
```

Questo file rappresenta l'ultimo output attivo.

### 6.2 Archivio

Quando un file canonico viene sovrascritto, la versione precedente viene spostata in:

```text
data/synthetic/archive/
```

Questo evita di perdere generazioni precedenti.

### 6.3 Run Versionati

Ogni generazione viene anche salvata con nome contenente modello e timestamp:

```text
data/synthetic/runs/
```

Esempio:

```text
glossary_synthetic__ollama_qwen2.5_32b__20260502T082824071296Z.csv
```

Questi file sono quelli usati per confrontare modelli diversi e costruire l'ensemble.

### 6.4 Checkpoint PDF

Il PDF mining supporta checkpoint intermedi in:

```text
data/synthetic/checkpoints/
```

Esempio:

```text
data/synthetic/checkpoints/rover_pdf_augmented__ollama_mistral-small3.2.csv
```

Il checkpoint serve a non perdere tutto il lavoro in caso di interruzione di una generazione lunga.

Alla ripartenza, la pipeline:

1. legge il checkpoint;
2. ricostruisce le frasi già completate;
3. rimuove dal lavoro residuo le frasi già presenti;
4. continua solo sulle frasi mancanti;
5. usa il checkpoint come fonte unica per il salvataggio finale.

## 7. Suite Multi-Modello

Per generare dataset con più modelli si usa:

```bash
python scripts/data/run_generation_suite.py \
  --models ollama/mistral-small3.2 ollama/gemma3:27b ollama/qwen2.5:32b google/gemini-2.5-flash ollama/aya-expanse:8b \
  --dataset-kind glossary \
  --benchmark-sample-size 20 \
  --skip-ppl
```

Per ogni modello, la suite:

1. imposta `conf.gen.model_id`;
2. istanzia il backend corretto;
3. genera il dataset richiesto;
4. salva il file canonico;
5. salva il file versionato;
6. registra il risultato nel manifest della suite.

La suite scrive i propri artefatti in:

```text
runs/generation_suite/<timestamp>/
```

Il manifest contiene l'elenco dei modelli eseguiti, i file generati, i confronti e gli eventuali report.

## 8. Valutazione e Confronto dei Dataset

Dopo la generazione, i dataset possono essere confrontati a coppie.

Il comando è:

```bash
python scripts/data/compare_synthetic.py \
  data/synthetic/runs/DATASET_A.csv \
  data/synthetic/runs/DATASET_B.csv
```

Il confronto misura:

- numero di righe;
- distribuzione per lingua;
- distribuzione per origine;
- overlap tra sorgenti;
- campioni aggiunti e rimossi;
- differenze qualitative visibili su esempi reali.

La suite multi-modello può generare automaticamente tutti i confronti pairwise.

## 9. Costruzione dell'Ensemble

Dopo aver generato più dataset versionati, si costruisce il dataset ensemble:

```bash
python scripts/leo.py data build-ensemble
```

L'ensemble builder legge i CSV da:

```text
data/synthetic/runs/
```

e applica:

1. rimozione dei duplicati esatti;
2. rimozione dei near-duplicates;
3. filtro qualità;
4. priorità tra modelli.

Il risultato finale è:

```text
data/synthetic/ensemble_training_set.csv
```

## 10. Priorità tra Modelli

Quando due modelli generano frasi quasi identiche, il sistema tiene la versione del modello con priorità più alta.

La priorità attuale privilegia:

1. `ollama/mistral-small3.2`
2. `ollama/gemma3:27b`
3. `ollama/qwen2.5:32b`
4. `google/gemini-2.5-flash`
5. `ollama/aya-expanse:8b`
6. `ollama/mistral-nemo`
7. `ollama/phi4`

Questa priorità non significa che il modello sia sempre migliore in assoluto. Serve solo a decidere quale riga conservare quando più generatori producono contenuti molto simili.

## 11. Uso dell'Ensemble nel Training

Quando esiste:

```text
data/synthetic/ensemble_training_set.csv
```

il dataloader non carica indiscriminatamente tutti i CSV sintetici top-level.

Usa invece un pool controllato:

```text
data/synthetic/rover_synthetic_multilingual.csv
data/synthetic/competitor_synthetic.csv
data/synthetic/ensemble_training_set.csv
```

Questa scelta evita di allenare il modello su:

- vecchi file `glossary_synthetic.csv`;
- output intermedi non controllati;
- duplicati generati da modelli diversi;
- dataset sperimentali non validati.

## 12. Controlli di Qualità Operativi

Durante o dopo la generazione, è utile verificare che i file siano stati creati:

```bash
ls -lh data/synthetic/runs | tail -20
```

Per controllare righe, lingue e modelli:

```bash
python -c "import glob, pandas as pd
for p in sorted(glob.glob('data/synthetic/runs/*.csv')):
    df = pd.read_csv(p)
    print('\n', p)
    print('rows:', len(df))
    print('langs:', df['target_lang'].value_counts().to_dict() if 'target_lang' in df else {})
    print('models:', df['model_id'].value_counts().to_dict() if 'model_id' in df else {})"
```

Per controllare righe vuote o malformate:

```bash
python -c "import glob, pandas as pd
for p in sorted(glob.glob('data/synthetic/runs/*.csv')):
    df = pd.read_csv(p)
    required = ['source_text', 'target_text', 'source_lang', 'target_lang']
    bad = df[df[required].isna().any(axis=1)]
    empty = df[(df['source_text'].astype(str).str.strip() == '') | (df['target_text'].astype(str).str.strip() == '')]
    print(p, 'rows=', len(df), 'bad_na=', len(bad), 'empty=', len(empty))"
```

Per PDF mining, è utile controllare anche i checkpoint:

```bash
ls -lh data/synthetic/checkpoints/
```

## 13. Flusso Raccomandato

Il flusso raccomandato è:

1. generare dataset `glossary` con più modelli;
2. confrontare i dataset generati;
3. costruire l'ensemble;
4. fare training NLLB e Seamless sul dataset controllato;
5. confrontare i modelli fine-tuned su un test set umano;
6. eseguire PDF mining solo come fase separata, con checkpoint attivi;
7. validare manualmente campioni critici prima di includere grandi quantità di dati PDF nel training finale.

In questo modo la generazione sintetica resta tracciabile, riproducibile e controllabile, anche quando vengono usati modelli generativi diversi.
