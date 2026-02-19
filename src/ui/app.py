import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path

# --- 1. CONFIGURAZIONE PAGINA (DEVE ESSERE LA PRIMA ISTRUZIONE ST) ---
st.set_page_config(page_title="LEO Validator", page_icon="images/LEO_logo05.svg", layout="wide") #🦁

# --- 2. SETUP PERCORSI E IMPORT AUTH ---
# Risaliamo alla root del progetto
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(Path(__file__).resolve().parent)) # Aggiunge src/ui al path

# Ora possiamo importare auth
from auth import check_google_login

# --- 3. LOGIN GOOGLE (OPZIONALE PER LOCAL DEV) ---
# Se l'ambiente ha LEO_SKIP_AUTH="1", bypassiamo il login per velocità
if os.getenv("LEO_SKIP_AUTH") == "1":
    user_email = "local_admin@leo.dev"
    st.sidebar.info("🚀 **Dev Mode**: Login bypassato.")
else:
    # Se l'utente non è loggato, lo script si ferma qui dentro (Redirect a Google)
    user_email = check_google_login()

# --- 4. SIDEBAR UTENTE ---
st.sidebar.success(f"👤 Validator:\n**{user_email}**")

# Tasto Logout (Utile per testare o cambiare utente)
if st.sidebar.button("Esci / Logout", type="primary"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

# --- 5. CONFIGURAZIONE DATI ---
DATA_SYNTHETIC_DIR = ROOT_DIR / "data" / "synthetic"
GOLD_FILE = ROOT_DIR / "data" / "gold" / "rover_gold_dataset.csv"

# --- MAPPATURA VISIVA ---
LANG_MAP = {
    "ita_Latn": "🇮🇹 Italiano",
    "eng_Latn": "🇬🇧 Inglese (Tecnico)",
    "fra_Latn": "🇫🇷 Francese (Tecnico)",
    "spa_Latn": "🇪🇸 Spagnolo (Tecnico)"
}

import shutil
import time

def load_data():
    """Carica e unisce tutti i file CSV nella cartella synthetic. Quarantena i file corrotti."""
    if not DATA_SYNTHETIC_DIR.exists():
        return None
    
    all_files = list(DATA_SYNTHETIC_DIR.glob("*.csv"))
    if not all_files:
        return None
    
    dfs = []
    quarantine_dir = DATA_SYNTHETIC_DIR / "quarantined"
    
    for f in all_files:
        try:
            # Proviamo a leggere
            dfs.append(pd.read_csv(f))
        except Exception as e:
            # GESTIONE ERRORI E QUARANTENA
            st.error(f"⚠️ File corrotto rilevato: {f.name}. Spostamento in quarantena...")
            
            quarantine_dir.mkdir(exist_ok=True)
            
            # Timestamp per evitare conflitti di nome
            timestamp = int(time.time())
            new_name = f"{f.stem}_{timestamp}{f.suffix}"
            dest_path = quarantine_dir / new_name
            
            try:
                shutil.move(str(f), str(dest_path))
                st.warning(f"File spostato in: {dest_path}")
                st.markdown(f"**Errore:** `{e}`")
            except Exception as move_err:
                st.error(f"Impossibile spostare il file: {move_err}")
            
    if not dfs:
        return None
        
    return pd.concat(dfs, ignore_index=True)

def save_gold_row(row_data):
    """Salva una riga validata nel file Gold Standard (Append mode)"""
    # AGGIUNTA FONDAMENTALE: Salviamo chi ha fatto la validazione!
    row_data["validator_id"] = user_email
    
    df_row = pd.DataFrame([row_data])
    
    # Se il file non esiste, scrive l'header, altrimenti appende solo i dati
    header = not GOLD_FILE.exists()
    
    # Assicuriamo che la cartella esista
    GOLD_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    df_row.to_csv(GOLD_FILE, mode='a', header=header, index=False)

# --- 6. INTERFACCIA PRINCIPALE ---
_, col_mid, _ = st.columns([1, 6, 1])
with col_mid:
    st.image("images/LEO_logo05.svg", width=480)
    # Riduciamo lo spazio usando un margine superiore negativo nel CSS del paragrafo
    st.markdown("<p style='text-align: center; margin-top: -100px;'><b>Roverplastik Neural Translation Project</b> | <i>Validation Interface</i></p>", unsafe_allow_html=True)

# Caricamento Dati
full_df = load_data()

if full_df is None:
    st.error(f"❌ Nessun file CSV trovato in: {DATA_SYNTHETIC_DIR}")
    st.info("Esegui prima lo script 'scripts/run_gen.py' per generare i dati!")
    st.stop()

# --- 6. SIDEBAR: NAVIGATION & STATS ---
st.sidebar.divider()
app_mode = st.sidebar.radio(
    "🚀 Scegli Attività:",
    ["Validazione AI", "Inserimento Manuale"],
    help="Validazione: correggi i dati dell'IA. Inserimento: aggiungi nuove frasi da zero."
)

# Sezione Filtri (Solo per Validazione)
if app_mode == "Validazione AI":
    st.sidebar.subheader("🎯 Filtro Lingua")
    filter_labels = ["🌍 Tutte le lingue"] + [LANG_MAP[k] for k in LANG_MAP.keys()]
    selected_lang_label = st.sidebar.selectbox(
        "Seleziona lingua su cui lavorare:",
        options=filter_labels,
        index=0
    )
    
    # Convertiamo la label in codice lingua
    selected_lang = "Tutte"
    if selected_lang_label != "🌍 Tutte le lingue":
        selected_lang = [k for k, v in LANG_MAP.items() if v == selected_lang_label][0]
    
    # Filtriamo il dataframe per la validazione
    if selected_lang == "Tutte":
        df = full_df
    else:
        df = full_df[full_df['target_lang'] == selected_lang].reset_index(drop=True)
else: # app_mode == "Inserimento Manuale"
    selected_lang = "Tutte" # Default per inserimento, non usato per filtrare il df in questo contesto
    # Per l'inserimento manuale, non abbiamo bisogno di filtrare il df in anticipo,
    # ma per evitare errori di variabile non definita, possiamo assegnare full_df o un df vuoto.
    # Tuttavia, la logica di inserimento manuale non usa 'df' per iterare, quindi questa riga è più per coerenza.
    df = full_df # O un df vuoto se non si vuole mostrare nulla della validazione

# --- 7. LOGICA INSERIMENTO MANUALE ---
if app_mode == "Inserimento Manuale":
    st.markdown("### ✍️ Contribuzione Manuale")
    st.info("Usa questo form per aggiungere nuove traduzioni tecniche verificate che non sono presenti nel database dell'IA.")
    
    with st.form("manual_entry_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            src_text = st.text_area("🇮🇹 Testo Originale (Italiano)", height=150, placeholder="Es: Guarnizione coestrusa in materiale termoplastico...")
        
        with col2:
            tgt_lang_label = st.selectbox("🎯 Lingua di Destinazione", options=list(LANG_MAP.values()))
            tgt_lang_code = [k for k, v in LANG_MAP.items() if v == tgt_lang_label][0]
            tgt_text = st.text_area(f"✨ Traduzione in {tgt_lang_label}", height=150, placeholder="Inserisci la traduzione corretta...")
        
        term_focus = st.text_input("🔑 Termine Chiave (Opzionale)", placeholder="Es: guarnizione")
        
        submitted = st.form_submit_button("SALVA NEL DATASET GOLD", type="primary", use_container_width=True)
        
        if submitted:
            if not src_text.strip() or not tgt_text.strip():
                st.error("Per favore, compila sia il testo originale che la traduzione!")
            else:
                manual_entry = {
                    "source_text": src_text.strip(),
                    "target_text": tgt_text.strip(),
                    "source_lang": "ita_Latn",
                    "target_lang": tgt_lang_code,
                    "original_source": "manual_contribution",
                    "original_ai_prediction": "N/A",
                    "quality_check": "manual_entry",
                    "validator_id": user_email,
                    "term_keyword": term_focus.strip() if term_focus else "manual"
                }
                save_gold_row(manual_entry)
                st.success("✅ Frase salvata con successo nel Gold Dataset!")
                st.balloons()
    st.stop() # Fine logica inserimento

# --- 8. GESTIONE STATO SESSIONE (Solo per Validazione) ---
if 'session_started' not in st.session_state:
    st.session_state.session_started = False

if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

if 'session_count' not in st.session_state:
    st.session_state.session_count = 0

# --- 7. SCHERMATA DI SETUP (INIZIALE) ---
if not st.session_state.session_started:
    st.markdown("<h1 style='text-align: center;'>🚀 Benvenuto in L.E.O.</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Configura la tua sessione di validazione prima di iniziare.</p>", unsafe_allow_html=True)
    
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.divider()
        # Carichiamo i dati per mostrare le statistiche
        full_df = load_data()
        if full_df is not None:
            st.subheader("📊 Stato del lavoro")
            stats_list = []
            for lang_code, label in LANG_MAP.items():
                count = len(full_df[full_df['target_lang'] == lang_code])
                stats_list.append({"Lingua": label, "Frasi da validare": count})
            st.table(pd.DataFrame(stats_list))
            
            st.divider()
            st.subheader("🎯 Scegli il tuo obiettivo")
            filter_labels = ["🌍 Tutte le lingue"] + [LANG_MAP[k] for k in LANG_MAP.keys()]
            selected_label = st.selectbox("Quale lingua vuoi validare oggi?", options=filter_labels)
            
            if st.button("INIZIA SESSIONE DI LAVORO", type="primary", use_container_width=True):
                # Salviamo la scelta
                if selected_label == "🌍 Tutte le lingue":
                    st.session_state.target_lang_filter = "Tutte"
                else:
                    st.session_state.target_lang_filter = [k for k, v in LANG_MAP.items() if v == selected_label][0]
                
                st.session_state.session_started = True
                st.rerun()
        else:
            st.error("Nessun dato trovato per iniziare.")
    st.stop()

# --- 8. AREA DI LAVORO (SESSIONE AVVIATA) ---
full_df = load_data()
selected_lang = st.session_state.target_lang_filter

# Filtriamo il dataframe
if selected_lang == "Tutte":
    df = full_df
else:
    df = full_df[full_df['target_lang'] == selected_lang].reset_index(drop=True)

# Gestione UI Sidebar (Compatta durante il lavoro)
st.sidebar.divider()
st.sidebar.info(f"📍 Sessione: **{selected_lang}**")
if st.sidebar.button("Cambia Lingua / Setup"):
    st.session_state.session_started = False
    st.session_state.current_index = 0
    st.rerun()

idx = st.session_state.current_index

if len(df) == 0:
    st.warning(f"Nessuna frase trovata per il filtro selezionato.")
    if st.button("Torna al Setup"):
        st.session_state.session_started = False
        st.rerun()
    st.stop()

if idx < len(df):
    row = df.iloc[idx]
    
    # Progress Bar
    progress = (idx / len(df))
    st.progress(progress)
    st.caption(f"Riga {idx + 1} di {len(df)} | Validate ora: {st.session_state.session_count}")

    st.divider()
    
    # Visualizza il termine chiave
    st.markdown(f"### 🔑 Termine Focus: `{row.get('term_keyword', 'N/A')}`")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🇮🇹 Italiano")
        # Text Area per modificare la sorgente (Richiesto dall'utente)
        source_text = st.text_area("Testo Originale (Modifica se necessario)", value=row['source_text'], height=150, key=f"src_{idx}")
        
    with col2:
        tgt_lang_label = LANG_MAP.get(row['target_lang'], row['target_lang'])
        st.markdown(f"#### {tgt_lang_label}")
        # Text Area per correggere la traduzione
        target_text = st.text_area("Traduzione AI (Correggi qui)", value=row['target_text'], height=150, key=f"tgt_{idx}")

    # --- PULSANTIERA ---
    st.divider()
    b_col1, b_col2, b_col3 = st.columns([1, 1, 4])
    
    with b_col1:
            if st.button("✅ SALVA E CONTINUA", type="primary", use_container_width=True):
                
                # 2. LOGICA DI CORREZIONE DOPPIA
                original_src = row['source_text'].strip() if isinstance(row['source_text'], str) else ""
                original_tgt = row['target_text'].strip() if isinstance(row['target_text'], str) else ""
                
                final_src = source_text.strip()
                final_tgt = target_text.strip()
                
                # Determiniamo lo stato
                src_changed = (final_src != original_src)
                tgt_changed = (final_tgt != original_tgt)
                
                if src_changed and tgt_changed:
                    status = "both_corrected"
                elif src_changed:
                    status = "source_corrected"
                elif tgt_changed:
                    status = "target_corrected"
                else:
                    status = "ai_approved"
                
                # Prepara il dato pulito
                validated_entry = {
                    "source_text": final_src,
                    "target_text": final_tgt,
                    "source_lang": row['source_lang'],
                    "target_lang": row['target_lang'],
                    "original_source": original_src,
                    "original_ai_prediction": original_tgt,
                    "quality_check": status,
                    "validator_id": user_email
                }
                
                save_gold_row(validated_entry)
                
                # Feedback
                if src_changed or tgt_changed:
                    st.toast(f"Modifiche salvate! ({status})", icon="📝")
                else:
                    st.toast("Approvato senza modifiche.", icon="✅")
                    
                st.session_state.current_index += 1
                st.session_state.session_count += 1
                st.rerun()

    with b_col2:
        if st.button("❌ SCARTA", type="secondary", use_container_width=True):
            st.toast("Riga scartata", icon="🗑️")
            st.session_state.current_index += 1
            st.rerun()
    
    with b_col3:
        st.info("💡 **Novità**: Ora puoi correggere anche il testo in **Italiano** se trovi errori nel manuale originale.")

else:
    # FINE DEL DATASET
    st.balloons()
    st.success("🎉 Dataset Completato! Ottimo lavoro.")
    st.markdown(f"Hai validato **{st.session_state.session_count}** frasi in questa sessione.")
    if st.button("Ricomincia da capo"):
        st.session_state.current_index = 0
        st.rerun()