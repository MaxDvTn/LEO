import streamlit as st
from streamlit_oauth import OAuth2Component
import os
import base64
import json

# --- CONFIGURAZIONE ---
# Carica le chiavi in modo sicuro dai secrets di Streamlit (.streamlit/secrets.toml)
try:
    CLIENT_ID = st.secrets["google_oauth"]["client_id"]
    CLIENT_SECRET = st.secrets["google_oauth"]["client_secret"]
except KeyError:
    st.error("⚠️ Credenziali Google non trovate. Verifica .streamlit/secrets.toml")
    st.stop()

# Configurazione standard Google
AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_TOKEN_URL = "https://oauth2.googleapis.com/revoke"

# LISTA DEI DOMINI AUTORIZZATI
# Se lasci la lista vuota [], tutti gli account Google saranno accettati
ALLOWED_DOMAINS = ["@liceodavincitn.it", "@roverplastik.it"] 
REDIRECT_URI = "https://concretely-dendroid-florinda.ngrok-free.dev"

def check_google_login():
    """Gestisce il login e ritorna l'email dell'utente se autenticato."""
    
    # Se siamo già loggati, restituiamo l'email salvata
    if "user_email" in st.session_state:
        return st.session_state["user_email"]

    st.header("🔒 Accesso Riservato Team LEO")
    st.write("Accedi con il tuo account istituzionale per contribuire.")

    # Crea componente OAuth
    oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_URL, TOKEN_URL, TOKEN_URL, REVOKE_TOKEN_URL)
    
    # Tenta di determinare l'URL di redirect automaticamente
    # Se usi ngrok, streamlit a volte si confonde. 
    # È meglio passare l'URL corrente copiato dalla barra degli indirizzi se da problemi.
    # Per ora proviamo automatico.
    
    result = oauth2.authorize_button(
        name="Continua con Google",
        icon="https://www.google.com.tw/favicon.ico",
        redirect_uri=REDIRECT_URI, # Tenta di indovinare l'URL corrente
        scope="openid email profile",
        key="google_auth",
        use_container_width=True,
    )

    if result:
        # Decodifica il token per leggere l'email
        try:
            id_token = result["token"]["id_token"]
            # Il payload è la seconda parte del token (separato da punti)
            payload = id_token.split(".")[1]
            # Padding per decodifica base64
            payload += "=" * (-len(payload) % 4)
            decoded = json.loads(base64.b64decode(payload).decode("utf-8"))
            
            email = decoded["email"]
            name = decoded.get("name", "Studente")

            # --- CONTROLLO DOMINIO ---
            if ALLOWED_DOMAINS:
                if not any(email.endswith(domain) for domain in ALLOWED_DOMAINS):
                    st.error(f"Accesso negato! Devi usare una mail di uno di questi domini: {', '.join(ALLOWED_DOMAINS)}")
                    st.stop()
            
            # Login successo
            st.session_state["user_email"] = email
            st.session_state["user_name"] = name
            st.toast(f"Benvenuto, {name}!", icon="👋")
            st.rerun()
            
        except Exception as e:
            st.error(f"Errore nel login: {e}")
            
    st.stop() # Ferma l'app finché non si logga.