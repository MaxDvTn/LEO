#!/bin/bash
# Script per avviare il server Streamlit e Ngrok in una sessione tmux 'server'

SESSION="server"
TMUX_CMD="tmux"

# Chiedi conferma se la sessione esiste
if $TMUX_CMD has-session -t $SESSION 2>/dev/null; then
    echo "⚠️  La sessione '$SESSION' esiste già."
    echo "👉 Collegati con: tmux attach -t $SESSION"
    exit 0
fi

# Crea nuova sessione (detached)
$TMUX_CMD new-session -d -s $SESSION

# Pane 1: Streamlit App
# Assumiamo che conda sia inizializzato. Se no, potrebbe servire 'source ~/miniconda3/etc/profile.d/conda.sh'
# Ma in una nuova sessione tmux (login shell) dovrebbe funzionare.
$TMUX_CMD send-keys -t $SESSION:0.0 "source ~/miniconda3/etc/profile.d/conda.sh" C-m
$TMUX_CMD send-keys -t $SESSION:0.0 "conda activate LEO" C-m
$TMUX_CMD send-keys -t $SESSION:0.0 "streamlit run src/ui/app.py" C-m

# Pane 2: Ngrok
$TMUX_CMD split-window -h -t $SESSION:0
$TMUX_CMD send-keys -t $SESSION:0.1 "ngrok http --domain=concretely-dendroid-florinda.ngrok-free.dev 8501" C-m

echo "✅ Server avviato nella sessione tmux '$SESSION'."
echo "👉 L'app Streamlit è nel pannello di sinistra."
echo "👉 Il tunnel Ngrok è nel pannello di destra."
echo ""
echo "Per visualizzare il server, esegui:"
echo "  tmux attach -t $SESSION"
