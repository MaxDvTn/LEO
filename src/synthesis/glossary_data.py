# src/synthesis/glossary_data.py

# Lista curata di termini tecnici Roverplastik per generare frasi di training
ROVER_GLOSSARY = [
    # --- CATEGORIA: PRODOTTI E COMPONENTI (HARDWARE) ---
    {"term": "cassonetto coibentato", "context": "structure"},
    {"term": "guarnizione termoacustica", "context": "sealing"},
    {"term": "monoblocco", "context": "window_frame"},
    {"term": "spalla coibentata", "context": "insulation"},
    {"term": "sottobancale", "context": "structure"},
    {"term": "avvolgibile in alluminio", "context": "shading"},
    {"term": "chiusura oscurante", "context": "shading"},
    {"term": "nodo primario", "context": "construction"},
    {"term": "nodo secondario", "context": "construction"},
    {"term": "controtelaio", "context": "installation"},
    {"term": "barriera al vapore", "context": "insulation"},
    {"term": "nastro autoespandente", "context": "sealing"},
    {"term": "vite di fissaggio", "context": "hardware"},

    # --- CATEGORIA: FISICA E PRESTAZIONI (ABSTRACT) ---
    {"term": "ponte termico", "context": "physics"},
    {"term": "trasmittanza termica", "context": "physics"},
    {"term": "abbattimento acustico", "context": "physics"},
    {"term": "efficienza energetica", "context": "ecology"},
    {"term": "tenuta all'aria", "context": "performance"},
    {"term": "tenuta all'acqua", "context": "performance"},
    {"term": "carico del vento", "context": "performance"},
    {"term": "formazione di muffa", "context": "problem"},
    {"term": "condensa superficiale", "context": "problem"},
    {"term": "isolamento a cappotto", "context": "construction"},

    # --- CATEGORIA: AZIONI E PROCESSI (VERBS/ACTIONS) ---
    {"term": "riqualificazione del foro finestra", "context": "renovation"},
    {"term": "posa in opera qualificata", "context": "installation"},
    {"term": "sigillatura del giunto", "context": "action"},
    {"term": "coibentazione", "context": "action"},
    {"term": "ristrutturazione non invasiva", "context": "renovation"},

    # --- CATEGORIA: TERMINI DA SCRAPING WEB (NUOVI) ---
    {"term": "finestre da tetto", "context": "product"},
    {"term": "tunnel solari", "context": "product"},
    {"term": "tapparelle e tende per lucernari", "context": "product"},
    {"term": "zanzariere", "context": "product"},
    {"term": "veneziane", "context": "shading"},
    {"term": "soluzioni per foro finestra", "context": "solution"},
    {"term": "deventer seals", "context": "product"},
    {"term": "monoblocco coibentato presystem", "context": "product"},
]


def get_terms_list(with_context: bool = False):
    """Return glossary terms. with_context=True returns list of dicts, False returns flat list of strings."""
    if with_context:
        return list(ROVER_GLOSSARY)
    return [item["term"] for item in ROVER_GLOSSARY]


def add_new_term(term: str, context: str = "custom_entry") -> bool:
    """
    Append a new term to ROVER_GLOSSARY in this source file, then update the in-memory list.
    Uses an atomic write (temp file + rename) to avoid corruption on crash.
    """
    import os
    import tempfile
    import shutil

    file_path = os.path.abspath(__file__)
    if file_path.endswith(".pyc"):
        raise RuntimeError("Cannot modify compiled .pyc file; run from source.")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find the closing ']' of ROVER_GLOSSARY specifically — track when the list starts.
    in_glossary = False
    insert_idx = -1
    for i, line in enumerate(lines):
        if "ROVER_GLOSSARY" in line and "[" in line:
            in_glossary = True
        if in_glossary and line.strip() == "]":
            insert_idx = i
            break

    if insert_idx == -1:
        return False

    new_entry = f'    {{"term": "{term}", "context": "{context}"}},\n'
    lines.insert(insert_idx, new_entry)

    # Atomic write: write to a temp file in the same directory, then rename.
    dir_path = os.path.dirname(file_path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".py.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.writelines(lines)
        shutil.move(tmp_path, file_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    ROVER_GLOSSARY.append({"term": term, "context": context})
    return True
