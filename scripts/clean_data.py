import pandas as pd
from pathlib import Path
from langdetect import detect, DetectorFactory
from tqdm import tqdm
import os

# Ensure consistent results from langdetect
DetectorFactory.seed = 42

def clean_csv(file_path, output_path, quarantine_dir):
    if not file_path.exists():
        print(f"⚠️ File not found: {file_path}")
        return

    print(f"🔍 Cleaning {file_path.name}...")
    df = pd.read_csv(file_path)
    initial_count = len(df)
    
    cleaned_rows = []
    quarantined_rows = []
    
    for _, row in tqdm(df.iterrows(), total=initial_count, desc="Detecting language"):
        source_text = str(row['source_text'])
        source_lang = row.get('source_lang', 'ita_Latn')
        
        # Skip empty text
        if not source_text.strip():
            quarantined_rows.append(row)
            continue
            
        try:
            detected_lang = detect(source_text)
            
            # If expected lang is Italian, check if it's really Italian
            if source_lang == 'ita_Latn':
                if detected_lang == 'it':
                    cleaned_rows.append(row)
                else:
                    row_dict = row.to_dict()
                    row_dict['detected_lang'] = detected_lang
                    quarantined_rows.append(row_dict)
            else:
                # If it's not Italian, we just keep it for now as we only suspected IT tagging errors
                cleaned_rows.append(row)
                
        except Exception:
            # Language detection failed (likely too short or no alphabet)
            row_dict = row.to_dict()
            row_dict['detected_lang'] = 'unknown'
            quarantined_rows.append(row_dict)

    # Save cleaned data
    if cleaned_rows:
        pd.DataFrame(cleaned_rows).to_csv(output_path, index=False)
        
    # Save quarantined data
    if quarantined_rows:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        q_file = quarantine_dir / f"quarantined_{file_path.name}"
        pd.DataFrame(quarantined_rows).to_csv(q_file, index=False)
        print(f"📦 Quarantined {len(quarantined_rows)} rows to {q_file}")

    print(f"✅ Cleaned: {len(cleaned_rows)} rows saved to {output_path}")

def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
    QUARANTINE_DIR = DATA_DIR / "quarantined"
    
    # 1. Cleaning rover_pdf_augmented.csv
    pdf_file = DATA_DIR / "rover_pdf_augmented.csv"
    clean_csv(pdf_file, pdf_file, QUARANTINE_DIR)
    
    # 2. Cleaning rover_synthetic_multilingual.csv
    synth_file = DATA_DIR / "rover_synthetic_multilingual.csv"
    clean_csv(synth_file, synth_file, QUARANTINE_DIR)

if __name__ == "__main__":
    main()
