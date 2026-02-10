import pandas as pd
import requests
from pathlib import Path
import sys
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.common.config import conf

def download_files():
    metadata_path = conf.paths.data_raw / "extra_docs_metadata.csv"
    if not metadata_path.exists():
        print(f"❌ Metadata file not found: {metadata_path}")
        return

    df = pd.read_csv(metadata_path)
    output_dir = conf.paths.data_raw_pdfs
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Downloading {len(df)} documents to {output_dir}...")
    
    success_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        url = row['Link Download']
        if not isinstance(url, str) or not url.startswith("http"):
            print(f"⚠️ Invalid URL for {row.get('Titolo Documento', 'Unknown')}: {url}")
            continue
            
        # Create a safe filename
        filename = Path(url).name
        # If filename is too generic or weird, maybe use Title
        if not filename.endswith(".pdf"):
            filename = f"{row['Titolo Documento'].replace(' ', '_')}.pdf"
            
        dest_path = output_dir / filename
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                f.write(response.content)
            success_count += 1
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")

    print(f"✅ Downloaded {success_count}/{len(df)} files.")

if __name__ == "__main__":
    download_files()
