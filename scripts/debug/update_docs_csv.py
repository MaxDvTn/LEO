import pandas as pd
from pathlib import Path
from urllib.parse import urlparse
import sys

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.common.config import conf

NEW_URLS = [
    "https://profiles.semperitgroup.com/fileadmin/user_upload/MediaLibrary/Sealings/Products/RubberSheeting/PDFs/Holzfensterdichtungbroschuere_EN_web.pdf",
    "https://profiles.semperitgroup.com/fileadmin/user_upload/MediaLibrary/Sealings/Products/RubberSheeting/PDFs/1_SEMP_2012-5491-0087_DTB_2012_EN_01_Coex.pdf",
    "https://profiles.semperitgroup.com/fileadmin/user_upload/MediaLibrary/Sealings/Products/RubberSheeting/PDFs/3_SEMP_2012-5491-0087_DTB_2012_EN_03_Brands.pdf",
    "https://profiles.semperitgroup.com/fileadmin/user_upload/MediaLibrary/Sealings/PDF/Hybird_Ace_flyer_EN.pdf",
    "https://profiles.semperitgroup.com/fileadmin/user_upload/MediaLibrary/Sealings/Products/RubberSheeting/PDFs/EPDM-Flyer_new_2016_01.pdf",
    "https://site-1248247.mozfiles.com/files/1248247/Q-LON-3034_Timber-engl.pdf",
    "https://www.scribd.com/document/906914616/U0856000-Schlegel-Epdm-Qlon-Aluminium-Brochure-2019-en-Lr-2",
    "https://www.beck-heun.de/wp-content/uploads/ROKA-THERM-XP-EN-2023-03.pdf",
    "https://www.beck-heun.de/wp-content/uploads/AUFSATZKASTEN-EN-2023-07.pdf",
    "https://www.audagna.it/images/pdf/B+H_Cassonetti-Roka.pdf",
    "https://www.primo.com/media/mwldn3ib/gasket-catalogue-english-german.pdf",
    "https://www.primo.com/media/xvcfpgz5/window-and-door-profiles-brochure-english.pdf",
    "https://www.gfa-dichtungen.de/assets/pdf/portfolio.pdf",
    "https://www.heroal.de/website/PDF/technical-brochures/heroal-Rollladensysteme/Roller-shutter-systems-brochure.pdf",
    "https://www.heroal.de/website/PDF/technical-brochures/heroal_ready_3/heroal-Ready-Brochure.pdf",
    "https://www.heroal.de/website/PDF/partnerbroschueren/partnerbroschuere_heroal_rollladen.pdf",
    "https://www.heroal.de/website/PDF/brochure-tecniche/heroal_rolltorsysteme/heroal-roller-door-systems.pdf",
    "https://www.hella.info/media/9010/download/75000001_EN.pdf?v=4",
    "https://www.hella.info/media/9012/download/75480040_EN.pdf?v=2",
    "https://www.hella.info/media/1016/download/75900018_EN.pdf?v=3",
    "https://www.koemmerling.com/cms16/files/Koemmerling-88-centre-seal-main-brochure-2421130370-0825-web.pdf?download=1",
    "https://www.koemmerling.com/cms16/files/Koemmerling-76-double-seal-main-brochure-2421130150-0285-web.pdf?download=1",
    "https://www.koemmerling.com/cms16/files/Koemmerling-Sheets-and-building-profiles-brochure-2421170050-1025-web.pdf?download=1",
    "https://www.koemmerling.com/cms16/files/Koemmerling-KoemaCel-brochure-2421170053-1025-web.pdf?download=1",
    "https://www.koemmerling.com/cms16/files/Koemmerling-76-door-system-AluClip-Plus-brochure-2421130006-0825-web.pdf?download=1",
    "https://www.kjmgroup.co.uk/files/downloads/deceuninck-specification-guide-2018.pdf",
    "https://www.deceuninckna.com/wp-content/uploads/DEC23-03_279-Series-Brochure_Digital-1.pdf",
    "https://climate-eg.com/wp-content/uploads/2021/07/Deceuninck-new-catalogue-1.pdf",
    "https://www.rehau.com/downloads/893332/high-gloss-rauvolet-brilliant-line-roller-shutters.pdf",
    "https://www.hawkeyewindows.com/wp-content/uploads/2022/03/Hawkeye-Rolling-Shutters-Brochure-2022.pdf",
    "https://fenbro.com/wp-content/uploads/2022/12/Roller-shutters-EkoOkna-catalogue.pdf",
    "https://dovgilzaluzi.lv/catalogues/roller-shuterters.pdf",
    "https://www.heroal.de/website/PDF/technical-brochures/sonnenschutz_2/heroal-VS-Z-Textile-Assembly-box-Exte-Flyer.pdf"
]

def infer_producer(url):
    domain = urlparse(url).netloc.lower()
    if 'semperit' in domain: return 'Semperit'
    if 'beck-heun' in domain or 'audagna' in domain: return 'Beck+Heun' # Audagna hosts B+H
    if 'primo' in domain: return 'Primo'
    if 'heroal' in domain: return 'Heroal'
    if 'hella' in domain: return 'Hella'
    if 'koemmerling' in domain: return 'Koemmerling'
    if 'deceuninck' in domain or 'climate-eg' in domain or 'kjmgroup' in domain: return 'Deceuninck'
    if 'rehau' in domain: return 'Rehau'
    if 'hawkeye' in domain: return 'Hawkeye'
    if 'fenbro' in domain or 'ekookna' in url: return 'EkoOkna'
    if 'scribd' in domain or 'site-1248247' in domain: return 'Schlegel' # Inferred from context
    if 'gfa-dichtungen' in domain: return 'GFA'
    return 'Unknown'

def infer_country(url):
    domain = urlparse(url).netloc.lower()
    if domain.endswith('.de'): return 'DE'
    if domain.endswith('.it'): return 'IT'
    if domain.endswith('.at'): return 'AT'
    if domain.endswith('.lv'): return 'LV'
    return 'INT'

def main():
    metadata_path = conf.paths.data_raw / "extra_docs_metadata.csv"
    
    if metadata_path.exists():
        df = pd.read_csv(metadata_path)
    else:
        df = pd.DataFrame(columns=['Produttore', 'Paese', 'Categoria', 'Titolo Documento', 'Link Download'])
    
    existing_links = set(df['Link Download'].astype(str))
    
    new_rows = []
    for url in NEW_URLS:
        url = url.strip()
        if not url: continue
        
        # Simple duplicate check
        if url in existing_links:
            continue
            
        # Infer metadata
        prod = infer_producer(url)
        country = infer_country(url)
        
        # Infer title from filename
        path = urlparse(url).path
        filename = Path(path).name
        # Cleanup filename for title
        title = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ').title()
        
        new_rows.append({
            'Produttore': prod,
            'Paese': country,
            'Categoria': 'Manuali Tecnici', # Generic default
            'Titolo Documento': title,
            'Link Download': url
        })
        
        existing_links.add(url) # Prevent dupes within the new list
        
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # Append
        combined_df = pd.concat([df, new_df], ignore_index=True)
        combined_df.to_csv(metadata_path, index=False)
        print(f"✅ Added {len(new_rows)} new documents to {metadata_path}")
    else:
        print("🎉 No new documents to add (all duplicates).")

if __name__ == "__main__":
    main()
