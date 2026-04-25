import sys
from pathlib import Path
import pandas as pd

# Setup Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.synthesis.glossary_data import ROVER_GLOSSARY

def main():
    print("📖 Exporting Glossary to CSV...")
    
    # Create DataFrame
    df = pd.DataFrame(ROVER_GLOSSARY)
    
    # Define output path
    output_path = PROJECT_ROOT / "data" / "gold" / "technical_terms.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Glossary exported to: {output_path}")
    print(f"   Total terms: {len(df)}")
    print(df.head())

if __name__ == "__main__":
    main()
