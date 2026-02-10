import sys
import argparse
from pathlib import Path

# Setup Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.pipelines.factory import ModelFactory

def main():
    parser = argparse.ArgumentParser(description="NLLB Inference")
    parser.add_argument("--src_lang", type=str, default="eng_Latn", help="Source Language Code")
    parser.add_argument("--tgt_lang", type=str, default="ita_Latn", help="Target Language Code")
    parser.add_argument("--text", type=str, required=True, help="Text to translate")
    parser.add_argument("--checkpoint", type=str, help="Path to .ckpt file (optional)")
    
    args = parser.parse_args()
    
    ModelFactory().translate(
        text=args.text,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        checkpoint_path=args.checkpoint
    )

if __name__ == "__main__":
    main()
