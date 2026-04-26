import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.leo import main as leo_main

def main():
    parser = argparse.ArgumentParser(description="LEO inference compatibility wrapper")
    parser.add_argument("--src_lang", type=str, default="eng_Latn", help="Source Language Code")
    parser.add_argument("--tgt_lang", type=str, default="ita_Latn", help="Target Language Code")
    parser.add_argument("--text", type=str, required=True, help="Text to translate")
    parser.add_argument("--checkpoint", type=str, help="Path to .ckpt file (optional)")
    
    args = parser.parse_args()
    sys.argv = [
        "leo.py",
        "infer",
        "--src-lang",
        args.src_lang,
        "--tgt-lang",
        args.tgt_lang,
        "--text",
        args.text,
    ]
    if args.checkpoint:
        sys.argv.extend(["--checkpoint", args.checkpoint])
    leo_main()

if __name__ == "__main__":
    main()
