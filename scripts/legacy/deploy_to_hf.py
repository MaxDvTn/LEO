import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.hf import main as hf_main

def main():
    sys.argv = ["hf.py", "export"]
    hf_main()
    sys.argv = ["hf.py", "upload-model"]
    hf_main()
    sys.argv = ["hf.py", "deploy-space", "--restart"]
    hf_main()


if __name__ == "__main__":
    main()
