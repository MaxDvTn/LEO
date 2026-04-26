import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.leo import main as leo_main

def main():
    sys.argv = ["leo.py", "data", "generate"]
    leo_main()

if __name__ == "__main__":
    main()
