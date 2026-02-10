import sys
from pathlib import Path

# Setup Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.pipelines.factory import ModelFactory

def main():
    ModelFactory().run_benchmark()

if __name__ == "__main__":
    main()