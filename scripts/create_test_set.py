import sys
from pathlib import Path

# Setup Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.pipelines.factory import DataFactory

def main():
    DataFactory().create_test_set()

if __name__ == "__main__":
    main()