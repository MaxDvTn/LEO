import sys
from pathlib import Path
import pandas as pd

# Setup Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.pipelines.factory import DataFactory

if __name__ == "__main__":
    print("🔄 Regenerating Test Set...")
    DataFactory().create_test_set()
