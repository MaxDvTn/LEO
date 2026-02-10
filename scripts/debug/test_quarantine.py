import sys
from unittest.mock import MagicMock
from pathlib import Path

# 1. Mock Streamlit parts that are called at top-level
sys.modules["streamlit"] = MagicMock()
import streamlit as st
st.error = lambda x: print(f"ST_ERROR: {x}")
st.warning = lambda x: print(f"ST_WARNING: {x}")
st.markdown = lambda x: print(f"ST_MARKDOWN: {x}")
st.set_page_config = MagicMock()
st.sidebar = MagicMock()
# Mock session_state to support attribute access
class MockSessionState(dict):
    def __getattr__(self, key): return self.get(key)
    def __setattr__(self, key, value): self[key] = value

st.session_state = MockSessionState()

# Mock columns to return iterable (handles int and list)
def mock_columns(spec):
    n = spec if isinstance(spec, int) else len(spec)
    return [MagicMock() for _ in range(n)]
st.columns = mock_columns

# 2. Add src/ui to path
PROJECT_ROOT = Path("/home/mbosetti/LEO")
sys.path.append(str(PROJECT_ROOT / "src" / "ui"))

# 3. Mock Google Login to avoid blocking
sys.modules["auth"] = MagicMock()
sys.modules["auth"].check_google_login = lambda: "test_user@school.it"

# 4. Import app
from app import load_data, DATA_SYNTHETIC_DIR

def run_test():
    print(f"Testing load_data from {DATA_SYNTHETIC_DIR}")
    print(f"Files in dir: {[f.name for f in DATA_SYNTHETIC_DIR.glob('*.csv')]}")
    
    # Run loader
    df = load_data()
    
    # Check
    quarantine_dir = DATA_SYNTHETIC_DIR / "quarantined"
    if quarantine_dir.exists():
        files = list(quarantine_dir.glob("*.csv"))
        print(f"Quarantined files found: {[f.name for f in files]}")
        if len(files) > 0:
            print("✅ TEST PASSED: File was quarantined.")
        else:
            print("❌ TEST FAILED: Quarantine dir exists but empty.")
    else:
        print("❌ TEST FAILED: Quarantine dir does not exist.")

if __name__ == "__main__":
    run_test()
