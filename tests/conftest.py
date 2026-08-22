import sys
from pathlib import Path

# Automatically add the vlm directory to sys.path so tests can import modules directly
vlm_dir = Path(__file__).parent.parent
sys.path.insert(0, str(vlm_dir))
