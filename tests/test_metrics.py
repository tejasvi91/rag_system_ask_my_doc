import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.observability.metrics import print_dashboard

print_dashboard()