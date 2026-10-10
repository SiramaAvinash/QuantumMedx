"""QuantumMedx - quantum kernel classifier benchmarked on Processor A / B (QFF 2026, PS Q2)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSOR_DIR = ROOT / "processors"
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"
DATA_DIR = ROOT / "data"

for _d in (TABLE_DIR, FIG_DIR, DATA_DIR / "raw", DATA_DIR / "processed"):
    _d.mkdir(parents=True, exist_ok=True)
