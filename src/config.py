"""NeuroSelect project configuration.

Centralized project paths and basic configuration placeholders.
Phase 1 only: no EEG processing parameters yet.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"

RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"

BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

REQUIRED_DIRS = [
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    RESULTS_DIR,
    PLOTS_DIR,
    METRICS_DIR,
]

# create required directories if they do not exist
for _dir in REQUIRED_DIRS:
    _dir.mkdir(parents=True, exist_ok=True)