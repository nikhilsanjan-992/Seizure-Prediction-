"""NeuroSelect Phase 1 setup verification script.

Checks:
- Python version
- required package imports
- project directories
- configuration paths
"""

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PACKAGES = {
    "MNE": "mne",
    "NumPy": "numpy",
    "SciPy": "scipy",
    "Pandas": "pandas",
    "Matplotlib": "matplotlib",
    "Scikit-learn": "sklearn",
    "TensorFlow": "tensorflow",
    "FastAPI": "fastapi",
    "Uvicorn": "uvicorn",
    "dotenv": "dotenv",
}

IMPORT_PACKAGE_MAP = {
    "Scikit-learn": "sklearn",
    "Uvicorn": "uvicorn",
    "dotenv": "dotenv",
}

DIRECTORIES = [
    "src",
    "backend",
    "frontend",
    "scripts",
    "data/raw",
    "models",
    "results",
    "results/plots",
    "results/metrics",
]


def check_python() -> bool:
    version = sys.version_info
    ok = version >= (3, 10)
    print(f"Python: {ok and 'OK' or 'FAILED'} ({sys.version.split()[0]})")
    return ok


def check_packages() -> bool:
    all_ok = True
    for label, module in PACKAGES.items():
        import_name = IMPORT_PACKAGE_MAP.get(label, module)
        try:
            importlib.import_module(import_name)
            print(f"{label}: OK")
        except Exception:
            all_ok = False
            print(f"{label}: FAILED")
    return all_ok


def check_directories() -> bool:
    all_ok = True
    for rel in DIRECTORIES:
        if (PROJECT_ROOT / rel).is_dir():
            print(f"Directory {rel}: OK")
        else:
            all_ok = False
            print(f"Directory {rel}: FAILED")
    return all_ok


def check_config() -> bool:
    try:
        import config  # noqa: F401

        for attr in ("DATA_DIR", "RAW_DATA_DIR", "MODEL_DIR",
                     "RESULTS_DIR", "PLOTS_DIR", "METRICS_DIR"):
            value = getattr(config, attr)
            if not isinstance(value, Path) or not value.name:
                print(f"Config {attr}: FAILED")
                return False
        print("Config paths: OK")
        return True
    except Exception:
        print("Config paths: FAILED")
        return False


def main() -> None:
    print("NeuroSelect Phase 1 Setup")
    print("-" * 25)
    results = [
        check_python(),
        check_packages(),
        check_directories(),
        check_config(),
    ]
    print("-" * 25)
    if all(results):
        print("PHASE 1 SETUP: SUCCESS")
    else:
        print("PHASE 1 SETUP: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()