from __future__ import annotations

from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
MISCLASSIFICATION_DIR = OUTPUTS_DIR / "misclassification"
LOGS_DIR = OUTPUTS_DIR / "logs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

ALL_DIRS = [
    DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
    MISCLASSIFICATION_DIR,
    LOGS_DIR,
    PLOTS_DIR,
    CHECKPOINTS_DIR,
]


def ensure_directories() -> None:
    for folder in ALL_DIRS:
        folder.mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def local_path(*parts: str) -> Path:
    return BASE_DIR.joinpath(*parts)


def model_path(filename: str) -> Path:
    ensure_directories()
    return MODELS_DIR / filename


def checkpoint_path(filename: str) -> Path:
    ensure_directories()
    return CHECKPOINTS_DIR / filename


def output_path(filename: str) -> Path:
    ensure_directories()
    return OUTPUTS_DIR / filename


def report_path(filename: str) -> Path:
    ensure_directories()
    return REPORTS_DIR / filename


def misclassification_path(filename: str) -> Path:
    ensure_directories()
    return MISCLASSIFICATION_DIR / filename


def log_path(prefix: str = "run") -> Path:
    ensure_directories()
    return LOGS_DIR / f"{prefix}_{timestamp()}.txt"


ensure_directories()
