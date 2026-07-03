import os
from pathlib import Path


ML_CORE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_CORE_ROOT.parent

PRICE_CLASS_LABELS = {
    0: "budget",
    1: "economy",
    2: "mid_range",
    3: "upscale",
    4: "premium_luxury",
}

PRICE_CLASS_BOUNDS_VND = {
    0: (None, 500_000),
    1: (500_000, 1_000_000),
    2: (1_000_000, 2_000_000),
    3: (2_000_000, 5_000_000),
    4: (5_000_000, None),
}


def _resolve_price_classifier_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_default_price_classifier_path() -> Path:
    configured = os.environ.get("PRICE_CLASSIFIER_PATH")
    if configured:
        return _resolve_price_classifier_path(configured)

    repo_root_model = PROJECT_ROOT / "price_classification_v3_model.joblib"
    if repo_root_model.exists():
        return repo_root_model

    return ML_CORE_ROOT / "models/classify/v3/price_classification_v3_model.joblib"


LOW_CONFIDENCE_THRESHOLD = 0.55
