from pathlib import Path


ML_CORE_ROOT = Path(__file__).resolve().parents[1]

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

DEFAULT_PRICE_CLASSIFIER_PATH = ML_CORE_ROOT / "models/classify/v3/price_classification_v3_model.joblib"

LOW_CONFIDENCE_THRESHOLD = 0.55
