from pathlib import Path
from typing import Any

from .config import DEFAULT_PRICE_CLASSIFIER_PATH, PRICE_CLASS_BOUNDS_VND, PRICE_CLASS_LABELS
from .exceptions import ModelArtifactError
from .schemas import IntentRequest, PriceBandPrediction


def class_id_from_budget(min_vnd: float | None, max_vnd: float | None) -> int:
    if min_vnd is None and max_vnd is None:
        return 2
    if min_vnd is not None and max_vnd is not None:
        reference = (min_vnd + max_vnd) / 2
    else:
        reference = max_vnd if max_vnd is not None else min_vnd
    if reference is None:
        return 2
    if reference < 500_000:
        return 0
    if reference < 1_000_000:
        return 1
    if reference < 2_000_000:
        return 2
    if reference < 5_000_000:
        return 3
    return 4


class PriceBandService:
    def __init__(self, model_path: Path = DEFAULT_PRICE_CLASSIFIER_PATH):
        self.model_path = Path(model_path)
        self._artifact: dict[str, Any] | None = None

    def _load_artifact(self) -> dict[str, Any]:
        if self._artifact is not None:
            return self._artifact
        try:
            import joblib
        except ModuleNotFoundError as exc:
            raise ModelArtifactError("joblib is required to load the price classifier artifact.") from exc
        if not self.model_path.exists():
            raise ModelArtifactError(f"Price classifier artifact not found: {self.model_path}")
        self._artifact = joblib.load(self.model_path)
        return self._artifact

    def predict_price_band(self, intent: IntentRequest) -> PriceBandPrediction:
        class_id = class_id_from_budget(intent.budget_min_vnd, intent.budget_max_vnd)
        label = PRICE_CLASS_LABELS[class_id]
        low, high = PRICE_CLASS_BOUNDS_VND[class_id]
        probabilities = {name: 0.0 for name in PRICE_CLASS_LABELS.values()}
        probabilities[label] = 1.0 if intent.budget_min_vnd is not None or intent.budget_max_vnd is not None else 0.6

        return PriceBandPrediction(
            price_class_id=class_id,
            price_class_label=label,
            probabilities=probabilities,
            confidence=probabilities[label],
            price_min_vnd=intent.budget_min_vnd if intent.budget_min_vnd is not None else low,
            price_max_vnd=intent.budget_max_vnd if intent.budget_max_vnd is not None else high,
        )
