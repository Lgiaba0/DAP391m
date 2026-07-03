from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any

from .config import PRICE_CLASS_BOUNDS_VND, PRICE_CLASS_LABELS, get_default_price_classifier_path
from .exceptions import ModelArtifactError
from .schemas import IntentRequest, PriceBandPrediction


REGION_BY_CITY = {
    "da nang": "central",
    "nha trang": "central",
    "hoi an": "central",
    "ha noi": "north",
    "ho chi minh": "south",
    "ho chi minh city": "south",
    "sai gon": "south",
    "phu quoc": "south",
    "vung tau": "south",
    "ha long": "north",
    "sapa": "north",
}


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
    def __init__(self, model_path: Path | str | None = None):
        self.model_path = Path(model_path) if model_path is not None else get_default_price_classifier_path()
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
        self._register_legacy_v3_symbols()
        artifact = joblib.load(self.model_path)
        if not isinstance(artifact, dict):
            raise ModelArtifactError("Price classifier artifact must be a dict payload.")
        self._patch_legacy_sklearn_artifact(artifact)
        self._artifact = artifact
        return artifact

    def _register_legacy_v3_symbols(self) -> None:
        """Expose legacy training symbols for artifacts pickled from `__main__`."""
        try:
            from ML_core.scripts.classify.v3 import train_price_classification_v3 as v3
        except Exception:
            return

        main_module = sys.modules.get("__main__")
        if main_module is None:
            return

        for name in (
            "HierarchicalPriceClassifier",
            "make_lgbm_binary_or_multiclass",
            "to_stage_class",
            "compute_stage_weights",
            "compute_specialist_weights",
        ):
            if hasattr(v3, name):
                setattr(main_module, name, getattr(v3, name))

    def _patch_legacy_sklearn_artifact(self, artifact: Mapping[str, Any]) -> None:
        seen: set[int] = set()

        def visit(value: Any) -> None:
            value_id = id(value)
            if value_id in seen:
                return
            seen.add(value_id)

            if value.__class__.__name__ == "SimpleImputer" and not hasattr(value, "_fill_dtype"):
                setattr(value, "_fill_dtype", getattr(value, "_fit_dtype", object))

            if isinstance(value, Mapping):
                for nested in value.values():
                    visit(nested)
                return

            if isinstance(value, (list, tuple, set, frozenset)):
                for nested in value:
                    visit(nested)
                return

            nested_values = getattr(value, "__dict__", None)
            if isinstance(nested_values, dict):
                for nested in nested_values.values():
                    visit(nested)

        visit(artifact)

    def get_readiness_issues(self) -> list[str]:
        issues: list[str] = []
        try:
            import joblib  # noqa: F401
        except ModuleNotFoundError:
            issues.append("joblib is not installed.")
            return issues

        if not self.model_path.exists():
            issues.append(f"Price classifier artifact not found: {self.model_path}")
        return issues

    def predict_price_band(self, intent: IntentRequest) -> PriceBandPrediction:
        artifact = self._load_artifact()
        model = artifact.get("model")
        if model is None or not hasattr(model, "predict_proba"):
            raise ModelArtifactError("Price classifier artifact is missing a usable `model.predict_proba()` implementation.")

        frame = self._build_inference_frame(intent, artifact)
        raw_probabilities = model.predict_proba(frame)
        if len(raw_probabilities) != 1:
            raise ModelArtifactError("Price classifier returned an unexpected batch size.")
        probability_row = list(raw_probabilities[0])

        label_mapping = self._label_mapping(artifact)
        class_order = self._class_order(model, artifact, len(probability_row))
        probabilities = {
            label_mapping[class_id]: float(probability_row[index])
            for index, class_id in enumerate(class_order)
        }

        top_label = max(probabilities, key=probabilities.get)
        class_id = next(key for key, value in label_mapping.items() if value == top_label)
        label = label_mapping[class_id]
        low, high = PRICE_CLASS_BOUNDS_VND[class_id]
        return PriceBandPrediction(
            price_class_id=class_id,
            price_class_label=label,
            probabilities=probabilities,
            confidence=probabilities[label],
            price_min_vnd=intent.budget_min_vnd if intent.budget_min_vnd is not None else low,
            price_max_vnd=intent.budget_max_vnd if intent.budget_max_vnd is not None else high,
        )

    def _build_inference_frame(self, intent: IntentRequest, artifact: Mapping[str, Any]):
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise ModelArtifactError("pandas is required to build inference inputs for the price classifier.") from exc

        feature_cols = list(artifact.get("feature_cols") or [])
        if not feature_cols:
            raise ModelArtifactError("Price classifier artifact is missing `feature_cols`.")

        base_row = self._base_feature_values(intent)
        frame = pd.DataFrame([base_row])
        market_context_encoder = artifact.get("market_context_encoder")
        if market_context_encoder is not None and hasattr(market_context_encoder, "transform"):
            context_frame = market_context_encoder.transform(frame)
            for column in context_frame.columns:
                frame[column] = context_frame.iloc[0][column]

        feature_config = artifact.get("feature_config") or {}
        numeric_features = set(feature_config.get("numeric_features") or [])
        categorical_features = set(feature_config.get("categorical_features") or [])
        text_features = set(feature_config.get("text_features") or [])

        row: dict[str, Any] = {}
        for column in feature_cols:
            if column in frame.columns:
                row[column] = frame.iloc[0][column]
                continue
            row[column] = self._fallback_feature_value(column, numeric_features, categorical_features, text_features)
        return pd.DataFrame([row], columns=feature_cols)

    def _base_feature_values(self, intent: IntentRequest) -> dict[str, Any]:
        destination = intent.destination or "Unknown"
        property_type = intent.property_types[0] if intent.property_types else "hotel"
        guest_count = float(intent.guest_count or 2)
        room_count = float(intent.room_count or 1)
        amenities = set(intent.amenities)

        return {
            "city": destination,
            "source_city": destination,
            "district": "Unknown",
            "region": self._infer_region(destination),
            "property_type": property_type,
            "room_type_extracted": "standard",
            "bed_type": "Unknown",
            "local_currency": "VND",
            "score_tier": "unknown",
            "room_name": f"{property_type} room for {int(guest_count)} guests",
            "description": intent.raw_query or "",
            "star_rating_clean": 3.0,
            "room_quality_tier": 2.0,
            "luxury_tail_score": 0.0,
            "luxury_route_score": 0.0,
            "luxury_amenity_score": float(len(amenities)),
            "amenities_count": float(len(amenities)),
            "max_persons": guest_count,
            "bedroom_count": room_count,
            "amenity_outdoor_pool": float("pool" in amenities),
            "amenity_indoor_pool": 0.0,
            "amenity_restaurant": float("breakfast" in amenities),
            "amenity_fitness_center": 0.0,
            "amenity_spa": 0.0,
        }

    def _fallback_feature_value(
        self,
        column: str,
        numeric_features: set[str],
        categorical_features: set[str],
        text_features: set[str],
    ) -> Any:
        if column in numeric_features or column.endswith("_train"):
            return 0.0
        if column in text_features:
            return ""
        if column in categorical_features:
            return "Unknown"
        return 0.0 if self._looks_numeric(column) else "Unknown"

    def _infer_region(self, destination: str) -> str:
        return REGION_BY_CITY.get(destination.strip().lower(), "unknown")

    def _label_mapping(self, artifact: Mapping[str, Any]) -> dict[int, str]:
        raw_labels = artifact.get("price_class_labels") or PRICE_CLASS_LABELS
        return {int(class_id): str(label) for class_id, label in raw_labels.items()}

    def _class_order(self, model: Any, artifact: Mapping[str, Any], proba_width: int) -> list[int]:
        if hasattr(model, "classes_"):
            class_order = [int(class_id) for class_id in model.classes_]
        else:
            class_order = sorted(self._label_mapping(artifact))
        if len(class_order) != proba_width:
            raise ModelArtifactError("Price classifier probability output does not match the configured class order.")
        return class_order

    def _looks_numeric(self, column: str) -> bool:
        numeric_hints = (
            "count",
            "score",
            "rating",
            "price",
            "persons",
            "bedroom",
            "tail",
            "route",
            "class",
            "_train",
        )
        return any(hint in column for hint in numeric_hints)
