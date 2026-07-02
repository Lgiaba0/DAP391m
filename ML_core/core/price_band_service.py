from pathlib import Path
import sys
from typing import Any
import warnings

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

    def _register_v3_pickle_symbols(self) -> None:
        script_path = self.model_path.parents[3] / "scripts/classify/v3/train_price_classification_v3.py"
        if not script_path.exists():
            return

        import importlib.util

        module_name = "price_classification_v3_runtime"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            if spec is None or spec.loader is None:
                return
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

        main_module = sys.modules["__main__"]
        for name in (
            "HierarchicalPriceClassifier",
            "make_lgbm_binary_or_multiclass",
            "to_stage_class",
            "compute_stage_weights",
            "compute_specialist_weights",
        ):
            if hasattr(module, name):
                setattr(main_module, name, getattr(module, name))

    def _load_artifact(self) -> dict[str, Any]:
        if self._artifact is not None:
            return self._artifact
        try:
            import joblib
        except ModuleNotFoundError as exc:
            raise ModelArtifactError("joblib is required to load the price classifier artifact.") from exc
        if not self.model_path.exists():
            raise ModelArtifactError(f"Price classifier artifact not found: {self.model_path}")
        self._register_v3_pickle_symbols()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Trying to unpickle estimator .*")
            self._artifact = joblib.load(self.model_path)
        self._patch_sklearn_compatibility(self._artifact)
        return self._artifact

    def _patch_sklearn_compatibility(self, value: Any, seen: set[int] | None = None) -> None:
        if seen is None:
            seen = set()
        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)

        if value.__class__.__name__ == "SimpleImputer" and not hasattr(value, "_fill_dtype"):
            value._fill_dtype = getattr(value, "_fit_dtype", None)

        if isinstance(value, dict):
            for item in value.values():
                self._patch_sklearn_compatibility(item, seen)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                self._patch_sklearn_compatibility(item, seen)
            return
        if hasattr(value, "steps"):
            for _, step in value.steps:
                self._patch_sklearn_compatibility(step, seen)
        if hasattr(value, "transformers"):
            for _, transformer, _ in value.transformers:
                self._patch_sklearn_compatibility(transformer, seen)
        if hasattr(value, "__dict__"):
            for item in vars(value).values():
                self._patch_sklearn_compatibility(item, seen)

    @staticmethod
    def _first(values: list[str], default: str = "Unknown") -> str:
        return values[0] if values else default

    @staticmethod
    def _count_words(value: str) -> int:
        return len([part for part in value.split() if part])

    def build_model_input_features(self, intent: IntentRequest) -> dict[str, Any]:
        artifact = self._load_artifact()
        feature_config = artifact["feature_config"]
        features: dict[str, Any] = {}

        for col in feature_config["numeric_features"]:
            features[col] = 0.0
        for col in feature_config["categorical_features"]:
            features[col] = "Unknown"
        for col in feature_config["text_features"]:
            features[col] = ""

        amenities = set(intent.amenities)
        locations = set(intent.location_preferences)
        property_types = set(intent.property_types)
        destination = intent.destination or "Unknown"
        property_type = self._first(intent.property_types)
        raw_query = intent.raw_query or ""

        for col in ("city", "source_city"):
            if col in features:
                features[col] = destination
        if "property_type" in features:
            features["property_type"] = property_type
        if "local_currency" in features:
            features["local_currency"] = "VND"
        if "room_name" in features:
            features["room_name"] = raw_query
        if "description" in features:
            features["description"] = raw_query

        numeric_updates = {
            "max_persons": float(intent.guest_count or 0),
            "bedroom_count": float(intent.room_count or 0),
            "room_index": 0.0,
            "num_rate_options": 0.0,
            "desc_word_count": float(self._count_words(raw_query)),
            "amenities_count": float(len(amenities)),
            "amenity_density": float(len(amenities)),
            "amenity_outdoor_pool": float("pool" in amenities),
            "amenity_indoor_pool": 0.0,
            "breakfast_included": float("breakfast" in amenities),
            "has_breakfast_option": float("breakfast" in amenities),
            "has_beachfront": float("beach" in amenities or "near beach" in locations),
            "has_ocean_front": float("beach" in amenities or "near beach" in locations),
            "has_beach_or_ocean_front": float("beach" in amenities or "near beach" in locations),
            "name_has_beach": float("beach" in amenities or "near beach" in locations),
            "name_has_hotel": float("hotel" in property_types),
            "name_has_resort": float("resort" in property_types),
            "name_has_apartment": float("apartment" in property_types),
            "has_apartment": float("apartment" in property_types),
        }
        if "pool" in amenities:
            numeric_updates["luxury_amenity_score"] = max(numeric_updates.get("luxury_amenity_score", 0.0), 1.0)
        if "resort" in property_types:
            numeric_updates["luxury_route_score"] = max(numeric_updates.get("luxury_route_score", 0.0), 1.0)
            numeric_updates["is_resort_luxury"] = 1.0

        for col, value in numeric_updates.items():
            if col in features:
                features[col] = value

        return features

    def _predict_with_model(self, intent: IntentRequest) -> PriceBandPrediction:
        import pandas as pd

        artifact = self._load_artifact()
        model = artifact["model"]
        label_map = artifact.get("price_class_labels", PRICE_CLASS_LABELS)
        features = self.build_model_input_features(intent)
        feature_cols = artifact.get(
            "feature_cols",
            artifact["feature_config"]["numeric_features"]
            + artifact["feature_config"]["categorical_features"]
            + artifact["feature_config"]["text_features"],
        )
        X = pd.DataFrame([{col: features.get(col, 0.0) for col in feature_cols}])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
            proba = model.predict_proba(X)[0]
        class_id = int(proba.argmax())
        label = label_map[class_id]
        probabilities = {label_map[idx]: float(proba[idx]) for idx in range(len(proba))}
        low, high = PRICE_CLASS_BOUNDS_VND[class_id]

        return PriceBandPrediction(
            price_class_id=class_id,
            price_class_label=label,
            probabilities=probabilities,
            confidence=float(proba[class_id]),
            price_min_vnd=low,
            price_max_vnd=high,
            model_input_features=features,
        )

    def predict_price_band(self, intent: IntentRequest) -> PriceBandPrediction:
        try:
            return self._predict_with_model(intent)
        except Exception as exc:
            if self.model_path.exists():
                raise ModelArtifactError(f"Failed to run price classifier artifact: {exc}") from exc

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
            model_input_features={},
        )
