import unittest
from pathlib import Path
from unittest.mock import patch

from ML_core.core.config import get_default_price_classifier_path
from ML_core.core.exceptions import ModelArtifactError
from ML_core.core.price_band_service import PriceBandService
from ML_core.core.schemas import IntentRequest


class FakeModel:
    classes_ = [0, 1, 2, 3, 4]

    def predict_proba(self, frame):
        self.last_frame = frame
        return [[0.05, 0.12, 0.66, 0.11, 0.06]]


class PriceBandServiceTest(unittest.TestCase):
    def test_predict_price_band_uses_model_confidence_and_budget_bounds(self):
        fake_model = FakeModel()

        class FakePriceBandService(PriceBandService):
            def _load_artifact(self):
                return {
                    "model": fake_model,
                    "feature_cols": [
                        "city",
                        "source_city",
                        "region",
                        "property_type",
                        "room_name",
                        "description",
                        "amenities_count",
                        "max_persons",
                    ],
                    "feature_config": {
                        "numeric_features": ["amenities_count", "max_persons"],
                        "categorical_features": ["city", "source_city", "region", "property_type"],
                        "text_features": ["room_name", "description"],
                    },
                    "price_class_labels": {
                        0: "budget",
                        1: "economy",
                        2: "mid_range",
                        3: "upscale",
                        4: "premium_luxury",
                    },
                }

        service = FakePriceBandService()
        intent = IntentRequest(
            raw_query="Khach san Da Nang 1-2 trieu co ho boi",
            destination="Da Nang",
            guest_count=2,
            budget_min_vnd=1_000_000,
            budget_max_vnd=2_000_000,
            amenities=["pool"],
            property_types=["hotel"],
        )

        prediction = service.predict_price_band(intent)
        self.assertEqual(prediction.price_class_id, 2)
        self.assertEqual(prediction.price_class_label, "mid_range")
        self.assertAlmostEqual(prediction.confidence, 0.66)
        self.assertEqual(prediction.price_min_vnd, 1_000_000)
        self.assertEqual(prediction.price_max_vnd, 2_000_000)
        self.assertEqual(fake_model.last_frame.iloc[0]["city"], "Da Nang")
        self.assertEqual(fake_model.last_frame.iloc[0]["property_type"], "hotel")

    def test_predict_price_band_raises_when_artifact_unavailable(self):
        class MissingArtifactService(PriceBandService):
            def _load_artifact(self):
                raise ModelArtifactError("artifact missing")

        service = MissingArtifactService()
        with self.assertRaises(ModelArtifactError):
            service.predict_price_band(IntentRequest(raw_query="khach san"))

    def test_default_price_classifier_path_uses_repo_root_model_when_present(self):
        expected = Path(__file__).resolve().parents[1] / "price_classification_v3_model.joblib"
        with patch("ML_core.core.config.os.environ", {}):
            resolved = get_default_price_classifier_path()
        self.assertEqual(resolved, expected)


if __name__ == "__main__":
    unittest.main()
