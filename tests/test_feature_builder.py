import unittest

from intent.intent_parser import IntentParser
from ML_core.core.feature_builder import FeatureBuilder
from ML_core.core.schemas import PriceBandPrediction


class FeatureBuilderTest(unittest.TestCase):
    def test_build_vector_from_intent_and_price_band(self):
        intent = IntentParser().parse("Da Nang hotel 1-2 trieu pool near beach for 2")
        price_band = PriceBandPrediction(
            price_class_id=2,
            price_class_label="mid_range",
            probabilities={
                "budget": 0.05,
                "economy": 0.1,
                "mid_range": 0.7,
                "upscale": 0.1,
                "premium_luxury": 0.05,
            },
            confidence=0.7,
            price_min_vnd=1_000_000,
            price_max_vnd=2_000_000,
        )
        vector = FeatureBuilder().build(intent, price_band)
        self.assertEqual(vector.price_class_label, "mid_range")
        self.assertEqual(vector.amenity_pool, 1)
        self.assertEqual(vector.near_beach, 1)


if __name__ == "__main__":
    unittest.main()
