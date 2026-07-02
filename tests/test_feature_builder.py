import unittest

from intent.intent_parser import IntentParser
from ML_core.core.feature_builder import FeatureBuilder
from ML_core.core.price_band_service import PriceBandService


class FeatureBuilderTest(unittest.TestCase):
    def test_build_vector_from_intent_and_price_band(self):
        intent = IntentParser().parse("Da Nang hotel 1-2 trieu pool near beach for 2")
        price_band = PriceBandService().predict_price_band(intent)
        vector = FeatureBuilder().build(intent, price_band)
        self.assertIn(vector.price_class_label, {"budget", "economy", "mid_range", "upscale", "premium_luxury"})
        self.assertGreater(vector.price_class_confidence, 0)
        self.assertEqual(vector.amenity_pool, 1)
        self.assertEqual(vector.near_beach, 1)
        self.assertEqual(vector.model_input_features["max_persons"], 2.0)
        self.assertEqual(vector.model_input_features["amenity_outdoor_pool"], 1.0)


if __name__ == "__main__":
    unittest.main()
