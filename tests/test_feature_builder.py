import unittest

from intent.intent_parser import IntentParser
from ML_core.core.feature_builder import FeatureBuilder
from ML_core.core.price_band_service import PriceBandService


class FeatureBuilderTest(unittest.TestCase):
    def test_build_vector_from_intent_and_price_band(self):
        intent = IntentParser().parse("Da Nang hotel 1-2 trieu pool near beach for 2")
        price_band = PriceBandService().predict_price_band(intent)
        vector = FeatureBuilder().build(intent, price_band)
        self.assertEqual(vector.price_class_label, "mid_range")
        self.assertEqual(vector.amenity_pool, 1)
        self.assertEqual(vector.near_beach, 1)


if __name__ == "__main__":
    unittest.main()
