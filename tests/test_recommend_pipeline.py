import unittest

from pipelines.recommend_pipeline import recommend


class RecommendPipelineTest(unittest.TestCase):
    def test_recommend_returns_ranked_response(self):
        response = recommend("Tim phong khach san Da Nang cho 2 nguoi, gan bien, 1-2 trieu, co ho boi")
        self.assertEqual(response.parsed_intent.destination, "Da Nang")
        self.assertIn(
            response.price_band.price_class_label,
            {"budget", "economy", "mid_range", "upscale", "premium_luxury"},
        )
        self.assertGreater(response.price_band.confidence, 0)
        self.assertIn("model_input_features", response.feature_vector.to_dict())
        self.assertEqual(response.feature_vector.model_input_features["amenity_outdoor_pool"], 1.0)
        self.assertGreaterEqual(len(response.recommendations), 1)


if __name__ == "__main__":
    unittest.main()
