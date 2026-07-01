import unittest

from pipelines.recommend_pipeline import recommend


class RecommendPipelineTest(unittest.TestCase):
    def test_recommend_returns_ranked_response(self):
        response = recommend("Tim phong khach san Da Nang cho 2 nguoi, gan bien, 1-2 trieu, co ho boi")
        self.assertEqual(response.parsed_intent.destination, "Da Nang")
        self.assertEqual(response.price_band.price_class_label, "mid_range")
        self.assertGreaterEqual(len(response.recommendations), 1)


if __name__ == "__main__":
    unittest.main()
