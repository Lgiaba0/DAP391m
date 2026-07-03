import unittest

from ML_core.core.schemas import PriceBandPrediction
from pipelines.recommend_pipeline import RecommendPipeline
from recommendation.schemas import RecommendationCandidate


class RecommendPipelineTest(unittest.TestCase):
    def test_recommend_returns_ranked_response(self):
        class FakePriceBandService:
            def predict_price_band(self, intent):
                return PriceBandPrediction(
                    price_class_id=2,
                    price_class_label="mid_range",
                    probabilities={
                        "budget": 0.05,
                        "economy": 0.1,
                        "mid_range": 0.72,
                        "upscale": 0.09,
                        "premium_luxury": 0.04,
                    },
                    confidence=0.72,
                    price_min_vnd=intent.budget_min_vnd,
                    price_max_vnd=intent.budget_max_vnd,
                )

        class FakeWebSearchAgent:
            def search(self, vector):
                return (
                    [
                        RecommendationCandidate(
                            name="Da Nang Beach Hotel",
                            source_url="https://example.com/da-nang-beach-hotel",
                            price_vnd=1_500_000,
                            destination=vector.destination,
                            guest_capacity=2,
                            amenities=["pool"],
                            location_tags=["near beach"],
                            property_type="hotel",
                            source_quality=0.8,
                        )
                    ],
                    ["Da Nang hotel near beach"],
                )

        pipeline = RecommendPipeline(
            price_band_service=FakePriceBandService(),
            web_search_agent=FakeWebSearchAgent(),
        )
        response = pipeline.recommend("Tim phong khach san Da Nang cho 2 nguoi, gan bien, 1-2 trieu, co ho boi")
        self.assertEqual(response.parsed_intent.destination, "Da Nang")
        self.assertEqual(response.price_band.price_class_label, "mid_range")
        self.assertGreaterEqual(len(response.recommendations), 1)


if __name__ == "__main__":
    unittest.main()
