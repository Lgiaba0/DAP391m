import unittest

from ML_core.core.schemas import IntentRequest, PriceBandPrediction, SearchFeatureVector
from recommendation.schemas import RecommendationCandidate, RankedRecommendation, RecommendationResponse

import api_server


class FakeSearchAgent:
    def __init__(self):
        self._issues = []

    def describe_provider(self):
        return "serpapi"

    def get_readiness_issues(self):
        return list(self._issues)


class FakePriceBandService:
    model_path = "fake.joblib"

    def __init__(self):
        self._issues = []

    def get_readiness_issues(self):
        return list(self._issues)


class FakePipeline:
    def __init__(self):
        self.web_search_agent = FakeSearchAgent()
        self.price_band_service = FakePriceBandService()

    def recommend(self, raw_user_query):
        return RecommendationResponse(
            parsed_intent=IntentRequest(raw_query=raw_user_query, destination="Da Nang"),
            price_band=PriceBandPrediction(
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
            ),
            feature_vector=SearchFeatureVector(
                destination="Da Nang",
                guest_count=2,
                price_min_vnd=1_000_000,
                price_max_vnd=2_000_000,
                price_class_id=2,
                price_class_label="mid_range",
                price_class_confidence=0.7,
                amenity_pool=1,
                near_beach=1,
                property_type_hotel=1,
                raw_query=raw_user_query,
            ),
            recommendations=[
                RankedRecommendation(
                    candidate=RecommendationCandidate(
                        name="Sea Light Hotel",
                        source_url="https://example.com/hotel",
                        price_vnd=1_500_000,
                        destination="Da Nang",
                    ),
                    score=0.88,
                    reasons=["price_fit"],
                    tradeoffs=[],
                )
            ],
            debug={"search_queries": ["Da Nang hotel"]},
        )


class ApiServerTest(unittest.TestCase):
    @unittest.skipUnless(api_server.HAS_FLASK, "Flask is not installed in the current environment.")
    def test_recommend_endpoint_returns_pipeline_response(self):
        app = api_server.create_app(FakePipeline())
        client = app.test_client()

        response = client.post("/api/recommend", json={"query": "Khach san Da Nang"})
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["parsed_intent"]["destination"], "Da Nang")
        self.assertEqual(payload["recommendations"][0]["candidate"]["name"], "Sea Light Hotel")

    @unittest.skipUnless(api_server.HAS_FLASK, "Flask is not installed in the current environment.")
    def test_recommend_endpoint_rejects_empty_query(self):
        app = api_server.create_app(FakePipeline())
        client = app.test_client()

        response = client.post("/api/recommend", json={"query": "   "})
        payload = response.get_json()

        self.assertEqual(response.status_code, 400)
        self.assertEqual(payload["error"], "`query` is required.")

    @unittest.skipUnless(api_server.HAS_FLASK, "Flask is not installed in the current environment.")
    def test_health_endpoint_exposes_readiness(self):
        pipeline = FakePipeline()
        pipeline.web_search_agent._issues = ["search key missing"]
        pipeline.price_band_service._issues = ["model missing"]

        app = api_server.create_app(pipeline)
        client = app.test_client()

        response = client.get("/health")
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertFalse(payload["ready"])
        self.assertEqual(payload["search_provider"], "serpapi")
        self.assertEqual(payload["price_classifier_path"], "fake.joblib")
        self.assertEqual(payload["issues"], ["model missing", "search key missing"])


if __name__ == "__main__":
    unittest.main()
