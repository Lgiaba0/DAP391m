from agents.web_search_agent import WebSearchAgent
from intent.intent_parser import IntentParser
from ML_core.core.feature_builder import FeatureBuilder
from ML_core.core.price_band_service import PriceBandService
from recommendation.ranker import RecommendationRanker
from recommendation.schemas import RecommendationResponse


class RecommendPipeline:
    def __init__(
        self,
        intent_parser: IntentParser | None = None,
        price_band_service: PriceBandService | None = None,
        feature_builder: FeatureBuilder | None = None,
        web_search_agent: WebSearchAgent | None = None,
        ranker: RecommendationRanker | None = None,
    ):
        self.intent_parser = intent_parser or IntentParser()
        self.price_band_service = price_band_service or PriceBandService()
        self.feature_builder = feature_builder or FeatureBuilder()
        self.web_search_agent = web_search_agent or WebSearchAgent()
        self.ranker = ranker or RecommendationRanker()

    def recommend(self, raw_user_query: str) -> RecommendationResponse:
        intent = self.intent_parser.parse(raw_user_query)
        price_band = self.price_band_service.predict_price_band(intent)
        vector = self.feature_builder.build(intent, price_band)
        candidates, queries = self.web_search_agent.search(vector)
        recommendations = self.ranker.rank(vector, candidates)
        return RecommendationResponse(
            parsed_intent=intent,
            price_band=price_band,
            feature_vector=vector,
            recommendations=recommendations,
            debug={"search_queries": queries, "ranking_version": "v1"},
        )


def recommend(raw_user_query: str) -> RecommendationResponse:
    return RecommendPipeline().recommend(raw_user_query)
