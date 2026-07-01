from agents.search_query_builder import SearchQueryBuilder
from ML_core.core.schemas import SearchFeatureVector
from recommendation.schemas import RecommendationCandidate


class WebSearchAgent:
    def __init__(self, query_builder: SearchQueryBuilder | None = None):
        self.query_builder = query_builder or SearchQueryBuilder()

    def search(self, vector: SearchFeatureVector) -> tuple[list[RecommendationCandidate], list[str]]:
        queries = self.query_builder.build_queries(vector)
        candidates = [
            RecommendationCandidate(
                name=f"Mock {vector.destination or 'Vietnam'} {vector.price_class_label} stay",
                source_url=None,
                price_vnd=vector.price_min_vnd or vector.price_max_vnd,
                destination=vector.destination,
                guest_capacity=vector.guest_count,
                amenities=[name for flag, name in [(vector.amenity_pool, "pool"), (vector.amenity_breakfast, "breakfast")] if flag],
                location_tags=[name for flag, name in [(vector.near_beach, "near beach"), (vector.near_center, "near center")] if flag],
                property_type="hotel" if vector.property_type_hotel else None,
                source_quality=0.5,
                metadata={"mock": True},
            )
        ]
        return candidates, queries
