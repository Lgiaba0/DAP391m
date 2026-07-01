from ML_core.core.schemas import SearchFeatureVector
from recommendation.schemas import RankedRecommendation, RecommendationCandidate


RANKING_WEIGHTS = {
    "price_fit": 0.35,
    "amenity_fit": 0.25,
    "location_fit": 0.20,
    "capacity_fit": 0.10,
    "source_quality": 0.10,
}


def _price_fit(candidate: RecommendationCandidate, vector: SearchFeatureVector) -> float:
    if candidate.price_vnd is None:
        return 0.3
    min_vnd = vector.price_min_vnd
    max_vnd = vector.price_max_vnd
    if min_vnd is not None and candidate.price_vnd < min_vnd:
        return 0.6 if vector.expand_budget else 0.2
    if max_vnd is not None and candidate.price_vnd > max_vnd:
        return 0.6 if vector.expand_budget else 0.2
    return 1.0


def _amenity_fit(candidate: RecommendationCandidate, vector: SearchFeatureVector) -> float:
    wanted = []
    if vector.amenity_pool:
        wanted.append("pool")
    if vector.amenity_breakfast:
        wanted.append("breakfast")
    if vector.amenity_beach:
        wanted.append("beach")
    if not wanted:
        return 1.0
    available = set(candidate.amenities)
    return sum(1 for item in wanted if item in available) / len(wanted)


def _location_fit(candidate: RecommendationCandidate, vector: SearchFeatureVector) -> float:
    wanted = []
    if vector.near_beach:
        wanted.append("near beach")
    if vector.near_center:
        wanted.append("near center")
    if not wanted:
        return 1.0
    tags = set(candidate.location_tags)
    return sum(1 for item in wanted if item in tags) / len(wanted)


def _capacity_fit(candidate: RecommendationCandidate, vector: SearchFeatureVector) -> float:
    if vector.guest_count is None or candidate.guest_capacity is None:
        return 0.8
    return 1.0 if candidate.guest_capacity >= vector.guest_count else 0.0


class RecommendationRanker:
    def rank(
        self,
        vector: SearchFeatureVector,
        candidates: list[RecommendationCandidate],
        limit: int = 5,
    ) -> list[RankedRecommendation]:
        ranked = []
        for candidate in candidates:
            components = {
                "price_fit": _price_fit(candidate, vector),
                "amenity_fit": _amenity_fit(candidate, vector),
                "location_fit": _location_fit(candidate, vector),
                "capacity_fit": _capacity_fit(candidate, vector),
                "source_quality": max(0.0, min(candidate.source_quality, 1.0)),
            }
            score = sum(components[name] * RANKING_WEIGHTS[name] for name in RANKING_WEIGHTS)
            reasons = [name for name, value in components.items() if value >= 0.95]
            tradeoffs = [name for name, value in components.items() if value < 0.5]
            ranked.append(RankedRecommendation(candidate=candidate, score=round(score, 4), reasons=reasons, tradeoffs=tradeoffs))
        return sorted(ranked, key=lambda item: item.score, reverse=True)[:limit]
