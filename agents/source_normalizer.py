from typing import Any

from recommendation.schemas import RecommendationCandidate


def normalize_search_result(result: dict[str, Any]) -> RecommendationCandidate:
    return RecommendationCandidate(
        name=str(result.get("name") or result.get("title") or "Unknown stay"),
        source_url=result.get("source_url") or result.get("url"),
        price_vnd=result.get("price_vnd"),
        destination=result.get("destination"),
        guest_capacity=result.get("guest_capacity"),
        amenities=list(result.get("amenities") or []),
        location_tags=list(result.get("location_tags") or []),
        property_type=result.get("property_type"),
        source_quality=float(result.get("source_quality", 0.5)),
        metadata=dict(result.get("metadata") or {}),
    )
