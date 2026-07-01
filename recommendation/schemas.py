from dataclasses import asdict, dataclass, field
from typing import Any

from ML_core.core.schemas import IntentRequest, PriceBandPrediction, SearchFeatureVector


@dataclass
class RecommendationCandidate:
    name: str
    source_url: str | None = None
    price_vnd: float | None = None
    destination: str | None = None
    guest_capacity: int | None = None
    amenities: list[str] = field(default_factory=list)
    location_tags: list[str] = field(default_factory=list)
    property_type: str | None = None
    source_quality: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RankedRecommendation:
    candidate: RecommendationCandidate
    score: float
    reasons: list[str] = field(default_factory=list)
    tradeoffs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "score": self.score,
            "reasons": list(self.reasons),
            "tradeoffs": list(self.tradeoffs),
        }


@dataclass
class RecommendationResponse:
    parsed_intent: IntentRequest
    price_band: PriceBandPrediction
    feature_vector: SearchFeatureVector
    recommendations: list[RankedRecommendation]
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parsed_intent": self.parsed_intent.to_dict(),
            "price_band": self.price_band.to_dict(),
            "feature_vector": self.feature_vector.to_dict(),
            "recommendations": [item.to_dict() for item in self.recommendations],
            "debug": dict(self.debug),
        }
