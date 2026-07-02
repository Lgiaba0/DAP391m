from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class IntentRequest:
    raw_query: str
    destination: str | None = None
    check_in: str | None = None
    check_out: str | None = None
    guest_count: int | None = None
    room_count: int | None = None
    budget_min_vnd: float | None = None
    budget_max_vnd: float | None = None
    amenities: list[str] = field(default_factory=list)
    location_preferences: list[str] = field(default_factory=list)
    property_types: list[str] = field(default_factory=list)
    room_preferences: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PriceBandPrediction:
    price_class_id: int
    price_class_label: str
    probabilities: dict[str, float]
    confidence: float
    price_min_vnd: float | None = None
    price_max_vnd: float | None = None
    model_input_features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchFeatureVector:
    destination: str | None
    guest_count: int | None
    price_min_vnd: float | None
    price_max_vnd: float | None
    price_class_id: int
    price_class_label: str
    price_class_confidence: float
    amenity_pool: int = 0
    amenity_beach: int = 0
    amenity_breakfast: int = 0
    near_beach: int = 0
    near_center: int = 0
    property_type_hotel: int = 0
    property_type_apartment: int = 0
    property_type_resort: int = 0
    expand_budget: bool = False
    raw_query: str = ""
    model_input_features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
