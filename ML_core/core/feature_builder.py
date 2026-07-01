from .config import LOW_CONFIDENCE_THRESHOLD
from .schemas import IntentRequest, PriceBandPrediction, SearchFeatureVector


class FeatureBuilder:
    def build(self, intent: IntentRequest, price_band: PriceBandPrediction) -> SearchFeatureVector:
        amenities = set(intent.amenities)
        locations = set(intent.location_preferences)
        property_types = set(intent.property_types)
        return SearchFeatureVector(
            destination=intent.destination,
            guest_count=intent.guest_count,
            price_min_vnd=price_band.price_min_vnd,
            price_max_vnd=price_band.price_max_vnd,
            price_class_id=price_band.price_class_id,
            price_class_label=price_band.price_class_label,
            price_class_confidence=price_band.confidence,
            amenity_pool=int("pool" in amenities),
            amenity_beach=int("beach" in amenities),
            amenity_breakfast=int("breakfast" in amenities),
            near_beach=int("near beach" in locations),
            near_center=int("near center" in locations),
            property_type_hotel=int("hotel" in property_types),
            property_type_apartment=int("apartment" in property_types),
            property_type_resort=int("resort" in property_types),
            expand_budget=price_band.confidence < LOW_CONFIDENCE_THRESHOLD,
            raw_query=intent.raw_query,
        )
