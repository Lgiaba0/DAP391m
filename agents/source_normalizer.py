import re
from typing import Any
from urllib.parse import urlparse

from recommendation.schemas import RecommendationCandidate


PRICE_RANGE_PATTERN = re.compile(
    r"(?P<low>\d+(?:[.,]\d+)?)\s*[-–]\s*(?P<high>\d+(?:[.,]\d+)?)\s*(?P<unit>trieu|million|m|k|nghin|thousand)?",
    re.IGNORECASE,
)
PRICE_SINGLE_PATTERN = re.compile(
    r"(?P<value>\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)\s*(?P<unit>trieu|million|m|k|nghin|thousand|vnd)?",
    re.IGNORECASE,
)

PROPERTY_TYPE_PATTERNS = {
    "resort": re.compile(r"\bresort\b", re.IGNORECASE),
    "apartment": re.compile(r"\bapartment\b|\bcan ho\b", re.IGNORECASE),
    "hotel": re.compile(r"\bhotel\b|\bkhach san\b", re.IGNORECASE),
    "villa": re.compile(r"\bvilla\b", re.IGNORECASE),
    "homestay": re.compile(r"\bhomestay\b", re.IGNORECASE),
}

AMENITY_PATTERNS = {
    "pool": re.compile(r"\bpool\b|\bho boi\b|\bswimming\b", re.IGNORECASE),
    "breakfast": re.compile(r"\bbreakfast\b|\bbua sang\b", re.IGNORECASE),
    "beach": re.compile(r"\bbeach\b|\bbien\b", re.IGNORECASE),
    "spa": re.compile(r"\bspa\b", re.IGNORECASE),
    "wifi": re.compile(r"\bwifi\b|\bwi-fi\b", re.IGNORECASE),
}

LOCATION_PATTERNS = {
    "near beach": re.compile(r"\bnear beach\b|\bgan bien\b|\bsteps from the beach\b", re.IGNORECASE),
    "near center": re.compile(r"\bnear center\b|\bcity center\b|\bgan trung tam\b|\bdowntown\b", re.IGNORECASE),
}

SOURCE_QUALITY_BY_DOMAIN = {
    "booking.com": 0.92,
    "agoda.com": 0.9,
    "tripadvisor.com": 0.84,
    "expedia.com": 0.88,
    "hotels.com": 0.86,
    "traveloka.com": 0.87,
    "airbnb.com": 0.82,
    "ivivu.com": 0.8,
}


def _unit_multiplier(unit: str | None) -> float:
    normalized = (unit or "").lower()
    if normalized in {"trieu", "million", "m"}:
        return 1_000_000.0
    if normalized in {"k", "nghin", "thousand"}:
        return 1_000.0
    return 1.0


def _normalize_amount(raw_value: str, unit: str | None) -> float | None:
    compact = raw_value.replace(",", "").strip()
    if compact.count(".") > 1:
        compact = compact.replace(".", "")
    try:
        return float(compact) * _unit_multiplier(unit)
    except ValueError:
        return None


def _extract_price_vnd(text: str) -> float | None:
    range_match = PRICE_RANGE_PATTERN.search(text)
    if range_match:
        low = _normalize_amount(range_match.group("low"), range_match.group("unit"))
        high = _normalize_amount(range_match.group("high"), range_match.group("unit"))
        if low is not None and high is not None:
            return round((low + high) / 2, 2)

    for match in PRICE_SINGLE_PATTERN.finditer(text):
        value = _normalize_amount(match.group("value"), match.group("unit"))
        if value is not None and value >= 100_000:
            return value
    return None


def _infer_property_type(text: str) -> str | None:
    for property_type, pattern in PROPERTY_TYPE_PATTERNS.items():
        if pattern.search(text):
            return property_type
    return None


def _infer_tags(text: str, patterns: dict[str, re.Pattern[str]]) -> list[str]:
    return [name for name, pattern in patterns.items() if pattern.search(text)]


def _domain_quality(source_url: str | None) -> float:
    if not source_url:
        return 0.5
    domain = urlparse(source_url).netloc.lower()
    domain = domain.removeprefix("www.")
    for known_domain, score in SOURCE_QUALITY_BY_DOMAIN.items():
        if domain == known_domain or domain.endswith(f".{known_domain}"):
            return score
    return 0.6 if domain else 0.5


def normalize_search_result(result: dict[str, Any]) -> RecommendationCandidate:
    title = str(result.get("name") or result.get("title") or "Unknown stay")
    snippet = str(result.get("snippet") or result.get("description") or "")
    combined_text = f"{title} {snippet}".strip()
    source_url = result.get("source_url") or result.get("url") or result.get("link")

    explicit_amenities = list(result.get("amenities") or [])
    explicit_locations = list(result.get("location_tags") or [])
    inferred_amenities = _infer_tags(combined_text, AMENITY_PATTERNS)
    inferred_locations = _infer_tags(combined_text, LOCATION_PATTERNS)

    return RecommendationCandidate(
        name=title,
        source_url=source_url,
        price_vnd=result.get("price_vnd") or _extract_price_vnd(combined_text),
        destination=result.get("destination"),
        guest_capacity=result.get("guest_capacity"),
        amenities=list(dict.fromkeys(explicit_amenities + inferred_amenities)),
        location_tags=list(dict.fromkeys(explicit_locations + inferred_locations)),
        property_type=result.get("property_type") or _infer_property_type(combined_text),
        source_quality=float(result.get("source_quality", _domain_quality(source_url))),
        metadata=dict(result.get("metadata") or {}),
    )
