import re

from ML_core.core.schemas import IntentRequest


AMENITY_KEYWORDS = {
    "pool": ["pool", "ho boi", "be boi"],
    "breakfast": ["breakfast", "an sang"],
    "beach": ["beach", "bien"],
}

LOCATION_KEYWORDS = {
    "near beach": ["near beach", "gan bien"],
    "near center": ["near center", "trung tam", "gan trung tam"],
}

PROPERTY_TYPE_KEYWORDS = {
    "hotel": ["hotel", "khach san"],
    "apartment": ["apartment", "can ho"],
    "resort": ["resort"],
}

DESTINATION_KEYWORDS = {
    "Da Nang": ["da nang", "danang"],
    "Nha Trang": ["nha trang"],
    "Ho Chi Minh City": ["ho chi minh", "sai gon", "saigon"],
    "Ha Noi": ["ha noi", "hanoi"],
    "Da Lat": ["da lat", "dalat"],
    "Phu Quoc": ["phu quoc"],
}


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _extract_guest_count(text: str) -> int | None:
    patterns = [
        r"(\d+)\s*(?:nguoi|people|guests|khach)",
        r"for\s+(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def _money_to_vnd(value: str, unit: str | None) -> float:
    amount = float(value.replace(",", "."))
    if unit and unit.lower() in {"trieu", "m", "million"}:
        return amount * 1_000_000
    if unit and unit.lower() in {"k", "nghin"}:
        return amount * 1_000
    return amount


def _extract_budget(text: str) -> tuple[float | None, float | None]:
    range_match = re.search(
        r"(\d+(?:[\.,]\d+)?)\s*(?:-|to|den)\s*(\d+(?:[\.,]\d+)?)\s*(trieu|m|million|k|nghin)?",
        text,
    )
    if range_match:
        unit = range_match.group(3)
        return _money_to_vnd(range_match.group(1), unit), _money_to_vnd(range_match.group(2), unit)

    under_match = re.search(r"(?:under|duoi|max)\s*(\d+(?:[\.,]\d+)?)\s*(trieu|m|million|k|nghin)?", text)
    if under_match:
        return None, _money_to_vnd(under_match.group(1), under_match.group(2))

    around_match = re.search(r"(?:around|khoang)\s*(\d+(?:[\.,]\d+)?)\s*(trieu|m|million|k|nghin)?", text)
    if around_match:
        center = _money_to_vnd(around_match.group(1), around_match.group(2))
        return center * 0.8, center * 1.2

    return None, None


class IntentParser:
    def parse(self, raw_query: str) -> IntentRequest:
        normalized = raw_query.lower()
        budget_min, budget_max = _extract_budget(normalized)

        destination = None
        for label, keywords in DESTINATION_KEYWORDS.items():
            if _contains_any(normalized, keywords):
                destination = label
                break

        return IntentRequest(
            raw_query=raw_query,
            destination=destination,
            guest_count=_extract_guest_count(normalized),
            budget_min_vnd=budget_min,
            budget_max_vnd=budget_max,
            amenities=[label for label, keywords in AMENITY_KEYWORDS.items() if _contains_any(normalized, keywords)],
            location_preferences=[
                label for label, keywords in LOCATION_KEYWORDS.items() if _contains_any(normalized, keywords)
            ],
            property_types=[
                label for label, keywords in PROPERTY_TYPE_KEYWORDS.items() if _contains_any(normalized, keywords)
            ],
        )
