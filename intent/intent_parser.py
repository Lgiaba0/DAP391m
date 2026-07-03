import re
import unicodedata

from ML_core.core.schemas import IntentRequest


def _remove_diacritics(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn").replace("đ", "d").replace("Đ", "D")


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


import os
import json
from agents.openrouter_client import call_openrouter

class IntentParser:
    def parse(self, raw_query: str) -> IntentRequest:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if api_key:
            model = os.environ.get("OPENROUTER_INTENT_MODEL", "google/gemini-2.5-flash")
            prompt = (
                "You are a structured intent extractor for a travel search engine in Vietnam.\n"
                "Analyze the user's travel query and extract details into a JSON object matching this schema:\n"
                "{\n"
                "  \"destination\": string or null (extract normalized name: \"Da Nang\", \"Nha Trang\", \"Ho Chi Minh City\", \"Ha Noi\", \"Da Lat\", \"Phu Quoc\"),\n"
                "  \"guest_count\": integer or null,\n"
                "  \"budget_min_vnd\": number or null,\n"
                "  \"budget_max_vnd\": number or null,\n"
                "  \"amenities\": array of strings (subset of: [\"pool\", \"breakfast\", \"beach\"]),\n"
                "  \"location_preferences\": array of strings (subset of: [\"near beach\", \"near center\"]),\n"
                "  \"property_types\": array of strings (subset of: [\"hotel\", \"apartment\", \"resort\"])\n"
                "}\n\n"
                f"Query: \"{raw_query}\"\n\n"
                "Provide ONLY the raw JSON object, no explanation, no markdown blocks."
            )
            messages = [
                {"role": "user", "content": prompt}
            ]
            try:
                response_text = call_openrouter(model, messages, json_mode=False)
                from agents.openrouter_client import extract_json_from_text
                cleaned_text = extract_json_from_text(response_text)
                data = json.loads(cleaned_text)
                return IntentRequest(
                    raw_query=raw_query,
                    destination=data.get("destination"),
                    guest_count=data.get("guest_count"),
                    budget_min_vnd=data.get("budget_min_vnd"),
                    budget_max_vnd=data.get("budget_max_vnd"),
                    amenities=list(data.get("amenities") or []),
                    location_preferences=list(data.get("location_preferences") or []),
                    property_types=list(data.get("property_types") or []),
                )
            except Exception as e:
                # Log and fallback to regex parser below
                print(f"OpenRouter intent parsing failed, falling back to regex: {e}")

        normalized = _remove_diacritics(raw_query.lower())
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
