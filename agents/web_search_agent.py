import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from agents.search_query_builder import SearchQueryBuilder
from agents.source_normalizer import normalize_search_result
from ML_core.core.schemas import SearchFeatureVector
from recommendation.schemas import RecommendationCandidate


class WebSearchError(RuntimeError):
    """Base error for web-search failures."""


class WebSearchConfigurationError(WebSearchError):
    """Raised when no external search provider is configured."""


class WebSearchProviderError(WebSearchError):
    """Raised when the configured provider returns an unusable response."""


@dataclass(frozen=True)
class SearchProvider:
    name: str
    endpoint: str


JsonGetter = Callable[[str, float], dict[str, Any]]


class WebSearchAgent:
    def __init__(
        self,
        query_builder: SearchQueryBuilder | None = None,
        environment: Mapping[str, str] | None = None,
        http_get_json: JsonGetter | None = None,
        timeout_seconds: float = 10.0,
        result_limit: int = 8,
    ):
        self.query_builder = query_builder or SearchQueryBuilder()
        self.environment = os.environ if environment is None else environment
        self.http_get_json = http_get_json or self._default_http_get_json
        self.timeout_seconds = timeout_seconds
        self.result_limit = result_limit

    def describe_provider(self) -> str:
        provider = self._detect_provider(optional=True)
        return provider.name if provider is not None else "unconfigured"

    def get_readiness_issues(self) -> list[str]:
        provider = self._detect_provider(optional=True)
        if provider is not None:
            return []
        return ["Missing web-search provider configuration. Set SERPAPI_API_KEY or GOOGLE_API_KEY plus GOOGLE_CSE_ID."]

    def search(self, vector: SearchFeatureVector) -> tuple[list[RecommendationCandidate], list[str]]:
        queries = self._sanitize_queries(self.query_builder.build_queries(vector))
        provider = self._detect_provider()

        candidates: list[RecommendationCandidate] = []
        seen_keys: set[str] = set()
        for query in queries:
            raw_results = self._fetch_results(provider, query)
            for result in raw_results:
                payload = self._to_candidate_payload(result, provider, query, vector)
                candidate = normalize_search_result(payload)
                dedupe_key = (candidate.source_url or candidate.name).strip().lower()
                if not dedupe_key or dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                candidates.append(candidate)
                if len(candidates) >= self.result_limit:
                    return candidates, queries
        return candidates, queries

    def _sanitize_queries(self, queries: list[str]) -> list[str]:
        clean_queries: list[str] = []
        seen: set[str] = set()
        for query in queries:
            normalized = query.strip()
            if not normalized:
                continue
            dedupe_key = normalized.lower()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            clean_queries.append(normalized)
        if not clean_queries:
            raise WebSearchConfigurationError("No usable search query could be built from the feature vector.")
        return clean_queries

    def _detect_provider(self, optional: bool = False) -> SearchProvider | None:
        if self.environment.get("OPENROUTER_API_KEY") and self.environment.get("USE_MOCK_SEARCH", "").lower() != "true":
            return SearchProvider(name="openrouter_search", endpoint="openrouter")
        if self.environment.get("USE_MOCK_SEARCH", "").lower() == "true":
            return SearchProvider(name="mock", endpoint="mock")
        if self.environment.get("SERPAPI_API_KEY"):
            return SearchProvider(name="serpapi", endpoint="https://serpapi.com/search.json")
        if self.environment.get("GOOGLE_API_KEY") and self.environment.get("GOOGLE_CSE_ID"):
            return SearchProvider(name="google_cse", endpoint="https://www.googleapis.com/customsearch/v1")
        if optional:
            return None
        raise WebSearchConfigurationError(
            "Missing web-search provider configuration. Set OPENROUTER_API_KEY, SERPAPI_API_KEY or GOOGLE_API_KEY plus GOOGLE_CSE_ID."
        )

    def _fetch_results(self, provider: SearchProvider, query: str) -> list[dict[str, Any]]:
        if provider.name == "mock":
            return self._generate_mock_results(query)
        if provider.name == "openrouter_search":
            return self._fetch_openrouter_search_results(query)
        if provider.name == "serpapi":
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.environment["SERPAPI_API_KEY"],
                "num": self.result_limit,
            }
        elif provider.name == "google_cse":
            params = {
                "q": query,
                "key": self.environment["GOOGLE_API_KEY"],
                "cx": self.environment["GOOGLE_CSE_ID"],
                "num": min(self.result_limit, 10),
            }
        else:
            raise WebSearchProviderError(f"Unsupported search provider: {provider.name}")

        url = f"{provider.endpoint}?{urlencode(params)}"
        payload = self.http_get_json(url, self.timeout_seconds)
        results = self._extract_results(provider.name, payload)
        if not isinstance(results, list):
            raise WebSearchProviderError(f"{provider.name} returned an unexpected result shape.")
        return [item for item in results if isinstance(item, dict)]

    def _fetch_openrouter_search_results(self, query: str) -> list[dict[str, Any]]:
        model = self.environment.get("OPENROUTER_SEARCH_MODEL", "perplexity/sonar")
        prompt = (
            "You are a helpful travel search assistant.\n"
            "Search the web and find actual hotels in Vietnam matching the request.\n"
            f"Find hotels matching the query: \"{query}\"\n\n"
            "Respond ONLY with a JSON array of hotel objects matching this schema:\n"
            "[\n"
            "  {\n"
            "    \"title\": string (hotel name),\n"
            "    \"snippet\": string (brief description of hotel, price, amenities, and why it fits),\n"
            "    \"link\": string (actual Booking.com, Agoda, Traveloka, or hotel official URL),\n"
            "    \"destination\": string (e.g. \"Da Nang\", \"Phu Quoc\", \"Nha Trang\", \"Ha Noi\", \"Da Lat\"),\n"
            "    \"price_vnd\": number or null (price in VND per night),\n"
            "    \"property_type\": string (choose from: \"hotel\", \"resort\", \"apartment\"),\n"
            "    \"amenities\": array of strings (subset of: [\"pool\", \"breakfast\", \"beach\"]),\n"
            "    \"location_tags\": array of strings (subset of: [\"near beach\", \"near center\"])\n"
            "  }\n"
            "]\n\n"
            "Do not include any explanation or markdown formatting. Just return raw JSON."
        )
        messages = [
            {"role": "user", "content": prompt}
        ]
        try:
            from agents.openrouter_client import call_openrouter, extract_json_from_text
            response_text = call_openrouter(model, messages, json_mode=False)
            cleaned_text = extract_json_from_text(response_text)
            data = json.loads(cleaned_text)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "hotels" in data:
                return list(data["hotels"])
            return []
        except Exception as e:
            print(f"OpenRouter web search failed: {e}")
            return []

    def _extract_results(self, provider_name: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if provider_name == "serpapi":
            return list(payload.get("organic_results") or [])
        if provider_name == "google_cse":
            return list(payload.get("items") or [])
        raise WebSearchProviderError(f"Unsupported search provider: {provider_name}")

    def _to_candidate_payload(
        self,
        result: dict[str, Any],
        provider: SearchProvider,
        query: str,
        vector: SearchFeatureVector,
    ) -> dict[str, Any]:
        title = result.get("title")
        snippet = result.get("snippet")
        link = result.get("link")
        if provider.name == "google_cse":
            metadata = {
                "provider": provider.name,
                "query": query,
                "display_link": result.get("displayLink"),
            }
        elif provider.name == "openrouter_search":
            metadata = {
                "provider": provider.name,
                "query": query,
                "display_link": "openrouter.ai",
            }
        else:
            metadata = {
                "provider": provider.name,
                "query": query,
                "display_link": result.get("displayed_link") or result.get("source"),
            }
        return {
            "title": title,
            "snippet": snippet,
            "url": link,
            "destination": result.get("destination") or vector.destination,
            "price_vnd": result.get("price_vnd"),
            "guest_capacity": result.get("guest_capacity"),
            "amenities": result.get("amenities"),
            "location_tags": result.get("location_tags"),
            "property_type": result.get("property_type"),
            "metadata": metadata,
        }

    def _default_http_get_json(self, url: str, timeout_seconds: float) -> dict[str, Any]:
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "DAP391m/1.0",
            },
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = self._http_error_detail(exc)
            message = f"Search provider HTTP error {exc.code}."
            if detail:
                message = f"{message} {detail}"
            raise WebSearchProviderError(message) from exc
        except URLError as exc:
            raise WebSearchProviderError(f"Search provider request failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise WebSearchProviderError("Search provider returned invalid JSON.") from exc

    def _http_error_detail(self, exc: HTTPError) -> str:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            return ""
        error = payload.get("error") if isinstance(payload, dict) else None
        if not isinstance(error, dict):
            return ""
        message = error.get("message")
        status = error.get("status")
        if message and status:
            return f"{status}: {message}"
        if message:
            return str(message)
        return ""

    def _generate_mock_results(self, query: str) -> list[dict[str, Any]]:
        mock_database = [
            # Da Nang
            {
                "title": "InterContinental Danang Sun Peninsula Resort",
                "snippet": "Premium luxury resort in Da Nang. Features beachfront, infinity pool, spa, fitness center, and complimentary breakfast. Price: 12.500.000 VND per night.",
                "link": "https://www.booking.com/hotel/vn/intercontinental-danang.html",
                "source": "booking.com",
                "destination": "Da Nang",
                "price_vnd": 12500000.0,
                "property_type": "resort",
                "amenities": ["pool", "breakfast", "spa", "beach"],
                "location_tags": ["near beach"],
                "guest_capacity": 4
            },
            {
                "title": "TMS Hotel Danang Beach",
                "snippet": "Beautiful mid_range hotel located steps from My Khe beach in Da Nang. Includes rooftop pool, restaurant, and free wifi. Price: 1.800.000 VND per night.",
                "link": "https://www.agoda.com/tms-hotel-danang.html",
                "source": "agoda.com",
                "destination": "Da Nang",
                "price_vnd": 1800000.0,
                "property_type": "hotel",
                "amenities": ["pool", "wifi", "breakfast"],
                "location_tags": ["near beach"],
                "guest_capacity": 2
            },
            {
                "title": "Aria Grand Hotel & Spa Danang",
                "snippet": "Comfortable economy hotel near center of Da Nang. Close to My Khe beach, features small pool, spa, and breakfast. Price: 850.000 VND per night.",
                "link": "https://www.traveloka.com/hotel/vn/aria-grand-danang.html",
                "source": "traveloka.com",
                "destination": "Da Nang",
                "price_vnd": 850000.0,
                "property_type": "hotel",
                "amenities": ["pool", "breakfast", "beach", "spa"],
                "location_tags": ["near center"],
                "guest_capacity": 2
            },
            {
                "title": "Seashore Hotel & Apartment",
                "snippet": "Modern mid_range apartment style hotel near My Khe beach. Amenities include swimming pool, kitchen, wifi, and steps from the beach. Price: 1.400.000 VND per night.",
                "link": "https://www.booking.com/hotel/vn/seashore-danang.html",
                "source": "booking.com",
                "destination": "Da Nang",
                "price_vnd": 1400000.0,
                "property_type": "apartment",
                "amenities": ["pool", "wifi", "beach"],
                "location_tags": ["near beach"],
                "guest_capacity": 3
            },
            # Nha Trang
            {
                "title": "Six Senses Ninh Van Bay Nha Trang",
                "snippet": "Premium luxury villas at Ninh Van Bay. Ultimate privacy, beach, private pool, spa and breakfast. Price: 15.000.000 VND per night.",
                "link": "https://www.booking.com/hotel/vn/six-senses-ninh-van-bay.html",
                "source": "booking.com",
                "destination": "Nha Trang",
                "price_vnd": 15000000.0,
                "property_type": "resort",
                "amenities": ["pool", "breakfast", "beach", "spa"],
                "location_tags": ["near beach"],
                "guest_capacity": 4
            },
            {
                "title": "Regalia Gold Hotel Nha Trang",
                "snippet": "Mid_range hotel in Nha Trang city center. Features rooftop pool, fitness center, close to beach and breakfast. Price: 1.200.000 VND per night.",
                "link": "https://www.agoda.com/regalia-gold-nha-trang.html",
                "source": "agoda.com",
                "destination": "Nha Trang",
                "price_vnd": 1200000.0,
                "property_type": "hotel",
                "amenities": ["pool", "breakfast", "wifi"],
                "location_tags": ["near center"],
                "guest_capacity": 2
            },
            {
                "title": "Nha Trang Horizon Hotel",
                "snippet": "Mid_range hotel right by the beach in Nha Trang. High-rise sea view, infinity pool, wifi and spa. Price: 1.650.000 VND per night.",
                "link": "https://www.traveloka.com/hotel/vn/nha-trang-horizon.html",
                "source": "traveloka.com",
                "destination": "Nha Trang",
                "price_vnd": 1650000.0,
                "property_type": "hotel",
                "amenities": ["pool", "wifi", "spa", "beach"],
                "location_tags": ["near beach"],
                "guest_capacity": 2
            },
            {
                "title": "Duyen Ha Resort Cam Ranh",
                "snippet": "Upscale family resort near Nha Trang Cam Ranh. Features huge garden, large swimming pool, beach access, spa and restaurant. Price: 2.800.000 VND per night.",
                "link": "https://www.booking.com/hotel/vn/duyen-ha-cam-ranh.html",
                "source": "booking.com",
                "destination": "Nha Trang",
                "price_vnd": 2800000.0,
                "property_type": "resort",
                "amenities": ["pool", "beach", "spa", "breakfast"],
                "location_tags": ["near beach"],
                "guest_capacity": 4
            },
            # Phu Quoc
            {
                "title": "JW Marriott Phu Quoc Emerald Bay Resort & Spa",
                "snippet": "Premium luxury resort on Khem Beach, Phu Quoc. Unique design, beachfront, pools, spa, restaurant. Price: 8.500.000 VND per night.",
                "link": "https://www.booking.com/hotel/vn/jw-marriott-phu-quoc.html",
                "source": "booking.com",
                "destination": "Phu Quoc",
                "price_vnd": 8500000.0,
                "property_type": "resort",
                "amenities": ["pool", "beach", "spa", "breakfast"],
                "location_tags": ["near beach"],
                "guest_capacity": 4
            },
            {
                "title": "Lahana Resort Phu Quoc",
                "snippet": "Mid_range eco-friendly resort near center of Duong Dong. Surrounded by nature, infinity pool, restaurant. Price: 1.500.000 VND per night.",
                "link": "https://www.agoda.com/lahana-resort-phu-quoc.html",
                "source": "agoda.com",
                "destination": "Phu Quoc",
                "price_vnd": 1500000.0,
                "property_type": "resort",
                "amenities": ["pool", "wifi", "breakfast"],
                "location_tags": ["near center"],
                "guest_capacity": 2
            },
            # Ha Noi
            {
                "title": "Sofitel Legend Metropole Hanoi",
                "snippet": "Premium luxury historic hotel in Hanoi French Quarter. Features swimming pool, french restaurant, spa, and fitness center. Price: 7.200.000 VND per night.",
                "link": "https://www.booking.com/hotel/vn/sofitel-legend-metropole.html",
                "source": "booking.com",
                "destination": "Ha Noi",
                "price_vnd": 7200000.0,
                "property_type": "hotel",
                "amenities": ["pool", "breakfast", "spa"],
                "location_tags": ["near center"],
                "guest_capacity": 2
            },
            {
                "title": "Hanoi Golden Lake Hotel",
                "snippet": "Upscale gold-plated hotel near center Hanoi. Features infinity pool on rooftop, restaurant, bar, spa. Price: 3.500.000 VND per night.",
                "link": "https://www.agoda.com/hanoi-golden-lake.html",
                "source": "agoda.com",
                "destination": "Ha Noi",
                "price_vnd": 3500000.0,
                "property_type": "hotel",
                "amenities": ["pool", "wifi", "spa", "breakfast"],
                "location_tags": ["near center"],
                "guest_capacity": 2
            },
            # Generic fallback
            {
                "title": "Standard Stay Hotel & Suites",
                "snippet": "Comfortable mid_range hotel with standard amenities. Includes pool, breakfast, and free wifi. Price: 1.100.000 VND per night.",
                "link": "https://www.booking.com/hotel/vn/standard-stay.html",
                "source": "booking.com",
                "destination": None,
                "price_vnd": 1100000.0,
                "property_type": "hotel",
                "amenities": ["pool", "breakfast", "wifi"],
                "location_tags": ["near center"],
                "guest_capacity": 2
            }
        ]

        normalized_query = query.lower()
        
        # Check if query mentions a specific destination
        target_destination = None
        if "phu quoc" in normalized_query:
            target_destination = "Phu Quoc"
        elif "da nang" in normalized_query or "danang" in normalized_query:
            target_destination = "Da Nang"
        elif "nha trang" in normalized_query:
            target_destination = "Nha Trang"
        elif "ha noi" in normalized_query or "hanoi" in normalized_query:
            target_destination = "Ha Noi"

        # Filter the database by destination first
        filtered_db = mock_database
        if target_destination:
            filtered_db = [item for item in mock_database if item.get("destination") == target_destination]

        results = []
        for item in filtered_db:
            # Check if any query word matches the hotel name or description
            title_words = item["title"].lower().split()
            snippet_words = item["snippet"].lower().split()
            query_words = normalized_query.split()
            
            # Simple keyword matching
            matches = 0
            for qw in query_words:
                if len(qw) > 2: # Ignore short words
                    if any(qw in tw for tw in title_words) or any(qw in sw for sw in snippet_words):
                        matches += 1
            if matches > 0:
                results.append(item)

        if not results:
            # Fallback to standard entries inside filtered_db or all database
            results = [item for item in filtered_db if "Standard" in item["title"] or "TMS" in item["title"] or "Lahana" in item["title"] or "Horizon" in item["title"]]
            if not results:
                results = filtered_db[:3]
            
        return results
