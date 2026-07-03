import unittest

from agents.web_search_agent import WebSearchAgent, WebSearchConfigurationError
from ML_core.core.schemas import SearchFeatureVector


def make_vector():
    return SearchFeatureVector(
        destination="Da Nang",
        guest_count=2,
        price_min_vnd=1_000_000,
        price_max_vnd=2_000_000,
        price_class_id=2,
        price_class_label="mid_range",
        price_class_confidence=0.66,
        amenity_pool=1,
        near_beach=1,
        property_type_hotel=1,
        raw_query="Khach san Da Nang gan bien 1-2 trieu co ho boi",
    )


class WebSearchAgentTest(unittest.TestCase):
    def test_search_uses_serpapi_when_key_present(self):
        captured_urls = []

        def fake_http_get(url, timeout_seconds):
            captured_urls.append((url, timeout_seconds))
            return {
                "organic_results": [
                    {
                        "title": "Sea Light Hotel Da Nang",
                        "link": "https://www.booking.com/hotel/vn/sea-light.html",
                        "snippet": "Pool, breakfast, near beach. 1.500.000 VND per night.",
                        "source": "Booking.com",
                    }
                ]
            }

        agent = WebSearchAgent(
            environment={"SERPAPI_API_KEY": "demo"},
            http_get_json=fake_http_get,
        )
        candidates, queries = agent.search(make_vector())

        self.assertGreaterEqual(len(queries), 1)
        self.assertEqual(len(candidates), 1)
        self.assertIn("search.json", captured_urls[0][0])
        self.assertEqual(candidates[0].property_type, "hotel")
        self.assertIn("pool", candidates[0].amenities)
        self.assertIn("near beach", candidates[0].location_tags)
        self.assertEqual(candidates[0].metadata["provider"], "serpapi")

    def test_search_uses_google_cse_when_google_keys_present(self):
        captured_urls = []

        def fake_http_get(url, timeout_seconds):
            captured_urls.append((url, timeout_seconds))
            return {
                "items": [
                    {
                        "title": "Central Da Nang Hotel",
                        "link": "https://www.agoda.com/central-da-nang-hotel",
                        "snippet": "Breakfast included, near beach, from 1.800.000 VND.",
                        "displayLink": "agoda.com",
                    }
                ]
            }

        agent = WebSearchAgent(
            environment={"GOOGLE_API_KEY": "demo", "GOOGLE_CSE_ID": "demo"},
            http_get_json=fake_http_get,
        )
        candidates, _queries = agent.search(make_vector())

        self.assertEqual(len(candidates), 1)
        self.assertIn("customsearch", captured_urls[0][0])
        self.assertEqual(candidates[0].metadata["provider"], "google_cse")
        self.assertEqual(candidates[0].metadata["display_link"], "agoda.com")

    def test_search_raises_when_no_provider_is_configured(self):
        agent = WebSearchAgent(environment={})
        with self.assertRaises(WebSearchConfigurationError):
            agent.search(make_vector())

    def test_search_uses_mock_when_use_mock_search_true(self):
        agent = WebSearchAgent(
            environment={"USE_MOCK_SEARCH": "true"},
        )
        candidates, queries = agent.search(make_vector())
        self.assertGreaterEqual(len(queries), 1)
        self.assertGreater(len(candidates), 0)
        # Should contain some mock candidates
        self.assertEqual(candidates[0].metadata["provider"], "mock")


if __name__ == "__main__":
    unittest.main()
