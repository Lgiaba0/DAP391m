import unittest

from intent.intent_parser import IntentParser


class IntentParserTest(unittest.TestCase):
    def test_parse_basic_vietnamese_without_accents(self):
        intent = IntentParser().parse("Tim phong khach san Da Nang cho 2 nguoi, gan bien, 1-2 trieu, co ho boi")
        self.assertEqual(intent.destination, "Da Nang")
        self.assertEqual(intent.guest_count, 2)
        self.assertEqual(intent.budget_min_vnd, 1_000_000)
        self.assertEqual(intent.budget_max_vnd, 2_000_000)
        self.assertIn("pool", intent.amenities)
        self.assertIn("near beach", intent.location_preferences)


if __name__ == "__main__":
    unittest.main()
