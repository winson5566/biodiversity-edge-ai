import datetime as dt
import math
import unittest

import numpy as np

from biodiversity_edge_ai.metadata import encode_geo_features, year_fraction


class MetadataTests(unittest.TestCase):
    def test_legacy_year_fraction_is_one_based(self):
        self.assertAlmostEqual(year_fraction(dt.date(2025, 1, 1)), 1 / 365)
        self.assertAlmostEqual(year_fraction("2024-12-31 00:00:00+00:00"), 1.0)

    def test_feature_order_matches_legacy_project(self):
        encoded = encode_geo_features(0.0, 180.0, "2025-01-01")
        self.assertEqual(encoded.dtype, np.float32)
        self.assertEqual(encoded.shape, (6,))
        self.assertAlmostEqual(float(encoded[0]), 0.0, places=6)
        self.assertAlmostEqual(float(encoded[1]), -1.0, places=6)
        self.assertAlmostEqual(float(encoded[2]), 0.0, places=6)
        self.assertAlmostEqual(float(encoded[3]), 1.0, places=6)
        expected_date = (1 / 365) * 2 - 1
        self.assertAlmostEqual(float(encoded[4]), math.sin(math.pi * expected_date), places=6)

    def test_coordinates_are_validated(self):
        with self.assertRaises(ValueError):
            encode_geo_features(91, 0, "2025-01-01")


if __name__ == "__main__":
    unittest.main()
