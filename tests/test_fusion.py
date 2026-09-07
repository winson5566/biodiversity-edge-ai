import unittest

import numpy as np

from biodiversity_edge_ai.fusion import fuse_probabilities, top_k


class FusionTests(unittest.TestCase):
    def test_log_linear_result_is_normalized(self):
        result = fuse_probabilities(
            np.array([0.8, 0.2]),
            np.array([0.01, 0.99]),
            alpha=0.3,
        )
        self.assertAlmostEqual(float(result.sum()), 1.0, places=6)
        self.assertEqual(int(np.argmax(result)), 1)

    def test_invalid_location_falls_back_to_vision(self):
        result = fuse_probabilities(
            np.array([8.0, 2.0]),
            np.array([0.01, 0.99]),
            location_valid=False,
        )
        np.testing.assert_allclose(result, [0.8, 0.2], rtol=1e-6)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            fuse_probabilities(np.array([0.5, 0.5]), np.array([1.0]))

    def test_top_k_is_sorted(self):
        indices, scores = top_k(np.array([0.2, 0.7, 0.1]), k=2)
        np.testing.assert_array_equal(indices, [1, 0])
        np.testing.assert_allclose(scores, [0.7, 0.2])


if __name__ == "__main__":
    unittest.main()
