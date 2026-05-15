import unittest

from trackpy.utils import default_pos_columns
from trackpy.artificial import draw_array
from trackpy.spiff import apply_spiff_correction
from trackpy.tests.common import StrictTestCase, sort_positions
import trackpy as tp
import numpy as np

class TestSpiff(StrictTestCase):
    def _test_spiff(self, ndim):
        # Draw an image with 200 features and some noise
        expected, image = draw_array(200, 2, noise_level=1, ndim=ndim)

        columns = default_pos_columns(ndim)

        # Locate the features and calculate the deviation
        features = tp.locate(image, diameter=5)
        _, actual = sort_positions(features[columns].values, expected)
        deviation = np.sqrt(np.mean(np.sum((actual - expected) ** 2, 1)))

        # Apply SPIFF correction and calculate the deviation
        corrected_features = apply_spiff_correction(features)
        _, corrected = sort_positions(corrected_features[columns].values, expected)
        deviation_corrected = np.sqrt(np.mean(np.sum((corrected - expected) ** 2, 1)))

        # Verify that the SPIFF correction improves accuracy
        assert (deviation_corrected < deviation)

        # Verify that the SPIFF corrected features have less subpixel bias
        hist_dev = np.std(
            np.histogram(features[columns] % 1, bins=10, range=(0, 1))[0]
        )
        corrected_hist_dev = np.std(
            np.histogram(corrected_features[columns] % 1, bins=10, range=(0, 1))[0]
        )
        assert (corrected_hist_dev < hist_dev)

    def test_spiff_2d(self):
        self._test_spiff(ndim=2)
    def test_spiff_3d(self):
        self._test_spiff(ndim=3)

if __name__ == '__main__':
    unittest.main()
