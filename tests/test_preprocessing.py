import unittest

import numpy as np
from PIL import Image

from biodiversity_edge_ai.inference import preprocess_image
from biodiversity_edge_ai.manifest import ModelManifest
from biodiversity_edge_ai.models.catalog import input_scale_for


class PreprocessingTests(unittest.TestCase):
    def preprocess(self, backbone):
        info = ModelManifest("test", "vision", "tflite", 2, "test", [32, 32, 3],
                             "float32", input_scale_for(backbone))
        return preprocess_image(Image.new("RGB", (64, 48), (10, 20, 30)), info)

    def test_raw_pixel_backbones_keep_rgb_values(self):
        for backbone in ("efficientnet-b0", "mobilenet-v3-large", "convnext-tiny", "convnext-small"):
            values = self.preprocess(backbone)
            np.testing.assert_allclose(values[0, 0], [10, 20, 30])
            self.assertEqual(values.shape, (32, 32, 3))
            self.assertEqual(values.dtype, np.float32)

    def test_resnet_uses_bgr_and_caffe_means(self):
        for name in ("resnet-50", "resnet-101"):
            np.testing.assert_allclose(self.preprocess(name)[0, 0],
                                       [30 - 103.939, 20 - 116.779, 10 - 123.68], rtol=1e-6)

    def test_mobilenet_v2_uses_minus_one_to_one(self):
        np.testing.assert_allclose(self.preprocess("mobilenet-v2")[0, 0],
                                   np.array([10, 20, 30]) / 127.5 - 1, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
