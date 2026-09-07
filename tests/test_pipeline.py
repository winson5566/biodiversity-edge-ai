import unittest

import numpy as np
from PIL import Image

from biodiversity_edge_ai.manifest import ModelManifest, class_map_sha256
from biodiversity_edge_ai.pipeline import BiodiversityPipeline


class FakeRunner:
    def __init__(self, output):
        self.output = np.asarray(output, dtype=np.float32)
        self.calls = 0

    def predict(self, values):
        self.calls += 1
        return self.output


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.classes = ["species_a", "species_b"]
        digest = class_map_sha256(self.classes)
        self.vision_manifest = ModelManifest(
            model_id="vision",
            role="vision",
            format="tflite",
            num_classes=2,
            class_map_sha256=digest,
            input_shape=[2, 2, 3],
            input_dtype="float32",
            input_scale="0_255",
        )
        self.geo_manifest = ModelManifest(
            model_id="geo",
            role="geo_prior",
            format="tflite",
            num_classes=2,
            class_map_sha256=digest,
            input_shape=[6],
            input_dtype="float32",
            input_scale="encoded_geo",
        )
        self.image = Image.new("RGB", (4, 3), color=(20, 30, 40))

    def test_location_changes_prediction(self):
        pipeline = BiodiversityPipeline(
            vision_runner=FakeRunner([0.8, 0.2]),
            vision_manifest=self.vision_manifest,
            class_names=self.classes,
            geo_runner=FakeRunner([0.01, 0.99]),
            geo_manifest=self.geo_manifest,
            alpha=0.3,
        )
        vision_only = pipeline.predict(self.image, k=1)
        fused = pipeline.predict(
            self.image,
            latitude=-43.5,
            longitude=172.6,
            observation_date="2026-09-07",
            k=1,
        )
        self.assertEqual(vision_only[0].class_id, 0)
        self.assertEqual(fused[0].class_id, 1)

    def test_incomplete_location_does_not_invoke_geo_prior(self):
        geo_runner = FakeRunner([0.01, 0.99])
        pipeline = BiodiversityPipeline(
            vision_runner=FakeRunner([0.8, 0.2]),
            vision_manifest=self.vision_manifest,
            class_names=self.classes,
            geo_runner=geo_runner,
            geo_manifest=self.geo_manifest,
        )
        predictions = pipeline.predict(
            self.image,
            latitude=-43.5,
            longitude=172.6,
            observation_date=None,
            k=1,
        )
        self.assertEqual(predictions[0].class_id, 0)
        self.assertEqual(geo_runner.calls, 0)


if __name__ == "__main__":
    unittest.main()
