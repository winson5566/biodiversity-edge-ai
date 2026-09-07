import tempfile
import unittest
from pathlib import Path

from biodiversity_edge_ai.manifest import (
    ModelManifest,
    class_map_sha256,
    validate_fusion_compatibility,
)


class ManifestTests(unittest.TestCase):
    def make_manifest(self, role: str, digest: str) -> ModelManifest:
        return ModelManifest(
            model_id=role,
            role=role,
            format="tflite",
            num_classes=2,
            class_map_sha256=digest,
            input_shape=[2, 2, 3] if role == "vision" else [6],
            input_dtype="float32",
            input_scale="0_255" if role == "vision" else "encoded_geo",
        )

    def test_round_trip_and_compatibility(self):
        digest = class_map_sha256(["a", "b"])
        vision = self.make_manifest("vision", digest)
        geo = self.make_manifest("geo_prior", digest)
        validate_fusion_compatibility(vision, geo)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            vision.save(path)
            self.assertEqual(ModelManifest.load(path), vision)

    def test_class_map_mismatch_is_rejected(self):
        vision = self.make_manifest("vision", class_map_sha256(["a", "b"]))
        geo = self.make_manifest("geo_prior", class_map_sha256(["b", "a"]))
        with self.assertRaisesRegex(ValueError, "class-map hashes differ"):
            validate_fusion_compatibility(vision, geo)


if __name__ == "__main__":
    unittest.main()
