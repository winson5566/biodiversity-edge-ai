import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from biodiversity_edge_ai.export.tflite import build_parser, export_model
from biodiversity_edge_ai.manifest import ModelManifest, class_map_sha256


class ExportContractTests(unittest.TestCase):
    def test_mismatched_class_order_or_pixel_scaling_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            classes = root / "classes.json"
            classes.write_text(json.dumps(["A", "B"]))
            keras_path = root / "vision.keras"
            contract = ModelManifest(
                "efficientnet-b0", "vision", "keras", 2, class_map_sha256(["A", "B"]),
                [32, 32, 3], "float32", "0_255")
            contract.save(f"{keras_path}.manifest.json")
            common = ["--keras-model", str(keras_path), "--output", str(root / "model.tflite"),
                      "--class-map", str(classes), "--model-id", "test", "--role", "vision"]
            with patch("biodiversity_edge_ai.export.tflite._tensorflow") as tensorflow:
                with self.assertRaisesRegex(ValueError, "scaling differs"):
                    export_model(build_parser().parse_args(common + ["--input-scale", "minus1_1"]))
                classes.write_text(json.dumps(["B", "A"]))
                with self.assertRaisesRegex(ValueError, "class map differs"):
                    export_model(build_parser().parse_args(common))
                tensorflow.return_value.keras.models.load_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
