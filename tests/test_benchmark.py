import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from biodiversity_edge_ai.evaluation.benchmark import benchmark, build_parser
from biodiversity_edge_ai.pipeline import Prediction


class BenchmarkTests(unittest.TestCase):
    def test_top5_warmup_metadata_and_optional_energy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "images/a").mkdir(parents=True)
            Image.new("RGB", (32, 32)).save(root / "images/a/photo.png")
            (root / "metadata.csv").write_text(
                "filename,latitude,longitude,date,label_id\n"
                "a/photo.png,-43.5,172.6,2025-01-01,1\n")
            (root / "model.tflite").write_bytes(b"test")
            (root / "geo.tflite").write_bytes(b"test")
            calls = []

            def predict(image, **kwargs):
                calls.append(kwargs)
                return [Prediction(0, "A", .7), Prediction(1, "B", .3)]

            fake = SimpleNamespace(
                predict=predict, geo_runner=SimpleNamespace(last_inference_ms=1),
                vision_runner=SimpleNamespace(last_inference_ms=4),
                vision_manifest=SimpleNamespace(model_id="test", optimization="fp32"),
                geo_manifest=SimpleNamespace(model_id="geo", optimization="fp32"))
            args = build_parser().parse_args([
                "--images", str(root / "images"), "--metadata-csv", str(root / "metadata.csv"),
                "--vision-model", str(root / "model.tflite"), "--vision-manifest", "unused",
                "--geo-model", str(root / "geo.tflite"), "--geo-manifest", "unused",
                "--class-map", "unused", "--warmup", "1", "--net-power-w", "1.5",
                "--output", str(root / "result.json")])
            with patch("biodiversity_edge_ai.evaluation.benchmark.BiodiversityPipeline.from_files",
                       return_value=fake):
                result = benchmark(args)
            self.assertEqual(calls[0]["latitude"], -43.5)
            self.assertEqual(result["top1_accuracy"], 0)
            self.assertEqual(result["top5_accuracy"], 1)
            self.assertEqual(result["invoke_mean_ms"], 5)
            self.assertEqual(result["energy_per_inference_mj"], 7.5)
            self.assertEqual(json.loads((root / "result.json").read_text())["samples"], 1)


if __name__ == "__main__":
    unittest.main()
