import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from biodiversity_edge_ai.data.prepare import prepare_dataset
from biodiversity_edge_ai.training.vision import load_prepared_classes


class PrepareDataTests(unittest.TestCase):
    def _source_dataset(self, root: Path) -> tuple[Path, Path]:
        images_root = root / "raw"
        images = []
        annotations = []
        image_id = 100
        for category_id in (7, 42):
            for index in range(6):
                relative = Path("source") / f"{category_id}_{index}.jpg"
                destination = images_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 6), color=(category_id, index, 20)).save(destination)
                row = {
                    "id": image_id,
                    "file_name": relative.as_posix(),
                    "latitude": -43.5,
                    "longitude": 172.6,
                    "date": "2025-01-02",
                }
                if category_id == 42 and index == 0:
                    row["latitude"] = None
                images.append(row)
                annotations.append(
                    {"id": image_id, "image_id": image_id, "category_id": category_id}
                )
                image_id += 1
        source = {
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": 7, "name": "Species seven", "common_name": "Seven"},
                {"id": 42, "name": "Species forty-two"},
            ],
        }
        annotation_path = root / "annotations.json"
        annotation_path.write_text(json.dumps(source), encoding="utf-8")
        return annotation_path, images_root

    def _args(self, annotations: Path, images_root: Path, output: Path) -> argparse.Namespace:
        return argparse.Namespace(
            annotations=str(annotations),
            images_root=str(images_root),
            output=str(output),
            category_ids=None,
            num_classes=2,
            min_per_class=3,
            max_per_class=None,
            val_fraction=0.2,
            test_fraction=0.2,
            seed=42,
            transfer_mode="symlink",
            verify_images=True,
        )

    def test_prepares_aligned_image_and_metadata_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations, images_root = self._source_dataset(root)
            output = root / "prepared"
            manifest = prepare_dataset(self._args(annotations, images_root, output))

            self.assertEqual(manifest["counts"], {"train": 8, "val": 2, "test": 2})
            self.assertEqual(json.loads((output / "class_map.json").read_text()), ["Seven", "Species forty-two"])
            directories, names = load_prepared_classes(output)
            self.assertEqual(directories, ["00000", "00001"])
            self.assertEqual(names, ["Seven", "Species forty-two"])

            image_ids = set()
            for split in ("train", "val", "test"):
                with (output / "metadata" / f"{split}.csv").open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                for row in rows:
                    self.assertNotIn(row["image_id"], image_ids)
                    image_ids.add(row["image_id"])
                    prepared_image = output / "images" / split / row["filename"]
                    self.assertTrue(prepared_image.is_symlink())
                    self.assertTrue(prepared_image.is_file())
            self.assertEqual(len(image_ids), 12)

    def test_refuses_to_mix_with_an_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations, images_root = self._source_dataset(root)
            output = root / "prepared"
            output.mkdir()
            (output / "keep.txt").write_text("do not overwrite", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_dataset(self._args(annotations, images_root, output))


if __name__ == "__main__":
    unittest.main()
