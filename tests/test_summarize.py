import json
import tempfile
import unittest
from pathlib import Path

from biodiversity_edge_ai.evaluation.summarize import FIELDS, load_results, write_csv, write_markdown


class SummarizeTests(unittest.TestCase):
    def test_writes_comparable_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = {field: 1 for field in FIELDS}
            result["vision_model_id"] = "vision_drq"
            result["vision_optimization"] = "drq"
            source = root / "result.json"
            source.write_text(json.dumps(result), encoding="utf-8")
            loaded = load_results([str(source)])
            write_csv(root / "tradeoffs.csv", loaded)
            write_markdown(root / "tradeoffs.md", loaded)
            self.assertIn("vision_drq", (root / "tradeoffs.csv").read_text())
            self.assertIn("| vision_drq | drq |", (root / "tradeoffs.md").read_text())


if __name__ == "__main__":
    unittest.main()
