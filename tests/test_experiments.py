import contextlib
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from biodiversity_edge_ai.config import Experiment
from biodiversity_edge_ai.models.catalog import BACKBONES
from biodiversity_edge_ai.workflow import Workflow


class ExperimentTests(unittest.TestCase):
    def test_presets_are_executable_and_smoke_needs_no_download(self):
        for path in Path("configs").glob("*.json"):
            Experiment.load(path)
        smoke = Experiment.load("configs/smoke.json")
        self.assertTrue(smoke.synthetic)
        self.assertEqual(smoke.weights, "none")

    def test_backbones_have_separate_default_outputs(self):
        outputs = {Workflow(Experiment(backbone=name)).root for name in BACKBONES}
        self.assertEqual(len(outputs), 7)

    def test_unknown_keys_and_invalid_configuration_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text(json.dumps({"backbon": "resnet-50"}))
            with self.assertRaisesRegex(ValueError, "unknown configuration"):
                Experiment.load(path)
        for config in (Experiment(threads=0), Experiment(formats=("fp32", "fp32")),
                       Experiment(backbone="missing"), Experiment(input_size=16)):
            with self.assertRaises(ValueError):
                config.validate()

    def test_dry_run_does_not_write_and_includes_both_evaluations(self):
        output = io.StringIO()
        workflow = Workflow(Experiment(), "dry-run-test", dry_run=True)
        with contextlib.redirect_stdout(output):
            workflow.execute()
        self.assertIn("vision_fp32.json", output.getvalue())
        self.assertIn("fused_fp32.json", output.getvalue())
        self.assertNotIn("--input-scale minus1_1", output.getvalue())
        self.assertFalse(workflow.root.exists())

    def test_configuration_collision_and_tampered_outputs_are_rejected(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            try:
                os.chdir(temp)
                workflow = Workflow(Experiment(), "test")
                workflow.initialize()
                with self.assertRaisesRegex(ValueError, "new --run"):
                    Workflow(replace(Experiment(), backbone="resnet-50"), "test").initialize()
                artifact = workflow.models / "dummy.txt"

                def create(*args, **kwargs):
                    artifact.write_text("complete")
                    return type("Result", (), {"returncode": 0})()

                with patch("biodiversity_edge_ai.workflow.subprocess.run", side_effect=create) as run:
                    workflow.step("dummy", "unused", [], [artifact])
                    workflow.step("dummy", "unused", [], [artifact])
                    self.assertEqual(run.call_count, 1)
                    artifact.write_text("changed")
                    with self.assertRaisesRegex(ValueError, "changed"):
                        workflow.step("dummy", "unused", [], [artifact])
            finally:
                os.chdir(original)


if __name__ == "__main__":
    unittest.main()
