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
        for path in Path("configs").rglob("*.json"):
            Experiment.load(path)
        smoke = Experiment.load("configs/smoke.json")
        self.assertTrue(smoke.synthetic)
        self.assertEqual(smoke.weights, "none")

    def test_backbones_have_separate_default_outputs(self):
        outputs = {Workflow(Experiment(backbone=name)).root for name in BACKBONES}
        self.assertEqual(len(outputs), 7)

    def test_each_model_has_a_complete_independent_profile(self):
        paths = list(Path("configs/models").glob("*.json"))
        self.assertEqual({p.stem for p in paths}, set(BACKBONES))
        for path in paths:
            with self.subTest(model=path.stem):
                config = Experiment.load(path)
                self.assertEqual(config.backbone, path.stem)
                self.assertEqual(set(json.loads(path.read_text())), set(config.to_dict()))
                self.assertEqual(config.annotations, "raw/inat2021/train_mini.json")
                full = Experiment.load(path, data_source="full")
                self.assertEqual(full.annotations, "raw/inat2021/train.json")
                self.assertEqual(full.name, "full")
                self.assertEqual(replace(full, name=config.name,
                                         annotations=config.annotations), config)

    def test_data_source_override_and_custom_paths(self):
        path = "configs/models/resnet-50.json"
        config = Experiment.load(path, data_source="full", images_root="/data")
        self.assertEqual(config.annotations, "/data/train.json")
        config = Experiment.load(path, data_source="full", annotations="custom.json")
        self.assertEqual(config.annotations, "custom.json")
        with self.assertRaisesRegex(ValueError, "synthetic"):
            Experiment.load("configs/smoke.json", data_source="full")
        with self.assertRaisesRegex(ValueError, "data_source"):
            Experiment.load(path, data_source="typo")
        self.assertEqual(Experiment.load("configs/full_system.json").name, "full")

    def test_learning_rates_are_validated(self):
        for key in ("head_learning_rate", "finetune_learning_rate", "geo_learning_rate"):
            for value in (0, -1, float("nan"), float("inf"), True, "0.01"):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    replace(Experiment(), **{key: value}).validate()
        with self.assertRaisesRegex(ValueError, "geo_batch_size"):
            Experiment(geo_batch_size=0).validate()

    def test_profile_training_parameters_reach_both_trainers(self):
        config = replace(Experiment.load("configs/models/convnext-small.json"),
                         head_learning_rate=0.002, finetune_learning_rate=0.0002,
                         geo_learning_rate=0.0006, geo_batch_size=11)
        workflow = Workflow(config, "dry-run-profile-test", dry_run=True)
        with patch.object(workflow, "step") as step, contextlib.redirect_stdout(io.StringIO()):
            workflow.execute("train")
        commands = {call.args[0]: list(map(str, call.args[2])) for call in step.call_args_list}
        vision, geo = commands["train-vision"], commands["train-geo"]
        for command, flag, expected in (
            (vision, "--backbone", "convnext-small"),
            (vision, "--batch-size", "8"),
            (vision, "--head-learning-rate", "0.002"),
            (vision, "--finetune-learning-rate", "0.0002"),
            (geo, "--batch-size", "11"), (geo, "--learning-rate", "0.0006"),
        ):
            self.assertEqual(command[command.index(flag) + 1], expected)

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
