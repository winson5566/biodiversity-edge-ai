import hashlib
from pathlib import Path
import unittest

import numpy as np

from biodiversity_edge_ai.config import Experiment
from biodiversity_edge_ai.training.geo_prior import balanced_epoch_indices
from biodiversity_edge_ai.training.recipe import learning_rate_at_step, stages
from biodiversity_edge_ai.workflow import Workflow


class TrainingRecipeTests(unittest.TestCase):
    def test_recovered_profiles_match_saved_flags(self):
        recovered = 0
        mapping = {
            "head_epochs": "epochs_stage1", "finetune_epochs": "epochs_stage2",
            "resolution_epochs": "epochs_stage3", "input_size": "input_size",
            "resolution_input_size": "input_size_stage3", "batch_size": "batch_size",
            "head_learning_rate": "lr_stage1", "finetune_learning_rate": "lr_stage2",
            "resolution_learning_rate": "lr_stage3", "unfreeze_layers": "unfreeze_layers",
            "momentum": "momentum", "label_smoothing": "label_smoothing",
            "randaug_layers": "randaug_num_layers", "randaug_magnitude": "randaug_magnitude",
            "num_classes": "num_classes", "seed": "random_seed",
        }
        for path in Path("configs/models").glob("*.json"):
            config = Experiment.load(path)
            if config.recipe_status != "recovered":
                continue
            recovered += 1
            reference = Path(config.parameter_source["vision_config"])
            self.assertEqual(hashlib.sha256(reference.read_bytes()).hexdigest(),
                             config.parameter_source["vision_config_sha256"])
            flags = dict(line.removeprefix("--").split("=", 1)
                         for line in reference.read_text().splitlines())
            for field, flag in mapping.items():
                with self.subTest(model=config.backbone, field=field):
                    self.assertEqual(getattr(config, field), float(flags[flag]))
            self.assertEqual(config.optimizer, "sgd")
            self.assertTrue(config.scale_learning_rate)
            self.assertTrue(config.cosine_decay)
            self.assertEqual(config.lr_warmup_epochs, 0.3)
            self.assertEqual(config.geo_batch_size, 1024)
            self.assertEqual(config.geo_lr_decay, 0.98)
            self.assertEqual(config.geo_max_per_class, 100)
        self.assertEqual(recovered, 6)

    def test_stage_plan_keeps_continuation_only_recipe(self):
        config = Experiment.load("configs/models/mobilenet-v2.json")
        plan = stages(config)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["name"], "resolution")
        self.assertEqual(plan[0]["epochs"], 2)
        self.assertFalse(plan[0]["augment"])
        self.assertTrue(config.requires_initial_weights)

    def test_missing_recipe_and_checkpoint_fail_before_writing(self):
        for model, message in (("convnext-tiny", "recipe is missing"),
                               ("mobilenet-v2", "continuation-only")):
            config = Experiment.load(f"configs/models/{model}.json")
            workflow = Workflow(config, f"missing-artifact-test-{model}")
            with self.assertRaisesRegex(ValueError, message):
                workflow.execute()
            self.assertFalse(workflow.root.exists())

    def test_scaled_warmup_and_cosine_match_three_stage_schedule(self):
        initial = 0.1 * 32 / 256
        self.assertEqual(initial, 0.0125)
        self.assertAlmostEqual(learning_rate_at_step(initial, 1, 100, 10), initial / 10)
        self.assertAlmostEqual(learning_rate_at_step(initial, 10, 100, 10), initial)
        self.assertAlmostEqual(learning_rate_at_step(initial, 60, 100, 10), initial / 2)
        self.assertEqual(learning_rate_at_step(initial, 110, 100, 10), 0)
        self.assertEqual(learning_rate_at_step(initial, 1000, 100, 10, False), initial)

    def test_balanced_geo_sampling_preserves_capped_class_weights(self):
        labels = np.repeat([0, 1, 2], [5, 15, 100])
        a = balanced_epoch_indices(labels, 10, np.random.default_rng(42))
        b = balanced_epoch_indices(labels, 10, np.random.default_rng(42))
        self.assertEqual(len(a), 25)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all((a >= 0) & (a < len(labels))))
        rng = np.random.default_rng(42)
        totals = sum(np.bincount(labels[balanced_epoch_indices(labels, 10, rng)], minlength=3)
                     for _ in range(1000))
        np.testing.assert_allclose(totals / totals.sum(), [0.2, 0.4, 0.4], atol=0.02)
        full = balanced_epoch_indices(labels, -1, rng)
        np.testing.assert_array_equal(np.sort(full), np.arange(len(labels)))
