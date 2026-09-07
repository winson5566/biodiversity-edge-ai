"""Opt-in workstation tests: BIODIVERSITY_TF_TESTS=1 make test."""

import gc
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from biodiversity_edge_ai.inference import preprocess_image
from biodiversity_edge_ai.manifest import ModelManifest
from biodiversity_edge_ai.models.catalog import BACKBONES
from biodiversity_edge_ai.models.vision import build_vision_model
from biodiversity_edge_ai.training.vision import image_dataset
from biodiversity_edge_ai.training.recipe import configure_trainable_layers


@unittest.skipUnless(os.environ.get("BIODIVERSITY_TF_TESTS") == "1", "requires opt-in TensorFlow tests")
class TensorFlowContractTests(unittest.TestCase):
    def test_stage_freezing_and_resolution_weight_transfer(self):
        import tensorflow as tf
        tf.keras.backend.clear_session()
        model = build_vision_model(backbone="mobilenet-v2", num_classes=2, input_size=32,
                                   weights=None, trainable=False)
        base = configure_trainable_layers(model, 0)
        self.assertEqual(len(base.trainable_variables), 0)
        configure_trainable_layers(model, -1)
        full_count = len(base.trainable_variables)
        configure_trainable_layers(model, 18)
        self.assertGreater(len(base.trainable_variables), 0)
        self.assertLess(len(base.trainable_variables), full_count)
        self.assertTrue(all(not layer.trainable for layer in base.layers[:-18]))
        larger = build_vision_model(backbone="mobilenet-v2", num_classes=2, input_size=40,
                                    weights=None, trainable=False)
        larger.set_weights(model.get_weights())
        for before, after in zip(model.get_weights(), larger.get_weights()):
            np.testing.assert_array_equal(before, after)
        self.assertEqual(larger.input_shape[1:3], (40, 40))

    def test_augmented_dataset_runs_in_graph_mode(self):
        import tensorflow as tf
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "a").mkdir()
            pixels = np.random.default_rng(42).integers(0, 256, (43, 65, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(directory / "a/image.png")
            for scale in ("minus1_1", "caffe", "0_255"):
                info = ModelManifest("test", "vision", "keras", 1, "test", [32, 32, 3],
                                     "float32", scale)
                dataset = image_dataset(tf, directory, ["a"], info, 1, 42, True,
                                        randaug_layers=2, randaug_magnitude=2, categorical=True)
                values, labels = next(iter(dataset))
                self.assertEqual(tuple(values.shape), (1, 32, 32, 3))
                self.assertTrue(np.isfinite(values.numpy()).all())
                np.testing.assert_array_equal(labels.numpy(), [[1.]])

    def test_all_backbones_build_and_predict(self):
        import tensorflow as tf
        for name, scale in BACKBONES.items():
            with self.subTest(backbone=name):
                tf.keras.backend.clear_session()
                model = build_vision_model(backbone=name, num_classes=2, input_size=32,
                                           weights=None, trainable=False)
                info = ModelManifest(name, "vision", "keras", 2, "test", [32, 32, 3],
                                     "float32", scale)
                values = preprocess_image(Image.new("RGB", (47, 35), (40, 100, 200)), info)
                result = model(values[None], training=False).numpy()
                self.assertEqual(result.shape, (1, 2))
                self.assertTrue(np.isfinite(result).all())
                np.testing.assert_allclose(result.sum(), 1, rtol=1e-5)
                del model
                gc.collect()

    def test_training_pixels_equal_device_pixels(self):
        import tensorflow as tf
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "a").mkdir()
            values = np.random.default_rng(42).integers(0, 256, (43, 65, 3), dtype=np.uint8)
            path = directory / "a/image.png"
            Image.fromarray(values).save(path)
            for scale in set(BACKBONES.values()):
                info = ModelManifest("test", "vision", "keras", 1, "test", [32, 32, 3],
                                     "float32", scale)
                dataset = image_dataset(tf, directory, ["a"], info, 1, 42, False)
                actual, labels = next(iter(dataset))
                with Image.open(path) as image:
                    expected = preprocess_image(image, info)
                np.testing.assert_array_equal(actual.numpy()[0], expected)
                np.testing.assert_array_equal(labels.numpy(), [0])
