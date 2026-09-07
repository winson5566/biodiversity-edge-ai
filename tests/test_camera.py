import contextlib
import io
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from biodiversity_edge_ai.device.rpi_camera import main
from biodiversity_edge_ai.pipeline import Prediction


class CameraTests(unittest.TestCase):
    def test_camera_uses_rgb_array_format_and_closes(self):
        camera = MagicMock()
        camera.capture_array.return_value = np.full((32, 32, 3), [255, 0, 0], dtype=np.uint8)
        pipeline = SimpleNamespace(geo_runner=None, predict=MagicMock(
            return_value=[Prediction(0, "red", 1.0)]))
        with patch.dict(sys.modules, {"picamera2": SimpleNamespace(Picamera2=lambda: camera)}), \
             patch("biodiversity_edge_ai.device.rpi_camera.BiodiversityPipeline.from_files",
                   return_value=pipeline), \
             patch.object(sys, "argv", ["camera", "--vision-model", "model", "--vision-manifest",
                                        "manifest", "--class-map", "map", "--warmup-seconds", "0"]), \
             contextlib.redirect_stdout(io.StringIO()):
            main()
        self.assertEqual(camera.create_still_configuration.call_args.kwargs["main"]["format"],
                         "BGR888")
        self.assertEqual(pipeline.predict.call_args.args[0].getpixel((0, 0)), (255, 0, 0))
        camera.stop.assert_called_once()
        camera.close.assert_called_once()
