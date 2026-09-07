"""Capture one Raspberry Pi camera frame and run the integrated pipeline."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time

from PIL import Image

from biodiversity_edge_ai.pipeline import BiodiversityPipeline
from biodiversity_edge_ai.device.display import WaveshareST7789Display, render_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-model", required=True)
    parser.add_argument("--vision-manifest", required=True)
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--geo-model")
    parser.add_argument("--geo-manifest")
    parser.add_argument("--latitude", type=float)
    parser.add_argument("--longitude", type=float)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--save-image")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--brightness", type=int, default=50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        from picamera2 import Picamera2
    except ImportError as exc:
        raise SystemExit("picamera2 is required on Raspberry Pi") from exc
    pipeline = BiodiversityPipeline.from_files(
        vision_model=args.vision_model,
        vision_manifest=args.vision_manifest,
        class_map=args.class_map,
        geo_model=args.geo_model,
        geo_manifest=args.geo_manifest,
        num_threads=args.threads,
        alpha=args.alpha,
    )
    camera = Picamera2()
    camera.configure(
        camera.create_still_configuration(
            # Picamera2's BGR888 produces RGB channel order in capture_array.
            main={"size": (args.camera_width, args.camera_height), "format": "BGR888"}
        )
    )
    camera.start()
    try:
        time.sleep(args.warmup_seconds)
        image = Image.fromarray(camera.capture_array())
    finally:
        camera.stop()
        camera.close()
    if args.save_image:
        image.save(args.save_image)
    location_used = (
        pipeline.geo_runner is not None
        and args.latitude is not None
        and args.longitude is not None
    )
    predictions = pipeline.predict(
        image,
        latitude=args.latitude,
        longitude=args.longitude,
        observation_date=(dt.date.today() if location_used else None),
        k=5,
    )
    print(json.dumps([prediction.__dict__ for prediction in predictions], indent=2))
    if args.display:
        inference_ms = float(
            getattr(pipeline.vision_runner, "last_inference_ms", 0.0)
        )
        if location_used:
            inference_ms += float(
                getattr(pipeline.geo_runner, "last_inference_ms", 0.0)
            )
        display = WaveshareST7789Display(brightness=args.brightness)
        try:
            display.show(
                render_result(
                    image,
                    predictions[0],
                    inference_ms=inference_ms,
                    location_used=location_used,
                )
            )
            input("Result shown on ST7789. Press Enter to close. ")
        finally:
            display.close()


if __name__ == "__main__":
    main()
