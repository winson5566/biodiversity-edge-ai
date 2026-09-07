# Integration map

## Legacy sources

The originals remain read-only references. This repository replaces cross-repository imports with one `src/biodiversity_edge_ai` package.

| Legacy repository | Retained capability | New location |
|---|---|---|
| `inat2021ufam-master` | vision builders, staged training concepts, TFLite export, evaluation | `models/vision.py`, `training/vision.py`, `export/tflite.py`, `evaluation/benchmark.py` |
| `geo_prior_tf-master` | six-feature encoding, residual FCNet, presence-only training | `metadata.py`, `models/geo_prior.py`, `training/geo_prior.py` |
| `AI_Camera-main` | TFLite runtime, Pi camera capture, ST7789 display, latency testing | `inference.py`, `device/rpi_camera.py`, `device/display.py`, `evaluation/benchmark.py` |

The Waveshare `ST7789.py` and `config.py` files were vendored from the AI Camera project. The upstream permission notice remains in `device/waveshare/config.py`.

## Correctness boundaries

1. Image preprocessing is declared in every model manifest and shared by desktop and Pi inference.
2. Vision and geo-prior artifacts must have the same number of classes and class-map SHA-256 value.
3. Geo features retain the legacy order: sine/cosine longitude, sine/cosine latitude, sine/cosine date.
4. Fusion normalizes its output before presenting a value as confidence.
5. The fusion weight is selected on validation data, never on the final test set.
6. Missing location causes an explicit vision-only fallback.

## Environment split

The workstation environment uses TensorFlow for training and conversion. The Pi environment uses TFLite Runtime plus the OS-provided camera/GPIO libraries. The deployable geo-prior must therefore be exported to TFLite; loading the Keras FCNet on the Pi would reintroduce the full TensorFlow dependency.

## Known migration work

- Existing CNN `.tflite` files can be benchmarked after manifests are created.
- The legacy Geo Prior checkpoint must be loaded with its original architecture once, saved as a clean inference-only `.keras` model, then exported through `export/tflite.py`.
- The old live camera program was vision-only. The new `rpi_camera.py` accepts both model artifacts and applies fusion.
- Live L76K GPS input is not implemented yet. The integrated command accepts fixed coordinates and safely falls back when they are absent.
- `prepare_data.py` now converts the source annotation JSON into aligned image, Geo Prior, calibration, and benchmark inputs. Directory-based loading is used for practical-scale runs; the original sharded TFRecord pipeline is still not used for full-scale training.

## Legacy artifact commands

Create a clean inference-only Geo Prior model from the original checkpoint:

```bash
PYTHONPATH=src python scripts/migrate_geo_prior_checkpoint.py \
  --geo-repo /path/to/geo_prior_tf-master \
  --class-map artifacts/legacy/class_maps/inat2021_10000.json \
  --checkpoint artifacts/legacy/geo_prior_checkpoint/ckp.weights.h5 \
  --output artifacts/models/geo_prior_fp32.keras
```

After exporting a TFLite file, create or verify its manifest using the TFLite runtime installed on the target platform:

```bash
PYTHONPATH=src python scripts/create_legacy_manifest.py \
  --model artifacts/legacy/vision_models/model_efficientnet_b0_inat2021_drq.tflite \
  --class-map artifacts/legacy/class_maps/inat2021_10000.json \
  --output artifacts/models/efficientnet_b0_drq.manifest.json \
  --model-id efficientnet_b0_drq_legacy \
  --role vision --input-scale 0_255 --optimization drq
```
