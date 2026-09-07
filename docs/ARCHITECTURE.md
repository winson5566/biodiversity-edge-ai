# System architecture

## Package map

| Capability | Implementation |
|---|---|
| Dataset preparation and class mapping | `scripts/prepare_data.py`, `pipeline.py` |
| Vision model and training | `models/vision.py`, `training/vision.py` |
| Geo Prior and metadata encoding | `metadata.py`, `models/geo_prior.py`, `training/geo_prior.py` |
| TFLite export and model manifests | `export/tflite.py`, `manifest.py` |
| Quantization and magnitude pruning | `export/tflite.py`, `training/pruning.py` |
| Vision/Geo Prior fusion | `fusion.py`, `inference.py` |
| Benchmarking and trade-off summaries | `evaluation/benchmark.py`, `scripts/summarize_benchmarks.py` |
| Raspberry Pi camera and display | `device/rpi_camera.py`, `device/display.py` |

## Inference flow

The vision model receives one resized RGB image and produces a probability vector over the shared class map. The Geo Prior receives six cyclic latitude, longitude, and date features and produces a second probability vector. Log-linear fusion combines the two distributions:

```text
fused = normalize(vision ** alpha * geo ** (1 - alpha))
```

Select `alpha` using validation data only. Evaluate the selected configuration once on held-out test data. When location or date is absent, the pipeline returns the vision distribution without fusion.

## Artifact compatibility

Every exported model has a JSON manifest. The inference pipeline validates these invariants before running:

1. the vision and Geo Prior output dimensions equal the class-map length;
2. both manifests contain the same class-map SHA-256 hash;
3. the manifest declares the input shape, data type, scaling, role, and optimization mode;
4. image preprocessing is identical during training, evaluation, and device inference;
5. Geo Prior features use the fixed order: sine/cosine longitude, sine/cosine latitude, sine/cosine date.

An incompatible model fails explicitly instead of producing a silently misaligned prediction.

## Training and deployment environments

The workstation environment uses TensorFlow for model training, fine-tuning, pruning, and TFLite conversion. The Raspberry Pi runtime loads only TFLite artifacts and the class map, avoiding a full TensorFlow installation on the device.

The deployable bundle should contain:

```text
class_map.json
vision_model.tflite
vision_model.tflite.manifest.json
geo_prior.tflite                 # optional
geo_prior.tflite.manifest.json   # optional
```

## Measurement boundary

Desktop evaluation measures predictive quality and verifies model conversion. Raspberry Pi benchmarking measures deployment behavior: model size, invocation latency, end-to-end latency, memory, and optionally energy per image. Use the same held-out images, thread count, warm-up count, and repetition count for every model variant.
