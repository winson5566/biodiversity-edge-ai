# End-to-end smoke test

The complete small-data workflow was executed on 8 September 2026 using Python 3.12.8, NumPy 1.26.4, TensorFlow 2.16.2, Keras 3.8.0, and an Apple M4 workstation.

Run the same test with:

```bash
make smoke
```

The smoke source contains two synthetic classes with twelve 40x40 images per class. ImageNet weights are disabled, and each training phase runs for one epoch. The command exercises the software path; its accuracy and timing are not scientific model results.

## Verified stages

1. Generated iNaturalist-style source JSON and 24 images.
2. Prepared deterministic 16/4/4 train/validation/test splits.
3. Preserved the shared two-class mapping across image and geographic inputs.
4. Trained and saved the MobileNetV2 vision model.
5. Trained, validated, and saved the six-feature Geo Prior.
6. Exported vision FP32, dynamic-range quantized, and full-integer INT8 TFLite models.
7. Applied 50% magnitude pruning, fine-tuned with fixed masks, and exported the pruned model with dynamic-range quantization.
8. Exported the Geo Prior to TFLite.
9. Loaded every TFLite artifact through the integrated runtime and ran geo-fused predictions on the held-out test images.
10. Generated JSON results plus a combined CSV and Markdown trade-off table.

## Smoke-test output

| Model | Optimization | Input | Size | Test Top-1 | End-to-end median |
|---|---|---|---:|---:|---:|
| vision_fp32 | FP32 | float32 | 2,761,512 bytes | 50% | 0.208 ms |
| vision_drq | DRQ | float32 | 870,752 bytes | 50% | 0.202 ms |
| vision_int8 | full INT8 | int8 | 973,752 bytes | 50% | 0.173 ms |
| vision_pruned_50_drq | pruning + DRQ | float32 | 862,096 bytes | 50% | 0.265 ms |

The common Geo Prior artifact was 15,392 bytes. These timings come from a four-image workstation run with one warm-up and one repetition; use the Raspberry Pi benchmark settings for deployment conclusions.

The saved pruned Keras model was inspected after fine-tuning: 335,848 of 671,696 eligible kernel weights were zero, giving exactly 50% kernel sparsity.

Generated smoke artifacts are intentionally ignored by Git under `data/smoke_*` and `artifacts/smoke/`.

## Compatibility issue found and fixed

Direct conversion of a Keras 3 `.keras` model through TensorFlow 2.16 relied on a removed private method. The exporter now writes a public SavedModel export boundary first and converts from that representation, supporting the tested Keras 3 path without private APIs.
