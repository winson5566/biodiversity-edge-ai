# Embedded machine learning classroom demo

This case study turns the original biodiversity system into one computer-vision-centred embedded ML workflow. It can be taught as a 60--90 minute lecture plus a practical lab.

## Learning outcomes

Students should be able to:

1. explain the difference between training-time and deployment-time pipelines;
2. compare FP32, dynamic-range quantization, full-integer INT8, and magnitude pruning;
3. fuse an image classifier with a small spatio-temporal prior;
4. measure accuracy, model size, latency, memory, and energy on target hardware;
5. justify an embedded model choice from evidence rather than accuracy alone.

## Lecture sequence

### 1. Problem and constraints (10 minutes)

Start with offline species recognition on a Raspberry Pi Zero 2 W. Show why a server-only solution is unsuitable when connectivity, response time, privacy, and power matter. Introduce the 10,000-class iNaturalist system as the original project and a smaller class subset as the classroom dataset.

### 2. Vision baseline (15 minutes)

Trace one image through resizing, preprocessing, CNN inference, softmax scores, and Top-K output. Train a small MobileNetV2 model with `scripts/train_vision.py`, keeping the class map and data split fixed for all later comparisons.

Questions for students:

- Why can a desktop-accurate model fail on the Pi?
- Which preprocessing assumptions must travel with the model?
- Why must test data remain untouched during model selection?

### 3. Geo Prior and fusion (15 minutes)

Encode latitude, longitude, and day of year as six cyclic features. Compare vision-only output with log-linear fusion:

```text
P(class | image, geo) proportional to
P(class | image)^alpha * P(class | geo)^(1-alpha)
```

Use validation data to choose `alpha`; evaluate that choice once on the test set. Demonstrate the vision-only fallback by removing location metadata. Point out that live GPS was not completed in the original device program: fixed coordinates are enough to teach the model and fusion path without claiming otherwise.

### 4. Quantization and pruning (20 minutes)

Export the same baseline as FP32, DRQ, and full INT8. Then create new 30%, 50%, and 70% magnitude-pruned variants and optionally combine pruning with INT8.

Keep the provenance explicit:

- FP32, DRQ, Geo Prior fusion, and Pi deployment belong to the original project.
- Full-integer activation quantization and pruning are new teaching extensions.

Ask students to predict the outcome before measuring it. In particular, zeros in a dense tensor do not guarantee reduced TFLite size or latency.

### 5. Device evidence and decision (15 minutes)

Run `scripts/benchmark_rpi.py` on the same test images, thread count, warm-up count, and repetition count for every artifact. Complete a table containing:

| Variant | Accuracy | Vision bytes | Geo bytes | Invoke median | End-to-end P95 | RSS delta | Energy/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 baseline | | | | | | | |
| DRQ | | | | | | | |
| Full INT8 | | | | | | | |
| Pruned 30% | | | | | | | |
| Pruned 50% | | | | | | | |
| Pruned 70% | | | | | | | |
| Pruned 50% + INT8 | | | | | | | |

End with a deployment recommendation and one explicit trade-off, such as a small accuracy loss for a large reduction in storage or latency.

## Practical lab deliverables

Each student group submits:

1. the fixed class map and split description;
2. manifests for every exported model;
3. raw JSON benchmark outputs from the Pi;
4. one completed trade-off table;
5. a short recommendation supported by device measurements;
6. a statement separating reproduced original results from newly generated extension results.

## Recommended live-demo path

Use a 5--10 class image subset and a 128x128 MobileNetV2 for training during class. Bring pre-exported 10,000-class legacy models to demonstrate the scale of the original project. Run the expensive full-dataset results as reported evidence, not as a live reproduction.
