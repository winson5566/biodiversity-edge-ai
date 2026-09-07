# Performance evaluation

Model selection is based on measured predictive quality and device cost. Do not infer deployment performance from parameter count, quantization mode, or sparsity alone.

## Experiment matrix

Run every variant on the same held-out image set and Raspberry Pi configuration.

| Vision variant | Top-1 accuracy | Model bytes | Invoke median | End-to-end P95 | RSS delta | Energy/image |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | TBD | TBD | TBD | TBD | TBD | TBD |
| DRQ | TBD | TBD | TBD | TBD | TBD | TBD |
| Full INT8 | TBD | TBD | TBD | TBD | TBD | TBD |
| Pruned 50% + DRQ | TBD | TBD | TBD | TBD | TBD | TBD |

Optional experiments may add 30% and 70% pruning targets or combine pruning with full INT8 quantization.

## Controlled comparison

Keep the following fixed across runs:

- test split and class map;
- image preprocessing and input resolution;
- Geo Prior artifact and fusion weight;
- Raspberry Pi model, OS image, cooling, power supply, and CPU governor;
- TFLite thread count;
- warm-up and measured repetition counts.

Record model identifiers, manifests, raw JSON benchmark outputs, ambient conditions, and power-meter method with the result table.

## Decision rule

Choose a deployment model by declaring the acceptable accuracy loss and the constrained resource. For example, prefer the smallest model that stays within an agreed accuracy margin and meets the P95 latency limit. A pruned model is beneficial only when the exported artifact or target runtime realizes the sparsity advantage.
