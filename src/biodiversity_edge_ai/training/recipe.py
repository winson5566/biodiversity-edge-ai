"""Three-stage transfer-learning settings and batch-wise learning rates."""

import math


def stages(args):
    """Stage 1: head; stage 2: all layers; stage 3: higher-resolution tail."""
    return [
        {"name": name, "epochs": epochs, "input_size": size,
         "learning_rate": rate, "unfreeze": unfreeze,
         "augment": args.randaug_layers > 0 and name != "resolution"}
        for name, epochs, size, rate, unfreeze in (
            ("head", args.head_epochs, args.input_size, args.head_learning_rate, 0),
            ("finetune", args.finetune_epochs, args.input_size, args.finetune_learning_rate, -1),
            ("resolution", args.resolution_epochs, args.resolution_input_size,
             args.resolution_learning_rate, args.unfreeze_layers),
        ) if epochs > 0
    ]


def learning_rate_at_step(initial, step, decay_steps, warmup_steps=0, cosine=True):
    """One-based batch steps; warmup is separate from the cosine decay interval."""
    if step < warmup_steps:
        return initial * step / warmup_steps
    if not cosine:
        return initial
    progress = min(max(step - warmup_steps, 0), decay_steps) / max(decay_steps, 1)
    return initial * 0.5 * (1 + math.cos(math.pi * progress))


def configure_trainable_layers(model, unfreeze):
    base = model.layers[1]
    base.trainable = unfreeze != 0
    if unfreeze > 0:
        for layer in base.layers:
            layer.trainable = False
        for layer in base.layers[-unfreeze:]:
            layer.trainable = True
    return base
