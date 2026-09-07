"""Input contracts for the seven supported Keras application backbones."""

BACKBONES = {
    "mobilenet-v2": "minus1_1",
    "mobilenet-v3-large": "0_255",
    "efficientnet-b0": "0_255",
    "resnet-50": "caffe",
    "resnet-101": "caffe",
    "convnext-tiny": "0_255",
    "convnext-small": "0_255",
}


def input_scale_for(backbone: str) -> str:
    try:
        return BACKBONES[backbone]
    except KeyError as exc:
        raise ValueError(f"unsupported backbone: {backbone}") from exc
