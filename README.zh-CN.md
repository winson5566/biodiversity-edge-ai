# Biodiversity Edge AI

[English](README.md) · **简体中文**

一个可在树莓派离线运行的物种识别系统。它训练图像分类模型和时空 Geo Prior，导出 TFLite 模型，评估量化与剪枝，并支持相机推理。

## 快速开始

支持 Python 3.10–3.12，推荐 Python 3.12。

```bash
git clone https://github.com/winson5566/biodiversity-edge-ai.git
cd biodiversity-edge-ai
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,dev]'
make smoke
```

`make smoke` 无需下载数据集。它生成 24 张合成图片，训练两个模型，导出 FP32/DRQ/INT8，进行 50% 剪枝，并将对比表写入 `artifacts/smoke/results/tradeoffs.md`。

## 数据集

默认数据为 **iNaturalist 2021 Train Mini**：500,000 张图片、10,000 个物种。全量 Train 使用相同流程。

| 模式 | 下载量 | 运行命令 |
|---|---:|---|
| Mini（默认） | 42 GB 图片 + 45 MB 标注 | `make workstation` |
| 全量 Train | 224 GB 图片 + 221 MB 标注 | `make workstation DATA_SOURCE=full` |

下载默认 Mini、校验并解压到 `raw/inat2021`：

```bash
mkdir -p raw/inat2021 && cd raw/inat2021
curl -C - -O https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz
curl -C - -O https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz
md5sum train_mini.tar.gz train_mini.json.tar.gz
tar -xzf train_mini.tar.gz
tar -xzf train_mini.json.tar.gz
cd ../..
```

Mini 的 MD5 应为 `db6ed8330e634445efc8fec83ae81442` 和 `395a35be3651d86dc3b0d365b8ea5f92`。macOS 请将 `md5sum` 替换为 `md5`。

<details>
<summary>全部数据下载、哈希和 S3 地址</summary>

| 文件 | 大小 | MD5 | S3 地址 |
|---|---:|---|---|
| [Train Mini 图片](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz) | 42 GB | `db6ed8330e634445efc8fec83ae81442` | `s3://ml-inat-competition-datasets/2021/train_mini.tar.gz` |
| [Train Mini 标注](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz) | 45 MB | `395a35be3651d86dc3b0d365b8ea5f92` | `s3://ml-inat-competition-datasets/2021/train_mini.json.tar.gz` |
| [全量 Train 图片](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz) | 224 GB | `e0526d53c7f7b2e3167b2b43bb2690ed` | `s3://ml-inat-competition-datasets/2021/train.tar.gz` |
| [全量 Train 标注](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.json.tar.gz) | 221 MB | `38a7bb733f7a09214d44293460ec0021` | `s3://ml-inat-competition-datasets/2021/train.json.tar.gz` |
| [Validation 图片](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz) | 8.4 GB | `f6f6e0e242e3d4c9569ba56400938afc` | `s3://ml-inat-competition-datasets/2021/val.tar.gz` |
| [Validation 标注](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz) | 9.4 MB | `4d761e0f6a86cc63e8f7afc91f6a8f0b` | `s3://ml-inat-competition-datasets/2021/val.json.tar.gz` |
| [Public Test 图片](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/public_test.tar.gz) | 43 GB | `7124b949fe79bfa7f7019a15ef3dbd06` | `s3://ml-inat-competition-datasets/2021/public_test.tar.gz` |
| [Public Test 信息](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/public_test.json.tar.gz) | 21 MB | `7a9413db55c6fa452824469cc7dd9d3d` | `s3://ml-inat-competition-datasets/2021/public_test.json.tar.gz` |

图片均为最长边 500 像素的 JPEG。解压后分别产生 `train_mini/category/image.jpg`、`train/category/image.jpg` 或 `val/category/image.jpg`。下载前请阅读[官方数据说明与使用条款](https://github.com/visipedia/inat_comp/tree/master/2021)。

</details>

标准流程从所选训练源中确定性地生成 70%/15%/15% 的训练、验证、测试划分。`make workstation` 不会自动使用官方下载的 Validation 或 public Test。public Test 没有标签，不能计算本地准确率。

## 训练与导出

运行完整 Mini 流程：

```bash
make workstation
```

该命令完成数据准备、MobileNetV2 和六特征 Geo Prior 训练、TFLite 导出、视觉模型剪枝和基准对比。

小样本课堂演示：

```bash
make workstation NUM_CLASSES=10 MAX_PER_CLASS=50 \
  DATASET=data/prepared_demo \
  MODEL_DIR=artifacts/models_demo RESULT_DIR=artifacts/results_demo
```

| 阶段 | 命令 | 主要产物 |
|---|---|---|
| 数据准备 | `make prepare` | 固定划分、元数据 CSV、`class_map.json` |
| 训练 | `make train` | `vision_baseline.keras`、`geo_prior.keras` |
| 导出 | `make export` | FP32、DRQ、INT8 和 Geo Prior TFLite |
| 剪枝 | `make optimize` | 50% 剪枝后的 DRQ TFLite |
| 对比 | `make benchmark` | JSON、CSV 和 Markdown 折中表 |

## 系统架构

```mermaid
flowchart LR
  subgraph WS[工作站]
    RAW[图片 + 标签 + 地理元数据] --> PREP[准备固定划分]
    PREP --> VTRAIN[训练视觉模型]
    PREP --> GTRAIN[训练 Geo Prior]
    VTRAIN --> VEXPORT[导出并优化视觉 TFLite]
    GTRAIN --> GEXPORT[导出 Geo Prior TFLite]
  end

  subgraph ART[部署产物]
    MAP[class_map.json]
    VM[vision.tflite + manifest]
    GM[geo_prior.tflite + manifest]
  end

  subgraph PI[树莓派]
    IMAGE[相机画面或图片] --> VINF[视觉推理]
    META[纬度 + 经度 + 日期] --> GINF[Geo Prior 推理]
    VINF --> FUSE[验证产物并融合]
    GINF --> FUSE
    FUSE --> OUT[Top-K 物种预测]
  end

  VEXPORT --> VM
  GEXPORT --> GM
  MAP --> VINF
  MAP --> GINF
  VM --> VINF
  GM --> GINF
```

融合要求类别映射哈希和输出维度一致。缺少位置或日期时，系统使用纯视觉推理。

## 评测与部署

让标准模型使用相同的保留测试图片进行评测：

```bash
make benchmark
```

比较 FP32、DRQ、全 INT8 和剪枝 DRQ 的 Top-1、模型大小、推理耗时、端到端耗时、内存和能耗。各版本必须固定数据划分、预处理、Geo Prior、融合权重、树莓派配置、线程数、预热次数和重复次数。

识别单张图片：

```bash
PYTHONPATH=src python scripts/predict.py \
  --image /path/to/photo.jpg \
  --vision-model artifacts/models/vision_drq.tflite \
  --vision-manifest artifacts/models/vision_drq.tflite.manifest.json \
  --class-map data/prepared/class_map.json
```

树莓派相机需要安装 PiCamera2、GPIO/SPI 支持、项目的 `.[rpi]` 依赖和兼容的 TensorFlow Lite Runtime。然后运行：

```bash
PYTHONPATH=src python scripts/rpi_camera.py \
  --vision-model artifacts/models/vision_drq.tflite \
  --vision-manifest artifacts/models/vision_drq.tflite.manifest.json \
  --class-map data/prepared/class_map.json
```

追加 `--display` 使用 ST7789 屏幕；追加 `--geo-model`、`--geo-manifest`、`--latitude` 和 `--longitude` 使用 Geo Prior 融合。

<details>
<summary>已验证的微型流程与教学使用</summary>

微型流程在 2026 年 9 月 8 日使用 Python 3.12.8、TensorFlow 2.16.2、Keras 3.8.0 和 Apple M4 工作站完成，覆盖数据准备、双模型训练、FP32/DRQ/INT8 转换、50% 剪枝、融合推理和折中表生成。

| 版本 | 模型大小 | 合成数据 Top-1 | 端到端中位数 |
|---|---:|---:|---:|
| FP32 | 2,761,512 | 50% | 0.263 ms |
| DRQ | 870,752 | 50% | 0.237 ms |
| 全 INT8 | 973,752 | 50% | 0.192 ms |
| 剪枝 50% + DRQ | 862,096 | 50% | 0.230 ms |

这些是四张合成图片的工作站检查，不是实际数据或树莓派结果。60–90 分钟教学可使用 5–10 类子集，比较四种模型，再根据准确率、大小、延迟、内存和能耗做部署选择。

</details>

运行核心测试：`make test`。
