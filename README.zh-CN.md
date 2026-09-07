# Biodiversity Edge AI

[English](README.md) | **简体中文**

一个用于离线物种识别的嵌入式机器学习项目，涵盖图像分类、时空地理先验、量化、剪枝及树莓派推理，所有功能位于同一个 Python 包中。

[快速开始](#快速开始) · [数据集](#数据集) · [训练](#训练) · [模型优化](#模型优化) · [评测](#评测) · [推理](#推理) · [树莓派部署](#树莓派部署) · [系统架构](#系统架构) · [更多文档](#更多文档)

## 快速开始

支持 Python 3.10–3.12，推荐 Python 3.12。在 macOS 或 Linux 工作站上克隆仓库、创建环境并运行小数据全流程：

```bash
git clone https://github.com/winson5566/biodiversity-edge-ai.git
cd biodiversity-edge-ai
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train,dev]'
make smoke
```

后续工作站命令均在仓库根目录执行，并保持该环境处于激活状态。

微型测试生成 **2 类、24 张合成图片**，按 16/4/4 划分训练、验证与测试集，训练两个模型，导出 FP32/DRQ/INT8，进行 50% 剪枝和微调，再评测融合预测。无需下载 iNaturalist 数据或 ImageNet 权重。

输出位于 `artifacts/smoke/models/` 和 `artifacts/smoke/results/tradeoffs.md`。该流程已经实际跑通，详见[测试记录](docs/SMOKE_TEST.md)。合成数据准确率和工作站耗时用于验证软件链路，不能代表真实物种识别效果或树莓派性能。

## 数据集

项目使用通过 AWS Open Data Program 分发的 **iNaturalist 2021**。默认训练数据为 **Train Mini：500,000 张图片、10,000 类**，使用全部可用图片，不设每类图片数量上限。通过 `DATA_SOURCE=full` 可切换为全量 Train。下载前请阅读[官方数据说明与使用条款](https://github.com/visipedia/inat_comp/tree/master/2021)。

### 下载清单

| 文件 | 大小 | MD5 |
|---|---:|---|
| [Train Mini 图片（默认）](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz) | 42 GB | `db6ed8330e634445efc8fec83ae81442` |
| [Train Mini 标注](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz) | 45 MB | `395a35be3651d86dc3b0d365b8ea5f92` |
| [全量 Train 图片](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz) | 224 GB | `e0526d53c7f7b2e3167b2b43bb2690ed` |
| [全量 Train 标注](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.json.tar.gz) | 221 MB | `38a7bb733f7a09214d44293460ec0021` |
| [Validation 图片](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz) | 8.4 GB | `f6f6e0e242e3d4c9569ba56400938afc` |
| [Validation 标注](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz) | 9.4 MB | `4d761e0f6a86cc63e8f7afc91f6a8f0b` |
| [Test 图片](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/public_test.tar.gz) | 43 GB | `7124b949fe79bfa7f7019a15ef3dbd06` |
| [Test 信息](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/public_test.json.tar.gz) | 21 MB | `7a9413db55c6fa452824469cc7dd9d3d` |

所有图片均为 JPEG，最长边为 500 像素。解压可能耗时较长，磁盘需要同时容纳压缩包和解压后的文件。

<details>
<summary>全部文件的 S3 地址</summary>

```text
s3://ml-inat-competition-datasets/2021/train_mini.tar.gz
s3://ml-inat-competition-datasets/2021/train_mini.json.tar.gz
s3://ml-inat-competition-datasets/2021/train.tar.gz
s3://ml-inat-competition-datasets/2021/train.json.tar.gz
s3://ml-inat-competition-datasets/2021/val.tar.gz
s3://ml-inat-competition-datasets/2021/val.json.tar.gz
s3://ml-inat-competition-datasets/2021/public_test.tar.gz
s3://ml-inat-competition-datasets/2021/public_test.json.tar.gz
```

</details>

### 下载并准备 Mini

下载默认的图片与标注文件：

```bash
mkdir -p raw/inat2021
cd raw/inat2021
curl -C - -O https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz
curl -C - -O https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz
```

解压前将两个文件的 MD5 与上表对照。Linux 使用：

```bash
md5sum train_mini.tar.gz train_mini.json.tar.gz
```

macOS 使用：

```bash
md5 train_mini.tar.gz train_mini.json.tar.gz
```

确认校验值一致后解压：

```bash
tar -xzf train_mini.tar.gz
tar -xzf train_mini.json.tar.gz
cd ../..
```

使用全量 Train 时，通过上表下载 `train.tar.gz` 和 `train.json.tar.gz`，均解压到 `raw/inat2021`。Validation 和 Test 可按需下载，并解压到同一目录。

以下展示所有数据源的目录结构；运行时只需准备选定训练源对应的图片和标注：

```text
raw/inat2021/
  train_mini.json
  train_mini/category/image.jpg
  train.json
  train/category/image.jpg
  val.json
  val/category/image.jpg
  public_test.json
  public_test/image.jpg
```

`IMAGES_ROOT` 应设为 `raw/inat2021`。标注内的路径已经包含 `train_mini/`、`train/` 或 `val/` 前缀。

准备默认 Mini 数据：

```bash
make prepare
```

生成 `data/prepared/class_map.json`、`dataset_manifest.json`、`images/{train,val,test}/` 和 `metadata/{train,val,test}.csv`。数据清单记录输入文件哈希、类别映射、选择规则、样本数量及跳过的记录。

### 当前项目使用哪些划分？

默认流程从选定的**训练数据源**中确定性地划分本地训练、验证和测试集，比例约为 70%/15%/15%，每类分别取整。`make workstation` 不会自动使用下载清单中的官方 Validation 或 public Test。

官方 public Test 信息不含真实类别标注，因此不能传入监督数据准备命令，也不能用于计算本地 Top-1 准确率。官方 Validation 是独立的有标注数据源，评测时需要与训练模型的类别映射对齐。详见[官方标注说明](https://github.com/visipedia/inat_comp/tree/master/2021#annotation-format-notes)。

字段定义、图片复制方式和单独的数据准备命令见[数据准备与格式文档](docs/DATA_FORMATS.md)。

## 训练

### 默认：Mini

完成 `make prepare` 后，训练图像分类器和六特征 Geo Prior：

```bash
make train
```

默认视觉模型为 MobileNetV2，使用 ImageNet 初始化、128×128 输入和 0.5 宽度系数。Geo Prior 为残差 FCNet，输入为经度、纬度及日期的编码。两个模型共享同一类别映射。

需要连续执行数据准备、训练、导出、剪枝和评测时，运行：

```bash
make workstation
```

### 全量 Train

下载并解压全量图片和标注后：

```bash
make workstation DATA_SOURCE=full
```

| 模式 | 准备后的数据 | 模型 | 评测结果 |
|---|---|---|---|
| Mini | `data/prepared` | `artifacts/models` | `artifacts/results` |
| Full | `data/prepared_full` | `artifacts/models_full` | `artifacts/results_full` |

单独运行某个阶段时也需传入 `DATA_SOURCE=full`，例如 `make train DATA_SOURCE=full`。后续示例使用 Mini 路径；运行全量或自定义配置时，请替换为对应路径。

### 小样本与自定义路径

选择 10 类，每类最多 50 张图片：

```bash
make workstation NUM_CLASSES=10 MAX_PER_CLASS=50 \
  DATASET=data/prepared_demo \
  MODEL_DIR=artifacts/models_demo RESULT_DIR=artifacts/results_demo
```

使用存放在其他位置的数据：

```bash
make workstation RAW_JSON=/path/to/train_mini.json IMAGES_ROOT=/path/to/raw
```

更换源路径、类别选择或划分参数时，应使用新的数据、模型及结果目录。Make 会复用已有产物，不能自动识别命令行变量的变化。

通过 `HEAD_EPOCHS`、`FINETUNE_EPOCHS`、`GEO_EPOCHS` 和 `BATCH_SIZE` 调整训练。`MAX_PER_CLASS` 默认留空，表示不限制数量；`MIN_PER_CLASS=20` 控制类别入选的最少样本数。全量数据准备会在内存中建立索引，内存与磁盘需求随数据量增长。微型测试未执行全量训练。

## 模型优化

导出已训练的基线与 Geo Prior，再生成剪枝版本：

```bash
make export
make optimize
```

| 视觉模型文件 | 优化方式 |
|---|---|
| `vision_fp32.tflite` | FP32 基线 |
| `vision_drq.tflite` | 动态范围权重量化 |
| `vision_int8.tflite` | 全整型 INT8，使用训练图片校准 |
| `vision_pruned_50_drq.tflite` | 50% 幅值剪枝、微调，再进行 DRQ |

文件保存在 `artifacts/models/`。每个 TFLite 模型均配有 `.tflite.manifest.json` 清单。Geo Prior 导出文件为 `geo_prior_fp32.tflite`。

自定义稀疏率示例：

```bash
PYTHONPATH=src python scripts/prune_vision.py \
  --keras-model artifacts/models/vision_baseline.keras \
  --output artifacts/models/vision_pruned_30.keras \
  --sparsity 0.30 --data-dir data/prepared \
  --input-size 128 --input-scale minus1_1 --finetune-epochs 3
```

随后用 `scripts/export_tflite.py` 导出该 Keras 模型，并使用独立的文件名与模型 ID；命令格式见[导出示例](docs/DATA_FORMATS.md#5-export-deployment-models)。

## 评测

让四个标准视觉模型使用同一 Geo Prior 和测试图片进行评测：

```bash
make benchmark
```

输出包括原始 JSON、`artifacts/results/tradeoffs.csv` 和 `tradeoffs.md`。可比较 Top-1 准确率、模型大小、推理耗时、端到端耗时、吞吐量和 RSS 内存增量。能耗需要另行配置外部测量方法。

Makefile 当前固定使用 `alpha=0.3`，不会自动搜索融合权重。最终测试前应在验证集上选择权重，并固定各模型的测试划分、预处理、线程数、预热和重复次数。

在目标设备上运行一次受控评测：

```bash
PYTHONPATH=src python scripts/benchmark_rpi.py \
  --images data/prepared/images/test \
  --metadata-csv data/prepared/metadata/test.csv \
  --vision-model artifacts/models/vision_drq.tflite \
  --vision-manifest artifacts/models/vision_drq.tflite.manifest.json \
  --geo-model artifacts/models/geo_prior_fp32.tflite \
  --geo-manifest artifacts/models/geo_prior_fp32.tflite.manifest.json \
  --class-map data/prepared/class_map.json \
  --alpha 0.3 --threads 1 --warmup 10 --repetitions 5 \
  --output artifacts/results/pi_benchmark.json
```

权重为零不一定带来文件缩小或推理加速，应根据设备实测选择部署方案。详见[性能评测指南](docs/PERFORMANCE_EVALUATION.md)。

## 推理

识别单张图片：

```bash
PYTHONPATH=src python scripts/predict.py \
  --image /path/to/photo.jpg \
  --vision-model artifacts/models/vision_drq.tflite \
  --vision-manifest artifacts/models/vision_drq.tflite.manifest.json \
  --class-map data/prepared/class_map.json
```

启用地理融合时，将以下参数追加到同一命令，并填写观测的实际经纬度和日期：

```bash
--geo-model artifacts/models/geo_prior_fp32.tflite \
--geo-manifest artifacts/models/geo_prior_fp32.tflite.manifest.json \
--latitude -43.5321 --longitude 172.6362 --date 2026-09-08
```

缺少位置或日期时自动使用纯视觉推理。类别映射不兼容的模型会被拒绝加载。

## 树莓派部署

将项目、选定的 TFLite 文件、模型清单和 `class_map.json` 复制到树莓派，并保持命令使用的路径一致。评测时还需复制准备好的测试图片和元数据；跨机器传输时，应将图片软链接转换为实际文件。

安装系统提供的 Picamera2、GPIO 和 SPI 依赖，然后安装项目的轻量运行依赖：

```bash
python -m pip install -e '.[rpi]'
```

为树莓派的操作系统与 Python 版本安装兼容的 TensorFlow Lite Runtime。使用虚拟环境时，应确保环境能够访问系统安装的相机库。

拍摄一帧并识别：

```bash
PYTHONPATH=src python scripts/rpi_camera.py \
  --vision-model artifacts/models/vision_drq.tflite \
  --vision-manifest artifacts/models/vision_drq.tflite.manifest.json \
  --class-map data/prepared/class_map.json
```

使用 ST7789 屏幕时追加 `--display`。地理融合接受 `--geo-model`、`--geo-manifest`、`--latitude` 和 `--longitude`，相机推理使用当天日期。实时 GPS 接入尚未实现。相机、屏幕、延迟和功耗需要在真实硬件上验证。

## 系统架构

```text
Image ─────────────> Vision TFLite ────┐
                                      ├──> Fusion ──> Species prediction
Latitude/longitude/date ──> Geo Prior ─┘
```

工作站负责训练和导出，树莓派加载 TFLite 模型。共享清单声明输入形状、缩放、数据类型、优化模式和类别映射哈希，使两个模型的预测类别保持对齐。详见[架构与接口](docs/ARCHITECTURE.md)。

## 更多文档

以下深入文档目前为英文：

- [数据准备与格式](docs/DATA_FORMATS.md)
- [架构与接口](docs/ARCHITECTURE.md)
- [性能评测](docs/PERFORMANCE_EVALUATION.md)
- [小数据全流程测试](docs/SMOKE_TEST.md)
- [课堂讲解与实践安排](docs/LECTURE_DEMO.md)

运行核心测试：

```bash
make test
```
