# Efficient Tiny Object Detection on Resource-Constrained Devices
### Using Enhanced Lightweight YOLO Architectures

> **Arooj Kausar** & **Syeda Kisaa Fatima**
> Department of Artificial Intelligence, NUST Islamabad, Pakistan

---

## 📌 Overview

This repository contains the full implementation of our paper on efficient tiny object detection using YOLOv8-nano and YOLOv11-nano. We target objects ≤32×32 pixels in aerial/UAV imagery and investigate:

- High-resolution training (1280×1280) with aggressive augmentation
- Architecture-level injection of SE and CBAM attention modules
- A critical fix for the **gradient-flow flaw** in hook-based attention implementations
- Cross-dataset generalization from VisDrone2019 → UAVDT

**Key Result:** Our optimized baseline achieves **0.451 mAP@0.5** on VisDrone2019-DET — a **32% relative improvement** over standard YOLOv8n settings — without any architectural changes.

---

## 📊 Results Summary

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Params |
|-------|---------|--------------|-----------|--------|--------|
| YOLOv8n Standard (640, box=7.5) | 0.342 | 0.197 | 0.448 | 0.339 | 3.01M |
| **YOLOv8n Optimized (Ours)** | **0.451** | **0.259** | **0.557** | **0.451** | 3.01M |
| YOLOv8n + SE | 0.403 | 0.226 | 0.499 | 0.420 | 3.01M |
| YOLOv8n + CBAM | 0.404 | 0.226 | 0.500 | 0.418 | 3.01M |
| YOLOv11n Optimized (Ours) | 0.463 | 0.282 | 0.577 | 0.462 | ~3M |
| YOLOv11n + SE | 0.354 | 0.208 | 0.464 | 0.376 | ~3M |
| YOLOv11n + CBAM | 0.355 | 0.208 | 0.464 | 0.376 | ~3M |

**Cross-dataset (UAVDT):**

| Model | mAP@0.5 | Car | Truck | Bus | GPU FPS |
|-------|---------|-----|-------|-----|---------|
| YOLOv8n Optimized | 0.460 | 0.740 | 0.042 | 0.599 | ~45 |
| YOLOv11n Optimized | 0.460 | 0.761 | 0.038 | 0.530 | ~43 |

---

## 📁 Repository Structure

```
tiny-object-detection/
├── configs/
│   ├── yolov8n_optimized.yaml        # Optimized hyperparameters
│   ├── yolov8n_se.yaml               # YOLOv8n + SE config
│   ├── yolov8n_cbam.yaml             # YOLOv8n + CBAM config
│   ├── yolov11n_optimized.yaml       # YOLOv11n optimized config
│   └── augmentation.yaml             # Augmentation pipeline config
├── models/
│   ├── attention/
│   │   ├── __init__.py
│   │   ├── se_block.py               # Squeeze-and-Excitation block
│   │   └── cbam_block.py             # CBAM block
│   ├── yolov8n_se.py                 # YOLOv8n with SE injection
│   └── yolov11n_cbam.py              # YOLOv11n with CBAM injection
├── data/
│   ├── visdrone.yaml                 # VisDrone dataset config
│   ├── uavdt.yaml                    # UAVDT dataset config
│   └── download_datasets.sh          # Dataset download scripts
├── utils/
│   ├── __init__.py
│   ├── augmentation.py               # Albumentations CLAHE pipeline
│   ├── metrics.py                    # mAP, FPS evaluation utils
│   └── visualize.py                  # Prediction visualization
├── scripts/
│   ├── train.py                      # Main training script
│   ├── evaluate.py                   # Evaluation on val/test split
│   ├── export_onnx.py                # ONNX export + INT8 quantization
│   └── benchmark_cpu.py              # CPU inference benchmarking
├── notebooks/
│   ├── 01_dataset_analysis.ipynb     # VisDrone EDA, tiny object stats
│   ├── 02_training_experiments.ipynb # Full experiment runs
│   └── 03_results_visualization.ipynb
├── results/
│   └── .gitkeep
├── docs/
│   └── gradient_flow_fix.md          # Explanation of hook vs arch-level injection
├── requirements.txt
├── environment.yml
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/tiny-object-detection.git
cd tiny-object-detection
```

### 2. Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# OR using conda
conda env create -f environment.yml
conda activate tiny-od
```

### 3. Download Datasets

```bash
bash data/download_datasets.sh
```

Or manually:
- **VisDrone2019-DET**: https://github.com/VisDrone/VisDrone-Dataset
- **UAVDT**: https://sites.google.com/view/grli-uavdt

### 4. Train

```bash
# Optimized YOLOv8n baseline (our best config)
python scripts/train.py --config configs/yolov8n_optimized.yaml --data data/visdrone.yaml

# YOLOv8n + CBAM
python scripts/train.py --config configs/yolov8n_cbam.yaml --data data/visdrone.yaml

# YOLOv11n optimized
python scripts/train.py --config configs/yolov11n_optimized.yaml --data data/visdrone.yaml
```

### 5. Evaluate

```bash
python scripts/evaluate.py --weights runs/train/exp/weights/best.pt --data data/visdrone.yaml --imgsz 1280
```

### 6. Export to ONNX + INT8

```bash
python scripts/export_onnx.py --weights runs/train/exp/weights/best.pt
```

---

## ⚙️ Optimized Training Configuration

Our key hyperparameter changes from YOLO defaults:

| Parameter | Default | Ours | Reason |
|-----------|---------|------|--------|
| `imgsz` | 640 | **1280** | Preserve sub-32px resolution |
| `box` loss weight | 7.5 | **10.0** | Higher gradient pressure for localization |
| `cls` loss weight | 0.5 | **0.3** | Reduce large-object dominance |
| `label_smoothing` | 0.0 | **0.1** | Regularize ambiguous tiny instances |
| `copy_paste` | 0.0 | **0.7** | Tiny instance augmentation |
| `mosaic` | 1.0 | **1.0** | Keep default |
| `mixup` | 0.0 | **0.15** | Additional diversity |
| `epochs` | 100 | 60* | Kaggle T4 constraint |

*60 epochs due to GPU time limits (~5.25 hrs/run); 150+ recommended.

---

## 🔬 Critical Finding: Gradient Flow Fix

Prior attention implementations using `register_forward_hook` have a **silent gradient-flow flaw**:

```python
#  INCORRECT — gradients do NOT flow through attention weights
def attention_hook(module, inputs, outputs):
    return attention_module(outputs)  # detached from computation graph

layer.register_forward_hook(attention_hook)
```

```python
# CORRECT — architecture-level injection
backbone[idx] = nn.Sequential(
    original_layer,
    attention_module   # proper node in computation graph
)
```

See [`docs/gradient_flow_fix.md`](docs/gradient_flow_fix.md) for a detailed explanation.

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@article{kausar2024tinyod,
  title     = {Efficient Tiny Object Detection on Resource-Constrained Devices Using Enhanced Lightweight YOLO Architectures},
  author    = {Kausar, Arooj and Fatima, Syeda Kisaa},
  year      = {2024},
  institution = {NUST Islamabad}
}
```

---

