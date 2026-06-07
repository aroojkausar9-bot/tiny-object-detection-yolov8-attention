#!/usr/bin/env bash
# Download and prepare VisDrone2019-DET and UAVDT datasets.
# Run from the repository root: bash data/download_datasets.sh

set -e
DATASETS_DIR="./datasets"
mkdir -p "$DATASETS_DIR"

echo "============================================"
echo "  Tiny Object Detection — Dataset Download"
echo "============================================"

# ── VisDrone2019-DET ──────────────────────────────────────
echo ""
echo "[1/2] VisDrone2019-DET"
echo "  Due to institutional access requirements, please download manually:"
echo "  → https://github.com/VisDrone/VisDrone-Dataset"
echo ""
echo "  Expected layout after extraction:"
echo "  datasets/VisDrone/"
echo "    images/train/"
echo "    images/val/"
echo "    images/test/"
echo "    labels/train/   (YOLO format .txt)"
echo "    labels/val/"
echo "    labels/test/"
echo ""
echo "  If you have the raw VisDrone annotations (.txt, x y w h conf cls),"
echo "  use utils/convert_visdrone.py to convert to YOLO format."

# ── UAVDT ─────────────────────────────────────────────────
echo ""
echo "[2/2] UAVDT"
echo "  Please download manually from:"
echo "  → https://sites.google.com/view/grli-uavdt"
echo ""
echo "  Expected layout:"
echo "  datasets/UAVDT/"
echo "    images/train/"
echo "    images/test/"
echo "    labels/train/"
echo "    labels/test/"

echo ""
echo "After downloading, update path: fields in data/visdrone.yaml and data/uavdt.yaml"
echo "to point to your local dataset roots."
