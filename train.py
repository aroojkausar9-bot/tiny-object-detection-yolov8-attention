"""
Main training script for tiny object detection experiments.

Supports:
  - YOLOv8n / YOLOv11n optimized baseline
  - + SE attention injection
  - + CBAM attention injection

Usage:
  python scripts/train.py --config configs/yolov8n_optimized.yaml --data data/visdrone.yaml
  python scripts/train.py --config configs/yolov8n_cbam.yaml     --data data/visdrone.yaml --attention cbam
  python scripts/train.py --config configs/yolov11n_optimized.yaml --data data/uavdt.yaml
"""

import argparse
import sys
import yaml
from pathlib import Path

# Make sure repo root is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from models.attention import inject_se, inject_cbam


def parse_args():
    p = argparse.ArgumentParser(description="Train tiny object detection model")
    p.add_argument("--config",    type=str, required=True,
                   help="Path to hyperparameter YAML (configs/)")
    p.add_argument("--data",      type=str, required=True,
                   help="Path to dataset YAML (data/)")
    p.add_argument("--model",     type=str, default=None,
                   help="YOLO model variant, e.g. yolov8n.pt (overrides config)")
    p.add_argument("--attention", type=str, default=None,
                   choices=["se", "cbam"],
                   help="Attention module to inject (optional)")
    p.add_argument("--layers",    type=int, nargs="+", default=[2, 4],
                   help="Backbone layer indices for attention injection (default: 2 4)")
    p.add_argument("--project",   type=str, default="runs/train",
                   help="Output project directory")
    p.add_argument("--name",      type=str, default=None,
                   help="Experiment name (auto-generated if not set)")
    p.add_argument("--resume",    action="store_true",
                   help="Resume from last checkpoint")
    return p.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Determine model weights
    model_name = args.model or cfg.get("model", "yolov8n.pt")
    print(f"\n{'='*60}")
    print(f"  Model    : {model_name}")
    print(f"  Data     : {args.data}")
    print(f"  Attention: {args.attention or 'None (baseline)'}")
    print(f"{'='*60}\n")

    # Load YOLO model
    model = YOLO(model_name)

    # Inject attention if requested
    if args.attention == "se":
        model = inject_se(model, layer_indices=args.layers)
    elif args.attention == "cbam":
        model = inject_cbam(model, layer_indices=args.layers)

    # Determine experiment name
    exp_name = args.name
    if exp_name is None:
        base = Path(model_name).stem
        attn = f"_{args.attention}" if args.attention else "_baseline"
        exp_name = f"{base}{attn}"

    # Build training kwargs from config YAML
    train_kwargs = {
        "data":           args.data,
        "epochs":         cfg.get("epochs", 60),
        "imgsz":          cfg.get("imgsz", 1280),
        "batch":          cfg.get("batch", 8),
        "lr0":            cfg.get("lr0", 0.01),
        "lrf":            cfg.get("lrf", 0.01),
        "momentum":       cfg.get("momentum", 0.937),
        "weight_decay":   cfg.get("weight_decay", 0.0005),
        "box":            cfg.get("box", 10.0),
        "cls":            cfg.get("cls", 0.3),
        "dfl":            cfg.get("dfl", 1.5),
        "label_smoothing":cfg.get("label_smoothing", 0.1),
        "mosaic":         cfg.get("mosaic", 1.0),
        "copy_paste":     cfg.get("copy_paste", 0.7),
        "mixup":          cfg.get("mixup", 0.15),
        "degrees":        cfg.get("degrees", 5.0),
        "shear":          cfg.get("shear", 2.0),
        "scale":          cfg.get("scale", 0.5),
        "fliplr":         cfg.get("fliplr", 0.5),
        "hsv_h":          cfg.get("hsv_h", 0.015),
        "hsv_s":          cfg.get("hsv_s", 0.7),
        "hsv_v":          cfg.get("hsv_v", 0.4),
        "workers":        cfg.get("workers", 4),
        "device":         cfg.get("device", 0),
        "project":        args.project,
        "name":           exp_name,
        "exist_ok":       True,
        "resume":         args.resume,
        "verbose":        True,
    }

    print("Training hyperparameters:")
    for k, v in train_kwargs.items():
        print(f"  {k:20s} = {v}")
    print()

    # Train
    results = model.train(**train_kwargs)

    print(f"\nTraining complete. Best model: {results.save_dir}/weights/best.pt")
    return results


if __name__ == "__main__":
    main()
