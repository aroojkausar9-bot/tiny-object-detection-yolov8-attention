"""
Evaluate a trained model on VisDrone or UAVDT validation/test split.

Reports: mAP@0.5, mAP@0.5:0.95, Precision, Recall (per-class and mean).

Usage:
  python scripts/evaluate.py --weights runs/train/yolov8n_baseline/weights/best.pt \
                              --data data/visdrone.yaml \
                              --imgsz 1280
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate YOLO tiny-OD model")
    p.add_argument("--weights", type=str, required=True,
                   help="Path to model weights (.pt)")
    p.add_argument("--data",    type=str, required=True,
                   help="Dataset YAML (data/visdrone.yaml or data/uavdt.yaml)")
    p.add_argument("--imgsz",   type=int, default=1280)
    p.add_argument("--split",   type=str, default="val",
                   choices=["val", "test"], help="Dataset split to evaluate on")
    p.add_argument("--batch",   type=int, default=8)
    p.add_argument("--device",  type=int, default=0)
    p.add_argument("--save-json", action="store_true",
                   help="Save results to JSON for further analysis")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    print(f"\nEvaluating: {args.weights}")
    print(f"Dataset   : {args.data}  (split={args.split})")
    print(f"Image size: {args.imgsz}\n")

    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        split=args.split,
        device=args.device,
        save_json=args.save_json,
        verbose=True,
    )

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"  mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"  Precision    : {metrics.box.mp:.4f}")
    print(f"  Recall       : {metrics.box.mr:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
