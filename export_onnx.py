"""
Export best.pt to ONNX (opset 12, simplified, static input shape)
and quantize to INT8 via ONNX Runtime dynamic quantization.

Reproduces the edge-deployment pipeline described in the paper.

Usage:
  python scripts/export_onnx.py --weights runs/train/yolov8n_baseline/weights/best.pt
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Export YOLO model to ONNX + INT8")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--imgsz",   type=int, default=1280)
    p.add_argument("--opset",   type=int, default=12)
    p.add_argument("--simplify", action="store_true", default=True)
    p.add_argument("--no-quantize", action="store_true",
                   help="Skip INT8 quantization step")
    return p.parse_args()


def quantize_int8(onnx_path: str):
    """Dynamic INT8 quantization via ONNX Runtime (no calibration dataset)."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("[WARN] onnxruntime not found. Skipping INT8 quantization.")
        return

    out_path = onnx_path.replace(".onnx", "_int8.onnx")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=out_path,
        weight_type=QuantType.QInt8,
    )
    orig_mb  = Path(onnx_path).stat().st_size / 1e6
    quant_mb = Path(out_path).stat().st_size / 1e6
    print(f"\nINT8 quantization complete:")
    print(f"  FP32 : {onnx_path}  ({orig_mb:.1f} MB)")
    print(f"  INT8 : {out_path}  ({quant_mb:.1f} MB)")
    print(f"  Compression: {orig_mb/quant_mb:.1f}x")


def main():
    args = parse_args()
    model = YOLO(args.weights)

    print(f"\nExporting {args.weights} → ONNX (opset={args.opset}, imgsz={args.imgsz})")

    onnx_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=False,   # static input shape for Jetson/Pi
    )

    print(f"ONNX model saved to: {onnx_path}")

    if not args.no_quantize:
        quantize_int8(str(onnx_path))


if __name__ == "__main__":
    main()
