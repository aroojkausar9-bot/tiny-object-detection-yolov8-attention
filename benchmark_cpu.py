"""
Benchmark ONNX model inference on CPU (simulates Jetson Nano / Raspberry Pi 4).

Reports: mean latency (ms), FPS, model file size (MB).

Usage:
  python scripts/benchmark_cpu.py --onnx runs/train/yolov8n_baseline/weights/best.onnx
  python scripts/benchmark_cpu.py --onnx runs/train/yolov8n_baseline/weights/best_int8.onnx
"""

import argparse
import time
import numpy as np
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="CPU inference benchmark (ONNX Runtime)")
    p.add_argument("--onnx",     type=str, required=True, help="Path to .onnx model")
    p.add_argument("--imgsz",    type=int, default=1280)
    p.add_argument("--runs",     type=int, default=50,  help="Benchmark iterations")
    p.add_argument("--warmup",   type=int, default=5,   help="Warmup iterations")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("Install onnxruntime: pip install onnxruntime")

    # CPU-only session
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1   # single-threaded (edge device simulation)
    session = ort.InferenceSession(
        args.onnx,
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )

    input_name  = session.get_inputs()[0].name
    input_shape = (1, 3, args.imgsz, args.imgsz)
    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(args.warmup):
        session.run(None, {input_name: dummy_input})

    # Benchmark
    latencies = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    mean_ms = np.mean(latencies)
    std_ms  = np.std(latencies)
    fps     = 1000.0 / mean_ms
    size_mb = Path(args.onnx).stat().st_size / 1e6

    print(f"\n{'='*50}")
    print(f"  Model      : {args.onnx}")
    print(f"  Size       : {size_mb:.1f} MB")
    print(f"  Latency    : {mean_ms:.1f} ± {std_ms:.1f} ms")
    print(f"  CPU FPS    : {fps:.1f}")
    print(f"  Runs       : {args.runs}  (warmup={args.warmup})")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
