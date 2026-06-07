"""
Visualization utilities for tiny object detection results.
"""

import numpy as np
from pathlib import Path


def draw_predictions(image: np.ndarray, boxes, scores, class_ids,
                     class_names: list, score_thresh: float = 0.25) -> np.ndarray:
    """
    Draw bounding boxes on image. Highlights tiny objects (< 32×32 px) in red.

    Args:
        image: HWC BGR numpy array.
        boxes: (N, 4) xyxy absolute coordinates.
        scores: (N,) confidence scores.
        class_ids: (N,) integer class indices.
        class_names: List of class name strings.
        score_thresh: Minimum score to draw.

    Returns:
        Annotated image.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("Install opencv-python: pip install opencv-python")

    img = image.copy()
    for box, score, cls_id in zip(boxes, scores, class_ids):
        if score < score_thresh:
            continue

        x1, y1, x2, y2 = map(int, box)
        w = x2 - x1
        h = y2 - y1
        is_tiny = (w < 32 and h < 32)

        color = (0, 0, 255) if is_tiny else (0, 255, 0)   # red=tiny, green=normal
        thickness = 1 if is_tiny else 2

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        label = f"{class_names[cls_id]} {score:.2f}"
        font_scale = 0.35 if is_tiny else 0.5
        cv2.putText(img, label, (x1, max(y1 - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

    return img


def plot_class_map(class_names: list, map_values: list, title: str = "Per-Class mAP@0.5"):
    """
    Bar chart of per-class mAP values.

    Args:
        class_names: List of class name strings.
        map_values: Corresponding mAP@0.5 values.
        title: Plot title.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Install matplotlib: pip install matplotlib")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c" if v < 0.3 else "#2ecc71" for v in map_values]
    bars = ax.bar(class_names, map_values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Class")
    ax.set_ylabel("mAP@0.5")
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=np.mean(map_values), color="navy", linestyle="--", linewidth=1.5,
               label=f"Mean: {np.mean(map_values):.3f}")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig
