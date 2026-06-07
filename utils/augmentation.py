"""
Additional augmentation utilities used in the paper.
Primarily: CLAHE (Contrast Limited Adaptive Histogram Equalization)
via the Albumentations library for local contrast enhancement.
"""

import numpy as np


def build_albumentations_transform(clip_limit: float = 2.0,
                                   tile_grid: tuple = (8, 8),
                                   p: float = 0.5):
    """
    Build an Albumentations transform with CLAHE for tiny object contrast enhancement.

    CLAHE improves the visibility of low-contrast tiny objects in aerial imagery
    by performing histogram equalization locally rather than globally.

    Args:
        clip_limit: Contrast clipping threshold (higher = more contrast).
        tile_grid: Grid size for local histogram computation.
        p: Probability of applying CLAHE per image.

    Returns:
        albumentations.Compose transform, or None if albumentations not installed.
    """
    try:
        import albumentations as A
    except ImportError:
        print("[WARN] albumentations not installed. Skipping CLAHE augmentation.")
        print("       Install with: pip install albumentations")
        return None

    transform = A.Compose([
        A.CLAHE(clip_limit=clip_limit, tile_grid_size=tile_grid, p=p),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

    return transform


def apply_clahe(image: np.ndarray, bboxes: list, class_labels: list,
                transform=None) -> tuple:
    """
    Apply CLAHE transform to a single image.

    Args:
        image: HWC numpy array (uint8).
        bboxes: List of YOLO-format bounding boxes [[cx, cy, w, h], ...].
        class_labels: List of integer class IDs.
        transform: Albumentations transform (from build_albumentations_transform).

    Returns:
        (augmented_image, bboxes, class_labels) — unchanged if transform is None.
    """
    if transform is None:
        return image, bboxes, class_labels

    result = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    return result["image"], result["bboxes"], result["class_labels"]
