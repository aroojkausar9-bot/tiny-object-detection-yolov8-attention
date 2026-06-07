"""
Squeeze-and-Excitation (SE) Block
Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018

Architecture-level injection (NOT hook-based) to ensure proper gradient flow.
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Channel-wise recalibration via global average pooling + MLP excitation.

    For a feature map X of shape (B, C, H, W):
      1. Squeeze: global average pool -> (B, C)
      2. Excite:  FC(C -> C/r) -> ReLU -> FC(C/r -> C) -> Sigmoid
      3. Scale:   X * e  (broadcast over H, W)

    Args:
        channels (int): Number of input channels C.
        reduction (int): Reduction ratio r. Effective bottleneck = max(C // r, 4).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        bottleneck = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)        # (B, C, 1, 1)
        self.excite = nn.Sequential(
            nn.Flatten(),                              # (B, C)
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        scale = self.squeeze(x)                        # (B, C, 1, 1)
        scale = self.excite(scale)                     # (B, C)
        scale = scale.view(scale.size(0), scale.size(1), 1, 1)  # (B, C, 1, 1)
        return x * scale                               # broadcast multiply


def inject_se(model, layer_indices: list, reduction: int = 16):
    """
    Inject SE blocks at specified backbone layer indices using
    architecture-level sequential wrapping (not hooks).

    This ensures attention weights receive gradient signal from
    the detection loss — a critical fix over hook-based approaches.

    Args:
        model: Ultralytics YOLO model object (model.model).
        layer_indices: Backbone layer indices to inject at.
                       Recommended: [2, 4] (stride-4 and stride-8 layers).
        reduction: SE reduction ratio.

    Returns:
        model with SE blocks injected in-place.
    """
    backbone = model.model.model  # Ultralytics backbone module list

    for idx in layer_indices:
        original_layer = backbone[idx]
        # Determine channel count from the layer's output
        # P2 (idx=2): 32 channels; P3 (idx=4): 64 channels
        in_channels = _infer_channels(original_layer)
        se = SEBlock(channels=in_channels, reduction=reduction)

        # Architecture-level injection — proper computation graph node
        backbone[idx] = nn.Sequential(original_layer, se)
        print(f"[SE] Injected at backbone[{idx}] "
              f"(channels={in_channels}, bottleneck={max(in_channels // reduction, 4)})")

    return model


def _infer_channels(layer: nn.Module) -> int:
    """Infer output channels from common Ultralytics layer types."""
    # C2f / C3k2 modules store output channels in 'cv2'
    if hasattr(layer, 'cv2'):
        return layer.cv2.conv.out_channels
    # Fallback: check last conv in children
    for m in reversed(list(layer.modules())):
        if isinstance(m, nn.Conv2d):
            return m.out_channels
    raise ValueError(f"Cannot infer output channels from {type(layer)}")
