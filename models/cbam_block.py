"""
Convolutional Block Attention Module (CBAM)
Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018

Sequential channel attention then spatial attention.
Architecture-level injection for correct gradient flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel attention sub-module.
    Uses both avg-pool and max-pool descriptors through a shared MLP.

    Mc = sigmoid(MLP(AvgPool(X)) + MLP(MaxPool(X)))
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        bottleneck = max(channels // reduction, 4)
        # Shared MLP — applied to both avg and max pooled descriptors
        self.shared_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_desc = self.shared_mlp(self.avg_pool(x))   # (B, C)
        max_desc = self.shared_mlp(self.max_pool(x))   # (B, C)
        mc = torch.sigmoid(avg_desc + max_desc)
        return mc.view(mc.size(0), mc.size(1), 1, 1)   # (B, C, 1, 1)


class SpatialAttention(nn.Module):
    """
    Spatial attention sub-module.
    Concatenates channel-wise avg and max projections, then applies 7x7 conv.

    Ms = sigmoid(f^{7x7}([AvgPool(X'), MaxPool(X')]))
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_proj = x.mean(dim=1, keepdim=True)          # (B, 1, H, W)
        max_proj = x.max(dim=1, keepdim=True).values    # (B, 1, H, W)
        concat = torch.cat([avg_proj, max_proj], dim=1) # (B, 2, H, W)
        ms = torch.sigmoid(self.conv(concat))           # (B, 1, H, W)
        return ms


class CBAMBlock(nn.Module):
    """
    Full CBAM: channel attention -> spatial attention (sequential).

    X'' = (X ⊙ Mc) ⊙ Ms

    For tiny object detection:
    - Mc identifies channels encoding high-frequency edges / small-scale patterns
    - Ms focuses on pixel locations where those patterns actually occur,
      suppressing homogeneous sky/road backgrounds that dilute tiny-object gradients

    Args:
        channels (int): Input channel count.
        reduction (int): Channel attention reduction ratio.
        spatial_kernel (int): Conv kernel for spatial branch (default 7).
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mc = self.channel_att(x)     # (B, C, 1, 1)
        x_prime = x * mc             # channel-recalibrated feature map
        ms = self.spatial_att(x_prime)  # (B, 1, H, W)
        return x_prime * ms          # final CBAM output X''


def inject_cbam(model, layer_indices: list, reduction: int = 16, spatial_kernel: int = 7):
    """
    Inject CBAM blocks at specified backbone layer indices.

    Uses architecture-level sequential wrapping (not register_forward_hook)
    to ensure gradients flow correctly from detection loss through attention weights.

    Args:
        model: Ultralytics YOLO model.
        layer_indices: Backbone indices to inject at. Recommended: [2, 4].
        reduction: Channel attention reduction ratio.
        spatial_kernel: Spatial branch conv kernel size.

    Returns:
        model with CBAM injected in-place.
    """
    backbone = model.model.model

    for idx in layer_indices:
        original_layer = backbone[idx]
        in_channels = _infer_channels(original_layer)
        cbam = CBAMBlock(channels=in_channels,
                         reduction=reduction,
                         spatial_kernel=spatial_kernel)

        backbone[idx] = nn.Sequential(original_layer, cbam)

        ch_params = 2 * max(in_channels // reduction, 4) * in_channels
        sp_params = 2 * spatial_kernel * spatial_kernel
        total_params = ch_params + sp_params
        print(f"[CBAM] Injected at backbone[{idx}] "
              f"(channels={in_channels}, +{total_params} params)")

    return model


def _infer_channels(layer: nn.Module) -> int:
    """Infer output channels from common Ultralytics layer types."""
    if hasattr(layer, 'cv2'):
        return layer.cv2.conv.out_channels
    for m in reversed(list(layer.modules())):
        if isinstance(m, nn.Conv2d):
            return m.out_channels
    raise ValueError(f"Cannot infer output channels from {type(layer)}")
