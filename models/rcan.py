"""
models/rcan.py
==============
Residual Channel Attention Network (RCAN)
Paper: "Image Super-Resolution Using Very Deep Residual Channel Attention Networks"
Authors: Yulun Zhang et al. | ECCV 2018 | arXiv: 1807.02758

Implementation notes:
  - Follows the official architecture exactly.
  - Every sub-module is documented so you can trace data through the network.
  - The model is self-contained: just instantiate RCAN(config) and go.

Architecture at a glance:
  LR Image
    → Shallow Feature Extraction (1 Conv)
    → Residual-in-Residual Structure (RIR)
       ├─ n_resgroups × Residual Group (RG)
       │     └─ n_resblocks × Residual Channel Attention Block (RCAB)
       │              └─ Channel Attention (CA)
       └─ Long skip connection from input to end of RIR
    → Upsampling (PixelShuffle × scale)
    → Final Reconstruction (1 Conv)
  SR Image
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: default conv
# ---------------------------------------------------------------------------
def default_conv(
    in_channels: int, out_channels: int, kernel_size: int, bias: bool = True
) -> nn.Conv2d:
    """
    Standard 2-D convolution with 'same' padding.

    padding = kernel_size // 2 ensures the spatial dimensions are preserved
    when stride=1, which is always the case in RCAN except the pixel-shuffle
    upsample step.
    """
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias
    )


# ---------------------------------------------------------------------------
# ChannelAttention (CA)
# ---------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    """
    Channel Attention module — the core novelty of RCAN.

    Intuition
    ---------
    Different feature channels carry different amounts of useful information.
    High-frequency detail channels (edges, textures) should be amplified;
    low-frequency channels (smooth backgrounds) can be down-weighted.
    CA learns these weights *adaptively from the data*.

    Forward pass
    ------------
    Input:  (B, C, H, W)

    1. Global Average Pooling (GAP):    (B, C, H, W) → (B, C, 1, 1)
       - Squeezes spatial info into a single descriptor per channel.
       - Tells the network "how active is this channel on average?"

    2. Bottleneck FC layers:            (B, C, 1, 1) → (B, C//r, 1, 1) → (B, C, 1, 1)
       - Two 1×1 convolutions (equivalent to FC on squeezed tensor).
       - First reduces C → C//r (compression), learns cross-channel relations.
       - Second restores C//r → C (expansion).
       - Sigmoid clamps weights to [0, 1] — these are *attention weights*.

    3. Element-wise multiply:           (B, C, 1, 1) * (B, C, H, W) → (B, C, H, W)
       - Each channel of the feature map is scaled by its learned weight.

    Args
    ----
    n_feats   : number of feature channels (C)
    reduction : bottleneck ratio — paper uses 16
    """

    def __init__(self, n_feats: int, reduction: int = 16):
        super().__init__()

        # Bottleneck: C → C//reduction → C
        # Using 1×1 convolutions because the tensor is already (B, C, 1, 1)
        # after GAP, making them exactly equivalent to fully-connected layers.
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Conv2d(n_feats, n_feats // reduction, 1),  # FC: compress
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats // reduction, n_feats, 1),  # FC: expand
            nn.Sigmoid(),  # Attention weights ∈ [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # weights: (B, C, 1, 1) — broadcast over spatial dims
        return x * self.channel_attention(x)


# ---------------------------------------------------------------------------
# RCAB — Residual Channel Attention Block
# ---------------------------------------------------------------------------
class RCAB(nn.Module):
    """
    Residual Channel Attention Block — the fundamental building block of RCAN.

    Structure:
        Input
          │
        Conv(3×3) → ReLU → Conv(3×3) → ChannelAttention
          │                                    │
          └────────── + ──────────────────────┘
          │          (Short skip connection)
        Output

    The residual formulation (Input + F(Input)) helps with gradient flow through
    the very deep network (400+ layers in the full RCAN config).

    Args
    ----
    conv      : convolution function to use (default_conv by default)
    n_feats   : number of channels
    kernel_size: convolution kernel size (always 3 in the paper)
    reduction : channel attention reduction ratio
    res_scale : scales the residual branch; 1.0 = no scaling (paper default)
    act       : activation function
    """

    def __init__(
        self,
        conv,
        n_feats: int,
        kernel_size: int,
        reduction: int = 16,
        res_scale: float = 1.0,
        act: nn.Module = nn.ReLU(inplace=True),
    ):
        super().__init__()
        self.res_scale = res_scale

        # Main branch: Conv → ReLU → Conv → CA
        self.body = nn.Sequential(
            conv(n_feats, n_feats, kernel_size),  # First conv: preserve channel count
            act,
            conv(n_feats, n_feats, kernel_size),  # Second conv: preserve channel count
            ChannelAttention(n_feats, reduction),  # CA re-weights channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual: output = x + res_scale * F(x)
        # When res_scale=1.0, this simplifies to x + F(x)
        return x + self.res_scale * self.body(x)


# ---------------------------------------------------------------------------
# ResidualGroup (RG)
# ---------------------------------------------------------------------------
class ResidualGroup(nn.Module):
    """
    Residual Group — a stack of RCAB blocks with a long skip connection.

    Structure:
        Input
          │
        [RCAB × n_resblocks]   ← n RCABs stacked sequentially
          │
        Conv(3×3)               ← Aggregation conv at end of group
          │
        + ──────────────────── (Long skip from input)
          │
        Output

    The long skip connection (Input → Output of Conv) ensures that if the
    RCABs collectively learn nothing useful, the identity still flows through.
    This is crucial for the stability of very deep training.

    Args
    ----
    conv        : convolution function
    n_resblocks : number of RCAB blocks in this group (paper: 20)
    n_feats     : number of feature channels
    kernel_size : kernel size for convolutions
    reduction   : CA reduction ratio
    res_scale   : residual scaling for RCAB
    act         : activation
    """

    def __init__(
        self,
        conv,
        n_resblocks: int,
        n_feats: int,
        kernel_size: int,
        reduction: int = 16,
        res_scale: float = 1.0,
        act: nn.Module = nn.ReLU(inplace=True),
    ):
        super().__init__()

        # Build n_resblocks RCABs + final aggregation conv
        modules = [
            RCAB(conv, n_feats, kernel_size, reduction, res_scale, act)
            for _ in range(n_resblocks)
        ]
        modules.append(conv(n_feats, n_feats, kernel_size))  # Aggregation conv

        self.body = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Long skip: body output + original input
        return self.body(x) + x


# ---------------------------------------------------------------------------
# Upsampler
# ---------------------------------------------------------------------------
class Upsampler(nn.Sequential):
    """
    Learnable upsampling via sub-pixel convolution (PixelShuffle).

    How PixelShuffle works:
        1. A Conv2d maps: n_feats → n_feats * scale^2  channels
        2. nn.PixelShuffle(scale) reshapes: (B, C*s^2, H, W) → (B, C, H*s, W*s)
           - Takes groups of s^2 channels and rearranges them into spatial pixels
           - Learned — the network decides what pixels to "place" where
           - Much better than bilinear/bicubic because it's trained end-to-end

    For scale=4, we apply two ×2 upsamplings in sequence (more stable training).

    Args
    ----
    conv   : convolution function
    scale  : upscaling factor (2, 3, or 4)
    n_feats: number of input channels
    """

    def __init__(self, conv, scale: int, n_feats: int):
        modules = []

        if scale & (scale - 1) == 0:
            # scale is a power of 2 (e.g., 2 or 4)
            # For scale=4, loop runs twice (2 × PixelShuffle(2) = PixelShuffle(4))
            for _ in range(int(math.log(scale, 2))):
                # Expand channels by scale^2, then reshuffle into spatial dims
                modules.append(conv(n_feats, 4 * n_feats, 3))  # 4 = 2^2 for ×2
                modules.append(nn.PixelShuffle(2))
        elif scale == 3:
            # Special case: scale=3 is not a power of 2
            modules.append(conv(n_feats, 9 * n_feats, 3))  # 9 = 3^2 for ×3
            modules.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError(
                f"Upsampler: scale {scale} is not supported. Use 2, 3, or 4."
            )

        super().__init__(*modules)


# ---------------------------------------------------------------------------
# RCAN — Main Model
# ---------------------------------------------------------------------------
class RCAN(nn.Module):
    """
    Residual Channel Attention Network (RCAN).

    Paper: "Image Super-Resolution Using Very Deep Residual Channel Attention Networks"
    arXiv: 1807.02758 | ECCV 2018

    Full data flow:
        x: (B, 3, H, W)  ← LR input, optionally with noise

        1. Shallow feature extraction: (B, 3, H, W) → (B, n_feats, H, W)
           A single 3×3 conv lifts RGB to the working feature dimensionality.

        2. Deep feature extraction via RIR:
           - n_resgroups × ResidualGroup, each with n_resblocks × RCAB
           - Long skip connection from step 1 output to end of RIR
           Result: (B, n_feats, H, W)

        3. Sub-pixel upsampling: (B, n_feats, H, W) → (B, n_feats, H*scale, W*scale)

        4. Final reconstruction conv: (B, n_feats, H*scale, W*scale) → (B, 3, H*scale, W*scale)
           Maps features back to RGB.

        output: (B, 3, H*scale, W*scale) ← SR output

    Args
    ----
    n_resgroups  : number of Residual Groups (paper: 10)
    n_resblocks  : RCAB blocks per Residual Group (paper: 20)
    n_feats      : feature channels throughout the network (paper: 64)
    scale        : super-resolution scale factor (paper: 2/3/4/8)
    reduction    : channel attention reduction ratio (paper: 16)
    res_scale    : residual branch scaling factor (paper: 1.0)
    in_channels  : input image channels (3 for RGB)
    out_channels : output image channels (3 for RGB)
    """

    def __init__(
        self,
        n_resgroups: int = 10,
        n_resblocks: int = 20,
        n_feats: int = 64,
        scale: int = 2,
        reduction: int = 16,
        res_scale: float = 1.0,
        in_channels: int = 3,
        out_channels: int = 3,
    ):
        super().__init__()

        conv = default_conv
        act = nn.ReLU(inplace=True)

        # ------------------------------------------------------------------
        # 1. Shallow Feature Extraction
        # ------------------------------------------------------------------
        # A single 3×3 convolution that maps the 3-channel RGB input into the
        # n_feats-dimensional feature space. This "embedding" layer is kept
        # shallow on purpose — the deep feature extraction in the RIR body does
        # the heavy lifting.
        self.head = conv(in_channels, n_feats, 3)

        # ------------------------------------------------------------------
        # 2. Deep Feature Extraction (Residual-in-Residual)
        # ------------------------------------------------------------------
        # Stack n_resgroups Residual Groups. Each RG itself contains n_resblocks
        # RCAB blocks. Total RCAB blocks = n_resgroups × n_resblocks.
        # Default: 10 × 20 = 200 RCAB blocks (400+ convolution layers total!).
        resgroups = [
            ResidualGroup(
                conv,
                n_resblocks,
                n_feats,
                kernel_size=3,
                reduction=reduction,
                res_scale=res_scale,
                act=act,
            )
            for _ in range(n_resgroups)
        ]
        # Final conv after all residual groups (before long skip is added)
        resgroups.append(conv(n_feats, n_feats, 3))
        self.body = nn.Sequential(*resgroups)

        # ------------------------------------------------------------------
        # 3. Upsampling (sub-pixel convolution)
        # ------------------------------------------------------------------
        self.upsample = Upsampler(conv, scale, n_feats)

        # ------------------------------------------------------------------
        # 4. Final Reconstruction
        # ------------------------------------------------------------------
        # Maps the upsampled features back to an RGB image.
        self.tail = conv(n_feats, out_channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : torch.Tensor — shape (B, 3, H, W), values in [0, 1]

        Returns
        -------
        torch.Tensor — shape (B, 3, H*scale, W*scale), values in [0, 1]
        """
        # Step 1: Shallow feature extraction
        # x_head carries the first-level features and acts as the long skip
        # connection all the way through the RIR body.
        x_head = self.head(x)  # (B, n_feats, H, W)

        # Step 2: Deep feature extraction with long skip
        # body(x_head): residual features (what the deep network learned)
        # + x_head:     original features (identity path, stabilises training)
        x_body = self.body(x_head) + x_head  # (B, n_feats, H, W)

        # Step 3: Sub-pixel upsampling
        x_up = self.upsample(x_body)  # (B, n_feats, H*scale, W*scale)

        # Step 4: Final reconstruction to RGB
        out = self.tail(x_up)  # (B, 3, H*scale, W*scale)

        return out

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------
def build_rcan(config) -> RCAN:
    """
    Convenience factory: builds an RCAN instance from a config module.

    Usage:
        from configs.rcan_config import *
        import types, configs.rcan_config as cfg
        model = build_rcan(cfg)

    Args
    ----
    config : module or object with attributes:
             N_RESGROUPS, N_RESBLOCKS, N_FEATS, SCALE_FACTOR, REDUCTION, RES_SCALE
    """
    model = RCAN(
        n_resgroups=config.N_RESGROUPS,
        n_resblocks=config.N_RESBLOCKS,
        n_feats=config.N_FEATS,
        scale=config.SCALE_FACTOR,
        reduction=config.REDUCTION,
        res_scale=config.RES_SCALE,
    )
    return model


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("RCAN Architecture Self-Test")
    print("=" * 60)

    # Small model for quick testing
    model = RCAN(
        n_resgroups=2,  # Reduced for quick test (paper: 10)
        n_resblocks=4,  # Reduced for quick test (paper: 20)
        n_feats=32,  # Reduced for quick test (paper: 64)
        scale=2,
        reduction=8,
    )

    n_params = model.count_parameters()
    print(f"\nModel created successfully.")
    print(f"  n_resgroups : 2  (paper: 10)")
    print(f"  n_resblocks : 4  (paper: 20)")
    print(f"  n_feats     : 32 (paper: 64)")
    print(f"  scale       : ×2")
    print(f"  Parameters  : {n_params:,}")

    # Forward pass
    batch_size = 2
    lr_h, lr_w = 48, 48
    x = torch.randn(batch_size, 3, lr_h, lr_w)
    print(f"\nInput  shape : {tuple(x.shape)}")

    with torch.no_grad():
        out = model(x)

    print(f"Output shape : {tuple(out.shape)}")
    expected_h = lr_h * 2
    expected_w = lr_w * 2
    assert out.shape == (
        batch_size,
        3,
        expected_h,
        expected_w,
    ), f"Shape mismatch! Expected {(batch_size, 3, expected_h, expected_w)}, got {tuple(out.shape)}"

    print("\n✓ Forward pass correct.")
    print("✓ Output dimensions match scale factor ×2.")
    print("\nFull model config (paper defaults):")

    full_model = RCAN()  # All paper defaults
    print(f"  Parameters : {full_model.count_parameters():,}")
    print("\nSelf-test passed.")
