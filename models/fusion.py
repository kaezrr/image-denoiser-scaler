"""
models/fusion.py
Spatial Attention Fusion Network (SAFN) — the trainable DL contribution.

Architecture:
    Input: concatenated expert outputs (B, 9, H, W)   [3 experts x 3ch each]
        |
    Stem conv: 9 -> mid_channels
        |
    N x Residual blocks (mid_channels)
        |
    Attention head: mid_channels -> num_experts (= 3)
        |
    Softmax across expert dim -> per-pixel attention weights (B, 3, H, W)
        |
    Weighted sum of expert outputs -> (B, 3, H, W)  final output

Key insight: the softmax forces the network to CHOOSE between experts per pixel,
which is interpretable and efficient. High attention on Expert 1 = "trust Real-ESRGAN
here", high on Expert 2 = "trust EDSR here", high on Expert 3 = "simple bicubic is best".

Total trainable parameters: ~300k (tiny but effective)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Simple residual block: Conv -> ReLU -> Conv -> skip."""
    def __init__(self, c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class ChannelAttention(nn.Module):
    """Squeeze-and-excite style channel attention on feature maps."""
    def __init__(self, c: int, reduction: int = 8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, c // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(c // reduction, c),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class SpatialAttentionFusionNet(nn.Module):
    """
    Learns per-pixel soft attention weights over expert outputs.
    Only this network is trained — all experts are frozen.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        fc  = cfg["fusion"]
        in_c   = fc["in_channels"]    # 9 (3 experts x 3ch)
        mid_c  = fc["mid_channels"]   # 64
        n_res  = fc["num_res_blocks"] # 4
        n_exp  = fc["num_experts"]    # 3

        # Feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[ResBlock(mid_c) for _ in range(n_res)],
            ChannelAttention(mid_c),
        )

        # Attention map head: outputs per-pixel logits for each expert
        self.attn_head = nn.Sequential(
            nn.Conv2d(mid_c, mid_c // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c // 2, n_exp, 1),   # (B, n_exp, H, W) logits
        )
        self.n_exp = n_exp
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, expert_cat: torch.Tensor, expert_list: list) -> dict:
        """
        Args:
            expert_cat:  (B, 9, H, W) — concatenated expert outputs
            expert_list: list of 3 tensors each (B, 3, H, W) — individual expert outputs

        Returns:
            dict with:
              "output":       (B, 3, H, W) — fused image
              "attn_weights": (B, 3, H, W) — per-pixel attention (for visualization)
        """
        feat   = self.stem(expert_cat)
        feat   = self.res_blocks(feat)
        logits = self.attn_head(feat)                        # (B, 3, H, W)
        attn   = F.softmax(logits, dim=1)                   # (B, 3, H, W), sums to 1 per pixel

        # Weighted sum: for each expert, multiply its 3-channel output by its 1-channel weight
        output = sum(
            attn[:, i:i+1, :, :] * expert_list[i]
            for i in range(self.n_exp)
        )
        return {"output": output, "attn_weights": attn}
