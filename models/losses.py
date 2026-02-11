"""
models/losses.py
================
Loss functions for RCAN training.

Three losses are available:
  1. L1Loss        — primary pixel-level loss (MAE). Paper default.
  2. PerceptualLoss— feature-level loss via VGG19. Optional, improves textures.
  3. CombinedLoss  — weighted sum of L1 + Perceptual. Controlled by config weights.

Why L1 instead of MSE (L2)?
  - MSE averages over all pixels, heavily penalising large errors → encourages
    overly smooth, blurry outputs (the model plays it "safe").
  - L1 is less sensitive to large errors, preserving sharper predictions.
  - Empirically, L1 gives better PSNR/SSIM on SR tasks.

Why Perceptual Loss?
  - L1/L2 operate on raw pixels; human perception is not pixel-aligned.
  - VGG features capture semantic content (edges, textures, objects).
  - Adding perceptual loss encourages the model to produce images that *look*
    sharp, not just have low numerical pixel error.
  - In Phase 1 (RCAN as "Microscope") the default weight is 0 — purely pixel.
    You can enable it by setting LOSS_PERCEPTUAL_WEIGHT > 0 in the config.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# L1 Loss (primary)
# ---------------------------------------------------------------------------
class L1Loss(nn.Module):
    """
    Mean Absolute Error between the SR prediction and the HR ground truth.

    L1(SR, HR) = mean(|SR_i - HR_i|)

    Args
    ----
    reduction : 'mean' (default) | 'sum' | 'none'
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction=reduction)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(sr, hr)


# ---------------------------------------------------------------------------
# Perceptual Loss (optional)
# ---------------------------------------------------------------------------
class PerceptualLoss(nn.Module):
    """
    Feature-level perceptual loss using VGG19 feature maps.

    The key idea: instead of comparing pixels directly, compare the activations
    of a pretrained VGG19 network. This captures semantic similarity —
    two patches that look similar to humans will have similar VGG features,
    even if individual pixel values differ.

    Implementation details:
      - Uses VGG19 features up to relu3_4 (layer index 18) by default.
      - Deeper layers capture higher-level semantics; shallower = texture.
      - VGG19 weights are frozen (requires_grad=False) — we don't want to
        update the perceptual extractor during SR training.
      - Input normalization: VGG was trained on ImageNet-normalised images.
        We apply the same normalization before computing features.

    Args
    ----
    feature_layer : index of VGG19 feature layer to use (default 18 = relu3_4)
    """

    def __init__(self, feature_layer: int = 18):
        super().__init__()

        try:
            import torchvision.models as models
            vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
            # Extract only the layers up to feature_layer
            self.feature_extractor = nn.Sequential(
                *list(vgg.features)[:feature_layer + 1]
            )
        except ImportError:
            raise ImportError(
                "torchvision is required for PerceptualLoss. "
                "Install it with: pip install torchvision"
            )

        # Freeze VGG — we only use it as a fixed feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # ImageNet normalisation constants (VGG was trained on these)
        # Shape: (1, 3, 1, 1) for broadcasting
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize images from [0,1] to ImageNet stats."""
        return (x - self.mean) / self.std

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        # Normalize both to ImageNet stats
        sr_feat = self.feature_extractor(self._normalize(sr))
        hr_feat = self.feature_extractor(self._normalize(hr))
        # MSE between feature maps (L2 in feature space)
        return F.mse_loss(sr_feat, hr_feat)


# ---------------------------------------------------------------------------
# Combined Loss
# ---------------------------------------------------------------------------
class CombinedLoss(nn.Module):
    """
    Weighted combination of L1 and (optionally) Perceptual loss.

    Total = w_l1 * L1(SR, HR) + w_perceptual * Perceptual(SR, HR)

    If w_perceptual = 0 (default in config), perceptual loss is skipped entirely
    to avoid loading VGG19 unnecessarily.

    Args
    ----
    l1_weight          : weight for L1 term (default 1.0)
    perceptual_weight  : weight for Perceptual term (default 0.0 = disabled)
    """

    def __init__(
        self,
        l1_weight:         float = 1.0,
        perceptual_weight: float = 0.0,
    ):
        super().__init__()
        self.l1_weight         = l1_weight
        self.perceptual_weight = perceptual_weight

        self.l1_loss = L1Loss()

        if perceptual_weight > 0:
            self.perceptual_loss: Optional[PerceptualLoss] = PerceptualLoss()
        else:
            self.perceptual_loss = None

    def forward(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns
        -------
        total_loss : scalar tensor (backprop through this)
        loss_dict  : dict with individual component values for logging
        """
        l1 = self.l1_loss(sr, hr)
        total = self.l1_weight * l1
        loss_dict = {"l1": l1.item()}

        if self.perceptual_loss is not None:
            perceptual = self.perceptual_loss(sr, hr)
            total = total + self.perceptual_weight * perceptual
            loss_dict["perceptual"] = perceptual.item()

        loss_dict["total"] = total.item()
        return total, loss_dict


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_loss(config) -> CombinedLoss:
    """
    Build the loss function from the config module.

    Args
    ----
    config : module with LOSS_L1_WEIGHT and LOSS_PERCEPTUAL_WEIGHT attributes
    """
    return CombinedLoss(
        l1_weight=config.LOSS_L1_WEIGHT,
        perceptual_weight=config.LOSS_PERCEPTUAL_WEIGHT,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loss functions self-test")

    sr = torch.rand(2, 3, 96, 96)
    hr = torch.rand(2, 3, 96, 96)

    # L1 only
    loss_fn = CombinedLoss(l1_weight=1.0, perceptual_weight=0.0)
    total, info = loss_fn(sr, hr)
    print(f"  L1 loss: {info['l1']:.4f}  total: {info['total']:.4f}")
    print("  ✓ CombinedLoss (L1 only) passed.")
