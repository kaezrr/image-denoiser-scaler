"""Hybrid Inception-ResNet autoencoder for image denoising / super-resolution.

Architecture
------------
Encoder
    Stage 1 : InceptionBlock(64)   → MaxPool  [300→150]  — skip_1
    Stage 2 : ResidualBlock(128)×2 → MaxPool  [150→75]   — skip_2
    Stage 3 : InceptionBlock(256)  → MaxPool  [75→38]    — skip_3

Bottleneck
    Conv(512) → BN → ReLU                    [38×38]

Decoder  (U-Net style — concat skip at each level)
    Stage 1 : UpSample → concat(skip_3) → Conv(256)  [38→75]
    Stage 2 : UpSample → concat(skip_2) → Conv(128)  [75→150]
    Stage 3 : UpSample → concat(skip_1) → Conv(64)   [150→300]

Output  : Conv(3, sigmoid)                   [300×300×3]

Why this design
---------------
* Inception branches (1×1 / 3×3 / 5×5 / pool) capture multi-scale texture
  detail.  The 1×1 bottleneck before each large kernel keeps the parameter
  count manageable — this is the core GoogLeNet insight.
* Residual blocks let gradients flow cleanly through the deep encoder without
  vanishing — the ResNet insight.  They're placed in stage 2 where feature
  maps are mid-size, giving the most training benefit.
* U-Net skip connections bring spatial detail back into the decoder so it
  never has to reconstruct fine texture from the bottleneck alone — this is
  why the current simple decoder produces blurry results.
"""

import tensorflow as tf  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Input,
    MaxPooling2D,
    UpSampling2D,
    Activation,
    Lambda,
)
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------

def _crop_to_match(skip, target):
    """Center-crop `skip` so its H×W matches `target`.

    MaxPooling odd spatial dims (e.g. 75 → 37 after pool, 74 after upsample)
    causes a 1-pixel mismatch at each Concatenate.  Rather than padding or
    resizing (both distort features), we trim the skip tensor symmetrically.
    """
    sh = tf.shape(skip)
    th = tf.shape(target)
    dh = sh[1] - th[1]
    dw = sh[2] - th[2]
    return skip[
        :,
        dh // 2 : sh[1] - (dh - dh // 2),
        dw // 2 : sh[2] - (dw - dw // 2),
        :,
    ]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _bn_relu(x):
    """BatchNorm → ReLU helper."""
    x = BatchNormalization()(x)
    return Activation("relu")(x)


def inception_block(x, filters: int):
    """GoogLeNet-style Inception module.

    Runs four parallel branches and concatenates their outputs:
        branch_1 : 1×1 conv          (cheap, channel mixing)
        branch_2 : 1×1 bottleneck → 3×3 conv
        branch_3 : 1×1 bottleneck → 5×5 conv
        branch_4 : 3×3 MaxPool    → 1×1 conv  (pooling path)

    The 1×1 convolutions before 3×3/5×5 are the bottleneck that made
    GoogLeNet dramatically cheaper than a naïve multi-scale design.

    Parameters
    ----------
    x : tensor
        Input feature map.
    filters : int
        Number of output filters per branch.  Total output channels = 4×filters.
    """
    f4 = max(filters // 4, 1)      # bottleneck width

    # Branch 1 — direct 1×1
    b1 = Conv2D(filters, (1, 1), padding="same")(x)
    b1 = _bn_relu(b1)

    # Branch 2 — 1×1 bottleneck then 3×3
    b2 = Conv2D(f4, (1, 1), padding="same")(x)
    b2 = _bn_relu(b2)
    b2 = Conv2D(filters, (3, 3), padding="same")(b2)
    b2 = _bn_relu(b2)

    # Branch 3 — 1×1 bottleneck then 5×5
    b3 = Conv2D(f4, (1, 1), padding="same")(x)
    b3 = _bn_relu(b3)
    b3 = Conv2D(filters, (5, 5), padding="same")(b3)
    b3 = _bn_relu(b3)

    # Branch 4 — MaxPool then 1×1
    b4 = MaxPooling2D((3, 3), strides=(1, 1), padding="same")(x)
    b4 = Conv2D(filters, (1, 1), padding="same")(b4)
    b4 = _bn_relu(b4)

    return Concatenate()([b1, b2, b3, b4])   # → 4×filters channels


def residual_block(x, filters: int):
    """ResNet-style residual (identity) block.

    F(x) + x — the network learns only the *residual* correction rather
    than the full mapping.  This makes it much easier to learn near-identity
    transformations (e.g. "remove a little noise") and keeps gradients alive
    in deep stacks.

    A 1×1 projection conv is added to the shortcut when the channel count
    changes so the Add() dimensions always match.

    Parameters
    ----------
    x : tensor
        Input feature map.
    filters : int
        Number of output filters.
    """
    shortcut = x

    # Main path
    out = Conv2D(filters, (3, 3), padding="same")(x)
    out = _bn_relu(out)
    out = Conv2D(filters, (3, 3), padding="same")(out)
    out = BatchNormalization()(out)

    # Project shortcut if channel count differs
    if x.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding="same")(x)
        shortcut = BatchNormalization()(shortcut)

    out = Add()([out, shortcut])
    return Activation("relu")(out)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

def build_autoencoder(input_shape: tuple = (300, 300, 3)) -> Model:
    """Build and compile the hybrid Inception-ResNet denoising autoencoder.

    Parameters
    ----------
    input_shape : tuple
        Shape of a single input image.  Default (300, 300, 3).

    Returns
    -------
    tensorflow.keras.Model
        Compiled autoencoder ready for training.
    """
    inp = Input(shape=input_shape)

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    # Stage 1 — Inception  [300×300 → 150×150]
    e1 = inception_block(inp, filters=16)       # → 64 ch (4×16)
    skip_1 = e1                                  # save for decoder
    e1 = MaxPooling2D((2, 2))(e1)

    # Stage 2 — two stacked Residual blocks  [150×150 → 75×75]
    e2 = residual_block(e1, filters=128)
    e2 = residual_block(e2, filters=128)
    skip_2 = e2
    e2 = MaxPooling2D((2, 2))(e2)

    # Stage 3 — Inception  [75×75 → 38×38]
    e3 = inception_block(e2, filters=64)        # → 256 ch (4×64)
    skip_3 = e3
    e3 = MaxPooling2D((2, 2))(e3)

    # ------------------------------------------------------------------
    # Bottleneck  [38×38]
    # ------------------------------------------------------------------
    bn = Conv2D(512, (3, 3), padding="same")(e3)
    bn = _bn_relu(bn)

    # ------------------------------------------------------------------
    # Decoder  (each level: upsample → concat skip → conv)
    # ------------------------------------------------------------------

    # Stage 1  [37 → 74 → concat → 75]
    d1 = UpSampling2D((2, 2))(bn)
    skip_3_cropped = Lambda(lambda t: _crop_to_match(t[0], t[1]))([skip_3, d1])
    d1 = Concatenate()([d1, skip_3_cropped])
    d1 = Conv2D(256, (3, 3), padding="same")(d1)
    d1 = _bn_relu(d1)

    # Stage 2  [74 → 148 → concat → 150]
    d2 = UpSampling2D((2, 2))(d1)
    skip_2_cropped = Lambda(lambda t: _crop_to_match(t[0], t[1]))([skip_2, d2])
    d2 = Concatenate()([d2, skip_2_cropped])
    d2 = Conv2D(128, (3, 3), padding="same")(d2)
    d2 = _bn_relu(d2)

    # Stage 3  [148 → 296 → concat → 300]
    d3 = UpSampling2D((2, 2))(d2)
    skip_1_cropped = Lambda(lambda t: _crop_to_match(t[0], t[1]))([skip_1, d3])
    d3 = Concatenate()([d3, skip_1_cropped])
    d3 = Conv2D(64, (3, 3), padding="same")(d3)
    d3 = _bn_relu(d3)

    # ------------------------------------------------------------------
    # Output — resize back to exact input spatial size if needed
    # ------------------------------------------------------------------
    out_pre = Conv2D(3, (3, 3), padding="same")(d3)
    # Bilinear resize guarantees exact H×W match regardless of crop accumulation
    out = Lambda(
        lambda t: tf.image.resize(t, (input_shape[0], input_shape[1]))
    )(out_pre)
    out = Activation("sigmoid")(out)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
    model.summary()
    return model