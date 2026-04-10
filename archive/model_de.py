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
"""

import tensorflow as tf  # type: ignore
import keras             # type: ignore
from tensorflow.keras import mixed_precision  # type: ignore

mixed_precision.set_global_policy("mixed_float16")

from tensorflow.keras.layers import (  # type: ignore
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Input,
    MaxPooling2D,
    UpSampling2D,
    Activation,
)
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore


# ---------------------------------------------------------------------------
# Custom layers — replacing Lambda so models save/load without safe_mode
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class CropToMatch(keras.layers.Layer):
    """Center-crop the skip tensor so its H×W matches the target tensor.

    Why this exists
    ---------------
    MaxPooling on odd spatial dims (e.g. 75→37 after pool, 74 after
    upsample) causes a 1-pixel mismatch at each Concatenate in the decoder.
    Rather than padding (distorts features) or resizing (interpolation
    artefacts), we symmetrically trim the skip tensor to match.

    Replaces the Lambda(_crop_to_match) that caused load errors.
    """

    def call(self, inputs):
        skip, target = inputs
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

    def compute_output_shape(self, input_shape):
        # Output spatial dims match the target (second input), channels from skip
        skip_shape, target_shape = input_shape
        return (skip_shape[0], target_shape[1], target_shape[2], skip_shape[3])

    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable()
class ResizeTo(keras.layers.Layer):
    """Bilinear resize to a fixed (H, W) — used at the decoder output.

    Why this exists
    ---------------
    After three rounds of MaxPool+Upsample on odd spatial dims, the output
    can be 296×296 instead of 300×300.  A final bilinear resize guarantees
    an exact match to the input shape regardless of crop accumulation.

    Replaces the Lambda(tf.image.resize) that caused load errors.
    """

    def __init__(self, target_h: int, target_w: int, **kwargs):
        super().__init__(**kwargs)
        self.target_h = target_h
        self.target_w = target_w

    def call(self, x):
        return tf.image.resize(x, (self.target_h, self.target_w))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.target_h, self.target_w, input_shape[3])

    def get_config(self):
        # get_config lets Keras serialize this layer properly — this is what
        # makes load_model() work without safe_mode=False
        return {
            **super().get_config(),
            "target_h": self.target_h,
            "target_w": self.target_w,
        }


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _bn_relu(x):
    x = BatchNormalization()(x)
    return Activation("relu")(x)


def inception_block(x, filters: int):
    """GoogLeNet-style Inception module with four parallel branches.

    branch_1 : 1×1 conv          (cheap channel mixing)
    branch_2 : 1×1 bottleneck → 3×3 conv
    branch_3 : 1×1 bottleneck → 5×5 conv
    branch_4 : 3×3 MaxPool    → 1×1 conv

    Total output channels = 4 × filters.
    """
    f4 = max(filters // 4, 1)

    b1 = Conv2D(filters, (1, 1), padding="same")(x)
    b1 = _bn_relu(b1)

    b2 = Conv2D(f4, (1, 1), padding="same")(x)
    b2 = _bn_relu(b2)
    b2 = Conv2D(filters, (3, 3), padding="same")(b2)
    b2 = _bn_relu(b2)

    b3 = Conv2D(f4, (1, 1), padding="same")(x)
    b3 = _bn_relu(b3)
    b3 = Conv2D(filters, (5, 5), padding="same")(b3)
    b3 = _bn_relu(b3)

    b4 = MaxPooling2D((3, 3), strides=(1, 1), padding="same")(x)
    b4 = Conv2D(filters, (1, 1), padding="same")(b4)
    b4 = _bn_relu(b4)

    return Concatenate()([b1, b2, b3, b4])


def residual_block(x, filters: int):
    """ResNet identity block — learns residual correction F(x), adds to x.

    A 1×1 projection is added to the shortcut when channel count changes
    so Add() dimensions always match.
    """
    shortcut = x

    out = Conv2D(filters, (3, 3), padding="same")(x)
    out = _bn_relu(out)
    out = Conv2D(filters, (3, 3), padding="same")(out)
    out = BatchNormalization()(out)

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
    e1     = inception_block(inp, filters=8)    # → 32 ch
    skip_1 = e1
    e1     = MaxPooling2D((2, 2))(e1)

    # Stage 2 — two Residual blocks  [150×150 → 75×75]
    e2     = residual_block(e1, filters=64)
    e2     = residual_block(e2, filters=64)
    skip_2 = e2
    e2     = MaxPooling2D((2, 2))(e2)

    # Stage 3 — Inception  [75×75 → 38×38]
    e3     = inception_block(e2, filters=32)    # → 128 ch
    skip_3 = e3
    e3     = MaxPooling2D((2, 2))(e3)

    # ------------------------------------------------------------------
    # Bottleneck  [38×38]
    # ------------------------------------------------------------------
    bn = Conv2D(256, (3, 3), padding="same")(e3)
    bn = _bn_relu(bn)

    # ------------------------------------------------------------------
    # Decoder  (upsample → crop-and-concat skip → conv)
    # ------------------------------------------------------------------

    # Stage 1  [38 → 76]
    d1 = UpSampling2D((2, 2))(bn)
    d1 = Concatenate()([CropToMatch()([skip_3, d1]), d1])
    d1 = Conv2D(128, (3, 3), padding="same")(d1)
    d1 = _bn_relu(d1)

    # Stage 2  [76 → 152]
    d2 = UpSampling2D((2, 2))(d1)
    d2 = Concatenate()([CropToMatch()([skip_2, d2]), d2])
    d2 = Conv2D(64, (3, 3), padding="same")(d2)
    d2 = _bn_relu(d2)

    # Stage 3  [152 → 304 → resized to 300]
    d3 = UpSampling2D((2, 2))(d2)
    d3 = Concatenate()([CropToMatch()([skip_1, d3]), d3])
    d3 = Conv2D(32, (3, 3), padding="same")(d3)
    d3 = _bn_relu(d3)

    # ------------------------------------------------------------------
    # Output — resize back to exact input size, cast to float32
    # ------------------------------------------------------------------
    out = Conv2D(3, (3, 3), padding="same", dtype="float32")(d3)
    out = ResizeTo(input_shape[0], input_shape[1])(out)
    out = Activation("sigmoid")(out)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
    model.summary()
    return model