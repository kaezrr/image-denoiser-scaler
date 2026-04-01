"""Super-Resolution model — EDSR-lite with combined MAE + SSIM loss.

Architecture (unchanged from baseline)
---------------------------------------
Input   : Degraded LR patch  (150×150×3)
Head    : Conv2D(64)
Body    : ResidualBlock(64) × num_res_blocks  — no BN, no pooling
Tail    : Conv2D(12) → SubPixelConv2D(scale=2) → sigmoid
Output  : Clean HR patch     (300×300×3)

What changed
------------
Loss function
    Was:  MAE only
    Now:  0.8 × MAE  +  0.2 × (1 − SSIM)

    MAE keeps pixel accuracy sharp.
    SSIM loss pushes the model toward preserving edges and texture
    structure rather than just minimising average pixel error, which
    tends to produce blurry outputs.

Metrics
    PSNR and SSIM are logged every epoch as Keras metrics so you can
    watch perceptual quality improve in real time during training,
    rather than only seeing raw loss.

Custom objects (required for load_model)
    SubPixelConv2D  — pixel shuffle upscaling layer
"""

import tensorflow as tf
import keras
from tensorflow.keras.layers import Add, Conv2D, Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# ---------------------------------------------------------------------------
# Custom layer: pixel shuffle upscaling
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SubPixelConv2D(keras.layers.Layer):
    """Rearrange depth channels into spatial dims (pixel shuffle / depth-to-space).

    Input shape:  (B, H,      W,      C × scale²)
    Output shape: (B, H×scale, W×scale, C)
    """

    def __init__(self, scale_factor: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.scale_factor)

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            input_shape[1] * self.scale_factor if input_shape[1] else None,
            input_shape[2] * self.scale_factor if input_shape[2] else None,
            input_shape[3] // (self.scale_factor ** 2) if input_shape[3] else None,
        )

    def get_config(self):
        return {**super().get_config(), "scale_factor": self.scale_factor}


# ---------------------------------------------------------------------------
# SSIM loss + metrics
# ---------------------------------------------------------------------------

def ssim_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Differentiable SSIM loss: 1 − mean_SSIM over the batch.

    Returns a scalar in [0, 2]. Lower is better (0 = perfect).
    We use max_val=1.0 because images are normalised to [0,1].
    """
    return 1.0 - tf.reduce_mean(
        tf.image.ssim(y_true, y_pred, max_val=1.0)
    )


def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """0.8 × MAE  +  0.2 × (1 − SSIM).

    The 80/20 split keeps the loss dominated by pixel accuracy (which
    trains stably) while the SSIM term provides a structural gradient
    that prevents blurry outputs. Adjust the weights if you want more
    aggressive sharpening (push SSIM weight toward 0.4).
    """
    mae  = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim = ssim_loss(y_true, y_pred)
    return 0.8 * mae + 0.2 * ssim


class PSNRMetric(keras.metrics.Metric):
    """Keras metric wrapper for PSNR — logs mean PSNR over each epoch.

    PSNR = 10 × log10(1 / MSE)

    Higher is better. Typical values:
        < 25 dB  — noticeable degradation
        25–32 dB — acceptable quality
        > 32 dB  — good quality
    """

    def __init__(self, name: str = "psnr", **kwargs):
        super().__init__(name=name, **kwargs)
        self._sum   = self.add_weight(name="sum",   initializer="zeros")
        self._count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        psnr = tf.image.psnr(y_true, y_pred, max_val=1.0)
        # psnr is (batch,) — average over the batch then accumulate
        self._sum.assign_add(tf.reduce_sum(psnr))
        self._count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self._sum / self._count

    def reset_state(self):
        self._sum.assign(0.0)
        self._count.assign(0.0)


class SSIMMetric(keras.metrics.Metric):
    """Keras metric wrapper for SSIM — logs mean SSIM over each epoch.

    SSIM ∈ [0, 1]. Higher is better. Values above 0.9 are generally
    considered perceptually good.
    """

    def __init__(self, name: str = "ssim", **kwargs):
        super().__init__(name=name, **kwargs)
        self._sum   = self.add_weight(name="sum",   initializer="zeros")
        self._count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
        self._sum.assign_add(tf.reduce_sum(ssim))
        self._count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self._sum / self._count

    def reset_state(self):
        self._sum.assign(0.0)
        self._count.assign(0.0)


# ---------------------------------------------------------------------------
# Residual block (no BN — standard for SR)
# ---------------------------------------------------------------------------

def residual_block_sr(x, filters: int):
    """EDSR-style residual block — Conv → ReLU → Conv → Add.

    Batch Normalization is intentionally omitted: BN normalises feature
    statistics which can wash out the fine colour/contrast information that
    SR models need to preserve. EDSR showed this hurts perceptual quality.
    """
    shortcut = x
    out = Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    out = Conv2D(filters, (3, 3), padding="same")(out)
    return Add()([out, shortcut])


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

def build_sr_model(
    scale: int = 2,
    num_res_blocks: int = 8,
    input_shape: tuple = (150, 150, 3),
) -> Model:
    """Build and compile the robust EDSR-lite SR model.

    Parameters
    ----------
    scale : int
        Upscaling factor. 2 → 150×150 to 300×300.
    num_res_blocks : int
        Number of residual blocks in the body. 8 is the default lite config.
        Increase to 16 for higher quality at the cost of inference speed.
    input_shape : tuple
        LR input patch shape. Must match dataset_sr.LR_PATCH_SIZE.

    Returns
    -------
    keras.Model
        Compiled model with combined_loss, PSNRMetric, SSIMMetric.
    """
    inp = Input(shape=input_shape)

    # Head — initial feature extraction
    x      = Conv2D(64, (3, 3), padding="same")(inp)
    x_head = x   # global skip connection anchor

    # Body — deep feature extraction in LR space (no spatial downsampling)
    for _ in range(num_res_blocks):
        x = residual_block_sr(x, filters=64)

    # Global skip — add head features back so the body only needs to learn
    # the residual improvement, not reconstruct everything from scratch
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = Add()([x, x_head])

    # Tail — pixel shuffle upscaling
    # Need 3 × scale² output channels so depth_to_space gives 3-channel HR
    out_filters = 3 * (scale ** 2)   # = 12 for scale=2
    x   = Conv2D(out_filters, (3, 3), padding="same")(x)
    out = SubPixelConv2D(scale_factor=scale)(x)

    # Cast to float32 for mixed-precision safety, clip to [0,1]
    out = Activation("sigmoid", dtype="float32")(out)

    model = Model(inputs=inp, outputs=out)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=combined_loss,
        metrics=[PSNRMetric(), SSIMMetric()],
    )
    model.summary()
    return model