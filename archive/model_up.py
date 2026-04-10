"""Super-Resolution model using Deep Residual Networks and Sub-Pixel Convolution.

Architecture (EDSR-lite style)
------------------------------
Input   : Low-Resolution Image (e.g., 150x150x3)
Head    : Conv2D(64)
Body    : ResidualBlock(64) x 8-16 (No pooling! Spatial dimensions stay identical)
Tail    : Conv2D(3 * scale^2) -> SubPixelConv2D (Pixel Shuffle)
Output  : High-Resolution Image (e.g., 300x300x3)
"""

import tensorflow as tf
import keras
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    Input,
    Activation,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ---------------------------------------------------------------------------
# Custom Layer: The core of efficient upscaling
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SubPixelConv2D(keras.layers.Layer):
    """Pixel Shuffle layer for Super-Resolution.
    
    Takes a tensor of shape (B, H, W, C * scale^2) and rearranges the depth 
    into spatial dimensions, resulting in (B, H * scale, W * scale, C).
    """
    def __init__(self, scale_factor=2, **kwargs):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.scale_factor)

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0], 
            input_shape[1] * self.scale_factor, 
            input_shape[2] * self.scale_factor, 
            input_shape[3] // (self.scale_factor ** 2)
        )

    def get_config(self):
        return {**super().get_config(), "scale_factor": self.scale_factor}


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

def residual_block_sr(x, filters: int):
    """Modified ResNet block for Super-Resolution.
    
    Note: Standard SR models (like EDSR) often remove Batch Normalization 
    from the residual blocks because BN alters the color/contrast distributions 
    which hurts pixel-perfect SR generation.
    """
    shortcut = x

    out = Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    out = Conv2D(filters, (3, 3), padding="same")(out)

    out = Add()([out, shortcut])
    return out


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

def build_sr_model(scale: int = 2, num_res_blocks: int = 8, input_shape: tuple = (150, 150, 3)) -> Model:
    """Build and compile the super-resolution model.

    Parameters
    ----------
    scale : int
        The upscaling factor (e.g., 2 for 2x upscale).
    num_res_blocks : int
        Depth of the network. More blocks = better quality but slower inference.
    input_shape : tuple
        Shape of the Low-Resolution input patch.
    """
    inp = Input(shape=input_shape)

    # 1. Head (Extract initial features)
    x = Conv2D(64, (3, 3), padding="same")(inp)
    x_head = x  # Save for a global skip connection

    # 2. Body (Deep feature extraction in LR space)
    for _ in range(num_res_blocks):
        x = residual_block_sr(x, filters=64)

    # Global skip connection (adds the original image features back in)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = Add()([x, x_head])

    # 3. Tail (Upscale)
    # To output 3 channels (RGB) after pixel shuffle, we need 3 * scale^2 filters.
    # For a 2x scale, we need 3 * (2^2) = 12 filters.
    out_filters = 3 * (scale ** 2)
    x = Conv2D(out_filters, (3, 3), padding="same")(x)
    
    out = SubPixelConv2D(scale_factor=scale)(x)
    
    # Cast to float32 if you are using mixed precision, and bound to [0,1]
    out = Activation("sigmoid", dtype="float32")(out)

    model = Model(inputs=inp, outputs=out)
    
    # MAE (L1 loss) is standard for SR to prevent blurry edges
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="mae")
    model.summary()
    return model