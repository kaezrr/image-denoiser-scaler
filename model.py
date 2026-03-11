"""Convolutional autoencoder model for image denoising."""

from tensorflow.keras.layers import (  # type: ignore
    BatchNormalization,
    Conv2D,
    Input,
    MaxPooling2D,
    UpSampling2D,
)
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore


def build_autoencoder(input_shape: tuple = (300, 300, 3)) -> Model:
    """Build and compile the convolutional denoising autoencoder.

    Architecture
    ------------
    Encoder:
        Conv2D(64, 3, relu, same) → MaxPooling2D → BatchNorm
        Conv2D(32, 3, relu, same) → MaxPooling2D → BatchNorm

    Decoder:
        Conv2D(32, 3, relu, same) → UpSampling2D
        Conv2D(64, 3, relu, same) → UpSampling2D
        Conv2D(3,  3, sigmoid, same)

    Parameters
    ----------
    input_shape : tuple
        Shape of a single input image, default (300, 300, 3).

    Returns
    -------
    tensorflow.keras.Model
        Compiled autoencoder model.
    """
    inp = Input(shape=input_shape)

    # --- Encoder ---
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(inp)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)

    # --- Decoder ---
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D()(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D()(x)

    out = Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(), loss="mean_squared_error")
    model.summary()
    return model
