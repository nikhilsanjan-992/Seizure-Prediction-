"""NeuroSelect lightweight CNN-BiLSTM architecture.

Phase 5: one configurable binary-classification model for the 22/5/4
channel configurations. The SAME architecture is reused for all three
montages; only ``num_channels`` changes.

Data contract (must match Phase 3 pipeline output AFTER the documented
conversion in :func:`to_model_orientation`):

    model input  : (batch, timesteps, num_channels)
    model output : (batch, 1)  sigmoid in [0, 1] (seizure probability)

Phase 3 ``segment_eeg`` emits ``(samples, channels, timesteps)``; that is
NOT silently transposed inside the model - callers must use
:func:`to_model_orientation` and :func:`validate_model_arrays`.

Nothing in this module trains or evaluates on real EEG; no performance
metrics are computed.
"""

from __future__ import annotations

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
except Exception as exc:  # pragma: no cover - environment diagnostics only
    tf = None  # type: ignore[assignment]
    keras = None  # type: ignore[assignment]
    _TF_IMPORT_ERROR = exc
else:
    _TF_IMPORT_ERROR = None


# Default architecture sizes (lightweight; tuned for a student PC).
DEFAULT_CNN_FILTERS = (32, 64)
DEFAULT_CNN_KERNELS = (5, 3)
DEFAULT_LSTM_UNITS = 32
DEFAULT_DROPOUT = 0.3
DEFAULT_TIMESTEPS = 2560  # 10 s at 256 Hz (Phase 3 defaults)


def tensorflow_import_error():
    """Return the TensorFlow import error, or None if imports succeeded."""
    return _TF_IMPORT_ERROR


def set_reproducible_seed(seed: int = 42) -> None:
    """Set Python/NumPy/TensorFlow random seeds.

    Exact bit-level reproducibility may still depend on the specific
    hardware/software/TensorFlow backend.
    """
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)
        try:
            keras.utils.set_random_seed(seed)
        except Exception:
            pass


def build_cnn_bilstm(
    num_channels: int,
    timesteps: int = DEFAULT_TIMESTEPS,
    cnn_filters: tuple[int, int] = DEFAULT_CNN_FILTERS,
    cnn_kernels: tuple[int, int] = DEFAULT_CNN_KERNELS,
    lstm_units: int = DEFAULT_LSTM_UNITS,
    dropout: float = DEFAULT_DROPOUT,
    seed: int = 42,
):
    """Build the shared CNN-BiLSTM binary classifier.

    Architecture (input ``(timesteps, num_channels)``):

        Input
         -> 1D CNN (32) -> BatchNorm -> ReLU -> MaxPool
         -> 1D CNN (64) -> BatchNorm -> ReLU
         -> BiLSTM (lstm_units)
         -> Dropout
         -> Dense(1, sigmoid)

    The same architecture serves the 22-, 5- and 4-channel montages.
    """
    if _TF_IMPORT_ERROR is not None:
        raise RuntimeError(
            "TensorFlow is not importable; cannot build model. "
            f"Original error: {_TF_IMPORT_ERROR}"
        )
    if num_channels <= 0 or timesteps <= 0:
        raise ValueError(
            f"num_channels and timesteps must be positive "
            f"(got {num_channels}, {timesteps})."
        )

    set_reproducible_seed(seed)

    inputs = keras.Input(shape=(timesteps, num_channels), name="eeg_input")

    x = keras.layers.Conv1D(
        cnn_filters[0], cnn_kernels[0], padding="same", activation=None, name="conv1"
    )(inputs)
    x = keras.layers.BatchNormalization(name="bn1")(x)
    x = keras.layers.ReLU(name="relu1")(x)
    x = keras.layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    x = keras.layers.Conv1D(
        cnn_filters[1], cnn_kernels[1], padding="same", activation=None, name="conv2"
    )(x)
    x = keras.layers.BatchNormalization(name="bn2")(x)
    x = keras.layers.ReLU(name="relu2")(x)

    x = keras.layers.Bidirectional(
        keras.layers.LSTM(lstm_units, name="lstm"), name="bilstm"
    )(x)
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = keras.layers.Dense(
        1, activation="sigmoid", name="seizure_probability"
    )(x)

    model = keras.Model(inputs, outputs, name=f"cnn_bilstm_{num_channels}ch")
    return model


def compile_model(model, learning_rate: float = 1e-3):
    """Compile for binary seizure/non-seizure classification."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def model_param_counts(model) -> dict[str, int]:
    """Return total/trainable/non-trainable parameter counts."""
    total = int(model.count_params())
    trainable = int(sum(w.numpy().size for w in model.trainable_weights))
    non_trainable = total - trainable
    return {"total": total, "trainable": trainable, "non_trainable": non_trainable}


# ---------------------------------------------------------------------------
# Phase 3 -> Keras orientation helpers (explicit, validated, documented)
# ---------------------------------------------------------------------------
def to_model_orientation(segments: np.ndarray) -> np.ndarray:
    """Convert Phase 3 segments to the model's expected layout.

    Phase 3 ``segment_eeg`` output: ``(samples, channels, timesteps)``.
    Keras model input: ``(samples, timesteps, channels)``.
    This transposes time and channel axes EXPLICITLY.
    """
    segments = np.asarray(segments)
    if segments.ndim != 3:
        raise ValueError(
            f"Expected 3-D segments (samples, channels, timesteps); "
            f"got shape {segments.shape}."
        )
    return segments.transpose(0, 2, 1)


def validate_model_arrays(
    X: np.ndarray, y: np.ndarray, num_channels: int, timesteps: int
) -> None:
    """Validate that arrays match the model's input contract.

    ``X`` must have shape ``(n_samples, timesteps, num_channels)`` and
    ``y`` must be 1-D of matching length. Raises ``ValueError`` otherwise.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 3:
        raise ValueError(f"X must be 3-D (samples, timesteps, channels); got {X.shape}.")
    if X.shape[1] != timesteps:
        raise ValueError(
            f"X timesteps = {X.shape[1]}, expected {timesteps}."
        )
    if X.shape[2] != num_channels:
        raise ValueError(
            f"X channels = {X.shape[2]}, expected {num_channels}."
        )
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError(
            f"y must be 1-D with length {len(X)}; got shape {y.shape}."
        )