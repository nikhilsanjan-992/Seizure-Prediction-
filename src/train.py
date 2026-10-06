"""NeuroSelect training utilities (Phase 5).

Provides reusable, memory-safe pieces for the Phase 6 training run:

- model creation + compilation           (reuses src/model.py)
- class weights computed from TRAINING labels ONLY
- EarlyStopping + ModelCheckpoint (restore_best_weights)
- tf.data batching to avoid huge in-RAM tensors
- model saving under models/

NOTHING in this module trains a model automatically, and DEMO_MODE is not
invoked unless explicitly requested by the user.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from config import MODEL_DIR
from model import build_cnn_bilstm, compile_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Training configuration (used ONLY when the user explicitly starts training)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42                 # mirrors Phase 4 reproducibility setting
DEMO_MODE = False                # switch to True for a tiny smoke run

EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2
EARLY_STOP_PATIENCE = 5

# DEMO_MODE bounds: small real-data subset, tiny budget, clearly labeled.
DEMO_EPOCHS = 2
DEMO_BATCH_SIZE = 8
DEMO_MAX_SEGMENTS = 40


def set_training_seed(seed: int = RANDOM_SEED) -> None:
    """Central reproducibility hook (delegates to model.set_reproducible_seed)."""
    from model import set_reproducible_seed

    set_reproducible_seed(seed)


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Balanced class weights computed from training labels ONLY.

    Never call this with validation/test labels - class weights are part of
    the training setup and must not leak evaluation info into training.
    """
    from sklearn.utils.class_weight import compute_class_weight

    y = np.asarray(y)
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(cls): float(w) for cls, w in zip(classes, weights)}


def make_callbacks(
    model_name: str,
    monitor: str = "val_loss",
    patience: int = EARLY_STOP_PATIENCE,
    out_dir: str | Path = MODEL_DIR,
):
    """EarlyStopping (restore_best_weights) + ModelCheckpoint (best only)."""
    from tensorflow import keras

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"{model_name}.keras"
    return [
        keras.callbacks.EarlyStopping(
            monitor=monitor, patience=patience, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path), monitor=monitor,
            save_best_only=True, verbose=1,
        ),
    ]


def build_dataset(X: np.ndarray, y: np.ndarray, batch_size: int = BATCH_SIZE):
    """Lazy tf.data.Dataset (batch + prefetch) to keep RAM usage low.

    Prefetch uses AUTOTUNE; on a ~7.9 GB usable machine batching is modest.
    """
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def train_cnn_bilstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    num_channels: int,
    timesteps: int,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    patience: int = EARLY_STOP_PATIENCE,
    model_name: str | None = None,
):
    """Create, compile and fit a CNN-BiLSTM model (explicit call required).

    ``X_train``/``X_val`` must already be in model orientation
    ``(n, timesteps, num_channels)`` (see model.to_model_orientation) and
    y arrays must be 1-D binary labels. Returns ``(model, history)``.
    """
    from model import validate_model_arrays

    validate_model_arrays(X_train, y_train, num_channels, timesteps)
    if X_val is not None:
        validate_model_arrays(X_val, y_val, num_channels, timesteps)

    name = model_name or f"cnn_bilstm_{num_channels}"
    set_training_seed()

    model = compile_model(
        build_cnn_bilstm(num_channels=num_channels, timesteps=timesteps)
    )
    class_weight = compute_class_weights(y_train)

    train_ds = build_dataset(X_train, y_train, batch_size=batch_size)
    if X_val is not None:
        val_ds = build_dataset(X_val, y_val, batch_size=batch_size)
    else:
        val_ds = None
    callbacks = make_callbacks(name, patience=patience)

    if val_ds is not None:
        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        history = model.fit(
            train_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
    return model, history


def save_model(model, num_channels: int, out_dir: str | Path = MODEL_DIR) -> Path:
    """Save a model as ``cnn_bilstm_<num_channels>.keras``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cnn_bilstm_{num_channels}.keras"
    model.save(path)
    return path


def run_demo_training(
    X_demo: np.ndarray,
    y_demo: np.ndarray,
    num_channels: int,
    timesteps: int,
):
    """Explicitly-requested DEMO smoke run (yes, it trains a little).

    Uses at most ``DEMO_MAX_SEGMENTS`` real segments and ``DEMO_EPOCHS``
    epochs. Results are DEMO only and must never be cited as research
    results. Never invoked automatically.
    """
    if not DEMO_MODE:
        raise RuntimeError(
            "run_demo_training() called while DEMO_MODE is False; flip the flag "
            "first (explicit user action)."
        )
    X_demo = np.asarray(X_demo)
    y_demo = np.asarray(y_demo)
    X_demo = X_demo[:DEMO_MAX_SEGMENTS]
    y_demo = y_demo[:DEMO_MAX_SEGMENTS]

    from model import validate_model_arrays

    validate_model_arrays(X_demo, y_demo, num_channels, timesteps)

    # Tiny train/val split inside the demo subset (never reused elsewhere).
    split = int(len(X_demo) * 0.8)
    model, history = train_cnn_bilstm(
        X_demo[:split], y_demo[:split],
        X_demo[split:], y_demo[split:],
        num_channels, timesteps,
        epochs=DEMO_EPOCHS, batch_size=DEMO_BATCH_SIZE,
        model_name=f"demo_cnn_bilstm_{num_channels}",
    )
    return model, history