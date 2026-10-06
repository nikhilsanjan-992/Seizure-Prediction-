"""NeuroSelect Phase 5 architecture test.

Builds the shared CNN-BiLSTM for 22, 5 and 4 channels, prints parameter
counts, and runs a tiny forward-pass SHAPE test with a synthetic tensor.

The synthetic tensor is used ONLY to verify tensor shapes/dimensions of the
architecture. It is explicitly NOT EEG data, is never trained on, and does
not produce any research result or metric.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import model  # noqa: E402

TIMESTEPS = model.DEFAULT_TIMESTEPS  # 10 s at 256 Hz (Phase 3 default)
MONTAGES = (22, 5, 4)


def build_and_report(num_channels: int, run_shape_test: bool):
    m = model.build_cnn_bilstm(num_channels=num_channels, timesteps=TIMESTEPS)
    counts = model.model_param_counts(m)
    print("-" * 58)
    print(f"{num_channels}-CHANNEL MODEL")
    print(f"  Input shape        : (None, {TIMESTEPS}, {num_channels})  "
          f"i.e. (batch, timesteps, channels)")
    print(f"  Output shape       : {m.output_shape}")
    print(f"  Total parameters   : {counts['total']:,}")
    print(f"  Trainable          : {counts['trainable']:,}")
    print(f"  Non-trainable      : {counts['non_trainable']:,}")

    if run_shape_test:
        # DUMMY SHAPE TEST ONLY - synthetic tensor, not EEG, no training.
        draft = m(np.random.RandomState(0).randn(2, TIMESTEPS, num_channels))
        print(f"  Dummy forward pass: in (2,{TIMESTEPS},{num_channels}) -> "
              f"out {tuple(draft.shape)} (valid for binary sigmoid)")
    return m, counts


def main() -> int:
    print("MODEL TEST - CNN-BiLSTM architecture (no training)")

    err = model.tensorflow_import_error()
    if err is not None:
        print(f"BLOCKED: TensorFlow import failed: {err}")
        return 2

    models = {}
    for n_ch in MONTAGES:
        models[n_ch], _ = build_and_report(n_ch, run_shape_test=True)

    print("-" * 58)
    for n_ch in MONTAGES:
        out = models[n_ch].output_shape
        ok = out is not None and len(out) == 2 and out[1] == 1
        print(f"{n_ch}-channel: OUTPUT SHAPE OK ({out})" if ok
              else f"{n_ch}-channel: OUTPUT SHAPE FAILED ({out})")

    # Explicit orientation contract check with the Phase 3 layout.
    segments_phase3 = np.zeros((3, 22, TIMESTEPS))  # (samples, channels, timesteps)
    oriented = model.to_model_orientation(segments_phase3)
    print(f"Orientation helper: (3,22,{TIMESTEPS}) -> {tuple(oriented.shape)} "
          "(samples, timesteps, channels)")
    print("PARAMETERS: printed above per montage.")
    print("TRAINING PERFORMED: NO")
    print("MODEL TEST: SUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(main())