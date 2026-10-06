"""NeuroSelect Phase 8 FastAPI backend.

Serves the frontend contract (see ``frontend/src/api.js``):

    GET  /health    -> {"status", "model_ready", "channels_ready", "metrics_ready"}
    GET  /channels  -> {"available", "top_5", "top_4"}
    GET  /metrics   -> {"available", "rows": [...]}  (parsed from comparison.csv)
    POST /predict   -> multipart "file" + "configuration" ("22"|"5"|"4")
                           -> {"prediction", "probability", "segments_analyzed",
                               "channels_used", "configuration", "model"}

The backend reuses the existing src/ pipeline for every step. It never generates
synthetic EEG, never fabricates predictions or metrics, never retrains, and never
modifies research results.

Inference notes (documented, real-pipeline behavior):

- ``/predict`` requires an actual trained model under ``models/cnn_bilstm_<n>.keras``
  (the files produced by ``scripts/run_experiment.py``). Without those files every
  endpoint reports the true availability and ``/predict`` returns 503.
- The 22-channel montage resolves from ``results/metrics/experiment_config.json``
  (``montages.22.channels``); if that artifact is absent it falls back to the
  structural channels common across the files currently in ``data/raw/``.
- The 5- and 4-channel montages resolve from ``results/metrics/selected_channels.json``
  (Phase 4 output, ``top_5``/``top_4``). Nothing is hard-coded.
- Preprocessing mirrors ``src/dataset._prepare_recording`` (crop -> normalize label
  matching -> channel selection -> band-pass 0.5-40 Hz + notch 50 Hz) and segmentation
  mirrors ``src/preprocessing.segment_eeg`` (10 s windows, 5 s overlap).
- A recording whose sampling rate differs from the one the trained model expects is
  resampled to the model's input sample rate so its windows match the model input shape.
- Normalization uses per-recording z-score statistics via the existing
  ``src/preprocessing`` functions. Phase 6 training normalized with training-set
  channel statistics that the current artifacts do not persist; this inference-time
  choice keeps inputs in the same standardized domain and is logged here for honesty.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config  # noqa: E402
import dataset as ds  # noqa: E402
import eeg_pipeline as ep  # noqa: E402
import preprocessing as pp  # noqa: E402

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

logger = logging.getLogger("neuroselect.backend")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

# Threshold used at Phase 6 evaluation (mirrors evaluate.DEFAULT_THRESHOLD = 0.5).
THRESHOLD = 0.5
VALID_CONFIGS = ("22", "5", "4")
ACCEPTED_SUFFIXES = (".edf", ".edf.gz")
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB safety cap on uploads

app = FastAPI(title="NeuroSelect API", version="0.1.0")

# The React dev server (Vite) defaults to these origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_MODEL_CACHE: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
class _Timings:
    """Lightweight per-stage timing accumulator (loaded/preprocessing/...)."""

    def __init__(self) -> None:
        self.stages: list[tuple[str, float]] = []

    def start(self, name: str) -> float:
        ts = time.perf_counter()
        self.stages.append([name, ts])
        return ts

    def stop(self, ts: float) -> float:
        return (time.perf_counter() - ts) * 1000.0

    def ms(self) -> dict[str, float]:
        return {name: round(ms, 2) for name, ms in self._durations()}

    def _durations(self) -> list[tuple[str, float]]:
        return [(n, self.stop(t)) for n, t in self.stages]


def _model_path(config_id: str) -> Path:
    return config.MODEL_DIR / f"cnn_bilstm_{config_id}.keras"


def _model_exists(config_id: str) -> bool:
    return _model_path(config_id).is_file()


def _load_model(config_id: str):
    if config_id in _MODEL_CACHE:
        logger.info("Model %s served from cache", config_id)
        return _MODEL_CACHE[config_id]
    path = _model_path(config_id)
    if not path.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Trained model for configuration '{config_id}' is not available: "
                f"expected {path.name} under models/. Run scripts/"
                "run_experiment.py after adding a dataset and Phase 4 selection."
            ),
        )
    import tensorflow as tf

    model = tf.keras.models.load_model(path, compile=False)
    _MODEL_CACHE[config_id] = model
    logger.info("Loaded model %s (%s)", path.name, path.stat().st_size)
    return model


def _selected_channels():
    """Phase 4 top-5 / top-4 or None (never invented)."""
    return ds.load_selected_channels()


def _channels_for(config_id: str) -> list[str] | None:
    if config_id in ("5", "4"):
        sel = _selected_channels()
        if not sel:
            return None
        key = "top_5" if config_id == "5" else "top_4"
        chans = sel.get(key)
        return list(chans) if chans else None
    # "22": real experiment montage first, else common across data/raw.
    exp_cfg = config.METRICS_DIR / "experiment_config.json"
    if exp_cfg.is_file():
        try:
            with open(exp_cfg, encoding="utf-8") as fh:
                payload = json.load(fh)
            chans = payload.get("montages", {}).get("22", {}).get("channels")
            if isinstance(chans, list) and chans:
                return [str(c) for c in chans]
        except (OSError, ValueError) as exc:
            logger.warning("Could not read experiment_config.json: %s", exc)
    files = ds.discover_dataset()
    if files:
        common = ds.common_channels_across(files)
        if common:
            return common
    return None


def _parse_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.upper() in ("NA", "NAN", "NONE"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _load_metrics_rows() -> list[dict]:
    path = config.METRICS_DIR / "comparison.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return []
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            row = {
                "configuration": raw.get("configuration", ""),
                "channels": _parse_float(raw.get("channels")),
                "accuracy": _parse_float(raw.get("accuracy")),
                "precision": _parse_float(raw.get("precision")),
                "recall": _parse_float(raw.get("recall")),
                "specificity": _parse_float(raw.get("specificity")),
                "f1": _parse_float(raw.get("f1")),
                "roc_auc": _parse_float(raw.get("roc_auc")),
                "test_samples": _parse_float(raw.get("test_samples")),
            }
            if row["configuration"]:
                rows.append(row)
    return rows


def _save_upload(file: UploadFile) -> tuple[Path, str]:
    name = (file.filename or "upload.edf").lower()
    if not name.endswith(ACCEPTED_SUFFIXES):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{Path(name).name}'. This endpoint accepts "
                "real EDF recordings only (.edf or .edf.gz)."
            ),
        )
    tmp_dir = Path(tempfile.mkdtemp(prefix="neuroselect_upload_"))
    if file.filename and file.filename.strip():
        suffix = ".edf.gz" if name.endswith(".edf.gz") else ".edf"
    else:
        suffix = ".edf"
    dest = tmp_dir / f"upload{suffix}"
    written = 0
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Uploaded file exceeds the 2 GiB safety cap.",
                    )
                out.write(chunk)
    except HTTPException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    except OSError as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.exception("Failed to store upload")
        raise HTTPException(
            status_code=500, detail="Could not store the uploaded file on disk."
        ) from exc
    if written == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return dest, tmp_dir


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Report the ACTUAL availability of models, channels and metrics."""
    models = {
        cfg: {"present": _model_exists(cfg)} for cfg in VALID_CONFIGS
    }
    dataset_present = bool(ds.discover_dataset())
    selection_present = _selected_channels() is not None
    metrics_present = _load_metrics_rows() != []
    return {
        "status": "ok",
        "model_ready": any(m["present"] for m in models.values()),
        "channels_ready": selection_present,
        "metrics_ready": metrics_present,
        "models": models,
        "dataset_present": dataset_present,
        "threshold": THRESHOLD,
    }


@app.get("/channels")
def channels():
    """Actual channel configurations (Phase 4 selection / experiment montage)."""
    top5 = _selected_channels().get("top_5") if _selected_channels() else None
    top4 = _selected_channels().get("top_4") if _selected_channels() else None
    return {
        "available": bool(top5 and top4),
        "top_5": top5 or [],
        "top_4": top4 or [],
        "montage_22": _channels_for("22") or [],
    }


@app.get("/metrics")
def metrics():
    """Read parsed rows from results/metrics/comparison.csv (real values only)."""
    rows = _load_metrics_rows()
    return {"available": rows != [], "rows": rows}


def _run_predict_sync(upload_path: Path, config_id: str, channels: list, model,
                      model_timesteps: int, model_channels: int) -> dict:
    """All blocking CPU/IO work for /predict, safe to call in a thread pool."""
    from model import to_model_orientation  # noqa: E402

    timings = _Timings()
    total_start = timings.start("total")
    tmp_dir_inner = upload_path.parent

    try:
        try:
            raw = ep.load_edf(upload_path, preload=False)
        except Exception as exc:
            logger.exception("EDF parse failed")
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not read the uploaded EDF file. It may be corrupted "
                    "or not a valid EDF/EDF+GZ recording."
                )
            ) from exc

        try:
            duration = float(ep.get_duration(raw))
            if duration <= 0:
                raise HTTPException(
                    status_code=422, detail="The recording is empty (0 s)."
                )
            if duration > ds.MAX_SECONDS_PER_RECORDING:
                logger.info("Cropping recording to %s s", ds.MAX_SECONDS_PER_RECORDING)
                raw.crop(tmax=ds.MAX_SECONDS_PER_RECORDING)

            picked, _ignored, missing = pp.select_channels(
                raw, requested=channels
            )
            if picked is None or missing:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Missing required channels for configuration '{config_id}': "
                        f"{', '.join(missing)}. Expected {', '.join(channels)}."
                    ),
                )

            sfreq = float(ep.get_sampling_frequency(picked))
            target_sfreq = round(model_timesteps / ds.WINDOW_SECONDS, 4)
            if abs(sfreq - target_sfreq) > 1e-3:
                logger.info(
                    "Resampling recording %s Hz -> %s Hz to match the trained model",
                    sfreq, target_sfreq,
                )
                picked.resample(target_sfreq, verbose=False)
                sfreq = target_sfreq

            filtered = pp.filter_eeg(
                picked, copy=False,
                low_freq=ds.LOW_FREQ, high_freq=ds.HIGH_FREQ,
                notch_freq=ds.NOTCH_FREQ,
            )
            data = np.asarray(filtered.get_data(), dtype=np.float64)

            prep_ts = timings.start("preprocessing")
            mean, std = pp.compute_normalization_stats(data)
            data = pp.apply_normalization(data, mean, std)
            logger.info(
                "Normalized %d channels with per-recording z-score "
                "(training-set channel statistics are not persisted in current "
                "artifacts; this choice keeps inputs in the same standardized domain)",
                data.shape[0],
            )
            timings.stop(prep_ts)

            seg_ts = timings.start("segmentation")
            try:
                segments = pp.segment_eeg(
                    data,
                    window_size=ds.WINDOW_SECONDS,
                    overlap=ds.OVERLAP_SECONDS,
                    sfreq=sfreq,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Recording too short to segment: {exc}",
                ) from exc
            timings.stop(seg_ts)

            X = to_model_orientation(segments).astype(np.float32)
            n_segments = X.shape[0]
            if model_timesteps != X.shape[1] or model_channels != X.shape[2]:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Prepared segments have shape {tuple(X.shape)} but the model "
                        f"expects (batch, {model_timesteps}, {model_channels})."
                    ),
                )

            infer_ts = timings.start("inference")
            probs: list[np.ndarray] = []
            step = 64
            for i in range(0, n_segments, step):
                batch = X[i : i + step]
                probs.append(np.asarray(model.predict(batch, verbose=0)).reshape(-1))
            probability = float(np.concatenate(probs).mean())
            timings.stop(infer_ts)

            total_ms = timings.stop(total_start)
            prediction = "seizure" if probability >= THRESHOLD else "non_seizure"
            logger.info(
                "predict %s: %d channels, %d segments, prediction=%s prob=%.4f",
                config_id, len(channels), n_segments, prediction, probability,
            )

            stage_ms = timings.ms()
            stage_ms["total"] = round(total_ms, 2)
            return {
                "prediction": prediction,
                "probability": round(probability, 4),
                "segments_analyzed": int(n_segments),
                "channels_used": list(channels),
                "configuration": config_id,
                "model": "CNN-BiLSTM",
                "timings_ms": stage_ms,
            }
        finally:
            try:
                raw.close()
            except Exception:
                pass
    finally:
        shutil.rmtree(tmp_dir_inner, ignore_errors=True)
        logger.info("Cleaned up temporary upload directory: %s", tmp_dir_inner)


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    configuration: str = Form(...),
):
    """Predict seizure/non-seizure on a REAL uploaded EDF recording.

    Heavy CPU/IO work (MNE filtering + TF inference) is dispatched to a thread
    pool via run_in_executor so it never blocks the uvicorn event loop.
    """
    config_id = configuration.strip()
    if config_id not in VALID_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported configuration '{configuration}'. "
                f"Expected one of {', '.join(VALID_CONFIGS)}."
            ),
        )

    channels = _channels_for(config_id)
    if not channels:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No real channel configuration is available for '{config_id}' "
                "channels. Place EDF data and run scripts/select_electrodes.py "
                "(Phase 4) so selected_channels.json exists."
            ),
        )

    model = _load_model(config_id)
    input_shape = getattr(model, "input_shape", None)
    if not input_shape:
        raise HTTPException(
            status_code=500,
            detail="The loaded model does not expose an input shape; "
                   "it cannot be used for inference.",
        )
    model_timesteps = int(input_shape[1])
    model_channels = int(input_shape[2])
    if model_channels != len(channels):
        raise HTTPException(
            status_code=500,
            detail=(
                f"Model for '{config_id}' expects {model_channels} input "
                f"channels but the recorded montage has {len(channels)} "
                f"({', '.join(channels)}). The model artifact and the Phase 4 "
                "selection are inconsistent; provide a matching model."
            ),
        )

    # Save upload to temp file (fast IO is fine here)
    upload_path, tmp_dir = _save_upload(file)

    # Dispatch all heavy CPU/IO work to a thread pool so the event loop is free.
    loop = asyncio.get_event_loop()
    import functools
    try:
        result = await loop.run_in_executor(
            None,
            functools.partial(
                _run_predict_sync,
                upload_path, config_id, channels, model,
                model_timesteps, model_channels,
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in predict worker")
        raise HTTPException(
            status_code=500,
            detail=f"Internal prediction error: {exc}",
        ) from exc
    return result