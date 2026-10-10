"""
Whisper large-v3 speech-to-text service

POST /transcribe             multipart: file, language[, task, batch_size]  -> plain text
POST /transcribe/timestamps  multipart: file, language[, task, batch_size]  -> text + timed segments
GET  /health

Changes for benchmarking (search for "BENCH" to find them):
  * optional `batch_size` form field overrides BATCH_SIZE for that one request
  * every response carries a `metrics` object (timings, VRAM, GPU utilisation, tokens)
  * the PyTorch cache is released before and after every request, so VRAM numbers
    belong to that request only and idle memory goes back to other processes (Ollama)
"""
from __future__ import annotations

import logging
import logging.handlers
import math
import os
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

# Number GPUs the same way nvidia-smi does (must be set before torch initialises CUDA)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import av
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# LOGGING SETUP
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.getenv("LOG_FILE", os.path.join(LOG_DIR, "whisper.log"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s:%(process)d] %(message)s"


def _setup_logging() -> None:
    """Console handler plus a rotating file handler on the root logger."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Clear anything a previous basicConfig()/import may have installed
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    try:
        log_dir = os.path.dirname(os.path.abspath(LOG_FILE))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
        root.info("Logging to file: %s", os.path.abspath(LOG_FILE))
    except Exception:
        # Never let a bad log path take the service down
        root.exception("Could not open log file %r - continuing with console only", LOG_FILE)

    logging.captureWarnings(True)


_setup_logging()
log = logging.getLogger("whisper")


def _handle_uncaught(exc_type, exc_value, exc_tb) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    log.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))


sys.excepthook = _handle_uncaught

if hasattr(threading, "excepthook"):
    def _handle_thread_exc(args):
        log.critical(
            "Uncaught exception in thread %s",
            getattr(args.thread, "name", "?"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = _handle_thread_exc


def _env_int(name: str, default: int) -> int:
    """int(os.getenv(...)) but logs and falls back instead of dying at import time."""
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        log.warning("Invalid %s=%r, falling back to %d", name, raw, default)
        return default


MODEL_NAME = os.getenv("WHISPER_MODEL", "./model/openai/whisper-large-v3")
BATCH_SIZE = _env_int("BATCH_SIZE", 16)            # default: 30s chunks decoded in parallel
MAX_BATCH_SIZE = _env_int("MAX_BATCH_SIZE", 32)    # BENCH: upper limit for the per-request override
CHUNK_LENGTH_S = _env_int("CHUNK_LENGTH_S", 30)    # Whisper's native window
MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 4096)    # 0 = no limit
MAX_AUDIO_SECONDS = _env_int("MAX_AUDIO_SECONDS", 8 * 3600)  # 0 = no limit
HOST = os.getenv("HOST", "0.0.0.0")
PORT = _env_int("PORT", 8788)

# GPUs to try, in order of preference (nvidia-smi indices). The first one that exists
# and has at least MIN_FREE_GPU_MB free is used.
GPU_PREFERENCE = os.getenv("GPU_PREFERENCE", "1,0")
MIN_FREE_GPU_MB = _env_int("MIN_FREE_GPU_MB", 12000)

# Fail startup if the warmup pass fails (set to 0 to only warn)
WARMUP_FATAL = os.getenv("WARMUP_FATAL", "1").lower() not in ("0", "false", "no")

# BENCH: seconds between GPU utilisation samples while a transcription runs (0 = off)
GPU_SAMPLE_INTERVAL_S = float(os.getenv("GPU_SAMPLE_INTERVAL_S", "1.0"))

SAMPLE_RATE = 16000  # what Whisper expects
TASKS = ("transcribe", "translate")

if BATCH_SIZE < 1:
    log.warning("BATCH_SIZE=%d is invalid, using 1", BATCH_SIZE)
    BATCH_SIZE = 1
if MAX_BATCH_SIZE < BATCH_SIZE:
    MAX_BATCH_SIZE = BATCH_SIZE
if not 1 <= CHUNK_LENGTH_S <= 30:
    log.warning("CHUNK_LENGTH_S=%d is invalid, using 30", CHUNK_LENGTH_S)
    CHUNK_LENGTH_S = 30

# Language table for validating the `language` field. If the import path moves in a
# future transformers release we just skip validation and let generate() complain.
try:
    from transformers.models.whisper.tokenization_whisper import LANGUAGES, TO_LANGUAGE_CODE
except Exception:
    log.warning("Whisper language table not importable - language codes will not be pre-validated")
    LANGUAGES, TO_LANGUAGE_CODE = {}, {}

# Populated during startup
STATE: dict = {"pipe": None, "device": None, "error": None}

# One transcription on the model at a time; other requests wait their turn
_INFER_LOCK = threading.Lock()


class AudioError(ValueError):
    """The upload could not be turned into audio (client error)."""


def _is_oom(exc: BaseException) -> bool:
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def normalize_language(raw: str) -> str:
    """Accepts 'ja', 'JA', 'japanese' ... returns the Whisper language code."""
    lang = (raw or "").strip().lower()
    if not lang:
        raise ValueError("language must not be empty")
    if not LANGUAGES:
        return lang
    if lang in LANGUAGES:
        return lang
    if lang in TO_LANGUAGE_CODE:
        return TO_LANGUAGE_CODE[lang]
    raise ValueError(f"unsupported language {raw!r}; use a Whisper code such as 'en' or 'ja'")


# BENCH: GPU utilisation sampling
class GpuSampler:
    """
    Samples utilisation (%) of the whole GPU with nvidia-smi while a transcription runs.
    The figure covers every process on that card, so keep Ollama idle while benchmarking.
    Assumes torch's device index equals the nvidia-smi index (true with
    CUDA_DEVICE_ORDER=PCI_BUS_ID and no CUDA_VISIBLE_DEVICES remapping).
    """

    def __init__(self, device: str, interval_s: float = GPU_SAMPLE_INTERVAL_S):
        self.index = device.split(":", 1)[1] if device.startswith("cuda:") else None
        self.interval_s = interval_s
        self.samples: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _read(self) -> Optional[float]:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi", f"--id={self.index}",
                    "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=5, check=True,
            )
            return float(out.stdout.strip().splitlines()[0])
        except Exception:
            return None

    def _run(self) -> None:
        while not self._stop.is_set():
            value = self._read()
            if value is None:
                return  # nvidia-smi unusable: stop sampling, never disturb the request
            self.samples.append(value)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "GpuSampler":
        if self.index is not None and self.interval_s > 0:
            self._thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def stats(self) -> Dict[str, Any]:
        if not self.samples:
            return {"gpu_util_avg": None, "gpu_util_max": None, "gpu_util_samples": 0}
        return {
            "gpu_util_avg": round(sum(self.samples) / len(self.samples), 1),
            "gpu_util_max": max(self.samples),
            "gpu_util_samples": len(self.samples),
        }


def _estimate_chunks(duration_s: float) -> int:
    """How many windows the pipeline cuts the audio into (default stride = chunk/6 per side)."""
    if duration_s <= CHUNK_LENGTH_S:
        return 1
    step = CHUNK_LENGTH_S - 2 * (CHUNK_LENGTH_S / 6)
    return int(math.ceil((duration_s - CHUNK_LENGTH_S) / step)) + 1


def _count_tokens(pipe, segments: List[dict]) -> Optional[int]:
    """Text tokens in the final transcript (timestamp/special tokens are not counted)."""
    try:
        texts = [seg["text"] for seg in segments]
        if not texts:
            return 0
        ids = pipe.tokenizer(texts, add_special_tokens=False)["input_ids"]
        return int(sum(len(x) for x in ids))
    except Exception:
        log.warning("Could not count output tokens", exc_info=True)
        return None


# AUDIO DECODING
def decode_audio(source) -> np.ndarray:
    """
    Decode any audio/video container (mp3, wav, m4a, flac, ogg, mp4, mkv, webm ...) to
    16 kHz mono float32. Uses the FFmpeg libraries bundled inside PyAV, so no system
    ffmpeg is needed. For video files only the audio track is read.
    `source` is a path or a seekable binary file object.
    """
    try:
        container = av.open(source, mode="r")
    except (av.error.FFmpegError, OSError, ValueError) as exc:
        raise AudioError(f"not a readable audio/video file ({type(exc).__name__}: {exc})") from exc

    max_samples = MAX_AUDIO_SECONDS * SAMPLE_RATE if MAX_AUDIO_SECONDS > 0 else None
    chunks: List[np.ndarray] = []
    n_samples = 0
    skipped = 0

    try:
        if not container.streams.audio:
            raise AudioError("file contains no audio stream")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)

        def _collect(frames) -> None:
            nonlocal n_samples
            for out in frames:
                arr = out.to_ndarray().reshape(-1)
                chunks.append(arr)
                n_samples += arr.shape[0]
            if max_samples is not None and n_samples > max_samples:
                raise AudioError(
                    f"audio is longer than the {MAX_AUDIO_SECONDS}s limit (MAX_AUDIO_SECONDS)"
                )

        for packet in container.demux(stream):
            try:
                frames = packet.decode()
            except av.error.InvalidDataError:
                skipped += 1  # corrupt packet: drop it, keep going
                continue
            for frame in frames:
                _collect(resampler.resample(frame))
        _collect(resampler.resample(None))  # flush the resampler
    except AudioError:
        raise
    except (av.error.FFmpegError, OSError, ValueError) as exc:
        raise AudioError(f"failed to decode audio ({type(exc).__name__}: {exc})") from exc
    finally:
        container.close()

    if skipped:
        log.warning("Skipped %d corrupt packet(s) while decoding", skipped)
    if n_samples == 0:
        raise AudioError("file contains no decodable audio")

    return np.concatenate(chunks).astype(np.float32) / 32768.0


def pick_device() -> str:
    """First GPU in GPU_PREFERENCE with enough free memory; 'cpu' if there is no CUDA."""
    if not torch.cuda.is_available():
        return "cpu"

    count = torch.cuda.device_count()
    wanted: List[int] = []
    for part in GPU_PREFERENCE.split(","):
        part = part.strip()
        if part.isdigit() and int(part) < count and int(part) not in wanted:
            wanted.append(int(part))
        elif part:
            log.warning("GPU_PREFERENCE entry %r ignored (%d GPU(s) visible)", part, count)
    if not wanted:
        wanted = list(range(count))

    free_mb = {}
    for idx in wanted:
        try:
            free, total = torch.cuda.mem_get_info(idx)
            free_mb[idx] = free // (1024 * 1024)
            log.info("GPU %d: %d MB free of %d MB", idx, free_mb[idx], total // (1024 * 1024))
        except Exception:
            log.warning("Could not query GPU %d - skipping it", idx, exc_info=True)

    for idx in wanted:
        if free_mb.get(idx, 0) >= MIN_FREE_GPU_MB:
            return f"cuda:{idx}"

    raise RuntimeError(
        f"No GPU in GPU_PREFERENCE={GPU_PREFERENCE!r} has {MIN_FREE_GPU_MB} MB free "
        f"(free MB: {free_mb})"
    )


# MODEL LOADING
def load_model() -> None:
    try:
        device = pick_device()

        if device == "cpu":
            torch.set_num_threads(os.cpu_count() or 4)

        log.info("Loading %s on %s ...", MODEL_NAME, device)
        t0 = time.perf_counter()

        try:
            processor = AutoProcessor.from_pretrained(MODEL_NAME)
        except Exception:
            log.exception("Failed to load processor from %r", MODEL_NAME)
            raise

        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32
            )
        except Exception:
            log.exception("Failed to load model weights from %r", MODEL_NAME)
            raise

        try:
            model.to(device)
            model.eval()
        except Exception as exc:
            if _is_oom(exc):
                log.exception("Out of memory moving model to %s", device)
            else:
                log.exception("Failed to move model to %s", device)
            raise

        try:
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=device,
            )
        except Exception:
            log.exception("Failed to build ASR pipeline")
            raise

        STATE.update(pipe=pipe, device=device, error=None)
        log.info(
            "Loaded in %.1fs (default batch_size=%d, chunk_length=%ds)",
            time.perf_counter() - t0, BATCH_SIZE, CHUNK_LENGTH_S
        )
    except Exception as exc:
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        log.critical("Model load failed - service cannot serve requests", exc_info=True)
        raise

    # Warm up model on one second of silence
    try:
        transcribe_audio(np.zeros(SAMPLE_RATE, dtype=np.float32), "en", "transcribe")
        log.info("Warmup complete. Ready")
    except Exception:
        log.exception("Warmup inference failed")
        if WARMUP_FATAL:
            STATE["error"] = "warmup failed"
            raise
        log.warning("Continuing despite warmup failure (WARMUP_FATAL=0)")


def transcribe_audio(
    audio: np.ndarray, language: str, task: str, batch_size: Optional[int] = None
) -> dict:
    """
    Long-form transcription: the audio is cut into overlapping 30s windows, the windows
    are decoded `batch_size` at a time, and the pieces are stitched back together using
    Whisper's timestamp tokens.
    Returns {"text": str, "segments": [{start, end, text}], "metrics": {...}}.
    """
    pipe = STATE["pipe"]
    device = STATE["device"]

    if pipe is None:
        raise RuntimeError("Model is not loaded")

    batch = BATCH_SIZE if batch_size is None else int(batch_size)
    duration = audio.shape[0] / SAMPLE_RATE
    on_gpu = device.startswith("cuda")
    metrics: Dict[str, Any] = {"batch_size": batch, "n_chunks_est": _estimate_chunks(duration)}

    t_queue = time.perf_counter()
    with _INFER_LOCK:
        metrics["queue_wait_ms"] = round((time.perf_counter() - t_queue) * 1000, 1)
        sampler = GpuSampler(device)
        try:
            if on_gpu:
                # BENCH: start from an empty cache so the peaks below belong to this request
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            t_inf = time.perf_counter()
            with sampler:
                result = pipe(
                    {"raw": audio, "sampling_rate": SAMPLE_RATE},
                    chunk_length_s=CHUNK_LENGTH_S,
                    batch_size=batch,
                    return_timestamps=True,
                    generate_kwargs={"language": language, "task": task},
                )
            metrics["infer_ms"] = round((time.perf_counter() - t_inf) * 1000, 1)

            if on_gpu:
                gb = 1024 ** 3
                metrics["vram_peak_gb"] = round(torch.cuda.max_memory_allocated(device) / gb, 2)
                metrics["vram_reserved_peak_gb"] = round(torch.cuda.max_memory_reserved(device) / gb, 2)
                metrics["vram_allocated_after_gb"] = round(torch.cuda.memory_allocated(device) / gb, 2)
            metrics.update(sampler.stats())
        except Exception as exc:
            if _is_oom(exc):
                log.error(
                    "OOM during transcription (audio=%.0fs, batch_size=%d, device=%s). "
                    "Consider lowering the batch size.",
                    duration, batch, device,
                    exc_info=True,
                )
            else:
                log.exception("Inference failed (audio=%.0fs, batch_size=%d)", duration, batch)
            raise
        finally:
            if on_gpu:
                # Hand cached memory back so other processes on the card (Ollama) can use it
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    log.warning("torch.cuda.empty_cache() failed", exc_info=True)

    try:
        segments = []
        for chunk in result.get("chunks") or []:
            start, end = chunk.get("timestamp") or (None, None)
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            # Whisper sometimes leaves the very last end timestamp open
            start = float(start) if start is not None else (segments[-1]["end"] if segments else 0.0)
            end = float(end) if end is not None else duration
            segments.append({"start": round(start, 2), "end": round(end, 2), "text": text})

        metrics["output_tokens"] = _count_tokens(pipe, segments)
        log.info("metrics audio=%.0fs %s", duration, metrics)
        return {
            "text": (result.get("text") or "").strip(),
            "segments": segments,
            "metrics": metrics,
        }
    except Exception:
        log.exception("Post-processing failed (result type=%s)", type(result).__name__)
        raise


# API

class TranscribeResponse(BaseModel):
    model: str
    language: str
    task: str
    duration_s: float
    text: str
    took_ms: float
    batch_size: int                              # BENCH
    metrics: Optional[Dict[str, Any]] = None     # BENCH

class Segment(BaseModel):
    start: float
    end: float
    text: str

class TimestampResponse(TranscribeResponse):
    segments: List[Segment]

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except Exception:
        log.critical("Startup aborted", exc_info=True)
        raise
    try:
        yield
    finally:
        log.info("Shutting down, clearing model state")
        try:
            STATE.clear()
        except Exception:
            log.exception("Error while clearing state on shutdown")

app = FastAPI(title="Whisper large-v3 STT", version="1.1", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    log.warning("422 validation error on %s %s: %s", request.method, request.url.path, exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def on_unhandled_error(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}"},
    )


@app.get("/health")
def health():
    try:
        return {
            "status": "ok" if STATE.get("pipe") is not None else "loading",
            "model": MODEL_NAME,
            "device": STATE.get("device"),
            "batch_size": BATCH_SIZE,
            "max_batch_size": MAX_BATCH_SIZE,
            "chunk_length_s": CHUNK_LENGTH_S,
            "busy": _INFER_LOCK.locked(),
            "error": STATE.get("error"),
        }
    except Exception:
        log.exception("health check failed")
        raise HTTPException(500, "health check failed")


def _upload_size(upload: UploadFile) -> int:
    if upload.size is not None:
        return upload.size
    upload.file.seek(0, os.SEEK_END)
    size = upload.file.tell()
    upload.file.seek(0)
    return size


def _handle(upload: UploadFile, language: str, task: str, batch_size: Optional[int] = None) -> dict:
    """Shared by both endpoints: validate -> decode -> transcribe."""
    name = upload.filename or "<unnamed>"
    try:
        if STATE.get("pipe") is None:
            log.warning("503 transcribe rejected - model not loaded (error=%s)", STATE.get("error"))
            raise HTTPException(503, "Model is still loading")

        task = (task or "").strip().lower()
        if task not in TASKS:
            log.warning("400 transcribe rejected - bad task %r", task)
            raise HTTPException(400, f"task must be one of {list(TASKS)}")
        try:
            lang = normalize_language(language)
        except ValueError as exc:
            log.warning("400 transcribe rejected - %s", exc)
            raise HTTPException(400, str(exc))

        # BENCH: per-request batch size
        batch = BATCH_SIZE if batch_size is None else batch_size
        if not 1 <= batch <= MAX_BATCH_SIZE:
            log.warning("400 transcribe rejected - bad batch_size %r", batch_size)
            raise HTTPException(400, f"batch_size must be between 1 and {MAX_BATCH_SIZE}")

        size = _upload_size(upload)
        if size == 0:
            log.warning("400 transcribe rejected - empty file %r", name)
            raise HTTPException(400, "uploaded file is empty")
        if MAX_UPLOAD_MB > 0 and size > MAX_UPLOAD_MB * 1024 * 1024:
            log.warning("413 transcribe rejected - %r is %.0f MB", name, size / 1e6)
            raise HTTPException(413, f"file is larger than {MAX_UPLOAD_MB} MB")

        t0 = time.perf_counter()
        try:
            upload.file.seek(0)
            audio = decode_audio(upload.file)
        except AudioError as exc:
            log.warning("400 transcribe rejected - %r: %s", name, exc)
            raise HTTPException(400, str(exc))
        duration = audio.shape[0] / SAMPLE_RATE
        decode_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "decoded %r (%.1f MB) -> %.1fs of audio in %.0fms, transcribing "
            "(language=%s, task=%s, batch_size=%d)",
            name, size / 1e6, duration, decode_ms, lang, task, batch
        )

        try:
            out = transcribe_audio(audio, lang, task, batch)
        except Exception as exc:
            log.exception(
                "transcribe failed after %.0fms (file=%r, audio=%.0fs, batch_size=%d)",
                (time.perf_counter() - t0) * 1000, name, duration, batch
            )
            if _is_oom(exc):
                raise HTTPException(503, "Out of memory while transcribing; lower the batch size")
            raise HTTPException(500, f"Transcription failed: {type(exc).__name__}")
        took_ms = (time.perf_counter() - t0) * 1000

        metrics = dict(out.get("metrics") or {})
        metrics["decode_ms"] = round(decode_ms, 1)

        log.info(
            "transcribed %r: %.1fs audio in %.0fms (%.1fx realtime, batch_size=%d, "
            "%d segments, %d chars)",
            name, duration, took_ms, duration / max(took_ms / 1000, 1e-6), batch,
            len(out["segments"]), len(out["text"])
        )
        return {
            "model": MODEL_NAME,
            "language": lang,
            "task": task,
            "duration_s": round(duration, 2),
            "text": out["text"],
            "segments": out["segments"],
            "took_ms": took_ms,
            "batch_size": batch,
            "metrics": metrics,
        }
    finally:
        try:
            upload.file.close()  # also deletes the spooled temp file
        except Exception:
            log.warning("Could not close upload %r", name, exc_info=True)


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(
    file: UploadFile = File(..., description="Audio or video file"),
    language: str = Form(..., description="Spoken language, e.g. 'en', 'ja'"),
    task: str = Form("transcribe", description="'transcribe' or 'translate' (to English)"),
    batch_size: Optional[int] = Form(None, description="Chunks decoded in parallel (default: BATCH_SIZE)"),
):
    return _handle(file, language, task, batch_size)


@app.post("/transcribe/timestamps", response_model=TimestampResponse)
def transcribe_timestamps(
    file: UploadFile = File(..., description="Audio or video file"),
    language: str = Form(..., description="Spoken language, e.g. 'en', 'ja'"),
    task: str = Form("transcribe", description="'transcribe' or 'translate' (to English)"),
    batch_size: Optional[int] = Form(None, description="Chunks decoded in parallel (default: BATCH_SIZE)"),
):
    return _handle(file, language, task, batch_size)


if __name__ == "__main__":
    try:
        # log_config=None keeps our handlers, so uvicorn/access logs land in the file too
        uvicorn.run(app, host=HOST, port=PORT, log_level="info", log_config=None)
    except KeyboardInterrupt:
        log.info("Interrupted, exiting")
    except Exception:
        log.critical("Server exited with an unhandled exception", exc_info=True)
        raise
