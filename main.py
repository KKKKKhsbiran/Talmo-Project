"""
Talmo model API server.

Serves Norwood-Hamilton hair loss grading predictions from an ONNX model.
Replaces the previous mock implementation while preserving the established
API contract: field names, types, and error codes are unchanged.

Model: v1.0.0-norwood-convnext-tiny
  - Validation exact accuracy 77.61%, QWK 0.9256, MAE 0.263 grades
  - Confidence threshold 0.60 gates roughly 30% of requests to retake
"""

import os
import io
import time
import logging

import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, UploadFile, Form, Header, File, Query
from fastapi.responses import JSONResponse
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
EXPECTED_TOKEN = os.environ.get("MODEL_API_TOKEN")
MODEL_PATH = os.environ.get("MODEL_PATH", "talmo_norwood_v1.onnx")

MODEL_VERSION = "v1.0.0-norwood-convnext-tiny"
NUM_CLASSES = 7
IMG_SIZE = 224
RESIZE_SIZE = 256

# Preprocessing constants. These MUST match the training pipeline exactly;
# any deviation shifts the input distribution and degrades accuracy silently.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# Derived from validation-set calibration: accuracy is 87.9% above this
# threshold versus 53.2% below it (near the 51.4% majority baseline).
CONFIDENCE_THRESHOLD = 0.60

MAX_UPLOAD_BYTES = 8 * 1024 * 1024   # 8MB, per the API contract

GRADE_VALUES = np.arange(1, NUM_CLASSES + 1, dtype=np.float64)
MIN_GRADE, MAX_GRADE = float(GRADE_VALUES.min()), float(GRADE_VALUES.max())

if not EXPECTED_TOKEN:
    logger.warning("MODEL_API_TOKEN is not set. All requests will be rejected.")

# -----------------------------------------------------------------------------
# Model loading (once at startup, not per request)
# -----------------------------------------------------------------------------
_session = None


def get_session():
    """Lazily initialize the ONNX Runtime session.

    Loading on first use rather than at import time keeps the container's
    startup memory footprint lower, which matters on constrained free-tier
    hosting.
    """
    global _session
    if _session is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1      # Free-tier containers have 1 vCPU
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        _session = ort.InferenceSession(
            MODEL_PATH,
            sess_options=options,
            providers=['CPUExecutionProvider'],
        )
        logger.info("ONNX session initialized: %s", MODEL_PATH)

    return _session


# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------
def preprocess(image: Image.Image) -> np.ndarray:
    """Replicate the training-time validation transform.

    Pipeline: Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize

    Returns:
        float32 array of shape (1, 3, 224, 224)
    """
    image = image.convert('RGB')

    # Resize so the shorter side becomes RESIZE_SIZE, preserving aspect ratio.
    # This mirrors torchvision.transforms.Resize(256) with an int argument.
    width, height = image.size
    if width < height:
        new_width = RESIZE_SIZE
        new_height = int(round(height * RESIZE_SIZE / width))
    else:
        new_height = RESIZE_SIZE
        new_width = int(round(width * RESIZE_SIZE / height))

    image = image.resize((new_width, new_height), Image.BILINEAR)

    # Center crop to IMG_SIZE x IMG_SIZE
    left = (new_width - IMG_SIZE) // 2
    top = (new_height - IMG_SIZE) // 2
    image = image.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))

    # ToTensor: HWC uint8 [0,255] -> CHW float32 [0,1]
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = array.transpose(2, 0, 1)

    # Normalize
    array = (array - IMAGENET_MEAN) / IMAGENET_STD

    return array[np.newaxis, ...].astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
def run_inference(image: Image.Image) -> dict:
    """Predict the Norwood grade and progression index for one image."""
    session = get_session()
    input_tensor = preprocess(image)

    logits = session.run(None, {'input': input_tensor})[0]
    probs = softmax(logits)[0].astype(np.float64)

    pred_idx = int(probs.argmax())
    stage = int(GRADE_VALUES[pred_idx])
    confidence = float(probs[pred_idx])

    # Continuous index from the softmax expected value.
    # Example: P(3)=0.6, P(4)=0.4 -> expected 3.4 -> index 40.0
    expected_grade = float((probs * GRADE_VALUES).sum())
    progress_index = (expected_grade - MIN_GRADE) / (MAX_GRADE - MIN_GRADE) * 100.0

    return {
        "stage": stage,
        "progress_index": round(progress_index, 1),
        "confidence": round(confidence, 4),
        "quality_pass": confidence >= CONFIDENCE_THRESHOLD,
    }


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Liveness probe. Also warms the model on first call."""
    try:
        get_session()
        model_ready = True
    except Exception as exc:
        logger.error("Model unavailable: %s", exc)
        model_ready = False

    return {
        "ok": model_ready,
        "model_version": MODEL_VERSION,
        "model_loaded": model_ready,
    }


@app.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    view: str = Form(...),
    request_id: str = Form(...),
    authorization: str = Header(None),
    force: str = Query(None),   # Test flag: ?force=quality_fail or ?force=error
):
    started = time.perf_counter()

    # --- Authorization -------------------------------------------------------
    expected_header = f"Bearer {EXPECTED_TOKEN}" if EXPECTED_TOKEN else None
    if not expected_header or authorization != expected_header:
        return JSONResponse(
            status_code=401,
            content={"ok": False,
                     "error": {"code": "UNAUTHORIZED", "message": "토큰이 없습니다."}},
        )

    # --- Forced test responses (retained for integration testing) ------------
    if force == "error":
        return JSONResponse(
            status_code=400,
            content={"ok": False,
                     "error": {"code": "IMAGE_TOO_LARGE",
                               "message": "Simulated error: Image exceeds 8MB."}},
        )

    if force == "quality_fail":
        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "request_id": request_id,
                "views_received": [view],
                "completeness": 1.0,
                "quality": {"pass": False, "issues": ["too_dark", "blurry"]},
                "stage": None,
                "indices": {},
                "progress_index": None,
                "confidence": None,
                "model_version": MODEL_VERSION,
            },
        )

    # --- Read and validate the upload ----------------------------------------
    raw = await image.read()

    if len(raw) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=400,
            content={"ok": False,
                     "error": {"code": "IMAGE_TOO_LARGE",
                               "message": "Image exceeds 8MB."}},
        )

    try:
        pil_image = Image.open(io.BytesIO(raw))
        pil_image.load()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"ok": False,
                     "error": {"code": "INVALID_IMAGE",
                               "message": "Unable to decode the uploaded file."}},
        )

    # --- Inference -----------------------------------------------------------
    try:
        result = run_inference(pil_image)
    except FileNotFoundError as exc:
        logger.error("Model file missing: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"ok": False,
                     "error": {"code": "MODEL_UNAVAILABLE",
                               "message": "Model is not loaded."}},
        )
    except Exception as exc:
        logger.exception("Inference failed for request_id=%s", request_id)
        return JSONResponse(
            status_code=500,
            content={"ok": False,
                     "error": {"code": "INFERENCE_FAILED",
                               "message": "Prediction could not be completed."}},
        )

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info("request_id=%s stage=%s confidence=%.3f elapsed=%.0fms",
                request_id, result["stage"], result["confidence"], elapsed_ms)

    # --- Low confidence: return a quality failure rather than a weak answer ---
    # Below the threshold, measured accuracy is 53.2%, effectively at the
    # majority-class baseline. Serving such predictions would mislead users.
    if not result["quality_pass"]:
        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "request_id": request_id,
                "views_received": [view],
                "completeness": 1.0,
                "quality": {"pass": False, "issues": ["low_confidence"]},
                "stage": None,
                "indices": {},
                "progress_index": None,
                "confidence": result["confidence"],
                "model_version": MODEL_VERSION,
                "processed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )

    # --- Success -------------------------------------------------------------
    return {
        "ok": True,
        "request_id": request_id,
        "views_received": [view],
        "completeness": 1.0,
        "quality": {"pass": True, "issues": []},
        "stage": result["stage"],
        "indices": {view: result["progress_index"]},
        "progress_index": result["progress_index"],
        "confidence": result["confidence"],
        "model_version": MODEL_VERSION,
        "processed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
