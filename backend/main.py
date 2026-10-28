"""
FastAPI Backend for Faster R-CNN Object Detection.
Serves REST and WebSocket endpoints for image, video, and live webcam stream inference
with explicit target FPS control.
"""

import os
import sys
import time
import base64
import uuid
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import FasterRCNNInferencer, VOC_CLASSES, CLASS_COLORS_BGR

from contextlib import asynccontextmanager

# Storage directories
UPLOAD_DIR = PROJECT_ROOT / "backend_storage" / "uploads"
PROCESSED_DIR = PROJECT_ROOT / "backend_storage" / "processed"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Sample directory
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "data" / "VOCdevkit" / "VOC2007" / "JPEGImages"
SAMPLE_VIDEO_PATH = PROJECT_ROOT / "data" / "sample_demo.mp4"

WEIGHTS_PATH = str(PROJECT_ROOT / "model_weight.pth")
inferencer: Optional[FasterRCNNInferencer] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global inferencer
    print(f"Loading Faster R-CNN model from: {WEIGHTS_PATH}...")
    try:
        inferencer = FasterRCNNInferencer(weights_path=WEIGHTS_PATH)
        print(f"Faster R-CNN model loaded successfully on device: {inferencer.device}")
    except Exception as e:
        print(f"Failed to initialize model inferencer: {e}")
        raise e
    yield
    print("Shutting down model inferencer...")

app = FastAPI(
    title="Faster R-CNN Detection API",
    description="High-performance object detection API supporting pictures, videos, and webcam streams.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/info")
def get_api_info():
    """Return model architecture info, device, and class mappings."""
    if inferencer is None:
        raise HTTPException(status_code=503, detail="Model is still initializing.")

    # Convert BGR to Hex colors for frontend
    palette_hex = []
    for b, g, r in CLASS_COLORS_BGR:
        palette_hex.append(f"#{r:02x}{g:02x}{b:02x}")

    return {
        "status": "ready",
        "model_name": "Faster R-CNN (ResNet-18 Backbone + RPN)",
        "num_classes": inferencer.num_classes,
        "classes": inferencer.classes,
        "colors": palette_hex,
        "device": str(inferencer.device),
        "cuda_available": str(inferencer.device).startswith("cuda")
    }


@app.get("/api/samples")
def get_sample_images():
    """Return pre-selected diverse VOC samples for quick testing."""
    sample_list = [
        {"id": "000200.jpg", "title": "Bottles & People", "category": "Indoor Scene"},
        {"id": "001594.jpg", "title": "Sheep, Dog & Person", "category": "Animals / Pasture"},
        {"id": "000782.jpg", "title": "Transit Bus", "category": "Vehicles"},
        {"id": "004605.jpg", "title": "Bicycles & Cyclists", "category": "Street"},
        {"id": "007470.jpg", "title": "Aeroplane", "category": "Aviation"},
        {"id": "008909.jpg", "title": "Automobile", "category": "Vehicles"}
    ]
    return {"samples": sample_list}


@app.get("/api/sample/{filename}")
def get_sample_image(filename: str):
    """Serve a sample image from the dataset."""
    file_path = SAMPLE_IMAGES_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found")
    return FileResponse(file_path, media_type="image/jpeg")


@app.get("/api/video/sample")
def get_sample_video():
    """Serve the demo test video."""
    if not SAMPLE_VIDEO_PATH.exists():
        raise HTTPException(status_code=404, detail="Sample video not found")
    return FileResponse(SAMPLE_VIDEO_PATH, media_type="video/mp4")


@app.post("/api/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.35, ge=0.01, le=1.0),
    nms_threshold: float = Query(0.35, ge=0.01, le=1.0)
):
    """
    Predict bounding boxes and classes for an uploaded photo.
    Returns detection coordinates, scores, counts, and base64 annotated image.
    """
    if inferencer is None:
        raise HTTPException(status_code=503, detail="Model inferencer is not loaded.")

    contents = await file.read()
    try:
        result = inferencer.predict_image(
            image_input=contents,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            return_annotated=True
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image inference error: {str(e)}")

    # Encode annotated image to JPEG base64
    annotated_bgr = result["annotated_image"]
    _, buffer = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    base64_str = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

    # Aggregate counts by class
    counts = {}
    for d in result["detections"]:
        cname = d["class_name"]
        counts[cname] = counts.get(cname, 0) + 1

    return {
        "filename": file.filename,
        "width": result["width"],
        "height": result["height"],
        "inference_time_ms": result["inference_time_ms"],
        "total_detections": len(result["detections"]),
        "counts": counts,
        "detections": result["detections"],
        "annotated_image": base64_str
    }


@app.post("/api/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    target_fps: float = Query(10.0, ge=1.0, le=30.0, description="Target processing frames per second"),
    conf_threshold: float = Query(0.35, ge=0.01, le=1.0),
    nms_threshold: float = Query(0.35, ge=0.01, le=1.0)
):
    """
    Upload and process a video file at the requested target FPS rate.
    Returns processing performance metrics and downloadable annotated video.
    """
    if inferencer is None:
        raise HTTPException(status_code=503, detail="Model inferencer is not loaded.")

    file_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename or "video.mp4").suffix or ".mp4"
    temp_input = UPLOAD_DIR / f"{file_id}_input{ext}"
    output_filename = f"{file_id}_annotated.webm"
    output_path = PROCESSED_DIR / output_filename

    # Save uploaded video
    with open(temp_input, "wb") as f:
        f.write(await file.read())

    # Process video with target_fps
    try:
        cap = cv2.VideoCapture(str(temp_input))
        if not cap.isOpened():
            raise ValueError("Could not open uploaded video")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        effective_fps = min(float(target_fps), src_fps)
        step = max(1.0, src_fps / effective_fps)

        # Use VP80 WebM for broad browser video playback compatibility
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        out_writer = cv2.VideoWriter(str(output_path), fourcc, effective_fps, (frame_w, frame_h))

        frame_idx = 0
        processed_count = 0
        next_target_idx = 0.0
        total_inf_time = 0.0
        class_summary = {}
        frame_stats = []

        t0 = time.perf_counter()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx >= int(round(next_target_idx)):
                pred = inferencer.predict_frame(frame, conf_threshold=conf_threshold, nms_threshold=nms_threshold)
                annotated = inferencer.draw_detections(frame, pred["detections"])

                # Draw statistics overlay
                cv2.putText(
                    annotated,
                    f"Target FPS: {effective_fps:.1f} | Latency: {pred['inference_time_ms']}ms",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                out_writer.write(annotated)
                total_inf_time += pred["inference_time_ms"]
                processed_count += 1
                next_target_idx += step

                for det in pred["detections"]:
                    cn = det["class_name"]
                    class_summary[cn] = class_summary.get(cn, 0) + 1

                frame_stats.append({
                    "frame": frame_idx,
                    "time_s": round(frame_idx / src_fps, 2),
                    "detections": len(pred["detections"]),
                    "latency_ms": pred["inference_time_ms"]
                })

            frame_idx += 1

        cap.release()
        out_writer.release()
        total_time = time.perf_counter() - t0

    except Exception as e:
        if temp_input.exists():
            temp_input.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")
    finally:
        if temp_input.exists():
            temp_input.unlink(missing_ok=True)

    avg_processing_fps = (processed_count / total_time) if total_time > 0 else 0.0

    return {
        "video_id": file_id,
        "source_fps": round(src_fps, 2),
        "target_fps": round(effective_fps, 2),
        "total_source_frames": frame_idx,
        "processed_frames": processed_count,
        "total_elapsed_s": round(total_time, 2),
        "avg_processing_fps": round(avg_processing_fps, 2),
        "video_url": f"/api/video/{output_filename}",
        "class_summary": class_summary,
        "frame_stats": frame_stats[:100]  # First 100 sample frames
    }


@app.get("/api/video/{filename}")
def get_processed_video(filename: str):
    """Stream or download a processed video."""
    video_path = PROCESSED_DIR / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Processed video not found")

    media_type = "video/webm" if filename.endswith(".webm") else "video/mp4"
    return FileResponse(video_path, media_type=media_type)


@app.websocket("/api/ws/webcam")
async def websocket_webcam_stream(websocket: WebSocket):
    """
    Real-time bidirectional WebSocket stream for webcam inference.
    Client sends JSON with JPEG base64 frame and desired target_fps.
    Server rate-limits to target_fps and responds with detections and annotated frame.
    """
    await websocket.accept()
    if inferencer is None:
        await websocket.close(code=1013, reason="Inferencer not ready")
        return

    last_processed_time = time.perf_counter()

    try:
        while True:
            data = await websocket.receive_text()
            req = json.loads(data)

            # Extract parameters
            img_b64 = req.get("image", "")
            target_fps = float(req.get("target_fps", 15.0))
            conf_threshold = float(req.get("conf_threshold", 0.35))
            nms_threshold = float(req.get("nms_threshold", 0.35))
            return_annotated = bool(req.get("return_annotated", True))

            min_interval = 1.0 / max(1.0, min(30.0, target_fps))
            now = time.perf_counter()
            elapsed_since_last = now - last_processed_time

            # Throttling to enforce target FPS
            if elapsed_since_last < min_interval:
                # Optionally tell client skipped or sleep briefly
                time.sleep(max(0.001, min_interval - elapsed_since_last))

            # Decode base64 image
            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]

            img_bytes = base64.b64decode(img_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame_bgr is None:
                await websocket.send_text(json.dumps({"error": "Failed to decode frame"}))
                continue

            t_start = time.perf_counter()
            pred = inferencer.predict_frame(
                frame_bgr,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold,
                max_dim=640,
                top_k_proposals=200
            )

            actual_fps = 1.0 / max(1e-4, (time.perf_counter() - last_processed_time))
            last_processed_time = time.perf_counter()

            resp = {
                "detections": pred["detections"],
                "inference_time_ms": pred["inference_time_ms"],
                "target_fps": target_fps,
                "actual_fps": round(actual_fps, 1),
                "width": pred["width"],
                "height": pred["height"]
            }

            if return_annotated:
                annotated = inferencer.draw_detections(frame_bgr, pred["detections"])
                _, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                resp["annotated_image"] = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

            await websocket.send_text(json.dumps(resp))

    except WebSocketDisconnect:
        print("Webcam WebSocket client disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
