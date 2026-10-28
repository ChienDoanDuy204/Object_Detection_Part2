"""
Faster R-CNN Inference Module for Object Detection.
Supports single image prediction, video file/stream prediction, and live webcam stream
with configurable target FPS rate-limiting and benchmarking.
"""

import time
import os
from pathlib import Path
from typing import Union, List, Dict, Any, Generator, Optional, Tuple
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision.ops import batched_nms, clip_boxes_to_image
from PIL import Image

from model.FasterRCNN import FasterRCNN_model

# 20 Pascal VOC Classes in order of index (1-indexed, 0 is background)
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Curated harmonious color palette for 20 classes (BGR format for OpenCV)
CLASS_COLORS_BGR = [
    (245, 130, 48),    # aeroplane - orange
    (60, 180, 75),     # bicycle - green
    (230, 25, 75),     # bird - red
    (0, 130, 200),     # boat - blue
    (145, 30, 180),    # bottle - purple
    (70, 240, 240),    # bus - cyan
    (240, 50, 230),    # car - magenta
    (210, 245, 60),    # cat - lime
    (250, 190, 212),   # chair - pink
    (0, 128, 128),     # cow - teal
    (220, 190, 255),   # diningtable - lavender
    (170, 110, 40),    # dog - brown
    (255, 250, 200),   # horse - beige
    (128, 0, 0),       # motorbike - maroon
    (0, 255, 128),     # person - spring green
    (128, 128, 0),     # pottedplant - olive
    (0, 0, 128),       # sheep - navy
    (128, 128, 128),   # sofa - grey
    (255, 225, 25),    # train - yellow
    (0, 215, 255)      # tvmonitor - gold
]

# Normalization constants used during training
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


class FasterRCNNInferencer:
    """
    High-performance inference engine for Faster R-CNN with ResNet18 backbone.
    """
    def __init__(
        self,
        weights_path: str = "model_weight.pth",
        device: Optional[str] = None,
        classes: Optional[List[str]] = None
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.classes = classes or VOC_CLASSES
        self.num_classes = len(self.classes)
        self.colors = CLASS_COLORS_BGR

        # Instantiate model architecture
        self.model = FasterRCNN_model(num_classes=self.num_classes).to(self.device)

        # Load weights
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights file not found at: {weights_path}")

        weights = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(weights, strict=False)
        self.model.eval()

        # Normalization tensors on target device
        self.mean_t = torch.tensor(NORM_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std_t = torch.tensor(NORM_STD, device=self.device).view(1, 3, 1, 1)

        # Warmup model once
        self._warmup()

    def _warmup(self) -> None:
        """Run single dummy forward pass to compile graph and initialize CUDA kernels."""
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 300, 300, device=self.device)
                feat = self.model.backbone(dummy)
                delta = self.model.Regression_bbx_head(feat)
                score = self.model.classifier_head(feat)
                props = self.model.get_proposal(delta, score, (300, 300), top_k=50)
                _ = self.model.forward_fast_rcnn(feat, props)
        except Exception:
            pass

    def preprocess_image(
        self,
        img_rgb: np.ndarray,
        max_dim: Optional[int] = 800
    ) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[float, float]]:
        """
        Convert numpy RGB array into normalized Tensor (1, 3, H, W).
        Optionally rescales large images so maximum dimension <= max_dim for speed,
        returning scaling factors (scale_x, scale_y) to project boxes back to original size.
        """
        orig_h, orig_w = img_rgb.shape[:2]

        if max_dim is not None and max(orig_h, orig_w) > max_dim:
            scale = max_dim / float(max(orig_h, orig_w))
            new_w = int(round(orig_w * scale))
            new_h = int(round(orig_h * scale))
            img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img_rgb
            new_h, new_w = orig_h, orig_w

        scale_x = orig_w / float(new_w)
        scale_y = orig_h / float(new_h)

        # Convert to float tensor [0, 1]
        tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float().to(self.device).unsqueeze(0) / 255.0
        # Normalize
        tensor = (tensor - self.mean_t) / self.std_t

        return tensor, (new_h, new_w), (scale_x, scale_y)

    @torch.no_grad()
    def predict_frame(
        self,
        frame_bgr: np.ndarray,
        conf_threshold: float = 0.4,
        nms_threshold: float = 0.35,
        max_dim: Optional[int] = 800,
        top_k_proposals: int = 300
    ) -> Dict[str, Any]:
        """
        Predict objects on a single BGR OpenCV frame.
        
        Returns:
            Dictionary with detections list, inference_time_ms, and original frame size.
        """
        t_start = time.perf_counter()
        orig_h, orig_w = frame_bgr.shape[:2]

        # Convert BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor, (H_cur, W_cur), (scale_x, scale_y) = self.preprocess_image(frame_rgb, max_dim=max_dim)

        # Forward through Backbone & RPN
        feature_maps = self.model.backbone(tensor)
        proposal_delta = self.model.Regression_bbx_head(feature_maps)
        objectness_score = self.model.classifier_head(feature_maps)

        proposals = self.model.get_proposal(
            proposal_delta,
            objectness_score,
            (H_cur, W_cur),
            top_k=top_k_proposals,
            pre_nms_top_k=1000,
            iou_threshold=0.7
        )

        # Forward Fast R-CNN Head
        logits, box_deltas = self.model.forward_fast_rcnn(feature_maps, proposals)

        prop = proposals[0]
        if prop.shape[0] == 0:
            t_end = time.perf_counter()
            return {
                "detections": [],
                "inference_time_ms": round((t_end - t_start) * 1000, 2),
                "width": orig_w,
                "height": orig_h
            }

        # Decode refined bounding boxes
        w_roi = (prop[:, 2] - prop[:, 0]).clamp(min=1e-6)
        h_roi = (prop[:, 3] - prop[:, 1]).clamp(min=1e-6)
        cx_roi = prop[:, 0] + 0.5 * w_roi
        cy_roi = prop[:, 1] + 0.5 * h_roi
        rois_cxcywh = torch.stack([cx_roi, cy_roi, w_roi, h_roi], dim=-1)

        refined_boxes = self.model.decode_delta(rois_cxcywh, box_deltas)
        refined_boxes = clip_boxes_to_image(refined_boxes, (H_cur, W_cur))

        # Softmax probabilities over num_classes + 1 (background is index 0)
        scores = F.softmax(logits, dim=-1)
        fg_scores = scores[:, 1:]  # shape (N, 20)
        max_scores, class_indices = torch.max(fg_scores, dim=-1)

        # Confidence filter
        mask = max_scores >= conf_threshold
        filt_boxes = refined_boxes[mask]
        filt_scores = max_scores[mask]
        filt_classes = class_indices[mask]

        detections = []
        if filt_boxes.shape[0] > 0:
            keep = batched_nms(filt_boxes, filt_scores, filt_classes, iou_threshold=nms_threshold)
            final_boxes = filt_boxes[keep].cpu().numpy()
            final_scores = filt_scores[keep].cpu().numpy()
            final_classes = filt_classes[keep].cpu().numpy()

            for box, score, cls_id in zip(final_boxes, final_scores, final_classes):
                # Scale box back to original image dimensions
                x1 = float(max(0.0, min(orig_w, box[0] * scale_x)))
                y1 = float(max(0.0, min(orig_h, box[1] * scale_y)))
                x2 = float(max(0.0, min(orig_w, box[2] * scale_x)))
                y2 = float(max(0.0, min(orig_h, box[3] * scale_y)))

                cls_name = self.classes[int(cls_id)] if int(cls_id) < len(self.classes) else f"class_{cls_id}"
                detections.append({
                    "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "score": round(float(score), 4),
                    "class_id": int(cls_id) + 1,
                    "class_name": cls_name
                })

        t_end = time.perf_counter()
        return {
            "detections": detections,
            "inference_time_ms": round((t_end - t_start) * 1000, 2),
            "width": orig_w,
            "height": orig_h
        }

    def draw_detections(
        self,
        image_bgr: np.ndarray,
        detections: List[Dict[str, Any]],
        show_score: bool = True
    ) -> np.ndarray:
        """
        Draw clean bounding boxes, semi-transparent labels, and confidence tags.
        """
        annotated = image_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            cls_id = det["class_id"] - 1
            cls_name = det["class_name"]
            score = det["score"]

            color = self.colors[cls_id % len(self.colors)]

            # Draw outer rectangle with nice thickness
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Label text
            label = f"{cls_name} {score * 100:.1f}%" if show_score else cls_name
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1

            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            tag_y1 = max(0, y1 - th - 8)
            tag_y2 = y1
            tag_x2 = min(annotated.shape[1], x1 + tw + 10)

            # Draw label background tag
            cv2.rectangle(annotated, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1 + 4, tag_y2 - 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

        return annotated

    def predict_image(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image, bytes],
        conf_threshold: float = 0.4,
        nms_threshold: float = 0.35,
        return_annotated: bool = True
    ) -> Dict[str, Any]:
        """
        Predict objects for an image (path, numpy array, PIL Image, or raw bytes).
        """
        # Load image
        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input))
            if img_bgr is None:
                raise ValueError(f"Could not load image from: {image_input}")
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("Could not decode image bytes")
        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
            img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        res = self.predict_frame(img_bgr, conf_threshold=conf_threshold, nms_threshold=nms_threshold)
        if return_annotated:
            res["annotated_image"] = self.draw_detections(img_bgr, res["detections"])

        return res

    def predict_video(
        self,
        video_input: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        target_fps: Optional[float] = None,
        conf_threshold: float = 0.4,
        nms_threshold: float = 0.35,
        progress_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Predict objects in a video file with an optional target FPS parameter.
        If target_fps is provided, frames are sampled/processed to match the target FPS.
        """
        cap = cv2.VideoCapture(str(video_input))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_input}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Determine frame sampling step based on target_fps
        effective_fps = src_fps if (target_fps is None or target_fps <= 0) else min(float(target_fps), src_fps)
        step = max(1.0, src_fps / effective_fps)

        out_writer = None
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(str(output_path), fourcc, effective_fps, (frame_w, frame_h))

        frame_idx = 0
        processed_count = 0
        next_target_idx = 0.0
        frame_results = []
        total_inference_time = 0.0

        t0 = time.perf_counter()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx >= int(round(next_target_idx)):
                # Run inference on this frame
                pred = self.predict_frame(frame, conf_threshold=conf_threshold, nms_threshold=nms_threshold)
                annotated = self.draw_detections(frame, pred["detections"])

                # Draw current FPS watermark
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

                if out_writer is not None:
                    out_writer.write(annotated)

                total_inference_time += pred["inference_time_ms"]
                processed_count += 1
                next_target_idx += step

                frame_results.append({
                    "frame_idx": frame_idx,
                    "timestamp_s": round(frame_idx / src_fps, 3),
                    "detections": pred["detections"],
                    "inference_time_ms": pred["inference_time_ms"]
                })

                if progress_callback:
                    progress_callback(processed_count, total_frames, frame_idx)

            frame_idx += 1

        cap.release()
        if out_writer is not None:
            out_writer.release()

        total_elapsed = time.perf_counter() - t0
        avg_processing_fps = (processed_count / total_elapsed) if total_elapsed > 0 else 0.0

        return {
            "source_fps": round(src_fps, 2),
            "target_fps": round(effective_fps, 2),
            "total_source_frames": frame_idx,
            "processed_frames": processed_count,
            "total_elapsed_s": round(total_elapsed, 2),
            "avg_processing_fps": round(avg_processing_fps, 2),
            "output_path": str(output_path) if output_path else None,
            "frame_results": frame_results
        }

    def predict_video_stream(
        self,
        video_source: Union[int, str] = 0,
        target_fps: float = 15.0,
        conf_threshold: float = 0.4,
        nms_threshold: float = 0.35
    ) -> Generator[Tuple[np.ndarray, Dict[str, Any], float], None, None]:
        """
        Generator for video or webcam streaming with explicit FPS throttling.
        Yields (annotated_frame_bgr, detection_result, actual_fps).
        
        The FPS parameter controls how many frames per second the model processes.
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video stream source: {video_source}")

        target_interval = 1.0 / max(0.1, target_fps)
        last_process_time = time.perf_counter()
        fps_smooth = target_fps

        try:
            while cap.isOpened():
                now = time.perf_counter()
                elapsed = now - last_process_time
                if elapsed < target_interval:
                    time.sleep(max(0.001, target_interval - elapsed))
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                curr_time = time.perf_counter()
                actual_fps = 1.0 / max(1e-5, (curr_time - last_process_time))
                last_process_time = curr_time
                fps_smooth = 0.9 * fps_smooth + 0.1 * actual_fps

                pred = self.predict_frame(frame, conf_threshold=conf_threshold, nms_threshold=nms_threshold)
                annotated = self.draw_detections(frame, pred["detections"])

                # Overlay live stats
                cv2.putText(
                    annotated,
                    f"FPS: {fps_smooth:.1f} (Target: {target_fps:.1f}) | {pred['inference_time_ms']}ms",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                yield annotated, pred, fps_smooth
        finally:
            cap.release()


# Top-level functional interface as requested
def model_inference(
    mode: str,
    source: Union[str, int, np.ndarray, Image.Image, bytes],
    fps: Optional[float] = 15.0,
    conf_threshold: float = 0.4,
    nms_threshold: float = 0.35,
    weights_path: str = "model_weight.pth",
    output_path: Optional[str] = None
) -> Any:
    """
    Unified model inference function allowing prediction on:
    - 'picture' / 'image': Single photo inference
    - 'video': Video file inference with target FPS
    - 'webcam' / 'stream': Live stream generator with target FPS
    
    Args:
        mode: 'picture', 'video', or 'webcam' / 'stream'
        source: Image path/bytes/array, video path, or camera device index (0, 1...)
        fps: Target frames per second for video and webcam stream
        conf_threshold: Minimum confidence score to accept detection
        nms_threshold: IoU threshold for Non-Maximum Suppression
        weights_path: Path to trained Faster R-CNN .pth weights
        output_path: Optional path to save annotated output video
    """
    inferencer = FasterRCNNInferencer(weights_path=weights_path)

    mode_lower = mode.lower()
    if mode_lower in ("picture", "image", "photo"):
        return inferencer.predict_image(
            image_input=source,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold
        )
    elif mode_lower in ("video", "videofile"):
        return inferencer.predict_video(
            video_input=source,
            output_path=output_path,
            target_fps=fps,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold
        )
    elif mode_lower in ("webcam", "stream", "cam"):
        cam_source = int(source) if isinstance(source, (int, str)) and str(source).isdigit() else source
        return inferencer.predict_video_stream(
            video_source=cam_source,
            target_fps=fps or 15.0,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold
        )
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'picture', 'video', or 'webcam'.")
