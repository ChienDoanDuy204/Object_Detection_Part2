"use client";

import React, { useState, useEffect, useRef } from "react";
import { 
  Camera, 
  VideoOff, 
  Sliders, 
  FlipHorizontal, 
  Camera as CameraIcon, 
  Activity, 
  Zap, 
  Maximize,
  AlertCircle 
} from "lucide-react";

export default function CameraStudio({ systemInfo, onSnapshot }) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [targetFps, setTargetFps] = useState(15.0);
  const [confThreshold, setConfThreshold] = useState(0.35);
  const [actualFps, setActualFps] = useState(0);
  const [latencyMs, setLatencyMs] = useState(0);
  const [isMirrored, setIsMirrored] = useState(true);
  const [cameraError, setCameraError] = useState(null);
  const [liveDetections, setLiveDetections] = useState([]);
  const [availableDevices, setAvailableDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState("");

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const animFrameIdRef = useRef(null);
  const lastSendTimeRef = useRef(0);
  const isSendingFrameRef = useRef(false);

  // Enumerate video input devices
  useEffect(() => {
    async function getDevices() {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevs = devices.filter((d) => d.kind === "videoinput");
        setAvailableDevices(videoDevs);
        if (videoDevs.length > 0 && !selectedDeviceId) {
          setSelectedDeviceId(videoDevs[0].deviceId);
        }
      } catch (err) {
        console.error("Device enumeration error:", err);
      }
    }
    if (navigator.mediaDevices) {
      getDevices();
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  const startCamera = async () => {
    setCameraError(null);
    try {
      const constraints = {
        video: {
          deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined,
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }

      // Initialize WebSocket connection to FastAPI backend
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${protocol}//${window.location.hostname}:8000/api/ws/webcam`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsStreaming(true);
        startStreamingLoop();
      };

      ws.onmessage = (event) => {
        try {
          const res = JSON.parse(event.data);
          if (res.detections) {
            setLiveDetections(res.detections);
            setLatencyMs(res.inference_time_ms || 0);
            setActualFps(res.actual_fps || 0);
            drawOverlay(res.detections);
          }
        } catch (e) {
          console.error("WS Parse error:", e);
        } finally {
          isSendingFrameRef.current = false;
        }
      };

      ws.onerror = (e) => {
        console.error("WebSocket error:", e);
        setCameraError("WebSocket connection to backend failed. Make sure backend is running on port 8000.");
      };

      ws.onclose = () => {
        setIsStreaming(false);
      };
    } catch (err) {
      console.error("Camera access error:", err);
      setCameraError(`Unable to access camera: ${err.message}. Please check browser camera permissions.`);
      setIsStreaming(false);
    }
  };

  const stopCamera = () => {
    if (animFrameIdRef.current) {
      cancelAnimationFrame(animFrameIdRef.current);
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    setIsStreaming(false);
    setLiveDetections([]);
    setActualFps(0);
    setLatencyMs(0);
  };

  // Streaming loop respecting target FPS parameter
  const startStreamingLoop = () => {
    const captureCanvas = document.createElement("canvas");
    const captureCtx = captureCanvas.getContext("2d");

    const loop = (timestamp) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        return;
      }

      const minIntervalMs = 1000 / Math.max(1, targetFps);
      const elapsed = timestamp - lastSendTimeRef.current;

      if (elapsed >= minIntervalMs && !isSendingFrameRef.current && videoRef.current) {
        const video = videoRef.current;
        if (video.videoWidth > 0 && video.videoHeight > 0) {
          captureCanvas.width = video.videoWidth;
          captureCanvas.height = video.videoHeight;

          // Draw video frame to hidden canvas
          captureCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
          const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.75);

          const payload = {
            image: dataUrl,
            target_fps: targetFps,
            conf_threshold: confThreshold,
            nms_threshold: 0.35,
            return_annotated: false,
          };

          isSendingFrameRef.current = true;
          lastSendTimeRef.current = timestamp;
          wsRef.current.send(JSON.stringify(payload));
        }
      }

      animFrameIdRef.current = requestAnimationFrame(loop);
    };

    animFrameIdRef.current = requestAnimationFrame(loop);
  };

  const getClassColor = (className) => {
    const idx = (systemInfo?.classes || []).indexOf(className);
    if (idx !== -1 && systemInfo?.colors?.[idx]) {
      return systemInfo.colors[idx];
    }
    return "#38bdf8";
  };

  // Draw smooth bounding boxes on canvas overlay
  const drawOverlay = (detections) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return;

    canvas.width = vw;
    canvas.height = vh;
    ctx.clearRect(0, 0, vw, vh);

    detections.forEach((det) => {
      let [x1, y1, x2, y2] = det.box;
      const w = x2 - x1;
      const h = y2 - y1;
      const color = getClassColor(det.class_name);

      // If video is mirrored, flip x coordinates for overlay
      if (isMirrored) {
        const origX1 = x1;
        x1 = vw - x2;
        x2 = vw - origX1;
      }

      // Draw bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x1, y1, w, h);

      // Draw label background pill
      const labelText = `${det.class_name} ${Math.round(det.score * 100)}%`;
      ctx.font = "bold 13px Inter, sans-serif";
      const textMetrics = ctx.measureText(labelText);
      const tagW = textMetrics.width + 12;
      const tagH = 20;
      const tagY = Math.max(0, y1 - tagH);

      ctx.fillStyle = color;
      ctx.fillRect(x1, tagY, tagW, tagH);

      ctx.fillStyle = "#ffffff";
      ctx.fillText(labelText, x1 + 6, tagY + 14);
    });
  };

  // Take Snapshot to directly manipulate in Photo Studio mode
  const handleTakeSnapshot = () => {
    const video = videoRef.current;
    if (!video) return;

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const ctx = tempCanvas.getContext("2d");

    if (isMirrored) {
      ctx.translate(video.videoWidth, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
    const dataUrl = tempCanvas.toDataURL("image/jpeg", 0.95);

    onSnapshot({
      image: dataUrl,
      detections: liveDetections,
    });
  };

  // Tally counts of live detections
  const liveCounts = {};
  liveDetections.forEach((d) => {
    liveCounts[d.class_name] = (liveCounts[d.class_name] || 0) + 1;
  });

  return (
    <div className="studio-grid" id="camera-studio-container">
      {/* Left Stage Display */}
      <div className="glass-card display-stage">
        <div className="stage-toolbar">
          <div className="toolbar-group">
            {!isStreaming ? (
              <button
                id="start-camera-btn"
                className="btn btn-primary"
                onClick={startCamera}
              >
                <Camera size={16} />
                <span>Start Camera Stream</span>
              </button>
            ) : (
              <button
                id="stop-camera-btn"
                className="btn btn-danger"
                onClick={stopCamera}
              >
                <VideoOff size={16} />
                <span>Stop Stream</span>
              </button>
            )}

            {isStreaming && (
              <>
                <button
                  id="snapshot-camera-btn"
                  className="btn btn-secondary"
                  onClick={handleTakeSnapshot}
                  title="Capture snapshot and manipulate in Photo Studio"
                >
                  <CameraIcon size={15} />
                  <span>Snapshot to Photo Studio</span>
                </button>

                <button
                  id="mirror-camera-btn"
                  className={`btn btn-secondary ${isMirrored ? "active" : ""}`}
                  onClick={() => setIsMirrored(!isMirrored)}
                  title="Toggle Mirror Mode"
                >
                  <FlipHorizontal size={15} />
                </button>
              </>
            )}
          </div>

          {/* Live Performance HUD */}
          {isStreaming && (
            <div className="toolbar-group">
              <span className="status-badge" style={{ background: "rgba(56, 189, 248, 0.15)", borderColor: "var(--accent-cyan)", color: "var(--accent-cyan)" }}>
                <span>Actual: <strong>{actualFps} FPS</strong></span>
              </span>
              <span className="status-badge" style={{ background: "rgba(16, 185, 129, 0.15)", borderColor: "var(--accent-emerald)", color: "var(--accent-emerald)" }}>
                <span>Model Latency: <strong>{latencyMs}ms</strong></span>
              </span>
            </div>
          )}
        </div>

        {cameraError && (
          <div style={{ padding: "0.75rem 1rem", background: "rgba(239, 68, 68, 0.12)", border: "1px solid rgba(239, 68, 68, 0.3)", borderRadius: "var(--radius-sm)", color: "#fca5a5", fontSize: "0.82rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <AlertCircle size={16} />
            <span>{cameraError}</span>
          </div>
        )}

        {/* Live Camera Stream Display */}
        <div className="stage-canvas-container" id="camera-canvas-container">
          {!isStreaming ? (
            <div className="upload-dropzone" onClick={startCamera}>
              <div className="dropzone-icon">
                <Camera size={30} />
              </div>
              <h3 className="dropzone-title">Click "Start Camera Stream" to Begin</h3>
              <p className="dropzone-desc">Real-time object detection stream with adjustable FPS control</p>
            </div>
          ) : (
            <div style={{ position: "relative", width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <video
                ref={videoRef}
                playsInline
                muted
                style={{
                  maxWidth: "100%",
                  maxHeight: "640px",
                  borderRadius: "var(--radius-sm)",
                  transform: isMirrored ? "scaleX(-1)" : "none",
                  display: "block",
                }}
              />
              <canvas
                ref={canvasRef}
                style={{
                  position: "absolute",
                  top: "50%",
                  left: "50%",
                  transform: "translate(-50%, -50%)",
                  maxWidth: "100%",
                  maxHeight: "640px",
                  pointerEvents: "none",
                }}
              />
            </div>
          )}
        </div>

        {/* Real-time Detections Tally */}
        {isStreaming && Object.keys(liveCounts).length > 0 && (
          <div>
            <div style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", marginBottom: "0.35rem" }}>
              Live Detected Objects in Frame ({liveDetections.length})
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem" }}>
              {Object.entries(liveCounts).map(([cname, count]) => (
                <div
                  key={cname}
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "0.4rem",
                    padding: "0.25rem 0.65rem",
                    borderRadius: "var(--radius-full)",
                    background: "rgba(15, 23, 42, 0.8)",
                    border: `1px solid ${getClassColor(cname)}`,
                    fontSize: "0.75rem",
                  }}
                >
                  <span
                    style={{
                      width: "8px",
                      height: "8px",
                      borderRadius: "50%",
                      backgroundColor: getClassColor(cname),
                    }}
                  ></span>
                  <span>{cname}</span>
                  <span style={{ fontWeight: 700, color: "var(--accent-cyan)" }}>x{count}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Right Control Panel */}
      <div className="glass-card panel-section">
        <div className="section-title">
          <Sliders size={16} />
          <span>Stream & FPS Controls</span>
        </div>

        {/* TARGET FPS CONTROL SLIDER */}
        <div className="control-item">
          <div className="control-header">
            <span style={{ fontWeight: 600 }}>Target Processing FPS</span>
            <span className="control-val" id="webcam-target-fps-val" style={{ fontSize: "1.1rem" }}>
              {targetFps} FPS
            </span>
          </div>
          <input
            id="camera-fps-slider"
            type="range"
            min="1"
            max="30"
            step="1"
            value={targetFps}
            onChange={(e) => setTargetFps(parseFloat(e.target.value))}
          />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.7rem", color: "var(--text-muted)" }}>
            <span>1 FPS (Low load)</span>
            <span>15 FPS (Smooth)</span>
            <span>30 FPS (Full)</span>
          </div>

          {/* Quick Presets */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "0.35rem", marginTop: "0.4rem" }}>
            {[5, 10, 15, 30].map((preset) => (
              <button
                key={preset}
                className={`btn btn-secondary btn-sm ${targetFps === preset ? "active" : ""}`}
                style={{
                  padding: "0.25rem 0.4rem",
                  fontSize: "0.72rem",
                  borderColor: targetFps === preset ? "var(--accent-cyan)" : undefined,
                  color: targetFps === preset ? "var(--accent-cyan)" : undefined,
                }}
                onClick={() => setTargetFps(preset)}
              >
                {preset} FPS
              </button>
            ))}
          </div>

          <div
            style={{
              padding: "0.6rem 0.75rem",
              background: "rgba(56, 189, 248, 0.08)",
              border: "1px solid rgba(56, 189, 248, 0.2)",
              borderRadius: "var(--radius-sm)",
              fontSize: "0.72rem",
              color: "var(--text-secondary)",
              lineHeight: 1.4,
              marginTop: "0.5rem",
            }}
          >
            ⚡ <strong>Webcam FPS Throttler:</strong> Dynamically regulates client-to-server frame streaming intervals. Server processes exactly <code>{targetFps}</code> frames/second, reducing GPU power consumption and network bandwidth.
          </div>
        </div>

        {/* Confidence Threshold */}
        <div className="control-item">
          <div className="control-header">
            <span>Confidence Threshold</span>
            <span className="control-val">{Math.round(confThreshold * 100)}%</span>
          </div>
          <input
            id="camera-conf-slider"
            type="range"
            min="0.10"
            max="0.90"
            step="0.05"
            value={confThreshold}
            onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
          />
        </div>

        {/* Camera Device Selector */}
        {availableDevices.length > 1 && (
          <div className="control-item">
            <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>Camera Device</span>
            <select
              style={{
                background: "var(--bg-input)",
                border: "1px solid var(--border-subtle)",
                borderRadius: "var(--radius-sm)",
                color: "var(--text-primary)",
                padding: "0.5rem",
                fontSize: "0.8rem",
              }}
              value={selectedDeviceId}
              onChange={(e) => {
                setSelectedDeviceId(e.target.value);
                if (isStreaming) {
                  stopCamera();
                  setTimeout(() => startCamera(), 300);
                }
              }}
            >
              {availableDevices.map((dev, idx) => (
                <option key={dev.deviceId} value={dev.deviceId}>
                  {dev.label || `Camera ${idx + 1}`}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Live Gauges */}
        <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", marginTop: "0.5rem" }}>
          <span style={{ fontSize: "0.78rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase" }}>
            Live Stream Gauges
          </span>

          <div className="fps-gauge-wrap">
            <div>
              <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase" }}>Processed FPS</div>
              <div className="fps-badge-large">{actualFps.toFixed(1)}</div>
            </div>
            <div style={{ textAlign: "right" }}>
              <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase" }}>Target</div>
              <div style={{ fontSize: "1.2rem", fontWeight: 700, fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
                {targetFps} FPS
              </div>
            </div>
          </div>

          <div className="metrics-row">
            <div className="metric-card">
              <span className="metric-label">Model Latency</span>
              <span className="metric-value green">{latencyMs}ms</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Active Boxes</span>
              <span className="metric-value cyan">{liveDetections.length}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
