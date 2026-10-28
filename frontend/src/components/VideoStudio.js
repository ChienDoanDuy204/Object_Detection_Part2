"use client";

import React, { useState, useRef } from "react";
import { 
  Film, 
  Upload, 
  Play, 
  Pause, 
  CheckCircle, 
  Gauge, 
  Download, 
  Sliders, 
  Zap, 
  RefreshCw,
  Clock,
  Layers
} from "lucide-react";
import { getApiBaseUrl } from "@/utils/api";

export default function VideoStudio({ systemInfo }) {
  const [videoFile, setVideoFile] = useState(null);
  const [videoSrc, setVideoSrc] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [targetFps, setTargetFps] = useState(10.0);
  const [confThreshold, setConfThreshold] = useState(0.35);
  const [resultData, setResultData] = useState(null);
  const [activeVideoView, setActiveVideoView] = useState("annotated"); // "annotated" or "original"

  const fileInputRef = useRef(null);
  const videoPlayerRef = useRef(null);
  const apiBase = getApiBaseUrl();

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setVideoFile(file);
    setResultData(null);
    setVideoSrc(URL.createObjectURL(file));
  };

  const handleLoadSampleVideo = async () => {
    try {
      setIsProcessing(true);
      const resp = await fetch(`${apiBase}/api/video/sample`);
      if (!resp.ok) throw new Error("Could not fetch sample video");
      const blob = await resp.blob();
      const file = new File([blob], "sample_demo.mp4", { type: "video/mp4" });
      setVideoFile(file);
      setResultData(null);
      setVideoSrc(URL.createObjectURL(file));
    } catch (err) {
      console.error("Failed to load sample video:", err);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleProcessVideo = async () => {
    if (!videoFile) return;

    setIsProcessing(true);
    setResultData(null);

    const formData = new FormData();
    formData.append("file", videoFile);

    try {
      const res = await fetch(
        `${apiBase}/api/predict/video?target_fps=${targetFps}&conf_threshold=${confThreshold}&nms_threshold=0.35`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || "Video processing failed. Make sure the backend server is running on port 8000.");
      }

      const data = await res.json();
      setResultData(data);
      setActiveVideoView("annotated");
    } catch (err) {
      console.error("Video processing error:", err);
      alert(`Error processing video: ${err.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const getClassColor = (className) => {
    const idx = (systemInfo?.classes || []).indexOf(className);
    if (idx !== -1 && systemInfo?.colors?.[idx]) {
      return systemInfo.colors[idx];
    }
    return "#38bdf8";
  };

  return (
    <div className="studio-grid" id="video-studio-container">
      {/* Left Stage Display */}
      <div className="glass-card display-stage">
        <div className="stage-toolbar">
          <div className="toolbar-group">
            <button
              id="upload-video-btn"
              className="btn btn-primary"
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload size={15} />
              <span>Upload Video</span>
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              style={{ display: "none" }}
              onChange={handleFileUpload}
            />

            <button
              id="load-sample-video-btn"
              className="btn btn-secondary"
              onClick={handleLoadSampleVideo}
            >
              <Film size={15} />
              <span>Load Demo Video</span>
            </button>

            {resultData && (
              <div className="nav-tabs" style={{ padding: "0.2rem" }}>
                <button
                  id="video-view-annotated-btn"
                  className={`nav-tab-btn ${activeVideoView === "annotated" ? "active" : ""}`}
                  style={{ padding: "0.4rem 0.8rem", fontSize: "0.76rem" }}
                  onClick={() => setActiveVideoView("annotated")}
                >
                  <span>Annotated Video</span>
                </button>
                <button
                  id="video-view-original-btn"
                  className={`nav-tab-btn ${activeVideoView === "original" ? "active" : ""}`}
                  style={{ padding: "0.4rem 0.8rem", fontSize: "0.76rem" }}
                  onClick={() => setActiveVideoView("original")}
                >
                  <span>Original Video</span>
                </button>
              </div>
            )}
          </div>

          {resultData && (
            <div className="toolbar-group">
              <a
                id="download-video-btn"
                href={resultData.video_url.startsWith("http") ? resultData.video_url : `${apiBase}${resultData.video_url}`}
                download="faster_rcnn_processed.webm"
                className="btn btn-secondary btn-sm"
              >
                <Download size={14} />
                <span>Export WebM</span>
              </a>
            </div>
          )}
        </div>

        {/* Video Player Display Container */}
        <div className="stage-canvas-container" id="video-canvas-container">
          {!videoSrc ? (
            <div
              className="upload-dropzone"
              id="video-dropzone"
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="dropzone-icon">
                <Film size={28} />
              </div>
              <h3 className="dropzone-title">Upload a Video or Click "Load Demo Video"</h3>
              <p className="dropzone-desc">MP4, WebM, MOV with configurable target processing FPS</p>
            </div>
          ) : (
            <div style={{ position: "relative", width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <video
                key={activeVideoView === "annotated" && resultData ? resultData.video_url : videoSrc}
                ref={videoPlayerRef}
                controls
                playsInline
                autoPlay
                loop
                style={{
                  maxWidth: "100%",
                  maxHeight: "640px",
                  borderRadius: "var(--radius-sm)",
                  display: "block",
                }}
              >
                <source
                  src={
                    activeVideoView === "annotated" && resultData
                      ? (resultData.video_url.startsWith("http") ? resultData.video_url : `${apiBase}${resultData.video_url}`)
                      : videoSrc
                  }
                  type="video/webm"
                />
                <source
                  src={
                    activeVideoView === "annotated" && resultData
                      ? (resultData.video_url.startsWith("http") ? resultData.video_url : `${apiBase}${resultData.video_url}`)
                      : videoSrc
                  }
                  type="video/mp4"
                />
                Your browser does not support HTML5 video tag.
              </video>

              {isProcessing && (
                <div
                  style={{
                    position: "absolute",
                    inset: 0,
                    background: "rgba(3, 7, 18, 0.85)",
                    backdropFilter: "blur(4px)",
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: "1rem",
                    borderRadius: "var(--radius-sm)",
                    zIndex: 10,
                  }}
                >
                  <div className="spinner"></div>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ fontSize: "1rem", fontWeight: 700, color: "var(--accent-cyan)" }}>
                      Processing Video at {targetFps} FPS...
                    </div>
                    <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
                      Faster R-CNN ResNet-18 is decoding and annotating sampled frames
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Video Class Distribution Breakdown */}
        {resultData && resultData.class_summary && (
          <div style={{ marginTop: "0.5rem" }}>
            <div style={{ fontSize: "0.78rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", marginBottom: "0.5rem" }}>
              Object Detections Summary Across Video
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
              {Object.entries(resultData.class_summary).map(([cname, count]) => (
                <div
                  key={cname}
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "0.4rem",
                    padding: "0.3rem 0.65rem",
                    background: "rgba(15, 23, 42, 0.8)",
                    border: "1px solid var(--border-subtle)",
                    borderRadius: "var(--radius-full)",
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
                  <strong style={{ color: "var(--text-primary)" }}>{cname}</strong>
                  <span style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                    {count} instances
                  </span>
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
          <span>Video FPS & Parameters</span>
        </div>

        {/* TARGET FPS CONTROL SLIDER */}
        <div className="control-item">
          <div className="control-header">
            <span style={{ fontWeight: 600 }}>Target Processing FPS</span>
            <span className="control-val" id="target-fps-val" style={{ fontSize: "1.1rem" }}>
              {targetFps} FPS
            </span>
          </div>
          <input
            id="video-fps-slider"
            type="range"
            min="1"
            max="30"
            step="1"
            value={targetFps}
            onChange={(e) => setTargetFps(parseFloat(e.target.value))}
            disabled={isProcessing}
          />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.7rem", color: "var(--text-muted)" }}>
            <span>1 FPS (Skip heavy)</span>
            <span>15 FPS (Standard)</span>
            <span>30 FPS (Full)</span>
          </div>

          {/* Quick FPS Presets */}
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
                disabled={isProcessing}
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
            💡 <strong>FPS Parameter:</strong> Controls how many frame images per second the Faster R-CNN model processes. Higher FPS produces smoother detections; lower FPS accelerates processing throughput.
          </div>
        </div>

        {/* Confidence Threshold */}
        <div className="control-item">
          <div className="control-header">
            <span>Confidence Threshold</span>
            <span className="control-val">{Math.round(confThreshold * 100)}%</span>
          </div>
          <input
            id="video-conf-slider"
            type="range"
            min="0.10"
            max="0.90"
            step="0.05"
            value={confThreshold}
            onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
            disabled={isProcessing}
          />
        </div>

        {/* Action Button */}
        <button
          id="run-video-inference-btn"
          className="btn btn-primary"
          style={{ width: "100%", padding: "0.75rem", fontSize: "0.9rem" }}
          onClick={handleProcessVideo}
          disabled={!videoFile || isProcessing}
        >
          {isProcessing ? (
            <>
              <div className="spinner" style={{ width: "16px", height: "16px", borderWidth: "2px" }}></div>
              <span>Processing Video...</span>
            </>
          ) : (
            <>
              <Zap size={16} />
              <span>Process Video at {targetFps} FPS</span>
            </>
          )}
        </button>

        {/* Result Metrics */}
        {resultData && (
          <div style={{ display: "flex", flexDirection: "column", gap: "0.65rem", marginTop: "0.5rem" }}>
            <span style={{ fontSize: "0.78rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase" }}>
              Execution Metrics
            </span>
            <div className="metrics-row">
              <div className="metric-card">
                <span className="metric-label">Target FPS</span>
                <span className="metric-value cyan">{resultData.target_fps}</span>
              </div>
              <div className="metric-card">
                <span className="metric-label">Actual Speed</span>
                <span className="metric-value green">{resultData.avg_processing_fps} FPS</span>
              </div>
              <div className="metric-card">
                <span className="metric-label">Processed</span>
                <span className="metric-value purple">
                  {resultData.processed_frames}/{resultData.total_source_frames}
                </span>
              </div>
              <div className="metric-card">
                <span className="metric-label">Total Time</span>
                <span className="metric-value amber">{resultData.total_elapsed_s}s</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
