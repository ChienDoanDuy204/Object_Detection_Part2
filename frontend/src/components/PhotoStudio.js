"use client";

import React, { useState, useEffect, useRef } from "react";
import { 
  Upload, 
  Sparkles, 
  Sliders, 
  Download, 
  Eye, 
  Maximize2, 
  Crosshair, 
  Layers, 
  CheckCircle2, 
  AlertCircle 
} from "lucide-react";
import { getApiBaseUrl } from "@/utils/api";

export default function PhotoStudio({ systemInfo, snapshotData }) {
  const [imageSrc, setImageSrc] = useState(null);
  const [imageDims, setImageDims] = useState({ width: 0, height: 0 });
  const [detections, setDetections] = useState([]);
  const [annotatedImgSrc, setAnnotatedImgSrc] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [confThreshold, setConfThreshold] = useState(0.35);
  const [activeClassFilter, setActiveClassFilter] = useState({});
  const [selectedBox, setSelectedBox] = useState(null);
  const [hoveredBox, setHoveredBox] = useState(null);
  const [viewMode, setViewMode] = useState("interactive"); // "interactive", "annotated", "original"
  const [perfStats, setPerfStats] = useState(null);
  const [samples, setSamples] = useState([]);

  const imgRef = useRef(null);
  const fileInputRef = useRef(null);
  const apiBase = getApiBaseUrl();

  // Fetch preloaded samples from backend
  useEffect(() => {
    fetch(`${apiBase}/api/samples`)
      .then((res) => res.json())
      .then((data) => {
        if (data && data.samples) {
          setSamples(data.samples);
        }
      })
      .catch((err) => console.error("Failed to load samples:", err));
  }, []);

  // Handle incoming snapshot from Camera mode
  useEffect(() => {
    if (snapshotData) {
      setImageSrc(snapshotData.image);
      runInferenceOnBase64(snapshotData.image);
    }
  }, [snapshotData]);

  // Load a pre-selected VOC sample image
  const handleSelectSample = async (sampleId) => {
    setIsLoading(true);
    setSelectedBox(null);
    try {
      const imgUrl = `${apiBase}/api/sample/${sampleId}`;
      const resp = await fetch(imgUrl);
      const blob = await resp.blob();
      const file = new File([blob], sampleId, { type: "image/jpeg" });
      
      const reader = new FileReader();
      reader.onload = (e) => {
        setImageSrc(e.target.result);
      };
      reader.readAsDataURL(file);

      await runInferenceOnFile(file);
    } catch (err) {
      console.error("Error loading sample:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setSelectedBox(null);
    const reader = new FileReader();
    reader.onload = (ev) => {
      setImageSrc(ev.target.result);
    };
    reader.readAsDataURL(file);

    await runInferenceOnFile(file);
  };

  const runInferenceOnFile = async (file) => {
    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);

      // Low threshold to backend so frontend slider can dynamically filter interactively
      const res = await fetch(`${apiBase}/api/predict/image?conf_threshold=0.15&nms_threshold=0.35`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Inference request failed");
      const data = await res.json();

      setDetections(data.detections || []);
      setAnnotatedImgSrc(data.annotated_image);
      setImageDims({ width: data.width, height: data.height });
      setPerfStats({
        latency: data.inference_time_ms,
        count: data.total_detections,
        width: data.width,
        height: data.height,
      });

      // Default all detected classes to visible
      const classesMap = {};
      (data.detections || []).forEach((d) => {
        classesMap[d.class_name] = true;
      });
      setActiveClassFilter(classesMap);
    } catch (err) {
      console.error("Inference error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const runInferenceOnBase64 = async (b64Data) => {
    setIsLoading(true);
    try {
      const byteString = atob(b64Data.split(",")[1]);
      const ab = new ArrayBuffer(byteString.length);
      const ia = new Uint8Array(ab);
      for (let i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
      }
      const blob = new Blob([ab], { type: "image/jpeg" });
      const file = new File([blob], "snapshot.jpg", { type: "image/jpeg" });
      await runInferenceOnFile(file);
    } catch (err) {
      console.error("Base64 inference error:", err);
      setIsLoading(false);
    }
  };

  // Filter detections based on confidence threshold and class toggle
  const visibleDetections = detections.filter(
    (d) => d.score >= confThreshold && (activeClassFilter[d.class_name] !== false)
  );

  // Group counts for visible detections
  const classCounts = {};
  visibleDetections.forEach((d) => {
    classCounts[d.class_name] = (classCounts[d.class_name] || 0) + 1;
  });

  const toggleClassVisibility = (className) => {
    setActiveClassFilter((prev) => ({
      ...prev,
      [className]: !prev[className],
    }));
  };

  const getClassColor = (className) => {
    const idx = (systemInfo?.classes || []).indexOf(className);
    if (idx !== -1 && systemInfo?.colors?.[idx]) {
      return systemInfo.colors[idx];
    }
    return "#38bdf8";
  };

  return (
    <div className="studio-grid" id="photo-studio-container">
      {/* Left Stage Display */}
      <div className="glass-card display-stage">
        <div className="stage-toolbar">
          <div className="toolbar-group">
            <button
              id="upload-photo-btn"
              className="btn btn-primary"
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload size={15} />
              <span>Upload Photo</span>
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              style={{ display: "none" }}
              onChange={handleFileUpload}
            />

            {imageSrc && (
              <div className="nav-tabs" style={{ padding: "0.2rem" }}>
                <button
                  id="view-interactive-btn"
                  className={`nav-tab-btn ${viewMode === "interactive" ? "active" : ""}`}
                  style={{ padding: "0.4rem 0.8rem", fontSize: "0.76rem" }}
                  onClick={() => setViewMode("interactive")}
                >
                  <Crosshair size={14} />
                  <span>Interactive</span>
                </button>
                <button
                  id="view-annotated-btn"
                  className={`nav-tab-btn ${viewMode === "annotated" ? "active" : ""}`}
                  style={{ padding: "0.4rem 0.8rem", fontSize: "0.76rem" }}
                  onClick={() => setViewMode("annotated")}
                >
                  <Eye size={14} />
                  <span>Model Overlay</span>
                </button>
                <button
                  id="view-original-btn"
                  className={`nav-tab-btn ${viewMode === "original" ? "active" : ""}`}
                  style={{ padding: "0.4rem 0.8rem", fontSize: "0.76rem" }}
                  onClick={() => setViewMode("original")}
                >
                  <span>Original</span>
                </button>
              </div>
            )}
          </div>

          {perfStats && (
            <div className="toolbar-group">
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                Inference: <strong style={{ color: "var(--accent-cyan)" }}>{perfStats.latency}ms</strong> (~{Math.round(1000 / perfStats.latency)} FPS)
              </span>
              {annotatedImgSrc && (
                <a
                  id="download-annotated-btn"
                  href={annotatedImgSrc}
                  download="faster_rcnn_detection.jpg"
                  className="btn btn-secondary btn-sm"
                  title="Download Annotated Photo"
                >
                  <Download size={14} />
                  <span>Export</span>
                </a>
              )}
            </div>
          )}
        </div>

        {/* Interactive Image Canvas Area */}
        <div className="stage-canvas-container" id="stage-canvas-container">
          {!imageSrc ? (
            <div
              className="upload-dropzone"
              id="photo-dropzone"
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="dropzone-icon">
                <Upload size={28} />
              </div>
              <h3 className="dropzone-title">Upload a Photo or Choose a Sample Below</h3>
              <p className="dropzone-desc">Drag & drop PNG, JPG, or WebP up to 20MB</p>
            </div>
          ) : (
            <div style={{ position: "relative", display: "inline-block", maxWidth: "100%", maxHeight: "640px" }}>
              <img
                ref={imgRef}
                src={viewMode === "annotated" && annotatedImgSrc ? annotatedImgSrc : imageSrc}
                alt="Source Detection"
                style={{
                  display: "block",
                  maxWidth: "100%",
                  maxHeight: "640px",
                  objectFit: "contain",
                  borderRadius: "var(--radius-sm)",
                }}
              />

              {/* Direct Manipulation SVG Overlay (Interactive Bounding Boxes) */}
              {viewMode === "interactive" && imageDims.width > 0 && (
                <svg
                  className="svg-overlay"
                  viewBox={`0 0 ${imageDims.width} ${imageDims.height}`}
                  preserveAspectRatio="xMidYMid meet"
                >
                  {visibleDetections.map((det, i) => {
                    const [x1, y1, x2, y2] = det.box;
                    const w = x2 - x1;
                    const h = y2 - y1;
                    const color = getClassColor(det.class_name);
                    const isSelected = selectedBox === det;
                    const isHovered = hoveredBox === det;

                    return (
                      <g key={i} className="svg-box-group">
                        {/* Outer Bounding Box */}
                        <rect
                          x={x1}
                          y={y1}
                          width={w}
                          height={h}
                          fill={isSelected ? `${color}33` : isHovered ? `${color}22` : "transparent"}
                          stroke={color}
                          strokeWidth={isSelected ? 3.5 : isHovered ? 3 : 2}
                          className={`svg-box ${isSelected ? "selected" : ""}`}
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedBox(det);
                          }}
                          onMouseEnter={() => setHoveredBox(det)}
                          onMouseLeave={() => setHoveredBox(null)}
                        />

                        {/* Interactive Pill Tag */}
                        <g transform={`translate(${x1}, ${Math.max(18, y1)})`}>
                          <rect
                            x="0"
                            y="-18"
                            width={Math.max(70, det.class_name.length * 9 + 40)}
                            height="18"
                            fill={color}
                            rx="3"
                          />
                          <text
                            x="6"
                            y="-5"
                            fill="#ffffff"
                            fontSize="11"
                            fontWeight="600"
                            fontFamily="var(--font-sans)"
                          >
                            {det.class_name} {Math.round(det.score * 100)}%
                          </text>
                        </g>
                      </g>
                    );
                  })}
                </svg>
              )}

              {isLoading && (
                <div
                  style={{
                    position: "absolute",
                    inset: 0,
                    background: "rgba(3, 7, 18, 0.7)",
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: "0.75rem",
                    borderRadius: "var(--radius-sm)",
                  }}
                >
                  <div className="spinner"></div>
                  <span style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--accent-cyan)" }}>
                    Faster R-CNN Inferring...
                  </span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Quick Sample Gallery Bar */}
        <div>
          <div style={{ fontSize: "0.78rem", fontWeight: 600, color: "var(--text-muted)", marginBottom: "0.4rem" }}>
            QUICK DEMO SAMPLES (CLICK TO RUN)
          </div>
          <div className="samples-bar" id="sample-gallery-bar">
            {samples.map((s) => (
              <div
                key={s.id}
                className="sample-card"
                id={`sample-card-${s.id.replace('.jpg', '')}`}
                onClick={() => handleSelectSample(s.id)}
              >
                <img
                  src={`/api/sample/${s.id}`}
                  alt={s.title}
                  className="sample-thumb"
                  loading="lazy"
                />
                <div className="sample-title">{s.title}</div>
                <div className="sample-category">{s.category}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right Control & Inspector Panel */}
      <div className="glass-card panel-section">
        <div className="section-title">
          <Sliders size={16} />
          <span>Interactive Controls</span>
        </div>

        {/* Confidence Threshold Slider */}
        <div className="control-item">
          <div className="control-header">
            <span>Confidence Threshold</span>
            <span className="control-val">{Math.round(confThreshold * 100)}%</span>
          </div>
          <input
            id="photo-conf-slider"
            type="range"
            min="0.10"
            max="0.95"
            step="0.01"
            value={confThreshold}
            onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
          />
          <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
            Live filters active detections without re-computation
          </span>
        </div>

        {/* Performance Metrics */}
        {perfStats && (
          <div className="metrics-row">
            <div className="metric-card">
              <span className="metric-label">Latency</span>
              <span className="metric-value cyan">{perfStats.latency}ms</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Objects</span>
              <span className="metric-value green">{visibleDetections.length}</span>
            </div>
          </div>
        )}

        {/* Detected Classes Filter Pills */}
        {Object.keys(classCounts).length > 0 && (
          <div className="control-item">
            <span style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--text-secondary)" }}>
              Detected Classes ({Object.keys(classCounts).length})
            </span>
            <div className="class-pills-wrap">
              {Object.entries(classCounts).map(([cname, count]) => {
                const color = getClassColor(cname);
                const isActive = activeClassFilter[cname] !== false;
                return (
                  <button
                    key={cname}
                    className={`class-pill ${isActive ? "active" : ""}`}
                    onClick={() => toggleClassVisibility(cname)}
                    title={`Click to ${isActive ? "hide" : "show"} ${cname}`}
                  >
                    <span className="class-pill-dot" style={{ backgroundColor: color }}></span>
                    <span>{cname}</span>
                    <span style={{ opacity: 0.6 }}>({count})</span>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Direct Manipulation Box Inspector */}
        {selectedBox ? (
          <div className="inspector-panel" id="detection-inspector">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: "0.78rem", fontWeight: 700, color: "var(--accent-cyan)", textTransform: "uppercase" }}>
                Target Inspector
              </span>
              <button
                className="btn btn-secondary btn-sm"
                style={{ padding: "0.15rem 0.4rem", fontSize: "0.7rem" }}
                onClick={() => setSelectedBox(null)}
              >
                Close
              </button>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
              <span
                style={{
                  width: "12px",
                  height: "12px",
                  borderRadius: "50%",
                  backgroundColor: getClassColor(selectedBox.class_name),
                }}
              ></span>
              <strong style={{ fontSize: "1.1rem" }}>{selectedBox.class_name}</strong>
              <span style={{ marginLeft: "auto", fontFamily: "var(--font-mono)", color: "var(--accent-emerald)" }}>
                {(selectedBox.score * 100).toFixed(1)}%
              </span>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.4rem", fontSize: "0.75rem", fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
              <div>x1: {selectedBox.box[0]}</div>
              <div>y1: {selectedBox.box[1]}</div>
              <div>x2: {selectedBox.box[2]}</div>
              <div>y2: {selectedBox.box[3]}</div>
              <div>W: {Math.round(selectedBox.box[2] - selectedBox.box[0])}px</div>
              <div>H: {Math.round(selectedBox.box[3] - selectedBox.box[1])}px</div>
            </div>
          </div>
        ) : (
          <div style={{ padding: "1rem", textAlign: "center", border: "1px dashed var(--border-subtle)", borderRadius: "var(--radius-sm)", fontSize: "0.75rem", color: "var(--text-muted)" }}>
            Click on any bounding box in the photo to inspect its coordinates and dimensions.
          </div>
        )}
      </div>
    </div>
  );
}
