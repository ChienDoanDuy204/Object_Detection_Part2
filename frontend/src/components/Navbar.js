"use client";

import React from "react";
import { Camera, Film, Image as ImageIcon, Cpu, Sparkles } from "lucide-react";

export default function Navbar({ activeTab, setActiveTab, systemInfo }) {
  return (
    <header className="header-container" id="app-header">
      <div className="brand-wrapper">
        <div className="brand-icon">
          <Sparkles size={20} />
        </div>
        <div>
          <h1 className="brand-title">Faster R-CNN Studio</h1>
          <div className="brand-subtitle">End-to-End Object Detection with FPS Control</div>
        </div>
      </div>

      <nav className="nav-tabs" aria-label="Detection Modes">
        <button
          id="tab-photo-btn"
          className={`nav-tab-btn ${activeTab === "photo" ? "active" : ""}`}
          onClick={() => setActiveTab("photo")}
        >
          <ImageIcon size={16} />
          <span>Photo Studio</span>
        </button>

        <button
          id="tab-video-btn"
          className={`nav-tab-btn ${activeTab === "video" ? "active" : ""}`}
          onClick={() => setActiveTab("video")}
        >
          <Film size={16} />
          <span>Video Pipeline</span>
        </button>

        <button
          id="tab-camera-btn"
          className={`nav-tab-btn ${activeTab === "camera" ? "active" : ""}`}
          onClick={() => setActiveTab("camera")}
        >
          <Camera size={16} />
          <span>Live Camera</span>
        </button>
      </nav>

      <div className="header-status">
        <div className="status-badge" title="Model runtime backend status">
          <span className="status-dot"></span>
          <span>{systemInfo?.device ? `CUDA (${systemInfo.device.toUpperCase()})` : "READY"}</span>
        </div>
      </div>
    </header>
  );
}
