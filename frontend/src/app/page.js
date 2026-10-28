"use client";

import React, { useState, useEffect } from "react";
import Navbar from "@/components/Navbar";
import PhotoStudio from "@/components/PhotoStudio";
import VideoStudio from "@/components/VideoStudio";
import CameraStudio from "@/components/CameraStudio";
import { getApiBaseUrl } from "@/utils/api";

export default function Home() {
  const [activeTab, setActiveTab] = useState("photo"); // "photo", "video", "camera"
  const [systemInfo, setSystemInfo] = useState(null);
  const [snapshotData, setSnapshotData] = useState(null);

  // Fetch model info from FastAPI backend
  useEffect(() => {
    const apiBase = getApiBaseUrl();
    fetch(`${apiBase}/api/info`)
      .then((res) => res.json())
      .then((data) => {
        if (data && data.status === "ready") {
          setSystemInfo(data);
        }
      })
      .catch((err) => {
        console.warn("Could not fetch API info:", err);
      });
  }, []);

  // When snapshot is taken in camera mode, switch to photo tab to inspect
  const handleSnapshot = (snap) => {
    setSnapshotData(snap);
    setActiveTab("photo");
  };

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        systemInfo={systemInfo}
      />

      <main className="main-wrapper" id="main-content">
        {activeTab === "photo" && (
          <PhotoStudio
            systemInfo={systemInfo}
            snapshotData={snapshotData}
          />
        )}

        {activeTab === "video" && (
          <VideoStudio
            systemInfo={systemInfo}
          />
        )}

        {activeTab === "camera" && (
          <CameraStudio
            systemInfo={systemInfo}
            onSnapshot={handleSnapshot}
          />
        )}
      </main>

      <footer
        style={{
          marginTop: "auto",
          padding: "1.25rem 2rem",
          borderTop: "1px solid var(--border-subtle)",
          textAlign: "center",
          fontSize: "0.75rem",
          color: "var(--text-muted)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "0.5rem",
        }}
      >
        <div>
          Faster R-CNN Object Detection System • ResNet-18 Backbone • 20 Pascal VOC Classes
        </div>
        <div>
          Backend: FastAPI • Frontend: Next.js • Target FPS Processing Engine
        </div>
      </footer>
    </div>
  );
}
