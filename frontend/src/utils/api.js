/**
 * Dynamic API endpoint resolution.
 * Automatically targets the FastAPI backend on port 8000 using the current host,
 * preventing Next.js reverse-proxy body size truncation on large video uploads.
 */

export function getApiBaseUrl() {
  if (typeof window !== "undefined") {
    const hostname = window.location.hostname || "localhost";
    return `http://${hostname}:8000`;
  }
  return "http://localhost:8000";
}

export function getWsBaseUrl() {
  if (typeof window !== "undefined") {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const hostname = window.location.hostname || "localhost";
    return `${protocol}//${hostname}:8000`;
  }
  return "ws://localhost:8000";
}
