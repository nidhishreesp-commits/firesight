import { createContext, ReactNode, useContext, useEffect, useRef, useState } from "react";
import { api } from "./api";

type VideoResult = any;

interface VideoSessionValue {
  file: File | null;
  videoUrl: string | null;
  result: VideoResult | null;
  error: string;
  busy: boolean;
  elapsed: number;
  maxSeconds: number;
  sampleFps: number;
  lastTime: number;
  clipDuration: number;
  setMaxSeconds: (n: number) => void;
  setSampleFps: (n: number) => void;
  setLastTime: (n: number) => void;
  selectFile: (file: File | null) => void;
  run: () => Promise<void>;
  clear: () => void;
}

const VideoSessionContext = createContext<VideoSessionValue | null>(null);
const STORAGE_KEY = "firesight.videoSession.v1";

export function VideoSessionProvider({ children }: { children: ReactNode }) {
  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [result, setResult] = useState<VideoResult | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [maxSeconds, setMaxSeconds] = useState(120);
  const [sampleFps, setSampleFps] = useState(5);
  const [lastTime, setLastTime] = useState(0);
  const [clipDuration, setClipDuration] = useState(0);
  const urlRef = useRef<string | null>(null);

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      if (raw) {
        const saved = JSON.parse(raw);
        if (saved.result) setResult(saved.result);
        if (saved.maxSeconds) setMaxSeconds(saved.maxSeconds);
        if (saved.sampleFps) setSampleFps(saved.sampleFps);
        if (saved.lastTime) setLastTime(saved.lastTime);
        if (saved.fileName) setError(`Re-upload "${saved.fileName}" to replay the video; the saved stats/timeline are below.`);
      }
    } catch {
      /* ignore */
    }
    return () => {
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    };
  }, []);

  useEffect(() => {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify({
        result, maxSeconds, sampleFps, fileName: file?.name ?? null
      }));
    } catch {
      /* quota */
    }
  }, [result, maxSeconds, sampleFps, file]);

  const selectFile = (next: File | null) => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    const url = next ? URL.createObjectURL(next) : null;
    urlRef.current = url;
    setVideoUrl(url);
    setFile(next);
    setResult(null);
    setError("");
    setLastTime(0);
    setClipDuration(0);
    if (url) {
      const probe = document.createElement("video");
      probe.preload = "metadata";
      probe.src = url;
      probe.onloadedmetadata = () => {
        const dur = Math.ceil(probe.duration || 0);
        setClipDuration(dur);
        setMaxSeconds(Math.min(Math.max(dur, 5), 300));
      };
    }
  };

  const run = async () => {
    if (!file) return;
    setBusy(true);
    setError("");
    setResult(null);
    setElapsed(0);
    const started = Date.now();
    const timer = window.setInterval(() => setElapsed(Math.round((Date.now() - started) / 1000)), 500);
    try {
      setResult(await api.detectVideo(file, maxSeconds, sampleFps));
    } catch (e) {
      setError(`${file.name}: ${String(e)}. If this is a long/4K clip, reduce "analyse first" seconds or sample fps.`);
    } finally {
      window.clearInterval(timer);
      setBusy(false);
    }
  };

  const clear = () => {
    selectFile(null);
    setResult(null);
    setError("");
    try { sessionStorage.removeItem(STORAGE_KEY); } catch { /* ignore */ }
  };

  return (
    <VideoSessionContext.Provider value={{
      file, videoUrl, result, error, busy, elapsed, maxSeconds, sampleFps, lastTime, clipDuration,
      setMaxSeconds, setSampleFps, setLastTime, selectFile, run, clear
    }}>
      {children}
    </VideoSessionContext.Provider>
  );
}

export function useVideoSession() {
  const ctx = useContext(VideoSessionContext);
  if (!ctx) throw new Error("useVideoSession must be used inside VideoSessionProvider");
  return ctx;
}
