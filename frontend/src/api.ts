const BASE = import.meta.env.VITE_API_BASE ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<any>("/health"),
  testsets: () => request<any>("/api/testsets"),
  testImages: (name: string, filter: string, offset: number, limit = 24, threshold = 0.5) =>
    request<any>(`/api/testsets/${name}/images?filter=${filter}&offset=${offset}&limit=${limit}&threshold=${threshold}`),
  benchmark: () => request<any>("/api/benchmark/latest"),
  benchRResults: () => request<any>("/api/benchmark/benchR"),
  models: () => request<any>("/api/models"),
  cameras: () => request<any>("/api/live/cameras"),
  startCamera: (id: string) => request<any>(`/api/live/${id}/start`, { method: "POST" }),
  stopCamera: (id: string) => request<any>(`/api/live/${id}/stop`, { method: "POST" }),
  feedback: (payload: Record<string, unknown>) =>
    request<any>("/api/feedback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }),
  reviewQueue: () => request<any>("/api/review/queue"),
  bench100: () => request<any>("/api/bench100"),
  benchReview: (payload: Record<string, unknown>) =>
    request<any>("/api/bench100/review", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }),
  benchLock: (reviewer: string, allowOverride = false) =>
    request<any>(`/api/bench100/lock?reviewer=${encodeURIComponent(reviewer)}&allow_composition_override=${allowOverride}`, { method: "POST" }),
  benchR: () => request<any>("/api/benchR"),
  benchRReview: (payload: Record<string, unknown>) =>
    request<any>("/api/benchR/review", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }),
  benchRProposals: () => request<any>("/api/benchR/proposals"),
  benchRSpotcheck: () => request<any>("/api/benchR/spotcheck"),
  benchRSpotSave: (payload: Record<string, unknown>) =>
    request<any>("/api/benchR/spotcheck", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }),
  benchRLock: (reviewer: string) =>
    request<any>(`/api/benchR/lock?reviewer=${encodeURIComponent(reviewer)}`, { method: "POST" }),
  firePolicy: () => request<any>("/api/policy/fire-types"),
  setFirePolicy: (include: string[], exclude: string[]) =>
    request<any>("/api/policy/fire-types", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ include, exclude }) }),
  detectVideo: async (file: File, maxSeconds = 60, sampleFps = 5) => {
    const form = new FormData();
    form.append("file", file);
    form.append("max_seconds", String(maxSeconds));
    form.append("sample_fps", String(sampleFps));
    return request<any>("/api/detect/video", { method: "POST", body: form });
  },
  detectImage: async (file: File, threshold = 0.3) => {
    const form = new FormData();
    form.append("file", file);
    form.append("threshold", String(threshold));
    return request<any>("/api/detect/image", { method: "POST", body: form });
  }
};
