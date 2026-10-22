import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        timeout: 1800000,
        proxyTimeout: 1800000
      },
      "/health": "http://localhost:8000",
      "/metrics": "http://localhost:8000"
    }
  }
});
