import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// API は FastAPI(localhost:8100)を別途起動しておく
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5180,
    strictPort: true,
    proxy: {
      "/api": "http://localhost:8100",
    },
  },
});
