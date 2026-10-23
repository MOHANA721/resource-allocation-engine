import { defineConfig } from "@playwright/test";

const python = process.platform === "win32" ? "..\\.venv\\Scripts\\python.exe" : "../.venv/bin/python";

export default defineConfig({
  testDir: "e2e",
  timeout: 60_000,
  use: { baseURL: "http://localhost:5173", browserName: "chromium" },
  webServer: [
    {
      command: `${python} -m uvicorn app.main:app --app-dir ../backend --host 127.0.0.1 --port 8000`,
      url: "http://127.0.0.1:8000/api/health",
      reuseExistingServer: true,
    },
    {
      command: "npm run dev -- --port 5173 --strictPort",
      url: "http://localhost:5173",
      reuseExistingServer: true,
    },
  ],
});
