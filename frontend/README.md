# Chest Pathology XAI frontend

A single-screen React workspace for uploading a chest X-ray, running the
FastAPI model, and inspecting per-class predictions and HiResCAM heatmaps.

## Setup

```bash
cp .env.example .env
npm install
npm run dev
```

The app expects the backend at `VITE_API_URL`, which defaults to
`http://localhost:8000`.

```bash
# From the repository root:
python -m backend.main --host 127.0.0.1 --port 8000
```

## Scripts

```bash
npm run dev        # Vite dev server
npm run lint       # ESLint
npm run typecheck  # TypeScript project build check
npm run build      # typecheck + production bundle
npm run preview    # preview the production bundle
```

## Behavior

- Uploads accept PNG and JPEG up to 5 MB on the client; the backend enforces
  its own 10 MB limit.
- All 14 predictions are shown. Findings at or above their tuned threshold are
  highlighted; below-threshold findings remain selectable so their heatmap can
  be inspected.
- The backend returns the selected `target`, and the UI keeps the highlighted
  row in sync with the displayed heatmap.
- Failed or malformed API responses produce a visible error banner instead of a
  blank screen.
- In-flight requests are aborted when a new file or target is selected.

## Limitations

This is a research prototype, not a medical device. Do not use it with patient
data or for clinical decisions.
