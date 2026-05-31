# DogPose — construction-sketch overlay for dog photos

Upload a photo of a dog and get a **gesture-drawing construction overlay** on top
of it — head sphere, ribcage/pelvis volumes, shoulder/haunch masses, a spine,
multi-segment legs with joint balls, and a silhouette backdrop — so it can be
used as a guide for figure-style sketching (Loomis / Walt Stanchfield style).

The overlay is built from real model output, not hand-placed: a detector finds
the dog, a pose model predicts body keypoints, and a renderer turns those into
anatomical construction primitives. Low-confidence anchors are draggable so the
user can correct the fit.

## Pipeline

```
photo
  ├─ YOLOv8n-seg          → dog bbox + silhouette polygon
  ├─ DLC SuperAnimal-Quad → 39 body keypoints
  └─ renderer (p5.js)     → construction primitives + 10 draggable anchors
```

A Python/Flask server runs detection + pose; a browser sketch (p5.js) does the
rendering and interaction.

## Repo layout

| Path | What |
|---|---|
| `dog-pose/server/` | Flask server — YOLO detection + DLC pose, returns keypoints, bbox, silhouette |
| `dog-pose/sketch/` | Browser overlay — `index.html` + `sketch.js` (primitives, drag UI, downloads) |
| `dog-pose/bite_server/` | Phase 3 experiment scaffolding (see below) — only on the `phase3-spike` branch |
| `PLAN.md` | Phase 1 plan + design (anchors, fallbacks, geometry) |
| `PLAN-phase2.md` | Phase 2 plan — anatomical volumes + silhouette (shipped) |
| `PLAN-phase3.md` | Phase 3 plan — 3D body fit (BITE) + sketchy rendering (planned) |

Model weights (DLC snapshot, MegaDetector, YOLO `.pt`) are **not versioned** —
they're large binaries pulled separately (see `.gitignore` /
`dog-pose/server/requirements.txt`).

## Running it

```bash
cd dog-pose/server
pip install -r requirements.txt        # plus the DLC SuperAnimal weights
python server.py                       # Flask on :5000
# then open dog-pose/sketch/index.html in a browser and upload a dog photo
```

Note: the server runs CPU-only by design — `tensorflow-metal` is excluded
because it crashes on Apple-Silicon (M-series) Macs.

## Status / roadmap

- **Phase 1 (done)** — keypoint backbone, 10 draggable anchors, construction
  primitives, server robustness fixes. See `PLAN.md`.
- **Phase 2 (done)** — silhouette layer, shoulder/haunch masses, 4-segment
  legs, neck/ears/jaw. See `PLAN-phase2.md`.
- **Phase 3 (planned/spike)** — fit a 3D parametric dog model (BITE/SMAL) for
  anatomically-correct, depth-ordered volumes, plus hand-drawn stroke
  rendering. Plan + a Colab validation spike live on the **`phase3-spike`**
  branch (`PLAN-phase3.md`, `dog-pose/bite_server/`).
