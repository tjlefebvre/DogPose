# Phase 3 Plan — 3D Body Fit (BITE) + Stylized Rendering

---

## ⚠️ Plan Review — Corrections & Errata (2026-05-31)

Verified against the actual [runa91/bite_release](https://github.com/runa91/bite_release)
repo (its `environment.yaml`, README, demo command) and this machine. The
findings below **supersede** any conflicting statement later in this document.
Ground truth pulled from BITE's env file: `python=3.7.10`, `pytorch==1.6.0`,
`torchvision==0.7.0`, `cudatoolkit=10.1`, `pytorch3d==0.2.5`, `numpy=1.18.5`.
Demo entry point: `scripts/ttoptimization_stage3.py` over a folder of
pre-cropped, roughly-square, dog-centred images (`datasets/test_image_crops`).

1. **🔴 SHOWSTOPPER — BITE's stack is incompatible with this Mac (arm64 / M4).**
   PyTorch 1.6.0 / cudatoolkit 10.1 / PyTorch3D 0.2.5 is a frozen 2020-era
   CUDA/Linux stack with no Apple-Silicon wheels; PyTorch 1.6 predates MPS.
   "Runs on CPU/MPS/CUDA" and "install on CPU, skip CUDA" (below) are WRONG —
   not a toggle, the dependency tree won't resolve natively. Realistic
   runtimes: a Linux+GPU box, a cloud GPU, Docker `linux/amd64` (emulated,
   slow), or Colab. **Chunk A must first decide the runtime, not just "does it
   run."** Same constraint already bit this project (`tensorflow-metal …
   crashes on M4`, per `server/requirements.txt`).

2. **🔴 WRONG CONTRACT — BITE does NOT consume our keypoints/silhouette as
   inputs.** The released demo takes a *cropped image* and predicts keypoints +
   silhouette internally, then runs test-time optimization (with a learned
   ground-contact model). Passing DLC/YOLO outputs *in* would require modifying
   BITE's internals, not a wrapper. **Corrected design:** BITE gets the image
   crop only; our DLC keypoints + YOLO silhouette become an independent
   *agreement check* on BITE's fit, not inputs. The `/fit` request drops the
   `keypoints`/`silhouette` form fields; the response is unchanged.

3. **🟠 BITE has no detector — it needs a centred square crop.** Reuse the YOLO
   bbox/crop the DLC server already computes; do not send the full frame and
   expect BITE to find the dog.

4. **🟠 Missing coordinate mapping.** BITE projects in crop space (~256px
   square). Its 2D output must be unmapped to original-image coords — the same
   unmap `server/model.py` already does for DLC keypoints. The contract's
   `mesh_2d_projection` / `joints_2d` must be documented as *original-image
   coords after unmapping*, and the server must do that unmapping.

5. **🟠 ttopt is slow & iterative, not a forward pass.** The demo *is*
   `ttoptimization_stage3.py` — seconds-to-minutes per image, worse under
   emulation. Treat BITE as a **precompute/batch step with result caching keyed
   on image hash**, not an interactive call. The "interactive 3D fit" UX and
   "seconds" estimate below are too optimistic.

6. **🟠 License is a hard constraint, not an open question.** SMAL/BARC/BITE
   ship under MPI **non-commercial research** licenses, and model/data files
   are **registration-gated** (no `git clone` of weights). If DogPose ever ships
   commercially, this stack is disqualified as-is. Decide intent before Chunk A.

7. **🟡 Smaller corrections:** `conda` is not on the shell PATH (don't assume
   the toolchain); `bite_server` needs its own `CORS(app)`; `smal_data/
   part_seg.pkl` (below) is a guessed filename — verify against the repo; BARC
   is **not** a platform fallback (same 2020 stack) — the real fork is
   *local vs. cloud*, not BITE vs. BARC.

**Sequencing implication:** Chunk E (Rough.js) is fully independent of BITE and
improves the working Phase 2 output today. Strong case to **ship Chunk E
first** for immediate visible progress, then gate the BITE work (A–D) behind a
runtime decision (local-emulated / cloud / Colab) made in a revised Chunk A.

---

## Goal

Close the remaining gap to the user's reference sketches by giving the renderer
**real 3D body understanding** and a **hand-drawn stroke aesthetic**:

1. **3D body fit** — replace heuristic 2D ellipses with primitives projected
   from a posed 3D dog mesh (SMAL via BITE). Solves: depth-correct limb
   ordering, breed-aware proportions, anatomically-grounded ribcage / pelvis /
   scapula / cranium volumes.
2. **Stylized renderer** — replace clean p5 strokes with multi-pass jittered
   strokes (Rough.js or hand-rolled equivalent) so the output reads as a
   gesture sketch instead of a CAD diagram.

We **keep** the Phase 1 + Phase 2 foundation: DLC keypoints, YOLOv8n-seg
silhouette, 10 draggable anchors, all existing primitives. BITE and the
stylized layer are **strictly additive** — if BITE fails on an image, the
Phase 2 renderer is the fallback and the output is still useful.

## Why BITE (locked decision)

[BITE (CVPR 2023, MPI-IS)](https://bite.is.tue.mpg.de/) fits the SMAL
parametric dog model to a single photo and outperforms BARC / WLDO on real-world
photos. Pretrained weights are public but registration-gated and
non-commercial (Erratum 6). BITE is pinned to a 2020 CUDA/Linux stack that does
NOT run natively on Apple Silicon (Erratum 1). It takes a cropped image and
predicts keypoints + silhouette internally, then test-time-optimises the SMAL
fit — it does NOT consume our DLC/YOLO outputs as inputs (Erratum 2). The output
we need — a posed 3D mesh with per-vertex body-part labels and per-bone segment
endpoints — is what BITE returns, in crop space; we unmap it (Erratum 4).

Alternatives considered & rejected:
- **BARC** — BITE's predecessor. Use only if BITE setup blocks. Same data
  contract; slightly worse fits on in-the-wild photos.
- **SMALify direct** — too low-level; we'd be re-implementing BITE.
- **Diffusion-based stylization (ControlNet)** — for the *rendering* layer.
  Rejected because it will invent anatomy that isn't in the photo, defeating
  the tool's purpose.

## Pipeline after Phase 3

```
photo
  │
  ├── YOLOv8n-seg ──────────── silhouette polygon          [Phase 2 — unchanged]
  │
  ├── DLC SuperAnimal-Quad ─── 39 keypoints                [Phase 1 — unchanged]
  │
  └── BITE (NEW) ───────────── 3D SMAL mesh (posed)
                                + per-vertex body-part labels
                                + 3D joint positions
                                + 2D projection (camera)
                                + per-part depth (for z-order)
                       │
                       ▼
            primitive extractor (NEW)
              head sphere     ← head verts → min-enclosing-sphere → project
              ribcage ellipse ← ribcage verts → 2D PCA ellipse
              pelvis sphere   ← pelvis verts → min-enclosing-sphere → project
              scapula plate   ← scapula verts (per side)
              neck cylinder   ← neck bone (head-base → withers)
              leg cylinders   ← bone segments per leg, depth-ordered
                       │
                       ▼
            stylized renderer (NEW)
              every primitive ⇒ multi-pass jittered strokes (Rough.js style)
              respects existing render order
              respects "hide handles" toggle for clean print
```

Phase 1/2 primitives become the **fallback path** when BITE returns no fit or
a low-confidence fit.

---

## Locked Design Decisions

### Environment isolation

BITE has heavy & finicky deps (PyTorch3D / neural-mesh-renderer, SMAL pickle
files, possibly Chumpy). We will **not** add them to the existing DLC server's
conda env — high risk of breaking the working stack.

- **Separate Python service** at `dog-pose/bite_server/` running on port 5001.
- New conda env `dog-pose-bite` with PyTorch + PyTorch3D + BITE deps.
- Flask app, single `/fit` endpoint. Accepts the dog-centred image crop (the
  client sends the YOLO-bbox crop, or the bbox for the server to crop —
  Erratum 3). It does NOT accept keypoints/silhouette as inputs; BITE predicts
  those internally (Erratum 2). DLC/YOLO outputs stay client-side as an
  agreement check on the fit.
- Existing DLC server (`server.py` on port 5000) stays untouched. Client
  orchestrates: POST image to `:5000/predict` → POST image + result to
  `:5001/fit` → merge.

If BITE service is down or returns 5xx, client proceeds with Phase 2 output
only. No fatal coupling.

### What BITE returns to the client

```jsonc
{
  "ok": true,
  "mesh_2d_projection": {
    // Per-body-part 2D projected vertices in original-image coords.
    // Used by the primitive extractor.
    "head":       [[x, y, z_camera], ...],
    "neck":       [...],
    "torso":      [...],
    "ribcage":    [...],
    "pelvis":     [...],
    "scapula_l":  [...],
    "scapula_r":  [...],
    "femur_l":    [...],
    "tibia_l":    [...],
    "humerus_l":  [...],
    // ... etc. SMAL has ~33 body parts; we ship the ones we render.
  },
  "joints_2d": [{"name": "withers", "x": ..., "y": ..., "z": ...}, ...],
  "fit_confidence": 0.0–1.0,   // composite of silhouette IoU + kp reproj error
  "camera": { "f": ..., "cx": ..., "cy": ... }
}
```

The client never sees raw 3D mesh vertices — too heavy. Server flattens to
2D projection + per-part. z is preserved for occlusion ordering only.

### Primitive extraction rules

Each primitive is derived from its corresponding SMAL body-part vertex group:

| Primitive | Source verts | Extraction |
|---|---|---|
| Head sphere | `head` | Welzl min-enclosing-circle on 2D projection |
| Muzzle wedge | `head` lower-front quadrant | Hull → simplified to 4 pts |
| Ribcage ellipse | `ribcage` | 2D PCA → major/minor axes |
| Pelvis sphere | `pelvis` | Welzl min-enclosing-circle |
| Scapula plate | `scapula_l/r` | 2D convex hull, simplified |
| Neck cylinder | `neck` | Bone endpoints + per-vertex thickness sampling |
| Leg segments | bone joints from `joints_2d` | Already line segments |
| Joint balls | joint xy | Same radii as Phase 2 |

z-order per primitive comes from mean z of its source verts. Renderer draws
back-to-front per primitive (not per pixel).

### Stylized rendering

- Use **Rough.js** (~9 KB gzipped, MIT, zero deps). Bundled via CDN `<script>`
  tag in `index.html` — no build step.
- Every primitive renders through a `roughDraw(...)` wrapper that calls the
  equivalent Rough.js primitive (`circle`, `ellipse`, `line`, `polygon`).
- Tunable per-primitive: `roughness` (0–3), `bowing` (0–3), `strokeWidth`,
  `seed` (deterministic per session).
- Construction shapes (ribcage, pelvis, head) get higher roughness; legs and
  spine get lower (cleaner gesture lines). Matches the references.
- "Hide handles" + download paths inherit the stylized renderer automatically
  (no separate render path).

### Out of scope for Phase 3

- Making projected primitives draggable. They follow BITE. Phase 4 question.
- Animating fit over a video. Single-image only.
- Multi-dog scenes. Largest-subject rule, same as today.
- Adapting BITE's training (it's pretrained — we don't retrain).
- Replacing DLC. BITE *uses* DLC keypoints as constraints.
- Texture / shading. Construction lines only.

---

## Implementation Chunks

Each chunk = one atomic commit on `main` (or its own branch + merge if you'd
rather review). Each chunk leaves the app in a working state. We verify in
the browser preview against `/Users/tjlefebvre/Desktop/image_proxy.webp` + at
least one new pose at every chunk boundary. Commit message format follows the
existing convention: `Phase 3.X — <one-line summary>`.

### Chunk A — BITE spike (no integration)

**Goal**: decide *where* BITE will run, then prove it runs end-to-end on one
image — before writing any glue. Per Erratum 1, native arm64 is effectively
ruled out, so this chunk is a **runtime decision** first, a smoke test second.
Highest-risk chunk; do it in isolation so we can change runtime (or abandon
the 3D path) before committing production code.

- **Decide runtime first.** Native arm64 won't resolve BITE's pinned 2020
  CUDA/Linux stack. Evaluate, in order of least effort:
  1. **Google Colab** (free/cheap GPU, their env largely matches) — fastest way
     to a first mesh; good enough to validate quality before investing.
  2. **Cloud Linux+GPU** (e.g. a spot instance) — for the eventual batch service.
  3. **Docker `linux/amd64`** on this Mac (emulated, CPU-only, slow) — last
     resort; ttopt may take minutes/image.
  Record the choice + rationale in `dog-pose/bite_server/SETUP.md`.
- `git checkout -b phase3-spike` (throwaway; do NOT merge).
- Clone BITE outside this repo; register on the project page and download the
  registration-gated model/data files (Erratum 6). Recreate their env with
  `conda env create -f environment.yaml` (env name `dog_pose_env_pytorch3d`).
  Note every friction point (PyTorch3D build, kornia==0.4.0 pin, model
  downloads) in `SETUP.md` as it happens.
- Run the actual demo entry point — `scripts/ttoptimization_stage3.py` with the
  refinement config — over a folder of **pre-cropped, dog-centred square**
  images (BITE has no detector, Erratum 3): make crops from the YOLO bboxes of
  - one repo PNG at the project root,
  - `/Users/tjlefebvre/Desktop/image_proxy.webp`,
  - one reference-style photo (clear side-on dog).
- Save the mesh overlays in `dog-pose/bite_server/spike_outputs/` and **record
  wall-clock per image** (informs the precompute/caching design, Erratum 5).

**Verify**: (a) the chosen runtime actually produced a fit; (b) visual
inspection — does the mesh roughly conform to each dog, limbs roughly placed?;
(c) per-image latency noted. If quality good → continue. If BITE blocks → try
BARC (same stack, single forward pass, faster — but same platform constraint),
or fall back to the "richer 2D + Rough.js" path (ship Chunk E alone).

**Commit**: nothing committed to `main`. Spike branch may be tagged
`phase3-spike-snapshot` and deleted after.

**Time estimate**: 2–4 hr. Risk: high (environment setup on M-series).

### Chunk B — `bite_server` skeleton

**Goal**: stand up a separate Flask service that wraps BITE, returns a
well-formed JSON contract. No client wiring yet.

- `git checkout main && git checkout -b phase3-bite-server`.
- New dir `dog-pose/bite_server/` with:
  - `server.py` — Flask app on port 5001, `POST /fit` endpoint.
  - `bite_wrapper.py` — single function `fit(image_np, keypoints, silhouette)
    -> dict` matching the JSON contract in "What BITE returns" above.
  - `requirements.txt` — BITE-side deps only. Do **not** touch
    `dog-pose/server/requirements.txt`.
  - `SETUP.md` — exact conda + pip incantation reproducing the spike env,
    including any M-series workarounds discovered in Chunk A.
  - `verify.py` — analogue of the existing `dog-pose/server/verify.py`:
    runs `fit()` against the same fixture images and prints the per-part
    vertex counts + fit_confidence + a sanity check on `joints_2d`.
- Body-part vertex groups: pull from SMAL's published part segmentation
  (`smal_data/part_seg.pkl` in the BITE repo) and hard-code the mapping
  from SMAL part-id → our part-name in `bite_wrapper.py`. Document the
  mapping in a comment.
- z-order: include per-part mean z in the projection output.

**Verify**: `python verify.py` produces:
- `ok: true` for the two clean side-on photos,
- non-zero vertex counts for `head`, `ribcage`, `pelvis`, `femur_l`, etc.,
- `fit_confidence` ≥ 0.3 on at least one of them,
- valid JSON (`json.dumps` round-trip).

Also: `curl -F image=@... :5001/fit` returns the same.

**Commit**: `Phase 3.B — bite_server skeleton + /fit endpoint`.

**Time**: 2–3 hr. Risk: medium (JSON contract + part-id mapping).

### Chunk C — Client orchestration: call both servers, merge

**Goal**: client hits `:5000` then `:5001`, stashes the merged result, but
does **not yet draw anything new**. Existing Phase 2 render is unchanged.

- In `dog-pose/sketch/sketch.js`, after `sendToServer` resolves:
  - POST the image + DLC response to `http://localhost:5001/fit`.
  - On success, stash `serverResponse.bite = <bite payload>`.
  - On any error (network, 5xx, malformed JSON), log a warning and continue —
    `bite` stays `null`.
  - Add a tiny status indicator in `index.html` next to the upload button:
    "3D fit: ok / failed / off" so we can see what happened without opening
    devtools.
- Add a constant `BITE_SERVER_URL = "http://localhost:5001/fit"` at the top
  of `sketch.js`. If the env variable / URL is empty, skip the call entirely.

**Verify**: drop a photo, confirm the status indicator turns green and
devtools shows the `bite` payload on `serverResponse`. With `bite_server`
stopped, indicator shows "failed" and the page still renders Phase 2 output.

**Commit**: `Phase 3.C — client calls bite_server, stashes payload`.

**Time**: 1 hr. Risk: low.

### Chunk D — Projected primitives (BITE path only, no styling yet)

**Goal**: when `bite` is present and `fit_confidence` clears a threshold,
use BITE-derived primitives in place of the Phase 2 heuristic ones. Same
clean p5 strokes — no Rough.js yet. This isolates the "is BITE actually
better?" question from "do stylized strokes look right?"

- New module-style block at the bottom of `sketch.js`:
  `computeBitePrimitives(bite) -> { head, ribcage, pelvis, scapulaL,
  scapulaR, neck, legs[], jointBalls[], zOrder[] }`.
- Primitive extraction per the "Primitive extraction rules" table above.
  Use a 2D Welzl implementation (~30 lines) for min-enclosing circles;
  PCA on a vertex array for ellipses (`numeric.js` is overkill — hand-roll
  the 2×2 eigen decomposition).
- In `drawConstruction`, branch:
  - `if (serverResponse.bite && bite.fit_confidence >= BITE_CONFIDENCE_MIN)`
    → render BITE primitives in z-order.
  - Else → render Phase 2 primitives as today.
- Constants at the top of `sketch.js`:
  ```
  BITE_CONFIDENCE_MIN = 0.35   // fall back to Phase 2 below this
  ```
- The 10 draggable handles + silhouette backdrop remain Phase 1/2-driven —
  not from BITE. So the user can still nudge anchors if BITE is off.

**Verify**: against the reference-style side-on dog photo, the BITE-derived
ribcage / pelvis / head should sit visibly more correctly on the body than
the Phase 2 ellipses (no more silhouette-overshoot, correct depth ordering
on legs). Compare side-by-side via the existing download path — save one
with `BITE_CONFIDENCE_MIN = 999` (forces Phase 2) and one with the real
threshold. Put both screenshots in `dog-pose/bite_server/comparison/`.

**Commit**: `Phase 3.D — BITE-projected primitives w/ Phase 2 fallback`.

**Time**: 3–4 hr. Risk: medium (primitive extraction maths + z-order).

### Chunk E — Stylized renderer (Rough.js)

**Goal**: every primitive in `drawConstruction` renders through Rough.js so
output reads as a sketch, not a diagram. Works for **both** BITE and Phase 2
fallback paths — they share the same draw calls.

- Add Rough.js via CDN `<script>` in `index.html`. Pin a specific version
  (e.g. `roughjs@4.6.6`).
- New helper `roughCanvas` initialised once per `setup()`.
- New wrapper functions: `roughCircle(x,y,r,opts)`, `roughEllipse(...)`,
  `roughLine(...)`, `roughPolygon(pts, opts)`. Each delegates to Rough.js
  but applies project-wide defaults (deterministic seed, default stroke
  width, default roughness).
- Replace every `circle()`, `ellipse()`, `line()` etc. call in
  `drawConstruction` with the wrapper. Keep the existing colours and
  ordering.
- Per-primitive style overrides:
  ```
  STYLE_CONSTRUCTION = { roughness: 2.2, bowing: 1.5 }  // ribcage, pelvis, head
  STYLE_GESTURE      = { roughness: 0.8, bowing: 0.3 }  // spine, legs
  STYLE_SILHOUETTE   = { roughness: 1.0, bowing: 0.5 }  // backdrop polygon
  ```
- Seed: hash of image filename + frame count. Sketch stays still while
  dragging (rerolling each frame is dizzying); we re-seed only on new image.

**Verify**: side-by-side vs the user's reference sketches. Strokes should
read as pencil-ish, with the slight "drawn twice" feel visible in the refs.
Hide handles → download → confirm the print output is also stylized.

**Commit**: `Phase 3.E — Rough.js stylized rendering for all primitives`.

**Time**: 1.5 hr. Risk: low.

### Chunk F — Tuning + docs

**Goal**: dial in per-part style, fit-confidence threshold, and write Phase 3
deltas into `PLAN.md`.

- Calibrate `BITE_CONFIDENCE_MIN` against 5+ photos covering: clear side-on,
  3/4 view, lying down, partial occlusion, harsh perspective. We want Phase 2
  to take over **only** when BITE is visibly wrong.
- Calibrate `roughness` / `bowing` per primitive class.
- Append a "Phase 3 deltas" section to `PLAN.md` mirroring the Phase 2
  pattern (architecture, render order, new tuning constants, what changed).
- Update `dog-pose/server/verify.py` to optionally call out to `:5001/fit`
  and print whether BITE is reachable. Don't make it a hard dependency.
- Add a Makefile or short `dog-pose/RUN.md` documenting "start both servers
  + the sketch" so the dev-loop is one command per pane.

**Files**: `PLAN.md`, `dog-pose/sketch/sketch.js`,
`dog-pose/server/verify.py`, `dog-pose/RUN.md` (new).
**Commit**: `Phase 3.F — tuning + docs`.

**Time**: 1.5 hr. Risk: low.

---

## Sequencing & Rollback

```
A (spike, throwaway)
   │
   ▼
B (bite_server) ──▶ C (client orchestration) ──▶ D (BITE primitives) ──▶ E (rough.js) ──▶ F (tune + docs)
                                                       │
                          server only │  client only   │  client only      │  client only
```

- A is a kill-switch: if BITE doesn't work on this machine, we replan
  (BARC or stay 2D) before writing any production code.
- B is independently testable via `bite_server/verify.py` and `curl`.
- C–F are client-only and additive — each one leaves the app fully working
  with a partial Phase 3 stack.
- Rollback at any point: `git revert <commit>` or, if the chunk is the tip,
  `git reset --hard <prev>`.

Branching: each chunk on its own branch `phase3-<letter>-<slug>`, merged via
fast-forward to `main` once verified. Spike branch (Chunk A) is the
exception — never merged.

## Outcome checks against desired sketches

We're measuring against the user's four reference images (howling pose, 3/4
back view, side-on construction, multi-pose page). For each chunk, the
"verify" step calls out the specific quality bar; the cumulative bar at the
end of Phase 3:

| Reference trait | Source | Verified by |
|---|---|---|
| Anatomically-correct ribcage egg position | BITE → ribcage verts → ellipse | Chunk D side-by-side |
| Depth-correct leg ordering (near leg over far leg) | BITE z-ordering | Chunk D — multi-pose photo test |
| Scapula plate visible behind shoulder | BITE → scapula verts | Chunk D |
| Cranium + muzzle as separate volumes, not one circle | BITE head part split | Chunk D |
| Loose, sketchy stroke quality | Rough.js | Chunk E |
| Multiple "drawn twice" lines | Rough.js multi-pass | Chunk E |
| Construction reads as gesture, not CAD | Per-primitive style tuning | Chunk F |
| Still works on photos BITE can't fit (lying / occluded) | Phase 2 fallback | Chunk F multi-photo test |

If after Chunk F any row in that table still fails on its named test image,
that's a Phase 3 bug — fix before declaring done. If a row fails only on
*new* edge cases beyond the test set, that's Phase 4 scope.

## Time estimate

| Chunk | Est | Risk |
|---|---|---|
| A — BITE spike | 2–4 hr | **high** |
| B — bite_server skeleton | 2–3 hr | medium |
| C — client orchestration | 1 hr | low |
| D — BITE primitives | 3–4 hr | medium |
| E — Rough.js rendering | 1.5 hr | low |
| F — tuning + docs | 1.5 hr | low |
| **Total** | **~12–16 hr** | dominated by A + D |

## Open questions (deferred to Phase 4 unless they bite us)

- Should projected primitives be draggable in any way? (Probably no — but
  worth a re-check after seeing real use.)
- Should we add a SMAL **shape** slider so the user can dial breed
  proportions if BITE's shape fit is off? Cheap to add if needed.
- BITE runtime is unresolved on this machine (Erratum 1): native arm64 is
  out; ttopt is slow (Erratum 5). The live question is local-emulated vs.
  cloud vs. Colab + result caching — decided in the revised Chunk A.
- Caching BITE results keyed on image hash, so re-renders don't re-fit.
- Tuning Rough.js per device pixel ratio for crisp Retina output.
