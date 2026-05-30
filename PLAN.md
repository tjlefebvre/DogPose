# Dog Pose Sketch Overlay — Plan

## Goal

A live overlay on uploaded dog photos that draws a volumetric armature in the Loomis / Preston-Blair style (two body circles, head sphere with cross-axis, muzzle box, ear triangles, stick legs with knee knots, tail curve) so the user can use it as a construction guide for sketching the dog. Target: working end-to-end in ~1 day.

The current pipeline (YOLOv8 → DLC SuperAnimal Quadruped → 39 keypoints, rendered as colored asterisks in p5) is unreliable: on small-in-frame dogs the keypoints collapse onto the head (see `Bounding box efforts resolution too large.png`). The plan replaces the unreliable raw-keypoint overlay with a robust construction-primitive overlay backed by a thin drag-correction UI, plus narrow server-side fixes to the worst model-failure mode.

## Locked Design

### Anchor set (α) — 10 anchors

L+R eyes, nose, withers (`neck_end`, id 16), hip (`back_end`, id 20), tail_base (22), tail_end (23), 4 paws (26 FL, 29 FR, 30 BL, 35 BR).

Knees inferred as midpoint(slot, paw). Ears decorative, computed from the head circle. Body widths use a single chest-depth formula, not the model's `body_middle_*` or `belly_bottom` keypoints.

### Fallback chain — what happens when an anchor is missing or low-confidence

- **Head circle** needs ≥1 eye + nose. If only one eye, draw with the visible eye + a default eye-distance; the user can drag the missing eye into place.
- **Front body circle** needs withers. If withers missing, fall back to `bbox_top + (bbox_height / 3)`.
- **Back body circle** needs hip. If missing, fall back to a fixed ratio along the spine `withers → tail_base`.
- **Legs**: each leg independent. If paw missing for a leg, draw the thigh stub from the anatomical slot only.
- **Hard floor**: if eyes + nose + both spine anchors all fail (model-collapse case), do NOT draw construction. Show the YOLO bbox + 10 ghost handles in default template positions; user drags them into place and primitives appear.

### Interaction model (i + i.1)

- 10 draggable handles always visible after inference.
- Primitives recompute live during drag (client-side; no server round-trip).
- **Confidence-gated visual**: confidence ≥ 0.6 → small, faded handle. Confidence < 0.6 or value was inferred via fallback → large, bright, ringed handle (visually draws the eye to the dubious ones).
- **Double-click a handle** → snap back to detected position (per-anchor reset, no global reset).
- **Hide-handles toggle** → clean construction-only view, used by the print and composite download paths.
- **No auto-mirror**: dragging one eye does not move the other. Re-evaluate after seeing real usage.

### Geometry formulas

Let `S = dist(withers, hip)`. Spine direction `spine_dir = (hip - withers).normalize()`. Belly direction `belly_dir = perp(spine_dir)` chosen so it points toward `mean(paws)` (computed once per image from whichever paws are present).

- **Head**:
  - center = midpoint(left_eye, right_eye)
  - radius = `1.5 × dist(eyes)`
  - cross-axis = line through eyes + perpendicular through nose
  - muzzle = tapered shape along head-center → nose, length = `dist(head_edge, nose)`, width ≈ `0.4 × head_radius`
  - ears = two triangles on the upper arc of the head circle (decorative, not from ear keypoints)
- **Body circles** (5a-i + 5b-i):
  - radius `R = 0.30 × S` for both circles
  - front center = `withers + belly_dir × R`
  - back center = `hip + belly_dir × R`
- **Legs** (5c-i):
  - Anatomical slot angles relative to spine-forward direction on each circle:
    - Front-left: 225° on ribcage   • Front-right: 315° on ribcage
    - Back-left: 225° on pelvis     • Back-right: 315° on pelvis
  - Slot position = circle_center + radius × (cos θ, sin θ) in the spine/belly frame
  - Leg = polyline (slot → knee → paw); knee = midpoint(slot, paw)
  - Small knee circle, radius ≈ `0.15 × R`
- **Tail**: single line `tail_base → tail_end`.

### Model fixes (P1, ~90 min, server-side)

- **A1 — Crop-size normalization**: after the YOLO crop, letterbox-resize the crop to 400×400 (square, padded) before feeding DLC. Map keypoints back: crop coords → letterbox coords → original-image coords. Root cause of the park-image collapse.
- **A2 — Confidence threshold split**: raise the construction-drawing threshold from 0.05 → 0.30. Keep 0.05 only as the floor for "where to initially place a low-confidence handle."
- **A3 — Failure detection**: compute keypoint spread (e.g. bounding box of the high-confidence keypoints) and compare against the YOLO bbox diagonal. If spread < 20% of the bbox diagonal → return `{ "failed": true, "bbox": ... }` instead of `keypoints`. Client shows the hard-floor ghost-handle template.

## Implementation Plan — Four Chunks

### Chunk 1 — Server-side robustness (≈ 90 min)

Edit `dog-pose/server/model.py` to add A1, A2, A3:

- Refactor `_detect_crop` to return both the crop and its bounding box in small-frame coords.
- Add a `_letterbox_to_400(crop)` helper that returns `(padded_400x400, scale, pad_x, pad_y)`.
- Run DLC on the 400×400, then unpad / unscale keypoints back to crop coords, then through the existing offset/scale chain to original-image coords.
- Compute the high-confidence-keypoint bounding box; if its diagonal is < 0.2 × YOLO bbox diagonal, set a `failed` flag.
- Update `server.py` response shape to `{ keypoints: [...], bbox: [x1, y1, x2, y2], failed: bool, image_size: [w, h] }`.

Smoke-test against both PNGs at the repo root: the park-dog case should now produce keypoints spread across the dog or trip the `failed` flag.

### Chunk 2 — Construction renderer (≈ 2.5 hr)

In `dog-pose/sketch/sketch.js`, replace the existing asterisk-overlay code with a construction renderer driven by 10 anchors:

- **Anchor extraction** — pull the 10 anchors by id from the keypoints array, applying the fallback chain. Tag each anchor with `{ x, y, confidence, source: 'detected' | 'inferred' | 'ghost' }`.
- **Frame computation** — once per image: spine direction, belly direction (from mean paw position), forward direction (from nose).
- **Head primitive** — circle + cross-axis + muzzle + ear triangles.
- **Body primitives** — front + back circles using the belly-offset formula.
- **Legs** — compute the four anatomical slot positions, draw slot→knee→paw polylines with small knee circles.
- **Tail** — single line.
- Render at the same scale/offset the existing image-drawing code uses, so primitives line up with the photo.

Throw away the existing per-keypoint asterisk rendering and the back-polyline `[20, 21, 22, 23]` — they're superseded.

### Chunk 3 — Drag-correct UI + print/composite refresh (≈ 2.5 hr)

- Render the 10 anchors as draggable handles. Hit-testing on `mousePressed`; selected anchor follows `mouseDragged`; release on `mouseReleased`. Recompute primitives every frame while dragging.
- Confidence-gated handle styling (faded small / bright large + ringed).
- Double-click detection (track last-click time per anchor) → reset that anchor to its detected position.
- "Hide handles" toggle button next to the existing download buttons in `index.html`.
- Hard-floor branch: when server returns `failed: true`, place the 10 ghost anchors at fixed positions inside the returned bbox and tag them `ghost`.
- Update `downloadOverlay` and `downloadComposite` to render with handles hidden, with the new construction primitives instead of the asterisk overlay. Redesign the legend to describe the construction (head circle / ribcage / pelvis / spine / leg axes) rather than 39 named keypoints.

### Chunk 4 — Test + tune (≈ 45 min)

Run a handful of real dog photos through the pipeline and tune:

- **K = 0.30** (body radius coefficient) — may need to be 0.25 or 0.33 depending on how the circles read against typical photos.
- **Head radius coefficient** (currently 1.5 × eye-distance) — same.
- **Slot angles** (225° / 315°) — adjust if leg attachments look off.
- **Failure threshold** (20% of bbox diagonal) — calibrate against a few cases.

Confirm the hide-handles → download path produces a clean print sheet.

## Out of Scope (deliberately)

- Switching the pose model. P3 was rejected: high probability of losing a day to darwin/M-series dependency issues for marginal gain.
- Multi-scale TTA (A4). May add later if residual failures after A1+A3 are common enough to warrant.
- Auto-mirror for symmetric anchors. Add only if the user repeatedly hand-mirrors the same anchors.
- Perspective scaling of the rear body circle in foreshortened views. The two-overlapping-2D-circles approach self-resolves the topology (overlap → looks like one circle) which is good enough for v1.
- Breed-adaptive proportions (greyhound vs bulldog body width). User can drag to adjust.

## Tuning Parameters — Single Source of Truth

Once the code lands, the following should live as named constants at the top of `sketch.js` so tuning is one place:

```
BODY_RADIUS_K        = 0.30   // R = K × dist(withers, hip)
HEAD_RADIUS_K        = 1.5    // R = K × dist(eyes)
KNEE_CIRCLE_K        = 0.15   // knee radius = K × body radius
SLOT_FRONT_LEFT_DEG  = 225
SLOT_FRONT_RIGHT_DEG = 315
SLOT_BACK_LEFT_DEG   = 225
SLOT_BACK_RIGHT_DEG  = 315
CONFIDENCE_DRAW      = 0.30   // below this → handle only, no primitive contribution
CONFIDENCE_GHOST     = 0.05   // below this → handle starts at default template position
CONFIDENCE_CONFIDENT = 0.60   // above this → small faded handle, otherwise large bright
```

And server-side in `model.py`:

```
DLC_INPUT_SIZE       = 400    // letterbox target before DLC
FAILURE_SPREAD_RATIO = 0.20   // failed if kp-spread < ratio × bbox diagonal
```

---

## Phase 2 deltas (shipped)

See [PLAN-phase2.md](PLAN-phase2.md) for the full plan and chunk breakdown. The
sections above reflect Phase 1 only. Below is what actually changed in Phase 2.

### Anchor set delta

No new draggable anchors. Phase 2 adds *derived* points (ear bases, ear ends,
mouth ends, four thigh joints, four knee joints) used by the renderer but
sourced directly from the model — not part of the 10-anchor drag surface.

### New primitives

- **Silhouette backdrop** — YOLOv8n-seg polygon shipped from the server,
  rendered as a faint closed polyline behind every other primitive.
- **Haunch / shoulder mass ovals** — rotated ellipses anchored between
  `hip ↔ back_thai` and `withers ↔ front_thai` (per side), drawn behind the
  body circles so the circles read as torso volumes overlaid on the
  shoulder/haunch masses.
- **4-segment legs** — replaces the slot→thigh→knee→paw 3-segment leg with
  `thigh → knee/elbow → hock/wrist → paw`. Hock/wrist has no DLC keypoint —
  it's an interpolation along knee→paw at `HOCK_WRIST_T`.
- **Neck cylinder** — two side lines tapering between the front body-circle
  edge and the head sphere along the head→withers axis.
- **Real ears + jaw line** — earbase/earend keypoints drive real ear
  triangles; mouth-end keypoints drive the jaw line (auto-skipped on
  breeds where the corners aren't visible to the model).
- **Extended head cross-axes** — Loomis-style centre planes spanning the
  full head sphere (nose↔back-of-skull, eye line across).

### Geometry / behaviour deltas

- Belly direction now comes from the silhouette centroid when present,
  falling back to mean-paw.
- Body radius `R` is clamped to `silhouette_half_width × 1.15` when the
  keypoint-derived `R` overshoots by > 30%.
- Hip falls back to `tail_base`-derived position when `back_end` is detected
  far from `tail_base` (model often misplaces back_end).
- Head centre / radius derives from ear-midpoint ↔ nose when both earbases
  are detected (more stable than the eye-based estimate).
- Leg "slot" concept eliminated — the `thai` keypoint *is* the body
  attachment.

### Render order (back to front)

```
silhouette → spine → haunch/shoulder ovals → ribcage + pelvis circles →
neck cylinder → 4-segment legs → tail → head sphere + axes + muzzle +
ears + jaw → draggable handles
```

### Phase 2 tuning constants (`sketch.js`)

```
HAUNCH_AXIS_RATIO       = 0.65
SHOULDER_AXIS_RATIO     = 0.45
SHOULDER_LENGTH_SCALE   = 0.55   // shrinks shoulder span vs full withers↔thai
MASS_LENGTH_OVERSHOOT   = 0.20
JOINT_R_HIP_SHOULDER    = 0.18
JOINT_R_KNEE_ELBOW      = 0.13
JOINT_R_HOCK_WRIST      = 0.09
HOCK_WRIST_T            = 0.55
DETAIL_CONFIDENCE       = 0.25   // floor for joints / ears / jaw / mass ovals
SILHOUETTE_OPACITY      = 75
```

### Phase 2 server constants (`model.py`)

```
SILHOUETTE_EPSILON_RATIO = 0.01   // Douglas-Peucker epsilon, × bbox diagonal
SILHOUETTE_MAX_POINTS    = 64     // hard cap on polygon vertex count
```
