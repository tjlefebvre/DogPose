# Phase 2 Plan — Anatomical Volumes + Silhouette

## Goal

Close the gap between the current overlay and the user's reference sketches
(loose Loomis / Walt Stanchfield construction style). The references show three
things our renderer doesn't: **distinct rear haunch & front shoulder masses**,
**4-segment legs with joint balls at every bend**, and a **silhouette curve**
(underline / belly / topline) that keypoints alone cannot give us.

We are **not** redirecting. Phase 1 (DLC keypoint backbone + 10 draggable
anchors + construction primitives) stays as the foundation; Phase 2 layers on
top of it.

## Locked Design Decisions

### New volume primitives

- **Haunch oval** per rear leg — anchored between `hip` and the detected
  `back_*_thai` (id 31 / 32). Major axis = hip → thigh. Minor axis ≈ 0.55 ×
  major. Renders behind the rear-leg polyline.
- **Shoulder oval** per front leg — anchored between `withers` and the detected
  `front_*_thai` (id 24 / 27). Same construction as haunch, but smaller (front
  shoulders read as less massive in references). Optional on a per-pose basis;
  always drawn if both anchors present.

### 4-segment legs

| leg | seg 1 | joint a | seg 2 | joint b | seg 3 | joint c | seg 4 |
|---|---|---|---|---|---|---|---|
| **front** | shoulder | elbow | upper arm → forearm | wrist | forearm | — | paw |
| **rear**  | hip      | stifle (knee) | femur → tibia | hock     | metatarsus | — | paw |

Keypoint mapping (DLC SuperAnimal Quadruped):
- Front-L: `thai=24` (shoulder), `knee=25` (elbow), `paw=26` — wrist
  interpolated at 80% along knee→paw.
- Front-R: 27 / 28 / 29, wrist interpolated.
- Rear-L:  `thai=31` (hip), `knee=33` (stifle), `paw=30` — hock interpolated.
- Rear-R:  32 / 34 / 35, hock interpolated.

Joint balls grow with anatomical importance: shoulder/hip largest, stifle/elbow
medium, wrist/hock smallest. Joint ball radius is a fraction of the body
circle radius `R`, capped by the segment length on either side so the ball
never visually swamps the leg.

### Silhouette layer

- **Model**: switch YOLO backbone from `yolov8n.pt` to `yolov8n-seg.pt`
  (~7 MB extra, COCO-pretrained, same `ultralytics` package). Stays in CPU mode.
- **Pick**: largest seg mask whose class is in `ANIMAL_CLASSES`. Same selection
  rule as the bbox today.
- **Output to client**: a downsampled polygon (Douglas-Peucker simplified to
  ≤ 64 points) in original-image coordinates, alongside `keypoints` / `bbox`.
  No raw mask bitmap shipped — only the polygon.
- **Client uses for** (in priority order):
  1. **Underline curve overlay**: render the polygon at very low opacity as a
     gestural backdrop so the sketcher sees the silhouette.
  2. **Body-circle radius sanity check**: each circle should fit inside the
     silhouette. If `R_keypoint > R_silhouette × 1.3`, log a warning; clamp
     optional.
  3. **Auto-belly direction**: replace the current "mean paw position" heuristic
     with the polygon's centroid below the spine.
- **Fallback**: if silhouette extraction fails (no mask, or polygon
  degenerate), client behaves exactly as Phase 1 — silhouette layer is
  strictly additive.

### Render order (back to front)

1. Silhouette polygon (faint backdrop)
2. Spine line
3. Haunch ovals + shoulder ovals
4. Ribcage + pelvis circles
5. Neck cylinder
6. 4-segment leg polylines + joint balls
7. Tail line
8. Head sphere + extended cross-axes + ears + muzzle
9. Handles (draggable anchors)

### Out of scope for Phase 2

- Making haunch / shoulder / 4-leg joints draggable. They follow keypoint
  inference + the existing 10 anchors. Adding new draggable handles is a
  Phase 3 question — we want to see how the volumes read first.
- Gesture-quality curves (slight bowing / tapering on legs and neck). Polylines
  stay polylines.
- Multi-dog handling. Same single-largest-subject rule as today.
- Switching pose models. SuperAnimal Quadruped stays.

---

## Implementation Chunks

Each chunk is a single atomic commit (or small commit cluster) that leaves the
overlay in a working state. We verify in the browser preview against the
existing test image (`/Users/tjlefebvre/Desktop/image_proxy.webp`) at every
chunk boundary.

### Chunk A — Haunch + shoulder ovals (client-only)

**Scope**: render two new ovals per side using existing keypoints. No server
changes.

- Add helper `computeMassEllipse(rootAnchor, jointKp, axisRatio, sizeBoost)`
  that returns `{ cx, cy, rx, ry, angle }` for a 2D rotated ellipse with major
  axis along root → joint.
- Compute haunch ovals for back-L and back-R from `(hip, thigh_bl)` and
  `(hip, thigh_br)`. Skip if `thai` not detected (≥ 0.25).
- Compute shoulder ovals for front-L and front-R from `(withers, thigh_fl)` and
  `(withers, thigh_fr)`. Skip if `thai` not detected.
- Render with `bodyColor`, lower opacity than the main circles, so they read
  as secondary masses. Line weight × 0.9.
- Add render call to `drawConstruction` between the body circles and the legs.

**Files**: `dog-pose/sketch/sketch.js` only.
**Verify**: reload preview, drop user_dog.webp, screenshot. Confirm visible
haunch ovals where the dog's rear thighs are; shoulders may be small / hidden
in this 3/4 pose, which is acceptable.
**Commit**: `Phase 2.A — haunch + shoulder mass ovals`

### Chunk B — 4-segment legs with anatomical joint balls

**Scope**: replace 3-segment `slot → thigh → knee → paw` with 4-segment
`thigh → elbow/stifle → wrist/hock → paw`. Joint ball sizes vary per joint
type.

- Drop the "slot" concept from legs. The thigh keypoint *is* the body
  attachment. The body-circle edge is no longer used for legs.
- For each leg: compute thigh (slot), knee (elbow/stifle), wrist/hock as a
  point along knee→paw (default `WRIST_T = 0.55`). When user wants to refine,
  they drag the paw; joints recompute via the detected positions + the
  interpolation for the synthetic joint.
- Joint ball sizes per anatomical class (as fractions of `R`):
  ```
  JOINT_R_HIP_SHOULDER = 0.18   // largest
  JOINT_R_KNEE_ELBOW   = 0.13   // medium
  JOINT_R_HOCK_WRIST   = 0.09   // smallest
  ```
  Capped by the shorter adjacent segment length × 0.45 to avoid swamping.
- Joint balls drawn as small open circles at all joints; the hip / shoulder
  ball is the same point as the haunch / shoulder oval center, so visually
  the ball nests inside the oval — exactly like the howling-dog reference.
- Remove `KNEE_CIRCLE_K` and `FALLBACK_THIGH_T` / `FALLBACK_KNEE_T` (no
  longer relevant) — replaced by the joint-class constants.

**Files**: `dog-pose/sketch/sketch.js` only.
**Verify**: 3 of 4 legs in user_dog.webp should show 4 segments with
appropriately-sized joint balls; back-right leg falls back gracefully (paw
< 0.30 conf).
**Commit**: `Phase 2.B — 4-segment legs + anatomical joint balls`

### Chunk C — Server-side YOLOv8n-seg integration

**Scope**: swap the YOLO bbox model for the seg variant, extract a simplified
polygon, ship it in the response. No client changes yet.

- Update `dog-pose/server/model.py`:
  - `yolo = YOLO("yolov8n-seg.pt")` (replaces `yolov8n.pt`).
  - In `_detect_crop`, after picking `best`, pull `results[0].masks.xy[best]`
    — already in `(x, y)` pairs at original-image scale per the ultralytics
    docs (verify in `verify.py` — if seg coords are in resized space we
    unscale via the existing `/ scale` chain).
  - Simplify the polygon: implement a small Douglas-Peucker (or import from
    `shapely` if installed; if not, hand-write — it's ~20 lines). Target
    epsilon = 1% of bbox diagonal, max 64 points.
  - Add `silhouette: [[x,y], ...]` to the response shape. If mask missing,
    `silhouette: null`.
- Update `dog-pose/server/verify.py` to print `len(silhouette)` and the
  polygon's bounding box; run against both repo PNGs and the user's webp.
- No `requirements.txt` change — `ultralytics` already pulls the seg variant
  on demand.

**Files**: `dog-pose/server/model.py`, `dog-pose/server/verify.py`.
**Verify**: run `verify.py`; confirm polygon present, point count ≤ 64,
polygon bbox ≈ YOLO bbox (within ~10% on each side). The flask `/predict`
response should serialise to JSON without errors.
**Commit**: `Phase 2.C — YOLOv8n-seg silhouette polygon in server response`

### Chunk D — Client: silhouette backdrop + belly-direction refinement

**Scope**: ingest the silhouette polygon, render it faintly under everything
else, and use it to replace the "mean paw position" heuristic for belly
direction.

- In `sendToServer`'s response handler, stash `data.silhouette` on
  `serverResponse`.
- In `drawConstruction`, render the silhouette first (before any other
  primitive) as a closed polyline with very low stroke opacity (~40/255) and
  no fill, using the spine/neutral colour. Skip if `silhouette == null`.
- Update `computeFrame`: belly direction = `spineCenter → silhouette_centroid`
  when silhouette present and centroid is on the belly side of the spine
  (signed perpendicular check). Fall back to mean-paw when not.
- Update `extractAnchors` ghost-position branches to clamp into the silhouette
  polygon if available (so hard-floor fallbacks land inside the dog, not in
  an arbitrary bbox corner).

**Files**: `dog-pose/sketch/sketch.js` only.
**Verify**: silhouette traces the dog's outline. The belly-direction shift
should be invisible for typical photos (paws + silhouette agree), but for
unusual poses (e.g. lying down) the belly should still point in the right
direction.
**Commit**: `Phase 2.D — silhouette backdrop + silhouette-driven belly axis`

### Chunk E — Body-circle radius sanity check + render-order cleanup

**Scope**: small correctness pass once everything else is in place.

- In `computeFrame`, after computing `R` from `dist(withers, hip) × K`,
  also compute `R_silhouette` ≈ half-width of the silhouette near the spine
  centre. If `R > R_silhouette × 1.3`, clamp R to `R_silhouette × 1.15`
  (gentle, not aggressive).
- Re-order draw calls to match the back-to-front spec in this plan
  (silhouette → spine → haunches → body circles → neck → legs → tail →
  head → handles).
- Sanity-check that `Hide handles` still works and that the print/composite
  downloads include the new primitives.

**Files**: `dog-pose/sketch/sketch.js`.
**Verify**: end-to-end on user_dog.webp, the second test PNG, and at least
one new photo (any dog in a different pose). Compare side-by-side against
the reference sketches.
**Commit**: `Phase 2.E — radius clamp + render order + download polish`

### Chunk F — Tuning + docs

**Scope**: a single short pass to tune the new constants and update PLAN.md
to reflect Phase 2.

- Calibrate `JOINT_R_*`, haunch / shoulder oval size ratios, silhouette
  opacity. ~30 min of dial-twiddling against 3–4 photos.
- Append a "Phase 2 deltas" section to the original `PLAN.md` so the
  long-term plan reflects reality.
- Update `dog-pose/server/verify.py` so the smoke test also asserts
  silhouette presence + sanity.

**Files**: `PLAN.md`, `dog-pose/sketch/sketch.js`, `dog-pose/server/verify.py`.
**Commit**: `Phase 2.F — tuning + docs`

---

## Sequencing & Rollback

```
A ──▶ B ──▶ C ──▶ D ──▶ E ──▶ F
                    │
       client-only  │  server   │   client + tuning
```

- A, B are client-only and independent of the server work — they can ship
  first and stand alone if C–D are deferred.
- C is server-only and independently testable via `verify.py`.
- D depends on C.
- E depends on A, B, C, D (uses every new primitive).
- F is purely additive (tuning).

Each chunk is one commit on `main` (or its own branch + merge if you'd
rather review). Rollback at any point is `git reset --hard <prev>`.

## Estimated effort

| Chunk | Est | Risk |
|---|---|---|
| A — haunch + shoulder ovals | 45 min | low |
| B — 4-segment legs | 1.5 hr | low |
| C — server seg integration | 1.5 hr | medium (new model behaviour) |
| D — client silhouette + belly | 1 hr | low |
| E — radius clamp + render order | 30 min | low |
| F — tuning + docs | 45 min | low |
| **Total** | **~6 hr** | |

## Open questions (deferred to Phase 3 if relevant)

- Should joint balls be draggable so the user can refine a wrong knee bend?
  Decision deferred until we see how 4-segment legs read in real use.
- Should the silhouette be editable (e.g. dragging the underline curve)?
  Probably no — too much complexity for marginal gain over "drag the paw".
- Gesture-quality curves (bowed leg segments, tapered neck). Easy to add
  later; not required to match the references at a useful level.
- Auto-detected sit / lie / stand pose classification, to choose between
  oval sizes and angles per pose. Punt unless real-world use shows a gap.
