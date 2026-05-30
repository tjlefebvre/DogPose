// ── Tuning constants ─────────────────────────────────────────────────────────
const BODY_RADIUS_K        = 0.30;   // R = K × dist(withers, hip)
const HEAD_RADIUS_K         = 0.70;  // R = K × dist(ear_mid, nose)  — replaces eye-distance scheme
const HEAD_RADIUS_K_EYES    = 1.5;   // R = K × dist(eyes)            — kept for eyes-only fallback
// Joint ball radii as fractions of body radius R, ordered by anatomical
// importance. Capped at runtime by adjacent segment length × 0.45 so a small
// segment can never get swamped by its joint ball.
const JOINT_R_HIP_SHOULDER = 0.18;   // largest: where the leg meets the body
const JOINT_R_KNEE_ELBOW   = 0.13;   // medium:  major mid-leg bend
const JOINT_R_HOCK_WRIST   = 0.09;   // smallest: lower bend, near the paw
// Hock (rear) / wrist (front) sits along the knee→paw line at this fraction.
// (No DLC keypoint corresponds; we interpolate.)
const HOCK_WRIST_T         = 0.55;

// Silhouette backdrop opacity (0–255). Low — the polygon is contextual, not
// dominant. Bumped/dimmed in Chunk F tuning.
const SILHOUETTE_OPACITY   = 55;
// Slot angles in frame {x=spine_dir (withers→hip), y=belly_dir (toward paws)}.
// 0°=back of dog, 90°=down/belly, 180°=forward/nose, 270°=up/back-of-spine.
// Legs hang from the belly side: front legs forward+down (~135°), back legs back+down (~45°).
const SLOT_FRONT_LEFT_DEG  = 135;
const SLOT_FRONT_RIGHT_DEG = 135;
const SLOT_BACK_LEFT_DEG   = 45;
const SLOT_BACK_RIGHT_DEG  = 45;
const CONFIDENCE_DRAW      = 0.30;   // below this → handle only, no primitive contribution
const CONFIDENCE_GHOST     = 0.05;   // below this → treat as missing, use fallback
const CONFIDENCE_CONFIDENT = 0.60;   // above this → small faded handle; otherwise large bright+ring
// If back_end is detected farther than this × dist(nose, tail_base) from tail_base,
// treat it as a model error and use tail_base as the hip instead.
const HIP_SANITY_RATIO     = 0.55;

// Haunch (rear thigh) + shoulder (front thigh) mass ovals.
// Major axis runs root anchor (hip / withers) → detected thai keypoint.
// Minor axis is `axisRatio` × major axis.
const HAUNCH_AXIS_RATIO    = 0.60;   // rear thigh mass: chunky
const SHOULDER_AXIS_RATIO  = 0.55;   // front shoulder: leaner
// The oval extends beyond the root↔thai segment by this fraction at each end,
// so the visible ellipse wraps the joint instead of stopping at the keypoint.
const MASS_LENGTH_OVERSHOOT = 0.25;

// ── State ────────────────────────────────────────────────────────────────────
let img       = null;
let serverResponse = null;   // full response from /predict: {keypoints, bbox, failed, image_size}
let anchors   = null;        // extracted 10-anchor map
let loading   = false;
let hideHandles = false;

// Drag state
let dragIdx   = -1;          // index into ANCHOR_IDS being dragged
let lastClickTime = {};      // for double-click reset
let hoverKey  = null;        // anchor key currently under the mouse

// ── Anchor ID mapping ────────────────────────────────────────────────────────
// DLC SuperAnimal Quadruped keypoint ids we actually use
const KP = {
  nose:           0,
  upper_jaw:      1,
  lower_jaw:      2,
  mouth_end_r:    3,
  mouth_end_l:    4,
  right_eye:      5,
  right_earbase:  6,
  right_earend:   7,
  left_eye:       10,
  left_earbase:   11,
  left_earend:    12,
  neck_base:      15,
  withers:        16,  // neck_end
  back_base:      19,
  hip:            20,  // back_end
  tail_base:      22,
  tail_end:       23,
  thigh_fl:       24,  // front_left_thai
  knee_fl:        25,
  paw_fl:         26,
  thigh_fr:       27,
  knee_fr:        28,
  paw_fr:         29,
  paw_bl:         30,
  thigh_bl:       31,
  thigh_br:       32,
  knee_bl:        33,
  knee_br:        34,
  paw_br:         35,
};

// Confidence floor for joint/ear/jaw detail keypoints (separate from anchors).
const DETAIL_CONFIDENCE  = 0.25;

// Pull a single keypoint from the current serverResponse with a confidence floor.
function getRawKp(id, minConf = DETAIL_CONFIDENCE) {
  if (!serverResponse || !serverResponse.keypoints) return null;
  const k = serverResponse.keypoints.find(p => p.id === id);
  if (!k || k.confidence < minConf) return null;
  return { x: k.x, y: k.y, confidence: k.confidence };
}

// Ordered list so we can index by drag slot
const ANCHOR_IDS = [
  'right_eye','left_eye','nose',
  'withers','hip',
  'tail_base','tail_end',
  'paw_fl','paw_fr','paw_bl','paw_br'
];

const ANCHOR_LABELS = {
  right_eye:  'right eye',
  left_eye:   'left eye',
  nose:       'nose',
  withers:    'withers',
  hip:        'hip',
  tail_base:  'tail base',
  tail_end:   'tail end',
  paw_fl:     'front left paw',
  paw_fr:     'front right paw',
  paw_bl:     'back left paw',
  paw_br:     'back right paw',
};

// ── Setup ────────────────────────────────────────────────────────────────────
function setup() {
  const canvas = createCanvas(800, 600);
  canvas.parent(document.body);
  background(30);
  textAlign(CENTER, CENTER);
  fill(150);
  textSize(16);
  text("Upload a dog photo above", width / 2, height / 2);

  document.getElementById("fileInput").addEventListener("change", handleFile);
  document.getElementById("dlOverlay").addEventListener("click", downloadOverlay);
  document.getElementById("dlComposite").addEventListener("click", downloadComposite);
  document.getElementById("toggleHandles").addEventListener("click", () => {
    hideHandles = !hideHandles;
    document.getElementById("toggleHandles").textContent =
      hideHandles ? "Show handles" : "Hide handles";
    redraw();
  });
}

// ── File / server ─────────────────────────────────────────────────────────────
function handleFile(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    loadImage(ev.target.result, (loaded) => {
      img = loaded;
      serverResponse = null;
      anchors = null;
      redraw();
      sendToServer(file);
    });
  };
  reader.readAsDataURL(file);
}

function sendToServer(file) {
  loading = true;
  setStatus("Running inference…");
  redraw();

  const formData = new FormData();
  formData.append("image", file);

  fetch("http://127.0.0.1:5000/predict", { method: "POST", body: formData })
    .then((r) => r.json())
    .then((data) => {
      if (data.error) { setStatus("Error: " + data.error); return; }
      serverResponse = data;
      anchors = extractAnchors(data);
      loading = false;
      const failMsg = data.failed ? " [model failed — drag ghost handles]" : "";
      setStatus(`Inference complete${failMsg}`);
      redraw();
    })
    .catch((err) => {
      loading = false;
      setStatus("Server error — is server.py running?");
      console.error(err);
    });
}

// ── Anchor extraction ────────────────────────────────────────────────────────
function extractAnchors(data) {
  const kps = data.keypoints || [];
  const failed = data.failed || false;
  const bbox = data.bbox || [0, 0, (data.image_size||[800,600])[0], (data.image_size||[800,600])[1]];

  // Helper: pull a single keypoint by id with confidence floor
  function kp(id, minConf = CONFIDENCE_GHOST) {
    const k = kps.find(p => p.id === id);
    if (!k || k.confidence < minConf) return null;
    return { x: k.x, y: k.y, confidence: k.confidence };
  }

  // Ghost template positions relative to bbox — last-resort fallback only
  function ghostPos(slot) {
    const bx = bbox[0], by = bbox[1], bw = bbox[2]-bbox[0], bh = bbox[3]-bbox[1];
    const positions = {
      right_eye:  [bx + bw*0.60, by + bh*0.12],
      left_eye:   [bx + bw*0.70, by + bh*0.12],
      nose:       [bx + bw*0.65, by + bh*0.20],
      withers:    [bx + bw*0.55, by + bh*0.35],
      hip:        [bx + bw*0.30, by + bh*0.35],
      tail_base:  [bx + bw*0.18, by + bh*0.35],
      tail_end:   [bx + bw*0.08, by + bh*0.28],
      paw_fl:     [bx + bw*0.60, by + bh*0.85],
      paw_fr:     [bx + bw*0.50, by + bh*0.85],
      paw_bl:     [bx + bw*0.35, by + bh*0.85],
      paw_br:     [bx + bw*0.25, by + bh*0.85],
    };
    const p = positions[slot];
    return { x: p[0], y: p[1], confidence: 0, source: 'ghost' };
  }

  function tag(x, y, conf, source) {
    return { x, y, confidence: conf, source, detectedX: x, detectedY: y };
  }

  // Pull a single keypoint as a standard anchor (detected / inferred / null).
  function anchor(slot, kpId) {
    const raw = kp(kpId);
    if (failed) return ghostPos(slot);
    if (!raw) return null;
    const source = raw.confidence >= CONFIDENCE_DRAW ? 'detected' : 'inferred';
    return { x: raw.x, y: raw.y, confidence: raw.confidence, source,
             detectedX: raw.x, detectedY: raw.y };
  }

  // ── Head anchors ──────────────────────────────────────────────────────────
  // The eye keypoints often fail on dark-faced dogs. Earbases + nose are
  // detected much more reliably, so derive eyes from those when needed.
  const re_raw  = anchor('right_eye',     KP.right_eye);
  const le_raw  = anchor('left_eye',      KP.left_eye);
  const nose_raw = anchor('nose',         KP.nose);
  const re_ear  = anchor('right_earbase', KP.right_earbase);
  const le_ear  = anchor('left_earbase',  KP.left_earbase);

  // Treat low-confidence detected eye as "missing" for placement purposes
  // (the rendered handle is still draggable; the user can override).
  const eyeOk = (e) => e && e.confidence >= CONFIDENCE_DRAW;

  let right_eye, left_eye, nose;

  if (eyeOk(re_raw) && eyeOk(le_raw)) {
    // Best case: both eyes detected confidently.
    right_eye = re_raw;
    left_eye  = le_raw;
    nose      = nose_raw || (re_ear && le_ear ? deriveNoseFromEars(re_ear, le_ear, re_raw, le_raw) : ghostPos('nose'));
  } else if (re_ear && le_ear && nose_raw) {
    // Derive eyes from earbases + nose. Eyes sit about 40% of the way from
    // nose toward each earbase along a head-axis line.
    const ear_mid = { x: (re_ear.x + le_ear.x)/2, y: (re_ear.y + le_ear.y)/2 };
    const t = 0.55;  // 0 = at nose, 1 = at earbase
    right_eye = tag(nose_raw.x + (re_ear.x - nose_raw.x) * t,
                    nose_raw.y + (re_ear.y - nose_raw.y) * t,
                    Math.min(re_ear.confidence, nose_raw.confidence) * 0.5,
                    'inferred');
    left_eye  = tag(nose_raw.x + (le_ear.x - nose_raw.x) * t,
                    nose_raw.y + (le_ear.y - nose_raw.y) * t,
                    Math.min(le_ear.confidence, nose_raw.confidence) * 0.5,
                    'inferred');
    nose = nose_raw;
  } else if ((eyeOk(re_raw) || eyeOk(le_raw)) && nose_raw) {
    // Only one eye + nose: mirror the visible eye across the nose-head axis.
    nose = nose_raw;
    const visible = eyeOk(re_raw) ? re_raw : le_raw;
    const isRight = eyeOk(re_raw);
    const mirrored = mirrorEye(visible, nose_raw, re_ear || le_ear);
    if (isRight) { right_eye = visible; left_eye = mirrored; }
    else         { left_eye  = visible; right_eye = mirrored; }
  } else {
    // Last resort: ghost positions.
    right_eye = ghostPos('right_eye');
    left_eye  = ghostPos('left_eye');
    nose      = nose_raw || ghostPos('nose');
  }

  // ── Spine anchors with sanity check ──────────────────────────────────────
  const tail_base_raw = anchor('tail_base', KP.tail_base);
  const withers_raw   = anchor('withers',   KP.withers);
  const neck_base_raw = anchor('neck_base', KP.neck_base);
  const back_base_raw = anchor('back_base', KP.back_base);
  const hip_raw       = anchor('hip',       KP.hip);

  // Head reference point (for spine direction sanity)
  const head_ref = nose_raw
    || (re_ear && le_ear ? { x: (re_ear.x+le_ear.x)/2, y: (re_ear.y+le_ear.y)/2 } : null);

  let withers = withers_raw || back_base_raw || neck_base_raw;
  if (!withers) {
    // Fallback: 1/3 of the way from head to tail along the body axis.
    if (head_ref && tail_base_raw) {
      const wx = head_ref.x + (tail_base_raw.x - head_ref.x) * 0.30;
      const wy = head_ref.y + (tail_base_raw.y - head_ref.y) * 0.30;
      withers = tag(wx, wy, 0, 'inferred');
    } else {
      const bbox_midX = (bbox[0] + bbox[2]) / 2;
      const bbox_h = bbox[3] - bbox[1];
      withers = tag(bbox_midX, bbox[1] + bbox_h * 0.33, 0, 'inferred');
    }
  }

  // Hip = back_end (id 20). The model often emits it at low/mid confidence in
  // wildly wrong positions. Sanity-check against tail_base distance.
  let hip;
  const nose_to_tail = head_ref && tail_base_raw
    ? Math.hypot(tail_base_raw.x - head_ref.x, tail_base_raw.y - head_ref.y)
    : null;

  function hipFromTail(tb, w) {
    // Hip sits just in front of the tail base, biased toward withers.
    const dx = w.x - tb.x, dy = w.y - tb.y;
    const blend = 0.15;  // 0 = on tail, 1 = on withers
    return tag(tb.x + dx * blend, tb.y + dy * blend, 0, 'inferred');
  }

  if (hip_raw && tail_base_raw && nose_to_tail) {
    const d = Math.hypot(hip_raw.x - tail_base_raw.x, hip_raw.y - tail_base_raw.y);
    if (d > HIP_SANITY_RATIO * nose_to_tail) {
      // hip_raw is clearly mispredicted (e.g. landed near the head).
      hip = hipFromTail(tail_base_raw, withers);
    } else {
      hip = hip_raw;
    }
  } else if (hip_raw) {
    hip = hip_raw;
  } else if (tail_base_raw) {
    hip = hipFromTail(tail_base_raw, withers);
  } else if (withers && head_ref) {
    // Project past withers along head→withers direction.
    const dx = withers.x - head_ref.x, dy = withers.y - head_ref.y;
    hip = tag(withers.x + dx, withers.y + dy, 0, 'inferred');
  } else {
    hip = ghostPos('hip');
  }

  const tail_base = tail_base_raw || hipFromTail(hip, withers);
  const tail_end  = anchor('tail_end', KP.tail_end) || ghostPos('tail_end');

  const paw_fl = anchor('paw_fl', KP.paw_fl) || ghostPos('paw_fl');
  const paw_fr = anchor('paw_fr', KP.paw_fr) || ghostPos('paw_fr');
  const paw_bl = anchor('paw_bl', KP.paw_bl) || ghostPos('paw_bl');
  const paw_br = anchor('paw_br', KP.paw_br) || ghostPos('paw_br');

  return { right_eye, left_eye, nose, withers, hip, tail_base, tail_end,
           paw_fl, paw_fr, paw_bl, paw_br };
}

function deriveNoseFromEars(re_ear, le_ear, re, le) {
  // Project from ear midpoint through eye midpoint, out by ~eye-distance.
  const em = { x: (re_ear.x+le_ear.x)/2, y: (re_ear.y+le_ear.y)/2 };
  const eym = { x: (re.x+le.x)/2, y: (re.y+le.y)/2 };
  const dx = eym.x - em.x, dy = eym.y - em.y;
  return { x: eym.x + dx, y: eym.y + dy, confidence: 0, source: 'inferred',
           detectedX: eym.x + dx, detectedY: eym.y + dy };
}

function mirrorEye(visible, nose, earRef) {
  // Mirror `visible` across the nose-along-head-axis line.
  // axis direction = from nose toward an ear reference; if none, use horizontal.
  let ax, ay;
  if (earRef) { ax = earRef.x - nose.x; ay = earRef.y - nose.y; }
  else        { ax = 1; ay = 0; }
  const len2 = ax*ax + ay*ay || 1;
  // reflect (v - nose) across axis = 2 * proj - (v - nose)
  const vx = visible.x - nose.x, vy = visible.y - nose.y;
  const dot = (vx*ax + vy*ay) / len2;
  const px = dot * ax, py = dot * ay;
  const rx = 2*px - vx, ry = 2*py - vy;
  return { x: nose.x + rx, y: nose.y + ry, confidence: 0, source: 'inferred',
           detectedX: nose.x + rx, detectedY: nose.y + ry };
}

// ── Geometry helpers ──────────────────────────────────────────────────────────
function vec(a, b)    { return { x: b.x - a.x, y: b.y - a.y }; }
function len(v)       { return Math.hypot(v.x, v.y); }
function vnorm(v)     { const l = len(v) || 1; return { x: v.x/l, y: v.y/l }; }
function perp(v)      { return { x: -v.y, y: v.x }; }   // 90° CCW
function mid(a, b)    { return { x: (a.x+b.x)/2, y: (a.y+b.y)/2 }; }
function add(a, v, s) { return { x: a.x + v.x*s, y: a.y + v.y*s }; }   // a + v*s
function vdist(a, b)  { return Math.hypot(b.x-a.x, b.y-a.y); }

function silhouetteCentroid(poly) {
  // Shoelace centroid of a closed polygon. Falls back to vertex average for
  // degenerate (zero-area) input.
  if (!poly || poly.length < 3) return null;
  let area = 0, cx = 0, cy = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x0, y0] = poly[i];
    const [x1, y1] = poly[(i + 1) % poly.length];
    const cross = x0 * y1 - x1 * y0;
    area += cross;
    cx += (x0 + x1) * cross;
    cy += (y0 + y1) * cross;
  }
  area *= 0.5;
  if (Math.abs(area) < 1e-6) {
    // Degenerate — average the vertices.
    let sx = 0, sy = 0;
    for (const [x, y] of poly) { sx += x; sy += y; }
    return { x: sx / poly.length, y: sy / poly.length };
  }
  return { x: cx / (6 * area), y: cy / (6 * area) };
}

function computeFrame(an) {
  const spine_dir = vnorm(vec(an.withers, an.hip));
  const spineCenter = mid(an.withers, an.hip);

  // Prefer the silhouette centroid (more stable across unusual poses than the
  // mean paw position, which collapses when paws are missing or above the
  // spine for lying-down dogs). Fall back to mean-paw, then to perpendicular.
  let belly = null;
  const silhouette = serverResponse && serverResponse.silhouette;
  if (silhouette && silhouette.length >= 3) {
    const centroid = silhouetteCentroid(silhouette);
    if (centroid) {
      const d = vec(spineCenter, centroid);
      if (len(d) >= 5) belly = vnorm(d);
    }
  }
  if (!belly) {
    const paws = [an.paw_fl, an.paw_fr, an.paw_bl, an.paw_br];
    const meanPaw = {
      x: paws.reduce((s,p)=>s+p.x,0)/paws.length,
      y: paws.reduce((s,p)=>s+p.y,0)/paws.length,
    };
    if (len(vec(spineCenter, meanPaw)) >= 5) belly = vnorm(vec(spineCenter, meanPaw));
    else belly = perp(spine_dir);
  }

  const forward_dir = vnorm(vec(an.withers, an.nose));

  const S = vdist(an.withers, an.hip);
  let   R = BODY_RADIUS_K * S;

  // Silhouette-driven radius clamp: the keypoint-derived R sometimes blows up
  // (e.g. when withers / hip are mispredicted far apart). The silhouette gives
  // us the dog's actual half-width perpendicular to the spine — if R exceeds
  // that by > 30%, clamp it gently back to ~115% of the half-width.
  if (silhouette && silhouette.length >= 3) {
    const perpDir = perp(spine_dir);  // unit perpendicular to spine
    let maxAbsPerp = 0;
    for (const [vx, vy] of silhouette) {
      const dx = vx - spineCenter.x, dy = vy - spineCenter.y;
      const along = Math.abs(dx * perpDir.x + dy * perpDir.y);
      if (along > maxAbsPerp) maxAbsPerp = along;
    }
    if (maxAbsPerp > 1 && R > maxAbsPerp * 1.3) {
      R = maxAbsPerp * 1.15;
    }
  }

  return { spine_dir, belly_dir: belly, forward_dir, S, R };
}

function slotPos(center, frame, angleDeg) {
  const rad = (angleDeg * Math.PI) / 180;
  // Basis: x-axis = spine_dir (hip direction), y-axis = belly_dir (down)
  const cos = Math.cos(rad), sin = Math.sin(rad);
  const sx = frame.spine_dir, sy = frame.belly_dir;
  return {
    x: center.x + frame.R * (cos * sx.x + sin * sy.x),
    y: center.y + frame.R * (cos * sx.y + sin * sy.y),
  };
}

// ── Main draw ─────────────────────────────────────────────────────────────────
function draw() {
  background(30);

  if (!img) {
    fill(150); textSize(16); textAlign(CENTER, CENTER);
    text("Upload a dog photo above", width / 2, height / 2);
    return;
  }

  const sc = Math.min(width / img.width, height / img.height);
  const dw = img.width * sc, dh = img.height * sc;
  const ox = (width - dw) / 2, oy = (height - dh) / 2;

  image(img, ox, oy, dw, dh);

  if (loading) {
    fill(255, 200, 0, 200); textSize(14); textAlign(CENTER, CENTER);
    text("Analysing…", width / 2, 20);
    return;
  }

  if (!anchors) return;

  drawConstruction(anchors, ox, oy, sc, null, false);
}

// ── Construction renderer ─────────────────────────────────────────────────────
// ctx:     null → main canvas, else p5.Graphics
// forPrint: style tweak for white background
function drawConstruction(an, ox, oy, sc, ctx, forPrint) {
  const G = ctx || window;   // route drawing calls through ctx if provided

  // Map an anchor from image-space to canvas-space
  function px(a) { return ox + a.x * sc; }
  function py(a) { return oy + a.y * sc; }

  const frame = computeFrame(an);
  const { spine_dir, belly_dir, R, S } = frame;

  // Image-space + canvas-space body centres (shared across multiple sections).
  const frontC_img = add(an.withers, belly_dir, R);
  const backC_img  = add(an.hip,     belly_dir, R);
  const frontCX = ox + frontC_img.x * sc;
  const frontCY = oy + frontC_img.y * sc;
  const backCX  = ox + backC_img.x  * sc;
  const backCY  = oy + backC_img.y  * sc;
  const Rsc     = R * sc;

  // ── Head geometry (hoisted so the neck cylinder can use it) ────────────────
  // Prefer ear-midpoint↔nose axis (more stable on dark-faced dogs); fall back
  // to nose↔eye-midpoint, then to eye-distance.
  const r_earbase_raw = getRawKp(KP.right_earbase);
  const l_earbase_raw = getRawKp(KP.left_earbase);
  const eyeMid  = mid(an.right_eye, an.left_eye);
  const eyeDist = vdist(an.right_eye, an.left_eye);
  const noseToEyeMid = vdist(eyeMid, an.nose);
  let headC, headR;
  if (r_earbase_raw && l_earbase_raw) {
    const earMid = mid(r_earbase_raw, l_earbase_raw);
    headC = { x: earMid.x * 0.6 + an.nose.x * 0.4,
              y: earMid.y * 0.6 + an.nose.y * 0.4 };
    const earSpan = vdist(r_earbase_raw, l_earbase_raw);
    const noseToEar = vdist(earMid, an.nose);
    headR = Math.max(noseToEar * 0.65, earSpan * 0.55);
  } else if (noseToEyeMid > eyeDist * 0.4) {
    headC = { x: eyeMid.x + (eyeMid.x - an.nose.x) * 0.25,
              y: eyeMid.y + (eyeMid.y - an.nose.y) * 0.25 };
    headR = HEAD_RADIUS_K * noseToEyeMid * 1.6;
  } else {
    headC = eyeMid;
    headR = HEAD_RADIUS_K_EYES * eyeDist;
  }
  const headCX  = ox + headC.x * sc;
  const headCY  = oy + headC.y * sc;
  const headRsc = headR * sc;
  // Shared head axes.
  const noseVec = vnorm(vec(headC, an.nose));        // forward (skull → nose)
  const sideVec = { x: -noseVec.y, y: noseVec.x };   // perpendicular (left side)

  const lineW = forPrint ? 1.5 : 2;
  const bodyColor  = forPrint ? [60, 100, 180]  : [100, 160, 255];
  const headColor  = forPrint ? [180, 60,  60]  : [255, 100, 100];
  const legColor   = forPrint ? [60,  140,  60] : [100, 220, 120];
  const tailColor  = forPrint ? [140, 100,  40] : [220, 160,  60];
  const spineColor = forPrint ? [100, 100, 100] : [180, 180, 180];

  function col(c, a) {
    return ctx ? `rgba(${c[0]},${c[1]},${c[2]},${(a/255).toFixed(2)})`
               : color(c[0], c[1], c[2], a);
  }

  function strokeCol(c, a, w) {
    if (ctx) { ctx.stroke(c[0], c[1], c[2], a); ctx.strokeWeight(w); }
    else      { stroke(c[0], c[1], c[2], a); strokeWeight(w); }
  }
  function fillCol(c, a) {
    if (ctx) ctx.fill(c[0], c[1], c[2], a);
    else      fill(c[0], c[1], c[2], a);
  }
  function noF() { if (ctx) ctx.noFill(); else noFill(); }
  function noS() { if (ctx) ctx.noStroke(); else noStroke(); }

  function drawCircle(cx, cy, r) {
    if (ctx) ctx.circle(cx, cy, r*2);
    else circle(cx, cy, r*2);
  }
  function drawLine(x1,y1,x2,y2) {
    if (ctx) ctx.line(x1,y1,x2,y2);
    else line(x1,y1,x2,y2);
  }
  function drawEllipse(cx,cy,rw,rh) {
    if (ctx) ctx.ellipse(cx,cy,rw*2,rh*2);
    else ellipse(cx,cy,rw*2,rh*2);
  }
  // Rotated ellipse via push/translate/rotate. angle is in radians.
  function drawRotatedEllipse(cx, cy, rw, rh, angle) {
    if (ctx) {
      ctx.push(); ctx.translate(cx, cy); ctx.rotate(angle);
      ctx.ellipse(0, 0, rw*2, rh*2);
      ctx.pop();
    } else {
      push(); translate(cx, cy); rotate(angle);
      ellipse(0, 0, rw*2, rh*2);
      pop();
    }
  }
  // Compute and draw a mass oval whose major axis runs from `root` (image-space)
  // to `joint` (image-space). Returns null if joint missing.
  function drawMassEllipse(root, joint, axisRatio) {
    if (!root || !joint) return false;
    const cx_img = (root.x + joint.x) / 2;
    const cy_img = (root.y + joint.y) / 2;
    const dx = joint.x - root.x, dy = joint.y - root.y;
    const len = Math.hypot(dx, dy);
    if (len < 4) return false;  // degenerate
    const major = (len / 2) * (1 + MASS_LENGTH_OVERSHOOT);
    const minor = major * axisRatio;
    const angle = Math.atan2(dy, dx);
    drawRotatedEllipse(ox + cx_img * sc, oy + cy_img * sc,
                       major * sc, minor * sc, angle);
    return true;
  }
  function beginP() { if (ctx) ctx.beginShape(); else beginShape(); }
  function endP()   { if (ctx) ctx.endShape();   else endShape(); }
  function vtx(x,y) { if (ctx) ctx.vertex(x,y); else vertex(x,y); }

  // ── Silhouette backdrop ────────────────────────────────────────────────────
  // Faint outline of the dog from YOLO-seg; sits behind everything else so the
  // sketcher sees the full body shape as context, not as a dominant primitive.
  const silhouette = serverResponse && serverResponse.silhouette;
  if (silhouette && silhouette.length >= 3) {
    strokeCol(spineColor, SILHOUETTE_OPACITY, lineW * 0.8);
    noF();
    beginP();
    for (const [sx_img, sy_img] of silhouette) {
      vtx(ox + sx_img * sc, oy + sy_img * sc);
    }
    // close
    vtx(ox + silhouette[0][0] * sc, oy + silhouette[0][1] * sc);
    endP();
  }

  // ── Spine line ──────────────────────────────────────────────────────────────
  noF();
  strokeCol(spineColor, 120, lineW);
  drawLine(px(an.withers), py(an.withers), px(an.hip), py(an.hip));

  // ── Haunch + shoulder mass ovals (drawn behind body circles) ──────────────
  // Visible secondary masses where the rear thighs and front shoulders sit.
  // Major axis: hip → back_thai (haunch) or withers → front_thai (shoulder).
  // Skipped when the relevant thai keypoint is below DETAIL_CONFIDENCE.
  noF();
  strokeCol(bodyColor, 140, lineW * 0.9);
  const haunch_bl_kp = getRawKp(KP.thigh_bl);
  const haunch_br_kp = getRawKp(KP.thigh_br);
  const shoulder_fl_kp = getRawKp(KP.thigh_fl);
  const shoulder_fr_kp = getRawKp(KP.thigh_fr);
  drawMassEllipse(an.hip,     haunch_bl_kp,   HAUNCH_AXIS_RATIO);
  drawMassEllipse(an.hip,     haunch_br_kp,   HAUNCH_AXIS_RATIO);
  drawMassEllipse(an.withers, shoulder_fl_kp, SHOULDER_AXIS_RATIO);
  drawMassEllipse(an.withers, shoulder_fr_kp, SHOULDER_AXIS_RATIO);

  // ── Body circles (drawn on top of haunch/shoulder masses) ─────────────────
  noF();
  strokeCol(bodyColor, 200, lineW);
  drawCircle(frontCX, frontCY, Rsc);
  drawCircle(backCX,  backCY,  Rsc);

  // ── Neck cylinder ──────────────────────────────────────────────────────────
  // Two side lines between the front body-circle edge and the head-circle edge
  // along the head→withers direction. Read as a tapered tube.
  {
    const neckDir_img = vnorm(vec(frontC_img, headC));
    const neckSpan = vdist(frontC_img, headC);
    if (neckSpan > 1) {
      const neckBot = add(frontC_img, neckDir_img, R);
      const neckTop = add(headC,      neckDir_img, -headR);
      const sideN   = { x: -neckDir_img.y, y: neckDir_img.x };
      const wBot = R * 0.55;
      const wTop = headR * 0.55;
      const b1 = add(neckBot, sideN,  wBot);
      const b2 = add(neckBot, sideN, -wBot);
      const t1 = add(neckTop, sideN,  wTop);
      const t2 = add(neckTop, sideN, -wTop);
      strokeCol(bodyColor, 150, lineW * 0.9);
      noF();
      drawLine(ox + b1.x*sc, oy + b1.y*sc, ox + t1.x*sc, oy + t1.y*sc);
      drawLine(ox + b2.x*sc, oy + b2.y*sc, ox + t2.x*sc, oy + t2.y*sc);
    }
  }

  // ── Legs ───────────────────────────────────────────────────────────────────
  // 4-segment construction (Loomis / Walt Stanchfield style):
  //   thigh (hip/shoulder) → knee (stifle/elbow) → hock/wrist → paw
  // The `thai` keypoint *is* the body attachment — no separate body-circle slot.
  // Hock (rear) and wrist (front) have no DLC keypoint; we interpolate them
  // along knee→paw at HOCK_WRIST_T. Joint balls scale by anatomical importance.
  strokeCol(legColor, 200, lineW);
  noF();

  const legDefs = [
    { fallbackAngle: SLOT_FRONT_LEFT_DEG,  center: an.withers, bodyC: frontC_img, paw: an.paw_fl, thighId: KP.thigh_fl, kneeId: KP.knee_fl },
    { fallbackAngle: SLOT_FRONT_RIGHT_DEG, center: an.withers, bodyC: frontC_img, paw: an.paw_fr, thighId: KP.thigh_fr, kneeId: KP.knee_fr },
    { fallbackAngle: SLOT_BACK_LEFT_DEG,   center: an.hip,     bodyC: backC_img,  paw: an.paw_bl, thighId: KP.thigh_bl, kneeId: KP.knee_bl },
    { fallbackAngle: SLOT_BACK_RIGHT_DEG,  center: an.hip,     bodyC: backC_img,  paw: an.paw_br, thighId: KP.thigh_br, kneeId: KP.knee_br },
  ];

  function cappedJointR(base, ...adjacent) {
    return Math.min(base, ...adjacent.map(l => l * 0.45));
  }

  for (const leg of legDefs) {
    const pawValid = leg.paw && leg.paw.source !== 'ghost';

    // Thigh = the leg's body attachment (shoulder for front, hip for rear).
    const thighRaw = getRawKp(leg.thighId);
    let thigh_img;
    if (thighRaw) {
      thigh_img = thighRaw;
    } else if (pawValid) {
      // Paw-driven fallback: body-circle edge facing the paw.
      const dx = leg.paw.x - leg.bodyC.x, dy = leg.paw.y - leg.bodyC.y;
      const d = Math.hypot(dx, dy) || 1;
      thigh_img = { x: leg.bodyC.x + (dx/d) * R, y: leg.bodyC.y + (dy/d) * R };
    } else {
      thigh_img = slotPos(leg.center, frame, leg.fallbackAngle);
    }
    const tx = ox + thigh_img.x * sc, ty = oy + thigh_img.y * sc;
    const pw = px(leg.paw), ph = py(leg.paw);

    // Knee/elbow: detected or midpoint of thigh→paw.
    const kneeRaw = getRawKp(leg.kneeId);
    const knee_img = kneeRaw
      ? kneeRaw
      : { x: (thigh_img.x + leg.paw.x) / 2, y: (thigh_img.y + leg.paw.y) / 2 };
    const kx = ox + knee_img.x * sc, ky = oy + knee_img.y * sc;

    // Hock/wrist: interpolated along knee→paw.
    const wx = kx + (pw - kx) * HOCK_WRIST_T;
    const wy = ky + (ph - ky) * HOCK_WRIST_T;

    // Polyline: thigh → knee → hock/wrist → paw.
    drawLine(tx, ty, kx, ky);
    drawLine(kx, ky, wx, wy);
    drawLine(wx, wy, pw, ph);

    // Joint balls. Sizes capped by adjacent segment length so they never swamp.
    const segTK = Math.hypot(kx - tx, ky - ty);
    const segKW = Math.hypot(wx - kx, wy - ky);
    const segWP = Math.hypot(pw - wx, ph - wy);

    const rHipShoulder = cappedJointR(JOINT_R_HIP_SHOULDER * Rsc, segTK);
    const rKneeElbow   = cappedJointR(JOINT_R_KNEE_ELBOW   * Rsc, segTK, segKW);
    const rHockWrist   = cappedJointR(JOINT_R_HOCK_WRIST   * Rsc, segKW, segWP);

    drawCircle(tx, ty, rHipShoulder);
    drawCircle(kx, ky, rKneeElbow);
    drawCircle(wx, wy, rHockWrist);
  }

  // ── Tail ───────────────────────────────────────────────────────────────────
  strokeCol(tailColor, 200, lineW);
  noF();
  drawLine(px(an.tail_base), py(an.tail_base), px(an.tail_end), py(an.tail_end));

  // ── Head sphere + construction lines ──────────────────────────────────────
  // headC, headR, headCX/CY, headRsc, noseVec, sideVec computed up-top so the
  // neck cylinder section can also use them.
  strokeCol(headColor, 200, lineW);
  noF();
  drawCircle(headCX, headCY, headRsc);

  // Extended cross-axes spanning the full sphere (Loomis center planes):
  // 1. Nose axis: front (toward nose) to back of skull.
  const noseAxisFront = add(headC, noseVec,  headR);
  const noseAxisBack  = add(headC, noseVec, -headR);
  strokeCol(headColor, 150, lineW * 0.7);
  drawLine(ox + noseAxisBack.x*sc,  oy + noseAxisBack.y*sc,
           ox + noseAxisFront.x*sc, oy + noseAxisFront.y*sc);

  // 2. Eye axis: extend through the detected eye direction across the sphere.
  let eyeAxis;
  if (eyeDist > 1) {
    eyeAxis = vnorm(vec(an.right_eye, an.left_eye));
  } else {
    eyeAxis = sideVec;  // fall back to nose-perpendicular
  }
  const eyeAxisR = add(headC, eyeAxis, -headR);
  const eyeAxisL = add(headC, eyeAxis,  headR);
  drawLine(ox + eyeAxisR.x*sc, oy + eyeAxisR.y*sc,
           ox + eyeAxisL.x*sc, oy + eyeAxisL.y*sc);

  // Eye-to-eye solid line (highlights the actual detected eye positions).
  strokeCol(headColor, 200, lineW);
  drawLine(px(an.right_eye), py(an.right_eye), px(an.left_eye), py(an.left_eye));

  // Centre-to-nose solid stroke on top of the back-extended axis.
  drawLine(headCX, headCY, px(an.nose), py(an.nose));

  // ── Muzzle box (kept as the Loomis snout volume) ──────────────────────────
  const muzzleLen = Math.max(vdist(headC, an.nose) - headR * 0.5, headR * 0.3);
  const muzzleW   = headR * 0.4;
  const muzzleRootW = headR * 0.55;
  const rootPt  = add(headC, noseVec, headR * 0.7);
  const tipPt   = add(headC, noseVec, headR * 0.7 + muzzleLen);
  const r1 = add(rootPt, sideVec,  muzzleRootW);
  const r2 = add(rootPt, sideVec, -muzzleRootW);
  const t1 = add(tipPt,  sideVec,  muzzleW);
  const t2 = add(tipPt,  sideVec, -muzzleW);
  strokeCol(headColor, 130, lineW * 0.7);
  noF();
  beginP();
  vtx(ox + r1.x*sc, oy + r1.y*sc);
  vtx(ox + t1.x*sc, oy + t1.y*sc);
  vtx(ox + t2.x*sc, oy + t2.y*sc);
  vtx(ox + r2.x*sc, oy + r2.y*sc);
  endP();

  // ── Jaw line (from detected mouth-end keypoints when available) ───────────
  const mouth_r_raw = getRawKp(KP.mouth_end_r);
  const mouth_l_raw = getRawKp(KP.mouth_end_l);
  if (mouth_r_raw && mouth_l_raw) {
    strokeCol(headColor, 180, lineW * 0.8);
    noF();
    // Mouth-end to mouth-end across the muzzle.
    drawLine(ox + mouth_r_raw.x*sc, oy + mouth_r_raw.y*sc,
             ox + mouth_l_raw.x*sc, oy + mouth_l_raw.y*sc);
    // Faint strokes from each mouth-end back to the nose for the muzzle outline.
    strokeCol(headColor, 110, lineW * 0.6);
    drawLine(ox + mouth_r_raw.x*sc, oy + mouth_r_raw.y*sc, px(an.nose), py(an.nose));
    drawLine(ox + mouth_l_raw.x*sc, oy + mouth_l_raw.y*sc, px(an.nose), py(an.nose));
  }

  // ── Ears ──────────────────────────────────────────────────────────────────
  // Prefer real earbase + earend keypoints; fall back to decorative triangles
  // on the upper arc of the head circle when ears aren't detected.
  const r_earend_raw = getRawKp(KP.right_earend);
  const l_earend_raw = getRawKp(KP.left_earend);

  function drawDetectedEar(base, end) {
    if (!base) return false;
    if (!end) {
      // Just a stub from base toward "up" (away from belly) on the head circle.
      const upDir = vnorm(vec(an.withers || headC, headC));
      const tipP = add(base, upDir, headR * 0.3);
      strokeCol(headColor, 140, lineW * 0.8);
      noF();
      drawLine(ox + base.x*sc, oy + base.y*sc, ox + tipP.x*sc, oy + tipP.y*sc);
      return true;
    }
    // Triangle: base — earend (tip) — base offset perpendicular by ~30% length.
    const d = vdist(base, end);
    const dir = vnorm(vec(base, end));
    const sidePerp = { x: -dir.y, y: dir.x };
    const baseHalf = Math.max(d * 0.25, headR * 0.18);
    const bL = add(base, sidePerp,  baseHalf);
    const bR = add(base, sidePerp, -baseHalf);
    strokeCol(headColor, 150, lineW * 0.8);
    noF();
    beginP();
    vtx(ox + bL.x*sc,  oy + bL.y*sc);
    vtx(ox + end.x*sc, oy + end.y*sc);
    vtx(ox + bR.x*sc,  oy + bR.y*sc);
    vtx(ox + bL.x*sc,  oy + bL.y*sc);  // close the triangle
    endP();
    return true;
  }

  const rEarDrawn = drawDetectedEar(r_earbase_raw, r_earend_raw);
  const lEarDrawn = drawDetectedEar(l_earbase_raw, l_earend_raw);

  // Decorative-ear fallback for any side that wasn't drawn from real keypoints.
  if (!rEarDrawn || !lEarDrawn) {
    const earBase = 0.35;
    for (const [side, drawn] of [[-1, rEarDrawn], [1, lEarDrawn]]) {
      if (drawn) continue;
      const earAnchor = {
        x: headC.x + (-noseVec.y) * side * headR * 0.55 - noseVec.x * headR * 0.35,
        y: headC.y + (-noseVec.x) * side * headR * 0.55 - noseVec.y * headR * 0.35,
      };
      const earTip = {
        x: earAnchor.x + (-noseVec.y) * side * headR * earBase * 0.3
                       - noseVec.x * headR * earBase * 1.1,
        y: earAnchor.y + (-noseVec.x) * side * headR * earBase * 0.3
                       - noseVec.y * headR * earBase * 1.1,
      };
      const earL = { x: earAnchor.x + (-noseVec.y) * side * headR * earBase,
                     y: earAnchor.y + (-noseVec.x) * side * headR * earBase };
      const earR = { x: earAnchor.x - (-noseVec.y) * side * headR * earBase * 0.3,
                     y: earAnchor.y - (-noseVec.x) * side * headR * earBase * 0.3 };
      strokeCol(headColor, 130, lineW * 0.7);
      noF();
      beginP();
      vtx(ox + earL.x*sc, oy + earL.y*sc);
      vtx(ox + earTip.x*sc, oy + earTip.y*sc);
      vtx(ox + earR.x*sc, oy + earR.y*sc);
      endP();
    }
  }

  // ── Handles ────────────────────────────────────────────────────────────────
  if (!hideHandles && !forPrint) {
    drawHandles(an, ox, oy, sc, ctx);
  }
}

function drawHandles(an, ox, oy, sc, ctx) {
  const handleList = [
    { key: 'right_eye', a: an.right_eye },
    { key: 'left_eye',  a: an.left_eye  },
    { key: 'nose',      a: an.nose      },
    { key: 'withers',   a: an.withers   },
    { key: 'hip',       a: an.hip       },
    { key: 'tail_base', a: an.tail_base },
    { key: 'tail_end',  a: an.tail_end  },
    { key: 'paw_fl',    a: an.paw_fl    },
    { key: 'paw_fr',    a: an.paw_fr    },
    { key: 'paw_bl',    a: an.paw_bl    },
    { key: 'paw_br',    a: an.paw_br    },
  ];

  for (const { key, a } of handleList) {
    const cx = ox + a.x * sc;
    const cy = oy + a.y * sc;
    const confident = a.confidence >= CONFIDENCE_CONFIDENT;
    const isGhost   = a.source === 'ghost' || a.source === 'inferred';

    let hR, fillC, strokeC, alpha;
    if (confident) {
      hR = 5; fillC = [200, 200, 200]; strokeC = [255,255,255]; alpha = 120;
    } else {
      hR = 8; fillC = [255, 200, 50]; strokeC = [255, 255, 100]; alpha = 220;
    }

    if (ctx) {
      ctx.strokeWeight(1.5);
      ctx.stroke(strokeC[0], strokeC[1], strokeC[2], alpha);
      ctx.fill(fillC[0], fillC[1], fillC[2], confident ? 60 : 100);
      ctx.circle(cx, cy, hR * 2);
      if (!confident) {
        // outer ring
        ctx.noFill();
        ctx.stroke(strokeC[0], strokeC[1], strokeC[2], 120);
        ctx.circle(cx, cy, (hR + 4) * 2);
      }
    } else {
      strokeWeight(1.5);
      stroke(strokeC[0], strokeC[1], strokeC[2], alpha);
      fill(fillC[0], fillC[1], fillC[2], confident ? 60 : 100);
      circle(cx, cy, hR * 2);
      if (!confident) {
        noFill();
        stroke(strokeC[0], strokeC[1], strokeC[2], 120);
        circle(cx, cy, (hR + 4) * 2);
      }
    }
  }

  // Hover label
  if (hoverKey && anchors[hoverKey]) {
    const a  = anchors[hoverKey];
    const lx = ox + a.x * sc;
    const ly = oy + a.y * sc;
    const label = ANCHOR_LABELS[hoverKey] || hoverKey;
    const pad = 4, fs = 11;
    textSize(fs);
    const tw = textWidth(label);
    // Flip to left if too close to the right edge
    const bx = (lx + 14 + tw + pad * 2 > width) ? lx - 14 - tw - pad * 2 : lx + 14;
    const by = ly - fs / 2 - pad;
    noStroke();
    fill(0, 0, 0, 180);
    rect(bx - pad, by, tw + pad * 2, fs + pad * 2, 3);
    fill(255, 255, 255, 220);
    textAlign(LEFT, TOP);
    text(label, bx, by + pad);
  }
}

// ── Mouse interaction ──────────────────────────────────────────────────────────
function mousePressed() {
  if (!anchors) return;
  const sc = Math.min(width / img.width, height / img.height);
  const ox = (width - img.width * sc) / 2;
  const oy = (height - img.height * sc) / 2;

  const now = millis();
  for (let i = 0; i < ANCHOR_IDS.length; i++) {
    const key = ANCHOR_IDS[i];
    const a = anchors[key];
    if (!a) continue;
    const cx = ox + a.x * sc, cy = oy + a.y * sc;
    if (dist2(mouseX, mouseY, cx, cy) < 12) {
      // Double-click detection
      if (lastClickTime[key] && (now - lastClickTime[key]) < 400) {
        // Reset to detected position
        if (a.detectedX !== undefined) {
          anchors[key].x = a.detectedX;
          anchors[key].y = a.detectedY;
        }
        dragIdx = -1;
        redraw();
        return;
      }
      lastClickTime[key] = now;
      dragIdx = i;
      return;
    }
  }
  dragIdx = -1;
}

function mouseDragged() {
  if (dragIdx < 0 || !anchors || !img) return;
  const sc = Math.min(width / img.width, height / img.height);
  const ox = (width - img.width * sc) / 2;
  const oy = (height - img.height * sc) / 2;
  const key = ANCHOR_IDS[dragIdx];
  anchors[key].x = (mouseX - ox) / sc;
  anchors[key].y = (mouseY - oy) / sc;
  redraw();
}

function mouseReleased() { dragIdx = -1; }

function mouseMoved() {
  if (!anchors || !img || hideHandles) {
    if (hoverKey !== null) { hoverKey = null; redraw(); }
    return;
  }
  const sc = Math.min(width / img.width, height / img.height);
  const ox = (width - img.width * sc) / 2;
  const oy = (height - img.height * sc) / 2;
  let found = null;
  for (const key of ANCHOR_IDS) {
    const a = anchors[key];
    if (!a) continue;
    if (dist2(mouseX, mouseY, ox + a.x * sc, oy + a.y * sc) < 16) { found = key; break; }
  }
  if (found !== hoverKey) { hoverKey = found; redraw(); }
}

function dist2(x1, y1, x2, y2) { return Math.hypot(x1-x2, y1-y2); }

// ── Download ──────────────────────────────────────────────────────────────────
function downloadOverlay() {
  if (!img || !anchors) return;

  const LEGEND_W  = 180;
  const LEGEND_PAD = 8;
  const totalW    = width + LEGEND_W;

  const sc = Math.min(width / img.width, height / img.height);
  const dw = img.width * sc, dh = img.height * sc;
  const ox = (width - dw) / 2, oy = (height - dh) / 2;

  const g = createGraphics(totalW, height);
  g.background(255);

  // Faded photo
  g.tint(255, 60);
  g.image(img, ox, oy, dw, dh);
  g.noTint();

  // Construction primitives (handles hidden)
  drawConstruction(anchors, ox, oy, sc, g, true);

  // Legend panel
  const legX = width;
  g.fill(245); g.noStroke();
  g.rect(legX, 0, LEGEND_W, height);

  g.fill(0); g.textSize(9); g.textStyle(BOLD); g.textAlign(LEFT, TOP);
  g.text("Construction guide", legX + LEGEND_PAD, LEGEND_PAD);
  g.textStyle(NORMAL);

  const entries = [
    [[100,160,255], "ribcage circle"],
    [[100,160,255], "pelvis circle"],
    [[255,100,100], "head circle"],
    [[255,100,100], "muzzle / ears"],
    [[100,220,120], "leg axes + knees"],
    [[220,160, 60], "tail"],
    [[180,180,180], "spine"],
  ];
  let ly = LEGEND_PAD + 18;
  for (const [[r,c,b], label] of entries) {
    g.fill(r,c,b); g.noStroke();
    g.rect(legX + LEGEND_PAD, ly + 2, 7, 7);
    g.fill(0); g.textSize(8); g.textAlign(LEFT, TOP);
    g.text(label, legX + LEGEND_PAD + 11, ly);
    ly += 14;
  }

  saveCanvas(g, "dog_pose_print", "png");
  g.remove();
}

function downloadComposite() {
  if (!img || !anchors) { saveCanvas("dog_pose_composite", "png"); return; }

  const sc = Math.min(width / img.width, height / img.height);
  const dw = img.width * sc, dh = img.height * sc;
  const ox = (width - dw) / 2, oy = (height - dh) / 2;

  const g = createGraphics(width, height);
  g.background(30);
  g.image(img, ox, oy, dw, dh);
  drawConstruction(anchors, ox, oy, sc, g, true);   // forPrint=true → no handles
  saveCanvas(g, "dog_pose_composite", "png");
  g.remove();
}

function setStatus(msg) {
  document.getElementById("status").textContent = msg;
}
