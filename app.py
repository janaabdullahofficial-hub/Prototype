"""
Baseer – AI Early Multi-Modal Anomaly Detection & Triage Command Center
Full Stream Control with Smooth Frame-by-Frame Rendering

FIX NOTE (frame skipping issue):
The previous version processed the whole video inside one long Python
while-loop and called `cam_holder.image(...)` on every iteration. When
updates to the SAME placeholder arrive faster than the browser/websocket
can flush them, Streamlit's ForwardMsgQueue coalesces consecutive deltas
for that element and keeps only the latest one — so visually you only
ever see the first frame (sent before the backlog built up) and the very
last frame (the final state once the loop ends). No amount of
`time.sleep()` inside the loop fixes this, because the whole script is
still one uninterrupted run.

The fix: stop doing one giant loop. Instead, keep a background thread
producing frames into a queue (unchanged), and consume ONE frame per
tick using `st.fragment(run_every=...)`. Each fragment tick is a real,
independently-flushed rerun of just that fragment, so every frame
actually gets painted instead of being dropped.

Requires streamlit >= 1.33 (for st.fragment with run_every). If you're
on an older version: `pip install --upgrade streamlit`.

FIX NOTE 2 (broken image / "Missing file" icon):
`st.image()` writes each frame through Streamlit's MediaFileManager,
which serves it as a separate static file fetch. At ~30ms/frame that
fetch races with MediaFileManager's cleanup of the previous frame's
file, so the browser sometimes requests a file that's already been
evicted -> broken image icon + "MediaFileManager: Missing file" in the
logs (a long-standing Streamlit issue with rapid st.image updates).
Fix: skip MediaFileManager entirely by inlining each frame as a
base64 data URI inside the same st.markdown() call that already
carries the rest of the fragment's HTML — no separate file fetch, so
no race.

FIX NOTE 3 (frames still look like they're skipping / not smooth):
Even with the fragment fix, the producer thread used to read the whole
video as fast as cv2 could decode it and dump every frame into the
queue immediately. The consumer then drained the ENTIRE backlog on
every tick and displayed only the newest frame ("catch up" strategy).
That keeps the render *timing* smooth (steady 12 fps ticks) but the
*content* jumps — you'd see frame 1, then frame 40, then frame 90,
etc., which reads as choppy/skippy even though no Streamlit message
was ever dropped.

Real fix: pace the PRODUCER itself, using time.sleep, so it emits
frames at (roughly) the same rate the UI can actually paint them —
i.e. genuine real-time playback like a live broadcast, instead of
"decode everything instantly, then fast-forward-and-catch-up on
screen". The consumer now normally finds ~1 fresh frame per tick
instead of a backlog, so what's shown is the video in true order with
no visual jumps. A small drain cap is kept purely as a safety net for
transient hiccups (e.g. a slow detection pass on one frame), not as
the primary pacing mechanism anymore.

FIX NOTE 4 (accuracy — false positive anomaly alerts):
Two changes address most of the false-alarm rate:
1) Foreground segmentation moved from raw two-frame differencing to
   `cv2.createBackgroundSubtractorMOG2`, which builds an actual
   statistical background model. Frame differencing lights up on any
   pixel that changed between two frames — camera micro-jitter,
   compression noise, lighting flicker, shadows — all of which used to
   produce spurious motion blobs that could immediately trigger a
   classification. MOG2 + shadow-channel thresholding + morphological
   open/close removes the vast majority of that noise before it ever
   reaches the tracker.
2) Temporal confirmation ("hysteresis") on alerts. Previously a single
   noisy frame where a track's features happened to cross a threshold
   was enough to fire "fighting" / "sudden_fall" / etc. Now a track
   must be classified the SAME way for several consecutive frames
   (scaled by sensitivity) before anything is drawn or logged. This is
   the single biggest lever against one-off misclassifications, and it
   costs only a small, expected amount of detection latency.
"""

import base64
import math
import os
import queue
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

# ============================================================================
# PAGE CONFIG & STYLES
# ============================================================================

st.set_page_config(
    page_title="Baseer | AI Anomaly Detection Platform",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;900&family=JetBrains+Mono:wght@500;700&display=swap');
    * { font-family: 'Inter', -apple-system, sans-serif; }
    code, .mono { font-family: 'JetBrains Mono', monospace !important; }

    .block-container { padding-top: 1.2rem; max-width: 1440px; }
    .header-box {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg, #0B132B 0%, #1C2541 100%);
        padding: 1rem 1.4rem;
        border-radius: 12px;
        border: 1px solid #3A506B;
        margin-bottom: 1.2rem;
    }
    .system-title {
        font-size: 1.85rem;
        font-weight: 900;
        background: linear-gradient(90deg, #48CAE4, #00B4D8, #90E0EF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .system-sub { color: #94A3B8; font-size: 0.88rem; margin: 0.2rem 0 0 0; }

    .live-badge {
        background: #DC2626;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.08em;
    }
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.5rem;
        margin: 0.5rem 0 0.8rem 0;
    }
    .kpi-card {
        background: #0D1B2A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 0.5rem 0.4rem;
        text-align: center;
    }
    .kpi-num { font-size: 1.25rem; font-weight: 700; color: #38BDF8; font-family: 'JetBrains Mono', monospace; }
    .kpi-title { font-size: 0.72rem; color: #64748B; font-weight: 700; }

    .alert-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 0.9rem;
        margin-bottom: 0.8rem;
    }
    .triage-badge {
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 800;
        color: white;
        font-family: 'JetBrains Mono', monospace;
    }
    .card-title { font-size: 1.05rem; font-weight: 800; color: #F8FAFC; margin-top: 0.4rem; }
    .card-en { font-size: 0.82rem; color: #94A3B8; margin-bottom: 0.25rem; }
    .card-meta { color: #64748B; font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; }
    .category-tag {
        display: inline-block;
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border-radius: 4px;
        padding: 1px 6px;
        font-size: 0.7rem;
        margin-bottom: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# TAXONOMY MAPPING
# ============================================================================

TAXONOMY_RULES = {
    "sunstroke_heat_exhaustion": {
        "category": "Heat Emergencies & Insolation",
        "title": "Sunstroke / Extreme Heat Exhaustion",
        "en": "sunstroke_heat_exhaustion",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "☀️",
        "action": "Immediate evacuation to cooling shelter & emergency medical response.",
    },
    "fighting": {
        "category": "Physical Violence & Assaults",
        "title": "Physical Altercation / Fighting",
        "en": "fighting",
        "priority": "High",
        "color": "#F97316",
        "icon": "🥊",
        "action": "Dispatch security officers immediately to de-escalate.",
    },
    "sudden_fall": {
        "category": "Falls & Medical Emergencies",
        "title": "Sudden Fall & Sudden Balance Loss",
        "en": "sudden_fall",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "🚨",
        "action": "Dispatch Rapid Response Team to location.",
    },
    "sudden_fall_followed_by_seizure": {
        "category": "Falls & Medical Emergencies",
        "title": "Sudden Fall Followed by Seizure",
        "en": "sudden_fall_followed_by_seizure",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "⚡",
        "action": "Secure perimeter, protect head & dispatch paramedic urgent team.",
    },
    "lying_immobile": {
        "category": "Falls & Medical Emergencies",
        "title": "Person Lying Immobile / Unconscious",
        "en": "lying_immobile",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "🛑",
        "action": "Dispatch CPR medical response unit.",
    },
    "regular_limping": {
        "category": "Abnormal Gait & Fatigue",
        "title": "Steady Limp (Balanced, Consistent Gait)",
        "en": "regular_limping",
        "priority": "Medium",
        "color": "#F59E0B",
        "icon": "🚶",
        "action": "Dispatch mobile wheelchair transport.",
    },
    "irregular_limping": {
        "category": "Abnormal Gait & Fatigue",
        "title": "Unbalanced / Asymmetric Limp (Losing Steadiness)",
        "en": "irregular_limping",
        "priority": "High",
        "color": "#F97316",
        "icon": "👣",
        "action": "Alert field triage point for lower limb inspection.",
    },
    "Exhausted_walking": {
        "category": "Abnormal Gait & Fatigue",
        "title": "Severe Physical Exhaustion / Weak Gait",
        "en": "Exhausted_walking",
        "priority": "High",
        "color": "#F97316",
        "icon": "😫",
        "action": "Provide hydration and escort to rest station.",
    },
}

PRIORITY_COLOR = {"Critical": "#DC2626", "High": "#F97316", "Medium": "#F59E0B", "Low": "#3B82F6"}

LOCATIONS = [
    "Pilgrim Corridor 12 (Mashaer Walkway)",
    "King Fahd Gate - Central Courtyard",
    "Haramain Train Station - Terminal 2",
    "Field Hospital - Jamarat Bridge Precinct",
]

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Alert:
    id: str
    unique_key: str
    frame_idx: int
    video_time_s: float
    wall_clock: str
    location: str
    condition_key: str
    confidence: float


class Track:
    def __init__(self, track_id, centroid, bbox, frame_idx):
        self.id = track_id
        self.history = deque(maxlen=15)
        self.age = 0
        # Temporal-confirmation state (FIX NOTE 4): a classification only
        # "counts" once the SAME condition has been seen on several
        # consecutive frames for this track.
        self.pending_cond = None
        self.pending_count = 0
        self.update(centroid, bbox, frame_idx)

    def update(self, centroid, bbox, frame_idx):
        self.centroid = centroid
        self.bbox = bbox
        self.last_seen = frame_idx
        self.age += 1
        self.history.append({"c": centroid, "b": bbox, "f": frame_idx})

    def confirm(self, cond, needed):
        """Register this frame's raw classification and report whether it
        has now been seen `needed` times in a row. Returns True only on
        confirmation (so callers don't re-fire every subsequent frame)."""
        if cond == self.pending_cond:
            self.pending_count += 1
        else:
            self.pending_cond = cond
            self.pending_count = 1
        return cond is not None and self.pending_count == needed


def extract_features(track: Track):
    hist = list(track.history)
    # Require a slightly longer history than before (5 vs 3) so a track's
    # speed/jitter estimate isn't dominated by 1-2 noisy samples right
    # after it's created.
    if len(hist) < 5:
        return None

    heights = [h["b"][3] for h in hist]
    widths = [h["b"][2] for h in hist]
    cxs = [h["c"][0] for h in hist]
    cys = [h["c"][1] for h in hist]

    curr_h = max(heights[-1], 20.0)
    aspect_ratios = [w / max(h, 1.0) for w, h in zip(widths, heights)]

    aspect_curr = float(np.mean(aspect_ratios[-3:]))
    h_drop = (np.mean(heights[:3]) - np.mean(heights[-3:])) / max(np.mean(heights[:3]), 1.0)
    horiz_v = np.diff(cxs) / curr_h

    displacement = math.hypot(cxs[-1] - cxs[0], cys[-1] - cys[0]) / curr_h
    speed_mean = float(np.mean(np.abs(horiz_v))) if len(horiz_v) else 0.0
    speed_jitter = float(np.std(horiz_v)) if len(horiz_v) else 0.0

    return dict(
        aspect_curr=aspect_curr,
        h_drop=h_drop,
        displacement=displacement,
        speed_mean=speed_mean,
        speed_jitter=speed_jitter,
    )


def classify_taxonomy(f: dict, sensitivity: int):
    """Per-track classification. `sensitivity` widens/loosens every
    threshold (not just the reported confidence)."""
    s = sensitivity / 100.0

    # --- Heat exhaustion / sunstroke: very low net movement with a
    # persistent, moderate tremor/sway ---
    heat_jitter_lo = 0.010 - 0.005 * s
    heat_jitter_hi = 0.030 + 0.025 * s
    if (
        f["speed_mean"] < (0.010 + 0.006 * s)
        and heat_jitter_lo < f["speed_jitter"] < heat_jitter_hi
        and f["aspect_curr"] > 0.55
    ):
        return "sunstroke_heat_exhaustion", min(0.96, 0.72 + 0.22 * s)

    # --- Sudden fall / fall + seizure: rapid height collapse + widened
    # silhouette ---
    fall_h_drop_thresh = 0.26 * (1.15 - 0.35 * s)
    if f["aspect_curr"] > 1.0 and f["h_drop"] > fall_h_drop_thresh:
        if f["speed_jitter"] > (0.03 - 0.01 * s):
            return "sudden_fall_followed_by_seizure", min(0.98, 0.78 + 0.18 * s)
        return "sudden_fall", min(0.95, 0.76 + 0.20 * s)

    # --- Lying immobile: wide/flat silhouette, barely moving ---
    if f["aspect_curr"] > (1.20 - 0.15 * s) and f["displacement"] < (0.15 + 0.06 * s):
        return "lying_immobile", min(0.96, 0.78 + 0.18 * s)

    # --- Gait irregularities ---
    limp_hi = 0.095 - 0.025 * s
    limp_lo = 0.050 - 0.015 * s
    if f["speed_jitter"] > limp_hi:
        return "irregular_limping", min(0.92, 0.58 + f["speed_jitter"] * 2.6)
    if f["speed_jitter"] > limp_lo:
        return "regular_limping", min(0.88, 0.53 + f["speed_jitter"] * 2.6)

    # --- Exhaustion: slow, sagging gait ---
    if f["speed_mean"] < (0.020 + 0.010 * s) and f["h_drop"] > (0.10 - 0.03 * s):
        return "Exhausted_walking", min(0.85, 0.58 + 0.22 * s)

    return None, 0.0


def detect_fighting_pairs(track_features, sensitivity):
    """`track_features`: list of (track_id, centroid, height, features).
    Flags pairs of people who are close together AND both moving
    erratically at the same time — a proxy for a physical altercation.
    Returns the set of track ids involved in at least one such pair."""
    s = sensitivity / 100.0
    proximity_thresh = 1.5  # in units of average person height
    jitter_thresh = 0.05 - 0.02 * s

    flagged = set()
    for i in range(len(track_features)):
        tid1, c1, h1, feat1 = track_features[i]
        if feat1 is None or feat1["speed_jitter"] <= jitter_thresh:
            continue
        for j in range(i + 1, len(track_features)):
            tid2, c2, h2, feat2 = track_features[j]
            if feat2 is None or feat2["speed_jitter"] <= jitter_thresh:
                continue
            avg_h = max((h1 + h2) / 2.0, 20.0)
            dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1]) / avg_h
            if dist < proximity_thresh:
                flagged.add(tid1)
                flagged.add(tid2)
    return flagged


def confirm_frames_needed(sensitivity: int) -> int:
    """How many consecutive frames a track must hold the same
    classification before it's treated as a real alert (FIX NOTE 4).
    Higher sensitivity -> fewer frames needed (more responsive, a bit
    more false positives). Lower sensitivity -> more frames needed
    (slower, but much cleaner)."""
    s = sensitivity / 100.0
    return max(2, int(round(6 - 4 * s)))


# ============================================================================
# THREADED WORKER & ENGINE
# ============================================================================

def frame_producer(video_path, max_frames, frame_queue, target_fps, stop_event):
    """FIX NOTE 3: paced producer. Reads the source video's own fps and
    then sleeps between reads so frames are pushed into the queue at
    roughly `target_fps` (real-time-like), instead of dumping the whole
    clip into the queue as fast as cv2 can decode it. This is what
    actually makes playback look smooth/continuous rather than
    fast-forward-then-jump."""
    cap = cv2.VideoCapture(video_path)
    frame_interval = 1.0 / max(target_fps, 1)
    count = 0
    next_t = time.time()
    while cap.isOpened() and count < max_frames:
        if stop_event.is_set():
            break
        ret, frame = cap.read()
        if not ret:
            break
        count += 1

        now = time.time()
        sleep_for = next_t - now
        if sleep_for > 0:
            time.sleep(sleep_for)
        next_t += frame_interval

        try:
            frame_queue.put((count, frame), timeout=1.0)
        except queue.Full:
            # Consumer stalled hard (e.g. tab backgrounded) — drop this
            # frame rather than block the producer forever.
            pass
    cap.release()
    try:
        frame_queue.put((None, None), timeout=1.0)
    except queue.Full:
        pass


# ----------------------------------------------------------------------
# Person-shape refinement: OpenCV's built-in HOG pedestrian detector.
# Loaded once (no external model download / network access needed).
# We only run it on a small padded crop around an ALREADY-flagged track's
# motion bbox — not on every track/every frame — so the box gets fitted
# to the actual person without materially slowing down normal frames.
# ----------------------------------------------------------------------
try:
    _HOG = cv2.HOGDescriptor()
    _HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
except Exception:
    # Some cloud/headless OpenCV builds don't expose HOGDescriptor (or fail
    # to load its default SVM weights) depending on how the package was
    # built/installed. Degrade gracefully instead of crashing the whole
    # app — refine_person_box below just returns the original motion bbox
    # unchanged when this is None.
    _HOG = None


def refine_person_box(frame_bgr, bbox, pad=25):
    if _HOG is None:
        return bbox

    x, y, w, h = bbox
    H, W = frame_bgr.shape[:2]
    x0, y0 = max(int(x - pad), 0), max(int(y - pad), 0)
    x1, y1 = min(int(x + w + pad), W), min(int(y + h + pad), H)
    if x1 <= x0 or y1 <= y0:
        return bbox

    roi = frame_bgr[y0:y1, x0:x1]
    if roi.shape[0] < 24 or roi.shape[1] < 16:
        return bbox  # too small for HOG to say anything useful

    try:
        rects, weights = _HOG.detectMultiScale(
            roi, winStride=(6, 6), padding=(8, 8), scale=1.05
        )
    except Exception:
        return bbox

    if rects is None or len(rects) == 0:
        return bbox

    # pick the highest-confidence detection in the crop
    best = int(np.argmax(weights)) if len(weights) else 0
    rx, ry, rw, rh = rects[best]
    return (x0 + int(rx), y0 + int(ry), int(rw), int(rh))


def new_cv_state():
    return {
        # FIX NOTE 4: real background model (MOG2) instead of naive
        # two-frame differencing — far less sensitive to camera jitter,
        # lighting flicker and compression noise.
        "bg_sub": cv2.createBackgroundSubtractorMOG2(
            history=250, varThreshold=45, detectShadows=True
        ),
        "warmup": 0,
        "tracks": {},
        "next_id": 1,
        "global_cd": {},
    }


def process_video_frame(frame, frame_idx, state, sensitivity):
    canvas = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    fg_mask = state["bg_sub"].apply(gray)

    # Let the background model warm up before trusting it — the first
    # ~20 frames are mostly "everything is foreground" noise.
    state["warmup"] += 1
    if state["warmup"] < 20:
        return canvas, [], 0

    # MOG2 marks shadow pixels as mid-gray (127) when detectShadows=True.
    # Thresholding at 200 keeps only solid foreground and drops shadows,
    # which used to register as spurious extra motion blobs.
    _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # Opening removes small noise specks; closing re-merges a real
    # person's silhouette (limbs/folds) that the mask fragments into
    # several small blobs.
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for c in contours:
        # Keep the low area floor (catches distant/small people early),
        # relying on MOG2 + morphology above to keep noise out instead of
        # a high area cutoff.
        if cv2.contourArea(c) < 250:
            continue
        x, y, w, h = cv2.boundingRect(c)
        detections.append(((x + w / 2, y + h / 2), (x, y, w, h)))

    assigned = set()
    for (cx, cy), (x, y, w, h) in detections:
        best_id, best_d = None, 100.0
        for tid, tr in state["tracks"].items():
            if tid in assigned:
                continue
            d = math.hypot(tr.centroid[0] - cx, tr.centroid[1] - cy)
            if d < best_d:
                best_d, best_id = d, tid
        if best_id is not None:
            state["tracks"][best_id].update((cx, cy), (x, y, w, h), frame_idx)
            assigned.add(best_id)
        else:
            tid = state["next_id"]
            state["next_id"] += 1
            state["tracks"][tid] = Track(tid, (cx, cy), (x, y, w, h), frame_idx)
            assigned.add(tid)

    # Drop stale tracks
    for tid in [t for t, obj in state["tracks"].items() if frame_idx - obj.last_seen > 2]:
        del state["tracks"][tid]

    new_alerts = []
    confirm_needed = confirm_frames_needed(sensitivity)

    # --- Pass 1: extract motion features for every active track once ---
    track_feats = {}
    for tid, tr in state["tracks"].items():
        track_feats[tid] = extract_features(tr)

    # --- Pass 2: pairwise fighting check (needs every track's features
    # available up front, so it has to run before per-track classification) ---
    feats_list = [
        (tid, state["tracks"][tid].centroid, state["tracks"][tid].bbox[3], track_feats[tid])
        for tid in state["tracks"]
    ]
    fighting_ids = detect_fighting_pairs(feats_list, sensitivity)

    # --- Pass 3: classify, apply temporal confirmation, then draw ---
    for tid, tr in state["tracks"].items():
        f = track_feats[tid]
        if f is None:
            continue

        if tid in fighting_ids:
            cond, conf = "fighting", min(0.93, 0.62 + f["speed_jitter"] * 2.2)
        else:
            cond, conf = classify_taxonomy(f, sensitivity)

        # FIX NOTE 4: only act once the SAME condition has held for
        # `confirm_needed` consecutive frames — kills one-off noise spikes.
        confirmed_now = tr.confirm(cond, confirm_needed)

        if confirmed_now and cond in TAXONOMY_RULES:
            info = TAXONOMY_RULES[cond]
            label = f"ALERT: {info['en'].upper()} ({conf*100:.0f}%)"

            x, y, w, h = refine_person_box(canvas, tr.bbox)

            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 2)
            cv2.rectangle(canvas, (x, max(y - 20, 0)), (x + tw + 6, max(y, 20)), (0, 0, 255), -1)
            cv2.putText(canvas, label, (x + 3, max(y - 5, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            last_f = state["global_cd"].get(cond, -9999)
            if frame_idx - last_f > 60:
                state["global_cd"][cond] = frame_idx
                new_alerts.append((cond, conf))
        elif tr.pending_cond in TAXONOMY_RULES and tr.pending_count >= confirm_needed:
            # Already-confirmed, ongoing condition on a later frame — keep
            # the box visible without re-logging a duplicate alert.
            info = TAXONOMY_RULES[tr.pending_cond]
            x, y, w, h = refine_person_box(canvas, tr.bbox)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(canvas, info["en"].upper(), (x + 3, max(y - 5, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas, new_alerts, len(state["tracks"])


# ============================================================================
# STREAMLIT UI & FRAGMENT-BASED CONSUMER RUNTIME
# ============================================================================

RENDER_FPS = 12  # smooth, sustainable push rate for the browser/websocket

if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "metrics" not in st.session_state:
    st.session_state.metrics = {"frame": 0, "tracks": 0, "fps": 0.0, "time": 0.0}
if "streaming" not in st.session_state:
    st.session_state.streaming = False
if "engine" not in st.session_state:
    st.session_state.engine = None  # holds queue/thread/cv-state for the active run
if "last_frame_b64" not in st.session_state:
    st.session_state.last_frame_b64 = None  # cached last-painted frame as a data-URI, avoids MediaFileManager races

st.markdown(
    """
    <div class="header-box">
        <div>
            <div class="system-title">🚑 BASEER | AI Anomaly Detection Command Center</div>
            <div class="system-sub">Early Multi-Modal Medical & Safety Anomaly Detection</div>
        </div>
        <div class="live-badge">● LIVE DISPATCH SYSTEM</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🎛️ Command Center Controls")

    uploaded_vid = st.file_uploader("Upload Surveillance Clip (.mp4)", type=["mp4", "avi", "mov"])

    st.markdown("---")
    selected_zone = st.selectbox("Zone Location", LOCATIONS)
    sens = st.slider("Detection Sensitivity", 20, 100, 75,
                      help="Higher = alerts confirm faster (fewer consecutive frames required) but slightly more false positives. Lower = slower but cleaner.")

    st.markdown("---")
    play_speed = st.slider("Playback Speed (FPS)", 5, RENDER_FPS, RENDER_FPS,
                            help=f"Real-time playback pace, capped at {RENDER_FPS} FPS — the max rate the browser can paint smoothly. Frames are now paced to this speed as they're produced, so playback stays continuous instead of jumping ahead.")
    max_f = st.slider("Max Processing Frames", 100, 1500, 600, step=50)

    st.markdown("---")
    col1, col2 = st.columns(2)
    start_btn = col1.button("▶ Run Stream", use_container_width=True, type="primary")
    reset_btn = col2.button("⟲ Reset", use_container_width=True)

    if reset_btn:
        # Also tear down any active engine/thread state, not just alerts.
        eng = st.session_state.engine
        if eng is not None:
            stop_evt = eng.get("stop_event")
            if stop_evt is not None:
                stop_evt.set()
            if eng.get("tfile_path") and os.path.exists(eng["tfile_path"]):
                try:
                    os.remove(eng["tfile_path"])
                except Exception:
                    pass
        st.session_state.alerts = []
        st.session_state.metrics = {"frame": 0, "tracks": 0, "fps": 0.0, "time": 0.0}
        st.session_state.streaming = False
        st.session_state.engine = None
        st.session_state.last_frame_b64 = None
        st.rerun()

def draw_kpi_html(m):
    return f"""
    <div class="kpi-container">
        <div class="kpi-card"><div class="kpi-num">{m['frame']}</div><div class="kpi-title">Frame</div></div>
        <div class="kpi-card"><div class="kpi-num">{m['time']:.1f}s</div><div class="kpi-title">Time</div></div>
        <div class="kpi-card"><div class="kpi-num">{m['tracks']}</div><div class="kpi-title">Active</div></div>
        <div class="kpi-card"><div class="kpi-num">{m['fps']:.1f}</div><div class="kpi-title">FPS</div></div>
        <div class="kpi-card"><div class="kpi-num" style="color:#EF4444">{len(st.session_state.alerts)}</div><div class="kpi-title">Alerts</div></div>
    </div>
    """


def draw_triage_html():
    if not st.session_state.alerts:
        return "<div style='color:#94A3B8; padding:1rem; border:1px dashed #334155; border-radius:8px;'>No critical anomalies detected yet. Monitoring live feed...</div>"

    html_out = ""
    for idx, a in enumerate(reversed(st.session_state.alerts)):
        info = TAXONOMY_RULES.get(a.condition_key, TAXONOMY_RULES["sudden_fall"])
        b_color = PRIORITY_COLOR[info["priority"]]

        html_out += f"""
        <div class="alert-card" style="border-left: 6px solid {b_color};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="triage-badge" style="background:{b_color}">{info['priority']} PRIORITY</span>
                <span class="card-meta">#{a.id} · {a.wall_clock} · t={a.video_time_s:.1f}s</span>
            </div>
            <div style="margin-top:0.3rem;"><span class="category-tag">📂 {info['category']}</span></div>
            <div class="card-title">{info['icon']} {info['title']}</div>
            <div class="card-en"><b>Class:</b> <code>{info['en']}</code> (Conf: {a.confidence*100:.0f}%)</div>
            <div class="card-meta">📍 {a.location}</div>
            <div style="margin-top:0.4rem; font-size:0.8rem; color:#CBD5E1;"><b>Recommended Action:</b> {info['action']}</div>
        </div>
        """
    return html_out


def start_stream():
    """One-shot setup: save the upload, spin up the (now paced) producer
    thread, and arm the engine. Actual frame-by-frame rendering happens
    in the live_feed fragment below, one tick at a time — never in a
    blocking loop here."""
    if uploaded_vid is None:
        st.warning("Please upload a video file to run the analytical stream.")
        return

    st.session_state.alerts = []
    st.session_state.metrics = {"frame": 0, "tracks": 0, "fps": 0.0, "time": 0.0}

    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tf.write(uploaded_vid.read())
    tfile_path = tf.name
    tf.close()

    # FIX NOTE 3: cap the requested playback speed at RENDER_FPS so the
    # producer can never outpace what the UI paints — this is what keeps
    # the video looking like a continuous live feed instead of
    # decode-everything-then-fast-forward.
    target_fps = min(play_speed, RENDER_FPS)

    frame_queue = queue.Queue(maxsize=8)
    stop_event = threading.Event()
    prod_thread = threading.Thread(
        target=frame_producer,
        args=(tfile_path, max_f, frame_queue, target_fps, stop_event),
        daemon=True,
    )
    prod_thread.start()

    st.session_state.engine = {
        "queue": frame_queue,
        "stop_event": stop_event,
        "cv_state": new_cv_state(),
        "tfile_path": tfile_path,
        "w": 480,
        "h": 270,
        "fps_src": float(target_fps),
        "sens": sens,
        "zone": selected_zone,
        "proc": 0,
        "start_t": time.time(),
    }
    st.session_state.streaming = True


@st.fragment(run_every=1.0 / RENDER_FPS)
def live_feed():
    """Runs on its own timer, independent of the rest of the app.

    IMPORTANT: a fragment cannot write into placeholders/columns that were
    created outside of it (Streamlit raises
    'Fragments cannot write to elements outside of their container').
    So this fragment builds its ENTIRE layout (both columns) itself, every
    tick. That's the intended fragment pattern: each tick fully re-renders
    just this subtree, which is what makes every frame a real, individually
    flushed update instead of one being coalesced away.

    FIX NOTE 3: because the producer now paces itself to RENDER_FPS, each
    tick normally finds exactly ONE fresh frame waiting. We still drain up
    to a small cap as a safety net for transient stalls (e.g. one slow
    detection pass), but it's no longer the primary mechanism — so the
    picture on screen advances through the video in true order instead of
    jumping across a backlog.
    """
    eng = st.session_state.engine

    if st.session_state.streaming and eng is not None:
        last_canvas = None
        last_frame_idx = None
        last_tracks = 0
        finished = False
        drained = 0
        MAX_DRAIN = 3  # safety net only, not the pacing mechanism anymore

        while drained < MAX_DRAIN:
            try:
                frame_idx, raw = eng["queue"].get_nowait()
            except queue.Empty:
                break
            drained += 1

            if frame_idx is None:
                finished = True
                break

            raw = cv2.resize(raw, (eng["w"], eng["h"]))
            frame_bgr, evts, tracks = process_video_frame(raw, frame_idx, eng["cv_state"], eng["sens"])

            for cond, conf in evts:
                seq_num = len(st.session_state.alerts) + 1
                st.session_state.alerts.append(
                    Alert(
                        id=f"EMS-{seq_num:03d}",
                        unique_key=f"{seq_num}_{frame_idx}_{int(time.time()*1000)}",
                        frame_idx=frame_idx,
                        video_time_s=frame_idx / eng["fps_src"],
                        wall_clock=datetime.now().strftime("%H:%M:%S"),
                        location=eng["zone"],
                        condition_key=cond,
                        confidence=conf,
                    )
                )

            last_canvas = frame_bgr
            last_frame_idx = frame_idx
            last_tracks = tracks

        if finished:
            st.session_state.streaming = False
            if eng["tfile_path"] and os.path.exists(eng["tfile_path"]):
                try:
                    os.remove(eng["tfile_path"])
                except Exception:
                    pass

        if last_canvas is not None:
            ok, jpg_buf = cv2.imencode(".jpg", last_canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            if ok:
                st.session_state.last_frame_b64 = base64.b64encode(jpg_buf.tobytes()).decode("utf-8")

            eng["proc"] += 1
            elapsed = max(time.time() - eng["start_t"], 1e-6)
            st.session_state.metrics = {
                "frame": last_frame_idx,
                "tracks": last_tracks,
                "fps": eng["proc"] / elapsed,
                "time": last_frame_idx / eng["fps_src"],
            }

    # --- Redraw the full layout every tick, from cached state ---
    col_cam, col_triage = st.columns([1.35, 1])

    with col_cam:
        st.markdown("##### 📹 Analytical Feed (Continuous Stream)")
        if st.session_state.last_frame_b64 is not None:
            st.markdown(
                f'<img src="data:image/jpeg;base64,{st.session_state.last_frame_b64}" '
                f'style="width:100%;border-radius:10px;display:block;" />',
                unsafe_allow_html=True,
            )
        else:
            st.info("Upload a clip and press ▶ Run Stream to begin.")
        st.markdown(draw_kpi_html(st.session_state.metrics), unsafe_allow_html=True)

    with col_triage:
        st.markdown("##### 🚨 Live Triage & Dispatch Log")
        st.markdown(draw_triage_html(), unsafe_allow_html=True)


if start_btn:
    start_stream()

# Always mount the fragment; it self-schedules via run_every and simply
# redraws from cached state when idle (streaming == False).
live_feed()
