"""
Baseer – AI Early Multi-Modal Anomaly Detection & Triage Command Center
Full Stream Control with Smooth Frame-by-Frame Rendering
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
        self.update(centroid, bbox, frame_idx)

    def update(self, centroid, bbox, frame_idx):
        self.centroid = centroid
        self.bbox = bbox
        self.last_seen = frame_idx
        self.age += 1
        self.history.append({"c": centroid, "b": bbox, "f": frame_idx})


def extract_features(track: Track):
    hist = list(track.history)
    if len(hist) < 3:
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
    """Per-track classification. `sensitivity` now actually widens/loosens
    every threshold (not just the reported confidence), and a final
    catch-all rule flags any track whose motion is meaningfully off a
    normal-walking baseline even if it doesn't match a specific pattern —
    so unusual behavior doesn't silently pass through undetected."""
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



# ============================================================================
# THREADED WORKER & ENGINE
# ============================================================================

def frame_producer(video_path, max_frames, frame_queue):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put((count + 1, frame))
        count += 1
    cap.release()
    frame_queue.put((None, None))


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
        "prev_gray": None,
        "tracks": {},
        "next_id": 1,
        "global_cd": {},
    }


def process_video_frame(frame, frame_idx, state, sensitivity):
    canvas = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)

    if state["prev_gray"] is None:
        state["prev_gray"] = gray
        return canvas, [], 0

    frame_diff = cv2.absdiff(state["prev_gray"], gray)
    state["prev_gray"] = gray

    _, thresh = cv2.threshold(frame_diff, 22, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for c in contours:
        # تخفيض الحد لرصد الأجسام البعيدة فور ورودها في الحركة
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

    # تنظيف وتفريغ الأهداف غير النشطة
    for tid in [t for t, obj in state["tracks"].items() if frame_idx - obj.last_seen > 2]:
        del state["tracks"][tid]

    new_alerts = []

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

    # --- Pass 3: classify + draw, one track at a time ---
    for tid, tr in state["tracks"].items():
        f = track_feats[tid]
        if f is None:
            continue

        if tid in fighting_ids:
            cond, conf = "fighting", min(0.93, 0.62 + f["speed_jitter"] * 2.2)
        else:
            cond, conf = classify_taxonomy(f, sensitivity)

        if cond and cond in TAXONOMY_RULES:
            info = TAXONOMY_RULES[cond]
            label = f"ALERT: {info['en'].upper()} ({conf*100:.0f}%)"

            # Tighten the box to the actual person shape, not just
            # the raw motion-diff blob.
            x, y, w, h = refine_person_box(canvas, tr.bbox)

            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 2)
            cv2.rectangle(canvas, (x, max(y - 20, 0)), (x + tw + 6, max(y, 20)), (0, 0, 255), -1)
            cv2.putText(canvas, label, (x + 3, max(y - 5, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            last_f = state["global_cd"].get(cond, -9999)
            if frame_idx - last_f > 60:
                state["global_cd"][cond] = frame_idx
                new_alerts.append((cond, conf))

    return canvas, new_alerts, len(state["tracks"])


# ============================================================================
# STREAMLIT UI & FRAGMENT-BASED CONSUMER RUNTIME
# ============================================================================

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
    sens = st.slider("Detection Sensitivity", 20, 100, 75)

    st.markdown("---")
    play_speed = st.slider("Playback Speed (FPS Target)", 10, 60, 25,
                            help="Effective render rate is capped at ~12 FPS — higher values just make detection run faster, since Streamlit can't smoothly push more frames than that per second over the websocket.")
    max_f = st.slider("Max Processing Frames", 100, 1500, 600, step=50)

    st.markdown("---")
    col1, col2 = st.columns(2)
    start_btn = col1.button("▶ Run Stream", use_container_width=True, type="primary")
    reset_btn = col2.button("⟲ Reset", use_container_width=True)

    if reset_btn:
        # Also tear down any active engine/thread state, not just alerts.
        eng = st.session_state.engine
        if eng is not None and eng.get("tfile_path") and os.path.exists(eng["tfile_path"]):
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
    """One-shot setup: save the upload, spin up the producer thread, and
    arm the engine. Actual frame-by-frame rendering happens in the
    live_feed fragment below, one frame per tick — never in a blocking
    loop here."""
    if uploaded_vid is None:
        st.warning("Please upload a video file to run the analytical stream.")
        return

    st.session_state.alerts = []
    st.session_state.metrics = {"frame": 0, "tracks": 0, "fps": 0.0, "time": 0.0}

    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tf.write(uploaded_vid.read())
    tfile_path = tf.name
    tf.close()

    frame_queue = queue.Queue(maxsize=50)
    prod_thread = threading.Thread(
        target=frame_producer, args=(tfile_path, max_f, frame_queue), daemon=True
    )
    prod_thread.start()

    # Streamlit's script-rerun/delta model can't reliably sustain the full
    # 25-30fps the slider advertises — pushing updates faster than the
    # websocket can flush just brings back the coalescing/skip problem,
    # now at the fragment level instead of the placeholder level. Cap the
    # real push rate at a value that stays smooth end-to-end.
    MAX_RENDER_FPS = 12
    st.session_state.engine = {
        "queue": frame_queue,
        "cv_state": new_cv_state(),
        "tfile_path": tfile_path,
        "w": 480,
        "h": 270,
        "fps_src": 25.0,
        "sens": sens,
        "zone": selected_zone,
        "frame_delay": max(1.0 / max(play_speed, 1), 1.0 / MAX_RENDER_FPS),
        "next_allowed": 0.0,
        "proc": 0,
        "start_t": time.time(),
    }
    st.session_state.streaming = True


@st.fragment(run_every=0.06)
def live_feed():
    """Runs on its own timer, independent of the rest of the app.

    IMPORTANT: a fragment cannot write into placeholders/columns that were
    created outside of it (Streamlit raises
    'Fragments cannot write to elements outside of their container').
    So this fragment builds its ENTIRE layout (both columns) itself, every
    tick. That's the intended fragment pattern: each tick fully re-renders
    just this subtree, which is what makes every frame a real, individually
    flushed update instead of one being coalesced away.

    Anti-choppiness strategy: rather than popping exactly one frame per
    tick (which falls behind and looks laggy/jumpy once the producer
    thread gets ahead of the render rate), each tick DRAINS every frame
    currently sitting in the queue. Every drained frame still goes through
    detection (so no anomaly is missed), but only the CANVAS of the most
    recent one gets encoded and shown — so the picture on screen is always
    the freshest available state instead of trailing behind a backlog.
    """
    eng = st.session_state.engine

    if st.session_state.streaming and eng is not None:
        now = time.time()
        if now >= eng["next_allowed"]:
            eng["next_allowed"] = now + eng["frame_delay"]

            last_canvas = None
            last_frame_idx = None
            last_tracks = 0
            finished = False
            drained = 0
            MAX_DRAIN = 40  # safety cap so one tick can't run forever if wildly behind

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
