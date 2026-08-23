"""
Baseer – AI Early Multi-Modal Anomaly Detection & Triage Command Center
Parallel Threaded Streaming Pipeline & Dynamic Real-Time Bounding Box Removal
"""

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
# PAGE CONFIG & SYSTEM THEME
# ============================================================================

st.set_page_config(
    page_title="Baseer | AI Anomaly Detection & Triage Platform",
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
# ENGLISH TAXONOMY
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
    "slow_fall": {
        "category": "Falls & Medical Emergencies",
        "title": "Gradual / Slow Fall Incident",
        "en": "slow_fall",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "⬇️",
        "action": "Check vital signs and transport patient.",
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
        "title": "Regular Limping Gait Detected",
        "en": "regular_limping",
        "priority": "Medium",
        "color": "#F59E0B",
        "icon": "🚶",
        "action": "Dispatch mobile wheelchair transport.",
    },
    "irregular_limping": {
        "category": "Abnormal Gait & Fatigue",
        "title": "Irregular / Asymmetric Limping Gait",
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
        self.history = deque(maxlen=20)
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
    if len(hist) < 4:
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
    s = sensitivity / 100.0

    if f["speed_mean"] < 0.012 and 0.015 < f["speed_jitter"] < 0.035 and f["aspect_curr"] > 0.65:
        return "sunstroke_heat_exhaustion", min(0.96, 0.75 + 0.2 * s)

    if (f["aspect_curr"] > 1.1 and f["h_drop"] > 0.28 * (1.1 - 0.3 * s)):
        if f["speed_jitter"] > 0.035:
            return "sudden_fall_followed_by_seizure", min(0.98, 0.80 + 0.15 * s)
        return "sudden_fall", min(0.95, 0.78 + 0.18 * s)

    if f["aspect_curr"] > 1.15 and f["displacement"] < 0.15:
        return "lying_immobile", min(0.96, 0.80 + 0.15 * s)

    if f["speed_jitter"] > 0.05:
        if f["speed_jitter"] > 0.09:
            return "irregular_limping", min(0.90, 0.60 + f["speed_jitter"] * 2.5)
        return "regular_limping", min(0.88, 0.55 + f["speed_jitter"] * 2.5)

    if f["speed_mean"] < 0.025 and f["h_drop"] > 0.1:
        return "Exhausted_walking", min(0.85, 0.60 + 0.2 * s)

    return None, 0.0


# ============================================================================
# PARALLEL PROCESSING THREAD PIPELINE
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


def new_state():
    return {
        "prev_gray": None,
        "tracks": {},
        "next_id": 1,
        "global_cd": {},
    }


def process_video_frame(frame, frame_idx, state, sensitivity):
    canvas = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)

    if state["prev_gray"] is None:
        state["prev_gray"] = gray
        return canvas, [], 0

    frame_diff = cv2.absdiff(state["prev_gray"], gray)
    state["prev_gray"] = gray

    _, thresh = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for c in contours:
        if cv2.contourArea(c) < 600:
            continue
        x, y, w, h = cv2.boundingRect(c)
        detections.append(((x + w / 2, y + h / 2), (x, y, w, h)))

    assigned = set()
    for (cx, cy), (x, y, w, h) in detections:
        best_id, best_d = None, 120.0
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

    # حرق المسارات المتوقفة أو الخارجة فم فوراً لتفريغ الشاشة
    for tid in [t for t, obj in state["tracks"].items() if frame_idx - obj.last_seen > 5]:
        del state["tracks"][tid]

    new_alerts = []

    for tid, tr in state["tracks"].items():
        x, y, w, h = tr.bbox
        f = extract_features(tr)

        if f:
            cond, conf = classify_taxonomy(f, sensitivity)
            if cond and cond in TAXONOMY_RULES:
                info = TAXONOMY_RULES[cond]
                label = f"ALERT: {info['en'].upper()} ({conf*100:.0f}%)"

                # 🟢 رسم المربع الأحمر بشكل مؤقت وحين وجود العَرَض فقط
                cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 2)
                cv2.rectangle(canvas, (x, max(y - 20, 0)), (x + tw + 6, max(y, 20)), (0, 0, 255), -1)
                cv2.putText(canvas, label, (x + 3, max(y - 5, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

                last_f = state["global_cd"].get(cond, -9999)
                if frame_idx - last_f > 75:
                    state["global_cd"][cond] = frame_idx
                    new_alerts.append((cond, conf))

    return canvas, new_alerts, len(state["tracks"])


# ============================================================================
# STREAMLIT UI & CONSUMER RUNTIME
# ============================================================================

if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "metrics" not in st.session_state:
    st.session_state.metrics = {"frame": 0, "tracks": 0, "fps": 0.0, "time": 0.0}

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
    sens = st.slider("Detection Sensitivity", 20, 100, 70)

    st.markdown("---")
    play_speed = st.slider("Playback FPS Speed", 15, 30, 24)
    max_f = st.slider("Max Processing Frames", 100, 1500, 500, step=50)

    st.markdown("---")
    col1, col2 = st.columns(2)
    start_btn = col1.button("▶ Run Stream", use_container_width=True, type="primary")
    reset_btn = col2.button("⟲ Reset", use_container_width=True)

    if reset_btn:
        st.session_state.alerts = []
        st.session_state.metrics = {"frame": 0, "tracks": 0, "fps": 0.0, "time": 0.0}
        st.rerun()

col_cam, col_triage = st.columns([1.35, 1])

with col_cam:
    st.markdown("##### 📹 Analytical Feed (Continuous Stream)")
    cam_holder = st.empty()
    kpi_holder = st.empty()

with col_triage:
    st.markdown("##### 🚨 Live Triage & Dispatch Log")
    triage_holder = st.empty()


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


def run_detection():
    if uploaded_vid is None:
        st.warning("Please upload a video file to run the analytical stream.")
        return

    st.session_state.alerts = []
    state = new_state()

    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tf.write(uploaded_vid.read())
    tfile_path = tf.name
    tf.close()

    frame_queue = queue.Queue(maxsize=30)
    prod_thread = threading.Thread(target=frame_producer, args=(tfile_path, max_f, frame_queue), daemon=True)
    prod_thread.start()

    w, h = 640, 360
    fps_src = 25.0
    start_t = time.time()
    proc = 0
    target_delay = 1.0 / play_speed

    while True:
        loop_start = time.time()
        frame_idx, raw = frame_queue.get()

        if frame_idx is None:
            break

        raw = cv2.resize(raw, (w, h))
        frame_bgr, evts, tracks = process_video_frame(raw, frame_idx, state, sens)

        for cond, conf in evts:
            seq_num = len(st.session_state.alerts) + 1
            st.session_state.alerts.append(
                Alert(
                    id=f"EMS-{seq_num:03d}",
                    unique_key=f"{seq_num}_{frame_idx}_{int(time.time()*1000)}",
                    frame_idx=frame_idx,
                    video_time_s=frame_idx / fps_src,
                    wall_clock=datetime.now().strftime("%H:%M:%S"),
                    location=selected_zone,
                    condition_key=cond,
                    confidence=conf,
                )
            )

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        proc += 1
        elapsed = max(time.time() - start_t, 1e-6)

        st.session_state.metrics = {
            "frame": frame_idx,
            "tracks": tracks,
            "fps": proc / elapsed,
            "time": frame_idx / fps_src,
        }

        cam_holder.image(rgb, channels="RGB", use_container_width=True)
        kpi_holder.markdown(draw_kpi_html(st.session_state.metrics), unsafe_allow_html=True)
        triage_holder.markdown(draw_triage_html(), unsafe_allow_html=True)

        compute_duration = time.time() - loop_start
        sleep_time = target_delay - compute_duration
        if sleep_time > 0:
            time.sleep(sleep_time)

    if os.path.exists(tfile_path):
        try:
            os.remove(tfile_path)
        except Exception:
            pass


if start_btn:
    run_detection()
else:
    kpi_holder.markdown(draw_kpi_html(st.session_state.metrics), unsafe_allow_html=True)
    triage_holder.markdown(draw_triage_html(), unsafe_allow_html=True)
