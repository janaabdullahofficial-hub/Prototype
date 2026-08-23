"""
Early Medical Emergency Detection System (نظام الرصد والفرز الإسعافي المبكر)
=============================================================================
- Perspective-normalized kinematic feature extraction
- Global alert cooldown & debouncing (no duplicate spam)
- Crash-free dynamic alert cards rendering
- Bilingual (Arabic / English) command-center UI theme
"""

import math
import os
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import streamlit as st

# ============================================================================
# PAGE CONFIG & MODERN MEDICAL OPS THEME
# ============================================================================

st.set_page_config(
    page_title="نظام الكشف الإسعافي المبكر | Medical Emergency AI",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Tajawal', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container { padding-top: 1.4rem; max-width: 1400px; }
    
    .system-title { 
        font-size: 1.85rem; 
        font-weight: 800; 
        margin-bottom: 0.1rem;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .system-subtitle { color: #94A3B8; font-size: 0.95rem; margin-top: 0; margin-bottom: 1rem; }
    
    .status-strip {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
        gap: 0.5rem;
        margin: 0.6rem 0;
    }
    .kpi-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 0.45rem 0.6rem;
        text-align: center;
    }
    .kpi-val { font-size: 1.15rem; font-weight: 700; color: #F8FAFC; font-family: 'JetBrains Mono', monospace; }
    .kpi-lbl { font-size: 0.72rem; color: #64748B; font-weight: 600; text-transform: uppercase; }

    .alert-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-left: 5px solid #475569;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        color: white;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
    }
    .cond-title-ar { font-size: 1.05rem; font-weight: 700; color: #F1F5F9; margin-top: 0.3rem; }
    .cond-title-en { font-size: 0.82rem; color: #94A3B8; margin-bottom: 0.3rem; }
    .meta-line { color: #64748B; font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; }
    .dispatched-tag {
        color: #10B981; font-weight: 700; font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# MEDICAL TRIAGE RULE ENGINE (BILINGUAL)
# ============================================================================

TRIAGE_RULES = {
    "severe_gait_limping": {
        "ar": "إجهاد حراري / بوادر جفاف حاد",
        "en": "Heatstroke / Gait Instability",
        "priority": "Medium",
        "color": "#F59E0B",
        "icon": "🚶",
    },
    "stooped_walking_resting": {
        "ar": "اشتباه إغماء / هبوط ضغط مفاجئ",
        "en": "Syncope / Sudden Blood Pressure Drop",
        "priority": "High",
        "color": "#F97316",
        "icon": "🧍",
    },
    "severe_choking_on_ground": {
        "ar": "انسداد مجرى التنفس / ضائقة تنفسية",
        "en": "Acute Airway Obstruction / Distress",
        "priority": "Critical",
        "color": "#EF4444",
        "icon": "🫁",
    },
    "seizure_convulsion": {
        "ar": "حالة تشنج عصبي / نوبة صرع نشطة",
        "en": "Active Seizure / Convulsion",
        "priority": "Critical",
        "color": "#EF4444",
        "icon": "⚡",
    },
    "sudden_fall": {
        "ar": "سقوط مفاجئ / فقدان فوري للوعي",
        "en": "Sudden Fall / Immediate Unconsciousness",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "🚨",
    },
}

PRIORITY_RANK = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
PRIORITY_COLOR = {"Critical": "#DC2626", "High": "#F97316", "Medium": "#F59E0B", "Low": "#3B82F6"}

LOCATIONS = [
    "بوابة 01 – المدخل الرئيسي (Gate 01)",
    "كاميرا 07 – الساحة المركزية (Main Courtyard)",
    "كاميرا 12 – الممشى والممر الرئيسي (Main Corridor)",
    "محطة النقل 03 (Transport Hub 03)",
    "مخرج الطوارئ الشمالي (Emergency Exit North)",
    "موقع مخصص... (Custom)",
]

# ============================================================================
# DATA STRUCTURES & SMOOTH TRACKING
# ============================================================================

@dataclass
class Alert:
    id: str
    frame_idx: int
    video_time_s: float
    wall_clock: str
    location: str
    condition_key: str
    confidence: float
    dispatched: bool = False
    dispatched_time: Optional[str] = None


class Track:
    def __init__(self, track_id, centroid, bbox, frame_idx, maxlen=35):
        self.id = track_id
        self.history = deque(maxlen=maxlen)
        self.update(centroid, bbox, frame_idx)

    def update(self, centroid, bbox, frame_idx):
        self.centroid = centroid
        self.bbox = bbox
        self.last_seen = frame_idx
        self.history.append({"c": centroid, "b": bbox, "f": frame_idx})


# ============================================================================
# PERSPECTIVE-NORMALIZED FEATURE EXTRACTION & CLASSIFICATION
# ============================================================================

def extract_features(track: Track):
    hist = list(track.history)
    if len(hist) < 6:
        return None

    heights = [h["b"][3] for h in hist]
    widths = [h["b"][2] for h in hist]
    cxs = [h["c"][0] for h in hist]
    cys = [h["c"][1] for h in hist]

    current_h = max(heights[-1], 10.0)
    aspect_ratios = [w / max(hh, 1.0) for w, hh in zip(widths, heights)]

    height_now = float(np.mean(heights[-3:]))
    height_early = float(np.mean(heights[:3]))
    height_drop_ratio = (height_early - height_now) / max(height_early, 1e-3)

    aspect_now = float(np.mean(aspect_ratios[-3:]))
    aspect_early = float(np.mean(aspect_ratios[:3]))

    vert_v_norm = np.diff(cys) / current_h
    horiz_v_norm = np.diff(cxs) / current_h

    net_displacement_norm = math.hypot(cxs[-1] - cxs[0], cys[-1] - cys[0]) / current_h
    total_path_norm = sum(
        math.hypot(cxs[i + 1] - cxs[i], cys[i + 1] - cys[i]) for i in range(len(cxs) - 1)
    ) / current_h + 1e-3

    horiz_speed_std = float(np.std(horiz_v_norm)) if len(horiz_v_norm) else 0.0
    horiz_speed_mean = float(np.mean(np.abs(horiz_v_norm))) if len(horiz_v_norm) else 0.0

    return dict(
        height_drop_ratio=height_drop_ratio,
        aspect_now=aspect_now,
        aspect_early=aspect_early,
        net_displacement_norm=net_displacement_norm,
        total_path_norm=total_path_norm,
        horiz_speed_std=horiz_speed_std,
        horiz_speed_mean=horiz_speed_mean,
        max_vert_velocity_norm=float(np.max(np.abs(vert_v_norm))) if len(vert_v_norm) else 0.0,
    )


def classify(features: dict, sensitivity: int):
    f = features
    s = sensitivity / 100.0
    lenient = 1.0 - 0.4 * s

    # 1. Sudden fall
    if (f["aspect_now"] > 1.1 and f["height_drop_ratio"] > 0.38 * lenient) or (
        f["aspect_early"] < 0.9 and f["aspect_now"] > 1.15 and f["max_vert_velocity_norm"] > 0.06 * lenient
    ):
        conf = min(0.98, 0.65 + 0.2 * s)
        return "sudden_fall", conf

    # 2. Prone on ground
    prone = f["aspect_now"] > 1.15
    if prone and f["net_displacement_norm"] < 0.25 * lenient:
        conf = min(0.92, 0.55 + 0.2 * s)
        return "severe_choking_on_ground", conf

    # 3. Stooped posture
    if (
        0.18 * lenient < f["height_drop_ratio"] <= 0.38
        and f["horiz_speed_mean"] < 0.05 * lenient
        and f["aspect_now"] < 1.1
    ):
        conf = min(0.88, 0.45 + f["height_drop_ratio"] * 0.8)
        return "stooped_walking_resting", conf

    # 4. Severe gait instability
    path_eff = f["net_displacement_norm"] / f["total_path_norm"]
    if (
        not prone
        and f["horiz_speed_mean"] > 0.03
        and f["horiz_speed_std"] > 0.045 * lenient
        and path_eff < 0.65
    ):
        conf = min(0.85, 0.40 + f["horiz_speed_std"] * 5.0)
        return "severe_gait_limping", conf

    return None, 0.0


# ============================================================================
# REAL VIDEO OPENCV PIPELINE
# ============================================================================

def new_detector_state():
    bgsub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=35, detectShadows=True)
    return {"bgsub": bgsub, "tracks": {}, "next_id": 1, "global_cooldown": {}}


def process_real_frame(frame, frame_idx, state, sensitivity, min_area=800, cooldown_frames=120):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = state["bgsub"].apply(frame)
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        detections.append(((x + w / 2, y + h / 2), (x, y, w, h)))

    assigned = set()
    for (cx, cy), (x, y, w, h) in detections:
        best_id, best_dist = None, 100.0
        for tid, tr in state["tracks"].items():
            if tid in assigned:
                continue
            d = math.hypot(tr.centroid[0] - cx, tr.centroid[1] - cy)
            if d < best_dist:
                best_dist, best_id = d, tid
        if best_id is not None:
            state["tracks"][best_id].update((cx, cy), (x, y, w, h), frame_idx)
            assigned.add(best_id)
        else:
            tid = state["next_id"]
            state["next_id"] += 1
            state["tracks"][tid] = Track(tid, (cx, cy), (x, y, w, h), frame_idx)
            assigned.add(tid)

    stale = [tid for tid, tr in state["tracks"].items() if frame_idx - tr.last_seen > 18]
    for tid in stale:
        del state["tracks"][tid]

    annotated = frame.copy()
    new_events = []
    active_tracks = 0

    for tid, tr in state["tracks"].items():
        if tr.last_seen != frame_idx:
            continue
        active_tracks += 1
        x, y, w, h = tr.bbox
        feats = extract_features(tr)
        box_color, label = (50, 200, 100), f"ID {tid} · Normal"

        if feats:
            cond, conf = classify(feats, sensitivity)
            if cond:
                box_color = (40, 40, 235)
                info = TRIAGE_RULES[cond]
                label = f"{info['icon']} {info['en']}"
                
                last_fired = state["global_cooldown"].get(cond, -10_000)
                if frame_idx - last_fired > cooldown_frames:
                    state["global_cooldown"][cond] = frame_idx
                    new_events.append((tid, cond, conf))
            elif feats["horiz_speed_std"] > 0.03:
                box_color = (0, 200, 255)
                label = f"ID {tid} · Monitoring"

        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(
            annotated, label, (x, max(y - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2, cv2.LINE_AA,
        )

    return annotated, new_events, active_tracks


# ============================================================================
# SYNTHETIC SIMULATION PIPELINE
# ============================================================================

SIM_PHASES = ["normal_walk", "severe_limp", "stooped_resting", "sudden_fall", "severe_choking_on_ground"]
SIM_PHASE_LEN = 65


def process_sim_frame(frame_idx, state, sensitivity, width, height, rng, cooldown_frames=90):
    cycle_idx = frame_idx % (SIM_PHASE_LEN * len(SIM_PHASES))
    phase = SIM_PHASES[cycle_idx // SIM_PHASE_LEN]
    t = (cycle_idx % SIM_PHASE_LEN) / SIM_PHASE_LEN
    ground_y = height - 70

    if phase == "normal_walk":
        w, h = 46, 130
        cx, cy = width * 0.2 + t * width * 0.4, ground_y - h / 2
    elif phase == "severe_limp":
        w, h = 50, 128
        cx, cy = width * 0.6 + t * 40 + math.sin(t * 24) * 16, ground_y - h / 2
    elif phase == "stooped_resting":
        w, h = 55 + t * 20, 128 - t * 50
        cx, cy = width * 0.75, ground_y - h / 2
    elif phase == "sudden_fall":
        c_t = min(1.0, t / 0.3)
        w, h = 70 + c_t * 60, 120 - c_t * 90
        cx, cy = width * 0.75, ground_y - h / 2
    else:
        w, h = 130 + rng.uniform(-4, 4), 30 + rng.uniform(-3, 3)
        cx, cy = width * 0.75, ground_y - 18

    bbox = (cx - w / 2, cy - h / 2, w, h)
    centroid = (cx, cy)

    canvas = np.full((height, width, 3), (15, 23, 42), dtype=np.uint8)
    cv2.line(canvas, (0, ground_y), (width, ground_y), (51, 65, 85), 2)
    x, y, w_i, h_i = [int(v) for v in bbox]
    cv2.ellipse(canvas, (int(cx), int(cy)), (max(w_i // 2, 6), max(h_i // 2, 6)), 0, 0, 360, (203, 213, 225), -1)

    tid = 1
    if tid not in state["tracks"]:
        state["tracks"][tid] = Track(tid, centroid, bbox, frame_idx)
    else:
        state["tracks"][tid].update(centroid, bbox, frame_idx)

    feats = extract_features(state["tracks"][tid])
    new_events = []
    box_color, label = (50, 200, 100), "ID 1 · Normal"

    if feats:
        cond, conf = classify(feats, sensitivity)
        if cond:
            box_color = (40, 40, 235)
            info = TRIAGE_RULES[cond]
            label = f"{info['icon']} {info['en']}"
            last_fired = state["global_cooldown"].get(cond, -10_000)
            if frame_idx - last_fired > cooldown_frames:
                state["global_cooldown"][cond] = frame_idx
                new_events.append((tid, cond, conf))

    cv2.rectangle(canvas, (x, y), (x + w_i, y + h_i), box_color, 2)
    cv2.putText(canvas, label, (x, max(y - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2, cv2.LINE_AA)
    return canvas, new_events, 1


# ============================================================================
# SESSION STATE & DISPATCH LOGIC
# ============================================================================

if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "alert_seq" not in st.session_state:
    st.session_state.alert_seq = 0
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = {"frame_idx": 0, "active_tracks": 0, "fps_proc": 0.0, "video_time_s": 0.0}
if "last_frame_rgb" not in st.session_state:
    st.session_state.last_frame_rgb = None


def push_alert(location, condition_key, confidence, frame_idx, video_time_s):
    st.session_state.alert_seq += 1
    a = Alert(
        id=f"EMS-{st.session_state.alert_seq:03d}",
        frame_idx=frame_idx,
        video_time_s=video_time_s,
        wall_clock=datetime.now().strftime("%H:%M:%S"),
        location=location,
        condition_key=condition_key,
        confidence=confidence,
    )
    st.session_state.alerts.append(a)


# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

with st.sidebar:
    st.markdown("### 🎛️ غرفة التحكم والمراقبة")
    st.caption("Control Panel · System Parameters")

    simulation_mode = st.toggle(
        "وضع المحاكاة الافتراضي (Simulation Mode)",
        value=False,
        help="شغّل هذا الخيار لعرض حركة تجريبية بدون ملف فيديو، أو أطفئه لرفع فيديو حقيقي.",
    )

    uploaded_file = st.file_uploader(
        "رفع تسجيل كاميرا المراقبة (.mp4)",
        type=["mp4", "avi", "mov"],
        disabled=simulation_mode,
    )

    st.markdown("---")
    location_choice = st.selectbox("نطاق الكاميرا (Camera Zone)", LOCATIONS, index=1)
    if location_choice.startswith("موقع مخصص"):
        location_choice = st.text_input("اسم الموقع المخصص", value="Zone X – Central")

    sensitivity = st.slider("حساسية الرصد (Sensitivity Threshold)", 1, 100, 55)

    st.markdown("---")
    playback_fps = st.slider("سرعة العرض (Playback FPS)", 6, 30, 16)
    max_frames = st.slider("أقصى عدد إطارات للفحص (Max Frames)", 60, 800, 320, step=20)

    st.markdown("---")
    c1, c2 = st.columns(2)
    start_btn = c1.button("▶ تشغيل (Start)", use_container_width=True, type="primary")
    reset_btn = c2.button("⟲ إعادة ضبط", use_container_width=True)

    if reset_btn:
        st.session_state.alerts = []
        st.session_state.last_frame_rgb = None
        st.session_state.last_metrics = {"frame_idx": 0, "active_tracks": 0, "fps_proc": 0.0, "video_time_s": 0.0}
        st.rerun()

# ============================================================================
# MAIN DASHBOARD LAYOUT
# ============================================================================

st.markdown('<div class="system-title">🚑 نظام الرصد والفرز الإسعافي المبكر</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="system-subtitle">Early Pre-Collapse Anomaly Detection & AI Medical Triage Dashboard</div>',
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.3, 1])

with left_col:
    st.markdown("##### 📹 بث المراقبة الحي (Live Video Feed)")
    video_box = st.empty()
    metrics_box = st.empty()

with right_col:
    st.markdown("##### 🚨 سجل التنبيهات الإسعافية (Live Triage Feed)")
    alert_placeholder = st.container()


def render_metrics(m):
    metrics_box.markdown(
        f"""
        <div class="status-strip">
            <div class="kpi-card">
                <div class="kpi-val">{m['frame_idx']}</div>
                <div class="kpi-lbl">الإطار (Frame)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val">{m['video_time_s']:.1f}s</div>
                <div class="kpi-lbl">الزمن (Time)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val">{m['active_tracks']}</div>
                <div class="kpi-lbl">الأشخاص (Tracks)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val">{m['fps_proc']:.1f}</div>
                <div class="kpi-lbl">السرعة (FPS)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val" style="color:#EF4444">{len(st.session_state.alerts)}</div>
                <div class="kpi-lbl">البلاغات (Alerts)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_live_html_cards():
    """Renders HTML-only alert cards during the real-time processing loop."""
    if not st.session_state.alerts:
        alert_placeholder.info("لا توجد بلاغات إسعافية حتى الآن. سيتم فرز الحالات فور رصدها تلقائياً.")
        return

    html = ""
    for a in reversed(st.session_state.alerts):
        info = TRIAGE_RULES[a.condition_key]
        badge_color = PRIORITY_COLOR[info["priority"]]
        html += f"""
        <div class="alert-card" style="border-left-color: {badge_color};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="badge" style="background: {badge_color};">{info['priority']} PRIORITY</span>
                <span class="meta-line">#{a.id} · {a.wall_clock} · t={a.video_time_s:.1f}s</span>
            </div>
            <div class="cond-title-ar">{info['icon']} {info['ar']}</div>
            <div class="cond-title-en">{info['en']} (Confidence: {a.confidence*100:.0f}%)</div>
            <div class="meta-line">📍 {a.location}</div>
        </div>
        """
    alert_placeholder.markdown(html, unsafe_allow_html=True)


def render_interactive_alerts():
    """Renders full interactive cards with Dispatch buttons after loop finishes."""
    alert_placeholder.empty()
    with alert_placeholder:
        if not st.session_state.alerts:
            st.info("لا توجد بلاغات إسعافية حتى الآن. سيتم فرز الحالات فور رصدها تلقائياً.")
            return

        for a in reversed(st.session_state.alerts):
            info = TRIAGE_RULES[a.condition_key]
            badge_color = PRIORITY_COLOR[info["priority"]]
            st.markdown(
                f"""
                <div class="alert-card" style="border-left-color: {badge_color};">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="badge" style="background: {badge_color};">{info['priority']} PRIORITY</span>
                        <span class="meta-line">#{a.id} · {a.wall_clock} · t={a.video_time_s:.1f}s</span>
                    </div>
                    <div class="cond-title-ar">{info['icon']} {info['ar']}</div>
                    <div class="cond-title-en">{info['en']} (Confidence: {a.confidence*100:.0f}%)</div>
                    <div class="meta-line">📍 {a.location}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            b1, b2 = st.columns([1.2, 1])
            with b1:
                if not a.dispatched:
                    if st.button("🚑 توجيه فرقة إسعافية (Dispatch)", key=f"btn_{a.id}_{a.frame_idx}"):
                        a.dispatched = True
                        a.dispatched_time = datetime.now().strftime("%H:%M:%S")
                        st.rerun()
                else:
                    st.button("✅ تم توجيه الفرقة", key=f"done_{a.id}_{a.frame_idx}", disabled=True)
            with b2:
                if a.dispatched:
                    st.markdown(f'<span class="dispatched-tag">🚨 تم التوجيه: {a.dispatched_time}</span>', unsafe_allow_html=True)
            st.write("")


# ============================================================================
# PROCESSING EXECUTION LOOP
# ============================================================================

def run_loop():
    state = new_detector_state()
    rng = np.random.default_rng(7)
    progress_bar = st.progress(0.0, text="جاري المعالجة والتحليل...")
    tfile_name = None

    if simulation_mode:
        width, height = 640, 400
        fps_source = 25.0
        total_frames = max_frames
        cap = None
    else:
        if uploaded_file is None:
            st.warning("الرجاء رفع ملف فيديو أولاً أو تفعيل وضع المحاكاة.")
            return

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile_name = tfile.name
        tfile.close()

        cap = cv2.VideoCapture(tfile_name)
        if not cap.isOpened():
            st.error("تعذر فتح ملف الفيديو.")
            return

        fps_source = cap.get(cv2.CAP_PROP_FPS) or 25.0
        vid_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
        total_frames = min(max_frames, vid_total) if vid_total > 0 else max_frames
        width, height = 640, 400

    start_time = time.time()
    frame_idx = 0
    processed = 0

    while processed < total_frames:
        frame_idx += 1

        if simulation_mode:
            frame_bgr, events, active_tracks = process_sim_frame(
                frame_idx, state, sensitivity, width, height, rng
            )
        else:
            ok, raw = cap.read()
            if not ok:
                break
            raw = cv2.resize(raw, (width, height))
            frame_bgr, events, active_tracks = process_real_frame(
                raw, frame_idx, state, sensitivity
            )

        if events:
            for tid, cond, conf in events:
                push_alert(location_choice, cond, conf, frame_idx, frame_idx / fps_source)
            render_live_html_cards()

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        st.session_state.last_frame_rgb = rgb

        elapsed = max(time.time() - start_time, 1e-6)
        st.session_state.last_metrics = {
            "frame_idx": frame_idx,
            "active_tracks": active_tracks,
            "fps_proc": processed / elapsed if processed else 0.0,
            "video_time_s": frame_idx / fps_source,
        }

        video_box.image(rgb, use_container_width=True)
        render_metrics(st.session_state.last_metrics)

        processed += 1
        progress_bar.progress(processed / total_frames, text=f"تحليل الإطارات... {processed}/{total_frames}")
        time.sleep(1.0 / playback_fps)

    if cap is not None:
        cap.release()
    if tfile_name is not None and os.path.exists(tfile_name):
        try:
            os.remove(tfile_name)
        except Exception:
            pass

    progress_bar.empty()
    render_interactive_alerts()


if start_btn:
    run_loop()

# Static / initial load display
if st.session_state.last_frame_rgb is not None:
    video_box.image(st.session_state.last_frame_rgb, use_container_width=True)
else:
    placeholder_canvas = np.full((400, 640, 3), (15, 23, 42), dtype=np.uint8)
    cv2.putText(
        placeholder_canvas, "اضغط بدء للتشغيل | Press Start to begin", (120, 205),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 163, 184), 1, cv2.LINE_AA,
    )
    video_box.image(placeholder_canvas, use_container_width=True)

render_metrics(st.session_state.last_metrics)
render_interactive_alerts()