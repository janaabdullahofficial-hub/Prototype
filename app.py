"""
نظام بصير للرصد والفرز الإسعافي والأمني المبكر
Baseer – AI Early Multi-Modal Anomaly Detection & Triage Command Center
"""

import math
import os
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

# ============================================================================
# PAGE CONFIG & MODERN COMMAND CENTER THEME
# ============================================================================

st.set_page_config(
    page_title="بصير | منصة الرصد والفرز المبكر الموحدة",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;900&family=JetBrains+Mono:wght@500;700&display=swap');
    * { font-family: 'Tajawal', -apple-system, sans-serif; }
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
    .card-ar { font-size: 1.05rem; font-weight: 800; color: #F8FAFC; margin-top: 0.4rem; }
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
    .eta-box {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid #10B981;
        color: #10B981;
        padding: 6px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# MEDICAL & BEHAVIORAL TAXONOMY (WITH HEATSTROKE AS CORE TRIAGE)
# ============================================================================

TAXONOMY_RULES = {
    "heatstroke_exhaustion": {
        "category": "Medical & Respiratory Distress",
        "ar": "ضربة شمس حادة / إجهاد حراري وهبوط عام",
        "en": "heatstroke_exhaustion",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "☀️",
        "action": "توجيه فرقة إسعافية فورية مع معدات التبريد ومحاليل الإرواء",
    },
    "sudden_fall": {
        "category": "Falls & Abnormal Locomotion",
        "ar": "سقوط مفاجئ وفقدان فوري للتوازن",
        "en": "sudden_fall",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "🚨",
        "action": "توجيه فرقة الإنعاش القلبي والتدخل السريع فوراً",
    },
    "slow_fall": {
        "category": "Falls & Abnormal Locomotion",
        "ar": "سقوط بطيء وتدريجي (هبوط إعياء حاد)",
        "en": "slow_fall",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "⬇️",
        "action": "فحص العلامات الحيوية ونقل المصاب لمنطقة مظللة",
    },
    "severe_gait_limping": {
        "category": "Falls & Abnormal Locomotion",
        "ar": "عرج شديد ومطرد / بوادر ضربة شمس وجفاف",
        "en": "severe_gait_limping",
        "priority": "High",
        "color": "#F97316",
        "icon": "🚶",
        "action": "توجيه كرسي إسعافي متحرك ومسعف راجل للتقييم",
    },
    "stooped_walking_resting": {
        "category": "Falls & Abnormal Locomotion",
        "ar": "مشي بظهر منحنٍ واستناد للراحة عند الرصيف",
        "en": "stooped_walking_resting",
        "priority": "High",
        "color": "#F97316",
        "icon": "🧍",
        "action": "نقل المصاب إلى مظلة رعاية وتفقد الضغط والسكر",
    },
    "severe_choking_on_ground": {
        "category": "Medical & Respiratory Distress",
        "ar": "استلقاء أرضي ممتد مع اضطراب تنفسي",
        "en": "severe_choking_on_ground",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "🫁",
        "action": "تأمين مجرى التنفس والتدخل الإسعافي الفوري",
    },
    "seizure_convulsion": {
        "category": "Medical & Respiratory Distress",
        "ar": "تشنج عصبي نشط ونوبة صرع",
        "en": "seizure_convulsion",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "⚡",
        "action": "حماية رأس المصاب وتأمين المحيط لمنع التدافع",
    },
    "running_sprinting": {
        "category": "Fast Movement & Dynamic Activities",
        "ar": "جري وركض سريع في المسار",
        "en": "running_sprinting",
        "priority": "Low",
        "color": "#3B82F6",
        "icon": "🏃",
        "action": "مراقبة تدفق الحشود وتفادي التدافع",
    },
}

PRIORITY_COLOR = {"Critical": "#DC2626", "High": "#F97316", "Medium": "#F59E0B", "Low": "#3B82F6"}

LOCATIONS = [
    "ممشى المشاعر – ممر رقم 12 (Pilgrim Corridor 12)",
    "ساحة الحرم المركزية – بوابة الملك فهد (King Fahd Gate)",
    "محطة قطار الحرمين – الصالة 2 (Train Station Hub)",
    "المستشفى الميداني – محيط جسر الجمرات (Jamarat Bridge)",
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
    dispatched: bool = False


class Track:
    def __init__(self, track_id, centroid, bbox, frame_idx):
        self.id = track_id
        self.history = deque(maxlen=50)
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
    if len(hist) < 6:
        return None

    heights = [h["b"][3] for h in hist]
    widths = [h["b"][2] for h in hist]
    cxs = [h["c"][0] for h in hist]
    cys = [h["c"][1] for h in hist]

    curr_h = max(heights[-1], 20.0)
    aspect_ratios = [w / max(h, 1.0) for w, h in zip(widths, heights)]

    aspect_curr = float(np.mean(aspect_ratios[-4:]))
    aspect_prev = float(np.mean(aspect_ratios[:4]))

    h_drop = (np.mean(heights[:4]) - np.mean(heights[-4:])) / max(np.mean(heights[:4]), 1.0)
    vert_v = np.diff(cys) / curr_h
    horiz_v = np.diff(cxs) / curr_h

    displacement = math.hypot(cxs[-1] - cxs[0], cys[-1] - cys[0]) / curr_h
    speed_mean = float(np.mean(np.abs(horiz_v))) if len(horiz_v) else 0.0
    speed_jitter = float(np.std(horiz_v)) if len(horiz_v) else 0.0

    return dict(
        aspect_curr=aspect_curr,
        aspect_prev=aspect_prev,
        h_drop=h_drop,
        max_vert_v=float(np.max(np.abs(vert_v))) if len(vert_v) else 0.0,
        displacement=displacement,
        speed_mean=speed_mean,
        speed_jitter=speed_jitter,
    )


def classify_taxonomy(f: dict, sensitivity: int):
    s = sensitivity / 100.0

    # 1. Sudden Fall vs Slow Fall
    if (f["aspect_curr"] > 1.05 and f["h_drop"] > 0.28 * (1.1 - 0.3 * s)) or (
        f["aspect_prev"] < 0.92 and f["aspect_curr"] > 1.10
    ):
        return "sudden_fall", min(0.98, 0.80 + 0.18 * s)

    if 0.18 < f["h_drop"] <= 0.28 and f["aspect_curr"] > 1.0:
        return "slow_fall", min(0.92, 0.68 + 0.2 * s)

    # 2. Prolonged Ground Immobilization & Seizures
    prone = f["aspect_curr"] > 1.10
    if prone and f["displacement"] < 0.20:
        if f["speed_jitter"] > 0.04:
            return "seizure_convulsion", min(0.95, 0.72 + 0.2 * s)
        return "severe_choking_on_ground", min(0.92, 0.68 + 0.2 * s)

    # 3. Heatstroke & Gait Instability
    if not prone and f["speed_jitter"] > 0.032 * (1.1 - 0.3 * s):
        if f["h_drop"] > 0.10:
            return "heatstroke_exhaustion", min(0.94, 0.70 + f["speed_jitter"] * 4.0)
        return "severe_gait_limping", min(0.88, 0.58 + f["speed_jitter"] * 4.0)

    # 4. Stooped Walking / Resting
    if 0.12 < f["h_drop"] <= 0.28 and f["aspect_curr"] < 1.05:
        return "stooped_walking_resting", min(0.86, 0.58 + f["h_drop"] * 0.7)

    return None, 0.0


# ============================================================================
# OPENCV ENGINE
# ============================================================================

def new_state():
    bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=45, detectShadows=False)
    return {"bg": bg, "tracks": {}, "next_id": 1, "global_cd": {}}


def process_video_frame(frame, frame_idx, state, sensitivity, min_area=2000):
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    fgmask = state["bg"].apply(frame)
    _, fgmask = cv2.threshold(fgmask, 220, 255, cv2.THRESH_BINARY)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_close, iterations=3)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 25 or h < 25:
            continue
        detections.append(((x + w / 2, y + h / 2), (x, y, w, h), area))

    detections = sorted(detections, key=lambda d: d[2], reverse=True)[:2]

    assigned = set()
    for (cx, cy), (x, y, w, h), _ in detections:
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

    for tid in [t for t, obj in state["tracks"].items() if frame_idx - obj.last_seen > 12]:
        del state["tracks"][tid]

    canvas = frame.copy()
    new_alerts = []
    active_count = 0

    for tid, tr in state["tracks"].items():
        if tr.last_seen != frame_idx or tr.age < 5:
            continue

        active_count += 1
        x, y, w, h = tr.bbox
        f = extract_features(tr)
        color, tag = (40, 200, 100), f"ID {tid} - Normal (0)"

        if f:
            cond, conf = classify_taxonomy(f, sensitivity)
            if cond and cond in TAXONOMY_RULES:
                color = (40, 40, 235)
                info = TAXONOMY_RULES[cond]
                tag = f"Abnormal: {info['en']}"

                last_f = state["global_cd"].get(cond, -9999)
                if frame_idx - last_f > 95:
                    state["global_cd"][cond] = frame_idx
                    new_alerts.append((cond, conf))
            elif f["speed_jitter"] > 0.025:
                color, tag = (0, 190, 245), f"ID {tid} - Monitoring"

        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        cv2.putText(canvas, tag, (x, max(y - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

    return canvas, new_alerts, active_count


# ============================================================================
# SYNTHETIC SIMULATION PIPELINE (WITH HEATSTROKE DEMO)
# ============================================================================

def process_sim(frame_idx, state, sensitivity, w, h):
    phases = [
        ("Normal Walk", "normal_walk", 50),
        ("Severe Heatstroke Symptoms", "heatstroke_exhaustion", 65),
        ("Pre-Collapse Stoop", "stooped_walking_resting", 55),
        ("Sudden Fall Event", "sudden_fall", 50),
        ("Immobilized on Ground", "severe_choking_on_ground", 65),
    ]
    
    total_cycle = sum(p[2] for p in phases)
    curr_t = frame_idx % total_cycle
    
    accum = 0
    curr_phase_name = "Normal Walk"
    curr_cond = "normal_walk"
    phase_progress = 0.0
    
    for name, cond, duration in phases:
        if accum <= curr_t < accum + duration:
            curr_phase_name = name
            curr_cond = cond
            phase_progress = (curr_t - accum) / duration
            break
        accum += duration

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (15, 23, 42)
    
    ground_y = h - 70
    cv2.line(canvas, (0, ground_y), (w, ground_y), (51, 65, 85), 3)

    if curr_cond == "normal_walk":
        bw, bh = 48, 140
        cx = int(w * 0.2 + phase_progress * w * 0.3)
        cy = int(ground_y - bh / 2)
    elif curr_cond == "heatstroke_exhaustion":
        bw, bh = 54, int(135 - phase_progress * 15)
        cx = int(w * 0.5 + math.sin(phase_progress * 20) * 16)
        cy = int(ground_y - bh / 2)
    elif curr_cond == "stooped_walking_resting":
        bw, bh = int(55 + phase_progress * 20), int(120 - phase_progress * 40)
        cx = int(w * 0.55)
        cy = int(ground_y - bh / 2)
    elif curr_cond == "sudden_fall":
        fall_t = min(1.0, phase_progress / 0.4)
        bw = int(55 + fall_t * 85)
        bh = int(120 - fall_t * 90)
        cx = int(w * 0.58)
        cy = int(ground_y - bh / 2)
    else:
        bw, bh = 140, 32
        cx = int(w * 0.58)
        cy = int(ground_y - 18)

    cv2.ellipse(canvas, (cx, cy), (max(int(bw / 2), 6), max(int(bh / 2), 6)), 0, 0, 360, (56, 189, 248), -1)
    if bh > 40:
        cv2.circle(canvas, (cx, cy - int(bh / 2) + 12), 14, (125, 211, 252), -1)

    bbox = (cx - bw / 2, cy - bh / 2, bw, bh)
    tid = 1
    if tid not in state["tracks"]:
        state["tracks"][tid] = Track(tid, (cx, cy), bbox, frame_idx)
    else:
        state["tracks"][tid].update((cx, cy), bbox, frame_idx)

    feats = extract_features(state["tracks"][tid])
    new_alerts = []
    box_color, tag = (40, 200, 100), "ID 1 - Normal (0)"

    if feats:
        cond, conf = classify_taxonomy(feats, sensitivity)
        if cond and cond in TAXONOMY_RULES:
            box_color = (40, 40, 235)
            tag = f"Abnormal: {TAXONOMY_RULES[cond]['en']}"
            last_f = state["global_cd"].get(cond, -9999)
            if frame_idx - last_f > 65:
                state["global_cd"][cond] = frame_idx
                new_alerts.append((cond, conf))

    x, y = int(cx - bw / 2), int(cy - bh / 2)
    cv2.rectangle(canvas, (x, y), (x + int(bw), y + int(bh)), box_color, 2)
    cv2.putText(canvas, tag, (x, max(y - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"SIMULATION: {curr_phase_name}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 163, 184), 1, cv2.LINE_AA)

    return canvas, new_alerts, 1


# ============================================================================
# STATE & UI MANAGEMENT
# ============================================================================

if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "metrics" not in st.session_state:
    st.session_state.metrics = {"frame": 0, "tracks": 0, "fps": 0.0, "time": 0.0}
if "last_img" not in st.session_state:
    st.session_state.last_img = None

st.markdown(
    """
    <div class="header-box">
        <div>
            <div class="system-title">🚑 نظام بصير | AI Anomaly Detection & Triage</div>
            <div class="system-sub">منظومة الرصد والفرز الذكي للمؤشرات الحيوية والحركية وإدارة بلاغات ضربات الشمس والسقوط</div>
        </div>
        <div class="live-badge">● LIVE DISPATCH SYSTEM</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🎛️ غرفة العمليات والتحكم")
    st.caption("Operations & Taxonomy Control Hub")

    mode_choice = st.radio(
        "مصدر البث (Feed Source)",
        ["Simulation", "Upload Video"],
        format_func=lambda x: "وضع المحاكاة التفاعلي (Simulation Mode)" if x == "Simulation" else "رفع فيديو مراقبة (Upload Video)",
    )

    uploaded_vid = None
    if mode_choice == "Upload Video":
        uploaded_vid = st.file_uploader("اختر مقطع الكاميرا (.mp4)", type=["mp4", "avi", "mov"])

    st.markdown("---")
    selected_zone = st.selectbox("نطاق الكاميرا والموقع (Zone)", LOCATIONS)
    sens = st.slider("حساسية الرصد والاستجابة (Sensitivity)", 20, 100, 65)

    st.markdown("---")
    play_speed = st.slider("معدل العرض (FPS)", 6, 30, 18)
    max_f = st.slider("إجمالي الإطارات للفحص (Max Frames)", 80, 800, 300, step=20)

    st.markdown("---")
    col1, col2 = st.columns(2)
    start_btn = col1.button("▶ تشغيل الرصد", use_container_width=True, type="primary")
    reset_btn = col2.button("⟲ إعادة ضبط", use_container_width=True)

    if reset_btn:
        st.session_state.alerts = []
        st.session_state.metrics = {"frame": 0, "tracks": 0, "fps": 0.0, "time": 0.0}
        st.session_state.last_img = None
        st.rerun()

col_cam, col_triage = st.columns([1.35, 1])

with col_cam:
    st.markdown("##### 📹 البث التحليلي المباشر (Analytical Feed)")
    cam_holder = st.empty()
    kpi_holder = st.empty()

with col_triage:
    st.markdown("##### 🚨 سجل الفرز والتوجيه الميداني (Live Triage Log)")
    triage_holder = st.container()


def render_kpis(m):
    kpi_holder.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-card"><div class="kpi-num">{m['frame']}</div><div class="kpi-title">الإطار (Frame)</div></div>
            <div class="kpi-card"><div class="kpi-num">{m['time']:.1f}s</div><div class="kpi-title">الزمن (Time)</div></div>
            <div class="kpi-card"><div class="kpi-num">{m['tracks']}</div><div class="kpi-title">الأشخاص (Active)</div></div>
            <div class="kpi-card"><div class="kpi-num">{m['fps']:.1f}</div><div class="kpi-title">المعالجة (FPS)</div></div>
            <div class="kpi-card"><div class="kpi-num" style="color:#EF4444">{len(st.session_state.alerts)}</div><div class="kpi-title">البلاغات (Alerts)</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_triage():
    triage_holder.empty()
    with triage_holder:
        if not st.session_state.alerts:
            st.info("لا توجد بلاغات إسعافية أو أمنية حرجة حتى الآن. النظام يراقب المؤشرات الحركية...")
            return

        for idx, a in enumerate(reversed(st.session_state.alerts)):
            info = TAXONOMY_RULES.get(a.condition_key, TAXONOMY_RULES["sudden_fall"])
            b_color = PRIORITY_COLOR[info["priority"]]
            st.markdown(
                f"""
                <div class="alert-card" style="border-left: 6px solid {b_color};">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="triage-badge" style="background:{b_color}">{info['priority']} PRIORITY</span>
                        <span class="card-meta">#{a.id} · {a.wall_clock} · t={a.video_time_s:.1f}s</span>
                    </div>
                    <div style="margin-top:0.3rem;"><span class="category-tag">📂 {info['category']}</span></div>
                    <div class="card-ar">{info['icon']} {info['ar']}</div>
                    <div class="card-en"><b>Class:</b> <code>{info['en']}</code> (الثقة: {a.confidence*100:.0f}%)</div>
                    <div class="card-meta">📍 {a.location}</div>
                    <div style="margin-top:0.4rem; font-size:0.8rem; color:#CBD5E1;"><b>الإجراء الموصى به:</b> {info['action']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            b1, b2 = st.columns([1.3, 1])
            with b1:
                if not a.dispatched:
                    if st.button("🚑 توجيه فرقة التدخل السريع", key=f"btn_dsp_{a.unique_key}_{idx}", type="primary"):
                        a.dispatched = True
                        st.rerun()
                else:
                    st.button("✅ تم توجيه الفرقة بنجاح", key=f"btn_done_{a.unique_key}_{idx}", disabled=True)
            with b2:
                if a.dispatched:
                    st.markdown(f'<div class="eta-box">🚨 الفرقة في الطريق (وصول: دقيقة ونصف)</div>', unsafe_allow_html=True)
            st.write("")


def run_detection():
    st.session_state.alerts = []
    state = new_state()
    p_bar = st.progress(0.0, text="جاري فحص وتتبع حركة الحشود...")
    tfile_path = None
    is_sim = (mode_choice == "Simulation")

    if is_sim:
        w, h, cap, fps_src, total_frames = 640, 400, None, 25.0, max_f
    else:
        if uploaded_vid is None:
            st.warning("الرجاء رفع ملف فيديو أولاً.")
            return
        
        # Safe Cloud-compatible persistent temp writing
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_vid.getbuffer())
        tfile_path = tfile.name
        tfile.close()
        
        cap = cv2.VideoCapture(tfile_path)
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        v_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_f
        total_frames = min(max_f, v_total)
        w, h = 640, 400

    start_t, frame_idx, proc = time.time(), 0, 0

    while proc < total_frames:
        frame_idx += 1
        if is_sim:
            frame_bgr, evts, tracks = process_sim(frame_idx, state, sens, w, h)
        else:
            ok, raw = cap.read()
            if not ok or raw is None:
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
        st.session_state.last_img = rgb
        elapsed = max(time.time() - start_t, 1e-6)
        st.session_state.metrics = {
            "frame": frame_idx,
            "tracks": tracks,
            "fps": proc / elapsed if proc else 0.0,
            "time": frame_idx / fps_src,
        }

        cam_holder.image(rgb, use_container_width=True)
        render_kpis(st.session_state.metrics)

        proc += 1
        p_bar.progress(proc / total_frames, text=f"تحليل الإطارات الذكي... {proc}/{total_frames}")
        time.sleep(1.0 / play_speed)

    if cap:
        cap.release()
    if tfile_path and os.path.exists(tfile_path):
        try:
            os.remove(tfile_path)
        except Exception:
            pass

    p_bar.empty()
    render_triage()


if start_btn:
    run_detection()

if st.session_state.last_img is not None:
    cam_holder.image(st.session_state.last_img, use_container_width=True)
else:
    placeholder = np.full((400, 640, 3), (11, 19, 43), dtype=np.uint8)
    cv2.putText(placeholder, "BASEER MULTI-MODAL TRIAGE", (120, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (72, 202, 228), 2, cv2.LINE_AA)
    cv2.putText(placeholder, "اضغط بدء الرصد للتشغيل الميداني", (160, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (144, 224, 239), 1, cv2.LINE_AA)
    cam_holder.image(placeholder, use_container_width=True)

render_kpis(st.session_state.metrics)
render_triage()
