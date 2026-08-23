"""
نظام بصير للرصد والفرز الإسعافي والأمني المبكر
Baseer – AI Early Multi-Modal Anomaly Detection & Triage Command Center
========================================================================
- Streamlit Live Streaming Engine
- Custom Taxonomy Architecture
- Zero Key Collision & Anti-Ghosting Spatial Filtering
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
# PAGE CONFIG & COMMAND CENTER THEME
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
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# UPDATED TAXONOMY MAPPING
# ============================================================================

TAXONOMY_RULES = {
    # 1. Physical Violence & Assaults
    "fighting": {
        "category": "Physical Violence & Assaults",
        "ar": "شجار واشتباك جسدي",
        "en": "fighting",
        "priority": "High",
        "color": "#F97316",
        "icon": "🥊",
        "action": "توجيه دورية أمن الميدان فوراً لفض الاشتباك",
    },

    # 2. Falls & Complex Medical Emergencies
    "sudden_fall": {
        "category": "Falls & Complex Medical Emergencies",
        "ar": "سقوط مفاجئ وفقدان فوري للتوازن",
        "en": "sudden_fall",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "🚨",
        "action": "توجيه فرقة التدخل السريع",
    },
    "slow_fall": {
        "category": "Falls & Complex Medical Emergencies",
        "ar": "سقوط بطيء وتدريجي",
        "en": "slow_fall",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "⬇️",
        "action": "فحص العلامات الحيوية ونقل المصاب",
    },
    "sudden_fall_followed_by_seizure": {
        "category": "Falls & Complex Medical Emergencies",
        "ar": "سقوط مفاجئ متبوع بنوبة تشنج/صرع",
        "en": "sudden_fall_followed_by_seizure",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "⚡",
        "action": "تأمين محيط المصاب وحماية الرأس وتوجيه إسعاف عاجل",
    },
    "slow_fall_followed_by_seizure": {
        "category": "Falls & Complex Medical Emergencies",
        "ar": "سقوط بطيء متبوع بنوبة تشنج/صرع",
        "en": "slow_fall_followed_by_seizure",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "⚡",
        "action": "توجيه العناية المركزة الميدانية مباشرة",
    },
    "slow_fall_followed_by_opisthotonos": {
        "category": "Falls & Complex Medical Emergencies",
        "ar": "سقوط بطيء متبوع بتشنج ظهري (Opisthotonos)",
        "en": "slow_fall_followed_by_opisthotonos",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "☣️",
        "action": "تثبيت العمود الفقري واستدعاء الطبيب المناظر",
    },
    "rolling_and_severe_coughing": {
        "category": "Falls & Complex Medical Emergencies",
        "ar": "تدحرج على الأرض وسعال حاد",
        "en": "rolling_and_severe_coughing",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "🫁",
        "action": "توفير قناع أكسجين وتأمين مجرى التنفس",
    },
    "lying_immobile": {
        "category": "Falls & Complex Medical Emergencies",
        "ar": "استلقاء وسقوط بدون حركة (فقدان وعي)",
        "en": "lying_immobile",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "🛑",
        "action": "توجيه فريق الإنعاش القلبي الرئوي",
    },
    "crawling_on_floor": {
        "category": "Falls & Complex Medical Emergencies",
        "ar": "زحف كامل على الأرض وعدم قدرة على الوقوف",
        "en": "crawling_on_floor",
        "priority": "High",
        "color": "#F97316",
        "icon": "🚷",
        "action": "إرسال نقالة طبية عاجلة لمنع الدهس",
    },

    # 3. Abnormal Gait & Physical Fatigue
    "regular_limping": {
        "category": "Abnormal Gait & Fatigue",
        "ar": "عرج منتظم (خطوات متشابهة ومتكررة)",
        "en": "regular_limping",
        "priority": "Medium",
        "color": "#F59E0B",
        "icon": "🚶",
        "action": "توجيه كرسي إسعافي متحرك لنقل المصاب",
    },
    "irregular_limping": {
        "category": "Abnormal Gait & Fatigue",
        "ar": "عرج غير منتظم (خطوات متغيرة وغير متناسقة)",
        "en": "irregular_limping",
        "priority": "High",
        "color": "#F97316",
        "icon": "👣",
        "action": "تنبيه نقطة الرعاية الميدانية وتفقد القدم",
    },
    "Exhausted_walking": {
        "category": "Abnormal Gait & Fatigue",
        "ar": "مشي بإجهاد وإعياء حاد",
        "en": "Exhausted_walking",
        "priority": "High",
        "color": "#F97316",
        "icon": "😫",
        "action": "تقديم السوائل ونقل المصاب لمنطقة التبريد",
    },
    "arm_injury": {
        "category": "Abnormal Gait & Fatigue",
        "ar": "إصابة والتواء في الذراع / اليد",
        "en": "arm_injury",
        "priority": "Medium",
        "color": "#F59E0B",
        "icon": "🩹",
        "action": "توجيه حقيبة إسعافات أولية لتثبيت الذراع",
    },

    # 4. Fast Movement & Dynamic Activities
    "rapid_breathing": {
        "category": "Dynamic Activities & Respiration",
        "ar": "نهث وتسارع غير طبيعي في التنفس",
        "en": "rapid_breathing",
        "priority": "Medium",
        "color": "#F59E0B",
        "icon": "🫀",
        "action": "تهدئة المصاب وقياس نسبة تشبع الأكسجين",
    },
    "running_sprinting": {
        "category": "Dynamic Activities & Respiration",
        "ar": "جري وركض سريع في المسار",
        "en": "running_sprinting",
        "priority": "Low",
        "color": "#3B82F6",
        "icon": "🏃",
        "action": "مراقبة التدفق لمنع التدافع العشوائي",
    },
    "jumping": {
        "category": "Dynamic Activities & Respiration",
        "ar": "قفز حركي متكرر",
        "en": "jumping",
        "priority": "Low",
        "color": "#3B82F6",
        "icon": "🦘",
        "action": "مراقبة اعتيادية",
    },
    "dancing": {
        "category": "Dynamic Activities & Respiration",
        "ar": "حركات رقص أو استعراض",
        "en": "dancing",
        "priority": "Low",
        "color": "#3B82F6",
        "icon": "💃",
        "action": "مراقبة اعتيادية",
    },
    "situps_exercise": {
        "category": "Dynamic Activities & Respiration",
        "ar": "تمارين بدنية أرضية (Sit-ups)",
        "en": "situps_exercise",
        "priority": "Low",
        "color": "#3B82F6",
        "icon": "🧘",
        "action": "مراقبة اعتيادية",
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
# DATA STRUCTURES & FEATURE EXTRACTION
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
        self.history = deque(maxlen=45)
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
    if len(hist) < 8:
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

    # 1. Fall & Seizure Variations
    if (f["aspect_curr"] > 1.05 and f["h_drop"] > 0.32 * (1.1 - 0.3 * s)) or (
        f["aspect_prev"] < 0.90 and f["aspect_curr"] > 1.12 and f["max_vert_v"] > 0.05
    ):
        if f["speed_jitter"] > 0.04:
            return "sudden_fall_followed_by_seizure", min(0.98, 0.80 + 0.15 * s)
        return "sudden_fall", min(0.95, 0.78 + 0.18 * s)

    if 0.22 < f["h_drop"] <= 0.32 and f["aspect_curr"] > 1.0:
        if f["speed_jitter"] > 0.04:
            return "slow_fall_followed_by_seizure", min(0.94, 0.75 + 0.18 * s)
        return "slow_fall", min(0.91, 0.65 + 0.2 * s)

    # 2. Prone & Immobility Conditions
    prone = f["aspect_curr"] > 1.15
    if prone and f["displacement"] < 0.15:
        if f["speed_jitter"] < 0.01:
            return "lying_immobile", min(0.96, 0.80 + 0.15 * s)
        elif f["speed_jitter"] > 0.05:
            return "rolling_and_severe_coughing", min(0.93, 0.70 + 0.2 * s)

    # 3. Exhaustion & Gait Analysis
    if not prone and f["speed_jitter"] > 0.075 * (1.1 - 0.2 * s):
        # الفرق بين العرج المنتظم وغير المنتظم بناءً على تشتت السرعة الخطية
        if f["speed_jitter"] > 0.11:
            return "irregular_limping", min(0.90, 0.60 + f["speed_jitter"] * 3.0)
        else:
            return "regular_limping", min(0.88, 0.55 + f["speed_jitter"] * 3.5)

    if not prone and f["speed_mean"] < 0.02 and f["h_drop"] > 0.15:
        return "Exhausted_walking", min(0.85, 0.60 + 0.2 * s)

    return None, 0.0


# ============================================================================
# OPENCV ENGINE
# ============================================================================

_KERNEL_OPEN = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
_KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))


def new_state():
    bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    return {"bg": bg, "tracks": {}, "next_id": 1, "global_cd": {}}


def process_video_frame(frame, frame_idx, state, sensitivity, min_area=3200):
    kernel_open = _KERNEL_OPEN
    kernel_close = _KERNEL_CLOSE

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
        if w < 30 or h < 30:
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
        if tr.last_seen != frame_idx or tr.age < 8:
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
                if frame_idx - last_f > 300:
                    state["global_cd"][cond] = frame_idx
                    new_alerts.append((cond, conf))
            elif f["speed_jitter"] > 0.025:
                color, tag = (0, 190, 245), f"ID {tid} - Monitoring"

        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        cv2.putText(canvas, tag, (x, max(y - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

    return canvas, new_alerts, active_count


# ============================================================================
# SYNTHETIC SIMULATION PIPELINE
# ============================================================================

def process_sim(frame_idx, state, sensitivity, w, h):
    phases = ["normal_walk", "regular_limping", "Exhausted_walking", "sudden_fall_followed_by_seizure", "lying_immobile"]
    p_len = 65
    p_idx = (frame_idx % (p_len * len(phases))) // p_len
    phase = phases[p_idx]
    t = (frame_idx % p_len) / p_len
    ground = h - 60

    if phase == "normal_walk":
        bw, bh = 46, 125
        cx, cy = w * 0.2 + t * w * 0.4, ground - bh / 2
    elif phase == "regular_limping":
        bw, bh = 50, 120
        cx, cy = w * 0.6 + math.sin(t * 18) * 12, ground - bh / 2
    elif phase == "Exhausted_walking":
        bw, bh = 54, 100
        cx, cy = w * 0.65, ground - bh / 2
    elif phase == "sudden_fall_followed_by_seizure":
        c = min(1.0, t / 0.32)
        bw, bh = 50 + c * 70, 120 - c * 90
        cx, cy = w * 0.65, ground - bh / 2
    else:
        bw, bh = 125, 30
        cx, cy = w * 0.65, ground - 16

    canvas = np.full((h, w, 3), (11, 19, 43), dtype=np.uint8)
    cv2.line(canvas, (0, ground), (w, ground), (58, 80, 107), 2)
    cv2.ellipse(canvas, (int(cx), int(cy)), (max(int(bw / 2), 6), max(int(bh / 2), 6)), 0, 0, 360, (144, 224, 239), -1)

    tid = 1
    if tid not in state["tracks"]:
        state["tracks"][tid] = Track(tid, (cx, cy), (cx - bw / 2, cy - bh / 2, bw, bh), frame_idx)
    else:
        state["tracks"][tid].update((cx, cy), (cx - bw / 2, cy - bh / 2, bw, bh), frame_idx)

    feats = extract_features(state["tracks"][tid])
    new_alerts = []
    color, tag = (40, 200, 100), "Normal: 0"

    if feats:
        cond, conf = classify_taxonomy(feats, sensitivity)
        if cond and cond in TAXONOMY_RULES:
            color = (40, 40, 235)
            tag = f"Abnormal: {TAXONOMY_RULES[cond]['en']}"
            last_f = state["global_cd"].get(cond, -9999)
            if frame_idx - last_f > 150:
                state["global_cd"][cond] = frame_idx
                new_alerts.append((cond, conf))

    x, y = int(cx - bw / 2), int(cy - bh / 2)
    cv2.rectangle(canvas, (x, y), (x + int(bw), y + int(bh)), color, 2)
    cv2.putText(canvas, tag, (x, max(y - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
    return canvas, new_alerts, 1


# ============================================================================
# STATE & UI MANAGEMENT
# ============================================================================

if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "metrics" not in st.session_state:
    st.session_state.metrics = {"frame": 0, "tracks": 0, "fps": 0.0, "time": 0.0}

st.markdown(
    """
    <div class="header-box">
        <div>
            <div class="system-title">🚑 نظام بصير | AI Anomaly Detection & Triage</div>
            <div class="system-sub">منظومة الرصد والفرز الذكي للمؤشرات الحيوية والحركية الحرجة</div>
        </div>
        <div class="live-badge">● LIVE DISPATCH SYSTEM</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🎛️ غرفة العمليات والتحكم")
    st.caption("Operations & Taxonomy Control Hub")

    feed_mode = st.radio(
        "مصدر البث (Feed Source)",
        ["وضع المحاكاة التفاعلي (Simulation Mode)", "رفع فيديو مراقبة (Upload Video)"],
    )

    uploaded_vid = None
    if feed_mode == "رفع فيديو مراقبة (Upload Video)":
        uploaded_vid = st.file_uploader("اختر مقطع الكاميرا (.mp4)", type=["mp4", "avi", "mov"])

    st.markdown("---")
    selected_zone = st.selectbox("نطاق الكاميرا والموقع (Zone)", LOCATIONS)
    sens = st.slider("حساسية الرصد والاستجابة (Sensitivity)", 20, 100, 60)

    st.markdown("---")
    play_speed = st.slider("معدل العرض (FPS)", 6, 30, 16)
    max_f = st.slider("إجمالي الإطارات للفحص (Max Frames)", 80, 800, 320, step=20)

    st.markdown("---")
    col1, col2 = st.columns(2)
    start_btn = col1.button("▶ تشغيل الرصد", use_container_width=True, type="primary")
    reset_btn = col2.button("⟲ إعادة ضبط", use_container_width=True)

    if reset_btn:
        st.session_state.alerts = []
        st.session_state.metrics = {"frame": 0, "tracks": 0, "fps": 0.0, "time": 0.0}
        st.rerun()

col_cam, col_triage = st.columns([1.35, 1])

with col_cam:
    st.markdown("##### 📹 البث التحليلي المباشر (Analytical Feed)")
    cam_holder = st.empty()
    kpi_holder = st.empty()

with col_triage:
    st.markdown("##### 🚨 سجل الفرز والتوجيه الميداني (Live Triage Log)")
    triage_holder = st.empty()


def draw_kpi_html(m):
    return f"""
    <div class="kpi-container">
        <div class="kpi-card"><div class="kpi-num">{m['frame']}</div><div class="kpi-title">الإطار (Frame)</div></div>
        <div class="kpi-card"><div class="kpi-num">{m['time']:.1f}s</div><div class="kpi-title">الزمن (Time)</div></div>
        <div class="kpi-card"><div class="kpi-num">{m['tracks']}</div><div class="kpi-title">الأشخاص (Active)</div></div>
        <div class="kpi-card"><div class="kpi-num">{m['fps']:.1f}</div><div class="kpi-title">المعالجة (FPS)</div></div>
        <div class="kpi-card"><div class="kpi-num" style="color:#EF4444">{len(st.session_state.alerts)}</div><div class="kpi-title">البلاغات (Alerts)</div></div>
    </div>
    """


def draw_triage_html():
    if not st.session_state.alerts:
        return "<div style='color:#94A3B8; padding:1rem; border:1px dashed #334155; border-radius:8px;'>لا توجد بلاغات إسعافية حتى الآن. النظام يراقب البث...</div>"

    html_out = ""
    for idx, a in enumerate(reversed(st.session_state.alerts)):
        info = TAXONOMY_RULES.get(a.condition_key, TAXONOMY_RULES["sudden_fall"])
        b_color = PRIORITY_COLOR[info["priority"]]
        dispatch_status = "✅ تم توجيه الفرقة" if a.dispatched else "⚠️ قيد الانتظار"

        html_out += f"""
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
            <div style="margin-top:0.5rem; font-size:0.85rem; font-weight:bold; color:#38BDF8;">الحالة: {dispatch_status}</div>
        </div>
        """
    return html_out


def run_detection():
    st.session_state.alerts = []
    state = new_state()
    is_sim = feed_mode.startswith("وضع المحاكاة")
    tfile_path = None

    if is_sim:
        w, h, cap, fps_src, total_frames = 640, 400, None, 25.0, max_f
    else:
        if uploaded_vid is None:
            st.warning("الرجاء رفع ملف فيديو أولاً.")
            return

        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tf.write(uploaded_vid.read())
        tfile_path = tf.name
        tf.close()

        cap = cv2.VideoCapture(tfile_path)
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        v_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_f
        total_frames = min(max_f, v_total)
        w, h = 640, 400

    start_t = time.time()
    frame_idx = 0
    proc = 0

    target_delay = 1.0 / play_speed

    while proc < total_frames:
        loop_start = time.time()
        frame_idx += 1

        if is_sim:
            frame_bgr, evts, tracks = process_sim(frame_idx, state, sens, w, h)
        else:
            ok, raw = cap.read()
            if not ok:
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

        cam_holder.image(rgb, use_container_width=True, output_format="JPEG")
        kpi_holder.markdown(draw_kpi_html(st.session_state.metrics), unsafe_allow_html=True)
        triage_holder.markdown(draw_triage_html(), unsafe_allow_html=True)

        compute_duration = time.time() - loop_start
        sleep_time = target_delay - compute_duration
        if sleep_time > 0:
            time.sleep(sleep_time)

    if cap:
        cap.release()
    if tfile_path and os.path.exists(tfile_path):
        try:
            os.remove(tfile_path)
        except Exception:
            pass


if start_btn:
    run_detection()
else:
    kpi_holder.markdown(draw_kpi_html(st.session_state.metrics), unsafe_allow_html=True)
    triage_holder.markdown(draw_triage_html(), unsafe_allow_html=True)
