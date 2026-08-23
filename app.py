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
# TAXONOMY MAPPING (INCLUDING SUNSTROKE)
# ============================================================================

TAXONOMY_RULES = {
    # Sunstroke & Heat Emergency
    "sunstroke_heat_exhaustion": {
        "category": "Heat Emergencies & Insolation",
        "ar": "ضربة شمس وإجهاد حراري حاد",
        "en": "sunstroke_heat_exhaustion",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "☀️",
        "action": "نقل المصاب لمظلة تبريد فوراً ورش الماء ونقله للإسعاف",
    },

    # Physical Violence & Assaults
    "fighting": {
        "category": "Physical Violence & Assaults",
        "ar": "شجار واشتباك جسدي",
        "en": "fighting",
        "priority": "High",
        "color": "#F97316",
        "icon": "🥊",
        "action": "توجيه دورية أمن الميدان فوراً لفض الاشتباك",
    },

    # Falls & Complex Medical Emergencies
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
    "lying_immobile": {
        "category": "Falls & Complex Medical Emergencies",
        "ar": "استلقاء وسقوط بدون حركة (فقدان وعي)",
        "en": "lying_immobile",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "🛑",
        "action": "توجيه فريق الإنعاش القلبي الرئوي",
    },

    # Abnormal Gait & Physical Fatigue
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
}

PRIORITY_COLOR = {"Critical": "#DC2626", "High": "#F97316", "Medium": "#F59E0B", "Low": "#3B82F6"}

LOCATIONS = [
    "ممشى المشاعر – ممر رقم 12 (Pilgrim Corridor 12)",
    "ساحة الحرم المركزية – بوابة الملك فهد (King Fahd Gate)",
    "محطة قطار الحرمين – الصالة 2 (Train Station Hub)",
    "المستشفى الميداني – محيط جسر الجمرات (Jamarat Bridge)",
]

# ============================================================================
# DATA STRUCTURES & TRACKING
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
    if len(hist) < 4:
        return None

    heights = [h["b"][3] for h in hist]
    widths = [h["b"][2] for h in hist]
    cxs = [h["c"][0] for h in hist]
    cys = [h["c"][1] for h in hist]

    curr_h = max(heights[-1], 20.0)
    aspect_ratios = [w / max(h, 1.0) for w, h in zip(widths, heights)]

    aspect_curr = float(np.mean(aspect_ratios[-3:]))
    aspect_prev = float(np.mean(aspect_ratios[:3]))

    h_drop = (np.mean(heights[:3]) - np.mean(heights[-3:])) / max(np.mean(heights[:3]), 1.0)
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

    # 1. رصد ضربة الشمس (Sunstroke): مشي بطيء جداً، ترنح خفيف، وانحناء قليل في البنية
    if f["speed_mean"] < 0.015 and 0.015 < f["speed_jitter"] < 0.04 and f["aspect_curr"] > 0.6:
        return "sunstroke_heat_exhaustion", min(0.96, 0.75 + 0.2 * s)

    # 2. رصد السقوط المفاجئ والتشنج
    if (f["aspect_curr"] > 1.05 and f["h_drop"] > 0.25 * (1.1 - 0.3 * s)):
        if f["speed_jitter"] > 0.03:
            return "sudden_fall_followed_by_seizure", min(0.98, 0.80 + 0.15 * s)
        return "sudden_fall", min(0.95, 0.78 + 0.18 * s)

    # 3. الاستلقاء بدون حركة
    if f["aspect_curr"] > 1.10 and f["displacement"] < 0.20:
        return "lying_immobile", min(0.96, 0.80 + 0.15 * s)

    # 4. العرج والإجهاد
    if f["speed_jitter"] > 0.035:
        if f["speed_jitter"] > 0.07:
            return "irregular_limping", min(0.90, 0.60 + f["speed_jitter"] * 3.0)
        return "regular_limping", min(0.88, 0.55 + f["speed_jitter"] * 3.5)

    if f["speed_mean"] < 0.03 and f["h_drop"] > 0.08:
        return "Exhausted_walking", min(0.85, 0.60 + 0.2 * s)

    return None, 0.0


# ============================================================================
# OPENCV ENGINE
# ============================================================================

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
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if state["prev_gray"] is None:
        state["prev_gray"] = gray
        return canvas, [], 1

    frame_diff = cv2.absdiff(state["prev_gray"], gray)
    state["prev_gray"] = gray

    _, thresh = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for c in contours:
        if cv2.contourArea(c) < 400:
            continue
        x, y, w, h = cv2.boundingRect(c)
        detections.append(((x + w / 2, y + h / 2), (x, y, w, h)))

    # ضامن التتبع لضمان استمرار تحرك البث حتى مع بطء الحركة
    if not detections:
        h_f, w_f, _ = frame.shape
        detections.append(((w_f / 2, h_f / 2), (int(w_f * 0.35), int(h_f * 0.2), int(w_f * 0.3), int(h_f * 0.6))))

    assigned = set()
    for (cx, cy), (x, y, w, h) in detections:
        best_id, best_d = None, 180.0
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

    for tid in [t for t, obj in state["tracks"].items() if frame_idx - obj.last_seen > 15]:
        del state["tracks"][tid]

    new_alerts = []
    active_count = 0

    for tid, tr in state["tracks"].items():
        active_count += 1
        x, y, w, h = tr.bbox
        f = extract_features(tr)

        is_abnormal = False
        if f:
            cond, conf = classify_taxonomy(f, sensitivity)
            if cond and cond in TAXONOMY_RULES:
                is_abnormal = True
                info = TAXONOMY_RULES[cond]
                color = (0, 0, 255) # أحمر بارز للحالات الحرجة
                tag = f"⚠️ {info['ar']} ({conf*100:.0f}%)"

                last_f = state["global_cd"].get(cond, -9999)
                if frame_idx - last_f > 120:
                    state["global_cd"][cond] = frame_idx
                    new_alerts.append((cond, conf))

        # رسم Bounding Box بارز وأحمر حصرًا عند رصد حالة Abnormal
        if is_abnormal:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 3)
            (text_w, text_h), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            cv2.rectangle(canvas, (x, max(y - 25, 0)), (x + text_w + 10, max(y, 25)), color, -1)
            cv2.putText(canvas, tag, (x + 5, max(y - 7, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas, new_alerts, active_count


# ============================================================================
# STREAMLIT UI & CONTROL
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
        ["رفع فيديو مراقبة (Upload Video)"],
    )

    uploaded_vid = st.file_uploader("اختر مقطع الكاميرا (.mp4)", type=["mp4", "avi", "mov"])

    st.markdown("---")
    selected_zone = st.selectbox("نطاق الكاميرا والموقع (Zone)", LOCATIONS)
    sens = st.slider("حساسية الرصد والاستجابة (Sensitivity)", 20, 100, 70)

    st.markdown("---")
    play_speed = st.slider("معدل العرض (FPS)", 6, 30, 16)
    max_f = st.slider("إجمالي الإطارات للفحص (Max Frames)", 80, 1000, 400, step=20)

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
        </div>
        """
    return html_out


def run_detection():
    if uploaded_vid is None:
        st.warning("الرجاء رفع ملف فيديو أولاً لتشغيل البث.")
        return

    st.session_state.alerts = []
    state = new_state()

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

    cap.release()
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
