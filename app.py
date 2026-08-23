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
# BASEER — AI EARLY MULTI-MODAL ANOMALY DETECTION & TRIAGE COMMAND CENTER
# ============================================================================

st.set_page_config(
    page_title="بصير | منصة الرصد والفرز المبكر الموحدة",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# THEME
# ============================================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;900&family=JetBrains+Mono:wght@500;700&display=swap');

    * {
        font-family: 'Tajawal', -apple-system, sans-serif;
    }

    code, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .block-container {
        padding-top: 1.2rem;
        max-width: 1440px;
    }

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

    .system-sub {
        color: #94A3B8;
        font-size: 0.88rem;
        margin: 0.2rem 0 0 0;
    }

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

    .kpi-num {
        font-size: 1.25rem;
        font-weight: 700;
        color: #38BDF8;
        font-family: 'JetBrains Mono', monospace;
    }

    .kpi-title {
        font-size: 0.72rem;
        color: #64748B;
        font-weight: 700;
    }

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

    .card-ar {
        font-size: 1.05rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 0.4rem;
    }

    .card-en {
        font-size: 0.82rem;
        color: #94A3B8;
        margin-bottom: 0.25rem;
    }

    .card-meta {
        color: #64748B;
        font-size: 0.78rem;
        font-family: 'JetBrains Mono', monospace;
    }

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

    .status-box {
        background: #0D1B2A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 0.7rem;
        color: #94A3B8;
        margin-bottom: 0.7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# MEDICAL / EMERGENCY TRIAGE RULES
# ============================================================================

TAXONOMY_RULES = {
    "heatstroke_exhaustion": {
        "category": "Medical & Respiratory Distress",
        "ar": "ضربة شمس حادة / إجهاد حراري وهبوط عام",
        "en": "heatstroke_exhaustion",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "☀️",
        "action": "توجيه فرقة إسعافية مع معدات التبريد ومحاليل الإرواء",
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
        "ar": "عرج شديد ومطرد / بوادر جفاف واختلال توازن",
        "en": "severe_gait_limping",
        "priority": "High",
        "color": "#F97316",
        "icon": "🚶",
        "action": "توجيه مسعف راجل ومساندة النقل الميداني",
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
        "ar": "استلقاء أرضي ممتد مع ضائقة تنفسية",
        "en": "severe_choking_on_ground",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "🫁",
        "action": "تأمين مجرى التنفس والتدخل الإسعافي الفوري",
    },
}

PRIORITY_COLOR = {
    "Critical": "#DC2626",
    "High": "#F97316",
    "Medium": "#F59E0B",
}


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
        self.history = deque(maxlen=40)
        self.age = 0
        self.update(centroid, bbox, frame_idx)

    def update(self, centroid, bbox, frame_idx):
        self.centroid = centroid
        self.bbox = bbox
        self.last_seen = frame_idx
        self.age += 1

        self.history.append(
            {
                "c": centroid,
                "b": bbox,
                "f": frame_idx,
            }
        )


# ============================================================================
# SESSION STATE
# ============================================================================

if "alerts" not in st.session_state:
    st.session_state.alerts = []

if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "frame": 0,
        "tracks": 0,
        "fps": 0.0,
        "time": 0.0,
        "processed": 0,
    }

if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False

if "running" not in st.session_state:
    st.session_state.running = False


# ============================================================================
# HEADER
# ============================================================================

st.markdown(
    """
    <div class="header-box">
        <div>
            <div class="system-title">
                🚑 نظام بصير | AI Medical Emergency Triage
            </div>

            <div class="system-sub">
                منظومة الرصد والفرز الذكي للمؤشرات الحيوية والحركية
                وإدارة بلاغات ضربات الشمس والسقوط في الحشود
            </div>
        </div>

        <div class="live-badge">
            ● LIVE DISPATCH SYSTEM
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:

    st.markdown("### 🎛️ غرفة العمليات والتحكم")
    st.caption("Operations Hub · Live Triage Parameters")

    feed_mode = st.radio(
        "مصدر البث (Feed Source)",
        ["Simulation", "Upload Video"],
        format_func=lambda x:
            "وضع المحاكاة الافتراضي (Simulation Mode)"
            if x == "Simulation"
            else "رفع فيديو مراقبة (Upload Video)",
    )

    uploaded_vid = None

    if feed_mode == "Upload Video":

        uploaded_vid = st.file_uploader(
            "اختر مقطع الكاميرا",
            type=["mp4", "avi", "mov", "mkv"],
        )

    st.markdown("---")

    selected_zone = st.selectbox(
        "نطاق الكاميرا والموقع (Zone)",
        LOCATIONS,
    )

    sens = st.slider(
        "حساسية الرصد والاستجابة (Sensitivity)",
        20,
        100,
        65,
    )

    st.markdown("---")

    max_f = st.slider(
        "إجمالي الإطارات للفحص (Max Frames)",
        60,
        2000,
        500,
        step=20,
    )

    st.markdown("---")

    st.markdown("### ⚡ إعدادات الأداء")

    processing_stride = st.selectbox(
        "معالجة كل كم إطار؟",
        [1, 2, 3, 4],
        index=1,
        help=(
            "1 = معالجة كل إطار. "
            "2 = معالجة إطار من كل إطارين. "
            "القيم الأكبر أسرع."
        ),
    )

    display_every = st.selectbox(
        "تحديث الشاشة كل كم إطار؟",
        [1, 2, 3, 5, 10],
        index=2,
        help=(
            "كلما زادت القيمة قل ضغط Streamlit "
            "وأصبح العرض أكثر سلاسة."
        ),
    )

    output_width = st.selectbox(
        "دقة التحليل",
        [480, 640, 800, 960],
        index=1,
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        start_btn = st.button(
            "▶ تشغيل الرصد",
            use_container_width=True,
            type="primary",
        )

    with col2:
        reset_btn = st.button(
            "⟲ إعادة ضبط",
            use_container_width=True,
        )

    stop_btn = st.button(
        "⏹ إيقاف المعالجة",
        use_container_width=True,
    )


# ============================================================================
# RESET / STOP
# ============================================================================

if reset_btn:

    st.session_state.alerts = []

    st.session_state.metrics = {
        "frame": 0,
        "tracks": 0,
        "fps": 0.0,
        "time": 0.0,
        "processed": 0,
    }

    st.session_state.stop_requested = False
    st.session_state.running = False

    st.rerun()


if stop_btn:
    st.session_state.stop_requested = True


# ============================================================================
# MAIN LAYOUT
# ============================================================================

col_cam, col_triage = st.columns([1.35, 1])


with col_cam:

    st.markdown(
        "##### 📹 البث التحليلي المباشر (Analytical Feed)"
    )

    cam_holder = st.empty()

    status_holder = st.empty()

    kpi_holder = st.empty()


with col_triage:

    st.markdown(
        "##### 🚨 سجل الفرز والتوجيه الميداني (Live Triage Log)"
    )

    triage_holder = st.container()


# ============================================================================
# KPI RENDERING
# ============================================================================

def render_kpis(m):

    kpi_holder.markdown(
        f"""
        <div class="kpi-container">

            <div class="kpi-card">
                <div class="kpi-num">{m['frame']}</div>
                <div class="kpi-title">الإطار (Frame)</div>
            </div>

            <div class="kpi-card">
                <div class="kpi-num">{m['time']:.1f}s</div>
                <div class="kpi-title">الزمن (Time)</div>
            </div>

            <div class="kpi-card">
                <div class="kpi-num">{m['tracks']}</div>
                <div class="kpi-title">الأشخاص (Active)</div>
            </div>

            <div class="kpi-card">
                <div class="kpi-num">{m['fps']:.1f}</div>
                <div class="kpi-title">المعالجة (FPS)</div>
            </div>

            <div class="kpi-card">
                <div class="kpi-num" style="color:#EF4444">
                    {len(st.session_state.alerts)}
                </div>
                <div class="kpi-title">البلاغات (Alerts)</div>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# TRIAGE LOG
# ============================================================================

def render_triage():

    triage_holder.empty()

    with triage_holder:

        if not st.session_state.alerts:

            st.info(
                "لا توجد بلاغات إسعافية حرجة حتى الآن. "
                "النظام يعمل ويراقب المؤشرات الحركية..."
            )

            return

        for idx, alert in enumerate(
            reversed(st.session_state.alerts)
        ):

            info = TAXONOMY_RULES.get(
                alert.condition_key,
                TAXONOMY_RULES["sudden_fall"],
            )

            b_color = PRIORITY_COLOR[
                info["priority"]
            ]

            st.markdown(
                f"""
                <div class="alert-card"
                     style="border-left: 6px solid {b_color};">

                    <div style="
                        display:flex;
                        justify-content:space-between;
                        align-items:center;
                    ">

                        <span class="triage-badge"
                              style="background:{b_color}">

                            {info['priority']} PRIORITY

                        </span>

                        <span class="card-meta">

                            #{alert.id}
                            · {alert.wall_clock}
                            · t={alert.video_time_s:.1f}s

                        </span>

                    </div>

                    <div style="margin-top:0.3rem;">

                        <span class="category-tag">

                            📂 {info['category']}

                        </span>

                    </div>

                    <div class="card-ar">

                        {info['icon']} {info['ar']}

                    </div>

                    <div class="card-en">

                        <b>Class:</b>
                        <code>{info['en']}</code>

                        (الثقة:
                        {alert.confidence * 100:.0f}%)

                    </div>

                    <div class="card-meta">

                        📍 {alert.location}

                    </div>

                    <div style="
                        margin-top:0.4rem;
                        font-size:0.8rem;
                        color:#CBD5E1;
                    ">

                        <b>الإجراء الموصى به:</b>
                        {info['action']}

                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            b1, b2 = st.columns([1.3, 1])

            with b1:

                if not alert.dispatched:

                    if st.button(
                        "🚑 توجيه فرقة التدخل السريع",
                        key=f"btn_dsp_{alert.unique_key}_{idx}",
                        type="primary",
                    ):

                        alert.dispatched = True

                        st.rerun()

                else:

                    st.button(
                        "✅ تم توجيه الفرقة بنجاح",
                        key=f"btn_done_{alert.unique_key}_{idx}",
                        disabled=True,
                    )

            with b2:

                if alert.dispatched:

                    st.markdown(
                        """
                        <div class="eta-box">
                            🚨 الفرقة في الطريق
                            (وصول: دقيقة ونصف)
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.write("")


# ============================================================================
# ALERT CREATION
# ============================================================================

def add_alert(
    condition_key,
    confidence,
    frame_idx,
    fps_src,
    selected_zone,
):

    # Limit alert history so it doesn't grow forever.
    MAX_ALERTS = 100

    seq_num = len(st.session_state.alerts) + 1

    alert = Alert(
        id=f"EMS-{seq_num:03d}",

        unique_key=(
            f"{seq_num}_"
            f"{frame_idx}_"
            f"{int(time.time() * 1000)}"
        ),

        frame_idx=frame_idx,

        video_time_s=(
            frame_idx / max(fps_src, 1)
        ),

        wall_clock=datetime.now().strftime(
            "%H:%M:%S"
        ),

        location=selected_zone,

        condition_key=condition_key,

        confidence=confidence,
    )

    st.session_state.alerts.append(alert)

    if len(st.session_state.alerts) > MAX_ALERTS:
        st.session_state.alerts = (
            st.session_state.alerts[-MAX_ALERTS:]
        )


# ============================================================================
# SIMULATION FRAME
# ============================================================================

def generate_simulation_frame(
    frame_idx,
    w,
    h,
):
    """
    Generates the same deterministic demonstration
    scenario used in the original application.
    """

    phases = [
        (
            "Normal Walk",
            "normal_walk",
            45,
        ),

        (
            "Severe Heatstroke Symptoms",
            "heatstroke_exhaustion",
            55,
        ),

        (
            "Pre-Collapse Stoop",
            "stooped_walking_resting",
            45,
        ),

        (
            "Sudden Fall Event",
            "sudden_fall",
            45,
        ),

        (
            "Immobilized",
            "severe_choking_on_ground",
            50,
        ),
    ]

    total_cycle = sum(
        p[2] for p in phases
    )

    curr_t = frame_idx % total_cycle

    accum = 0
    curr_cond = "normal_walk"
    prog_phase = 0.0

    for _, cond, dur in phases:

        if accum <= curr_t < accum + dur:

            curr_cond = cond

            prog_phase = (
                (curr_t - accum) / dur
            )

            break

        accum += dur

    canvas = np.full(
        (h, w, 3),
        (15, 23, 42),
        dtype=np.uint8,
    )

    ground_y = h - 70

    cv2.line(
        canvas,
        (0, ground_y),
        (w, ground_y),
        (51, 65, 85),
        3,
    )

    if curr_cond == "normal_walk":

        bw, bh = 48, 140

        cx = int(
            w * 0.2
            + prog_phase * w * 0.3
        )

        cy = int(
            ground_y - bh / 2
        )

    elif curr_cond == "heatstroke_exhaustion":

        bw, bh = 54, 130

        cx = int(
            w * 0.5
            + math.sin(prog_phase * 20) * 16
        )

        cy = int(
            ground_y - bh / 2
        )

    elif curr_cond == "stooped_walking_resting":

        bw = int(
            55 + prog_phase * 20
        )

        bh = int(
            120 - prog_phase * 40
        )

        cx = int(w * 0.55)

        cy = int(
            ground_y - bh / 2
        )

    elif curr_cond == "sudden_fall":

        fall_t = min(
            1.0,
            prog_phase / 0.4,
        )

        bw = int(
            55 + fall_t * 85
        )

        bh = int(
            120 - fall_t * 90
        )

        cx = int(w * 0.58)

        cy = int(
            ground_y - bh / 2
        )

    else:

        bw, bh = 140, 32

        cx = int(w * 0.58)

        cy = int(
            ground_y - 18
        )

    cv2.ellipse(
        canvas,
        (cx, cy),
        (
            max(int(bw / 2), 6),
            max(int(bh / 2), 6),
        ),
        0,
        0,
        360,
        (56, 189, 248),
        -1,
    )

    if bh > 40:

        cv2.circle(
            canvas,
            (
                cx,
                cy - int(bh / 2) + 12,
            ),
            14,
            (125, 211, 252),
            -1,
        )

    bbox = (
        cx - bw / 2,
        cy - bh / 2,
        bw,
        bh,
    )

    return (
        canvas,
        curr_cond,
        cx,
        cy,
        bw,
        bh,
        bbox,
    )


# ============================================================================
# SIMULATION ANALYSIS
# ============================================================================

def analyze_simulation_frame(
    canvas,
    curr_cond,
    cx,
    cy,
    bw,
    bh,
    frame_idx,
    global_cd,
):

    evts = []

    if curr_cond in TAXONOMY_RULES:

        last_f = global_cd.get(
            curr_cond,
            -9999,
        )

        if frame_idx - last_f > 50:

            global_cd[curr_cond] = frame_idx

            evts.append(
                (
                    curr_cond,
                    0.92,
                )
            )

    b_color = (
        (40, 40, 235)
        if curr_cond in TAXONOMY_RULES
        else
        (40, 200, 100)
    )

    tag = (
        f"Abnormal: {curr_cond}"
        if curr_cond in TAXONOMY_RULES
        else
        "ID 1 - Normal"
    )

    cv2.rectangle(
        canvas,
        (
            int(cx - bw / 2),
            int(cy - bh / 2),
        ),
        (
            int(cx + bw / 2),
            int(cy + bh / 2),
        ),
        b_color,
        2,
    )

    cv2.putText(
        canvas,
        tag,
        (
            int(cx - bw / 2),
            max(
                int(cy - bh / 2) - 8,
                20,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        b_color,
        2,
        cv2.LINE_AA,
    )

    return evts


# ============================================================================
# REAL VIDEO ANALYSIS
# ============================================================================

def analyze_real_frame(
    raw,
    bg,
    global_cd,
    frame_idx,
    sens,
):

    evts = []

    # Background subtraction
    fgmask = bg.apply(raw)

    # Sensitivity affects the threshold.
    threshold_value = int(
        np.clip(
            250 - sens * 0.5,
            180,
            240,
        )
    )

    _, fgmask = cv2.threshold(
        fgmask,
        threshold_value,
        255,
        cv2.THRESH_BINARY,
    )

    # Remove small noise.
    kernel = np.ones(
        (3, 3),
        np.uint8,
    )

    fgmask = cv2.morphologyEx(
        fgmask,
        cv2.MORPH_OPEN,
        kernel,
    )

    contours, _ = cv2.findContours(
        fgmask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    active_count = 0

    # Sensitivity changes minimum contour area.
    min_area = max(
        1000,
        int(
            4000
            - sens * 25
        ),
    )

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area <= min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(
            contour
        )

        active_count += 1

        asp = bw / max(
            bh,
            1,
        )

        # Simple heuristic from original code.
        if asp > 1.1:
            cond = "sudden_fall"
        else:
            cond = "severe_gait_limping"

        last_f = global_cd.get(
            cond,
            -9999,
        )

        # Cooldown prevents an alert on every frame.
        if frame_idx - last_f > 75:

            global_cd[cond] = frame_idx

            evts.append(
                (
                    cond,
                    0.88,
                )
            )

        # Bounding box.
        box_color = (
            40,
            40,
            235,
        )

        cv2.rectangle(
            raw,
            (x, y),
            (
                x + bw,
                y + bh,
            ),
            box_color,
            2,
        )

        cv2.putText(
            raw,
            f"Abnormal: {cond}",
            (
                x,
                max(
                    y - 8,
                    16,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            2,
            cv2.LINE_AA,
        )

    return evts, active_count


# ============================================================================
# MAIN PROCESSING ENGINE
# ============================================================================

def execute_analysis():

    st.session_state.alerts = []

    st.session_state.stop_requested = False

    st.session_state.running = True

    # ---------------------------------------------------------
    # PROCESSING CONFIG
    # ---------------------------------------------------------

    is_sim = (
        feed_mode == "Simulation"
    )

    w = int(output_width)

    # Keep aspect ratio 16:10-ish like original 640x400.
    h = int(
        w * 400 / 640
    )

    fps_src = 25.0

    cap = None
    temp_path = None

    # ---------------------------------------------------------
    # VIDEO INPUT
    # ---------------------------------------------------------

    if not is_sim:

        if uploaded_vid is None:

            st.warning(
                "الرجاء رفع ملف فيديو أولاً."
            )

            st.session_state.running = False

            return

        # Save uploaded file once.
        suffix = os.path.splitext(
            uploaded_vid.name
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as tfile:

            tfile.write(
                uploaded_vid.getbuffer()
            )

            temp_path = tfile.name

        cap = cv2.VideoCapture(
            temp_path
        )

        if not cap.isOpened():

            st.error(
                "تعذر فتح ملف الفيديو."
            )

            st.session_state.running = False

            return

        fps_src = (
            cap.get(
                cv2.CAP_PROP_FPS
            )
            or 25.0
        )

        v_total = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
            or max_f
        )

        total_frames = min(
            max_f,
            v_total,
        )

    else:

        total_frames = max_f

    # ---------------------------------------------------------
    # DETECTION STATE
    # ---------------------------------------------------------

    bg = cv2.createBackgroundSubtractorMOG2(
        history=300,
        varThreshold=45,
        detectShadows=False,
    )

    tracks = {}

    next_id = 1

    global_cd = {}

    # ---------------------------------------------------------
    # UI PLACEHOLDERS
    # ---------------------------------------------------------

    progress = st.progress(
        0.0,
        text="جاري بدء التحليل...",
    )

    start_time = time.time()

    processed_count = 0

    active_count = 0

    last_display_time = start_time

    # ---------------------------------------------------------
    # FRAME LOOP
    # ---------------------------------------------------------

    for frame_idx in range(
        1,
        total_frames + 1,
    ):

        # -----------------------------------------------------
        # STOP CHECK
        # -----------------------------------------------------

        if st.session_state.stop_requested:

            status_holder.warning(
                "⏹ تم إيقاف المعالجة."
            )

            break

        # -----------------------------------------------------
        # READ FRAME
        # -----------------------------------------------------

        if is_sim:

            (
                canvas,
                curr_cond,
                cx,
                cy,
                bw,
                bh,
                bbox,
            ) = generate_simulation_frame(
                frame_idx,
                w,
                h,
            )

            # Update simple tracking.
            tid = 1

            if tid not in tracks:

                tracks[tid] = Track(
                    tid,
                    (cx, cy),
                    bbox,
                    frame_idx,
                )

            else:

                tracks[tid].update(
                    (cx, cy),
                    bbox,
                    frame_idx,
                )

            evts = analyze_simulation_frame(
                canvas,
                curr_cond,
                cx,
                cy,
                bw,
                bh,
                frame_idx,
                global_cd,
            )

            active_count = 1

            frame_rgb = cv2.cvtColor(
                canvas,
                cv2.COLOR_BGR2RGB,
            )

        else:

            ok, raw = cap.read()

            if not ok or raw is None:
                break

            # -------------------------------------------------
            # PROCESSING STRIDE
            # -------------------------------------------------

            # We still read every frame to keep the video
            # timeline correct, but detection can run only
            # on selected frames.
            should_analyze = (
                frame_idx % processing_stride == 0
                or frame_idx == 1
            )

            raw = cv2.resize(
                raw,
                (w, h),
                interpolation=cv2.INTER_AREA,
            )

            if should_analyze:

                evts, active_count = (
                    analyze_real_frame(
                        raw,
                        bg,
                        global_cd,
                        frame_idx,
                        sens,
                    )
                )

            else:

                evts = []

            frame_rgb = cv2.cvtColor(
                raw,
                cv2.COLOR_BGR2RGB,
            )

        # -----------------------------------------------------
        # ALERTS
        # -----------------------------------------------------

        for cond, conf in evts:

            add_alert(
                condition_key=cond,
                confidence=conf,
                frame_idx=frame_idx,
                fps_src=fps_src,
                selected_zone=selected_zone,
            )

        processed_count += 1

        # -----------------------------------------------------
        # METRICS
        # -----------------------------------------------------

        elapsed = max(
            time.time() - start_time,
            0.001,
        )

        current_fps = (
            processed_count / elapsed
        )

        current_video_time = (
            frame_idx / max(
                fps_src,
                1,
            )
        )

        st.session_state.metrics = {
            "frame": frame_idx,
            "tracks": (
                len(tracks)
                if is_sim
                else active_count
            ),
            "fps": current_fps,
            "time": current_video_time,
            "processed": processed_count,
        }

        # -----------------------------------------------------
        # DISPLAY
        # -----------------------------------------------------

        # IMPORTANT:
        # We don't send every frame through Streamlit.
        # This is one of the biggest performance improvements.
        if (
            frame_idx % display_every == 0
            or frame_idx == 1
            or frame_idx == total_frames
        ):

            cam_holder.image(
                frame_rgb,
                channels="RGB",
                use_container_width=True,
            )

            render_kpis(
                st.session_state.metrics
            )

            render_triage()

            progress.progress(
                frame_idx / max(
                    total_frames,
                    1,
                ),
                text=(
                    f"تحليل الإطارات..."
                    f" {frame_idx}/{total_frames}"
                ),
            )

            # Don't force 30 UI updates per second.
            now = time.time()

            if now - last_display_time > 0.5:

                status_holder.markdown(
                    f"""
                    <div class="status-box">

                    🟢 <b>النظام يعمل</b>
                    &nbsp; | &nbsp;
                    Frame: {frame_idx}
                    &nbsp; | &nbsp;
                    FPS: {current_fps:.1f}
                    &nbsp; | &nbsp;
                    Alerts: {len(st.session_state.alerts)}

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                last_display_time = now

    # ---------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------

    if cap is not None:

        cap.release()

    if temp_path is not None:

        try:
            os.remove(temp_path)
        except OSError:
            pass

    progress.empty()

    total_elapsed = max(
        time.time() - start_time,
        0.001,
    )

    final_fps = (
        processed_count
        / total_elapsed
    )

    st.session_state.metrics[
        "fps"
    ] = final_fps

    st.session_state.running = False

    # ---------------------------------------------------------
    # FINAL STATUS
    # ---------------------------------------------------------

    if st.session_state.stop_requested:

        status_holder.warning(
            f"""
            ⏹ تم إيقاف المعالجة.

            تمت معالجة {processed_count}
            إطار بسرعة {final_fps:.1f} FPS.
            """
        )

    else:

        status_holder.success(
            f"""
            ✅ اكتملت المعالجة.

            تمت معالجة {processed_count}
            إطار بسرعة {final_fps:.1f} FPS.
            """
        )

    render_kpis(
        st.session_state.metrics
    )

    render_triage()


# ============================================================================
# START
# ============================================================================

if start_btn:

    execute_analysis()


# ============================================================================
# INITIAL SCREEN
# ============================================================================

if not start_btn:

    placeholder = np.full(
        (400, 640, 3),
        (11, 19, 43),
        dtype=np.uint8,
    )

    cv2.putText(
        placeholder,
        "BASEER MULTI-MODAL TRIAGE",
        (120, 195),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (72, 202, 228),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        placeholder,
        "اضغط بدء الرصد للتشغيل الميداني",
        (160, 235),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (144, 224, 239),
        1,
        cv2.LINE_AA,
    )

    cam_holder.image(
        placeholder,
        channels="RGB",
        use_container_width=True,
    )

    render_kpis(
        st.session_state.metrics
    )

    render_triage()
