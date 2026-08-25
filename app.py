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

FIX NOTE 5 (playback speed — video was running slower than real time):
The producer was pacing purely off a fixed render-fps target (e.g. 12
fps) regardless of the source clip's own fps. If the source was 25-30
fps, pushing only 12 frames/sec meant the on-screen playback took
2-2.5x longer than the clip's real duration — technically smooth, but
in slow motion. Fixed by reading the source video's real fps and
DECIMATING (skipping frames evenly) down to the render rate while
pacing sleeps against the *original* frame timeline, so total playback
wall-time now matches the clip's real duration.

FIX NOTE 6 (false alarms from blob merges + missed a real heat-exhaustion case):
Three more changes target this directly:
1) Track jump-rejection: in a crowd, two people's motion blobs
   frequently merge then split, which can snap a track's centroid a
   large, physically-implausible distance in one step. That single
   jump used to read as huge "jitter"/"speed" and could immediately
   trigger fighting/limping. A track whose centroid moves further than
   is plausible for one frame now has its short-term history reset
   instead of feeding that jump into the motion features.
2) Small/partial detections are excluded from alert eligibility (see
   FIX NOTE 7) — most reported false alarms were on small, likely
   partial blobs, and those are the least reliable signal anyway.
3) The heat-exhaustion / sunstroke rule was too narrow — it required
   speed_jitter inside a specific band AND aspect_curr > 0.55, which a
   person standing mostly still (little to no sway) with a normal
   upright silhouette can easily fail. Replaced with a broader "barely
   moving, upright posture" rule keyed on total displacement instead of
   a jitter band, so both a swaying and a nearly-motionless heat-fatigued
   person are caught.

FIX NOTE 7 (bounding boxes too small / imprecise):
Two changes:
1) `refine_person_box` (the HOG-based tightening step) could return a
   box much smaller than the actual motion blob when HOG only picked
   up part of a person (e.g. torso only) — that undersized box was a
   recurring source of visibly-wrong-looking alerts. It's now
   sanity-bounded: a refined box that's drastically smaller than the
   original motion bbox is rejected and the original is used instead.
2) Every drawn alert box now gets a fixed padding margin added around
   it, so boxes read clearly on screen instead of hugging (or cutting
   into) the person.

FIX NOTE 8 (alert boxes appearing on inanimate/static objects):
The low-motion heat-emergency rule ("barely moving, upright aspect
ratio") is, by construction, the condition most easily satisfied by a
non-person foreground blob: a static object left in the scene, a
parked item, or a shadow/reflection that has stabilized is *literally*
motionless — which trivially passes speed_mean/displacement thresholds
that were designed to detect a real person standing still. Nothing in
the pipeline previously required the blob to actually look like a
person before firing one of these conditions (the HOG check only ran
AFTER confirmation, purely to tighten the drawn box — if HOG found no
person it silently kept the raw motion box instead of rejecting the
alert). Fix: `verify_person_present()` now GATES confirmation of the
heat-emergency conditions specifically — it runs OpenCV's pedestrian
HOG detector on the candidate box and the alert is only allowed to fire
if HOG actually finds a person-shaped silhouette there. If it doesn't,
the track's confirmation state is reset (so it has to re-accumulate
consecutive frames AND re-pass this check, rather than firing on the
next frame regardless).

FIX NOTE 9 (sunstroke vs. heat exhaustion conflated into one alert):
`sunstroke_heat_exhaustion` used to cover both "person is essentially
motionless" (a real collapse/fainting risk) and "person is showing only
mild sway/low activity" (an early-warning sign) under a single
condition and a single Critical priority. Split into two distinct
taxonomy entries with two distinct thresholds and two distinct
priorities:
  - `sunstroke_fainting` (Critical) — near-zero movement, the stronger
    signal, closer to an actual collapse.
  - `suspected_heat_exhaustion` (High) — mild activity/sway, an
    early-warning tier rather than an emergency-collapse tier.
The triage panel is also now sorted by clinical priority first
(Critical -> High -> Medium -> Low), then by recency within a tier, so
`sunstroke_fainting` always surfaces above `suspected_heat_exhaustion`,
which in turn surfaces above routine gait/fatigue alerts like limping —
instead of pure reverse-chronological order, which could bury a
Critical alert under a run of Medium ones.

FIX NOTE 10 (non-person surfaces still triggering alerts — a wall, an
on-screen graphic/watermark from editing):
`verify_person_present()` (FIX NOTE 8) was only being required for the
two heat-emergency conditions. That was too narrow: a static wall
segment (lit up by compression noise / a moving shadow / camera
micro-jitter) or a burned-in editing overlay (a logo, lower-third,
timestamp) can just as easily land inside the aspect-ratio band for
"regular_limping" or any other condition as it can for "barely moving,
upright" — none of that is specific to the heat rule. The gate is now
required for EVERY condition before an alert is allowed to confirm, not
just the heat ones — a track only gets to fire an alert at all once
OpenCV's pedestrian HOG detector actually finds a person-shaped
silhouette in its box.

FIX NOTE 11 (a person who deliberately sat/lay down to rest was
classified as "sudden fall followed by seizure" — no seizure occurred):
Two problems compounded here:
1) The fall rule only looked at the NET height drop across the whole
   history window (first-3-avg vs last-3-avg), with no requirement that
   the drop be fast. A person lowering themselves deliberately to sit
   or rest produces the same net height change as an actual collapse,
   just spread over more frames — so it passed the same "h_drop"
   threshold as a real fall. Added `drop_rate`: the single largest
   frame-to-frame height decrease (normalized for decimation), which is
   high for an abrupt collapse and low for a controlled, gradual
   descent. A fall is now only classified as "sudden" when the drop
   itself is fast, not merely when the end state is lower.
2) The seizure sub-classification reused `speed_jitter`, which is
   horizontal-only and computed over the SAME window that includes the
   fall/sit-down transition itself — so the ordinary postural wobble of
   controlledly lowering yourself was indistinguishable from a tremor.
   Replaced with `tremor`: a two-axis (horizontal + vertical) jitter
   measured only from the most recent samples (i.e. after the person is
   already down, not during the transition), which is what an actual
   convulsive movement looks like and a settled, still person does not.

FIX NOTE 12 (alert boxes far too large — usually on inanimate objects):
A real person's motion blob is bounded by how big a person can actually
be in frame; the oversized boxes were consistently on non-person cases
(a lit-up wall segment, a large lighting-change/shadow region, a big
static area absorbed unevenly by the background model) that can span a
much larger area than any single person ever would. Two caps were
added:
1) At detection time, a contour is now rejected — never even becomes a
   track — if its area, height, or width exceeds a plausible
   single-person size relative to the frame (`MAX_CONTOUR_AREA_RATIO`,
   `MAX_PERSON_HEIGHT_RATIO`, `MAX_PERSON_WIDTH_RATIO`). This is the
   main fix, since it stops an oversized blob from ever being tracked
   or classified at all.
2) As a safety net, a track is also excluded from alert eligibility if
   its box has since grown past the same height limit (mirrors the
   existing `MIN_ALERT_HEIGHT_PX` floor with a matching ceiling), and
   `pad_box()` now caps its added margin in pixels so the padding step
   itself can't balloon an already-large box further.

FIX NOTE 13 (still alerting on inanimate objects; wants liveness checked
BEFORE anything else, tracked continuously the whole time an object is
in frame):
Two changes:
1) `verify_person_present()` used to run ONCE, at the exact instant a
   track's motion pattern finished its consecutive-frame confirmation —
   a single HOG snapshot, which a patterned/textured surface (a
   railing, a tiled wall, a gate) can occasionally pass by coincidence.
   Person-liveness is now sampled repeatedly across the track's ENTIRE
   time in view (throttled to a handful of samples for cost, via
   `Track.person_checks`/`person_hits`), and a track only gets to fire
   an alert once a required minimum number of those samples have come
   back positive by a required majority (`PERSON_CHECK_MIN_SAMPLES`,
   `PERSON_HIT_RATIO_REQUIRED`) — i.e. the object has to keep looking
   like a person across many separate looks, not just one lucky frame.
   If the sample budget runs out without a confident majority, the
   pending classification is dropped instead of firing.
2) `verify_person_present()` also now requires a minimum HOG confidence
   score, not just the presence of any detection — a low-confidence hit
   on a repetitive texture no longer counts as "found a person" on its
   own; combined with (1), it now has to happen consistently.

FIX NOTE 14 (a person's track was dying and restarting mid-frame):
Tracks used to be deleted the moment they went 2 frames without a
matched detection, and re-matching was pure nearest-centroid at a
fixed pixel radius. In a crowd, a brief occlusion, a temporary blob
merge/split, or one frame where the foreground mask fragmented was
enough to kill the track — and when the same person reappeared a
moment later they got a brand-new track ID, resetting `history`,
confirmation state, AND all the accumulated person-liveness evidence
from FIX NOTE 13. That's the opposite of "track this person the whole
time they're in frame". Fixed with:
1) `Track.missed`: instead of deleting on the first gap, a track now
   "coasts" — stays alive, keeping all of its state — for up to
   `MAX_MISSED_FRAMES` frames with no matched detection, and is only
   dropped once it's really been gone that long.
2) `Track.predicted_centroid()`: while coasting, matching uses the
   track's last known velocity to predict where it should be next,
   instead of just its last-seen position, so a detection that
   reappears a few pixels away (as expected motion, not a jump) is
   still re-matched to the SAME track/ID rather than spawning a new one.
3) The matching radius now scales with the track's own height (a
   person-relative distance, like the jump-rejection check already
   used) instead of a flat pixel constant, so small/close and
   large/far tracks are matched with sensible tolerances instead of
   one-size-fits-all.

FIX NOTE 15 (a real fallen/lying person could fail the person-verification
gate and have their alert silently dropped):
`verify_person_present()` (FIX NOTE 8/10/13) relies on OpenCV's default
pedestrian HOG detector, which is trained on upright, standing/walking
silhouettes. A person who has actually fallen or is lying immobile —
exactly the cases `sudden_fall`, `lying_immobile`, and
`sunstroke_fainting` exist to catch — produces a wide/flat box that the
detector was never trained to recognize, so the liveness gate could
reject a genuine emergency as "not a person" and silently drop it via
`reject_pending()`. Fixed by adding a second HOG pass, tried only for
wide/flat boxes (width >= ~0.9x height, i.e. the shape a lying person
actually produces): the crop is rotated 90 degrees before detection, so
a horizontal silhouette presents to the detector the way an upright one
normally would. This keeps the anti-inanimate-object gate just as
strict for actual static objects while no longer penalizing the exact
poses the most critical alerts depend on.

FIX NOTE 16 (sunstroke vs. heat exhaustion weren't split on the right
clinical signal): the two heat conditions used to be distinguished
purely by HOW LITTLE the person moved (near-still -> sunstroke, mild
sway -> heat exhaustion). That's not the actual clinical distinction
requested: sunstroke (ضربة شمس) is defined by NEUROLOGICAL signs —
delirium, loss of consciousness, or convulsions — while heat exhaustion
(إجهاد حراري) is a MUSCULAR/FATIGUE sign — an unsteady gait/limp that
may end in a stumble or fall, but the person stays conscious and mobile
through it. Reworked `classify_taxonomy`'s heat-emergency section
end-to-end around that distinction:
1) Two new per-track features in `extract_features`: `tail_speed`
   (activity in only the most recent samples — used to tell "fell and
   kept moving" apart from "fell and went still") and
   `direction_reversals` (how often horizontal direction flips relative
   to real net progress — a proxy for confused/disoriented wandering
   vs. purposeful, if unsteady, movement).
2) `sunstroke_fainting` (Critical) now fires on any of three
   neurological signs: a genuine tremor/convulsion, near-total stillness
   (loss of consciousness), or high direction-reversal with little net
   progress (delirium) — checked in that order since they're
   increasingly less specific and the more urgent pattern should win if
   more than one happens to overlap.
3) `suspected_heat_exhaustion` (High) now specifically requires a limp
   signature (elevated speed_jitter) combined with an abrupt height-drop
   (a stumble/fall) AND continued post-drop movement (`tail_speed` above
   a "still conscious" floor) — i.e. limped, stumbled, kept moving. If
   the person instead went still after the stumble, that's caught by the
   `sunstroke_fainting` near-stillness check above instead, since staying
   conscious is exactly what's supposed to separate the two tiers.
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
    "sunstroke_fainting": {
        "category": "Heat Emergencies & Insolation",
        "title": "Sunstroke / Heat-Induced Fainting",
        "en": "sunstroke_fainting",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "☀️",
        "action": "Immediate evacuation to cooling shelter & emergency medical response.",
    },
    "suspected_heat_exhaustion": {
        "category": "Heat Emergencies & Insolation",
        "title": "Suspected Heat Exhaustion (Early Signs)",
        "en": "suspected_heat_exhaustion",
        "priority": "High",
        "color": "#F97316",
        "icon": "🌡️",
        "action": "Dispatch field medic for hydration check & shaded rest; monitor closely.",
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
# FIX NOTE 9: explicit rank so the triage panel can sort Critical first,
# then High, then Medium/Low — instead of pure reverse-chronological.
PRIORITY_RANK = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}

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
        # FIX NOTE 14: consecutive frames with no matched detection. The
        # track stays alive ("coasts") while this is below
        # MAX_MISSED_FRAMES, instead of being deleted on the first gap.
        self.missed = 0
        # Temporal-confirmation state (FIX NOTE 4): a classification only
        # "counts" once the SAME condition has been seen on several
        # consecutive frames for this track.
        self.pending_cond = None
        self.pending_count = 0
        # FIX NOTE 13: person-liveness evidence, accumulated across the
        # WHOLE time this track has existed in frame — not just checked
        # once at the instant a classification happens to confirm. See
        # process_video_frame for how this is gathered and required.
        self.person_checks = 0
        self.person_hits = 0
        # A motion-pattern classification that has finished its
        # consecutive-frame confirmation and is now just waiting on
        # enough accumulated person-evidence before it's allowed to fire.
        self.awaiting_cond = None
        self.awaiting_conf = 0.0
        # The condition that has actually fired (motion-confirmed AND
        # person-verified) and is still ongoing this frame.
        self.confirmed_cond = None
        self.update(centroid, bbox, frame_idx)

    def update(self, centroid, bbox, frame_idx):
        # FIX NOTE 14: a real detection matched this frame -> no longer
        # coasting.
        self.missed = 0

        # FIX NOTE 6.1: reject implausible one-frame jumps (usually a
        # crowd blob merging/splitting into a different track) instead of
        # letting them masquerade as real high-speed/jitter motion.
        if self.history:
            prev = self.history[-1]
            prev_h = max(prev["b"][3], 20.0)
            jump = math.hypot(centroid[0] - prev["c"][0], centroid[1] - prev["c"][1])
            if jump > 1.2 * prev_h:
                self.history.clear()
                self.pending_cond = None
                self.pending_count = 0
                # FIX NOTE 13: a jump this large usually means the track
                # identity itself is questionable (blob merge/split) —
                # any liveness evidence gathered so far no longer
                # reliably describes "this" object, so start over.
                self.person_checks = 0
                self.person_hits = 0
                self.awaiting_cond = None
                self.confirmed_cond = None

        self.centroid = centroid
        self.bbox = bbox
        self.last_seen = frame_idx
        self.age += 1
        self.history.append({"c": centroid, "b": bbox, "f": frame_idx})

    def predicted_centroid(self):
        """FIX NOTE 14: while coasting (no detection this frame), predict
        where the track should be next using its last known velocity, so
        it can still be re-matched to a detection that reappears a few
        pixels away instead of only ever matching at its exact last-seen
        spot. Falls back to the last known centroid if there isn't
        enough history yet to estimate a velocity."""
        if len(self.history) < 2:
            return self.centroid
        c_now = self.history[-1]["c"]
        c_prev = self.history[-2]["c"]
        vx, vy = c_now[0] - c_prev[0], c_now[1] - c_prev[1]
        steps = self.missed + 1
        return (c_now[0] + vx * steps, c_now[1] + vy * steps)

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

    def reject_pending(self):
        """FIX NOTE 8: used when a confirmed classification fails the
        person-verification gate — reset so the track has to
        re-accumulate consecutive frames (and re-pass the gate) instead
        of firing again on the very next tick with a stale count."""
        self.pending_cond = None
        self.pending_count = 0
        self.awaiting_cond = None
        self.confirmed_cond = None

    def record_person_check(self, is_person: bool):
        """FIX NOTE 13: log one liveness sample for this track. Called
        repeatedly over the track's life (throttled by a sample cap in
        process_video_frame) so the eventual person/not-person call is
        based on the object's behavior across many frames, not a single
        snapshot that a patterned wall or a logo could get lucky on."""
        self.person_checks += 1
        if is_person:
            self.person_hits += 1

    def person_confidence_ok(self, min_samples: int, min_ratio: float) -> bool:
        if self.person_checks < min_samples:
            return False
        return (self.person_hits / self.person_checks) >= min_ratio



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
    fidxs = [h["f"] for h in hist]

    curr_h = max(heights[-1], 20.0)
    aspect_ratios = [w / max(h, 1.0) for w, h in zip(widths, heights)]

    aspect_curr = float(np.mean(aspect_ratios[-3:]))
    h_drop = (np.mean(heights[:3]) - np.mean(heights[-3:])) / max(np.mean(heights[:3]), 1.0)

    # FIX NOTE 5/6: since the producer now decimates frames (skips some
    # raw frames to keep playback at real speed), consecutive history
    # entries can be more than 1 source-frame apart. Normalize by the
    # actual average frame-index gap so speed/jitter thresholds stay
    # meaningful regardless of the decimation factor.
    frame_gaps = np.diff(fidxs)
    avg_step = float(np.mean(frame_gaps)) if len(frame_gaps) else 1.0
    avg_step = max(avg_step, 1.0)

    horiz_v = (np.diff(cxs) / curr_h) / avg_step
    vert_v = (np.diff(cys) / curr_h) / avg_step

    displacement = math.hypot(cxs[-1] - cxs[0], cys[-1] - cys[0]) / curr_h
    speed_mean = float(np.mean(np.abs(horiz_v))) if len(horiz_v) else 0.0
    speed_jitter = float(np.std(horiz_v)) if len(horiz_v) else 0.0

    # FIX NOTE 11.1: how ABRUPT the steepest single-frame height loss was
    # (normalized for decimation) — high for a real collapse, low for a
    # person gradually/deliberately lowering themselves to sit or rest,
    # even though both can reach the same net h_drop over the window.
    h_diffs = -np.diff(heights)  # positive = height decreased this step
    if len(h_diffs):
        base_h = max(np.mean(heights[:3]), 1.0)
        drop_rate = float(np.max(h_diffs)) / base_h / avg_step
    else:
        drop_rate = 0.0

    # FIX NOTE 11.2: two-axis tremor, measured only from the most recent
    # samples so it reflects the state AFTER any fall/sit transition has
    # already happened, not the transition's own postural wobble.
    tail_n = 3
    horiz_tail = horiz_v[-tail_n:] if len(horiz_v) else horiz_v
    vert_tail = vert_v[-tail_n:] if len(vert_v) else vert_v
    tremor = float(
        math.hypot(
            float(np.std(horiz_tail)) if len(horiz_tail) else 0.0,
            float(np.std(vert_tail)) if len(vert_tail) else 0.0,
        )
    )

    # FIX NOTE 16.1: activity right AFTER the most recent moment — used
    # to tell "fell/stumbled and kept moving" (conscious) apart from
    # "fell and went still" (loss of consciousness), independent of the
    # fall/stumble's own transition.
    tail_speed = float(np.mean(np.abs(horiz_v[-2:]))) if len(horiz_v) else 0.0

    # FIX NOTE 16.2: how often the horizontal direction flips, as a
    # fraction of steps. A limp still generally makes forward progress
    # (moderate jitter, but net displacement keeps growing); confused/
    # disoriented wandering flips direction constantly while going
    # nowhere (real motion, but little net displacement) — this is the
    # proxy used for delirium.
    signs = np.sign(horiz_v)
    nonzero = signs[signs != 0]
    if len(nonzero) > 1:
        direction_reversals = float(np.sum(nonzero[1:] != nonzero[:-1])) / (len(nonzero) - 1)
    else:
        direction_reversals = 0.0

    return dict(
        aspect_curr=aspect_curr,
        h_drop=h_drop,
        displacement=displacement,
        speed_mean=speed_mean,
        speed_jitter=speed_jitter,
        drop_rate=drop_rate,
        tremor=tremor,
        tail_speed=tail_speed,
        direction_reversals=direction_reversals,
    )


def classify_taxonomy(f: dict, sensitivity: int):
    """Per-track classification. `sensitivity` widens/loosens every
    threshold (not just the reported confidence)."""
    s = sensitivity / 100.0

    # === FIX NOTE 16: heat-emergency severity, keyed on the actual
    # clinical distinction requested: sunstroke (ضربة شمس) is a
    # NEUROLOGICAL sign — delirium, loss of consciousness, or
    # convulsions. Heat exhaustion (إجهاد حراري) is a MUSCULAR/FATIGUE
    # sign — an unsteady gait/limp that may end in a stumble or fall, but
    # the person stays conscious and mobile through it. The neurological
    # checks run first (most specific/most urgent should win if more
    # than one condition happens to overlap in the same window). ===

    # --- Convulsions -> sunstroke (Critical), regardless of posture ---
    tremor_thresh = 0.05 - 0.015 * s
    if f["tremor"] > tremor_thresh:
        return "sunstroke_fainting", min(0.97, 0.80 + 0.15 * s)

    # --- Loss of consciousness -> sunstroke (Critical): essentially
    # motionless, whether fainted still standing or collapsed and
    # stayed down. Posture doesn't matter here, only that real movement
    # has actually stopped.
    near_still = f["speed_mean"] < (0.009 + 0.006 * s) and f["displacement"] < (0.08 + 0.04 * s)
    if near_still:
        return "sunstroke_fainting", min(0.97, 0.75 + 0.20 * s)

    # --- Delirium -> sunstroke (Critical): moving, but incoherently —
    # frequent direction reversals with little real net progress. A limp
    # still generally makes forward progress; this doesn't.
    delirium_reversal_thresh = 0.55 - 0.10 * s
    if (
        f["direction_reversals"] > delirium_reversal_thresh
        and f["speed_mean"] > (0.010 + 0.004 * s)
        and f["displacement"] < (0.16 + 0.06 * s)
    ):
        return "sunstroke_fainting", min(0.93, 0.68 + 0.20 * s)

    # --- Limp progressing into a stumble/fall the person stays
    # conscious through -> heat exhaustion (High): a limping jitter
    # signature AND an abrupt height-drop (stumble/fall) happened in the
    # same window, but the person kept moving afterward (tail_speed
    # above a "still conscious" floor) instead of going still — staying
    # conscious is exactly what rules this OUT of the sunstroke tier
    # above.
    limp_signature = f["speed_jitter"] > (0.045 - 0.015 * s)
    heat_fall_h_drop_thresh = 0.22 * (1.15 - 0.35 * s)
    heat_fall_drop_rate_thresh = 0.09 * (1.15 - 0.35 * s)
    stumbled = f["h_drop"] > heat_fall_h_drop_thresh and f["drop_rate"] > heat_fall_drop_rate_thresh
    stayed_conscious = f["tail_speed"] > (0.018 + 0.006 * s)
    if limp_signature and stumbled and stayed_conscious:
        return "suspected_heat_exhaustion", min(0.88, 0.58 + 0.20 * s)

    # --- Sudden fall / fall + seizure (general, not heat-specific):
    # rapid height collapse + widened silhouette. FIX NOTE 11.1: net
    # h_drop alone doesn't distinguish a real collapse from someone
    # deliberately/gradually lowering themselves to sit or rest — both
    # end up shorter. Require the drop to also have been ABRUPT
    # (drop_rate) before calling it a fall at all; a slow, controlled
    # descent no longer qualifies. ---
    fall_h_drop_thresh = 0.26 * (1.15 - 0.35 * s)
    fall_drop_rate_thresh = 0.11 * (1.15 - 0.35 * s)
    if (
        f["aspect_curr"] > 1.0
        and f["h_drop"] > fall_h_drop_thresh
        and f["drop_rate"] > fall_drop_rate_thresh
    ):
        # FIX NOTE 11.2: seizure requires a genuine post-fall tremor
        # (two-axis, tail-only jitter) — not the one-off wobble of the
        # fall/sit transition itself, which `tremor` deliberately excludes.
        if f["tremor"] > (0.05 - 0.015 * s):
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

def frame_producer(video_path, max_frames, frame_queue, target_fps, src_fps, stop_event):
    """FIX NOTE 3/5: paced AND decimated producer.

    Every raw frame is read and paced against the SOURCE video's own
    timeline (1/src_fps per frame) — so total wall-clock time to get
    through the clip matches its real duration, not a slowed-down
    version of it. Only every Nth frame (N = src_fps/target_fps) is
    actually pushed to the queue for display/detection, which is what
    keeps the render side at a sustainable ~target_fps. This is the
    combination that gives real-time-feeling playback instead of either
    slow motion (pacing at target_fps only) or a decode-everything/
    catch-up jump (no pacing at all)."""
    cap = cv2.VideoCapture(video_path)
    skip_every = max(1, round(src_fps / max(target_fps, 1)))
    frame_interval = 1.0 / max(src_fps, 1)
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

        if count % skip_every != 0:
            continue  # keep the source-timed pace, but don't render/detect this one

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
# Person-shape refinement/verification: OpenCV's built-in HOG pedestrian
# detector. Loaded once (no external model download / network access
# needed). We only run it on small padded crops around an
# already-flagged track's motion bbox — not on every track/every frame —
# so it doesn't materially slow down normal frames.
# ----------------------------------------------------------------------
try:
    _HOG = cv2.HOGDescriptor()
    _HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
except Exception:
    # Some cloud/headless OpenCV builds don't expose HOGDescriptor (or fail
    # to load its default SVM weights) depending on how the package was
    # built/installed. Degrade gracefully instead of crashing the whole
    # app — refine_person_box below just returns the original motion bbox
    # unchanged, and verify_person_present below just stops gating, when
    # this is None.
    _HOG = None


def _hog_detect(frame_bgr, bbox, pad):
    """Shared HOG lookup used by both refine_person_box (tightening) and
    verify_person_present (gating). Returns (rects, weights, roi_origin)
    or (None, None, None) if the crop is unusable / HOG unavailable."""
    if _HOG is None:
        return None, None, None

    x, y, w, h = bbox
    H, W = frame_bgr.shape[:2]
    x0, y0 = max(int(x - pad), 0), max(int(y - pad), 0)
    x1, y1 = min(int(x + w + pad), W), min(int(y + h + pad), H)
    if x1 <= x0 or y1 <= y0:
        return None, None, None

    roi = frame_bgr[y0:y1, x0:x1]
    if roi.shape[0] < 24 or roi.shape[1] < 16:
        return None, None, None  # too small for HOG to say anything useful

    try:
        rects, weights = _HOG.detectMultiScale(
            roi, winStride=(6, 6), padding=(8, 8), scale=1.05
        )
    except Exception:
        return None, None, None

    return rects, weights, (x0, y0)


def _hog_detect_rotated(frame_bgr, bbox, pad):
    """FIX NOTE 15: the default pedestrian HOG detector expects an
    upright standing/walking silhouette, so it can miss a fallen/lying
    person's box outright (width >= height). Rotate the padded crop 90
    degrees before running HOG so a horizontal silhouette presents to
    the detector the way an upright one normally would. Returns
    (rects, weights) or (None, None) if the crop is unusable/HOG is
    unavailable."""
    if _HOG is None:
        return None, None

    x, y, w, h = bbox
    H, W = frame_bgr.shape[:2]
    x0, y0 = max(int(x - pad), 0), max(int(y - pad), 0)
    x1, y1 = min(int(x + w + pad), W), min(int(y + h + pad), H)
    if x1 <= x0 or y1 <= y0:
        return None, None

    roi = frame_bgr[y0:y1, x0:x1]
    if roi.shape[0] < 16 or roi.shape[1] < 24:
        return None, None  # too small for HOG to say anything useful

    rotated = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)

    try:
        rects, weights = _HOG.detectMultiScale(
            rotated, winStride=(6, 6), padding=(8, 8), scale=1.05
        )
    except Exception:
        return None, None

    return rects, weights


def refine_person_box(frame_bgr, bbox, pad=25):
    rects, weights, origin = _hog_detect(frame_bgr, bbox, pad)
    if rects is None or len(rects) == 0:
        return bbox
    x0, y0 = origin

    # pick the highest-confidence detection in the crop
    best = int(np.argmax(weights)) if len(weights) else 0
    rx, ry, rw, rh = rects[best]
    refined = (x0 + int(rx), y0 + int(ry), int(rw), int(rh))

    # FIX NOTE 7.1: HOG sometimes only picks up part of a person (torso,
    # shoulders) and returns a box far smaller than the actual motion
    # blob — that undersized box was a recurring source of visibly-wrong
    # alert boxes. Reject the refinement if it shrank the box too much
    # and keep the original motion bbox instead.
    orig_area = max(bbox[2] * bbox[3], 1)
    refined_area = refined[2] * refined[3]
    if refined_area < 0.5 * orig_area:
        return bbox
    return refined


def verify_person_present(frame_bgr, bbox, pad=20, min_weight=0.55):
    """FIX NOTE 8: gate against inanimate/static objects ("jamadat")
    firing an alert. This matters most for the heat-emergency
    conditions, since a completely stationary non-person blob (a static
    object, a parked item, a stabilized shadow/reflection) trivially
    satisfies "barely moving" — the same signal a real person standing
    still would give. Require OpenCV's pedestrian HOG detector to
    actually find a person-shaped silhouette in the candidate box before
    letting one of those conditions confirm.

    FIX NOTE 13: also require a minimum HOG confidence (`min_weight`),
    not just any detection. A repetitive/textured surface (patterned
    wall, railing, gate) can occasionally produce a low-confidence HOG
    hit purely by texture coincidence; a real person's detection score
    is typically well above that. This alone isn't perfectly reliable on
    a single frame — which is exactly why it's now sampled repeatedly
    over the track's life (see Track.person_checks) instead of being
    trusted on one snapshot.

    FIX NOTE 15: the plain upright HOG pass above can miss a genuinely
    fallen/lying person (wide/flat box), which would wrongly fail this
    gate for exactly the most critical alerts (sudden_fall,
    lying_immobile, sunstroke_fainting). For wide/flat boxes only, also
    try a second HOG pass on a 90-degree-rotated crop before giving up —
    upright boxes already got a fair shot from the first pass, so this
    extra cost is only paid for the shapes where it's actually needed.

    If this build has no HOG detector available, we can't gate at all —
    return True so real alerts on this box still fire rather than being
    silently suppressed forever."""
    if _HOG is None:
        return True

    rects, weights, _origin = _hog_detect(frame_bgr, bbox, pad)
    if rects is not None and len(rects) and float(np.max(weights)) >= min_weight:
        return True

    x, y, w, h = bbox
    if w >= h * 0.9:
        rects_r, weights_r = _hog_detect_rotated(frame_bgr, bbox, pad)
        if rects_r is not None and len(rects_r) and float(np.max(weights_r)) >= min_weight:
            return True

    return False


def pad_box(bbox, frame_shape, pad_ratio=0.28, min_pad_px=8, max_pad_px=40):
    """FIX NOTE 7.2: expand a box by a margin before drawing so alert
    boxes read clearly instead of hugging (or cutting into) the person.
    FIX NOTE 12: the margin is also capped in absolute pixels so this
    step can't itself balloon an already-large box further."""
    x, y, w, h = bbox
    H, W = frame_shape[:2]
    px = min(max(int(w * pad_ratio), min_pad_px), max_pad_px)
    py = min(max(int(h * pad_ratio), min_pad_px), max_pad_px)
    nx = max(x - px, 0)
    ny = max(y - py, 0)
    nx2 = min(x + w + px, W)
    ny2 = min(y + h + py, H)
    return (nx, ny, max(nx2 - nx, 1), max(ny2 - ny, 1))


# Minimum tracked-person height (px, at the 480x270 processing resolution)
# for a track to be eligible for anomaly classification at all. Most
# reported false alarms were on small/partial blobs — those are also the
# least trustworthy signal for classification, so gating them out here
# addresses both problems at once.
MIN_ALERT_HEIGHT_PX = 42

# FIX NOTE 12: matching ceiling — a blob this tall/wide relative to the
# frame is no longer a plausible single person; it reads as a large
# static/lighting artifact instead. Ratios are relative to the
# processing frame's own dimensions so they hold regardless of
# resolution.
MAX_CONTOUR_AREA_RATIO = 0.18   # a person's motion blob shouldn't fill more than ~18% of the frame
MAX_PERSON_HEIGHT_RATIO = 0.85  # relative to frame height
MAX_PERSON_WIDTH_RATIO = 0.55   # relative to frame height (a standing person is taller than wide)

# FIX NOTE 13: liveness sampling — a track only gets to fire an alert
# once it's been checked for "is this actually a person" across several
# separate frames during its time in view (not just once, at whichever
# instant the motion pattern happened to confirm), and the majority of
# those checks came back positive.
PERSON_CHECK_MIN_SAMPLES = 3     # need at least this many samples before a verdict counts
PERSON_CHECK_MAX_SAMPLES = 6     # stop sampling after this many (cost control once evidence is enough)
PERSON_HIT_RATIO_REQUIRED = 0.66  # fraction of those samples that must be positive

# FIX NOTE 10/13: every condition — not just the heat-emergency ones —
# now requires accumulated person-liveness evidence (see Track.person_*
# and PERSON_CHECK_* above) before it's allowed to confirm at all.

# FIX NOTE 14: how many consecutive frames a track can go without a
# matched detection before it's actually considered "gone" and dropped.
# Kept generous enough to survive a brief occlusion/blob-merge in a
# crowd without losing the track's identity, history, and accumulated
# person-liveness evidence.
MAX_MISSED_FRAMES = 10


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

    # FIX NOTE 12: plausible single-person size ceiling, relative to this
    # frame's own dimensions.
    frame_h, frame_w = frame.shape[:2]
    max_area = MAX_CONTOUR_AREA_RATIO * frame_h * frame_w
    max_h = MAX_PERSON_HEIGHT_RATIO * frame_h
    max_w = MAX_PERSON_WIDTH_RATIO * frame_h

    detections = []
    for c in contours:
        # Keep the low area floor (catches distant/small people early),
        # relying on MOG2 + morphology above to keep noise out instead of
        # a high area cutoff.
        area = cv2.contourArea(c)
        if area < 250:
            continue
        x, y, w, h = cv2.boundingRect(c)
        # FIX NOTE 12: reject implausibly large blobs outright — never
        # even becomes a track — instead of only capping it later.
        if area > max_area or h > max_h or w > max_w:
            continue
        detections.append(((x + w / 2, y + h / 2), (x, y, w, h)))

    # FIX NOTE 14: match against each track's PREDICTED position (its
    # last known position projected forward by its velocity and how long
    # it's been coasting), not just its last-seen spot — and scale the
    # acceptance radius by the track's own size, widening slightly the
    # longer it's been missing, so a person who reappears after a brief
    # occlusion/merge is re-matched to their SAME track/ID instead of
    # spawning a new one.
    assigned = set()
    for (cx, cy), (x, y, w, h) in detections:
        best_id, best_d = None, None
        for tid, tr in state["tracks"].items():
            if tid in assigned:
                continue
            pc = tr.predicted_centroid()
            d = math.hypot(pc[0] - cx, pc[1] - cy)
            max_d = max(tr.bbox[3], 20.0) * (1.1 + 0.15 * tr.missed)
            if d < max_d and (best_d is None or d < best_d):
                best_d, best_id = d, tid
        if best_id is not None:
            state["tracks"][best_id].update((cx, cy), (x, y, w, h), frame_idx)
            assigned.add(best_id)
        else:
            tid = state["next_id"]
            state["next_id"] += 1
            state["tracks"][tid] = Track(tid, (cx, cy), (x, y, w, h), frame_idx)
            assigned.add(tid)

    # FIX NOTE 14: a track that found no matching detection this frame is
    # marked as "missed" (coasting) rather than deleted immediately — its
    # history, confirmation state, and accumulated person-liveness
    # evidence all stay intact in case it's re-matched on a following
    # frame.
    for tid, tr in state["tracks"].items():
        if tid not in assigned:
            tr.missed += 1

    # Drop tracks only once they've genuinely been gone long enough —
    # not on the very first missed frame.
    for tid in [t for t, obj in state["tracks"].items() if obj.missed > MAX_MISSED_FRAMES]:
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
        # FIX NOTE 14: a coasting track (missed this frame, no fresh
        # bbox) has nothing new to classify or draw this tick — skip it,
        # but it's still alive and will resume normally once re-matched.
        if tr.missed > 0:
            continue

        f = track_feats[tid]
        if f is None:
            continue

        # FIX NOTE 6.2 / 12: small/partial detections are excluded from
        # alert eligibility entirely (least reliable signal), and so are
        # blobs that have grown past a plausible single-person size
        # (safety net matching the detection-time cap above) — still
        # tracked either way, just not classified.
        max_alert_h = MAX_PERSON_HEIGHT_RATIO * canvas.shape[0]
        if tr.bbox[3] < MIN_ALERT_HEIGHT_PX or tr.bbox[3] > max_alert_h:
            continue

        if tid in fighting_ids:
            cond, conf = "fighting", min(0.93, 0.62 + f["speed_jitter"] * 2.2)
            confirm_needed_here = confirm_needed + 1  # extra caution on fighting specifically
        else:
            cond, conf = classify_taxonomy(f, sensitivity)
            confirm_needed_here = confirm_needed

        # FIX NOTE 4: only act once the SAME condition has held for
        # several consecutive frames — kills one-off noise spikes.
        confirmed_now = tr.confirm(cond, confirm_needed_here)

        # FIX NOTE 13: sample liveness evidence every frame this track is
        # alert-eligible (throttled by a sample cap for cost), across its
        # WHOLE time in view — not only at the instant a classification
        # happens to confirm. This is what lets a track that entered
        # frame as (say) a wall-shadow get judged on many looks instead
        # of the one unlucky/lucky frame where thresholds lined up.
        if tr.person_checks < PERSON_CHECK_MAX_SAMPLES:
            tr.record_person_check(verify_person_present(canvas, tr.bbox))

        # A pending motion-pattern classification that no longer matches
        # what we were waiting to fire is stale — drop it so it doesn't
        # block a future, different condition from ever being awaited.
        if tr.awaiting_cond is not None and tr.awaiting_cond != tr.pending_cond:
            tr.awaiting_cond = None
        if tr.confirmed_cond is not None and tr.confirmed_cond != tr.pending_cond:
            tr.confirmed_cond = None

        if confirmed_now and cond in TAXONOMY_RULES:
            tr.awaiting_cond = cond
            tr.awaiting_conf = conf

        if tr.awaiting_cond is not None and tr.awaiting_cond == tr.pending_cond:
            if tr.person_confidence_ok(PERSON_CHECK_MIN_SAMPLES, PERSON_HIT_RATIO_REQUIRED):
                info = TAXONOMY_RULES[tr.awaiting_cond]
                label = f"ALERT: {info['en'].upper()} ({tr.awaiting_conf*100:.0f}%)"

                box = refine_person_box(canvas, tr.bbox)
                x, y, w, h = pad_box(box, canvas.shape)

                cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 3)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(canvas, (x, max(y - 24, 0)), (x + tw + 10, max(y, 24)), (0, 0, 255), -1)
                cv2.putText(canvas, label, (x + 5, max(y - 7, 17)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                last_f = state["global_cd"].get(tr.awaiting_cond, -9999)
                if frame_idx - last_f > 60:
                    state["global_cd"][tr.awaiting_cond] = frame_idx
                    new_alerts.append((tr.awaiting_cond, tr.awaiting_conf))

                tr.confirmed_cond = tr.awaiting_cond
                tr.awaiting_cond = None
            elif tr.person_checks >= PERSON_CHECK_MAX_SAMPLES:
                # FIX NOTE 8/13: evidence budget exhausted without a
                # confident majority of "yes, this is a person" — treat
                # as a non-person and give up on this classification
                # rather than waiting forever.
                tr.reject_pending()
            # else: motion is confirmed but liveness evidence is still
            # accumulating — wait, don't draw yet, don't re-fire later.
        elif tr.confirmed_cond is not None and tr.confirmed_cond == tr.pending_cond:
            # Already fired (motion-confirmed AND person-verified) and
            # still ongoing this frame — keep the box visible without
            # re-logging a duplicate alert.
            info = TAXONOMY_RULES[tr.confirmed_cond]
            box = refine_person_box(canvas, tr.bbox)
            x, y, w, h = pad_box(box, canvas.shape)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(canvas, info["en"].upper(), (x + 5, max(y - 7, 17)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


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

    # FIX NOTE 9: order by clinical priority first — sunstroke/fainting
    # (Critical) above suspected heat exhaustion (High) above routine
    # gait/fatigue alerts (Medium/Low) — and only fall back to recency
    # within the same tier, instead of pure reverse-chronological order
    # which could bury a Critical alert under a run of Medium ones.
    ordered = sorted(
        enumerate(st.session_state.alerts),
        key=lambda pair: (
            PRIORITY_RANK.get(TAXONOMY_RULES.get(pair[1].condition_key, {}).get("priority", "Low"), 3),
            -pair[0],
        ),
    )

    html_out = ""
    for _orig_idx, a in ordered:
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

    # FIX NOTE 5: read the clip's real fps so the producer can pace
    # against its true timeline instead of assuming a fixed rate.
    probe = cv2.VideoCapture(tfile_path)
    src_fps = probe.get(cv2.CAP_PROP_FPS)
    probe.release()
    if not src_fps or src_fps <= 1 or src_fps > 120:
        src_fps = 25.0

    # FIX NOTE 3: cap the requested playback speed at RENDER_FPS so the
    # producer can never outpace what the UI paints — this is what keeps
    # the video looking like a continuous live feed instead of
    # decode-everything-then-fast-forward.
    target_fps = min(play_speed, RENDER_FPS)

    frame_queue = queue.Queue(maxsize=8)
    stop_event = threading.Event()
    prod_thread = threading.Thread(
        target=frame_producer,
        args=(tfile_path, max_f, frame_queue, target_fps, src_fps, stop_event),
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
        "fps_src": float(src_fps),
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
