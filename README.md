# Early Medical Emergency Detection System — Prototype

An interactive Streamlit + OpenCV prototype for **pre-collapse anomaly detection**
in crowded surveillance environments (holy sites, transport hubs, stadiums).

## What it does

It watches for early warning movement patterns — **before** a person fully loses
consciousness — and maps them to a medical triage rule engine:

| Detected pattern | Probable condition | Priority |
|---|---|---|
| `severe_gait_limping` | Heatstroke / Severe Dehydration Warning | Medium |
| `stooped_walking_resting` | Syncope / Sudden Blood Pressure Drop | High |
| `severe_choking_on_ground` | Acute Airway Obstruction / Respiratory Distress | Critical |
| `seizure_convulsion` | Active Seizure / Convulsion | Critical |
| `sudden_fall` | Cardiac Arrest Precursor / Unconsciousness | Critical |

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens with **Simulation mode ON** by default, so it works immediately with
zero setup: it runs a scripted synthetic feed through the exact same
detection/classification pipeline used for real footage, cycling through all five
emergency patterns plus normal walking, so every part of the system (bounding
boxes, triage engine, alert feed, dispatch button) is demonstrable out of the box.

To analyze real footage instead: turn **Simulation mode** off in the sidebar and
upload an `.mp4` clip.

## How detection works (no external model weights required)

1. **Real video path**: OpenCV `BackgroundSubtractorMOG2` extracts moving
   foreground blobs → contours are filtered by area → nearest-centroid tracking
   assigns each blob a persistent track ID across frames.
2. **Feature extraction**: for each track's recent ~1.5s window, the system
   computes bounding-box kinematics — height-drop ratio, aspect-ratio flip
   (standing → prone), centroid jitter, path efficiency, motion energy
   (frame-to-frame contour-area volatility), and peak vertical velocity.
3. **Classification**: a transparent, tunable rule engine (`classify()` in
   `app.py`) maps these kinematic signatures to the five conditions above. The
   **sensitivity slider** scales how easily borderline motion counts as
   anomalous.
4. **Simulation path**: when no video is supplied, scripted bounding-box
   kinematics are fed through the *same* `extract_features()` / `classify()`
   functions — so the triage logic being demonstrated is real, only the sensor
   input is synthetic.

`test_logic.py` is a standalone script that validates each of the five
conditions correctly fires during its corresponding simulation phase.

## Production roadmap (not implemented here, noted in-app)

- Replace the motion-heuristic classifier with a pose-estimation backbone
  (MediaPipe / RTMPose) feeding a temporal model (LSTM / ST-GCN / Transformer)
  trained on a clinically-labeled fall/seizure/gait dataset.
- Multi-camera re-identification and sensor fusion.
- Human triage confirmation step before any real-world dispatch.
- Integration with a CAD / emergency-dispatch API and radio system.
- On-device inference for privacy and latency.

## Disclaimer

This is a demonstration prototype, not a certified or clinically validated
medical device. Do not use it for real triage decisions.
