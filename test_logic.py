"""Standalone test: verifies the 6 scripted simulation phases each excite the
correct condition in the real classify()/extract_features() pipeline."""
import sys, types

class Dummy:
    def __getattr__(self, name):
        return lambda *a, **k: Dummy()
    def __call__(self, *a, **k): return Dummy()
    def __enter__(self): return self
    def __exit__(self, *a): return False

st_stub = types.ModuleType("streamlit")
for name in ["set_page_config","markdown","sidebar","toggle","file_uploader","selectbox",
             "text_input","slider","columns","button","write","empty","container","progress",
             "warning","error","info","success","caption","expander","rerun","image","title",
             "subheader","header"]:
    setattr(st_stub, name, lambda *a, **k: Dummy())
st_stub.session_state = {}
sys.modules["streamlit"] = st_stub

ns = {"__name__": "app_under_test"}
try:
    exec(compile(open("app.py").read(), "app.py", "exec"), ns)
except Exception:
    pass  # UI section fails with the stub; function/class defs above it already ran into ns

Track = ns["Track"]
extract_features = ns["extract_features"]
classify = ns["classify"]
SIM_PHASES = ns["SIM_PHASES"]
SIM_PHASE_LEN = ns["SIM_PHASE_LEN"]
sim_bbox_for_frame = ns["sim_bbox_for_frame"]

import numpy as np
rng = np.random.default_rng(7)
width, height = 640, 400
sensitivity = 60

results = {}
track = None
frame_idx = 0
for cycle in range(2):  # run two full cycles through all phases
    for phase_num, phase in enumerate(SIM_PHASES):
        fired = None
        for local_idx in range(SIM_PHASE_LEN):
            frame_idx += 1
            bbox, centroid = sim_bbox_for_frame(local_idx, phase, width, height, rng)
            if track is None:
                track = Track(1, centroid, bbox, frame_idx)
            else:
                track.update(centroid, bbox, frame_idx)
            feats = extract_features(track)
            if feats:
                cond, conf = classify(feats, sensitivity)
                if cond:
                    fired = (cond, round(conf, 2), local_idx)
        results.setdefault(phase, []).append(fired)

print("Phase -> (condition_fired, confidence, frame_within_phase) per cycle:")
for phase, fires in results.items():
    print(f"  {phase:20s} -> {fires}")

expected_map = {
    "severe_limp": "severe_gait_limping",
    "stooped_resting": "stooped_walking_resting",
    "sudden_fall": "sudden_fall",
    "seizure": "seizure_convulsion",
    "choking_on_ground": "severe_choking_on_ground",
}
print("\nValidation:")
all_ok = True
for phase, expected_cond in expected_map.items():
    fires = results[phase]
    ok = any(f is not None and f[0] == expected_cond for f in fires)
    print(f"  {phase:20s} expected='{expected_cond:28s}' -> {'PASS' if ok else 'FAIL'}  (last observed: {fires[-1]})")
    all_ok = all_ok and ok

print("\nALL PASS" if all_ok else "\nSOME FAILED")
