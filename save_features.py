import os
import csv
import re
import math
from typing import Optional, Dict, Any, List
import numpy as np

METRICS_DIR = "serve_metrics/"
OUT_CSV = "serve_features.csv"

def extract_speed_from_filename(name: str) -> Optional[float]:
    m = re.search(r"_(\d+)_serve", name)
    return float(m.group(1))

def load_metrics_csv(path: str) -> List[Dict[str, float]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows

def compute_serve_features(metrics_rows: List[Dict[str, str]],
                           serve_id: str) -> Dict[str, Any]:
    # knee_bend_deg, torso_tilt_deg, racket_drop, hip_shoulder_sep_deg, torso_stillness_deg, hip_angle_deg
    n = len(metrics_rows)
    total_frames = n

    def col(name):
        return np.array([(r.get(name, np.nan)) for r in metrics_rows], dtype=float)

    knee = col("knee_bend_deg")
    torso_tilt = col("torso_tilt_deg")
    racket = col("racket_drop")
    hipsep = col("hip_shoulder_sep_deg")
    torso_still = col("torso_stillness_deg")
    hip_angle = col("hip_angle_deg")

    knee_max_idx = int(np.nanargmax((knee)))
    racket_min_idx = int(np.nanargmin(racket))
    hipsep_max_idx = int(np.nanargmax((hipsep)))

    # Get contact frame from metrics (computed in serve_mechanics.py)
    contact_idx = int(float(metrics_rows[0].get("contact_frame", 0)))

    # Magnitude features
    max_knee_bend = float(knee[knee_max_idx])
    knee_min = float(np.nanmin(knee))
    knee_range = max_knee_bend - knee_min
    knee_min_idx = int(np.nanargmin(knee))  # Add this line
    knee_push_off = (knee_min_idx - knee_max_idx) / total_frames  # Add this line
    min_racket_drop = float(racket[racket_min_idx])
    max_hip_shoulder_sep = float(hipsep[hipsep_max_idx])

    torso_tilt_at_contact = float(torso_tilt[contact_idx])
    hip_angle_at_contact = float(hip_angle[contact_idx])

    # torso stillness std during loading phase: start -> max knee bend
    start = 0
    end = knee_max_idx
    segment = torso_still[start:end + 1]
    torso_stillness_std_loading = float(np.nanstd(segment))

    knee_max_time = (knee_max_idx) / total_frames
    racket_vs_knee = (racket_min_idx - knee_max_idx) / total_frames
    hipsep_vs_knee = (hipsep_max_idx - knee_max_idx) / total_frames
    contact_vs_knee = (contact_idx - knee_max_idx) / total_frames
    contact_vs_hipsep = (contact_idx - hipsep_max_idx) / total_frames

    feature: Dict[str, Any] = {
        "serve_id": serve_id,
        "total_frames": int(total_frames),
        "max_knee_bend": max_knee_bend,
        "knee_bend_range": knee_range,
        "min_racket_drop": min_racket_drop,
        "max_hip_shoulder_sep": max_hip_shoulder_sep,
        "torso_tilt_at_contact": torso_tilt_at_contact,
        "hip_angle_at_contact": hip_angle_at_contact,
        "torso_stillness_std_loading": torso_stillness_std_loading,
        "knee_push_off": knee_push_off,

        # timing
        "knee_max_time": knee_max_time,
        "racket_vs_knee": racket_vs_knee,
        "hipsep_vs_knee": hipsep_vs_knee,
        "contact_vs_knee": contact_vs_knee,
        "contact_vs_hipsep": contact_vs_hipsep,
    }
    
    feature["serve_speed_label"] = extract_speed_from_filename(serve_id)

    return feature

def append_features_to_csv(out_path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    header = list(rows[0].keys())
    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

def process_all():
    files = sorted([f for f in os.listdir(METRICS_DIR) if f.endswith("_metrics.csv")])

    features = []
    for mf in files:
        metrics_path = os.path.join(METRICS_DIR, mf)
        serve_id = mf.replace("_metrics.csv", "")
        metrics_rows = load_metrics_csv(metrics_path)
        feature = compute_serve_features(metrics_rows, serve_id)
        features.append(feature)

    append_features_to_csv(OUT_CSV, features)
    print(f"Wrote {len(features)} feature rows to {OUT_CSV}")

process_all()
