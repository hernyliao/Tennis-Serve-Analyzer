import os
import csv
import math
from typing import List, Tuple
import numpy as np
import statistics
import glob

handedness = 'right'

def vec(a: Tuple[float,float,float], b: Tuple[float,float,float]) -> np.ndarray:
    return np.array([b[0]-a[0], b[1]-a[1], b[2]-a[2]], dtype=float)

def length(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    cosv = np.dot(a, b) / (na * nb)
    cosv = float(np.clip(cosv, -1.0, 1.0))
    return math.degrees(math.acos(cosv))

def point_from_row(row: List[str], header_map: dict, lm_name: str) -> Tuple[float,float,float]:
    return (float(row[header_map[f"{lm_name}_x"]]),
            float(row[header_map[f"{lm_name}_y"]]),
            float(row[header_map[f"{lm_name}_z"]]))

def detect_contact_frame_from_pose(pose_rows: List[dict], prefer_right: bool = True) -> int:
    """Detect contact frame by finding max downward wrist velocity."""
    first = pose_rows[0]

    right_y = "RIGHT_WRIST_y"
    left_y = "LEFT_WRIST_y"
    use_col = None
    if prefer_right and right_y in first:
        use_col = right_y
    elif left_y in first:
        use_col = left_y
    elif right_y in first:
        use_col = right_y
    else:
        for k in first.keys():
            if k.endswith("_WRIST_y"):
                use_col = k
                break

    if use_col is None:
        return 0

    yvals = np.array([float(r.get(use_col, np.nan)) for r in pose_rows], dtype=float)
    vel = np.diff(yvals)
    idx = int(np.argmax(vel))
    return idx + 1

def compute_frame_metrics(row: List[str], header_map: dict) -> dict:
    if handedness == 'right':
        HIP1 = point_from_row(row, header_map, "RIGHT_HIP")
        HIP2 = point_from_row(row, header_map, "LEFT_HIP")
        KNEE = point_from_row(row, header_map, "RIGHT_KNEE")
        ANKLE = point_from_row(row, header_map, "RIGHT_ANKLE")
        SHO1 = point_from_row(row, header_map, "RIGHT_SHOULDER")
        SHO2 = point_from_row(row, header_map, "LEFT_SHOULDER")
        WRIST = point_from_row(row, header_map, "RIGHT_WRIST")
        ELB = point_from_row(row, header_map, "RIGHT_ELBOW")
        NOSE = point_from_row(row, header_map, "NOSE")
    if handedness == 'left':
        HIP1 = point_from_row(row, header_map, "LEFT_HIP")
        HIP2 = point_from_row(row, header_map, "RIGHT_HIP")
        KNEE = point_from_row(row, header_map, "LEFT_KNEE")
        ANKLE = point_from_row(row, header_map, "LEFT_ANKLE")
        SHO1 = point_from_row(row, header_map, "LEFT_SHOULDER")
        SHO2 = point_from_row(row, header_map, "RIGHT_SHOULDER")
        WRIST = point_from_row(row, header_map, "LEFT_WRIST")
        ELB = point_from_row(row, header_map, "LEFT_ELBOW")
        NOSE = point_from_row(row, header_map, "NOSE")
    # Notes: What makes a good serve: knee bend, torso tilt, racket drop, hip-shoulder separation, torso stillness, hip angle (to ground)
    # shoulder distance scale
    shoulder_dist = length(vec(SHO2, SHO1))
    if shoulder_dist < 1e-6:
        shoulder_dist = 1.0

    metrics = {}

    v1 = vec(KNEE, HIP1)    # hip relative to knee
    v2 = vec(KNEE, ANKLE)  # ankle relative to knee
    knee_angle = angle_between(v1, v2)
    metrics["knee_bend_deg"] = knee_angle

    # 2) Torso tilt: angle of vector from hip-mid to shoulder-mid relative to vertical axis.
    mid_shoulder = ((SHO1[0] + SHO2[0]) / 2.0, (SHO1[1] + SHO2[1]) / 2.0, (SHO1[2] + SHO2[2]) / 2.0)
    mid_hip = ((HIP1[0] + HIP2[0]) / 2.0, (HIP1[1] + HIP2[1]) / 2.0, (HIP1[2] + HIP2[2]) / 2.0)
    spine_vec = vec(mid_hip, mid_shoulder)  # points upward along torso
    vert = np.array([0.0, -1.0, 0.0])

    torso_tilt = angle_between(spine_vec, vert)
    metrics["torso_tilt_deg"] = torso_tilt

    # 3) Racket drop depth: right wrist y relative to right shoulder y.
    # Negative number indicates wrist below shoulder (common racket drop is below shoulder).
    racket_drop = (WRIST[1] - SHO1[1])
    metrics["racket_drop"] = racket_drop

    # 4) Hip-shoulder separation (coil): angle between shoulder line and hip line projected into horizontal plane (x,z).
    shoulder_vec = np.array([SHO1[0]-SHO2[0], SHO1[2]-SHO2[2]])  # x,z only
    hip_vec = np.array([HIP1[0]-HIP2[0], HIP1[2]-HIP2[2]])       # x,z only
    if np.linalg.norm(shoulder_vec) < 1e-8 or np.linalg.norm(hip_vec) < 1e-8:
        hip_shoulder_sep = 0.0
    else:
        # compute 2D angle (in degrees) between shoulder_vec and hip_vec
        dot2 = float(np.dot(shoulder_vec, hip_vec) / (np.linalg.norm(shoulder_vec)*np.linalg.norm(hip_vec)))
        dot2 = float(np.clip(dot2, -1.0, 1.0))
        hip_shoulder_sep = math.degrees(math.acos(dot2))
    metrics["hip_shoulder_sep_deg"] = hip_shoulder_sep
    
    # 5) Torso stillness: angle of line between shoulders to ground
    shoulder_line_vec = vec(SHO2, SHO1)
    torso_stillness = angle_between(shoulder_line_vec, np.array([1.0, 0.0, 0.0]))
    metrics["torso_stillness_deg"] = torso_stillness

    # 6) Hip angle: angle of line between hips to the ground
    hip_vec = vec(HIP1, HIP2)
    hip_angle = angle_between(hip_vec, np.array([1.0, 0.0, 0.0]))
    metrics["hip_angle_deg"] = hip_angle
    return metrics


def process_file(input_csv: str, output_dir: str):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(input_csv)

    player = os.path.splitext(os.path.basename(input_csv))[0]
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f"{player}_metrics.csv")

    with open(input_csv, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError("CSV seems empty or missing header/rows")

    header = rows[0]
    data = rows[1:]

    # build header_map for quick lookup
    header_map = {name: idx for idx, name in enumerate(header)}
    
    # Convert rows to dicts for contact frame detection
    pose_dicts = [{header[i]: row[i] for i in range(len(header))} for row in data]
    contact_frame = detect_contact_frame_from_pose(pose_dicts)

    serve_start = 0
    serve_end = len(data)

    # compute metrics for each frame in serve window
    frames_out = []
    for idx in range(serve_start, serve_end):
        row = data[idx]
        frame_number = row[header_map.get("frame_number", 0)]
        timestamp = row[header_map.get("timestamp_sec", 1)]
        metrics = compute_frame_metrics(row, header_map)
        out_row = {
            "frame": frame_number,
            "timestamp": timestamp,
            "contact_frame": contact_frame,
            **metrics
        }
        frames_out.append(out_row)

    # write CSV
    fieldnames = ["frame", "timestamp", "contact_frame", "knee_bend_deg", "torso_tilt_deg",
                  "racket_drop", "hip_shoulder_sep_deg", "torso_stillness_deg", "hip_angle_deg"]
    with open(out_csv, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in frames_out:
            writer.writerow(r)

    # print summary
    if frames_out:
        knee_vals = [float(r["knee_bend_deg"]) for r in frames_out]
        td_vals = [float(r["torso_tilt_deg"]) for r in frames_out]
        rd_vals = [float(r["racket_drop"]) for r in frames_out]
        hs_vals = [float(r["hip_shoulder_sep_deg"]) for r in frames_out]
        print(f"Wrote {out_csv}  | frames: {len(frames_out)}")
        print("Summary (serve window):")
        print(f" knee_bend_deg: min={min(knee_vals):.2f}  median={statistics.median(knee_vals):.2f}  max={max(knee_vals):.2f}")
        print(f" torso_tilt_deg: min={min(td_vals):.2f}  median={statistics.median(td_vals):.2f}  max={max(td_vals):.2f}")
        print(f" racket_drop_norm: min={min(rd_vals):.3f}  median={statistics.median(rd_vals):.3f}  max={max(rd_vals):.3f}")
        print(f" hip_shoulder_sep_deg: min={min(hs_vals):.2f}  median={statistics.median(hs_vals):.2f}  max={max(hs_vals):.2f}")
        print(f" torso_stillness_deg: min={min([float(r['torso_stillness_deg']) for r in frames_out]):.2f}  median={statistics.median([float(r['torso_stillness_deg']) for r in frames_out]):.2f}  max={max([float(r['torso_stillness_deg']) for r in frames_out]):.2f}")
    else:
        print("No frames in serve window. No output written.")

video_paths = glob.glob("pose_data/*.csv")

for path in video_paths:
    process_file(path, output_dir="serve_metrics")