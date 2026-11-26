import cv2
import csv
import mediapipe as mp
import numpy as np
import os
import glob

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

video_files = glob.glob("videos/*.mp4")
video_basenames = [os.path.basename(f).replace(".mp4", "") for f in video_files]

pose_data_files = set(os.path.basename(f).replace(".csv", "") for f in glob.glob("pose_data/*.csv")) if os.path.exists("pose_data/afterdata") else set()

# Filter for videos not yet processed
unprocessed = [v for v in video_basenames if v not in pose_data_files]

# Use the full paths from glob directly
video_paths = [f for f in video_files if os.path.basename(f).replace(".mp4", "") in unprocessed]

DESIRED_SHOULDER_FRAC = 0.5
TARGET_CENTER = (0.5, 0.5)

def transform_to_shoulder_target(landmarks, frame_w, frame_h,
                                 left_idx, right_idx,
                                 desired_fraction=0.1,
                                 target_mid_norm=(0.5, 0.5)):
    eps = 1e-6
    W, H = float(frame_w), float(frame_h)

    L = landmarks[left_idx]
    R = landmarks[right_idx]
    P1_px = np.array([L.x * W, L.y * H], dtype=float)
    P2_px = np.array([R.x * W, R.y * H], dtype=float)

    v_px = P2_px - P1_px
    current_len = np.linalg.norm(v_px)
    target_len = desired_fraction * W
    scale = target_len / max(current_len, eps)

    M_px = (P1_px + P2_px) / 2.0
    M_target_px = np.array([target_mid_norm[0] * W, target_mid_norm[1] * H], dtype=float)

    out = []
    for lm in landmarks:
        L_px = np.array([lm.x * W, lm.y * H], dtype=float)
        r_px = L_px - M_px
        L_new_px = M_target_px + r_px * scale
        out.append((L_new_px[0] / W, L_new_px[1] / H, lm.z * scale))

    return out

def detect_serve_window(rows, header):
    lwrist_idx = header.index("LEFT_WRIST_y")
    rwrist_idx = header.index("RIGHT_WRIST_y")
    rhip_idx = header.index("RIGHT_HIP_y")

    lwrist_y = np.array([float(r[lwrist_idx]) for r in rows])
    rwrist_y = np.array([float(r[rwrist_idx]) for r in rows])
    rhip_y = np.array([float(r[rhip_idx]) for r in rows])

    vel_lwrist = np.diff(lwrist_y)
    vel_rwrist = np.diff(rwrist_y)

    serve_start = 0
    for i in range(len(vel_rwrist)):
        if vel_rwrist[i] < -0.01 and rwrist_y[i] < rhip_y[i] and lwrist_y[i] < rhip_y[i]:
            serve_start = i - 20
            break

    contact_frame = np.argmax(vel_rwrist) + 1
    serve_end = len(rwrist_y) - 1
    for i in range(contact_frame, len(rwrist_y)):
        if rwrist_y[i] > rhip_y[i] and vel_rwrist[i-1] < 0.01:
            serve_end = i + 5
            break

    print(f"Detected serve from frame {serve_start} to {serve_end} (length={serve_end-serve_start})")
    return serve_start, serve_end

def process_video(video_path):
    print(f"Processing {video_path}")

    cap = cv2.VideoCapture(video_path)
    all_rows = []

    header = ["frame_number", "timestamp_sec"]
    for lm in mp_pose.PoseLandmark:
        header += [f"{lm.name}_x", f"{lm.name}_y", f"{lm.name}_z"]

    with mp_pose.Pose(static_image_mode=False,
                      model_complexity=2,
                      enable_segmentation=False,
                      min_detection_confidence=0.8,
                      min_tracking_confidence=0.8) as pose:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            timestamp = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 3)
            h, w = frame.shape[:2]

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                row = [frame_count, timestamp]
                for lm in results.pose_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                all_rows.append(row)

    cap.release()

    if not all_rows:
        print(f"No pose data found for {video_path}. Skipping.")
        return

    serve_start, serve_end = detect_serve_window(all_rows, header)

    trimmed = all_rows[serve_start:serve_end + 1]

    base = os.path.basename(video_path).replace(".mp4", ".csv")
    output_csv = f"pose_data/{base}"

    os.makedirs("pose_data/", exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(trimmed)

    print(f"Wrote {output_csv}\n")

for path in video_paths:
    process_video(path)