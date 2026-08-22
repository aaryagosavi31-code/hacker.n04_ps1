import os
import sys

# 1. Suppress MediaPipe & TensorFlow C++ log spam
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import glob
import time
import cv2
import h5py
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
FACE_MODEL_PATH = os.path.join(MODEL_DIR, 'face_landmarker.task')
POSE_MODEL_PATH = os.path.join(MODEL_DIR, 'pose_landmarker_lite.task')

DATASET_RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset_raw'))
OUTPUT_HDF5_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'cheating_keypoints.h5'))

CLASSES = {
    '0_normal': 0,
    '1_phone': 1,
    '1_head_turn': 2,
    '2_paper_pass': 3,
    '3_missing': 4
}

FRAME_STEP = 4  # Subsample every 4th frame for high processing speed


def extract_frame_landmarks(frame, fm_detector, pose_detector):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    f_res = fm_detector.detect(image)
    p_res = pose_detector.detect(image)

    row = []

    # 1. Face Landmarks (5 key points -> 15 features)
    if f_res.face_landmarks:
        face_lms = f_res.face_landmarks[0]
        for idx in [1, 33, 263, 61, 291]:
            lm = face_lms[idx]
            row.extend([lm.x, lm.y, lm.z])
    else:
        row.extend([0.0] * 15)

    # 2. Pose Landmarks (13 key upper body points -> 52 features)
    if p_res.pose_landmarks:
        pose_lms = p_res.pose_landmarks[0]
        target_indices = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]
        for idx in target_indices:
            lm = pose_lms[idx]
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        row.extend([0.0] * 52)

    return np.array(row, dtype=np.float32)


def process_all_videos():
    print("=" * 60)
    print("STARTING HDF5 FEATURE EXTRACTION (FAST MODE)")
    print(f"Dataset root: {DATASET_RAW_DIR}")
    print("=" * 60)

    # MediaPipe 0.10.35 uses the Tasks API instead of mp.solutions.
    fm_detector = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=FACE_MODEL_PATH),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
    )
    pose_detector = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=POSE_MODEL_PATH),
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
    )

    if os.path.exists(OUTPUT_HDF5_PATH):
        os.remove(OUTPUT_HDF5_PATH)

    h5_file = h5py.File(OUTPUT_HDF5_PATH, 'w')
    total_processed = 0

    all_found_videos = []
    for root, _, files in os.walk(DATASET_RAW_DIR):
        for f in files:
            if f.lower().endswith(('.mkv', '.mp4', '.avi', '.mov')):
                all_found_videos.append(os.path.join(root, f))

    print(f"[*] Total video files detected inside dataset_raw: {len(all_found_videos)}")

    for vid_path in all_found_videos:
        file_name = os.path.basename(vid_path).lower()
        parent_folder = os.path.basename(os.path.dirname(vid_path)).lower()

        assigned_label = None
        assigned_class = None

        for class_name, class_code in CLASSES.items():
            if class_name in parent_folder or class_name in file_name or class_name.split('_')[1] in file_name:
                assigned_label = class_code
                assigned_class = class_name
                break

        if assigned_label is None:
            assigned_label = 1 if 'phone' in file_name else 0
            assigned_class = '1_phone' if assigned_label == 1 else '0_normal'

        print(f"\n[+] Processing: {os.path.basename(vid_path)} -> Class: '{assigned_class}' (Label {assigned_label})")

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f" [!] Could not open video: {os.path.basename(vid_path)}")
            continue

        sequence = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_STEP == 0:
                # Downscale large frames to 640px width for 4x faster CPU processing
                h, w = frame.shape[:2]
                if w > 640:
                    frame = cv2.resize(frame, (640, int(h * (640 / w))))

                lm_vector = extract_frame_landmarks(frame, fm_detector, pose_detector)
                sequence.append(lm_vector)

            frame_idx += 1
            if frame_idx % 40 == 0:
                print(f"   ...processed {frame_idx} frames", end="\r", flush=True)

        cap.release()

        if len(sequence) >= 10:
            seq_matrix = np.array(sequence, dtype=np.float32)
            dataset_name = f"video_{total_processed}"
            dset = h5_file.create_dataset(dataset_name, data=seq_matrix, compression="gzip")
            dset.attrs['label'] = assigned_label
            dset.attrs['file_name'] = os.path.basename(vid_path)

            print(f" -> Saved {dataset_name}: shape {seq_matrix.shape}, label {assigned_label}")
            total_processed += 1
        else:
            print(f" [!] Skipped {os.path.basename(vid_path)}: Too short ({len(sequence)} frames)")

    h5_file.close()
    print("\n" + "=" * 60)
    print(f"EXTRACTION COMPLETE! Total videos processed: {total_processed}")
    print(f"Dataset saved to: {OUTPUT_HDF5_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    process_all_videos()