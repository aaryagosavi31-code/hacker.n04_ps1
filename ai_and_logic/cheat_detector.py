import os
import time
import cv2
import requests
import numpy as np
import mediapipe as mp
from urllib.request import urlretrieve
from datetime import datetime

# MediaPipe 0.10.35 removed the legacy solutions API. Keep both paths so the
# detector works with older environments and current installations.
try:
    mp_face_mesh = mp.solutions.face_mesh  # type: ignore
    mp_pose = mp.solutions.pose            # type: ignore
    USE_TASKS_API = False
except AttributeError:
    from mediapipe.tasks import python as mp_tasks  # type: ignore
    from mediapipe.tasks.python import vision as mp_vision  # type: ignore
    USE_TASKS_API = True

# Initialize models
if USE_TASKS_API:
    MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(MODEL_DIR, exist_ok=True)

    face_model = os.path.join(MODEL_DIR, 'face_landmarker.task')
    pose_model = os.path.join(MODEL_DIR, 'pose_landmarker_lite.task')
    if not os.path.exists(face_model):
        urlretrieve(
            'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
            face_model
        )
    if not os.path.exists(pose_model):
        urlretrieve(
            'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
            pose_model
        )

    face_mesh = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=face_model),
            num_faces=4,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
    )
    pose_detector = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=pose_model),
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
    )
else:
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=4,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    pose_detector = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
# Evidence images save location in backend directory
SNAPSHOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend/static/snapshots'))
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

BACKEND_API_URL = "http://localhost:5000/api/incident"

SUSPICION_THRESHOLD = 75.0
COOLDOWN_SECONDS = 4.0
PARTICIPANT_COOLDOWN_MAP = {}
MISSING_FRAME_COUNTERS = {}


def categorize_cheating_type(cheat_type_raw):
    mapping = {
        'head_turn': 'Excessive Head Movement / Looking Away',
        'whisper': 'Auditory Anomaly / Whispering Detected',
        'multiple_faces': 'Multiple People Detected in Frame',
        'no_face': 'Absence from Bench / Candidate Missing',
        'paper_pass': 'Physical Exchange / Paper Passing Detected',
        'hand_reach': 'Unauthorized Hand Extension Across Bench Zone'
    }
    return mapping.get(cheat_type_raw.lower().strip(), cheat_type_raw.title())


def process_cheating_event(participant_id, cheat_type_raw, confidence_score, frame):
    current_time = time.time()

    if confidence_score < SUSPICION_THRESHOLD:
        return None

    last_logged = PARTICIPANT_COOLDOWN_MAP.get(participant_id, 0)
    if (current_time - last_logged) < COOLDOWN_SECONDS:
        return None

    PARTICIPANT_COOLDOWN_MAP[participant_id] = current_time

    cheat_type = categorize_cheating_type(cheat_type_raw)
    print(
        f"[CHEATING DETECTED] student={participant_id} "
        f"type={cheat_type} confidence={confidence_score:.1f}%"
    )
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    sanitized_type = cheat_type.replace(' ', '_').replace('/', '').replace('\\', '')
    filename = f"{participant_id}_{sanitized_type}_{timestamp_str}.jpg"
    filepath = os.path.join(SNAPSHOT_DIR, filename)

    # Save evidence frame
    cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    relative_image_url = f"/static/snapshots/{filename}"

    # Send payload to Flask API
    payload = {
        'student_id': participant_id,
        'cheat_type': cheat_type,
        'confidence_score': round(confidence_score, 2),
        'snapshot_path': relative_image_url
    }

    try:
        response = requests.post(BACKEND_API_URL, json=payload, timeout=2)
        if response.status_code == 201:
            print(f"[AI & LOGIC] Incident logged & broadcast: {payload}")
            return response.json()
    except Exception as e:
        print(f"[AI & LOGIC ERROR] Could not reach backend server: {e}")

    return None


def detect_cheating_mediapipe(frame, participant_id="P001"):
    """
    Evaluates video frame for behavior-based cheating modalities:
    - Head rotation and side looking
    - Paper passing or cross-body arm extension
    - Multiple faces in view
    - Candidate absence
    """
    if frame is None:
        return

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 1. FACE MESH ANALYSIS (Head Turn & Multiple Faces)
    if USE_TASKS_API:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        face_results = face_mesh.detect(image)
        detected_faces = face_results.face_landmarks
    else:
        face_results = face_mesh.process(rgb_frame)
        detected_faces = face_results.multi_face_landmarks

    if detected_faces:
        MISSING_FRAME_COUNTERS[participant_id] = 0

        # Detect multiple people in frame
        if len(detected_faces) > 1:
            process_cheating_event(participant_id, 'multiple_faces', 95.0, frame)
            cv2.putText(
                frame,
                "WARNING: MULTIPLE FACES",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        for face_landmarks in detected_faces:
            landmarks = face_landmarks if USE_TASKS_API else face_landmarks.landmark
            nose_tip = landmarks[1]
            left_eye = landmarks[33]
            right_eye = landmarks[263]

            # Head Turn Check (Horizontal symmetry ratio)
            dist_left = abs((nose_tip.x * w) - (left_eye.x * w))
            dist_right = abs((nose_tip.x * w) - (right_eye.x * w))

            if dist_right > 0.001:
                yaw_ratio = dist_left / dist_right
                if yaw_ratio > 1.8 or yaw_ratio < 0.45:
                    print(
                        f"\n[FLAGGED] Head turn detected for {participant_id}! "
                        f"Ratio: {round(yaw_ratio, 2)}"
                    )
                    process_cheating_event(participant_id, 'head_turn', 88.0, frame)
                    cv2.putText(
                        frame,
                        f"ALERT: HEAD TURN DETECTED ({participant_id})",
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )
                    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
    else:
        MISSING_FRAME_COUNTERS[participant_id] = MISSING_FRAME_COUNTERS.get(participant_id, 0) + 1
        if MISSING_FRAME_COUNTERS[participant_id] > 30:
            process_cheating_event(participant_id, 'no_face', 90.0, frame)
            cv2.putText(
                frame,
                f"ALERT: CANDIDATE MISSING ({participant_id})",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    # 2. POSE ANALYSIS (Hand Near Head / Phone Use)
    if USE_TASKS_API:
        pose_results = pose_detector.detect(image)
        detected_pose = pose_results.pose_landmarks[0] if pose_results.pose_landmarks else None
        nose_index, left_wrist_index, right_wrist_index = 0, 15, 16
    else:
        pose_results = pose_detector.process(rgb_frame)
        detected_pose = pose_results.pose_landmarks.landmark if pose_results.pose_landmarks else None
        nose_index = mp_pose.PoseLandmark.NOSE
        left_wrist_index = mp_pose.PoseLandmark.LEFT_WRIST
        right_wrist_index = mp_pose.PoseLandmark.RIGHT_WRIST

    if detected_pose:
        landmarks = detected_pose
        nose_y = landmarks[nose_index].y
        left_wrist = landmarks[left_wrist_index]
        right_wrist = landmarks[right_wrist_index]

        # Trigger if either hand/wrist goes up near face level
        if (left_wrist.y < nose_y + 0.1 and left_wrist.visibility > 0.5) or \
           (right_wrist.y < nose_y + 0.1 and right_wrist.visibility > 0.5):
            process_cheating_event(participant_id, 'phone', 92.0, frame)
            cv2.putText(
                frame,
                f"ALERT: PHONE / HAND NEAR FACE ({participant_id})",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
            hand_reaching_across = (
                left_wrist.visibility > 0.5 and left_wrist.x > right_shoulder.x
            ) or (
                right_wrist.visibility > 0.5 and right_wrist.x < left_shoulder.x
            )
            if hand_reaching_across:
                process_cheating_event(participant_id, 'paper_pass', 94.0, frame)
                cv2.putText(
                    frame,
                    f"ALERT: PAPER PASSING / HAND REACH ({participant_id})",
                    (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )