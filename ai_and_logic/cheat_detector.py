import os
import time
import cv2
import requests
from datetime import datetime

# Evidence images save location in backend directory
SNAPSHOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend/static/snapshots'))
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

BACKEND_API_URL = "http://localhost:5000/api/incident"

SUSPICION_THRESHOLD = 75.0
COOLDOWN_SECONDS = 5.0
BENCH_COOLDOWN_MAP = {}


def categorize_cheating_type(cheat_type_raw):
    mapping = {
        'phone': 'Unauthorised Material / Mobile Device',
        'head_turn': 'Excessive Head Movement / Looking Away',
        'whisper': 'Auditory Anomaly / Whispering Detected',
        'multiple_faces': 'Multiple People Detected in Frame',
        'no_face': 'Absence from Bench / Candidate Missing',
        'paper_pass': 'Physical Exchange / Paper Passing Detected',
        'hand_reach': 'Unauthorized Hand Extension Across Bench Zone'
    }
    return mapping.get(cheat_type_raw.lower().strip(), cheat_type_raw.title())


def process_cheating_event(bench_id, cheat_type_raw, confidence_score, frame):
    current_time = time.time()

    if confidence_score < SUSPICION_THRESHOLD:
        return None

    last_logged = BENCH_COOLDOWN_MAP.get(bench_id, 0)
    if (current_time - last_logged) < COOLDOWN_SECONDS:
        return None

    BENCH_COOLDOWN_MAP[bench_id] = current_time

    cheat_type = categorize_cheating_type(cheat_type_raw)
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    sanitized_type = cheat_type.replace(' ', '_').replace('/', '').replace('\\', '')
    filename = f"{bench_id}_{sanitized_type}_{timestamp_str}.jpg"
    filepath = os.path.join(SNAPSHOT_DIR, filename)

    # Save evidence frame
    cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    relative_image_url = f"/static/snapshots/{filename}"

    # Send payload to Flask API
    payload = {
        'bench_id': bench_id,
        'student_id': bench_id,
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