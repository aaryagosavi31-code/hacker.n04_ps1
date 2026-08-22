import cv2
import threading
import requests
import os
from datetime import datetime
from ultralytics import YOLO

from input import frame_generator
from person_registry import PersonRegistry

BACKEND_URL = "http://localhost:5000/api/incident"


class CameraWorker:

    def __init__(
        self,
        camera_id,
        source,
        registry,
        model_path="yolo11n-pose.pt"
    ):
        self.camera_id = camera_id
        self.source = source
        self.registry = registry
        self.model = YOLO(model_path)
        self.stopped = False

    def send_cheat_alert(self, participant_id, cheat_type, confidence, frame):
        """Saves a snapshot locally and posts incident data to PostgreSQL via Flask API."""
        os.makedirs("../backend/static/snapshots", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{participant_id}_{timestamp}.jpg"
        file_path = os.path.join("../backend/static/snapshots", filename)
        snapshot_url = f"/static/snapshots/{filename}"

        cv2.imwrite(file_path, frame)

        payload = {
            "student_id": participant_id,
            "cheat_type": cheat_type,
            "confidence_score": float(confidence),
            "snapshot_path": snapshot_url
        }

        try:
            res = requests.post(BACKEND_URL, json=payload, timeout=2)
            if res.status_code == 201:
                print(f"\n[SUCCESS] Logged to DB for {participant_id}: {cheat_type}")
        except Exception as e:
            print(f"\n[ERROR] Backend connection failed: {e}")

    def run(self):
        print(f"\n[{self.camera_id}] Worker started")
        print("-----------------------------------")
        print("Press 'p' -> Simulate Mobile Phone Detection")
        print("Press 'h' -> Simulate Head Turn / Looking Away")
        print("Press 'x' -> Simulate Paper Passing")
        print("Press 'q' -> Stop Worker")
        print("-----------------------------------\n")

        for frame in frame_generator(self.source):
            if self.stopped:
                break

            results = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

            result = results[0]
            current_participants = []

            # Track detected people & assign P001, P002...
            if result.boxes.id is not None and result.keypoints is not None:
                track_ids = result.boxes.id.int().cpu().tolist()

                for track_id in track_ids:
                    participant_id = self.registry.get_participant(self.camera_id, track_id)

                    # Auto-register new tracks to permanent IDs
                    if participant_id is None:
                        participant_id = self.registry.register(
                            camera_id=self.camera_id,
                            track_id=track_id
                        )

                    current_participants.append(participant_id)

            annotated_frame = result.plot()
            cv2.imshow(self.camera_id, annotated_frame)

            # Keyboard Listener for Manual Testing
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                self.stopped = True
                break

            elif key in [ord('p'), ord('h'), ord('x')]:
                # Pick the first tracked person or fallback to P001
                target_id = current_participants[0] if current_participants else "P001"

                if key == ord('p'):
                    cheat_type = "Unauthorised Material / Mobile Device"
                    confidence = 92.5
                elif key == ord('h'):
                    cheat_type = "Head Turn / Looking Away"
                    confidence = 88.0
                elif key == ord('x'):
                    cheat_type = "Paper Passing / Communication"
                    confidence = 95.0

                print(f"\n[KEY PRESS DETECTED] Triggering alert for: {target_id}")
                self.send_cheat_alert(target_id, cheat_type, confidence, annotated_frame)

        print(f"[{self.camera_id}] Worker stopped")

    def stop(self):
        self.stopped = True


class MultiCameraSystem:

    def __init__(self):
        self.registry = PersonRegistry()
        self.cameras = {}
        self.threads = []

    def add_camera(self, camera_id, source):
        if camera_id in self.cameras:
            return False
        worker = CameraWorker(
            camera_id=camera_id,
            source=source,
            registry=self.registry
        )
        self.cameras[camera_id] = worker
        return True

    def start(self):
        for camera_id, worker in self.cameras.items():
            thread = threading.Thread(target=worker.run, daemon=True)
            self.threads.append(thread)
            thread.start()

        for thread in self.threads:
            thread.join()

    def stop(self):
        for worker in self.cameras.values():
            worker.stop()
        cv2.destroyAllWindows()