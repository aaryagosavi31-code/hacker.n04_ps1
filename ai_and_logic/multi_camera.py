import cv2
import queue
import threading
from collections import deque
from ultralytics import YOLO

from input import frame_generator
from person_registry import PersonRegistry
# Import the MediaPipe cheating detection function
from cheat_detector import detect_cheating_mediapipe

class CameraWorker:

    def __init__(
        self,
        camera_id,
        source,
        registry,
        model_path="yolo11n-pose.pt",
        yolo_interval=2,
        mediapipe_interval=3,
        inference_size=416
    ):
        self.camera_id = camera_id
        self.source = source
        self.registry = registry
        self.model = YOLO(model_path)
        self.stopped = False
        self.yolo_interval = max(1, yolo_interval)
        self.mediapipe_interval = max(1, mediapipe_interval)
        self.inference_size = inference_size
        self.inference_queue = queue.Queue(maxsize=1)
        self.inference_thread = None
        self.fps = 30
        self.frame_buffer = deque(maxlen=self.fps * 5)
        self.event_active = False
        self.event_frames = []
        self.track_bench_map = {}

    def run(self):
        print(f"\n[{self.camera_id}] MediaPipe & YOLO Worker started...")

        self.inference_thread = threading.Thread(
            target=self._run_inference,
            name=f"inference-{self.camera_id}",
            daemon=True
        )
        self.inference_thread.start()
        for frame in frame_generator(self.source):
            if self.stopped:
                break

            self.frame_buffer.append(frame.copy())
            try:
                self.inference_queue.put_nowait(frame.copy())
            except queue.Full:
                try:
                    self.inference_queue.get_nowait()
                    self.inference_queue.put_nowait(frame.copy())
                except queue.Empty:
                    pass

            cv2.imshow(self.camera_id, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stopped = True
                break

        print(f"[{self.camera_id}] Worker stopped.")

    def _run_inference(self):
        frame_number = 0
        current_participant = "P001"
        result = None

        while not self.stopped:
            try:
                frame = self.inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            frame_number += 1

            if result is None or frame_number % self.yolo_interval == 0:
                results = self.model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    imgsz=self.inference_size,
                    max_det=4,
                    verbose=False
                )

                result = results[0]

                if result.boxes.id is not None:
                    track_ids = result.boxes.id.int().cpu().tolist()

                    participants = []

                    for track_id in track_ids:
                        bench_no = self.track_bench_map.setdefault(
                            track_id,
                            len(self.track_bench_map) + 1
                        )
                        participant_id = self.registry.get_participant(self.camera_id, track_id)

                        if participant_id is None:
                            participant_id = self.registry.register(
                                camera_id=self.camera_id,
                                track_id=track_id,
                                bench_no=bench_no
                            )

                        participants.append(participant_id)

                    if participants:
                        current_participant = participants[0]

            if frame_number % self.mediapipe_interval == 0:
                event = detect_cheating_mediapipe(
                    frame,
                    participant_id=current_participant
                )

                if event is not None:
                    print(f"[EVENT] {current_participant} -> {event}")
    def stop(self):
        self.stopped = True



class MultiCameraSystem:

    def __init__(self):
        self.registry = PersonRegistry()
        self.workers = {}
        self.threads = {}

    def add_camera(self, camera_id, source, model_path="yolo11n-pose.pt"):
        if camera_id in self.workers:
            raise ValueError(f"Camera {camera_id} is already registered")

        self.workers[camera_id] = CameraWorker(
            camera_id=camera_id,
            source=source,
            registry=self.registry,
            model_path=model_path
        )

    def register_participant(
        self,
        participant_id,
        camera_id,
        track_id,
        role_no=None,
        bench_no=None
    ):
        return self.registry.register(
            camera_id=camera_id,
            track_id=track_id,
            participant_id=participant_id,
            role_no=role_no,
            bench_no=bench_no
        )

    def start(self):
        for camera_id, worker in self.workers.items():
            thread = threading.Thread(
                target=worker.run,
                name=f"camera-{camera_id}",
                daemon=True
            )
            self.threads[camera_id] = thread
            thread.start()

        for thread in self.threads.values():
            thread.join()

    def stop(self):
        for worker in self.workers.values():
            worker.stop()

        for thread in self.threads.values():
            if thread.is_alive():
                thread.join(timeout=2)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = MultiCameraSystem()
    system.add_camera("CAM_01", 0)
    system.start()
