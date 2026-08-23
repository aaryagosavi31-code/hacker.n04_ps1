import os
import queue
import threading
import traceback
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import cv2

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
RECORDINGS_DIR = BACKEND_DIR / "static" / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

import sys

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from db import save_recording_to_db

VIDEO_PRE_SECONDS = 5
VIDEO_POST_SECONDS = 5
VIDEO_COOLDOWN_SECONDS = 10
DEFAULT_FPS = 30


class EvidenceVideoRecorder:
    def __init__(self, fps=DEFAULT_FPS, rolling_buffer=None):
        self.fps = fps if fps and fps > 0 else DEFAULT_FPS
        self.lock = threading.Lock()
        self.rolling_frames = rolling_buffer if rolling_buffer is not None else deque()
        self.recording_active = False
        self.last_recording_at = None
        self.pending_recording = None
        self.post_frames = []
        self.post_queue = queue.Queue()
        print("[VIDEO] Recorder initialized")

    def add_frame(self, frame):
        if frame is None:
            return

        frame_copy = frame.copy()
        with self.lock:
            self.rolling_frames.append(frame_copy)
            max_frames = int(self.fps * VIDEO_PRE_SECONDS)
            while len(self.rolling_frames) > max_frames:
                self.rolling_frames.popleft()
            active = self.recording_active

        if active:
            self.post_queue.put(frame_copy)

    def trigger(self, participant_id, event_type, frame, detection_time=None):
        detection_time = detection_time or datetime.now()
        event_name = self._event_name(event_type)

        with self.lock:
            if self.recording_active:
                print("[VIDEO] Recording already active; event ignored")
                return False
            if (
                self.last_recording_at is not None
                and (detection_time - self.last_recording_at).total_seconds()
                < VIDEO_COOLDOWN_SECONDS
            ):
                print("[VIDEO] Recording cooldown active; event ignored")
                return False

            while True:
                try:
                    self.post_queue.get_nowait()
                except queue.Empty:
                    break

            pre_frames = [saved.copy() for saved in self.rolling_frames]
            detection_frame = frame.copy()
            if not pre_frames or pre_frames[-1].shape != detection_frame.shape:
                pre_frames.append(detection_frame)
            self.recording_active = True
            self.pending_recording = (
                participant_id,
                event_name,
                detection_time,
                pre_frames,
                detection_frame
            )
            self.post_frames = []
            self.last_recording_at = detection_time

        print(f"[VIDEO] Detection received: {participant_id} -> {event_name}")
        print(f"[VIDEO] Pre-event frames captured: {len(pre_frames)}")
        print("[VIDEO] Recording post-event frames...")
        threading.Thread(
            target=self._collect_post_frames,
            name="evidence-post-frame-collector",
            daemon=True
        ).start()
        return True

    def _collect_post_frames(self):
        target_frames = int(self.fps * VIDEO_POST_SECONDS)
        collected = []
        while len(collected) < target_frames:
            try:
                collected.append(self.post_queue.get(timeout=1))
            except queue.Empty:
                with self.lock:
                    if not self.recording_active:
                        return

        with self.lock:
            if not self.pending_recording:
                return
            participant_id, event_name, detection_time, pre_frames, detection_frame = self.pending_recording
            post_frames = [frame.copy() for frame in collected]
            self.recording_active = False
            self.pending_recording = None

        self.post_frames = post_frames
        self._write_recording(
            participant_id,
            event_name,
            detection_time,
            pre_frames,
            detection_frame,
            post_frames
        )

    def _write_recording(
        self,
        participant_id,
        event_name,
        detection_time,
        pre_frames,
        detection_frame,
        post_frames
    ):
        try:
            frames = pre_frames + [detection_frame] + post_frames
            if not frames:
                raise RuntimeError("No frames available for evidence video")

            height, width = frames[0].shape[:2]
            started_at = detection_time - timedelta(seconds=VIDEO_PRE_SECONDS)
            ended_at = detection_time + timedelta(seconds=VIDEO_POST_SECONDS)
            timestamp = detection_time.strftime("%Y%m%d_%H%M%S")
            filename = f"{participant_id}_{event_name}_{timestamp}.mp4"
            filepath = RECORDINGS_DIR / filename
            writer = cv2.VideoWriter(
                str(filepath),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError("Could not open video writer")

            print("[VIDEO] Finalizing recording...")
            try:
                for frame in frames:
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))
                    writer.write(frame)
            finally:
                writer.release()

            if not filepath.exists() or filepath.stat().st_size == 0:
                raise RuntimeError(f"Video file was not created correctly: {filepath}")

            duration = len(frames) / self.fps
            video_path = f"/static/recordings/{filename}"
            print(f"[VIDEO] Video saved: {filepath}")
            print(f"[VIDEO] Duration: {duration:.2f}")
            print(f"[VIDEO] File size: {filepath.stat().st_size} bytes")
            print("[VIDEO] Saving recording to PostgreSQL...")
            recording_id = save_recording_to_db(
                participant_id,
                started_at,
                ended_at,
                video_path,
                duration
            )
            if recording_id is None:
                raise RuntimeError("PostgreSQL recording insert failed")
            print(f"[VIDEO] PostgreSQL recording inserted successfully")
            print(f"[VIDEO] Database recording_id: {recording_id}")
        except Exception as error:
            print(f"[VIDEO ERROR] {error}")
            traceback.print_exc()

    @staticmethod
    def _event_name(event_type):
        if isinstance(event_type, dict):
            event_type = event_type.get("cheat_type", "incident")
        return str(event_type).lower().replace(" ", "_").replace("/", "")

    def stop(self):
        with self.lock:
            self.recording_active = False
            self.pending_recording = None
