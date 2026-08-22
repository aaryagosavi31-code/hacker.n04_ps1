import cv2
import threading
import time

def get_source():
    print("\nSelect Camera Input Source:")
    print("1. Webcam")
    print("2. Media video file")
    print("3. CCTV / IP camera footage (RTSP/HTTP)")
    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == "1":
        return 0
    elif choice == "2":
        return input("Enter video file path: ").strip()
    elif choice == "3":
        return input("Enter RTSP/HTTP URL: ").strip()
    else:
        print("Invalid choice. Defaulting to Webcam (0).")
        return 0


class VideoStream:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()
        self.frame_id = 0
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            with self.lock:
                self.ret, self.frame = ret, frame
                self.frame_id += 1

    def read(self):
        with self.lock:
            if self.frame is None:
                return self.ret, None
            return self.ret, self.frame.copy()

    def read_with_id(self):
        with self.lock:
            if self.frame is None:
                return self.ret, None, self.frame_id
            return self.ret, self.frame.copy(), self.frame_id

    def stop(self):
        self.stopped = True
        self.cap.release()


def frame_generator(source=0):
    stream = VideoStream(source)
    last_frame_id = -1
    try:
        while True:
            ret, frame, frame_id = stream.read_with_id()
            if not ret or frame is None:
                time.sleep(0.001)
                continue
            if frame_id == last_frame_id:
                time.sleep(0.001)
                continue
            last_frame_id = frame_id
            yield frame
    finally:
        stream.stop()