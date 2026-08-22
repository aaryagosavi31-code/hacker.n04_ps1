import cv2
import threading

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
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return self.ret, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.cap.release()


def frame_generator(source=0):
    stream = VideoStream(source)
    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                continue
            yield frame
    finally:
        stream.stop()