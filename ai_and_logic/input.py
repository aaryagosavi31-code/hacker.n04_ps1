import cv2
import threading
import time


# ======================================================================
# get_source - lets you choose the input type before anything opens
# ======================================================================
# cv2.VideoCapture(source) accepts THREE kinds of input:
#   - an integer (0, 1, 2...)  -> webcam
#   - a file path string       -> a media video file already on disk
#   - an RTSP/HTTP URL string  -> CCTV / IP camera footage
#
# NOTE ON AUDIO: OpenCV's VideoCapture only reads video frames, it does
# NOT read audio. If you ever need audio from a file/stream, that needs
# a separate library (e.g. pyaudio or moviepy) running alongside this -
# not something VideoCapture itself can give you.
def get_source():
    print("Choose input source:")
    print("1. Webcam")
    print("2. Media video file")
    print("3. CCTV / IP camera footage")
    choice = input("Enter 1/2/3: ")

    if choice == "1":
        return 0
    elif choice == "2":
        path = input("Enter video file path: ")
        return path
    elif choice == "3":
        url = input("Enter RTSP/HTTP URL: ")
        return url
    else:
        print("Invalid choice, defaulting to webcam")
        return 0


# ======================================================================
# VideoStream - background-threaded frame grabber (always keeps latest
# frame only, so nothing lags behind if processing is slower than the
# camera/video's frame rate)
# ======================================================================
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


# ======================================================================
# frame_generator - the handoff function your teammate (Role 2/YOLO)
# will loop over, e.g.:
#     for frame in frame_generator(source):
#         keypoints = yolo_model(frame)
# ======================================================================
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


# ======================================================================
# MAIN - single clean run: pick a source, then stream it with FPS shown
# ======================================================================
if __name__ == "__main__":
    source = get_source()
    prev_time = time.time()

    for frame in frame_generator(source):
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Output", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()