import cv2
import threading
from ultralytics import YOLO

from input import frame_generator
from person_registry import PersonRegistry


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

        # Each camera gets its own YOLO instance
        self.model = YOLO(model_path)

        self.stopped = False


    # ============================================================
    # PROCESS CAMERA
    # ============================================================

    def run(self):

        print(
            f"[{self.camera_id}] Worker started"
        )

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


            # ----------------------------------------------------
            # Get tracked people
            # ----------------------------------------------------

            if (
                result.boxes.id is not None
                and result.keypoints is not None
            ):

                track_ids = (
                    result.boxes.id
                    .int()
                    .cpu()
                    .tolist()
                )

                keypoints = result.keypoints.xy


                for i, track_id in enumerate(track_ids):

                    # Check whether this track is registered
                    participant_id = (
                        self.registry.get_participant(
                            self.camera_id,
                            track_id
                        )
                    )


                    if participant_id is not None:

                        print(
                            f"[{self.camera_id}] "
                            f"{participant_id} "
                            f"-> Track {track_id}"
                        )

                    else:

                        print(
                            f"[{self.camera_id}] "
                            f"UNREGISTERED "
                            f"-> Track {track_id}"
                        )


            # ----------------------------------------------------
            # Display
            # ----------------------------------------------------

            annotated_frame = result.plot()

            cv2.imshow(
                self.camera_id,
                annotated_frame
            )


            # ----------------------------------------------------
            # Q stops all camera workers
            # ----------------------------------------------------

            if cv2.waitKey(1) & 0xFF == ord("q"):

                self.stopped = True
                break


        print(
            f"[{self.camera_id}] Worker stopped"
        )


    def stop(self):

        self.stopped = True


# =================================================================
# MULTI-CAMERA SYSTEM
# =================================================================

class MultiCameraSystem:

    def __init__(self):

        self.registry = PersonRegistry()

        self.cameras = {}

        self.threads = []


    # ============================================================
    # ADD CAMERA
    # ============================================================

    def add_camera(
        self,
        camera_id,
        source
    ):

        if camera_id in self.cameras:

            print(
                f"{camera_id} already exists."
            )

            return False


        worker = CameraWorker(
            camera_id=camera_id,
            source=source,
            registry=self.registry
        )


        self.cameras[camera_id] = worker

        print(
            f"Added {camera_id}"
        )

        return True


    # ============================================================
    # REGISTER PARTICIPANT
    # ============================================================

    def register_participant(
        self,
        participant_id,
        camera_id,
        track_id,
        role_no=None,
        bench_no=None
    ):

        return self.registry.register(
            participant_id=participant_id,
            camera_id=camera_id,
            track_id=track_id,
            role_no=role_no,
            bench_no=bench_no
        )


    # ============================================================
    # START ALL CAMERAS
    # ============================================================

    def start(self):

        print(
            "\nStarting all camera workers...\n"
        )


        for camera_id, worker in self.cameras.items():

            thread = threading.Thread(
                target=worker.run,
                daemon=True
            )

            self.threads.append(thread)

            thread.start()


        # Wait for workers
        for thread in self.threads:

            thread.join()


    # ============================================================
    # STOP ALL CAMERAS
    # ============================================================

    def stop(self):

        for worker in self.cameras.values():

            worker.stop()


        cv2.destroyAllWindows()