import cv2
from ultralytics import YOLO

from input import frame_generator
from person_registry import PersonRegistry


class VisionSystem:

    def __init__(self, model_path="yolo11n-pose.pt"):

        # --------------------------------------------------------
        # YOLO model
        # --------------------------------------------------------

        self.model = YOLO(model_path)

        # --------------------------------------------------------
        # Permanent participant registry
        # --------------------------------------------------------

        self.registry = PersonRegistry()

        # --------------------------------------------------------
        # Camera configuration
        #
        # Example:
        #
        #{
        #     "CAM_01": 0
        # }     --------------------------------------------------------

        self.cameras = {}


    # ============================================================
    # ADD CAMERA
    # ============================================================

    def add_camera(self, camera_id, source):

        if camera_id in self.cameras:

            print(
                f"Camera {camera_id} already exists."
            )

            return False

        self.cameras[camera_id] = {
            "source": source
        }

        print(
            f"Added camera: {camera_id}"
        )

        return True


    # ============================================================
    # REGISTER PARTICIPANT
    #
    # This is now the ONLY place where a permanent participant
    # should be created.
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
    # PROCESS ONE FRAME
    # ============================================================


    def process_frame(self, frame, camera_id):

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        tracking_data = []

        result = results[0]

        # --------------------------------------------------------
        # Check if people were detected
        # --------------------------------------------------------

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

            # ----------------------------------------------------
            # Process every tracked person
            # ----------------------------------------------------

            for i, track_id in enumerate(track_ids):

                # ------------------------------------------------
                # Check whether this track belongs to a registered
                # participant.
                # ------------------------------------------------

                participant_id = self.registry.get_participant(
                    camera_id,
                    track_id
                )

                if participant_id is None:
                    participant_id = self.registry.register(
                        camera_id,
                        track_id
                    )

                # ------------------------------------------------
                # Get participant details
                # ------------------------------------------------

                participant_data = None

                if participant_id is not None:

                    participant_data = (
                        self.registry.get_participant_data(
                            participant_id
                        )
                    )

                # ------------------------------------------------
                # Build structured tracking data
                # ------------------------------------------------

                person_data = {

                    "participant_id": participant_id,

                    "camera_id": camera_id,

                    "track_id": track_id,

                    "role_no": (
                        participant_data["role_no"]
                        if participant_data is not None
                        else None
                    ),

                    "bench_no": (
                        participant_data["bench_no"]
                        if participant_data is not None
                        else None
                    ),

                    "keypoints": (
                        keypoints[i]
                        .cpu()
                        .tolist()
                    )
                }

                tracking_data.append(
                    person_data
                )

        return results, tracking_data

    def run_camera(self, camera_id):

        if camera_id not in self.cameras:

            print(
                f"Camera {camera_id} not found."
            )

            return


        source = self.cameras[camera_id]["source"]


        print(
            f"Starting camera: {camera_id}"
        )


        for frame in frame_generator(source):

            results, tracking_data = (
                self.process_frame(
                    frame,
                    camera_id
                )
            )


            # ----------------------------------------------------
            # Display tracking information
            # ----------------------------------------------------

            for person in tracking_data:

                if person["participant_id"] is not None:

                    print(
                    person["participant_id"],
                    "->",
                    person["camera_id"],
                    "+ Track",
                    person["track_id"]
                    )
                else:
                    print(
                        "UNREGISTERED ->",
                        person["camera_id"],
                        "+ Track",
                        person["track_id"]
                    )

            # ----------------------------------------------------
            # Display YOLO output
            # ----------------------------------------------------

            annotated_frame = (
                results[0].plot()
            )


            cv2.imshow(
                camera_id,
                annotated_frame
            )


            # ----------------------------------------------------
            # Press Q to stop
            # ----------------------------------------------------

            if cv2.waitKey(1) & 0xFF == ord("q"):

                break


        cv2.destroyWindow(
            camera_id
        )


    # ============================================================
    # GET ALL PARTICIPANTS
    # ============================================================

    def get_participants(self):

        return self.registry.all_participants()