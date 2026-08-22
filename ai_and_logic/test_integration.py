import cv2
from ultralytics import YOLO
from input import frame_generator, get_source
from person_registry import PersonRegistry


# ============================================================
# SETUP
# ============================================================

model = YOLO("yolo11n-pose.pt")

registry = PersonRegistry()

camera_id = "CAM_01"


# ============================================================
# PROCESS ONE FRAME
# ============================================================

def process_frame(frame, camera_id):

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    tracking_data = []

    result = results[0]

    if result.boxes.id is not None and result.keypoints is not None:

        track_ids = result.boxes.id.int().cpu().tolist()
        keypoints = result.keypoints.xy

        for i, track_id in enumerate(track_ids):

            # ------------------------------------------------
            # Register this person if not already registered
            # ------------------------------------------------

            participant_id = registry.get_participant(
                camera_id,
                track_id
            )

            if participant_id is None:

                participant_id = registry.register(
                    camera_id,
                    track_id
                )

            # ------------------------------------------------
            # Create structured tracking data
            # ------------------------------------------------

            person_data = {
                "participant_id": participant_id,
                "camera_id": camera_id,
                "track_id": track_id,
                "keypoints": keypoints[i].cpu().tolist()
            }

            tracking_data.append(person_data)

    return results, tracking_data


# ============================================================
# MAIN
# ============================================================

source = get_source()

for frame in frame_generator(source):

    results, tracking_data = process_frame(
        frame,
        camera_id
    )

    # --------------------------------------------------------
    # Print newly created participant information
    # --------------------------------------------------------

    for person in tracking_data:

        print(
            person["participant_id"],
            "->",
            person["camera_id"],
            "+ Track",
            person["track_id"]
        )

    cv2.imshow(
        "YOLO Tracking",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cv2.destroyAllWindows()


# ============================================================
# SHOW FINAL REGISTRY
# ============================================================

print("\nFINAL PARTICIPANT REGISTRY:")

for participant_id, data in registry.all_participants().items():

    print(
        participant_id,
        "->",
        data
    )