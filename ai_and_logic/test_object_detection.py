import cv2
from ultralytics import YOLO

from input import frame_generator, get_source


# ============================================================
# LOAD OBJECT DETECTION MODEL
# ============================================================

model = YOLO("yolo11n.pt")


# ============================================================
# GET VIDEO SOURCE
# ============================================================

source = get_source()


# ============================================================
# PROCESS VIDEO
# ============================================================

for frame in frame_generator(source):

    results = model(
        frame,
        verbose=False
    )

    result = results[0]

    # --------------------------------------------------------
    # DETECTED OBJECTS
    # --------------------------------------------------------

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            # ------------------------------------------------
            # ONLY PRINT RELEVANT OBJECTS FOR NOW
            # ------------------------------------------------

            if (
                class_name =="cell phone"
                and confidence >=0.30
            ):

                print(
                    "DETECTED:",
                    class_name,
                    "| Confidence:",
                    round(confidence, 2)
                )

    # --------------------------------------------------------
    # SHOW DETECTIONS
    # --------------------------------------------------------

    annotated_frame = result.plot()

    cv2.imshow(
        "Object Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


cv2.destroyAllWindows()