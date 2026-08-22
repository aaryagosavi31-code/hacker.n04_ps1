from vision_system import VisionSystem


# ============================================================
# CREATE VISION SYSTEM
# ============================================================

vision = VisionSystem()


# ============================================================
# ADD TWO LOGICAL CAMERAS
#
# Both temporarily use the same webcam.
# This is ONLY an architecture test.
# ============================================================

vision.add_camera(
    "CAM_01",
    0
)

vision.add_camera(
    "CAM_02",
    0
)


# ============================================================
# REGISTER SAME TRACK NUMBER ON BOTH CAMERAS
# ============================================================

vision.register_participant(
    participant_id="P001",
    camera_id="CAM_01",
    track_id=1,
    role_no=3,
    bench_no=12
)

vision.register_participant(
    participant_id="P002",
    camera_id="CAM_02",
    track_id=1,
    role_no=4,
    bench_no=15
)


# ============================================================
# TEST REGISTRY
# ============================================================

print("\nMULTI-CAMERA TEST\n")


print(
    "CAM_01 + Track 1 ->",
    vision.registry.get_participant(
        "CAM_01",
        1
    )
)


print(
    "CAM_02 + Track 1 ->",
    vision.registry.get_participant(
        "CAM_02",
        1
    )
)


# ============================================================
# SHOW REGISTRY
# ============================================================

print("\nFINAL REGISTRY:")

for participant_id, data in vision.get_participants().items():

    print(
        participant_id,
        "->",
        data
    )