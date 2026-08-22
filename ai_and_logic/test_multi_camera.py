from multi_camera import MultiCameraSystem


# ============================================================
# CREATE SYSTEM
# ============================================================

system = MultiCameraSystem()


# ============================================================
# ADD CAMERAS
#
# Both temporarily use the same webcam.
#
# THIS IS ONLY AN ARCHITECTURE TEST.
# ============================================================

system.add_camera(
    "CAM_01",
    0
)

system.add_camera(
    "CAM_02",
    0
)


# ============================================================
# REGISTER PARTICIPANTS
#
# Notice both cameras can have Track 1.
# ============================================================

system.register_participant(
    participant_id="P001",
    camera_id="CAM_01",
    track_id=1,
    role_no=3,
    bench_no=12
)

system.register_participant(
    participant_id="P002",
    camera_id="CAM_02",
    track_id=1,
    role_no=4,
    bench_no=15
)


# ============================================================
# START ALL CAMERAS
# ============================================================

try:

    system.start()

finally:

    system.stop()


# ============================================================
# FINAL REGISTRY
# ============================================================

print(
    "\nFINAL PARTICIPANT REGISTRY:"
)


for participant_id, data in (
    system.registry.all_participants().items()
):

    print(
        participant_id,
        "->",
        data
    )