from person_registry import PersonRegistry


registry = PersonRegistry()


# ============================================================
# REGISTER PARTICIPANT
# ============================================================

registry.register(
    participant_id="P001",
    camera_id="CAM_01",
    track_id=1,
    role_no=3,
    bench_no=12
)


# ============================================================
# REGISTER SECOND PARTICIPANT
# ============================================================

registry.register(
    participant_id="P002",
    camera_id="CAM_01",
    track_id=2,
    role_no=5,
    bench_no=14
)


# ============================================================
# SHOW REGISTRY
# ============================================================

print("\nINITIAL REGISTRY:")

for participant_id, data in registry.all_participants().items():

    print(
        participant_id,
        "->",
        data
    )


# ============================================================
# SIMULATE TRACK ID CHANGE
#
# P001 was Track 1.
# Now the tracker identifies them as Track 30.
# ============================================================

print("\nUPDATING P001 TRACK...")

registry.update_track(
    "P001",
    "CAM_01",
    30
)


# ============================================================
# CHECK RESULT
# ============================================================

print("\nUPDATED REGISTRY:")

for participant_id, data in registry.all_participants().items():

    print(
        participant_id,
        "->",
        data
    )


# ============================================================
# TEST LOOKUP
# ============================================================

print("\nLOOKUP TEST:")

print(
    "CAM_01 + Track 30 ->",
    registry.get_participant(
        "CAM_01",
        30
    )
)

print(
    "CAM_01 + Track 1 ->",
    registry.get_participant(
        "CAM_01",
        1
    )
)