from seating_arrangement import get_student_id


class PersonRegistry:

    def __init__(self):

        # Permanent participant records
        #
        # Example:
        #
        # P001 -> {
        #     "role_no": 3,
        #     "bench_no": 12,
        #     "camera_id": "CAM_01",
        #     "track_id": 4
        # }
        self.participants = {}

        # Current tracking lookup
        #
        # (camera_id, track_id) -> participant_id
        self.track_lookup = {}

        # Used only when participant_id is not supplied
        self.next_id = 1


    # ============================================================
    # REGISTER PARTICIPANT
    # ============================================================

    def register(
        self,
        camera_id,
        track_id,
        participant_id=None,
        role_no=None,
        bench_no=None
    ):

        # --------------------------------------------------------
        # If no permanent ID was supplied, generate one
        # --------------------------------------------------------

        if participant_id is None:

            if bench_no is not None:
                participant_id = get_student_id(bench_no)

            if participant_id is None:
                participant_id = f"P{self.next_id:03d}"
                self.next_id += 1

        else:

            # Prevent duplicate participant registration
            if participant_id in self.participants:

                print(
                    f"{participant_id} is already registered."
                )

                return participant_id


        # --------------------------------------------------------
        # Prevent the same camera + track from being assigned
        # to two different participants
        # --------------------------------------------------------

        existing_participant = self.get_participant(
            camera_id,
            track_id
        )

        if existing_participant is not None:

            print(
                f"({camera_id}, Track {track_id}) "
                f"is already assigned to "
                f"{existing_participant}"
            )

            return existing_participant


        # --------------------------------------------------------
        # Store permanent participant
        # --------------------------------------------------------

        self.participants[participant_id] = {

            "role_no": role_no,

            "bench_no": bench_no,

            "camera_id": camera_id,

            "track_id": track_id
        }


        # --------------------------------------------------------
        # Store current tracking lookup
        # --------------------------------------------------------

        self.track_lookup[
            (camera_id, track_id)
        ] = participant_id


        print(
            f"REGISTERED: {participant_id} "
            f"-> {camera_id} + Track {track_id}"
        )


        return participant_id


    # ============================================================
    # GET PARTICIPANT FROM CURRENT TRACK
    # ============================================================

    def get_participant(
        self,
        camera_id,
        track_id
    ):

        return self.track_lookup.get(
            (camera_id, track_id)
        )


    # ============================================================
    # GET PARTICIPANT DETAILS
    # ============================================================

    def get_participant_data(
        self,
        participant_id
    ):

        return self.participants.get(
            participant_id
        )


    # ============================================================
    # UPDATE CURRENT TRACK
    #
    # Permanent participant ID stays the same.
    # ============================================================

    def update_track(
        self,
        participant_id,
        new_camera_id,
        new_track_id
    ):

        participant = self.participants.get(
            participant_id
        )

        if participant is None:

            print(
                f"{participant_id} does not exist."
            )

            return False


        # --------------------------------------------------------
        # Remove old tracking lookup
        # --------------------------------------------------------

        old_key = (
            participant["camera_id"],
            participant["track_id"]
        )

        self.track_lookup.pop(
            old_key,
            None
        )


        # --------------------------------------------------------
        # Update participant's current tracking information
        # --------------------------------------------------------

        participant["camera_id"] = new_camera_id

        participant["track_id"] = new_track_id


        # --------------------------------------------------------
        # Create new lookup
        # --------------------------------------------------------

        self.track_lookup[
            (new_camera_id, new_track_id)
        ] = participant_id


        print(
            f"UPDATED: {participant_id} "
            f"-> {new_camera_id} + Track {new_track_id}"
        )


        return True


    # ============================================================
    # UPDATE ROLE / BENCH
    # ============================================================

    def update_details(
        self,
        participant_id,
        role_no=None,
        bench_no=None
    ):

        participant = self.participants.get(
            participant_id
        )

        if participant is None:
            return False


        if role_no is not None:

            participant["role_no"] = role_no


        if bench_no is not None:

            participant["bench_no"] = bench_no


        return True


    # ============================================================
    # REMOVE PARTICIPANT
    # ============================================================

    def remove(
        self,
        participant_id
    ):

        participant = self.participants.pop(
            participant_id,
            None
        )

        if participant is None:
            return False


        old_key = (
            participant["camera_id"],
            participant["track_id"]
        )

        self.track_lookup.pop(
            old_key,
            None
        )


        print(
            f"REMOVED: {participant_id}"
        )


        return True


    # ============================================================
    # GET ALL PARTICIPANTS
    # ============================================================

    def all_participants(self):

        return self.participants.copy()