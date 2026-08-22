from vision_system import VisionSystem


def main():

    system = VisionSystem()

    print("=== DRISTI AI - Vision System ===")
    print("1. Run Webcam")
    print("2. Run Demo Video File")

    choice = input("Select input source (1 or 2): ").strip()

    if choice == "1":

        system.add_camera(
            "CAM_01",
            0
        )

    elif choice == "2":

        video_path = input(
            "Enter video file path: "
        ).strip()

        system.add_camera(
            "CAM_01",
            video_path
        )

    else:

        print("Invalid choice.")
        return

    try:

        system.run_camera("CAM_01")

    except KeyboardInterrupt:

        print("\nStopping vision system...")


if __name__ == "__main__":
    main()