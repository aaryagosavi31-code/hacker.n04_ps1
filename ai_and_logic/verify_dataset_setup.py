import os

DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset_raw'))

CLASSES = ['0_normal', '1_head_turn', '2_paper_pass', '3_missing', '4_phone']

def verify_setup():
    print("=" * 50)
    print("DATASET ENVIRONMENT CHECK (.mkv format)")
    print("=" * 50)
    
    if not os.path.exists(DATASET_DIR):
        print(f"[!] Directory '{DATASET_DIR}' not found. Creating it now...")
        os.makedirs(DATASET_DIR, exist_ok=True)

    total_videos = 0
    for class_name in CLASSES:
        class_path = os.path.join(DATASET_DIR, class_name)
        os.makedirs(class_path, exist_ok=True)
        
        # Scans for .mkv along with other common video extensions
        video_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.mkv', '.mp4', '.avi', '.mov'))]
        count = len(video_files)
        total_videos += count
        print(f" -> Class '{class_name}': {count} video(s) found.")

    print("-" * 50)
    print(f"Total raw training videos found: {total_videos}")
    print("=" * 50)

if __name__ == "__main__":
    verify_setup()