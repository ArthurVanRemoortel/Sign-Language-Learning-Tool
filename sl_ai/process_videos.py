"""
Makes raw videos files in dataset ready to be used. Required FFMPEG to be installed on your system.
Currently, only resizes.
"""
import os
from sl_ai import ffmpeg_resize
from pathlib import Path

HEIGHT = 360
WIDTH = 640

if __name__ == '__main__':
    ROOT_DIRS = [
        Path('./ai_data/vgt-train'),
        Path('./ai_data/vgt-test'),
        Path('./ai_data/vgt-all'),
        Path('./ai_data/camera_recordings')
    ]

    for ROOT_DIR in ROOT_DIRS:
        if not ROOT_DIR.exists():
            print(f"Skipping {ROOT_DIR}. It does not exist.")
            continue
        OUT_DIR = Path(f'./{ROOT_DIR.parent / ROOT_DIR.name}-{HEIGHT}')
        os.makedirs(OUT_DIR, exist_ok=True)
        for gesture_name in os.listdir(ROOT_DIR):
            gesture_dir = ROOT_DIR / gesture_name
            gesture_out_dir = OUT_DIR / gesture_name
            os.makedirs(gesture_out_dir, exist_ok=True)

            for gesture_video_file in os.listdir(gesture_dir):
                print(f"Processing {str(gesture_dir / gesture_video_file)}...")
                # clip = mp.VideoFileClip(str(gesture_dir / gesture_video_file))
                ffmpeg_resize(
                    vid_path=str(gesture_dir / gesture_video_file),
                    output_path=str(gesture_out_dir / gesture_video_file),
                    width=WIDTH,
                    height=HEIGHT,
                )
