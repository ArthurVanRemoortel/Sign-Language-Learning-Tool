import os
from pathlib import Path

VIDEO_EXTENSIONS = {'mp4', 'mkv', 'mov'}


def clean_listdir(path: Path):
    return list(filter(lambda item: not item.startswith("."), os.listdir(path)))


def is_video(path: Path | str):
    if isinstance(path, Path):
        ext = path.suffix[1:].lower()
    else:
        ext = path.split('.')[-1].lower()
    return ext in VIDEO_EXTENSIONS
