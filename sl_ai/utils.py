import os
from pathlib import Path


def clean_listdir(path: Path):
    return list(filter(lambda item: not item.startswith("."), os.listdir(path)))
