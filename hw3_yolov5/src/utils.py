import os
from pathlib import Path
from datetime import datetime

# Корень проекта (hw3_yolov5)
BASE_DIR = Path(__file__).resolve().parent.parent

def get_timestamp():
    """Возвращает строку с текущей датой и временем для меток"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_run_dir():
    """Создаёт папку results/run_<timestamp> и возвращает её путь"""
    timestamp = get_timestamp()
    run_dir = BASE_DIR / "results" / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir