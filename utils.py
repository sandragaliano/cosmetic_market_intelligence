import os
import pandas as pd
from datetime import datetime

# FOLDERS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_FOLDER = os.path.join(BASE_DIR, 'videos')
AUDIO_FOLDER = os.path.join(BASE_DIR, 'audios')
DATA_FOLDER = os.path.join(BASE_DIR, 'data')
CLEAN_DATA_FOLDER = os.path.join(DATA_FOLDER, 'clean_data')

# FILES
URL_DATA = os.path.join(DATA_FOLDER, 'url_data.xlsx')
URL_DATA_CLEANED = os.path.join(CLEAN_DATA_FOLDER, 'url_data_cleaned.xlsx')
SEPHORA_DATA_CLEANED = os.path.join(CLEAN_DATA_FOLDER, 'sephora_website_cleaned.csv')


def ensure_directories():
    """Crea las carpetas necesarias si no existen."""
    for folder in [VIDEO_FOLDER, AUDIO_FOLDER]:
        os.makedirs(folder, exist_ok=True)
