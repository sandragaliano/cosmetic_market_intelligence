import os
import pandas as pd
from datetime import datetime
from datetime import datetime

# DIRECTORIES
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BERT_DIR = os.path.join(BASE_DIR, 'analysis', 'Bert')

# FOLDERS
VIDEO_FOLDER = os.path.join(BASE_DIR, 'videos')
AUDIO_FOLDER = os.path.join(BASE_DIR, 'audios')
DATA_FOLDER = os.path.join(BASE_DIR, 'data')
CLEAN_DATA_FOLDER = os.path.join(DATA_FOLDER, 'clean_data')
DATA_FOR_BERT_DIR = os.path.join(BASE_DIR, 'data_for_bert')

BERT_DIR = os.path.join(BASE_DIR, 'analysis', 'Bert')
BERT_DATA_FOLDER = os.path.join(BERT_DIR, 'data_bert')
DETECTION_DIR = os.path.join(BASE_DIR, 'analysis', 'Brands_Detected')
DETECTION_RESULTS_FOLDER = os.path.join(BERT_DIR, 'detection_results')


# FILES
URL_DATA = os.path.join(DATA_FOLDER, 'url_data.xlsx')
URL_DATA_CLEANED = os.path.join(CLEAN_DATA_FOLDER, 'url_data_cleaned.xlsx')
SEPHORA_DATA_CLEANED = os.path.join(CLEAN_DATA_FOLDER, 'sephora_website_cleaned.csv')
SENTENCES_TRANSCRIPTIONS_FILE = os.path.join(DETECTION_DIR, 'sentences_transcriptions.xlsx')
LABELED_SENTENCES_FILE = os.path.join(BERT_DATA_FOLDER, 'labeled_sentences.xlsx')

BRANDS_REGEX_JSON = os.path.join(DETECTION_DIR, 'all_brands_products_regex.json')
SENTENCES_TRANSCRIPTIONS_DETECTION = os.path.join(DETECTION_DIR, 'sentences_transcriptions.xlsx')
RESULTS_BRAND_DETECTION = os.path.join(DETECTION_DIR, 'results_brand_detection.xlsx')
BRAND_MENTIONS_SUMMARY = os.path.join(DETECTION_DIR, 'brand_mentions_sentiment_summary.xlsx')
TOP_BRANDS_GRAPH = os.path.join(DETECTION_DIR, 'top_brands_mentions_sentiment.png')

def ensure_directories():
    """Crea las carpetas necesarias si no existen."""
    for folder in [VIDEO_FOLDER, AUDIO_FOLDER]:
        os.makedirs(folder, exist_ok=True)

def get_timestamp():
    """Returns current timestamp as a string, e.g. 2025-05-11_18-42-00"""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
