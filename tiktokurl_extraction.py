# DEPENDENCIES:    
import os
import re
import time
import logging
import subprocess
import pandas as pd
import whisper
import yt_dlp
from typing import Optional

from dataclasses import dataclass
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from utils import (
    VIDEO_FOLDER,
    AUDIO_FOLDER,
    URL_DATA,
    ensure_directories
)

# Create folders if they don't exist:
ensure_directories()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class Paths:
    video_folder: str
    audio_folder: str
    url_data: str

@dataclass
class TikTokVideoData:
    username: str
    followers: int
    video_likes: int
    video_views: int  
    url: str
    description: str
    publish_date: str
    error: Optional[str] = None

class TikTokVideoProcessor:
    def __init__(self, paths: Paths):
        self.paths = paths
        self._initialize_folders()
        self._setup_browser()
        self._whisper_model = None
    # Create folders if they don't exist:
    def _initialize_folders(self) -> None:
        for folder in [self.paths.video_folder, self.paths.audio_folder]:
            os.makedirs(folder, exist_ok=True)

    # Chrome config to avoid detection and run headless:
    def _setup_browser(self):
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option('excludeSwitches', ['enable-automation'])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        self.driver = webdriver.Chrome(options=options)
        # hide webdriver detection
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    def _load_whisper_model(self) -> None:
        # Initialize Whisper model only if needed
        if self._whisper_model is None:
            logging.info("Loading Whisper model...") 
            self._whisper_model = whisper.load_model("medium")

    def _download_and_convert_audio(self, url: str) -> str:
        try:
            video_id = url.split('/')[-1]
            audio_path = os.path.join(self.paths.audio_folder, f'audio_{video_id}.mp3')
            # Skip if already processed 
            if os.path.exists(audio_path):
                logging.info(f"Audio file already exists for {video_id}")
                return audio_path

            # Download video with yt-dlp
            video_path = os.path.join(self.paths.video_folder, f'{video_id}.mp4')
            ydl_opts = {
                'outtmpl': video_path,
                'format': 'bestvideo+bestaudio/best',
                'quiet': True
            }

            # Extract audio with ffmpeg
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            subprocess.run([
                'ffmpeg', '-i', video_path,
                '-vn', '-ar', '44100', '-ac', '2', '-ab', '192k',
                '-f', 'mp3', audio_path
            ], capture_output=True)

            # Clean up video file
            os.remove(video_path)
            return audio_path
        except Exception as e:
            logging.error(f"Error in download/convert for {url}: {e}")
            raise

    def _transcribe_audio(self, audio_path: str) -> str:
        try:
            self._load_whisper_model()
            result = self._whisper_model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            logging.error(f"Error transcribing audio {audio_path}: {e}")
            raise

    @staticmethod
    def _extract_username_from_url(url: str) -> str:
        # Handle various TikTok URL formats
        patterns = [
            r'@([\w.]+)',
            r'tiktok\.com/(@[\w.]+)',
            r'video/(@[\w.]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1).replace('@', '')
        
        raise ValueError(f"Could not extract username from: {url}")

    @staticmethod
    def _convert_count_to_number(count_str: str) -> int:
        # Convert TikTok's K/M/B format to numbers (thousands/millions/billions)
        if not count_str:
            return 0
            
        multipliers = {'K': 1000, 'M': 1000000, 'B': 1000000000}
        count_str = count_str.strip().upper()
        
        try:
            if count_str[-1] in multipliers:
                number = float(count_str[:-1])
                return int(number * multipliers[count_str[-1]])
            return int(count_str)
        except (ValueError, IndexError):
            return 0
    # Scrape profile page for follower count
    def get_profile_info(self, username: str) -> int:
        profile_url = f"https://www.tiktok.com/@{username}"
        self.driver.get(profile_url)
        time.sleep(2)
        
        try:
            followers_element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '[data-e2e="followers-count"]'))
            )
            followers_text = followers_element.text
            return self._convert_count_to_number(followers_text)
        except Exception as e:
            logging.error(f"Error getting followers for {username}: {e}")
            return 0

    # Extract video metadata without downloading
    def get_video_info(self, url: str) -> tuple:
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'force_json': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                timestamp = info.get('timestamp')
                publish_date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d') if timestamp else ''
                
                return (
                    info.get('like_count', 0),
                    info.get('view_count', 0), 
                    info.get('description', ''),
                    publish_date
                )
        except Exception as e:
            logging.error(f"Error getting video information: {e}")
            return 0, 0, '', ''

    def process_video_data(self, url: str) -> TikTokVideoData:
        # Main processing function for a single video
        try:
            username = self._extract_username_from_url(url)
            followers = self.get_profile_info(username)
            likes, views, description, publish_date = self.get_video_info(url) 
            
            # Download and transcribe audio
            audio_path = self._download_and_convert_audio(url)
            transcription = self._transcribe_audio(audio_path)

            # Clean up audio file after transcription
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return TikTokVideoData(
                username=username,
                followers=followers,
                video_likes=likes,
                video_views=views,  
                description=description,
                publish_date=publish_date,
                url=url
            ), transcription
            
        except Exception as e:
            logging.error(f"Error processing {url}: {e}")
            return TikTokVideoData(
                username="",
                followers=0,
                video_likes=0,
                video_views=0,  
                description="",
                publish_date="",
                url=url,
                error=str(e)
            ), ""

    def process_excel_data(self):
        try:
            df = pd.read_excel(self.paths.url_data, engine='openpyxl')
            # Setup new columns if they don't exist
            new_columns = {
                'tiktok_username': '',
                'followers_count': 0,
                'video_likes': 0,
                'video_views': 0,
                'video_description': '',
                'publish_date': '',
                'transcription': '',
                'extraction_error': ''
            }
            
            for col, default_value in new_columns.items():
                if col not in df.columns:
                    df[col] = default_value
            
            total_rows = len(df)
    
            for index, row in df.iterrows():
                url = row['url_video']
                
                # Check what data we need to fetch:
                needs_username = pd.isna(row['tiktok_username']) or row['tiktok_username'] == ''
                needs_followers = pd.isna(row['followers_count']) or row['followers_count'] == 0
                needs_video_info = (pd.isna(row['video_likes']) or row['video_likes'] == 0 or
                                pd.isna(row['video_views']) or row['video_views'] == 0 or
                                pd.isna(row['video_description']) or row['video_description'] == '' or
                                pd.isna(row['publish_date']) or row['publish_date'] == '')
                needs_transcription = pd.isna(row['transcription']) or row['transcription'] == ''
                
                if any([needs_username, needs_followers, needs_video_info, needs_transcription]):
                    logging.info(f"Processing {index + 1}/{total_rows}: {url}")
                    
                    try:
                        # Extract or use existing username
                        if needs_username:
                            username = self._extract_username_from_url(url)
                            df.at[index, 'tiktok_username'] = username
                        else:
                            username = row['tiktok_username']
                        # Get profile info
                        if needs_followers:
                            followers = self.get_profile_info(username)
                            df.at[index, 'followers_count'] = followers
                        # Get video metadata
                        if needs_video_info:
                            likes, views, description, publish_date = self.get_video_info(url)
                            df.at[index, 'video_likes'] = likes
                            df.at[index, 'video_views'] = views
                            df.at[index, 'video_description'] = description
                            df.at[index, 'publish_date'] = publish_date

                        # Transcribe audio
                        if needs_transcription:
                            audio_path = self._download_and_convert_audio(url)
                            transcription = self._transcribe_audio(audio_path)
                            df.at[index, 'transcription'] = transcription
                            
                            if os.path.exists(audio_path):
                                os.remove(audio_path)

                        # Save progress periodically
                        df.at[index, 'extraction_error'] = ''
                        if (index + 1) % 5 == 0:
                            df.to_excel(self.paths.url_data, index=False, engine='openpyxl')
                            logging.info(f"Progress saved: {index + 1} videos processed")
                        
                        time.sleep(2) # Avoid overwhelming the server
                        
                    except Exception as e:
                        logging.error(f"Error in URL {url}: {e}")
                        df.at[index, 'extraction_error'] = str(e)
                else:
                    logging.info(f"Skipping fully processed URL {index + 1}/{total_rows}: {url}")

            # Final save
            df.to_excel(self.paths.url_data, index=False, engine='openpyxl')
            logging.info("Process completed. Excel updated.")
            
        except Exception as e:
            logging.error(f"Error processing Excel file: {e}")
            raise

    def close(self):
        if hasattr(self, 'driver'):
            self.driver.quit()


def main():
    paths = Paths(
        video_folder= VIDEO_FOLDER,
        audio_folder= AUDIO_FOLDER,
        url_data= URL_DATA,
    )

    processor = TikTokVideoProcessor(paths)
    try:
        processor.process_excel_data()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        processor.close()

if __name__ == "__main__":
    main()
