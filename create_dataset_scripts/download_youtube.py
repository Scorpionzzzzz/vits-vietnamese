#!/usr/bin/env python3
"""
YouTube Downloader
Downloads YouTube videos and saves audio to a folder
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List
import yt_dlp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YouTubeDownloader:
    def __init__(self, output_dir: str = "../data/downloaded_audio"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
    
    def download_video(self, url: str) -> str:
        """Download audio from YouTube video"""
        logger.info(f"Downloading: {url}")
        
        # Generate filename from URL
        video_id = url.split('v=')[-1].split('&')[0]
        filename = f"{video_id}.wav"
        output_path = self.output_dir / filename
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(output_path).replace('.wav', ''),
            'noplaylist': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'postprocessor_args': [
                '-ar', '22050',  # resample to 22050Hz
                '-ac', '1',      # mono
            ],
            'overwrites': True,
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            if output_path.exists():
                logger.info(f"✅ Downloaded: {filename}")
                return str(output_path)
            else:
                raise FileNotFoundError(f"Download failed: {filename}")
                
        except Exception as e:
            logger.error(f"❌ Error downloading {url}: {e}")
            raise
    
    def download_multiple(self, urls: List[str]):
        """Download multiple videos"""
        logger.info(f"Downloading {len(urls)} videos...")
        
        downloaded_files = []
        for i, url in enumerate(urls, 1):
            try:
                logger.info(f"Progress: {i}/{len(urls)}")
                file_path = self.download_video(url)
                downloaded_files.append(file_path)
            except Exception as e:
                logger.error(f"Failed to download {url}: {e}")
                continue
        
        logger.info(f"✅ Download complete! {len(downloaded_files)} files downloaded")
        return downloaded_files

def main():
    parser = argparse.ArgumentParser(description="YouTube Audio Downloader")
    parser.add_argument("--urls", nargs="+", help="YouTube URLs to download")
    parser.add_argument("--urls_file", help="File containing YouTube URLs (one per line)")
    parser.add_argument("--output_dir", default="../data/downloaded_audio", help="Output directory")
    
    args = parser.parse_args()
    
    # Get URLs
    urls = []
    if args.urls:
        urls.extend(args.urls)
    elif args.urls_file:
        with open(args.urls_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
    else:
        # Interactive mode
        print("Enter YouTube URLs (one per line, press Enter twice to finish):")
        while True:
            url = input().strip()
            if not url:
                break
            urls.append(url)
    
    if not urls:
        print("No URLs provided!")
        return
    
    # Download videos
    downloader = YouTubeDownloader(args.output_dir)
    downloader.download_multiple(urls)

if __name__ == "__main__":
    main() 