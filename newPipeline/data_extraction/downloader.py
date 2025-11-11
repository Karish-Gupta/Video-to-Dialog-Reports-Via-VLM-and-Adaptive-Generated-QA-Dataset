import yt_dlp
import os

def download_audio(youtube_url, output_path="audio"):
    """Download audio from YouTube video"""
    print(f"Downloading audio from: {youtube_url}")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        'quiet': False,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        audio_file = ydl.prepare_filename(info)
        return audio_file, info

def download_video(youtube_url, output_path="outputs2"):
    """Download video from YouTube using yt-dlp and save to outputs2 by default."""
    print(f"Downloading video from: {youtube_url}")
    os.makedirs(output_path, exist_ok=True)

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        'quiet': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_file = ydl.prepare_filename(info)

    print(f"Video downloaded: {video_file}")
    return video_file, info