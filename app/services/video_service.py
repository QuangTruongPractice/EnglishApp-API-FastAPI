import os
import time
import gc
import logging
import torch
import whisperx
import yt_dlp
from datetime import datetime
from typing import Dict, List, Optional
from ..core.config import settings

from .model_manager import model_manager

logger = logging.getLogger(__name__)

class VideoService:
    def __init__(self):
        # Tạo thư mục tạm nếu chưa có
        self.upload_folder = settings.UPLOAD_FOLDER
        os.makedirs(self.upload_folder, exist_ok=True)

    def download_audio(self, url: str) -> Dict:
        # Tạo dấu thời gian để tránh trùng lặp file
        timestamp = int(time.time())
        opts = {
            'format': 'bestaudio/best',
            # Đặt tên file đơn giản để dễ quản lý
            'outtmpl': os.path.join(self.upload_folder, f'yt_{timestamp}_%(id)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
            # Dùng IPv4 để tránh bị YouTube chặn dải IPv6
            'source_address': '0.0.0.0',
            # Giả lập các client khác nhau để vượt qua bot detection của YouTube
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'ios', 'web'],
                    'skip': ['hls', 'dash']
                }
            },
            'nocheckcertificate': True,
            'ignoreerrors': False,
            'logtostderr': False,
            'no_entry_info': True,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                # Lấy đường dẫn file thực tế sau khi đã convert sang mp3
                temp_path = ydl.prepare_filename(info)
                audio_file = os.path.splitext(temp_path)[0] + ".mp3"
                
                return {
                    'success': True, 
                    'audio_file': audio_file, 
                    'title': info.get('title', 'Unknown'), 
                    'video_id': info.get('id'), 
                    'duration': info.get('duration'), 
                    'language': info.get('language', 'en'), 
                    'youtube_url': url
                }
        except Exception as e:
            logger.error(f"Download error: {e}")
            return {'success': False, 'error': str(e)}

    def transcribe_audio(self, audio_file: str) -> Dict:
        try:
            audio = whisperx.load_audio(audio_file)
            # Dùng model whisper để chuyển âm thanh thành chữ
            result = model_manager.whisper_model.transcribe(audio, batch_size=16)
            
            # Căn chỉnh lại thời gian cho khớp từng từ
            try:
                aligned = whisperx.align(
                    result["segments"], 
                    model_manager.align_model, 
                    model_manager.align_metadata,
                    audio,
                    model_manager.device
                )
                segments = aligned["segments"]
            except Exception as e:
                logger.warning(f"Alignment failed: {e}")
                segments = result["segments"]

            # Dọn dẹp bộ nhớ
            del audio
            gc.collect()

            # Lọc và định dạng dữ liệu trả về (dùng segment gốc của WhisperX)
            valid = []
            for i, s in enumerate(segments):
                txt = s.get('text', '').strip()
                if txt:
                    valid.append({
                        'start': s['start'], 
                        'end': s['end'], 
                        'text': txt, 
                        'segment_id': i
                    })
            
            return {'success': True, 'segments': valid, 'language': result.get("language", "en")} if valid else {'success': False, 'error': 'No content'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def format_video_data(self, download_result: Dict, transcribe_result: Dict) -> Dict:
        # Định dạng dữ liệu trả về cho client
        video_data = {
            "videoId": download_result['video_id'],
            "title": download_result['title'],
            "youtubeUrl": download_result['youtube_url'],
            "duration": download_result.get('duration'),
            "language": transcribe_result['language'],
            "status": "PROCESSED",
            "segmentsCount": len(transcribe_result['segments'])
        }

        subtitles_data = []
        for segment in transcribe_result['segments']:
            subtitle = {
                "segmentId": segment['segment_id'],
                "startTime": float(segment['start']),
                "endTime": float(segment['end']),
                "originalText": segment['text'],
                "confidence": float(segment.get('confidence', 0.0))
            }
            subtitles_data.append(subtitle)

        return {"video": video_data, "subtitles": subtitles_data}

    def cleanup_temp_files(self, file_path: str):
        # Xóa file tạm sau khi đã xử lý xong để giải phóng bộ nhớ
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
                
video_service = VideoService()
