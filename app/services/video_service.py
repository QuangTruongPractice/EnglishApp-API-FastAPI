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

logger = logging.getLogger(__name__)

class VideoService:
    """✅ Optimized video processing service with model caching, based on app.py root."""
    
    _transcribe_model = None  # ✅ Cache models as class attributes
    _align_model = None
    _align_metadata = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if torch.cuda.is_available() else "int8"
        self.upload_folder = settings.UPLOAD_FOLDER
        os.makedirs(self.upload_folder, exist_ok=True)

    def _load_models(self):
        """✅ Lazy load models once"""
        if self._transcribe_model is None:
            print("[MODEL] Loading Whisper model (base)...")
            self._transcribe_model = whisperx.load_model(
                "base", device=self.device, compute_type=self.compute_type
            )
        if self._align_model is None:
            print("[MODEL] Loading alignment model (en)...")
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code="en", device=self.device
            )

    def download_audio(self, youtube_url: str) -> Dict:
        """✅ Download audio from YouTube with multiple strategies (ios, android, web)"""
        strategies = ['ios', 'android', 'web_simple']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        for strategy in strategies:
            try:
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': os.path.join(self.upload_folder, f'{timestamp}_%(title)s.%(ext)s'),
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                        'preferredquality': '192',
                    }],
                    'no_warnings': True,
                    'socket_timeout': 30,
                    'retries': 1,
                    'nocheckcertificate': True,
                }

                # Strategy-specific configs
                if strategy == 'ios':
                    ydl_opts.update({
                        'user_agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15',
                        'extractor_args': {'youtube': {'player_client': ['ios']}}
                    })
                elif strategy == 'android':
                    ydl_opts.update({
                        'user_agent': 'Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36',
                        'extractor_args': {'youtube': {'player_client': ['android']}}
                    })

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # Strategy 'download=True' is used here to ensure fetch
                    info = ydl.extract_info(youtube_url, download=True)
                    audio_file = None
                    
                    # Find downloaded file
                    for file in os.listdir(self.upload_folder):
                        if file.startswith(timestamp) and file.endswith('.wav'):
                            audio_file = os.path.join(self.upload_folder, file)
                            break

                    if audio_file and os.path.exists(audio_file) and os.path.getsize(audio_file) > 1000:
                        return {
                            'success': True,
                            'audio_file': audio_file,
                            'title': info.get('title', 'Unknown Title'),
                            'video_id': info.get('id', 'unknown'),
                            'duration': info.get('duration', 0),
                            'language': info.get('language', 'en'),
                            'youtube_url': youtube_url
                        }

            except Exception as e:
                logger.warning(f"Download strategy {strategy} failed for {youtube_url}: {e}")
                continue  # ✅ Trial next strategy

        return {'success': False, 'error': 'Unable to download video after trying all strategies.'}

    def transcribe_audio(self, audio_file: str) -> Dict:
        """✅ Transcribe audio - reuse cached models"""
        try:
            self._load_models()  # ✅ Load once, reuse

            audio = whisperx.load_audio(audio_file)
            result = self._transcribe_model.transcribe(audio, batch_size=16)
            detected_language = result.get("language", "en")

            segments = result.get("segments", [])

            # Align segments
            try:
                result_aligned = whisperx.align(
                    segments, self._align_model, self._align_metadata,
                    audio, self.device, return_char_alignments=False
                )
                segments = result_aligned.get("segments", segments)
            except Exception as e:
                logger.warning(f"Alignment failed: {e}")
                pass  # ✅ Return unaligned segments if alignment fails

            del audio
            gc.collect()

            # Format segments
            valid_segments = [
                {
                    'start': seg.get('start', 0),
                    'end': seg.get('end', 0),
                    'text': seg.get('text', '').strip(),
                    'confidence': seg.get('confidence', 0.0),
                    'segment_id': i
                }
                for i, seg in enumerate(segments) if seg.get('text', '').strip()
            ]

            if not valid_segments:
                return {'success': False, 'error': 'No transcribable content found in audio.'}

            return {'success': True, 'segments': valid_segments, 'language': detected_language}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def format_video_data(self, download_result: Dict, transcribe_result: Dict) -> Dict:
        """✅ Format video and subtitle data for API response"""
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

    def cleanup_temp_file(self, file_path: str):
        """✅ Silent cleanup of temporary files"""
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Cleanup failed for {file_path}: {e}")

# Global singleton instance
video_service = VideoService()
