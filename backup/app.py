from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
import os
import gc
import warnings
import tempfile
import uvicorn
from pydantic import BaseModel
from pyngrok import ngrok
import yt_dlp
from datetime import datetime
import logging

# ✅ FIX: Add torchaudio AudioMetaData compatibility patch
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw["weights_only"] = False
    return _orig_load(*a, **kw)
torch.load = _patched_load

if not hasattr(torchaudio, "AudioMetaData"):
    try:
        from torchaudio.backend.common import AudioMetaData
        torchaudio.AudioMetaData = AudioMetaData
    except ImportError:
        from dataclasses import dataclass
        @dataclass
        class _AM:
            sample_rate: int = 0
            num_frames: int = 0
            num_channels: int = 0
            bits_per_sample: int = 0
            encoding: str = ""
        torchaudio.AudioMetaData = _AM

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

# ✅ Now import whisperx AFTER patches
import whisperx

warnings.filterwarnings("ignore")
os.environ['PYTORCH_LIGHTNING_UTILITIES_WARNINGS'] = 'ignore'

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

app = FastAPI(title="Video Processor Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])

UPLOAD_FOLDER = 'temp_downloads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class VideoProcessRequest(BaseModel):
    url: str


class VideoProcessResponse(BaseModel):
    success: bool
    data: dict = None
    error: str = None


def cleanup_temp_file(file_path):
    """✅ Silent cleanup"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.warning(f"Cleanup failed: {file_path}")


def format_video_data(download_result, transcribe_result):
    """✅ Format video data"""
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


# YouTube Processor Class (OPTIMIZED)
class YouTubeProcessor:
    _transcribe_model = None  # ✅ Cache model
    _align_model = None
    _align_metadata = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if torch.cuda.is_available() else "int8"

    def _load_models(self):
        """✅ Lazy load models once"""
        if self._transcribe_model is None:
            print("[MODEL] Loading Whisper model...")
            self._transcribe_model = whisperx.load_model("base", device=self.device, compute_type=self.compute_type)
        if self._align_model is None:
            print("[MODEL] Loading alignment model...")
            self._align_model, self._align_metadata = whisperx.load_align_model(language_code="en", device=self.device)

    def download_audio(self, youtube_url):
        """✅ Download audio from YouTube"""
        strategies = ['ios', 'android', 'web_simple']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        for strategy in strategies:
            try:
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': os.path.join(UPLOAD_FOLDER, f'{timestamp}_%(title)s.%(ext)s'),
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                        'preferredquality': '192',
                    }],
                    'no_warnings': True,
                    'socket_timeout': 30,
                    'retries': 1,
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
                    info = ydl.extract_info(youtube_url, download=False)
                    title = info.get('title', 'Unknown Title')
                    video_id = info.get('id', 'unknown')
                    duration = info.get('duration', 0)
                    language = info.get('language', 'en')

                    ydl.download([youtube_url])

                    # Find downloaded file
                    audio_file = None
                    for file in os.listdir(UPLOAD_FOLDER):
                        if file.startswith(timestamp) and file.endswith('.wav'):
                            audio_file = os.path.join(UPLOAD_FOLDER, file)
                            break

                    if audio_file and os.path.exists(audio_file) and os.path.getsize(audio_file) > 1000:
                        return {
                            'success': True,
                            'audio_file': audio_file,
                            'title': title,
                            'video_id': video_id,
                            'duration': duration,
                            'language': language,
                            'youtube_url': youtube_url
                        }

            except Exception as e:
                continue

        return {'success': False, 'error': 'Unable to download video'}

    def transcribe_audio(self, audio_file):
        """✅ Transcribe audio - reuse models"""
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
            except Exception:
                pass  # ✅ Silent fail

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
                return {'success': False, 'error': 'No transcribable content found'}

            return {'success': True, 'segments': valid_segments, 'language': detected_language}

        except Exception as e:
            return {'success': False, 'error': str(e)}


# Global instances
youtube_processor = None


@app.on_event("startup")
async def startup_event():
    """✅ Minimal startup log"""
    global youtube_processor
    print("[INIT] Video Processor initializing...")
    youtube_processor = YouTubeProcessor()
    print("[INIT] Video Processor ready!")


@app.get("/")
async def root():
    return {"message": "Video Processor Service", "status": "healthy", "version": "1.0.0"}


@app.post("/process-video", response_model=VideoProcessResponse)
async def process_video(request: VideoProcessRequest):
    """✅ Process YouTube video"""
    if not youtube_processor:
        raise HTTPException(status_code=500, detail="Service not initialized")

    try:
        # Download audio
        download_result = youtube_processor.download_audio(request.url)
        if not download_result['success']:
            raise HTTPException(status_code=400, detail=download_result['error'])

        # Transcribe audio
        transcribe_result = youtube_processor.transcribe_audio(download_result['audio_file'])

        # Cleanup
        cleanup_temp_file(download_result['audio_file'])

        if not transcribe_result['success']:
            raise HTTPException(status_code=500, detail=transcribe_result['error'])

        # Format response
        formatted_data = format_video_data(download_result, transcribe_result)

        return VideoProcessResponse(success=True, data=formatted_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cleanup-temp")
async def cleanup_temp_files():
    """✅ Cleanup temp files"""
    try:
        files_removed = 0
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                files_removed += 1
        return {"success": True, "files_removed": files_removed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    public_url = ngrok.connect(8000, domain="satyr-dashing-officially.ngrok-free.app")
    print("Public URL:", public_url)
    uvicorn.run(app, host="0.0.0.0", port=8000)