import os
import requests
import uuid
import time
from typing import Optional
from fastapi import Request
from ..core.config import settings

class MurfService:
    def __init__(self):
        self.api_key = settings.MURF_API_KEY
        self.output_dir = settings.AUDIO_OUTPUT_DIR

    def generate_audio(self, text: str, request: Request) -> Optional[str]:
        """Generates an audio file using Murf AI Stream API."""
        url = "https://global.api.murf.ai/v1/speech/stream"
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "voice_id": "Matthew",
            "text": text,
            "locale": "en-US",
            "model": "FALCON",
            "format": "MP3",
            "sampleRate": 24000,
            "channelType": "MONO"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, stream=True)
            if response.status_code == 200:
                filename = f"chat_{uuid.uuid4().hex[:8]}.mp3"
                filepath = os.path.join(self.output_dir, filename)
                
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                
                # Construct the link
                base_url = str(request.base_url).rstrip("/")
                return f"{base_url}/static/audios/{filename}"
            else:
                print(f"[ERR] Murf AI API: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"[ERR] Murf AI: {e}")
            return None

murf_service = MurfService()
