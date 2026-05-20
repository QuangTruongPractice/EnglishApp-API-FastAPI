from pydantic import BaseModel
from typing import Optional

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "en-US-GuyNeural"

class TTSResponse(BaseModel):
    audio_url: str
