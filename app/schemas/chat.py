from pydantic import BaseModel
from typing import List, Optional

class AnalyzeRequest(BaseModel):
    text: str
    words: List[str]

class ChatResponse(BaseModel):
    success: bool
    user_id: str
    user_text: str
    response_text: str
    audio_link: Optional[str]
    processing_time: float
