from pydantic import BaseModel
from typing import List, Dict, Optional

class VideoProcessRequest(BaseModel):
    url: str

class VideoProcessResponse(BaseModel):
    success: bool
    data: Optional[Dict] = None
    error: Optional[str] = None
