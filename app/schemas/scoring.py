from pydantic import BaseModel
from typing import List, Dict, Optional

class ScoringRequest(BaseModel):
    expected_text: str

class WordDetail(BaseModel):
    word: str
    start: float
    end: float
    similarity_score: float
    feedback: str

class PhonemeDetail(BaseModel):
    phoneme: str
    status: str
    tip: str

class ScoringResponse(BaseModel):
    success: bool
    score: Optional[float] = None
    processing_time: float
    step1_audio_similarity: Optional[Dict] = None
    step2_phoneme_analysis: Optional[Dict] = None
    reason: Optional[str] = None
    recognition_ratio: Optional[float] = None
    message: Optional[str] = None
    error: Optional[str] = None
