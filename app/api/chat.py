import os
import time
import tempfile
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..models.chat_history import ChatHistory
from ..services.chat_service import chat_service
from ..schemas.chat import AnalyzeRequest, ChatResponse

router = APIRouter(tags=["Chat"])

@router.post("/analyze-usage")
async def analyze_usage(req: AnalyzeRequest):
    """Analyzes usage of a list of words in a given text using restored logic."""
    t0 = time.time()
    result = chat_service.analyze_word_usage_groq(req.text, req.words)
    result["processing_time"] = round(time.time() - t0, 2)
    return result

@router.post("/chat", response_model=ChatResponse)
async def chat_with_friend(
    request: Request,
    user_id: str = Form(...),
    text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    reset: bool = Form(False),
    db: Session = Depends(get_db)
):
    """Processes chat input using restored logic from chat_service.py."""
    t0 = time.time()
    
    # Handle reset
    if reset:
        chat_service.reset_chat_history(user_id, db)
        
        # Nếu chỉ gọi reset mà không gửi thêm text/audio thì trả về luôn
        if not text and not audio:
            return {
                "success": True,
                "user_id": user_id,
                "user_text": "",
                "response_text": "Lịch sử trò chuyện đã được làm mới.",
                "audio_link": None,
                "processing_time": round(time.time() - t0, 2)
            }

    temp_path = None
    try:
        user_text = text
        if not user_text:
            if not audio:
                raise HTTPException(400, "Please provide either 'text' or 'audio'")
            content = await audio.read()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(content)
                temp_path = tmp.name
            user_text = chat_service.transcribe_audio_groq(temp_path)
            if not user_text:
                raise HTTPException(400, "Could not understand audio")

        # LLM Response
        response_text = await chat_service.get_groq_chat_response(user_id, user_text, db)

        # TTS Link
        audio_link = await chat_service.generate_audio_file(response_text, request)

        return {
            "success": True,
            "user_id": user_id,
            "user_text": user_text,
            "response_text": response_text,
            "audio_link": audio_link,
            "processing_time": round(time.time() - t0, 2)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERR] API Chat: {e}")
        raise HTTPException(500, str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
