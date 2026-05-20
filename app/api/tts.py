from fastapi import APIRouter, HTTPException, Request
from ..services.tts_service import tts_service
from ..schemas.tts import TTSRequest, TTSResponse

router = APIRouter(tags=["TTS"])

@router.post("/tts", response_model=TTSResponse)
async def text_to_speech(payload: TTSRequest, request: Request):
    """
    Chuyển đổi văn bản thành giọng nói bằng Edge TTS và trả về link file MP3.
    Các file âm thanh được lưu tạm trên server để tăng tốc độ xử lý.
    """
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    try:
        audio_path = await tts_service.generate_tts_url(payload.text, payload.voice)
        
        # Build absolute URL using the request domain
        absolute_url = str(request.base_url).rstrip('/') + audio_path
        
        return TTSResponse(audio_url=absolute_url)
    except Exception as e:
        print(f"[ERR] TTS API: {e}")
        raise HTTPException(status_code=500, detail=f"TTS Engine Error: {str(e)}")

