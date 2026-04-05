import os
import tempfile
import time
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from ..services.scoring_service import scoring_service
from ..schemas.scoring import ScoringResponse

router = APIRouter(tags=["Scoring"])

@router.post("/v2/score", response_model=ScoringResponse)
async def score_audio_v2(
    expected_text: str = Form(...),
    audio: UploadFile = File(...),
):
    """Pronunciation Scoring V2 pipeline (Similarity + Phonemes)"""
    path = None
    try:
        content = await audio.read()
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.write(fd, content)
        os.close(fd)
        
        result = await scoring_service.score_v2_logic(expected_text, path)
        
        # Consistent error handling: 
        # If success is False but has a reason (like low_recognition), 
        # return the JSON with reason instead of 400 ERROR.
        if not result.get("success", False) and "reason" not in result:
             raise HTTPException(400, result.get("error", "Scoring failed"))
            
        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERR] API Score V2: {e}")
        raise HTTPException(500, str(e))
    finally:
        if path and os.path.exists(path):
            os.remove(path)
