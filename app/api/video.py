import os
from fastapi import APIRouter, HTTPException
from ..services.video_service import video_service
from ..schemas.video import VideoProcessRequest, VideoProcessResponse

router = APIRouter(tags=["Video"])

@router.post("/process-video", response_model=VideoProcessResponse)
async def process_video(request: VideoProcessRequest):
    """Downloads and transcribes YouTube video audio using restored logic."""
    try:
        # 1. Download
        download_result = video_service.download_audio(request.url)
        if not download_result['success']:
            raise HTTPException(status_code=400, detail=download_result['error'])

        # 2. Transcribe
        transcribe_result = video_service.transcribe_audio(download_result['audio_file'])

        # 3. Cleanup
        if download_result.get('audio_file') and os.path.exists(download_result['audio_file']):
            os.remove(download_result['audio_file'])

        if not transcribe_result['success']:
            raise HTTPException(status_code=500, detail=transcribe_result['error'])

        # 4. Format
        formatted_data = video_service.format_video_data(download_result, transcribe_result)

        return VideoProcessResponse(success=True, data=formatted_data)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERR] API Video: {e}")
        raise HTTPException(status_code=500, detail=str(e))
