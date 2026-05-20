# 1. Nạp patch đầu tiên để chạy ổn định trên Windows
import app.core.patches 

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pyngrok import ngrok

from contextlib import asynccontextmanager
from .api import scoring, video, chat, tts
from .core.config import settings
from .core.database import init_db
from .services.model_manager import model_manager
from .services.video_service import video_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Initializing Environment...")
    init_db()
    
    # Khởi động trước các model để tránh bị chậm lúc đầu
    print(f"[STARTUP] Models warming up on {model_manager.device}...")
    
    try:
        public_url = ngrok.connect(8000, domain=settings.NGROK_DOMAIN)
        print(f"[*] Ngrok Tunnel Active: {public_url}")
    except Exception as e:
        print(f"[!] Ngrok failed: {e}")
        
    yield
    print("[SHUTDOWN] Cleaning up...")

app = FastAPI(
    title="English Learning App - Unified API",
    description="Optimized FastAPI service for Video Transcription and Pronunciation Scoring",
    version="2.0.0",
    lifespan=lifespan
)

# Cấu hình CORS để cho phép truy cập từ Web/Mobile
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình thư mục chứa file tĩnh (audio, tts...)
os.makedirs(settings.STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

# Nạp các đường dẫn API
app.include_router(scoring.router, prefix="/ai")
app.include_router(video.router, prefix="/ai")
app.include_router(chat.router, prefix="/ai")
app.include_router(tts.router, prefix="/ai")

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "English Learning App API",
        "version": "2.0.0",
        "endpoints": ["/process-video", "/v2/score", "/chat", "/analyze-usage"]
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
