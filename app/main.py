# 1. THE ABSOLUTE FIRST IMPORT (Crucial for Windows stability)
import app.core.patches 

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pyngrok import ngrok

from contextlib import asynccontextmanager
from .api import scoring, video, chat
from .core.config import settings
from .core.database import init_db
from .services.video_service import video_service
from .services.scoring_service import models as scoring_models

@asynccontextmanager
async def lifespan(app: FastAPI):
    """✅ Modern lifespan handler for startup/shutdown"""
    print("[STARTUP] Initializing Environment...")
    init_db()
    
    # Pre-load video models to avoid lag on first request
    # Scoring models are already loading via module-level import in scoring_service.py
    print("[STARTUP] Warming up video models...")
    video_service._load_models()
    
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

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Files Mounting
os.makedirs(settings.STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

# Include Routers
app.include_router(scoring.router)
app.include_router(video.router)
app.include_router(chat.router)

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "English Learning App API",
        "version": "2.0.0",
        "endpoints": ["/process-video", "/v2/score", "/chat", "/analyze-usage"]
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False) # ✅ Disable reload for production-like performance
