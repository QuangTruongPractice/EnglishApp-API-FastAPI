import os
import time
import requests
import tempfile
import uuid
from typing import Optional, List
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session
from groq import Groq
from dotenv import load_dotenv

from database import SessionLocal, ChatHistory, init_db

load_dotenv()

app = FastAPI()

# --- Static Files Setup ---
STATIC_DIR = "static"
AUDIO_OUTPUT_DIR = os.path.join(STATIC_DIR, "audios")
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Initialize API Clients ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MURF_API_KEY = os.getenv("MURF_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")
if not MURF_API_KEY:
    raise ValueError("MURF_API_KEY not found in .env")

groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize Database
init_db()

# --- Dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Helper Functions ---

def transcribe_audio_groq(file_path: str) -> str:
    """Uses Groq Whisper API to transcribe audio to text."""
    try:
        with open(file_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(os.path.basename(file_path), file.read()),
                model="whisper-large-v3",
                response_format="text",
            )
        return transcription
    except Exception as e:
        print(f"[ERR] Groq Transcription: {e}")
        return ""

def get_groq_chat_response(user_id: str, user_text: str, db: Session) -> str:
    """Generates a 'close friend' response using Groq LLM with context."""
    # 1. Fetch last 10 messages for context
    history = db.query(ChatHistory).filter(ChatHistory.user_id == user_id).order_by(ChatHistory.created_at.desc()).limit(10).all()
    history.reverse()

    messages = [
        {"role": "system", "content": "You are a very close, empathetic, and supportive best friend. Talk casually and warmly. Keep your responses very brief and concise (max 1-2 sentences)."}
    ]
    
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})
    
    messages.append({"role": "user", "content": user_text})

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        response_text = completion.choices[0].message.content
        
        # 2. Save both to database
        user_msg = ChatHistory(user_id=user_id, role="user", content=user_text)
        assistant_msg = ChatHistory(user_id=user_id, role="assistant", content=response_text)
        db.add(user_msg)
        db.add(assistant_msg)
        db.commit()
        
        return response_text
    except Exception as e:
        print(f"[ERR] Groq LLM: {e}")
        return "I'm having a bit of trouble thinking right now, but I'm here for you!"

def generate_murf_audio_file(text: str, request: Request) -> Optional[str]:
    """Generates an audio file using Murf AI Stream API and returns a local link."""
    url = "https://global.api.murf.ai/v1/speech/stream"
    headers = {
        "api-key": MURF_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "voice_id": "Matthew",
        "text": text,
        "locale": "en-US",
        "model": "FALCON",
        "format": "MP3",
        "sampleRate": 24000,
        "channelType": "MONO"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, stream=True)
        if response.status_code == 200:
            filename = f"chat_{uuid.uuid4().hex[:8]}.mp3"
            filepath = os.path.join(AUDIO_OUTPUT_DIR, filename)
            
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            
            # Construct the link
            base_url = str(request.base_url).rstrip("/")
            return f"{base_url}/static/audios/{filename}"
        else:
            print(f"[ERR] Murf AI API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"[ERR] Murf AI: {e}")
        return None



def analyze_word_usage_groq(text: str, words: List[str]) -> dict:
    """Analyzes each keyword in the text and provides granular feedback."""
    keywords_str = ", ".join([f'"{w}"' for w in words])
    prompt = f"""
    You are a professional English Writing Coach. 
    Analyze if the user's text uses each of the target keywords correctly (grammar and semantics).
    User Text: "{text}"
    Target Keywords: [{keywords_str}]
    
    Return your response strictly in JSON format:
    {{
      "overall_score": (int 1-10),
      "word_analysis": [
        {{
          "word": (string),
          "status": (string "correct" or "incorrect"),
          "feedback": (string specific feedback for this word in this context)
        }}
      ],
      "general_feedback": (string expert overall feedback),
      "improved_sentence": (string an improved version of the entire sentence)
    }}
    """
    
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        import json
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"[ERR] Word Analysis: {e}")
        return {
            "overall_score": 0, "word_analysis": [], 
            "general_feedback": "Analysis failed.", "improved_sentence": ""
        }

# --- Routes ---

@app.get("/")
async def root():
    return {"status": "ok"}



class AnalyzeRequest(BaseModel):
    text: str
    words: List[str]

@app.post("/v1/analyze-usage")
async def analyze_usage(req: AnalyzeRequest):
    """Analyzes usage of a list of words in a given text using JSON input."""
    t0 = time.time()
    result = analyze_word_usage_groq(req.text, req.words)
    result["processing_time"] = round(time.time() - t0, 2)
    return result

@app.post("/v1/chat")
async def chat_with_friend(
    request: Request,
    user_id: str = Form(...),
    text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    reset: bool = Form(False),
    db: Session = Depends(get_db)
):
    """Processes either text or audio chat input and returns a friendly response + local audio link."""
    t0 = time.time()
    
    # 0. Handle reset
    if reset:
        db.query(ChatHistory).filter(ChatHistory.user_id == user_id).delete()
        db.commit()

    temp_path = None
    try:
        user_text = text
        
        # 1. Handle audio if text is not provided
        if not user_text:
            if not audio:
                raise HTTPException(400, "Please provide either 'text' or 'audio'")
                
            content = await audio.read()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(content)
                temp_path = tmp.name

            user_text = transcribe_audio_groq(temp_path)
            if not user_text:
                raise HTTPException(400, "Could not understand audio")

        # 2. Text -> Response (Groq LLM with context)
        response_text = get_groq_chat_response(user_id, user_text, db)

        # 3. Response -> Audio Link (Stream & Save Locally)
        audio_link = generate_murf_audio_file(response_text, request)

        return {
            "success": True,
            "user_id": user_id,
            "user_text": user_text,
            "response_text": response_text,
            "audio_link": audio_link,
            "processing_time": round(time.time() - t0, 2)
        }
        
    except Exception as e:
        print(f"[ERR] Chat Service: {e}")
        raise HTTPException(500, str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
