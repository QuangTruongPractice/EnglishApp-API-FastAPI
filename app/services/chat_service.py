import os, time, tempfile, uuid, json
import edge_tts
from typing import Optional, List
from sqlalchemy.orm import Session
from groq import Groq, AsyncGroq
from ..core.config import settings
from ..models.chat_history import ChatHistory

class ChatService:
    def __init__(self):
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        self.async_groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.audio_output_dir = settings.AUDIO_OUTPUT_DIR

    def transcribe_audio_groq(self, file_path: str) -> str:
        try:
            with open(file_path, "rb") as file:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=(os.path.basename(file_path), file.read()),
                    model="whisper-large-v3", response_format="text"
                )
            return transcription
        except Exception as e:
            print(f"[ERR] Groq Transcription: {e}")
            return ""

    def reset_chat_history(self, user_id: str, db: Session) -> bool:
        try:
            db.query(ChatHistory).filter(ChatHistory.user_id == user_id).delete()
            db.commit()
            return True
        except Exception as e:
            print(f"[ERR] Reset Chat History: {e}")
            db.rollback()
            return False

    async def get_groq_chat_response(self, user_id: str, user_text: str, db: Session) -> str:
        history = db.query(ChatHistory).filter(ChatHistory.user_id == user_id).order_by(ChatHistory.created_at.desc()).limit(10).all()
        history.reverse()
        messages = [{"role": "system", "content": "You are a very close, empathetic, and supportive best friend. Talk casually and warmly. Keep your responses very brief and concise (max 1-2 sentences)."}]
        for msg in history: messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": user_text})
        try:
            completion = await self.async_groq_client.chat.completions.create(model="llama-3.1-8b-instant", messages=messages, temperature=0.7, max_tokens=100)
            response_text = completion.choices[0].message.content
            user_msg = ChatHistory(user_id=user_id, role="user", content=user_text)
            assistant_msg = ChatHistory(user_id=user_id, role="assistant", content=response_text)
            db.add(user_msg); db.add(assistant_msg); db.commit()
            return response_text
        except Exception as e:
            print(f"[ERR] Groq LLM: {e}")
            return "I'm here for you!"

    async def generate_audio_file(self, text: str, request) -> Optional[str]:
        """Generates audio using edge-tts (Microsoft Neural Voice, free & fast)."""
        try:
            filename = f"chat_{uuid.uuid4().hex[:8]}.mp3"
            filepath = os.path.join(self.audio_output_dir, filename)
            communicate = edge_tts.Communicate(text, "en-US-GuyNeural")
            await communicate.save(filepath)
            base_url = str(request.base_url).rstrip("/")
            return f"{base_url}/static/audios/{filename}"
        except Exception as e:
            print(f"[ERR] Edge TTS: {e}")
            return None

    def analyze_word_usage_groq(self, text: str, words: List[str]) -> dict:
        keywords_str = ", ".join([f'"{w}"' for w in words])
        prompt = (
            f"Analyze if the user's text uses each of the target keywords correctly. "
            f"User Text: \"{text}\"\n"
            f"Target Keywords: [{keywords_str}]\n"
            "Return strictly in JSON format with the following keys:\n"
            "- \"score\": a number from 0 to 10 (scale of 10)\n"
            "- \"target_keywords\": a list of objects, each with \"keyword\", \"correct_usage\" (bool), and \"reason\"\n"
            "- \"improved_sentence\": a better version of the user's text"
        )
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile", 
                messages=[{"role": "user", "content": prompt}], 
                response_format={"type": "json_object"}, 
                temperature=0.3
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            print(f"[ERR] Word Analysis: {e}")
            return {"score": 0, "target_keywords": [], "improved_sentence": "Analysis failed."}

chat_service = ChatService()
