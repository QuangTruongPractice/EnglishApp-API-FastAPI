import os
from typing import List
from groq import Groq
from ..core.config import settings

class GroqService:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    def transcribe_audio(self, file_path: str) -> str:
        """Uses Groq Whisper API to transcribe audio to text."""
        try:
            with open(file_path, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(os.path.basename(file_path), file.read()),
                    model="whisper-large-v3",
                    response_format="text",
                )
            return transcription
        except Exception as e:
            print(f"[ERR] Groq Transcription: {e}")
            return ""

    def get_chat_response(self, messages: List[dict]) -> str:
        """Generates a response using Groq LLM."""
        try:
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.7,
                max_tokens=150,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"[ERR] Groq LLM: {e}")
            return "I'm having a bit of trouble thinking right now, but I'm here for you!"

    def analyze_word_usage(self, text: str, words: List[str]) -> dict:
        """Analyzes word usage in text using Groq LLM."""
        keywords_str = ", ".join([f'"{w}"' for w in words])
        prompt = f"""
        You are a professional English Writing Coach. 
        Analyze if the user's text uses each of the target keywords correctly (grammar and semantics).
        User Text: "{text}"
        Target Keywords: [{keywords_str}]
        
        Return your response strictly in JSON format:
        {{
          "score": (int 1-10),
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
            completion = self.client.chat.completions.create(
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
                "score": 0, "word_analysis": [], 
                "general_feedback": "Analysis failed.", "improved_sentence": ""
            }

groq_service = GroqService()
