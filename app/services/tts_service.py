import edge_tts
import io
import os
import hashlib
from ..core.config import settings

class TTSService:
    def __init__ (self):
        # Tạo thư mục tạm nếu chưa có
        self.tts_dir = os.path.join(settings.STATIC_DIR, "tts")
        os.makedirs(self.tts_dir, exist_ok=True)

    async def generate_tts_url(self, text: str, voice: str = "en-US-GuyNeural") -> str:
        hash_input = (text + voice).encode('utf-8')
        hash_name = hashlib.md5(hash_input).hexdigest()
        file_name = f"{hash_name}.mp3"
        file_path = os.path.join(self.tts_dir, file_name)

        if not os.path.exists(file_path):
            comm = edge_tts.Communicate(text, voice)
            await comm.save(file_path)
            
        return f"/static/tts/{file_name}"

tts_service = TTSService()