import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    MURF_API_KEY = os.getenv("MURF_API_KEY")

    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:123456@localhost:1106/chatdb")

    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    STATIC_DIR = os.path.join(BASE_DIR, "static")
    AUDIO_OUTPUT_DIR = os.path.join(STATIC_DIR, "audios")
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "temp_downloads")
    REFERENCE_DIR = os.path.join(BASE_DIR, "audio")

    # Ngrok (Hardcoded as per user request for now or from env)
    NGROK_DOMAIN = "satyr-dashing-officially.ngrok-free.app"

    def __init__(self):
        # Ensure directories exist
        os.makedirs(self.AUDIO_OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(self.REFERENCE_DIR, exist_ok=True)

settings = Settings()
