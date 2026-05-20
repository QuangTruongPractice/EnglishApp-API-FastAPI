import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Cấu hình Model
    WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

    # Các mã API Key
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Cơ sở dữ liệu
    DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://4LGbsrBLSmPw4K8.root:yTj7fhI0tzERXGuC@gateway01.ap-southeast-1.prod.aws.tidbcloud.com:4000/chatdb?ssl_verify_cert=true&ssl_verify_identity=true")

    # Đường dẫn thư mục
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    STATIC_DIR = os.path.join(BASE_DIR, "static")
    AUDIO_OUTPUT_DIR = os.path.join(STATIC_DIR, "audios")
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "temp_downloads")
    REFERENCE_DIR = os.path.join(BASE_DIR, "audio")
    REFERENCE_CACHE_PATH = os.path.join(BASE_DIR, "reference_cache.pt")

    # Tên miền Ngrok
    NGROK_DOMAIN = "satyr-dashing-officially.ngrok-free.app"

    def __init__(self):
        # Tạo các thư mục nếu chưa có
        os.makedirs(self.AUDIO_OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(self.REFERENCE_DIR, exist_ok=True)

settings = Settings()
