import os
import time
import re
import gc
import torch
import whisperx
from transformers import (
    Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC,
    Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer,
)
from ..core.config import settings

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._init_models()
        
        # Với các máy không có GPU, kiểu dữ liệu int8 sẽ chạy nhanh hơn
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"

    def _init_models(self):
        """Tải tất cả các model một lần để dùng chung cho toàn bộ app"""
        t0 = time.time()
        
        # Với các máy không có GPU, kiểu dữ liệu int8 sẽ chạy nhanh hơn
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        print(f"[INIT] Loading models on {self.device} with {self.compute_type}...")

        # 1. WhisperX: Dùng để chuyển âm thanh thành chữ
        self.whisper_model = whisperx.load_model(
            settings.WHISPER_MODEL_SIZE, 
            self.device, 
            compute_type=self.compute_type
        )
        
        # 2. Alignment Model (Khớp thời gian)
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code="en", device=self.device
        )

        # 3. Wav2Vec2 Models (Dùng để chấm điểm phát âm)
        self.emb_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.emb_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.emb_model.eval()
        if self.device == "cuda":
            self.emb_model.to(self.device)

        # 3. Wav2Vec2 Phoneme Model (Dùng để nhận diện âm tiết thuần âm thanh)
        _ph_name = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        self.ph_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(_ph_name)
        self.ph_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(_ph_name)
        self.ph_model = Wav2Vec2ForCTC.from_pretrained(_ph_name)
        self.ph_model.eval()
        if self.device == "cuda":
            self.ph_model.to(self.device)

        self.ref_cache: dict[str, torch.Tensor] = {}
        self._initialized = True
        
        print(f"[INIT] All models ready in {time.time() - t0:.1f}s")
        self._precompute_references()

    def _precompute_references(self):
        """Lưu trữ sẵn các đặc trưng âm thanh mẫu để so sánh"""
        if not os.path.exists(settings.REFERENCE_DIR):
            return
        
        t0 = time.time()
        cache_data = {}
        if os.path.exists(settings.REFERENCE_CACHE_PATH):
            try:
                cache_data = torch.load(settings.REFERENCE_CACHE_PATH, map_location="cpu")
                for k, v in cache_data.items():
                    self.ref_cache[k] = v["emb"]
                print(f"[INIT] Loaded {len(cache_data)} embeddings from cache")
            except Exception as e:
                print(f"[INIT] Error loading cache: {e}")

        loaded_count = 0
        skipped_count = 0
        new_items_added = False
        
        files = [f for f in os.listdir(settings.REFERENCE_DIR) if f.endswith((".mp3", ".wav"))]
        
        for f in files:
            file_path = os.path.join(settings.REFERENCE_DIR, f)
            key = re.sub(r"[^\w]", "", os.path.splitext(f)[0].lower())
            mtime = os.path.getmtime(file_path)

            if key in cache_data and cache_data[key].get("mtime") == mtime:
                skipped_count += 1
                continue

            try:
                wav_arr = whisperx.load_audio(file_path)
                wav = torch.from_numpy(wav_arr).unsqueeze(0)
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                
                # Chuyển dữ liệu lên GPU nếu có
                inp = self.emb_processor(wav.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
                if self.device == "cuda":
                    inp = {k: v.to(self.device) for k, v in inp.items()}

                with torch.inference_mode():
                    emb = self.emb_model(**inp).last_hidden_state.mean(dim=1)
                
                # Lưu vào RAM để dùng lâu dài
                emb_cpu = emb.cpu()
                self.ref_cache[key] = emb_cpu
                cache_data[key] = {"emb": emb_cpu, "mtime": mtime}
                loaded_count += 1
                new_items_added = True

            except Exception as e:
                print(f"[INIT] Failed to process {f}: {e}")
        
        if new_items_added:
            try:
                torch.save(cache_data, settings.REFERENCE_CACHE_PATH)
            except Exception as e:
                print(f"[INIT] Failed to save cache: {e}")

        print(f"[INIT] Reference scan done. New: {loaded_count}, Cached: {skipped_count}. Time: {time.time() - t0:.1f}s")

    def get_ref_emb(self, word: str):
        return self.ref_cache.get(re.sub(r"[^\w]", "", word.lower()))

# Tạo một instance duy nhất để dùng khắp nơi
model_manager = ModelManager()
