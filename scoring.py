from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import whisperx
import torch
import tempfile
import os
import uvicorn
import re
import subprocess
import time
import numpy as np
from transformers import (
    Wav2Vec2Processor, 
    Wav2Vec2Model,
    Wav2Vec2ForCTC,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer
)
from torch.nn.functional import cosine_similarity
from difflib import SequenceMatcher
from functools import lru_cache

app = FastAPI()

# CONFIGURATION
SAMPLE_RATE = 16000
REFERENCE_DIR = "example"

# MODEL MANAGER
class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized: return
            
        self.device = "cpu"
        compute_type = "int8" 
        
        print(f"[INIT] Loading models for Laptop CPU...")
        st = time.time()
        
        # 1. WhisperX (Using int8 for speed)
        self.whisper_model = whisperx.load_model("tiny", self.device, compute_type=compute_type)
        self.align_model, self.align_metadata = whisperx.load_align_model(language_code="en", device=self.device)
        
        # 2. Embedding Model (Small & Fast)
        self.emb_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.emb_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.emb_model.eval()
        
        # 3. Phoneme Model (XLSR-53)
        phoneme_id = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        self.ph_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(phoneme_id)
        self.ph_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(phoneme_id)
        self.ph_model = Wav2Vec2ForCTC.from_pretrained(phoneme_id)
        self.ph_model.eval()
        
        # Cache for word reference embeddings
        self.ref_cache = {}
        self._initialized = True
        print(f"[INIT] All models ready in {time.time()-st:.2f}s")
    
    def get_ref_emb(self, word: str):
        word_norm = re.sub(r"[^\w]", "", word.lower())
        if word_norm in self.ref_cache: return self.ref_cache[word_norm]
        
        for ext in [".mp3", ".wav"]:
            path = os.path.join(REFERENCE_DIR, f"{word_norm}{ext}")
            if os.path.exists(path):
                # Faster loading than whisperx for small files
                audio = whisperx.load_audio(path)
                inputs = self.emb_processor(audio, sampling_rate=16000, return_tensors="pt")
                with torch.no_grad():
                    emb = self.emb_model(**inputs).last_hidden_state.mean(dim=1)
                self.ref_cache[word_norm] = emb
                return emb
        return None

models = ModelManager()

@lru_cache(maxsize=100)
def get_phonemes_expected(text: str):
    text = text.lower().replace(".", "").replace(",", "").strip()
    if text == "i admire you": return ["aɪ", "æ", "d", "m", "aɪ", "ɹ", "j", "uː"]
    
    cmd = ["espeak-ng", "-q", "--ipa=3", "-v", "en-us", text]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True, timeout=2)
        ipa = res.stdout.strip().replace("ˈ", "").replace("ˌ", "").replace(" ", "")
        return re.findall(r'([a-zæɑɔəɛɪʊʌ][ːɪʊə]|[tʃdʒðθŋʃʒɹ]|[a-zæɑɔəɛɪʊʌ])', ipa)
    except: return []

@app.get("/")
async def root(): return {"status": "ok", "message": "Optimized Scoring API"}

@app.post("/v2/score")
async def score_v2(expected_text: str = Form(...), audio: UploadFile = File(...)):
    t_start = time.time()
    path = None
    try:
        # 1. IO & Load 
        content = await audio.read()
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.write(fd, content)
        os.close(fd)
        # whisperx.load_audio is optimized for resampling
        audio_arr = whisperx.load_audio(path)
        t_load = time.time()
        
        # 2. Transcription & Alignment
        result = models.whisper_model.transcribe(audio_arr)
        if not result["segments"]: raise HTTPException(400, "No speech detected")
        aligned = whisperx.align(result["segments"], models.align_model, models.align_metadata, audio_arr, models.device)
        word_segments = aligned.get("word_segments", [])
        t_whisper = time.time()
        
        # 3. ANALYSIS
        # 3a. Phonemes (Single pass on phonetic model)
        ph_inputs = models.ph_feature_extractor(audio_arr, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            ph_logits = models.ph_model(ph_inputs.input_values).logits
        
        pred_ids = torch.argmax(ph_logits, dim=-1)[0]
        user_phonemes_raw = models.ph_tokenizer.decode(pred_ids)
        user_phonemes = re.findall(r'([a-zæɑɔəɛɪʊʌ][ːɪʊə]|[tʃdʒðθŋʃʒɹ]|[a-zæɑɔəɛɪʊʌ])', user_phonemes_raw)
        
        # 3b. Word Similarity (Using the small embedding model)
        # We pre-calculate the full audio embedding once to speed up slicing
        emb_inputs = models.emb_processor(audio_arr, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            full_hidden_states = models.emb_model(**emb_inputs).last_hidden_state
        
        # Slicing ratio for base-960h hidden states
        ratio = full_hidden_states.shape[1] / (len(audio_arr) / 16000)
        
        word_results = []
        total_sim, sim_count = 0.0, 0
        
        for ws in word_segments:
            word = ws.get("word")
            if not word or "start" not in ws: continue
            
            ref_emb = models.get_ref_emb(word)
            if ref_emb is not None:
                idx_s, idx_e = int(ws["start"] * ratio), int(ws["end"] * ratio)
                word_emb = full_hidden_states[:, idx_s:idx_e, :].mean(dim=1)
                
                sim = cosine_similarity(word_emb, ref_emb).item()
                word_results.append({
                    "word": word, "start": ws["start"], "end": ws["end"],
                    "similarity_score": round(sim, 3),
                    "feedback": "Excellent" if sim > 0.85 else "Good" if sim > 0.7 else "Needs practice"
                })
                total_sim += sim
                sim_count += 1
            else:
                word_results.append({"word": word, "feedback": "No reference"})
        
        avg_sim = round(total_sim / sim_count, 3) if sim_count > 0 else 0
        t_analysis = time.time()
        
        # 4. Phoneme Matching
        expected_phonemes = get_phonemes_expected(expected_text)
        matcher = SequenceMatcher(None, expected_phonemes, user_phonemes)
        details, correct = [], 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for i in range(i1, i2):
                    details.append({"phoneme": expected_phonemes[i], "status": "correct", "tip": ""})
                    correct += 1
            elif tag in ["delete", "replace"]:
                for i in range(i1, i2):
                    p = expected_phonemes[i]
                    details.append({"phoneme": p, "status": "mispronounced" if tag=="replace" else "missing", "tip": p})
            elif tag == "insert":
                for j in range(j1, j2):
                    details.append({"phoneme": user_phonemes[j], "status": "extra", "tip": "Âm thừa"})
        
        p_score = round(correct / len(expected_phonemes), 3) if expected_phonemes else 0
        
        total_time = time.time() - t_start
        print(f"[OPT] Load: {t_load-t_start:.2f}s | Whisper: {t_whisper-t_load:.2f}s | AI: {t_analysis-t_whisper:.2f}s | Total: {total_time:.2f}s")
        
        return {
            "success": True, 
            "overall_score": round((avg_sim + p_score) / 2, 3),
            "processing_time": round(total_time, 2),
            "step1_audio_similarity": {"average_score": avg_sim, "word_details": word_results},
            "step2_phoneme_analysis": {"accuracy": p_score, "expected_phonemes": expected_phonemes, "user_phonemes": user_phonemes, "details": details}
        }
    except Exception as e:
        print(f"[ERR] {str(e)}")
        raise HTTPException(500, str(e))
    finally:
        if path and os.path.exists(path): os.remove(path)

if __name__ == "__main__":
    uvicorn.run("scoring:app", host="127.0.0.1", port=5050, reload=True)