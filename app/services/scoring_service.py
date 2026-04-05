import os
import re
import time
import tempfile
import subprocess
import asyncio
import unicodedata
import gc
import torch
import numpy as np
import whisperx
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from difflib import SequenceMatcher
from torch.nn.functional import cosine_similarity
from transformers import (
    Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC,
    Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer,
)
from ..core.config import settings
from ..core import patches  # ✅ Ensure patches are applied for torch/torchaudio compatibility

SAMPLE_RATE = 16000
MIN_RECOGNITION_RATIO = 0.50

_EXECUTOR = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
_INFERENCE_SEM = asyncio.Semaphore(5)  # ✅ Parallel concurrency

IPA_RE = re.compile(
    r"(tʃ|dʒ"
    r"|[aeiouæɑɒɔɐɜɝəɛɪʊʌ][ɪʊəː]"
    r"|[ðθŋʃʒɹʔ]"
    r"|[aeiouæɑɒɔɐɜɝəɛɪʊʌ]"
    r"|[b-df-hj-np-tv-z])"
)

class ModelManager:
    """✅ Specialized singleton model manager for Pronunciation Scoring (from scoringv2.py)"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """✅ Standard initialization triggered at module-level (File 1 pattern)"""
        self._init_models()

    def _init_models(self):
        """✅ Actual model loading logic (Restored to CPU + int8 for maximum speed)"""
        if self._initialized:
            return
        
        # ✅ BACK TO BASICS: User confirms CPU + int8 is faster for this specific task
        self.device = "cpu" 
        self.compute_type = "int8"
        t0 = time.time()

        # ✅ Model Loading (Tiny model is lightweight for CPU)
        self.whisper_model = whisperx.load_model("tiny", self.device, compute_type=self.compute_type)
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code="en", device=self.device
        )

        self.emb_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.emb_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.emb_model.eval()

        _ph_name = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        self.ph_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(_ph_name)
        self.ph_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(_ph_name)
        self.ph_model = Wav2Vec2ForCTC.from_pretrained(_ph_name)
        self.ph_model.eval()

        self.ref_cache: dict[str, torch.Tensor] = {}
        self._initialized = True
        print(f"[INIT] Scoring models ready in {time.time() - t0:.1f}s")
        self._precompute_references()

    def _precompute_references(self):
        """✅ Cache reference audio embeddings (stored in audio/ directory)"""
        if not os.path.exists(settings.REFERENCE_DIR):
            return
        
        t0 = time.time()
        loaded_count = 0
        for f in os.listdir(settings.REFERENCE_DIR):
            if not f.endswith((".mp3", ".wav")):
                continue
            key = re.sub(r"[^\w]", "", os.path.splitext(f)[0].lower())
            if key in self.ref_cache:
                continue
            try:
                wav_arr = whisperx.load_audio(os.path.join(settings.REFERENCE_DIR, f))
                wav = torch.from_numpy(wav_arr).unsqueeze(0)
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                
                inp = self.emb_processor(wav.squeeze().numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
                with torch.inference_mode():
                    self.ref_cache[key] = self.emb_model(**inp).last_hidden_state.mean(dim=1)
                loaded_count += 1
            except:
                pass
        
        if loaded_count > 0:
            print(f"[INIT] {loaded_count} reference embeddings cached in {time.time() - t0:.1f}s")

    def get_ref_emb(self, word: str):
        return self.ref_cache.get(re.sub(r"[^\w]", "", word.lower()))

# Global singleton instance — loads models immediately upon module import
models = ModelManager()

@lru_cache(maxsize=1024)
def _espeak_ipa(text: str) -> tuple:
    """✅ Get IPA phonemes using espeak-ng (cached)"""
    text = text.lower().replace(".", "").replace(",", "").strip()
    cmd = _get_espeak_cmd()
    if not cmd:
        return ()
    
    try:
        r = subprocess.run(
            [cmd, "-q", "--ipa", "-v", "en-us", text],
            capture_output=True, timeout=2, check=True,
        )
        ipa = unicodedata.normalize("NFC", r.stdout.decode("utf-8", errors="replace"))
        # Cleanup IPA symbols
        ipa = ipa.strip().replace("ˈ","").replace("ˌ","").replace("_","").replace(".","").replace("\n","").replace(" ","")
        return tuple(IPA_RE.findall(ipa))
    except:
        return ()

@lru_cache(maxsize=1)
def _get_espeak_cmd():
    """✅ Find espeak-ng binary path"""
    cmds = [
        "espeak-ng",
        r"C:\Program Files\eSpeak NG\espeak-ng.exe",
        r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe",
    ]
    for cmd in cmds:
        try:
            subprocess.run([cmd, "-q", "--version"], capture_output=True, timeout=1)
            return cmd
        except:
            continue
    return None

def _extract_embedding(audio: np.ndarray):
    """(Thread) Extract audio embeddings for similarity scoring"""
    inp = models.emb_processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.inference_mode():
        h = models.emb_model(**inp).last_hidden_state
    return h, h.shape[1] / (len(audio) / SAMPLE_RATE)

def _extract_phonemes(audio: np.ndarray):
    """(Thread) Extract predicted phonemes using Wav2Vec2"""
    inp = models.ph_feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.inference_mode():
        logits = models.ph_model(inp.input_values).logits
    return models.ph_tokenizer.decode(torch.argmax(logits, dim=-1)[0])

class ScoringService:
    """✅ Pronunciation scoring service with high concurrency (Fixed with instant module-level load)"""

    async def score_v2_logic(self, expected_text: str, audio_path: str):
        """✅ Unified scoring pipeline (Fixed: Removed lazy load call inside semaphore)"""
        t_start = time.time()
        
        async with _INFERENCE_SEM:
            try:
                audio_arr = whisperx.load_audio(audio_path)

                # 1. Transcription (WhisperX tiny)
                result = models.whisper_model.transcribe(audio_arr)
                if not result["segments"]:
                    return {"success": False, "error": "No speech detected in audio."}

                # 2. Forced Alignment
                aligned = whisperx.align(
                    result["segments"], models.align_model,
                    models.align_metadata, audio_arr, models.device,
                )
                word_segments = aligned.get("word_segments", [])
                
                # 3. Recognition Ratio Check
                recognized = [ws["word"] for ws in word_segments if ws.get("word")]
                ratio = self._match_ratio(recognized, expected_text)
                
                if ratio < MIN_RECOGNITION_RATIO:
                    return {
                        "success": False,
                        "reason": "low_recognition",
                        "recognition_ratio": round(ratio, 3),
                        "message": f"Nhận diện {ratio:.0%} – vui lòng nói rõ hơn.",
                        "processing_time": round(time.time() - t_start, 2),
                    }

                # 4. Multi-threaded processing (Gather results)
                loop = asyncio.get_event_loop()
                (hidden, fps), ph_raw, ph_exp = await asyncio.gather(
                    loop.run_in_executor(_EXECUTOR, _extract_embedding, audio_arr),
                    loop.run_in_executor(_EXECUTOR, _extract_phonemes, audio_arr),
                    loop.run_in_executor(_EXECUTOR, _espeak_ipa, expected_text),
                )

                # 5. Audio Similarity Scoring (Embedding Cosine Sim)
                words_details, total_sim, n_valid = [], 0.0, 0
                for ws in word_segments:
                    w = ws.get("word")
                    if not w or "start" not in ws: continue
                    
                    ref_emb = models.get_ref_emb(w)
                    if ref_emb is not None:
                        s_idx = int(ws["start"] * fps)
                        e_idx = max(s_idx + 1, int(ws["end"] * fps))
                        sim = cosine_similarity(hidden[:, s_idx:e_idx, :].mean(dim=1), ref_emb).item()
                        
                        words_details.append({
                            "word": w, "start": ws["start"], "end": ws["end"],
                            "similarity_score": round(sim, 3),
                            "feedback": "Excellent" if sim > 0.85 else "Good" if sim > 0.70 else "Needs practice",
                        })
                        total_sim += sim; n_valid += 1
                    else:
                        words_details.append({"word": w, "feedback": "No reference available"})
                
                avg_sim = round(total_sim / n_valid, 3) if n_valid else 0.0

                # 6. Phoneme Level Analysis (IPA Comparison)
                user_ph = list(IPA_RE.findall(unicodedata.normalize("NFC", ph_raw)))
                matcher = SequenceMatcher(None, ph_exp, user_ph)
                ph_details, correct_count = [], 0
                
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag == "equal":
                        for i in range(i1, i2):
                            ph_details.append({"phoneme": ph_exp[i], "status": "correct", "tip": ""})
                            correct_count += 1
                    elif tag in ("delete", "replace"):
                        for i in range(i1, i2):
                            ph_details.append({
                                "phoneme": ph_exp[i],
                                "status": "mispronounced" if tag == "replace" else "missing",
                                "tip": ph_exp[i],
                            })
                    elif tag == "insert":
                        for j in range(j1, j2):
                            ph_details.append({"phoneme": user_ph[j], "status": "extra", "tip": "Âm thừa"})

                p_accuracy = round(correct_count / len(ph_exp), 3) if ph_exp else 0.0

                return {
                    "success": True,
                    "score": round((avg_sim + p_accuracy) / 2, 3),
                    "processing_time": round(time.time() - t_start, 2),
                    "step1_audio_similarity": {"average_score": avg_sim, "word_details": words_details},
                    "step2_phoneme_analysis": {
                        "accuracy": p_accuracy,
                        "expected_phonemes": list(ph_exp),
                        "user_phonemes": user_ph,
                        "details": ph_details,
                    },
                }

            except Exception as e:
                return {"success": False, "error": f"Scoring engine error: {str(e)}"}
            finally:
                gc.collect()

    @staticmethod
    def _match_ratio(recognized: list[str], expected: str) -> float:
        """Calculate word-level similarity ratio between recognized and target text."""
        exp_list = expected.lower().split()
        if not exp_list: return 0.0
        return SequenceMatcher(None, exp_list, [w.lower() for w in recognized]).ratio()

# Global singleton instance
scoring_service = ScoringService()
