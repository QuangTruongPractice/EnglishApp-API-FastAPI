import os
import re
import time
import tempfile
import subprocess
import asyncio
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import torch
import numpy as np
import torchaudio
import whisperx
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pyngrok import ngrok
from torch.nn.functional import cosine_similarity
from difflib import SequenceMatcher
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Model,
    Wav2Vec2ForCTC,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
)
import imageio_ffmpeg

app = FastAPI()

# ── CONFIG ────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
REFERENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")

# Ensure ffmpeg is in PATH for whisperx
_FFMPEG_DIR = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
if _FFMPEG_DIR not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + _FFMPEG_DIR

# Shared thread-pool for CPU-bound model inference (2 workers = emb + phoneme)
_EXECUTOR = ThreadPoolExecutor(max_workers=2)

# ── IPA PHONEME REGEX ─────────────────────────────────────────────────────────
# Thứ tự quan trọng: affricates trước, rồi diphthong (CHỈ nguyên âm đứng đầu),
# rồi phụ âm IPA đặc biệt, rồi nguyên âm đơn, cuối cùng phụ âm la-tinh.
IPA_RE = re.compile(
    r'(tʃ|dʒ'                               # affricates (ch, j)
    r'|[aeiouæɑɒɔɐɜɝəɛɪʊʌ][ɪʊəː]'         # diphthong / long vowel — CHỈ vowel đứng đầu
    r'|[ðθŋʃʒɹʔ]'                           # phụ âm IPA đặc biệt
    r'|[aeiouæɑɒɔɐɜɝəɛɪʊʌ]'                # nguyên âm đơn
    r'|[b-df-hj-np-tv-z]'                   # phụ âm la-tinh đơn
    r')'
)


# ── MODEL MANAGER ─────────────────────────────────────────────────────────────
class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.device       = "cpu"
        compute_type      = "int8"

        print("[INIT] Loading models …")
        t0 = time.time()

        # 1. WhisperX
        self.whisper_model  = whisperx.load_model("tiny", self.device, compute_type=compute_type)
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code="en", device=self.device
        )

        # 2. Embedding model
        self.emb_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.emb_model     = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.emb_model.eval()

        # 3. Phoneme model
        _ph = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        self.ph_tokenizer        = Wav2Vec2CTCTokenizer.from_pretrained(_ph)
        self.ph_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(_ph)
        self.ph_model            = Wav2Vec2ForCTC.from_pretrained(_ph)
        self.ph_model.eval()

        # ── torch.compile (PyTorch 2.x) – one-time JIT warm-up ──────────────
        # Uncomment if you are on PyTorch >= 2.0 and want ~20-30% extra speed:
        # self.emb_model = torch.compile(self.emb_model, mode="reduce-overhead")
        # self.ph_model  = torch.compile(self.ph_model,  mode="reduce-overhead")

        self.ref_cache    = {}
        self._initialized = True
        print(f"[INIT] Models ready in {time.time()-t0:.2f}s")

        self._precompute_references()

    # ── REFERENCE PRE-COMPUTATION ─────────────────────────────────────────────
    def _precompute_references(self):
        if not os.path.exists(REFERENCE_DIR):
            print(f"[WARN] Reference dir '{REFERENCE_DIR}' not found.")
            return

        print(f"[INIT] Pre-computing reference embeddings …")
        t0    = time.time()
        files = [f for f in os.listdir(REFERENCE_DIR) if f.endswith((".mp3", ".wav"))]

        for filename in files:
            word_norm = re.sub(r"[^\w]", "", os.path.splitext(filename)[0].lower())
            if word_norm in self.ref_cache:
                continue
            path = os.path.join(REFERENCE_DIR, filename)
            try:
                waveform, sr = torchaudio.load(path)
                if sr != SAMPLE_RATE:
                    waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                audio_np = waveform.squeeze().numpy()

                inputs = self.emb_processor(audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                with torch.inference_mode():
                    emb = self.emb_model(**inputs).last_hidden_state.mean(dim=1)
                self.ref_cache[word_norm] = emb
            except Exception as e:
                print(f"[ERR] Ref '{filename}': {e}")

        print(f"[INIT] {len(self.ref_cache)} refs in {time.time()-t0:.2f}s")

    def get_ref_emb(self, word: str):
        return self.ref_cache.get(re.sub(r"[^\w]", "", word.lower()))


models = ModelManager()


# ── PHONEME HELPER (cached) ────────────────────────────────────────────────────
@lru_cache(maxsize=512)
def get_phonemes_expected(text: str) -> tuple:  # tuple để lru_cache an toàn
    text = text.lower().replace(".", "").replace(",", "").strip()
    try:
        res = subprocess.run(
            ["espeak-ng", "-q", "--ipa", "-v", "en-us", text],  # bỏ =3, dùng --ipa thường
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=2, check=True,
        )
        ipa = res.stdout.decode("utf-8", errors="replace")

        # Normalize unicode — fix bug aɪ bị tách thành a + ɪ do code point khác nhau
        ipa = unicodedata.normalize("NFC", ipa)

        ipa = (
            ipa.strip()
            .replace("ˈ", "").replace("ˌ", "")
            .replace("_", "").replace(".", "")   # espeak-ng sinh ra _ và . boundary
            .replace("\n", " ").replace(" ", "")
        )
        return tuple(IPA_RE.findall(ipa))        # tuple thay vì list
    except Exception as e:
        print(f"[WARN] espeak-ng: {e}")
        return ()


# ── CPU-BOUND INFERENCE HELPERS ───────────────────────────────────────────────
def _run_embedding(audio_arr: np.ndarray):
    """Return full hidden states tensor + slicing ratio."""
    inputs = models.emb_processor(audio_arr, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.inference_mode():
        hidden = models.emb_model(**inputs).last_hidden_state  # (1, T, D)
    ratio = hidden.shape[1] / (len(audio_arr) / SAMPLE_RATE)
    return hidden, ratio


def _run_phoneme(audio_arr: np.ndarray):
    """Return decoded phoneme string."""
    inputs = models.ph_feature_extractor(
        audio_arr, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    )
    with torch.inference_mode():
        logits = models.ph_model(inputs.input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)[0]
    return models.ph_tokenizer.decode(pred_ids)


# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "message": "Optimized Scoring API v2"}


@app.post("/v2/score")
async def score_v2(
    expected_text: str = Form(...),
    audio: UploadFile = File(...),
):
    t_start = time.time()
    path    = None

    try:
        # ── 1. Async I/O ──────────────────────────────────────────────────────
        content = await audio.read()
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.write(fd, content); os.close(fd)
        audio_arr = whisperx.load_audio(path)   # float32 numpy array, 16 kHz
        t_load = time.time()

        # ── 2. Transcription + Alignment (still single-threaded; whisperx GIL) ──
        result = models.whisper_model.transcribe(audio_arr)
        if not result["segments"]:
            raise HTTPException(400, "No speech detected")
        aligned      = whisperx.align(
            result["segments"], models.align_model,
            models.align_metadata, audio_arr, models.device,
        )
        word_segments = aligned.get("word_segments", [])
        t_whisper = time.time()

        # ── 3. PARALLEL: embedding + phoneme in separate threads ──────────────
        loop = asyncio.get_event_loop()
        emb_future = loop.run_in_executor(_EXECUTOR, _run_embedding, audio_arr)
        ph_future  = loop.run_in_executor(_EXECUTOR, _run_phoneme,   audio_arr)

        # Also kick off espeak-ng (it's cached so near-instant on 2nd call)
        ph_expected_future = loop.run_in_executor(
            _EXECUTOR, get_phonemes_expected, expected_text
        )

        # Await all three concurrently
        (full_hidden, ratio), user_phonemes_raw, expected_phonemes = await asyncio.gather(
            emb_future, ph_future, ph_expected_future
        )
        t_analysis = time.time()

        # ── 4. Word similarity scoring ────────────────────────────────────────
        # Dùng cùng IPA_RE cho user phonemes để nhất quán
        user_phonemes = list(IPA_RE.findall(
            unicodedata.normalize("NFC", user_phonemes_raw)
        ))

        word_results              = []
        total_sim, sim_count = 0.0, 0

        for ws in word_segments:
            word = ws.get("word")
            if not word or "start" not in ws:
                continue
            ref_emb = models.get_ref_emb(word)
            if ref_emb is not None:
                idx_s = int(ws["start"] * ratio)
                idx_e = max(idx_s + 1, int(ws["end"] * ratio))
                word_emb = full_hidden[:, idx_s:idx_e, :].mean(dim=1)
                sim      = cosine_similarity(word_emb, ref_emb).item()
                word_results.append({
                    "word":             word,
                    "start":            ws["start"],
                    "end":              ws["end"],
                    "similarity_score": round(sim, 3),
                    "feedback": (
                        "Excellent" if sim > 0.85
                        else "Good"  if sim > 0.70
                        else "Needs practice"
                    ),
                })
                total_sim += sim
                sim_count += 1
            else:
                word_results.append({"word": word, "feedback": "No reference"})

        avg_sim = round(total_sim / sim_count, 3) if sim_count else 0.0

        # ── 5. Phoneme matching ───────────────────────────────────────────────
        matcher = SequenceMatcher(None, expected_phonemes, user_phonemes)
        details, correct = [], 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for i in range(i1, i2):
                    details.append({"phoneme": expected_phonemes[i], "status": "correct", "tip": ""})
                    correct += 1
            elif tag in ("delete", "replace"):
                for i in range(i1, i2):
                    p = expected_phonemes[i]
                    details.append({
                        "phoneme": p,
                        "status":  "mispronounced" if tag == "replace" else "missing",
                        "tip":     p,
                    })
            elif tag == "insert":
                for j in range(j1, j2):
                    details.append({"phoneme": user_phonemes[j], "status": "extra", "tip": "Âm thừa"})

        p_score    = round(correct / len(expected_phonemes), 3) if expected_phonemes else 0.0
        total_time = time.time() - t_start

        print(
            f"[OPT] IO:{t_load-t_start:.2f}s | "
            f"Whisper:{t_whisper-t_load:.2f}s | "
            f"AI(parallel):{t_analysis-t_whisper:.2f}s | "
            f"Total:{total_time:.2f}s"
        )

        return {
            "success":          True,
            "overall_score":    round((avg_sim + p_score) / 2, 3),
            "processing_time":  round(total_time, 2),
            "step1_audio_similarity": {
                "average_score": avg_sim,
                "word_details":  word_results,
            },
            "step2_phoneme_analysis": {
                "accuracy":          p_score,
                "expected_phonemes": expected_phonemes,
                "user_phonemes":     user_phonemes,
                "details":           details,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERR] {e}")
        raise HTTPException(500, str(e))
    finally:
        if path and os.path.exists(path):
            os.remove(path)


# ── ENTRY POINT ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Setup ngrok tunnel
    public_url = ngrok.connect(8000, domain="satyr-dashing-officially.ngrok-free.app")
    print("Public URL:", public_url)

    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000)