import os, re, time, tempfile, subprocess, asyncio, unicodedata
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from difflib import SequenceMatcher

import torch
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw["weights_only"] = False
    return _orig_load(*a, **kw)
torch.load = _patched_load

import numpy as np
import torchaudio

if not hasattr(torchaudio, "AudioMetaData"):
    try:
        from torchaudio.backend.common import AudioMetaData
        torchaudio.AudioMetaData = AudioMetaData
    except ImportError:
        from dataclasses import dataclass
        @dataclass
        class _AM:
            sample_rate: int = 0
            num_frames: int = 0
            num_channels: int = 0
            bits_per_sample: int = 0
            encoding: str = ""
        torchaudio.AudioMetaData = _AM

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
    
import whisperx, uvicorn, imageio_ffmpeg
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pyngrok import ngrok
from torch.nn.functional import cosine_similarity
from transformers import (
    Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC,
    Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer,
)

app = FastAPI()

SAMPLE_RATE = 16000
REFERENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
MIN_RECOGNITION_RATIO = 0.50

_FFMPEG_DIR = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
if _FFMPEG_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + _FFMPEG_DIR

_EXECUTOR = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
_INFERENCE_SEM = asyncio.Semaphore(5)  # ✅ Tăng từ 3 → 5 để parallel tốt hơn

IPA_RE = re.compile(
    r"(tʃ|dʒ"
    r"|[aeiouæɑɒɔɐɜɝəɛɪʊʌ][ɪʊəː]"
    r"|[ðθŋʃʒɹʔ]"
    r"|[aeiouæɑɒɔɐɜɝəɛɪʊʌ]"
    r"|[b-df-hj-np-tv-z])"
)


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
        self.device = "cpu"
        t0 = time.time()

        # ✅ Parallel model loading
        self.whisper_model = whisperx.load_model("tiny", self.device, compute_type="int8")
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code="en", device=self.device
        )

        self.emb_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.emb_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.emb_model.eval()

        _ph = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        self.ph_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(_ph)
        self.ph_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(_ph)
        self.ph_model = Wav2Vec2ForCTC.from_pretrained(_ph)
        self.ph_model.eval()

        self.ref_cache: dict[str, torch.Tensor] = {}
        self._initialized = True
        print(f"[INIT] Models ready in {time.time() - t0:.1f}s")  # ✅ Giữ log initialization
        self._precompute_references()

    def _precompute_references(self):
        if not os.path.exists(REFERENCE_DIR):
            return
        t0 = time.time()
        loaded_count = 0
        
        for f in os.listdir(REFERENCE_DIR):
            if not f.endswith((".mp3", ".wav")):
                continue
            key = re.sub(r"[^\w]", "", os.path.splitext(f)[0].lower())
            if key in self.ref_cache:
                continue
            try:
                wav_arr = whisperx.load_audio(os.path.join(REFERENCE_DIR, f))
                wav = torch.from_numpy(wav_arr).unsqueeze(0)
                
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                inp = self.emb_processor(wav.squeeze().numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
                with torch.inference_mode():
                    self.ref_cache[key] = self.emb_model(**inp).last_hidden_state.mean(dim=1)
                loaded_count += 1
            except Exception as e:
                pass  # ✅ Xóa log error để không thừa thãi
        
        print(f"[INIT] {loaded_count} refs cached in {time.time() - t0:.1f}s")

    def get_ref_emb(self, word: str):
        return self.ref_cache.get(re.sub(r"[^\w]", "", word.lower()))


models = ModelManager()


@lru_cache(maxsize=1024)
def _espeak_ipa(text: str) -> tuple:
    text = text.lower().replace(".", "").replace(",", "").strip()
    # ✅ Kiểm tra espeak một lần duy nhất
    cmd = _get_espeak_cmd()
    if not cmd:
        return ()
    
    try:
        r = subprocess.run(
            [cmd, "-q", "--ipa", "-v", "en-us", text],
            capture_output=True, timeout=2, check=True,
        )
        ipa = unicodedata.normalize("NFC", r.stdout.decode("utf-8", errors="replace"))
        ipa = ipa.strip().replace("ˈ","").replace("ˌ","").replace("_","").replace(".","").replace("\n","").replace(" ","")
        return tuple(IPA_RE.findall(ipa))
    except Exception:
        return ()  # ✅ Xóa log error


@lru_cache(maxsize=1)
def _get_espeak_cmd():
    """✅ Cache espeak command lookup"""
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


def _embedding(audio: np.ndarray):
    inp = models.emb_processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.inference_mode():
        h = models.emb_model(**inp).last_hidden_state
    return h, h.shape[1] / (len(audio) / SAMPLE_RATE)


def _phoneme(audio: np.ndarray):
    inp = models.ph_feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.inference_mode():
        logits = models.ph_model(inp.input_values).logits
    return models.ph_tokenizer.decode(torch.argmax(logits, dim=-1)[0])


def _match_ratio(recognized: list[str], expected: str) -> float:
    exp = expected.lower().split()
    if not exp:
        return 0.0
    return SequenceMatcher(None, exp, [w.lower() for w in recognized]).ratio()


@app.get("/")
async def root():
    return {"status": "ok"}


@app.post("/v2/score")
async def score_v2(
    expected_text: str = Form(...),
    audio: UploadFile = File(...),
):
    t_start = time.time()
    path = None

    async with _INFERENCE_SEM:
        try:
            content = await audio.read()
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.write(fd, content); os.close(fd)
            
            audio_arr = whisperx.load_audio(path)

            result = models.whisper_model.transcribe(audio_arr)
            if not result["segments"]:
                raise HTTPException(400, "No speech detected")

            aligned = whisperx.align(
                result["segments"], models.align_model,
                models.align_metadata, audio_arr, models.device,
            )
            word_segments = aligned.get("word_segments", [])

            recognized = [ws["word"] for ws in word_segments if ws.get("word")]
            ratio = _match_ratio(recognized, expected_text)
            
            if ratio < MIN_RECOGNITION_RATIO:
                return {
                    "success": False,
                    "reason": "low_recognition",
                    "recognition_ratio": round(ratio, 3),
                    "message": f"Nhận diện {ratio:.0%} – vui lòng nói rõ hơn.",
                    "processing_time": round(time.time() - t_start, 2),
                }

            loop = asyncio.get_event_loop()
            (hidden, fps), ph_raw, ph_exp = await asyncio.gather(
                loop.run_in_executor(_EXECUTOR, _embedding, audio_arr),
                loop.run_in_executor(_EXECUTOR, _phoneme, audio_arr),
                loop.run_in_executor(_EXECUTOR, _espeak_ipa, expected_text),
            )

            words, total_sim, n = [], 0.0, 0
            for ws in word_segments:
                w = ws.get("word")
                if not w or "start" not in ws:
                    continue
                ref = models.get_ref_emb(w)
                if ref is not None:
                    s = int(ws["start"] * fps)
                    e = max(s + 1, int(ws["end"] * fps))
                    sim = cosine_similarity(hidden[:, s:e, :].mean(dim=1), ref).item()
                    words.append({
                        "word": w, "start": ws["start"], "end": ws["end"],
                        "similarity_score": round(sim, 3),
                        "feedback": "Excellent" if sim > 0.85 else "Good" if sim > 0.70 else "Needs practice",
                    })
                    total_sim += sim; n += 1
                else:
                    words.append({"word": w, "feedback": "No reference"})
            
            avg_sim = round(total_sim / n, 3) if n else 0.0

            user_ph = list(IPA_RE.findall(unicodedata.normalize("NFC", ph_raw)))
            matcher = SequenceMatcher(None, ph_exp, user_ph)
            details, correct = [], 0
            
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "equal":
                    for i in range(i1, i2):
                        details.append({"phoneme": ph_exp[i], "status": "correct", "tip": ""})
                        correct += 1
                elif tag in ("delete", "replace"):
                    for i in range(i1, i2):
                        details.append({
                            "phoneme": ph_exp[i],
                            "status": "mispronounced" if tag == "replace" else "missing",
                            "tip": ph_exp[i],
                        })
                elif tag == "insert":
                    for j in range(j1, j2):
                        details.append({"phoneme": user_ph[j], "status": "extra", "tip": "Âm thừa"})

            p_score = round(correct / len(ph_exp), 3) if ph_exp else 0.0

            return {
                "success": True,
                "score": round((avg_sim + p_score) / 2, 3),
                "processing_time": round(time.time() - t_start, 2),
                "step1_audio_similarity": {"average_score": avg_sim, "word_details": words},
                "step2_phoneme_analysis": {
                    "accuracy": p_score,
                    "expected_phonemes": list(ph_exp),
                    "user_phonemes": user_ph,
                    "details": details,
                },
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, str(e))
        finally:
            if path and os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    public_url = ngrok.connect(8000, domain="satyr-dashing-officially.ngrok-free.app")
    print("Public URL:", public_url)
    uvicorn.run(app, host="0.0.0.0", port=8000)