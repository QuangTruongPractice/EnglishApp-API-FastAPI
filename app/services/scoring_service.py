import os
import re
import time
import subprocess
import asyncio
import unicodedata
import torch
import numpy as np
import whisperx
from .model_manager import model_manager
from .ipa_tips import IPA_VIET_TIPS
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from difflib import SequenceMatcher
from torch.nn.functional import cosine_similarity
from ..core import patches  # Đảm bảo các bản vá lỗi cho torch/torchaudio đã được nạp

SAMPLE_RATE = 16000
MIN_RECOGNITION_RATIO = 0.50

_EXECUTOR = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
_INFERENCE_SEM = asyncio.Semaphore(5)  # Giới hạn số lượng xử lý cùng lúc

IPA_RE = re.compile(
    r"(tʃ|dʒ|d͡ʒ|t͡ʃ"
    r"|[aeiouæɑɒɔɐɜɝəɛɪʊʌ][ɪʊəː:]"
    r"|[ðθŋʃʒɹʔr]"
    r"|[aeiouæɑɒɔɐɜɝəɛɪʊʌ]"
    r"|[b-df-hj-np-tv-z])"
)

_ESPEAK_CMD_CACHE = None

def _get_espeak_cmd():
    """Tìm đường dẫn file thực thi của espeak-ng"""
    global _ESPEAK_CMD_CACHE
    if _ESPEAK_CMD_CACHE:
        return _ESPEAK_CMD_CACHE
        
    cmds = [
        "espeak-ng",
        r"C:\Program Files\eSpeak NG\espeak-ng.exe"
    ]
    for cmd in cmds:
        try:
            r = subprocess.run([cmd, "-q", "--version"], capture_output=True, timeout=5)
            if r.returncode == 0:
                _ESPEAK_CMD_CACHE = cmd
                return cmd
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"_get_espeak_cmd warning for {cmd}: {e}")
            continue
    return None

@lru_cache(maxsize=1024)
def _espeak_ipa(text: str) -> tuple[str, tuple[str, ...]]:
    """Lấy phiên âm IPA bằng espeak-ng (có dùng cache)"""
    if not text:
        return "", ()
    
    text = text.lower().replace(".", "").replace(",", "").strip()
    cmd = _get_espeak_cmd()
    if not cmd:
        print("_espeak_ipa: Không tìm thấy espeak-ng hoặc biến môi trường bị lỗi!")
        return "", ()
    
    try:
        r = subprocess.run(
            [cmd, "-q", "--ipa", "-v", "en-us", text],
            capture_output=True, timeout=5, check=True,
        )
        ipa = unicodedata.normalize("NFC", r.stdout.decode("utf-8", errors="replace"))
        # Dọn dẹp các ký hiệu thừa trong chuỗi IPA
        ipa = ipa.strip().replace("ˈ","").replace("ˌ","").replace("_","").replace(".","").replace("\n","").replace(" ","")
        # Trả về cả chuỗi gốc đã làm sạch và danh sách token
        return ipa, tuple(IPA_RE.findall(ipa))
    except subprocess.CalledProcessError as e:
        print(f"espeak-ng lỗi: {e}, stderr: {e.stderr}")
        return "", ()
    except Exception as getattr_e:
        print(f"espeak-ng ngoại lệ khác: {getattr_e}")
        return "", ()

def _extract_embedding(audio: np.ndarray):
    """Trích xuất vector đặc trưng âm thanh để chấm điểm tương đồng"""
    inp = model_manager.emb_processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    if model_manager.device == "cuda":
        inp = {k: v.to(model_manager.device) for k, v in inp.items()}
    with torch.inference_mode():
        h = model_manager.emb_model(**inp).last_hidden_state
    return h, h.shape[1] / (len(audio) / SAMPLE_RATE)

def _extract_phonemes(audio: np.ndarray):
    """Dự đoán các âm tiết (phonemes) từ giọng nói của người dùng"""
    inp = model_manager.ph_feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    if model_manager.device == "cuda":
        inp = {k: v.to(model_manager.device) for k, v in inp.items()}

    if isinstance(inp, dict):
        input_values = inp["input_values"]
    else:
        input_values = inp.input_values

    with torch.inference_mode():
        logits = model_manager.ph_model(input_values).logits
    return model_manager.ph_tokenizer.decode(torch.argmax(logits, dim=-1)[0])

class ScoringService:
    """Dịch vụ chấm điểm phát âm (xử lý song song nhiều request)"""

    async def score_v2_logic(self, expected_text: str, audio_path: str):
        """Quy trình chấm điểm tổng hợp (Whisper + Wav2Vec2 + IPA)"""
        t_start = time.time()
        
        async with _INFERENCE_SEM:
            try:
                audio_arr = whisperx.load_audio(audio_path)
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(_EXECUTOR, model_manager.whisper_model.transcribe, audio_arr)
                if not result["segments"]:
                    return {"success": False, "error": "No speech detected in audio."}

                # 2. Khớp thời gian chi tiết từng từ - chạy trong executor để tránh chặn event loop
                aligned = await loop.run_in_executor(
                    _EXECUTOR,
                    whisperx.align,
                    result["segments"], model_manager.align_model,
                    model_manager.align_metadata, audio_arr, model_manager.device,
                )
                word_segments = aligned.get("word_segments", [])
                
                # 3. Kiểm tra tỷ lệ nhận diện từ thành công
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

                # 4. Trích xuất đặc trưng âm thanh và âm tiết tuần tự để tránh nghẽn tài nguyên
                loop = asyncio.get_event_loop()
                (hidden, fps), ph_raw, (ph_exp_raw, ph_exp) = await asyncio.gather(
                    loop.run_in_executor(_EXECUTOR, _extract_embedding, audio_arr),
                    loop.run_in_executor(_EXECUTOR, _extract_phonemes, audio_arr),
                    loop.run_in_executor(_EXECUTOR, _espeak_ipa, expected_text),
                )
                print(f"[PHONEME] Raw Expected: {ph_exp_raw}")
                print(f"[PHONEME] Raw User:     {ph_raw}")

                # 5. Chấm điểm độ tương đồng âm thanh (dùng vector nhúng)
                words_details, total_sim, n_valid = [], 0.0, 0
                for ws in word_segments:
                    w = ws.get("word")
                    if not w or "start" not in ws: continue
                    
                    ref_emb = model_manager.get_ref_emb(w)
                    if ref_emb is not None:
                        s_idx = int(ws["start"] * fps)
                        e_idx = max(s_idx + 1, int(ws["end"] * fps))
                        
                        # Chuyển dữ liệu lên GPU nếu máy có hỗ trợ
                        chunk = hidden[:, s_idx:e_idx, :].mean(dim=1)
                        if model_manager.device == "cuda":
                            ref_emb = ref_emb.to(model_manager.device)
                            
                        sim = cosine_similarity(chunk, ref_emb).item()
                        
                        words_details.append({
                            "word": w, "start": ws["start"], "end": ws["end"],
                            "similarity_score": round(sim, 3),
                            "feedback": "Excellent" if sim > 0.85 else "Good" if sim > 0.70 else "Needs practice",
                        })
                        total_sim += sim; n_valid += 1
                    else:
                        words_details.append({"word": w, "feedback": "No reference available"})
                
                avg_sim = round(total_sim / n_valid, 3) if n_valid else 0.0

                # 6. Phân tích chi tiết ở mức âm tiết (So sánh IPA)
                user_ph = list(IPA_RE.findall(unicodedata.normalize("NFC", ph_raw)))
                
                # Chuẩn hóa để so sánh nhưng giữ mapping để hiển thị IPA gốc
                def normalize_ph(plist):
                    norm_list, norm_map = [], []
                    for i, p in enumerate(plist):
                        # Loại bỏ dấu độ dài, dấu nhấn và chuẩn hóa ký tự 'r'
                        p_norm = p.replace("ː", "").replace(":", "").replace("ɹ", "r").replace("ˈ", "").replace("ˌ", "")
                        if p_norm == "r":
                            continue  # Bỏ qua âm r theo yêu cầu (tránh sai lệch giọng vùng miền)
                        if p_norm:
                            norm_list.append(p_norm)
                            norm_map.append(i)
                    return norm_list, norm_map
                
                ph_exp_norm, ph_exp_map = normalize_ph(ph_exp)
                user_ph_norm, user_ph_map = normalize_ph(user_ph)
                
                print(f"[PHONEME] Tokens Exp (Norm):  {ph_exp_norm}")
                print(f"[PHONEME] Tokens User (Norm): {user_ph_norm}")
                
                matcher = SequenceMatcher(None, ph_exp_norm, user_ph_norm)
                ph_details, correct_count = [], 0
                
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag == "equal":
                        for k in range(i1, i2):
                            orig_idx = ph_exp_map[k]
                            ph_details.append({"phoneme": ph_exp[orig_idx], "status": "correct", "tip": ""})
                            correct_count += 1
                    elif tag in ("delete", "replace"):
                        for k in range(i1, i2):
                            orig_idx = ph_exp_map[k]
                            ph = ph_exp[orig_idx]
                            # Tìm mẹo phát âm trong từ điển hoặc gợi ý mặc định
                            tip = IPA_VIET_TIPS.get(ph, f"Mẹo: Tập phát âm âm /{ph}/")
                            
                            ph_details.append({
                                "phoneme": ph,
                                "status": "mispronounced" if tag == "replace" else "missing",
                                "tip": tip,
                            })
                    elif tag == "insert":
                        for k in range(j1, j2):
                            orig_idx = user_ph_map[k]
                            ph_details.append({"phoneme": user_ph[orig_idx], "status": "extra", "tip": "Âm thừa"})

                p_accuracy = round(correct_count / len(ph_exp_norm), 3) if ph_exp_norm else 0.0

                return {
                    "success": True,
                    "score": round((avg_sim + p_accuracy) / 2, 3),
                    "processing_time": round(time.time() - t_start, 2),
                    "step1_audio_similarity": {"average_score": avg_sim, "word_details": words_details},
                    "step2_phoneme_analysis": {
                        "accuracy": p_accuracy,
                        "details": ph_details,
                    },
                }

            except Exception as e:
                print(f"Scoring engine error occurred: {e}")
                return {"success": False, "error": f"Scoring engine error: {str(e)}"}

    @staticmethod
    def _match_ratio(recognized: list[str], expected: str) -> float:
        """Tính tỷ lệ khớp giữa từ máy nghe được và từ gốc (đã loại bỏ dấu câu để tránh lệch điểm)"""
        cleaned_expected = re.sub(r"[.,\/#!$%\^&\*;:{}=\-_`~()?]", "", expected.lower())
        exp_list = cleaned_expected.split()
        if not exp_list:
            return 0.0
        cleaned_recognized = [re.sub(r"[.,\/#!$%\^&\*;:{}=\-_`~()?]", "", w.lower()) for w in recognized]
        return SequenceMatcher(None, exp_list, cleaned_recognized).ratio()

# Tạo instance duy nhất để sử dụng ở các nơi khác
scoring_service = ScoringService()