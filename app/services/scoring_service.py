import os
import re
import csv
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

# Tái sử dụng regex gốc của bạn, đồng thời cập nhật để hỗ trợ
IPA_RE = re.compile(
    r"(tʃ|dʒ|d͡ʒ|t͡ʃ"
    r"|[aeiouæɑɒɔɐɜɝəɛɪʊʌ][ɪʊəː:]"
    r"|[ðθŋʃʒɹʔrɡɾ]"
    r"|[aeiouæɑɒɔɐɜɝəɛɪʊʌɚɝ]"
    r"|[b-df-hj-np-tv-z])"
)

# TẢI MA TRẬN TƯƠNG ĐỒNG ÂM TIẾT (PHONEME SIMILARITY MATRIX)
PHONEME_MATRIX_ALIASES: dict[str, str] = {
    "ɚ": "ə˞",
    "ɝ": "ɜ˞",
}

def load_similarity_matrix(file_name: str) -> dict[tuple[str, str], float]:
    """Tải dữ liệu ma trận độ tương đồng từ file CSV ngoài."""
    matrix_path = os.path.join(os.path.dirname(__file__), "..", "..", file_name)
    print(f"[MATRIX] ✅ Loaded: {os.path.abspath(matrix_path)}")

    similarity: dict[tuple[str, str], float] = {}
    try:
        with open(matrix_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            if not header:
                return {}
            phonemes = header[1:]
            for row in reader:
                if not row or len(row) < 2:
                    continue
                p1 = row[0]
                for p2, value in zip(phonemes, row[1:]):
                    if not value:
                        continue
                    try:
                        score = float(value)
                    except ValueError:
                        continue
                    if score <= 0.0:
                        continue
                    similarity[(p1, p2)] = score
    except Exception as e:
        print(f"Warning: Không thể load ma trận {file_name}: {e}")
    return similarity

EXTERNAL_PHONEME_SIMILARITY: dict[tuple[str, str], float] = {}
EXTERNAL_PHONEME_SIMILARITY.update(load_similarity_matrix("vowel_similarity_matrix.csv"))
EXTERNAL_PHONEME_SIMILARITY.update(load_similarity_matrix("consonant_similarity_matrix.csv"))

def normalize_matrix_phoneme(phoneme: str) -> str:
    return PHONEME_MATRIX_ALIASES.get(phoneme, phoneme)

def phoneme_similarity(p1: str, p2: str) -> float:
    """Trả về độ tương đồng ∈ [0.0, 1.0] giữa 2 phoneme IPA."""
    if p1 == p2:
        return 1.0
    p1 = normalize_matrix_phoneme(p1)
    p2 = normalize_matrix_phoneme(p2)
    score = EXTERNAL_PHONEME_SIMILARITY.get((p1, p2))
    if score is not None:
        return score
    score = EXTERNAL_PHONEME_SIMILARITY.get((p2, p1))
    if score is not None:
        print(f"[MATRIX HIT] '{p1}' <-> '{p2}' = {score}")  # ← thêm đây
        return score
    return 0.0

def normalize_phoneme_generic(phoneme_str: str) -> str:
    """Loại bỏ các dấu trọng âm và khoảng trắng."""
    if not phoneme_str:
        return ""
    normalized = unicodedata.normalize("NFC", phoneme_str)
    normalized = re.sub(r"[ˈˌSubˑ·]", "", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized

# THUẬT TOÁN WEIGHTED EDIT DISTANCE & ALIGNMENT TRACEBACK
def calculate_weighted_edit_distance(
    seq1: tuple[str, ...],
    seq2: tuple[str, ...],
    token_word_indices: list[int] = None
):
    """
    Tính khoảng cách chỉnh sửa có trọng số và trả về luồng khớp nối tối ưu.
    seq1: espeak_tokens (Chuẩn), seq2: user_tokens (Thực tế từ audio)
    """
    m, n = len(seq1), len(seq2)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = float(i)
    for j in range(n + 1):
        dp[0][j] = float(j)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sim = phoneme_similarity(seq1[i - 1], seq2[j - 1])
            sub_cost = 1.0 - sim
            dp[i][j] = min(
                dp[i - 1][j] + 1.0,               # deletion
                dp[i][j - 1] + 1.0,               # insertion
                dp[i - 1][j - 1] + sub_cost,      # substitution
            )

    # --- Traceback dựng cấu trúc chi tiết ph_details dựa trên ma trận độ tương đồng ---
    ph_details = []
    i, j = m, n
    
    while i > 0 or j > 0:
        word_idx = None
        if token_word_indices and len(token_word_indices) > 0:
            if i > 0:
                word_idx = token_word_indices[i - 1]
            else:
                word_idx = token_word_indices[0]

        if i > 0 and j > 0:
            sim = phoneme_similarity(seq1[i - 1], seq2[j - 1])
            sub_cost = 1.0 - sim
            if abs(dp[i][j] - (dp[i - 1][j - 1] + sub_cost)) < 1e-9:
                p_exp = seq1[i - 1]
                p_user = seq2[j - 1]
                sim_val = float(sim)
                
                # --- CẬP NHẬT LOGIC PHÂN CẤP STATUS THEO ĐIỂM SỐ TƯƠNG ĐỒNG ---
                if sim_val > 0.80:
                    status = "correct"
                    tip = ""
                elif 0.50 < sim_val <= 0.80:
                    status = "slight"
                    tip = IPA_VIET_TIPS.get(p_exp, f"Mẹo: Bạn phát âm âm /{p_exp}/ hơi yếu, nghe gần giống âm /{p_user}/.")
                else:
                    status = "missed"
                    tip = IPA_VIET_TIPS.get(p_exp, f"Mẹo: Sai âm /{p_exp}/ (Bạn đọc thành /{p_user}/). Hãy luyện tập thêm.")
                
                detail = {
                    "phoneme": p_exp,
                    "status": status,
                    "similarity_score": round(sim_val, 3),
                    "tip": tip
                }
                if word_idx is not None:
                    detail["word_index"] = word_idx
                ph_details.append(detail)
                i -= 1
                j -= 1
                continue

        # Trường hợp thiếu âm (Bỏ sót âm trong từ gốc)
        if i > 0 and abs(dp[i][j] - (dp[i - 1][j] + 1.0)) < 1e-9:
            p_exp = seq1[i - 1]
            tip = IPA_VIET_TIPS.get(p_exp, f"Mẹo: Bạn đã bỏ sót không phát âm âm /{p_exp}/.")
            detail = {
                "phoneme": p_exp,
                "status": "missed",
                "similarity_score": 0.0,
                "tip": tip
            }
            if word_idx is not None:
                detail["word_index"] = word_idx
            ph_details.append(detail)
            i -= 1
            
        # Trường hợp thừa âm (Phát âm dư ký tự)
        elif j > 0:
            p_user = seq2[j - 1]
            ins_word_idx = None
            if token_word_indices and len(token_word_indices) > 0:
                ins_word_idx = token_word_indices[i] if i < len(token_word_indices) else token_word_indices[-1]
            detail = {
                "phoneme": p_user,
                "status": "extra",
                "similarity_score": 0.0,
                "tip": f"Âm thừa: Bạn phát âm dư âm /{p_user}/."
            }
            if ins_word_idx is not None:
                detail["word_index"] = ins_word_idx
            ph_details.append(detail)
            j -= 1

    ph_details.reverse()
    return dp[m][n], ph_details

# ESPEAK & EMBEDDING EXTRACTORS
_ESPEAK_CMD_CACHE = None

def _get_espeak_cmd():
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
        # Dọn dẹp thô sơ
        ipa = ipa.strip().replace("_","").replace(".","").replace("\n","").replace(" ","")
        return ipa, tuple(IPA_RE.findall(ipa))
    except subprocess.CalledProcessError as e:
        print(f"espeak-ng lỗi: {e}, stderr: {e.stderr}")
        return "", ()
    except Exception as getattr_e:
        print(f"espeak-ng ngoại lệ khác: {getattr_e}")
        return "", ()

def _extract_embedding(audio: np.ndarray):
    inp = model_manager.emb_processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    if model_manager.device == "cuda":
        inp = {k: v.to(model_manager.device) for k, v in inp.items()}
    with torch.inference_mode():
        h = model_manager.emb_model(**inp).last_hidden_state
    return h, h.shape[1] / (len(audio) / SAMPLE_RATE)

def _extract_phonemes(audio: np.ndarray):
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


# MAIN SCORING SERVICE
class ScoringService:
    """Dịch vụ chấm điểm phát âm sử dụng Ma Trận Độ Tương Đồng Âm Tiết"""

    async def score_v2_logic(self, expected_text: str, audio_path: str):
        t_start = time.time()
        
        async with _INFERENCE_SEM:
            try:
                # 1. Load audio bằng whisperx (trả về numpy array 1D)
                audio_arr = whisperx.load_audio(audio_path)
                print(f"[AUDIO] Loaded audio shape: {audio_arr.shape}, dtype: {audio_arr.dtype}")
                
                # CHẶT CHẼ: Kiểm tra nếu file âm thanh bị lỗi hoặc rỗng (0 samples)
                if audio_arr is None or audio_arr.shape[0] == 0:
                    print(f"[AUDIO ERROR] File audio tại '{audio_path}' bị rỗng hoặc lỗi định dạng!")
                    return {
                        "success": False, 
                        "error": "Audio file is empty or invalid. Please check your recording/upload."
                    }

                loop = asyncio.get_event_loop()
                
                # Truyền trực tiếp audio_arr (numpy array) vì WhisperX yêu cầu np.ndarray
                result = await loop.run_in_executor(
                    _EXECUTOR, 
                    model_manager.whisper_model.transcribe, 
                    audio_arr
                )
                
                if not result or not result.get("segments"):
                    return {"success": False, "error": "No speech detected in audio."}

                # 2. Khớp thời gian chi tiết từng từ
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

                # 4. Trích xuất đặc trưng âm thanh và âm tiết tuần tự song song
                loop = asyncio.get_event_loop()
                (hidden, fps), ph_raw, (ph_exp_raw, ph_exp) = await asyncio.gather(
                    loop.run_in_executor(_EXECUTOR, _extract_embedding, audio_arr),
                    loop.run_in_executor(_EXECUTOR, _extract_phonemes, audio_arr),
                    loop.run_in_executor(_EXECUTOR, _espeak_ipa, expected_text),
                )

                # 5. Chấm điểm độ tương đồng âm thanh (dùng vector nhúng)
                words_details, total_sim, n_valid = [], 0.0, 0
                for ws in word_segments:
                    w = ws.get("word")
                    if not w or "start" not in ws: continue
                    
                    ref_emb = model_manager.get_ref_emb(w)
                    if ref_emb is not None:
                        s_idx = int(ws["start"] * fps)
                        e_idx = max(s_idx + 1, int(ws["end"] * fps))
                        
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

                # 6. Phân tích mức âm tiết bằng thuật toán Weighted Edit Distance dựa trên Ma Trận
                # Tách expected_text thành từng từ để lấy phonemes riêng lẻ nhằm giữ lại ranh giới từ (word boundaries)
                words = expected_text.strip().split()
                espeak_tokens_list = []
                token_word_indices = []
                
                # Biến lưu trữ dạng chuỗi phoneme của expected_text với khoảng cách để trả về cho frontend
                expected_phoneme_words = []

                for word_idx, w in enumerate(words):
                    w_clean = re.sub(r"[.,\/#!$%\^&\*;:{}=\-_`~()?]", "", w)
                    if not w_clean:
                        continue
                    _, tokens = _espeak_ipa(w_clean)
                    
                    # Chuẩn hóa loại bỏ stress markers nhiễu cho từng token
                    clean_word_tokens = []
                    for t in tokens:
                        t_norm = normalize_phoneme_generic(t)
                        t_tokens = IPA_RE.findall(unicodedata.normalize("NFC", t_norm))
                        clean_word_tokens.extend(t_tokens)
                    
                    if clean_word_tokens:
                        espeak_tokens_list.extend(clean_word_tokens)
                        token_word_indices.extend([word_idx] * len(clean_word_tokens))
                        expected_phoneme_words.append(" ".join(clean_word_tokens))
                
                espeak_tokens = tuple(espeak_tokens_list)
                
                # Tạo chuỗi phoneme của toàn bộ expected_text với 2 khoảng trắng giữa các từ để hiển thị đẹp
                expected_phoneme_str = "  ".join(expected_phoneme_words)

                # Chuẩn hóa và tokenize user phonemes
                model_norm_str = normalize_phoneme_generic(ph_raw)
                model_tokens = tuple(IPA_RE.findall(unicodedata.normalize("NFC", model_norm_str)))

                print(f"[PHONEME] Cleaned Expected Tokens: {espeak_tokens}")
                print(f"[PHONEME] Cleaned Model Tokens   : {model_tokens}")
                print(f"[PHONEME] Token Word Indices      : {token_word_indices}")
                print(f"[PHONEME] Expected Phoneme String : {expected_phoneme_str}")

                # Chạy Thuật toán Trọng số & Khớp nối (Dự phòng trường hợp chuỗi rỗng)
                if not espeak_tokens and not model_tokens:
                    w_distance, ph_details, p_accuracy = 0.0, [], 1.0
                elif not espeak_tokens or not model_tokens:
                    max_len = max(len(espeak_tokens), len(model_tokens))
                    w_distance = float(max_len)
                    p_accuracy = 0.0
                    ph_details = []
                    for idx, p in enumerate(espeak_tokens):
                        detail = {
                            "phoneme": p, 
                            "status": "missed", 
                            "similarity_score": 0.0, 
                            "tip": "Thiếu dữ liệu âm tiết"
                        }
                        if token_word_indices:
                            detail["word_index"] = token_word_indices[idx]
                        ph_details.append(detail)
                else:
                    w_distance, ph_details = calculate_weighted_edit_distance(
                        espeak_tokens, 
                        model_tokens, 
                        token_word_indices=token_word_indices
                    )
                    max_len = max(len(espeak_tokens), len(model_tokens))
                    # Công thức: % độ chính xác âm tiết = (1 - Khoảng_Cách_Trọng_Số / Chiều_Dài_Lớn_Nhất)
                    p_accuracy = round(max(0.0, 1.0 - (w_distance / max_len)), 3)

                # Chèn các token khoảng trắng (space) vào ph_details dựa trên sự thay đổi của word_index
                final_ph_details = []
                last_word_idx = None
                for detail in ph_details:
                    curr_word_idx = detail.get("word_index")
                    if last_word_idx is not None and curr_word_idx is not None and curr_word_idx != last_word_idx:
                        # Thêm detail ranh giới từ (khoảng trắng)
                        final_ph_details.append({
                            "phoneme": " ",
                            "status": "space",
                            "similarity_score": 1.0,
                            "tip": ""
                        })
                    final_ph_details.append(detail)
                    last_word_idx = curr_word_idx
                
                print(f"[PHONEME] Final PH Details        : {final_ph_details}")

                return {
                    "success": True,
                    "score": round((avg_sim + p_accuracy) / 2, 3),
                    "processing_time": round(time.time() - t_start, 2),
                    "step1_audio_similarity": {"average_score": avg_sim, "word_details": words_details},
                    "step2_phoneme_analysis": {
                        "accuracy": p_accuracy,
                        "weighted_distance": round(w_distance, 3),
                        "details": final_ph_details,
                    },
                    "expected_phoneme": expected_phoneme_str,
                }

            except Exception as e:
                print(f"Scoring engine error occurred: {e}")
                import traceback
                traceback.print_exc()
                return {"success": False, "error": f"Scoring engine error: {str(e)}"}

    @staticmethod
    def _match_ratio(recognized: list[str], expected: str) -> float:
        cleaned_expected = re.sub(r"[.,\/#!$%\^&\*;:{}=\-_`~()?]", "", expected.lower())
        exp_list = cleaned_expected.split()
        if not exp_list:
            return 0.0
        cleaned_recognized = [re.sub(r"[.,\/#!$%\^&\*;:{}=\-_`~()?]", "", w.lower()) for w in recognized]
        return SequenceMatcher(None, exp_list, cleaned_recognized).ratio()

scoring_service = ScoringService()