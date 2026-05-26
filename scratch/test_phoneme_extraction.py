import os
import sys
import re
import csv
import random
import unicodedata

import torch
import numpy as np
import whisperx
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

sys.path.append(os.getcwd())
from app.services.scoring_service import _espeak_ipa, IPA_RE


# ==============================================================================
# 1. PHONEME SIMILARITY LOADING
# ==============================================================================

def load_similarity_matrix(file_name: str) -> dict[tuple[str, str], float]:
    """Tải dữ liệu ma trận độ tương đồng từ tệp CSV bên ngoài."""
    matrix_path = os.path.join(os.path.dirname(__file__), "..", file_name)
    if not os.path.exists(matrix_path):
        return {}

    similarity: dict[tuple[str, str], float] = {}
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
    return similarity


EXTERNAL_PHONEME_SIMILARITY: dict[tuple[str, str], float] = {}
EXTERNAL_PHONEME_SIMILARITY.update(load_similarity_matrix("vowel_similarity_matrix.csv"))
EXTERNAL_PHONEME_SIMILARITY.update(load_similarity_matrix("consonant_similarity_matrix.csv"))

PHONEME_MATRIX_ALIASES: dict[str, str] = {
    # Chuẩn hóa các ký hiệu nguyên âm uốn lưỡi về định dạng được sử dụng trong tiêu đề ma trận.
    "ɚ": "ə˞",
    "ɝ": "ɜ˞",
}

# ==============================================================================
# 2. PHONEME SIMILARITY FUNCTION
# ==============================================================================

def normalize_matrix_phoneme(phoneme: str) -> str:
    return PHONEME_MATRIX_ALIASES.get(phoneme, phoneme)


def phoneme_similarity(p1: str, p2: str) -> float:
    """
    Trả về điểm tương đồng ∈ [0.0, 1.0] giữa hai ký hiệu âm vị IPA.

    Quy tắc (theo thứ tự ưu tiên):
      1. Hai ký hiệu giống hệt nhau → 1.0
      2. Cặp ký hiệu (theo đúng thứ tự hoặc đảo ngược) được tìm thấy trong ma trận độ tương đồng bên ngoài → giá trị tương ứng
      3. Trường hợp khác → 0.0 (hai âm vị hoàn toàn khác nhau)

    Chi phí thay thế được sử dụng trong tính toán khoảng cách chỉnh sửa có trọng số là:
        chi_phí = 1.0 - phoneme_similarity(p1, p2)
    """
    if p1 == p2:
        return 1.0

    p1 = normalize_matrix_phoneme(p1)
    p2 = normalize_matrix_phoneme(p2)

    score = EXTERNAL_PHONEME_SIMILARITY.get((p1, p2))
    if score is not None:
        return score

    score = EXTERNAL_PHONEME_SIMILARITY.get((p2, p1))
    if score is not None:
        return score

    return 0.0


# ==============================================================================
# 3. WEIGHTED EDIT DISTANCE
# ==============================================================================

def calculate_weighted_edit_distance(
    seq1: tuple[str, ...],
    seq2: tuple[str, ...],
) -> tuple[float, list[tuple[str, str, float]]]:
    """
    Tính toán khoảng cách chỉnh sửa *có trọng số* giữa hai chuỗi âm vị.

    Khác với khoảng cách chỉnh sửa tiêu chuẩn (chi phí 0 hoặc 1 cho mỗi phép toán), 
    chi phí thay thế ở đây là một số thập phân:
        chi_phí_thay_thế(p1, p2) = 1.0 - phoneme_similarity(p1, p2)

    Chi phí chèn / xóa vẫn là 1.0 (một âm vị bị thiếu sẽ bị phạt điểm tối đa).

    Trả về
    -------
    (distance, matched_pairs)
        distance      – số thực, tổng chi phí có trọng số
        matched_pairs – danh sách các bộ (p_espeak, p_model, độ_tương_đồng) cho các bước
                        thay thế trong cách gióng hàng tối ưu (hữu ích cho việc ghi log)
    """
    m, n = len(seq1), len(seq2)

    # dp[i][j] = chi phí có trọng số nhỏ nhất để gióng hàng seq1[:i] với seq2[:j]
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = float(i)
    for j in range(n + 1):
        dp[0][j] = float(j)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sim = phoneme_similarity(seq1[i - 1], seq2[j - 1])
            sub_cost = 1.0 - sim                  # 0.0 nếu khớp hoàn toàn

            dp[i][j] = min(
                dp[i - 1][j] + 1.0,               # xóa (bỏ qua âm vị seq1)
                dp[i][j - 1] + 1.0,               # chèn (thêm âm vị seq2)
                dp[i - 1][j - 1] + sub_cost,      # thay thế (có trọng số)
            )

    # --- Traceback (truy vết ngược) để khôi phục các cặp thay thế đã khớp ---
    matched_pairs: list[tuple[str, str, float]] = []
    i, j = m, n
    while i > 0 and j > 0:
        sim = phoneme_similarity(seq1[i - 1], seq2[j - 1])
        sub_cost = 1.0 - sim
        if abs(dp[i][j] - (dp[i - 1][j - 1] + sub_cost)) < 1e-9:
            matched_pairs.append((seq1[i - 1], seq2[j - 1], sim))
            i -= 1
            j -= 1
        elif abs(dp[i][j] - (dp[i - 1][j] + 1.0)) < 1e-9:
            i -= 1                                 # xóa — không ghi log cặp nào
        else:
            j -= 1                                 # chèn — không ghi log cặp nào

    matched_pairs.reverse()
    return dp[m][n], matched_pairs


# ==============================================================================
# 4. WEIGHTED SIMILARITY SCORE
# ==============================================================================

def compute_similarity(
    espeak_tokens: tuple[str, ...],
    model_tokens: tuple[str, ...],
) -> tuple[float, float, list[tuple[str, str, float]]]:
    """
    Tính toán độ tương đồng âm vị có trọng số (0–100 %).

    Công thức:
        độ_tương_đồng = (1 - khoảng_cách_có_trọng_số / độ_dài_lớn_nhất) * 100

    Trả về
    -------
    (similarity_pct, weighted_distance, matched_pairs)
    """
    if not espeak_tokens and not model_tokens:
        return 100.0, 0.0, []
    if not espeak_tokens or not model_tokens:
        return 0.0, float(max(len(espeak_tokens), len(model_tokens))), []

    distance, matched_pairs = calculate_weighted_edit_distance(espeak_tokens, model_tokens)
    max_len = max(len(espeak_tokens), len(model_tokens))
    similarity = (1.0 - distance / max_len) * 100.0
    return similarity, distance, matched_pairs


def calculate_exact_edit_distance(
    seq1: tuple[str, ...],
    seq2: tuple[str, ...],
) -> float:
    """Tính toán khoảng cách Levenshtein chính xác giữa hai chuỗi âm vị."""
    m, n = len(seq1), len(seq2)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = float(i)
    for j in range(1, n + 1):
        dp[0][j] = float(j)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0.0 if seq1[i - 1] == seq2[j - 1] else 1.0
            dp[i][j] = min(
                dp[i - 1][j] + 1.0,
                dp[i][j - 1] + 1.0,
                dp[i - 1][j - 1] + cost,
            )

    return dp[m][n]


def compute_exact_similarity(
    espeak_tokens: tuple[str, ...],
    model_tokens: tuple[str, ...],
) -> float:
    if not espeak_tokens and not model_tokens:
        return 100.0
    if not espeak_tokens or not model_tokens:
        return 0.0

    distance = calculate_exact_edit_distance(espeak_tokens, model_tokens)
    max_len = max(len(espeak_tokens), len(model_tokens))
    return max(0.0, (1.0 - distance / max_len) * 100.0)


# ==============================================================================
# 5. NORMALIZATION (chuẩn hóa mức độ nhẹ - giữ nguyên các phân biệt âm vị)
# ==============================================================================

def normalize_phoneme_generic(phoneme_str: str) -> str:
    """
    Chuẩn hóa nhẹ nhàng nhằm loại bỏ nhiễu do *định dạng*, trong khi vẫn giữ nguyên
    các điểm khác biệt có ý nghĩa về mặt âm vị (ví dụ: độ dài nguyên âm 'iː').

    Những gì ĐƯỢC loại bỏ:
      - Dấu trọng âm ˈ ˌ (không liên quan đến bản chất âm vị)
      - Dấu ranh giới âm tiết ˑ ·
      - Khoảng trắng thừa

    Những gì KHÔNG ĐƯỢC loại bỏ:
      - Ký hiệu độ dài nguyên âm 'ː' → mô hình đôi khi phân biệt nguyên âm dài
      - Ký hiệu chất lượng nguyên âm (ɐ, ɚ, v.v.) → được xử lý bởi ma trận tương đồng
      - Cụm nguyên âm đôi (aɪ, oʊ, …) → được giữ nguyên để IPA_RE phân tích
    """
    if not phoneme_str:
        return ""

    # Bước 1: Chuẩn hóa Unicode canonical composition (vd: kết hợp dấu diacritics → dạng tổ hợp precomposed)
    normalized = unicodedata.normalize("NFC", phoneme_str)

    # Bước 2: Chỉ loại bỏ các dấu trọng âm / dấu phân cách ranh giới
    normalized = re.sub(r"[ˈˌˑ·]", "", normalized)

    # Bước 3: Thu gọn các chuỗi khoảng trắng liên tiếp (đầu ra CTC đôi khi có dư khoảng trắng)
    normalized = re.sub(r"\s+", "", normalized)

    return normalized


# ==============================================================================
# 6. MODEL & TOKENIZER LOADING
# ==============================================================================

_PH_MODEL_NAME = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(_PH_MODEL_NAME)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(_PH_MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(_PH_MODEL_NAME)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


# ==============================================================================
# 7. WORD LIST
# ==============================================================================

word_file_path = os.path.join(os.path.dirname(__file__), "..", "word.txt")
with open(word_file_path, "r", encoding="utf-8") as f:
    raw_words = f.read()

all_words = [w.strip() for w in re.split(r"[,\n]", raw_words) if w.strip()]
selected_words = random.sample(all_words, min(50, len(all_words)))
test_files = [(word, f"audio/{word}.wav") for word in selected_words]


# ==============================================================================
# 8. EVALUATION LOOP
# ==============================================================================

output_lines: list[str] = []
total_similarity = 0.0
total_exact_similarity = 0.0
valid_word_count = 0

os.makedirs("scratch", exist_ok=True)

for word, path in test_files:
    if not os.path.exists(path):
        output_lines.append(f"File {path} does not exist.\n\n")
        continue

    # --- Suy luận mô hình âm thanh (Inference) ---
    audio_arr = whisperx.load_audio(path)
    inp = feature_extractor(audio_arr, sampling_rate=16000, return_tensors="pt")
    if device == "cuda":
        inp = {k: v.to(device) for k, v in inp.items()}
    with torch.inference_mode():
        logits = model(**inp).logits
    pred_ids = torch.argmax(logits, dim=-1)[0]
    pred_phonemes_raw = tokenizer.decode(pred_ids)

    # --- Chuẩn hóa đầu ra của mô hình (Mức độ nhẹ) ---
    normalized_model_str = normalize_phoneme_generic(pred_phonemes_raw)
    model_tokens_final = tuple(
        IPA_RE.findall(unicodedata.normalize("NFC", normalized_model_str))
    )

    # --- Phiên âm tham chiếu từ eSpeak ---
    _espeak_raw, espeak_tokens = _espeak_ipa(word)
    espeak_normalized_str = normalize_phoneme_generic("".join(espeak_tokens))
    espeak_tokens_final = tuple(
        IPA_RE.findall(unicodedata.normalize("NFC", espeak_normalized_str))
    )

    # --- Chấm điểm cơ sở dựa trên khớp chính xác (Exact-match) ---
    exact_similarity = compute_exact_similarity(espeak_tokens_final, model_tokens_final)

    # --- Chấm điểm tương đồng có trọng số (Weighted similarity) ---
    similarity, w_distance, matched_pairs = compute_similarity(
        espeak_tokens_final, model_tokens_final
    )

    total_exact_similarity += exact_similarity
    total_similarity += similarity
    valid_word_count += 1

    # --- Ghi chú log chi tiết ---
    output_lines.append(f"Word: {word}\n")
    output_lines.append(f"  Espeak tokens (Final): {espeak_tokens_final}\n")
    output_lines.append(f"  Model tokens  (Final): {model_tokens_final}\n")
    output_lines.append(f"  Weighted Distance     : {w_distance:.3f}\n")
    output_lines.append(f"  Exact Similarity      : {exact_similarity:.2f}%\n")
    output_lines.append(f"  Word Accuracy        : {similarity:.2f}%\n")
    output_lines.append(f"  Improvement          : {similarity - exact_similarity:+.2f} pp\n")

    # Hiển thị các cặp thay thế có độ tương đồng < 1.0 (nghĩa là không khớp hoàn toàn)
    substitutions = [(e, m, s) for e, m, s in matched_pairs if s < 1.0]
    if substitutions:
        output_lines.append("  Matched Similarity Pairs (substitutions only):\n")
        for e_ph, m_ph, sim in substitutions:
            label = (
                "exact" if sim == 1.0 else
                "near" if sim >= 0.80 else
                "partial" if sim >= 0.40 else
                "mismatch"
            )
            output_lines.append(
                f"    [{label:8s}]  eSpeak={e_ph!r:8s}  model={m_ph!r:8s}  sim={sim:.2f}\n"
            )

    output_lines.append("-" * 48 + "\n")


# ==============================================================================
# 9. SUMMARY & OUTPUT
# ==============================================================================

if valid_word_count > 0:
    avg_accuracy = total_similarity / valid_word_count
    avg_exact_accuracy = total_exact_similarity / valid_word_count
    summary = (
        f"=== THỐNG KÊ TỔNG QUAN ===\n"
        f"Tổng số từ chạy thành công : {valid_word_count}\n"
        f"Độ tương đồng trung bình (Exact)    : {avg_exact_accuracy:.2f}%\n"
        f"Độ tương đồng trung bình (Weighted) : {avg_accuracy:.2f}%\n"
        f"Lợi ích trung bình                 : {avg_accuracy - avg_exact_accuracy:+.2f} pp\n"
        f"Scoring method                     : weighted edit distance + phoneme similarity map\n"
        + "=" * 56 + "\n\n"
    )
    output_lines.insert(0, summary)
    print(
        f"Done! Exact={avg_exact_accuracy:.2f}%, Weighted={avg_accuracy:.2f}%,"
        f" Gain={avg_accuracy - avg_exact_accuracy:+.2f} pp"
    )
else:
    print("Không tìm thấy tệp âm thanh hợp lệ.")

with open("scratch/test_phoneme_output.txt", "w", encoding="utf-8") as f:
    f.writelines(output_lines)