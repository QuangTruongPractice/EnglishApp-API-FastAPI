# 🚀 English Learning App - AI Services API (FastAPI)

Hệ thống API chuyên biệt hỗ trợ xử lý AI cho ứng dụng học tiếng Anh, tập trung vào chấm điểm phát âm (Pronunciation Scoring) và xử lý phụ đề thông minh từ YouTube.

---

## 🛠 Công Nghệ Sử Dụng

- **Framework:** FastAPI (Python 3.10.9)
- **AI Models:** 
  - **WhisperX:** Chuyển đổi âm thanh thành văn bản và căn chỉnh thời gian cấp độ từ.
  - **Wav2Vec2 (Facebook):** Trích xuất đặc trưng âm thanh để chấm điểm phát âm.
  - **Groq AI (Llama 3):** Xử lý hội thoại thông minh và phản hồi nhanh.
- **Database:** SQLite (Lưu lịch sử hội thoại).
- **Phân phối:** Ngrok (Tạo đường hầm public cho mobile app truy cập).
- **Phụ thuộc hệ thống (Bắt buộc):** FFmpeg, espeak-ng.

---

## ⚙️ Yêu cầu cài đặt hệ thống (System Requirements)

Để chạy được các tính năng AI, bạn **BẮT BUỘC** phải cài đặt các công cụ sau vào máy:

### 1. FFmpeg (Xử lý âm thanh)
- **Windows:**
  1. Tải bản **`ffmpeg-8.1-essentials_build.zip`** (Khuyến nghị) tại [gyan.dev](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip).
  2. Giải nén và thêm thư mục `bin` vào **System Environment Variables (PATH)**.
  3. Kiểm tra bằng lệnh: `ffmpeg -version`
- **Linux (Ubuntu/Debian):** `sudo apt update && sudo apt install ffmpeg -y`

### 2. espeak-ng (Trích xuất âm vị)
- **Windows:**
  1. Tải file **`espeak-ng.msi`** tại [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases).
  2. Cài đặt và thêm thư mục cài đặt (thường là `C:\Program Files\eSpeak NG`) vào **PATH**.
  3. Kiểm tra bằng lệnh: `espeak-ng --version`
- **Linux (Ubuntu/Debian):** `sudo apt update && sudo apt install espeak-ng -y`

---

## ⚠️ LƯU Ý QUAN TRỌNG (Đọc kỹ trước khi chạy)

### 1. Thời gian khởi động lần đầu (Critical)
Trong lần đầu tiên khởi chạy ứng dụng, hệ thống sẽ tiến hành quét toàn bộ thư mục `/audio` (chứa hơn 5000 file âm thanh mẫu). 
- **Quá trình:** Hệ thống trích xuất đặc trưng (embeddings) của các file này và lưu vào `reference_cache.pt`.
- **Thời gian:** Có thể mất từ **3 - 7 phút** tùy vào cấu hình máy. Các lần khởi động sau sẽ chỉ mất vài giây nhờ vào file cache này.

### 2. Môi trường Windows & Bản vá DLL
Hệ thống đã tích hợp sẵn tệp `app/core/patches.py` để xử lý triệt để lỗi `WinError 1114` (DLL initialization failed) phổ biến trên Windows khi chạy PyTorch. **Không nên xóa các dòng cấu hình môi trường trong file này.**

### 3. Yêu cầu phần cứng
- **RAM:** Tối thiểu 8GB (Khuyến nghị 16GB).
- **Ổ cứng:** Trống tối thiểu 5GB để tải các Model AI từ HuggingFace.
- **Python:** Bắt buộc sử dụng **Python 3.10.x** để đảm bảo tính tương thích của các thư viện AI.

---

## 🚀 Hướng Dẫn Cài Đặt

1. **Tạo môi trường ảo:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
