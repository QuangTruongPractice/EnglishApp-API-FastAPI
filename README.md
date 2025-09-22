# EnglishApp-API-FastAPI
# 📘 FastAPI Services

Dự án này cung cấp các dịch vụ API hỗ trợ ứng dụng học tiếng Anh, bao gồm chấm điểm phát âm và xử lý phụ đề YouTube.  
Các API này được xây dựng bằng **FastAPI**, dễ dàng tích hợp với backend **Spring Boot** và frontend **React Native**.

---

## 🔧 Kiến trúc hệ thống

- **React Native (Frontend):** Gửi request từ ứng dụng di động đến Spring Boot hoặc trực tiếp đến FastAPI.  
- **Spring Boot (Backend chính):** Đóng vai trò trung gian, quản lý người dùng, bảo mật, và gọi đến FastAPI khi cần xử lý AI/ngôn ngữ.  
- **FastAPI (Service API):**
  - Chấm điểm phát âm (dựa vào độ chính xác và độ thuần thục).
  - Lấy phụ đề từ YouTube.

Luồng hoạt động mẫu:
1. Người dùng ghi âm → React Native gửi file âm thanh → Spring Boot → FastAPI → trả kết quả về Spring Boot → frontend.
2. Người dùng yêu cầu phụ đề YouTube → React Native/Spring Boot → FastAPI → trả về phụ đề JSON.

---

## 🚀 Các API chính

### 1. API chấm điểm phát âm
- **Endpoint:** `/score/score-pronunciation`
- **Phương thức:** `POST`
- **Input:** `multipart/form-data` gồm:
  - `text`: câu/đoạn cần đọc (string)
  - `audio_file`: file ghi âm (audio/wav, audio/mp3, v.v.)
- **Output mẫu:**
  ```json
  {
    "accuracy": 0.85,
    "fluency": 0.78,
    "overall": 0.82
  }
### 2. API xử lý video YouTube
- **Endpoint:** `/score/process-video`
- **Phương thức:** `POST`
- **Input:** `json` gồm:
  - `url`: "link-youtubbe"
 
Hướng dẫn cài đặt
1. Clone dự án
git clone https://github.com/your-repo/fastapi-services.git
cd fastapi-services

2. Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3. Cài đặt dependencies
pip install -r requirements.txt
