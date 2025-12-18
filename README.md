# Chatbot Pháp Luật (Legal AI Assistant)

Dự án Chatbot hỗ trợ tra cứu và giải đáp thắc mắc về pháp luật Việt Nam sử dụng kỹ thuật **RAG (Retrieval Augmented Generation)** kết hợp với mô hình ngôn ngữ lớn **Google Gemini**.

## 🚀 Tính năng nổi bật

* **Tìm kiếm lai (Hybrid Search):** Kết hợp tìm kiếm từ khóa (BM25) và tìm kiếm ngữ nghĩa (Vector Search với FAISS) để đảm bảo độ bao phủ và chính xác.
* **Xếp hạng lại (Reranking):** Sử dụng Cross-Encoder để sắp xếp lại các văn bản tìm được, chọn ra những đoạn luật phù hợp nhất.
* **Trả lời thông minh:** Sử dụng Google Gemini 2.0 Flash để tổng hợp thông tin và trả lời câu hỏi dựa trên các văn bản luật được cung cấp.
* **Trích dẫn nguồn:** Hiển thị rõ ràng nguồn luật (Điều khoản, tên văn bản) được sử dụng để trả lời.

## 🛠 Yêu cầu hệ thống

* **Python:** 3.10 trở lên.
* **Google API Key:** Cần có khóa API từ Google AI Studio để sử dụng mô hình Gemini.

## 📦 Cài đặt

1.  **Clone dự án:**
    ```bash
    git clone [https://github.com/maniahuv/chatbotphapluat.git](https://github.com/maniahuv/chatbotphapluat.git)
    cd chatbotphapluat
    ```

2.  **Tạo môi trường ảo (Khuyến nghị):**
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # Linux/Mac:
    source venv/bin/activate
    ```

3.  **Cài đặt thư viện:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Cấu hình môi trường:**
    Tạo file `.env` tại thư mục gốc và thêm API Key của bạn vào:
    ```env
    GOOGLE_API_KEY=your_google_api_key_here
    ```

## ⚙️ Chuẩn bị dữ liệu (Data Pipeline)

Trước khi chạy chatbot, bạn cần xây dựng cơ sở tri thức và đánh chỉ mục (index) cho dữ liệu.

1.  **Xây dựng Knowledge Base (JSON):**
    Chuyển đổi các văn bản luật thô (từ `data/cleaned/`) thành định dạng JSON có cấu trúc.
    ```bash
    python scripts/build_knowledge_base.py
    ```
    *Output:* Các file `.json` sẽ được lưu trong `data/knowledge_base/`.

2.  **Tạo Index tìm kiếm (Vector & BM25):**
    Tạo các file chỉ mục để phục vụ việc tìm kiếm nhanh.
    ```bash
    python experiments/build_indexes.py
    ```
    *Output:* Các file `bm25.pkl`, `faiss.index`, `docs.json` sẽ được lưu trong `experiments/artifacts/`.

    *> Lưu ý: Đảm bảo file cấu hình `experiments/config.yaml` đã trỏ đúng đến các thư mục dữ liệu.*

## ▶️ Cách sử dụng

Sau khi đã cài đặt và chuẩn bị dữ liệu xong, bạn có thể khởi chạy chatbot:

```bash
python scripts/chatbot_legal.py
