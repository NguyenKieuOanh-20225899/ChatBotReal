import os
import sys
import time
import re
from dotenv import load_dotenv
import google.generativeai as genai
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 1. Setup đường dẫn để import được src và experiments
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

# Import bộ tìm kiếm nâng cao (Hybrid + Rerank)
from src.retrieval_service import LegalRetriever

# 2. Config API
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("❌ Lỗi: Chưa tìm thấy GOOGLE_API_KEY trong file .env")
    sys.exit(1)

genai.configure(api_key=API_KEY)

# 3. Khởi tạo (Chỉ chạy 1 lần khi bật app)
print("⏳ Đang khởi tạo hệ thống tìm kiếm (Hybrid Search + Rerank)...")
# Lưu ý: Đảm bảo file config.yaml nằm đúng chỗ hoặc truyền đường dẫn tuyệt đối
retriever = LegalRetriever(config_path="experiments/config.yaml")
model = genai.GenerativeModel("gemini-2.0-flash")
print("✅ Hệ thống đã sẵn sàng!\n")

def extract_source_from_text(text):
    """Hàm phụ trợ để tách tên nguồn từ chuỗi format '[Nguồn]: Nội dung'"""
    match = re.match(r"\[(.*?)]:", text)
    if match:
        return match.group(1)
    return "Không rõ nguồn"

def query_advanced(query: str):
    """
    Quy trình RAG nâng cao:
    1. Retrieve (BM25 + Vector)
    2. Rerank (Cross-Encoder)
    3. Generate (Gemini)
    """
    t0 = time.perf_counter()

    # --- BƯỚC 1 & 2: TÌM KIẾM & RERANK ---
    # Hàm này trả về list các chuỗi: "[Source]: Content"
    context_list = retriever.retrieve(query)

    # Tách nguồn để hiển thị cho đẹp
    sources = [extract_source_from_text(c) for c in context_list]
    # Lọc trùng nguồn
    unique_sources = list(set(sources))

    # --- BƯỚC 3: CHUẨN BỊ PROMPT ---
    if not context_list:
        return "Xin lỗi, tôi không tìm thấy văn bản pháp luật nào liên quan đến câu hỏi của bạn trong cơ sở dữ liệu.", [], time.perf_counter() - t0

    context_text = "\n\n".join(context_list)

    prompt = f"""
Bạn là trợ lý luật sư AI chuyên nghiệp, am hiểu pháp luật Việt Nam.
Nhiệm vụ: Trả lời câu hỏi dựa trên CÁC ĐOẠN VĂN BẢN ĐƯỢC CUNG CẤP dưới đây.

Yêu cầu quan trọng:
1. Trả lời chính xác, đi thẳng vào vấn đề. KHÔNG dùng các từ đệm gây mâu thuẫn (Ví dụ: Tránh nói "Có, không được phép..." mà hãy nói thẳng "Không, pháp luật không cho phép..." hoặc "Chồng không có quyền...").
2. BẮT BUỘC phải trích dẫn điều luật cụ thể (Ví dụ: Theo Điều 5, Luật...).
3. Nếu thông tin không có trong ngữ cảnh, hãy nói "Tôi không tìm thấy quy định trong dữ liệu hiện tại".
4. Giọng văn khách quan, trang trọng, dứt khoát.

--- DỮ LIỆU THAM KHẢO ---
{context_text}
--- KẾT THÚC DỮ LIỆU ---

Câu hỏi: {query}
Câu trả lời:
"""

    # --- BƯỚC 4: GỌI GEMINI ---
    try:
        resp = model.generate_content(prompt)
        answer = resp.text
    except Exception as e:
        answer = f"Lỗi khi gọi Google API: {e}"

    latency = time.perf_counter() - t0
    return answer, unique_sources, latency

if __name__ == "__main__":
    print("🤖 CHATBOT PHÁP LUẬT (Hybrid RAG + Gemini 2.0)")
    print("---------------------------------------------")
    while True:
        q = input("\n👤 Nhập câu hỏi (gõ 'exit' để thoát): ")
        if q.lower() in ["exit", "quit", "thoat"]:
            break

        if not q.strip():
            continue

        print("🔍 Đang tra cứu và phân tích...")
        ans, srcs, t = query_advanced(q)

        print("\n🤖 Trả lời:")
        print(ans)

        print("\n📚 Nguồn tham khảo:")
        if srcs:
            for s in srcs:
                print(f"  → {s}")
        else:
            print("  → Không có nguồn.")

        print(f"\n⚡ Thời gian xử lý: {t:.2f}s")