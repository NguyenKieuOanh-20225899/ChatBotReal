import os
import sys
import time
import re
from dotenv import load_dotenv
import google.generativeai as genai

# Ẩn thông báo log không cần thiết của gRPC và TensorFlow
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 1. Setup đường dẫn tuyệt đối để tránh lỗi File Not Found
# Lấy đường dẫn đến thư mục gốc của dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Import bộ tìm kiếm nâng cao (Hybrid + Rerank) từ dự án
try:
    from src.retrieval_service import LegalRetriever
except ImportError as e:
    print(f"❌ Lỗi: Không thể tìm thấy thư mục 'src'. Hãy đảm bảo bạn đang chạy từ thư mục gốc của dự án. Chi tiết: {e}")
    sys.exit(1)

# 2. Cấu hình API
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("❌ Lỗi: Chưa tìm thấy GOOGLE_API_KEY trong file .env")
    print("👉 Hãy lấy API Key miễn phí tại: https://aistudio.google.com/")
    sys.exit(1)

genai.configure(api_key=API_KEY)

# 3. Khởi tạo hệ thống (Chỉ chạy 1 lần)
print("⏳ Đang khởi tạo hệ thống tìm kiếm (Hybrid Search + Reranker)...")
CONFIG_PATH = os.path.join(BASE_DIR, "experiments", "config.yaml")

try:
    # Khởi tạo bộ truy xuất dữ liệu
    retriever = LegalRetriever(config_path=CONFIG_PATH)

    # --- KHẮC PHỤC LỖI 404: Tự động tìm tên mô hình khả dụng ---
    print("🔍 Đang xác thực mô hình Gemini 1.5 Flash...")
    models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    
    # Ưu tiên các phiên bản của gemini-1.5-flash để dùng Free Tier
    target_model_name = None
    for m in ["models/gemini-1.5-flash", "models/gemini-1.5-flash-latest", "models/gemini-1.5-flash-001"]:
        if m in models:
            target_model_name = m
            break
            
    if not target_model_name:
        # Nếu không thấy bản Flash, lấy đại diện đầu tiên có trong danh sách
        target_model_name = next((m for m in models if "flash" in m), models[0])

    print(f"✅ Đang sử dụng mô hình: {target_model_name}")
    model = genai.GenerativeModel(model_name=target_model_name)
    print("✅ Hệ thống đã sẵn sàng!\n")

except Exception as e:
    print(f"❌ Lỗi khởi tạo hệ thống: {e}")
    sys.exit(1)

def extract_source_from_text(text):
    """Làm sạch tên nguồn: loại bỏ phần mở rộng file và format lại cho đẹp"""
    match = re.match(r"\[(.*?)]:", text)
    if match:
        source_name = match.group(1)
        # Xóa các hậu tố file để hiển thị ngắn gọn
        source_name = re.sub(r"(_knowledge\.json|_clean\.txt|\.txt)$", "", source_name)
        return source_name
    return "Không rõ nguồn"

def query_advanced(query: str):
    """
    Quy trình RAG nâng cao:
    1. Retrieve (Hybrid Search)
    2. Rerank (Cross-Encoder)
    3. Generate (Gemini 1.5 Flash)
    """
    t0 = time.perf_counter()

    try:
        # BƯỚC 1 & 2: TRUY XUẤT VÀ XẾP HẠNG LẠI
        context_list = retriever.retrieve(query)

        if not context_list:
            return ("Tôi không tìm thấy văn bản pháp luật nào liên quan đến câu hỏi này trong cơ sở dữ liệu.", 
                    [], time.perf_counter() - t0)

        # Lấy danh sách nguồn duy nhất (đã làm sạch tên)
        unique_sources = sorted(list(set(extract_source_from_text(c) for c in context_list)))
        context_text = "\n\n".join(context_list)

        # BƯỚC 3: XÂY DỰNG PROMPT
        prompt = f"""
Bạn là trợ lý luật sư AI am hiểu sâu sắc pháp luật Việt Nam.
Nhiệm vụ: Trả lời câu hỏi dựa trên CÁC ĐOẠN VĂN BẢN ĐƯỢC CUNG CẤP dưới đây.

Yêu cầu:
1. Trả lời chính xác, dứt khoát. 
2. BẮT BUỘC trích dẫn Điều luật cụ thể (Ví dụ: Theo Điều 5...).
3. Nếu thông tin không nằm trong dữ liệu cung cấp, hãy nói "Tôi không tìm thấy quy định này trong cơ sở dữ liệu hiện tại".
4. Giọng văn trang trọng, khách quan.

--- DỮ LIỆU THAM KHẢO ---
{context_text}
--- KẾT THÚC DỮ LIỆU ---

Câu hỏi: {query}
Câu trả lời:
"""

        # BƯỚC 4: GỌI GEMINI SINH PHẢN HỒI
        response = model.generate_content(prompt)
        answer = response.text

    except Exception as e:
        answer = f"⚠️ Lỗi xử lý yêu cầu: {str(e)}"
        unique_sources = []

    latency = time.perf_counter() - t0
    return answer, unique_sources, latency

if __name__ == "__main__":
    print("🤖 CHATBOT PHÁP LUẬT (RAG + Gemini 1.5 Flash)")
    print("---------------------------------------------")
    
    while True:
        user_query = input("\n👤 Nhập câu hỏi (gõ 'exit' để thoát): ").strip()
        
        if user_query.lower() in ["exit", "quit", "thoat", "t"]:
            print("👋 Cảm ơn bạn đã sử dụng hệ thống!")
            break

        if not user_query:
            continue

        print("🔍 Đang phân tích dữ liệu pháp luật...")
        ans, srcs, t_elapsed = query_advanced(user_query)

        print("\n🤖 Trả lời:")
        print(ans)

        if srcs:
            print("\n📚 Nguồn trích dẫn:")
            for s in srcs:
                print(f"  → {s}")

        print(f"\n⚡ Thời gian xử lý: {t_elapsed:.2f}s")