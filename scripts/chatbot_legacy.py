# scripts/chatbot_legacy.py
import os, time
import sys

# Thêm đường dẫn root để tránh lỗi import nếu cần
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

# Load API Key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("❌ Lỗi: Chưa có GOOGLE_API_KEY trong .env")
    sys.exit(1)

genai.configure(api_key=API_KEY)

# --- Cấu hình đường dẫn ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Lưu ý: Đảm bảo bạn đã có vector_db cũ ở đây.
# Nếu bạn đã xóa folder vector_db để chạy cái mới thì code này sẽ lỗi.
# Nếu lỗi, bạn cần trỏ đúng đường dẫn hoặc chạy lại script tạo vector cũ.
VECTOR_DIR = os.path.join(BASE_DIR, "data", "vector_db")

# Init Model
embedding = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=API_KEY
)

# Thử load vector db, nếu không có thì báo lỗi
try:
    vector_db = FAISS.load_local(VECTOR_DIR, embedding, allow_dangerous_deserialization=True)
except Exception as e:
    print(f"❌ Không tìm thấy Vector DB cũ tại {VECTOR_DIR}.")
    print("Bạn có thể cần chạy lại script tạo vector cũ (nếu đã lỡ xóa) để test.")
    sys.exit(1)

model = genai.GenerativeModel("gemini-2.0-flash")

def query_vector(query: str, k: int = 5):
    """Vector-only RAG (Hệ thống Cũ)."""
    t0 = time.perf_counter()

    # 1. Chỉ tìm kiếm bằng Vector (Semantic Search)
    results = vector_db.similarity_search(query, k=k)

    context = "\n\n".join(r.page_content for r in results)

    prompt = f"""
Bạn là trợ lý pháp lý am hiểu luật Việt Nam.
Hãy trả lời ngắn gọn, chính xác và có dẫn Điều luật liên quan.

Câu hỏi: {query}

Các đoạn luật tham khảo:
{context}
"""
    # 2. Gọi Gemini trả lời
    resp = model.generate_content(prompt)

    latency = time.perf_counter() - t0
    sources = [r.metadata.get("source", "Unknown") for r in results]
    return resp.text, sources, latency

if __name__ == "__main__":
    print("🤖 CHATBOT CŨ (Vector Search Only)")
    print("----------------------------------")
    while True:
        q = input("\n👤 Nhập câu hỏi (old bot): ")
        if q.lower() in ["exit", "quit"]:
            break

        ans, srcs, t = query_vector(q)
        print("\n🤖 Trả lời:")
        print(ans)
        print("\n📚 Nguồn tham khảo (Top 5 Vector):")
        for s in srcs: print("→", s)
        print(f"\n⚡ Latency: {t:.2f}s")