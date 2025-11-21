import json
import os
import sys
from tqdm import tqdm
from dotenv import load_dotenv
# Import các thư viện cũ bạn dùng
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Import hàm tính điểm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.eval_metrics import recall_at_k, mrr, ndcg_at_k

# 1. Cấu hình
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DIR = os.path.join(BASE_DIR, "data", "vector_db")
DEVSET_PATH = os.path.join(BASE_DIR, "experiments", "devset", "dev_100.jsonl")

def normalize_id(doc_id):
    """
    Chuẩn hóa ID siêu mạnh: Cắt bỏ mọi hậu tố rườm rà để lấy tên văn bản gốc.
    Ví dụ input:
      - "VanBanGoc_52.2014.QH13_knowledge_Điều 103"
      - "VanBanGoc_52.2014.QH13_chunks#12"
      - "117:2024:NĐ-CP_clean.txt"
    Output chung: "VanBanGoc_52.2014.QH13" hoặc "117_2024_NĐ-CP"
    """
    if not doc_id: return ""
    s = str(doc_id)

    # 1. Lấy tên file (bỏ đường dẫn thư mục)
    s = os.path.basename(s)

    # 2. Thay thế các ký tự đặc biệt gây lỗi
    s = s.replace(":", "_") # Sửa lỗi 117:2024

    # 3. Bỏ đuôi file
    for ext in [".json", ".txt", "_clean"]:
        s = s.replace(ext, "")

    # 4. CẮT BỎ HẬU TỐ (Quan trọng nhất)
    # Cắt ngay khi gặp các từ khóa này
    keywords_to_cut = ["_knowledge", "_chunks", "_Điều", ":Điều"]
    for kw in keywords_to_cut:
        if kw in s:
            s = s.split(kw)[0]

    return s.strip()

def main():
    print(f"🔄 Đang tải Vector DB cũ từ: {VECTOR_DIR}")
    try:
        embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)
        vector_db = FAISS.load_local(VECTOR_DIR, embedding, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"❌ Lỗi: Không tìm thấy hoặc không load được Vector DB cũ.\n{e}")
        return

    print(f"📖 Đang đọc bộ dữ liệu kiểm tra...")
    dev_data = [json.loads(l) for l in open(DEVSET_PATH, "r", encoding="utf-8")]

    R10, MRR, N10 = [], [], []

    print("🚀 Bắt đầu đánh giá...")
    debug_count = 0

    for ex in tqdm(dev_data):
        query = ex["question"]
        gold_ids = [normalize_id(g) for g in ex["gold"]]

        # Tìm kiếm top 10
        docs = vector_db.similarity_search(query, k=10)

        # Lấy ID từ metadata (ưu tiên 'source' hoặc 'file_name')
        raw_retrieved = [d.metadata.get("source") or d.metadata.get("file_name") or "NO_SOURCE" for d in docs]
        retrieved_ids = [normalize_id(r) for r in raw_retrieved]

        # --- Debug lại để chắc chắn ID đã khớp ---
        if debug_count < 3:
            print(f"\n--- DEBUG Query {debug_count+1} ---")
            print(f"Câu hỏi: {query}")
            print(f"Gold (Chuẩn):   {gold_ids}")
            print(f"Retr (Tìm đc):  {retrieved_ids}")
            debug_count += 1
        # ---------------------------------------

        R10.append(recall_at_k(retrieved_ids, gold_ids, k=10))
        MRR.append(mrr(retrieved_ids, gold_ids))
        N10.append(ndcg_at_k(retrieved_ids, gold_ids, k=10))

    print("\n📊 === KẾT QUẢ ĐÁNH GIÁ CHATBOT CŨ (VECTOR ONLY) ===")
    print(f"Recall@10 = {sum(R10)/len(R10):.3f}")
    print(f"MRR       = {sum(MRR)/len(MRR):.3f}")
    print(f"nDCG@10   = {sum(N10)/len(N10):.3f}")

if __name__ == "__main__":
    main()