# File: scripts/create_vector_index.py
import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# Import các thư viện AI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rank_bm25 import BM25Okapi

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNK_DIR = os.path.join(BASE_DIR, "data", "chunks")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "data", "artifacts")

# Tạo thư mục artifacts nếu chưa có
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    print("❌ Lỗi: Chưa có GOOGLE_API_KEY trong file .env")
    exit(1)

def tokenize_vn(text):
    """
    Hàm tách từ đơn giản cho tiếng Việt để chạy BM25.
    (Tách theo khoảng trắng và lowercase)
    """
    return text.lower().split()

def main():
    print("🚀 Bắt đầu tạo Index cho Hybrid Search (Vector + BM25)...")
    print(f"   - Embeddings: Google (text-embedding-004)")
    print(f"   - Keyword: BM25Okapi")

    # 1. Đọc dữ liệu từ Chunks
    docs = []   # Lưu nội dung text
    metas = []  # Lưu metadata (tên file, nguồn...)

    if not os.path.exists(CHUNK_DIR):
        print(f"❌ Không tìm thấy thư mục {CHUNK_DIR}. Hãy chạy split_text.py trước.")
        exit(1)

    files = [f for f in os.listdir(CHUNK_DIR) if f.endswith(".json")]
    if not files:
        print("❌ Thư mục chunks rỗng!")
        exit(1)

    print("📦 Đang tải dữ liệu chunks...")
    for filename in tqdm(files):
        path = os.path.join(CHUNK_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            for chunk in chunks:
                # Xử lý tương thích cả format cũ (str) và mới (dict)
                if isinstance(chunk, dict):
                    text = chunk.get("page_content", "")
                    meta = chunk.get("metadata", {})
                    # Nếu metadata chưa có source, lấy từ tên file
                    if "source" not in meta:
                        meta["source"] = filename.replace("_chunks.json", ".pdf")
                else:
                    text = str(chunk)
                    meta = {"source": filename.replace("_chunks.json", ".pdf")}

                if text.strip(): # Chỉ lấy đoạn có nội dung
                    docs.append(text)
                    metas.append(meta)
        except Exception as e:
            print(f"⚠️ Lỗi đọc file {filename}: {e}")

    print(f"✅ Đã tải {len(docs)} đoạn văn bản.")

    # 2. Tạo & Lưu BM25 (Cho Keyword Search)
    print("🔠 Đang tạo chỉ mục BM25...")
    tokenized_docs = [tokenize_vn(doc) for doc in tqdm(docs, desc="Tokenizing")]
    bm25 = BM25Okapi(tokenized_docs)

    with open(os.path.join(ARTIFACTS_DIR, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    print("   -> Đã lưu data/artifacts/bm25.pkl")

    # 3. Lưu Docs & Metas (Quan trọng cho HybridSearcher)
    print("💾 Đang lưu docs.json và metas.json...")
    with open(os.path.join(ARTIFACTS_DIR, "docs.json"), "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)

    with open(os.path.join(ARTIFACTS_DIR, "metas.json"), "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False)

    # 4. Tạo & Lưu FAISS (Cho Semantic Search)
    print("🧠 Đang tạo Vector Index (FAISS)...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=google_api_key
    )

    # Tạo vector store
    vector_db = FAISS.from_texts(docs, embeddings, metadatas=metas)

    # Lưu index FAISS vào artifacts
    vector_db.save_local(ARTIFACTS_DIR, index_name="faiss")
    print(f"   -> Đã lưu FAISS index vào {ARTIFACTS_DIR}")

    print("\n🎉 HOÀN TẤT! Dữ liệu đã sẵn sàng cho Hybrid Search.")

if __name__ == "__main__":
    main()
