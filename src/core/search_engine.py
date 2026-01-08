import json, faiss, numpy as np, pickle
import sys, os
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

# --- Thay đổi quan trọng: Dùng thư viện của Google thay vì SentenceTransformer ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings

try:
    from src.utils.text_utils import tokenize_vn, preprocess_text
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from text_utils import tokenize_vn, preprocess_text

def rrf_fuse(ranked_lists, weights=None, K=60, topk=10):
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores = defaultdict(float)
    for lst, w in zip(ranked_lists, weights):
        for rank, idx in enumerate(lst, start=1):
            scores[idx] += w * (1.0 / (K + rank))

    sorted_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [i for i, _ in sorted_indices][:topk]

class HybridSearcher:
    def __init__(self, cfg):
        self.cfg = cfg
        load_dotenv() # Load biến môi trường để lấy API Key

        arts = Path(cfg["paths"]["artifacts_dir"])
        print("Loading artifacts...")

        # Load metadata
        self.docs = json.load(open(arts/"docs.json","r",encoding="utf-8"))
        self.metas = json.load(open(arts/"metas.json","r",encoding="utf-8"))
        self.bm25 = pickle.load(open(arts/"bm25.pkl","rb"))

        # Load FAISS
        # Đổi thành:
        self.faiss = faiss.read_index(str(arts/"faiss.faiss"))
        self.faiss.nprobe = cfg["index"].get("faiss_nprobe", 10)

        # --- FIX LỖI Ở ĐÂY ---
        # Thay vì dùng SentenceTransformer, ta khởi tạo Google Embeddings
        # để khớp với file create_vector_index.py
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️ Cảnh báo: Không tìm thấy GOOGLE_API_KEY. Vector Search sẽ lỗi.")

        self.emb = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", # Model bạn dùng để tạo index
            google_api_key=api_key
        )
        print("✅ Đã load Google Embeddings (text-embedding-004)")

        self.bm25_topk = cfg["retrieval"]["bm25_topk"]
        self.dense_topk = cfg["retrieval"]["dense_topk"]
        self.rrf_K = cfg["retrieval"]["rrf_K"]
        self.final_topk = cfg["retrieval"]["final_topk"]

    def search(self, query, k=None, mode="hybrid"):
        """
        mode: 'hybrid', 'vector_only', 'bm25_only'
        """
        query = preprocess_text(query)
        current_topk = k if k is not None else self.final_topk

        # 1. BM25 Search
        bm25_rank = []
        if mode in ["hybrid", "bm25_only"]:
            tokenized_q = tokenize_vn(query)
            bm25_scores = self.bm25.get_scores(tokenized_q)
            bm25_rank = np.argsort(-bm25_scores)[:self.bm25_topk].tolist()

            if mode == "bm25_only":
                return self._format_results(bm25_rank, current_topk)

        # 2. Dense Search (FAISS)
        dense_rank = []
        if mode in ["hybrid", "vector_only"]:
            try:
                # --- SỬA LẠI CÁCH GỌI EMBEDDING ---
                # Google trả về list float, cần convert sang numpy array (1, 768)
                vector_embedding = self.emb.embed_query(query)
                qv = np.array([vector_embedding], dtype=np.float32)

                D, I = self.faiss.search(qv, self.dense_topk)
                dense_rank = I[0].tolist()
            except Exception as e:
                print(f"❌ Lỗi Vector Search: {e}")
                dense_rank = []

            if mode == "vector_only":
                return self._format_results(dense_rank, current_topk)

        # 3. Fusion (Hybrid)
        weights = self.cfg["retrieval"].get("rrf_weights", [1.0, 1.0])
        fused_indices = rrf_fuse(
            [bm25_rank, dense_rank],
            weights=weights,
            K=self.rrf_K,
            topk=current_topk
        )

        # Format kết quả trả về
        results = []
        for rank, idx in enumerate(fused_indices):
            if idx < len(self.docs) and idx != -1:
                results.append({
                    "rank": rank + 1,
                    "doc": self.docs[idx],
                    "meta": self.metas[idx],
                    "bm25_hit": idx in bm25_rank,
                    "dense_hit": idx in dense_rank
                })
        return results

    def _format_results(self, indices, k):
        results = []
        for rank, idx in enumerate(indices):
            if idx < len(self.docs) and idx != -1:
                results.append({
                    "rank": rank + 1,
                    "doc": self.docs[idx],
                    "meta": self.metas[idx]
                })
        return results[:k]
