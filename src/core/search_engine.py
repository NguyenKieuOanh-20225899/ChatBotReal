import json, faiss, numpy as np, pickle
import sys, os  # <--- Bổ sung import thiếu
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer

try:
    # Trường hợp 1: Chạy từ root (python scripts/chatbot_legal.py)
    from src.utils.text_utils import tokenize_vn, preprocess_text
except ImportError:
    # Trường hợp 2: Chạy trực tiếp (python experiments/run_E2_hybrid.py)
    # Force Python tìm trong thư mục hiện tại
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from text_utils import tokenize_vn, preprocess_text

def rrf_fuse(ranked_lists, weights=None, K=60, topk=10):
    if weights is None:
        weights = [1.0] * len(ranked_lists)
        
    scores = defaultdict(float)
    # Duyệt qua từng danh sách kèm theo trọng số của nó
    for lst, w in zip(ranked_lists, weights):
        for rank, idx in enumerate(lst, start=1):
            # Công thức RRF có trọng số: weight * (1 / (K + rank))
            scores[idx] += w * (1.0 / (K + rank))
    
    sorted_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [i for i, _ in sorted_indices][:topk]

class HybridSearcher:
    def __init__(self, cfg):
        self.cfg = cfg  # <--- Bổ sung: Lưu config để dùng trong hàm search
        arts = Path(cfg["paths"]["artifacts_dir"])
        print("Loading artifacts...")
        self.docs = json.load(open(arts/"docs.json","r",encoding="utf-8"))
        self.metas = json.load(open(arts/"metas.json","r",encoding="utf-8"))
        self.bm25 = pickle.load(open(arts/"bm25.pkl","rb"))
        
        self.faiss = faiss.read_index(str(arts/"faiss.index"))
        # Cài đặt nprobe để tìm kiếm chính xác hơn
        self.faiss.nprobe = cfg["index"].get("faiss_nprobe", 10)
        
        self.emb = SentenceTransformer(cfg["index"]["embedding_model"])

        self.bm25_topk = cfg["retrieval"]["bm25_topk"]
        self.dense_topk = cfg["retrieval"]["dense_topk"]
        self.rrf_K = cfg["retrieval"]["rrf_K"]
        self.final_topk = cfg["retrieval"]["final_topk"]

    def search(self, query, k=None): # <--- Bổ sung: tham số k tùy chọn
        query = preprocess_text(query)
        
        # 1. BM25 Search
        tokenized_q = tokenize_vn(query)
        bm25_scores = self.bm25.get_scores(tokenized_q)
        # Lấy topk index có điểm cao nhất
        bm25_rank = np.argsort(-bm25_scores)[:self.bm25_topk].tolist()

        # 2. Dense Search (FAISS)
        qv = self.emb.encode([query], normalize_embeddings=True)
        D, I = self.faiss.search(qv.astype(np.float32), self.dense_topk)
        dense_rank = I[0].tolist()

        # 3. Fusion
        # Lấy trọng số từ config, mặc định là [1.0, 1.0] nếu không có
        # Config yaml cần thêm dòng: rrf_weights: [2.0, 1.0] (để ưu tiên BM25)
        weights = self.cfg["retrieval"].get("rrf_weights", [1.0, 1.0])
        
        # Nếu user truyền k vào thì dùng k, nếu không thì dùng default từ config
        current_topk = k if k is not None else self.final_topk

        fused_indices = rrf_fuse(
            [bm25_rank, dense_rank], 
            weights=weights, 
            K=self.rrf_K, 
            topk=current_topk
        )
        
        results = []
        for rank, idx in enumerate(fused_indices):
            if idx < len(self.docs): # Check bound an toàn
                results.append({
                    "rank": rank + 1,
                    "doc": self.docs[idx],
                    "meta": self.metas[idx],
                    # Debug info (optional)
                    "bm25_hit": idx in bm25_rank,
                    "dense_hit": idx in dense_rank
                })
        return results