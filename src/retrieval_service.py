import sys
import os
import yaml

# Thêm đường dẫn để import được các file trong folder experiments
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.search_pipeline import HybridSearcher
from experiments.rerank import CrossEncoderReranker

class LegalRetriever:
    def __init__(self, config_path="experiments/config.yaml"):
        print("🔄 Đang khởi động hệ thống tìm kiếm...")
        self.cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))

        # 1. Load bộ tìm kiếm Hybrid (BM25 + FAISS)
        self.searcher = HybridSearcher(self.cfg)

        # 2. Load bộ Reranker
        self.reranker = CrossEncoderReranker(self.cfg["reranker"]["model_name"])
        self.keep_topk = self.cfg["reranker"]["keep_topk"]

        print("✅ Hệ thống tìm kiếm đã sẵn sàng!")

    def retrieve(self, query: str):
        """
        Input: Câu hỏi người dùng
        Output: List các đoạn văn bản (text) phù hợp nhất
        """
        # Bước 1: Tìm kiếm sơ bộ (Lấy khoảng 50-100 kết quả)
        candidates = self.searcher.search(query)

        # Bước 2: Sắp xếp lại (Rerank) để chọn ra top k tốt nhất
        reranked_results, _ = self.reranker.rerank(query, candidates, keep_topk=self.keep_topk)

        # Bước 3: Trích xuất text để đưa vào LLM
        # (Bạn có thể lấy thêm meta nếu cần trích dẫn nguồn)
        context_list = []
        for item in reranked_results:
            doc_text = item["doc"]
            source = item["meta"].get("source_file", "Unknown")
            # Format: [Nguồn] Nội dung
            context_list.append(f"[{source}]: {doc_text}")

        return context_list

# Test nhanh nếu chạy trực tiếp file này
if __name__ == "__main__":
    bot = LegalRetriever()
    results = bot.retrieve("Điều 31 quy định gì?")
    for r in results:
        print("-" * 20)
        print(r)