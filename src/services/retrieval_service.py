import os
import yaml
from typing import List, Dict
from src.core.search_engine import HybridSearcher
from src.core.reranker import CrossEncoderReranker

class LegalRetriever:
    def __init__(self, config_path: str = "config/config.yaml"):
        print("🔄 Đang khởi động hệ thống tìm kiếm (LegalRetriever)...")

        self.config_path = os.path.abspath(config_path)
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Không tìm thấy config tại: {self.config_path}")

        self.cfg = yaml.safe_load(open(self.config_path, "r", encoding="utf-8"))

        # 1. Load Searcher
        self.searcher = HybridSearcher(self.cfg)

        # 2. Load Reranker
        rerank_cfg = self.cfg.get("reranker", {})
        self.reranker = CrossEncoderReranker(rerank_cfg.get("model_name", "BAAI/bge-reranker-v2-m3"))
        self.keep_topk = rerank_cfg.get("keep_topk", 5)

        print("✅ LegalRetriever đã sẵn sàng!")

    def retrieve(self, query: str) -> List[str]:
        candidates = self.searcher.search(query)

        if self.cfg.get("reranker", {}).get("apply", False):
            reranked_results, _ = self.reranker.rerank(query, candidates, keep_topk=self.keep_topk)
        else:
            reranked_results = candidates[:self.keep_topk]

        context_list = []
        for item in reranked_results:
            doc_text = item.get("doc", "")
            source = item.get("meta", {}).get("source_file", "Unknown")
            context_list.append(f"[{source}]: {doc_text}")

        return context_list
