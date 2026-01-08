# File: scripts/compare_search_methods.py
import sys
import os
import json
import pandas as pd
import yaml
from tqdm import tqdm
from dotenv import load_dotenv

# Thêm đường dẫn root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import các service
from src.core.search_engine import HybridSearcher
try:
    from src.services.graph_rag_service import GraphRAGService
except ImportError:
    GraphRAGService = None

load_dotenv()

def load_config():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_comparison():
    print("🚀 BẮT ĐẦU SO SÁNH 4 PHƯƠNG PHÁP TÌM KIẾM (FIXED)")
    print("="*60)

    cfg = load_config()

    # 1. Khởi tạo Searcher
    print("📦 Đang khởi tạo HybridSearcher...")
    try:
        searcher = HybridSearcher(cfg)
    except Exception as e:
        print(f"❌ Lỗi khởi tạo HybridSearcher: {e}")
        return

    print("🕸️ Đang khởi tạo GraphRAGService...")
    graph_service = GraphRAGService() if GraphRAGService else None

    # 2. Load bộ câu hỏi
    test_file = "data/test_set_essay.json"
    if not os.path.exists(test_file):
        test_file = "data/test_set_mcq.json"

    if not os.path.exists(test_file):
        print("❌ Không có dữ liệu test.")
        return

    with open(test_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # test_cases = test_cases[:10] # Bỏ comment nếu muốn test nhanh 10 câu

    results = []

    print(f"\n⚡ Đang xử lý {len(test_cases)} câu hỏi...")

    for idx, case in tqdm(enumerate(test_cases), total=len(test_cases)):
        q = case["question"]
        gt = case.get("ground_truth", "")

        row = {
            "STT": idx + 1,
            "Câu hỏi": q,
            "Đáp án chuẩn": gt
        }

        # --- Hàm lấy nội dung an toàn ---
        def get_result_safe(docs):
            if not docs:
                return "Không tìm thấy"
            # Lấy metadata an toàn bằng .get()
            first_doc = docs[0]
            if isinstance(first_doc, dict):
                content = first_doc.get("doc", str(first_doc))
                meta = first_doc.get("metadata", {})
                src = meta.get("source", "Unknown")
            else:
                # Trường hợp trả về object (Langchain Document)
                content = getattr(first_doc, "page_content", str(first_doc))
                meta = getattr(first_doc, "metadata", {})
                src = meta.get("source", "Unknown")

            return f"[{src}]\n{content[:300]}..."

        # --- 1: BM25 (Keyword) ---
        try:
            docs = searcher.search(q, k=1, mode="bm25_only")
            row["BM25 Result"] = get_result_safe(docs)
        except Exception as e:
            row["BM25 Result"] = f"Lỗi: {e}"

        # --- 2: VECTOR (Semantic) ---
        try:
            docs = searcher.search(q, k=1, mode="vector_only")
            row["Vector Result"] = get_result_safe(docs)
        except Exception as e:
            row["Vector Result"] = f"Lỗi: {e}"

        # --- 3: HYBRID (Kết hợp) ---
        try:
            docs = searcher.search(q, k=1, mode="hybrid")
            row["Hybrid Result"] = get_result_safe(docs)
        except Exception as e:
            row["Hybrid Result"] = f"Lỗi: {e}"

        # --- 4: GRAPH RAG ---
        if graph_service:
            try:
                ans, meta, _ = graph_service.query(q)
                n_graph = meta.get("graph_edges_used", 0)
                n_vec = len(meta.get("vector_sources", []))
                row["GraphRAG Answer"] = ans
                row["Graph Info"] = f"Graph:{n_graph} + Vector:{n_vec}"
            except Exception as e:
                row["GraphRAG Answer"] = f"Lỗi: {e}"
        else:
            row["GraphRAG Answer"] = "Off"

        results.append(row)

    # 3. Xuất file Excel
    output_file = "bang_so_sanh_chi_tiet.xlsx"
    df = pd.DataFrame(results)

    # Sắp xếp cột
    cols = ["STT", "Câu hỏi", "BM25 Result", "Vector Result", "Hybrid Result", "GraphRAG Answer", "Graph Info"]
    final_cols = [c for c in cols if c in df.columns]
    df = df[final_cols]

    df.to_excel(output_file, index=False)
    print(f"\n✅ ĐÃ XONG! File: {output_file}")

if __name__ == "__main__":
    run_comparison()
