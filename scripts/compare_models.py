import sys
import os
import textwrap

# Add root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.retrieval_service import HybridRAGService
from src.services.graph_rag_service import GraphRAGService

def print_box(title, content, color_code="\033[94m"):
    print(f"{color_code}┌{'─'*60}┐\033[0m")
    print(f"{color_code}│ {title.center(58)} │\033[0m")
    print(f"{color_code}├{'─'*60}┤\033[0m")
    lines = textwrap.wrap(content, width=58)
    for line in lines:
        print(f"{color_code}│ {line.ljust(58)} │\033[0m")
    print(f"{color_code}└{'─'*60}┘\033[0m")

def main():
    print("🚀 Đang khởi tạo các mô hình để so sánh...")
    try:
        hybrid_bot = HybridRAGService()
        graph_bot = GraphRAGService()
    except Exception as e:
        print(f"❌ Lỗi khởi tạo: {e}")
        return

    print("\n✅ Sẵn sàng so sánh! Nhập 'exit' để thoát.")

    while True:
        query = input("\n⚖️  Nhập câu hỏi so sánh: ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        if not query:
            continue

        print("\n" + "="*80)

        # --- CHẠY HYBRID MODEL ---
        print("1️⃣  Đang chạy Hybrid RAG (BM25 + Vector)...")
        ans_h, meta_h, time_h = hybrid_bot.query(query)

        # --- CHẠY GRAPH RAG MODEL ---
        print("2️⃣  Đang chạy Graph RAG (Vector + Knowledge Graph)...")
        ans_g, meta_g, time_g = graph_bot.query(query)

        # --- HIỂN THỊ KẾT QUẢ ---
        print("\n" + "⚔️  KẾT QUẢ SO SÁNH ⚔️".center(80))

        print(f"\n⏱️  Thời gian xử lý:")
        print(f"   - Hybrid: {time_h:.2f}s")
        print(f"   - Graph : {time_g:.2f}s")

        print_box("MODEL 1: HYBRID RAG", ans_h, "\033[96m") # Cyan
        print_box("MODEL 2: GRAPH RAG", ans_g, "\033[92m") # Green

        # So sánh Metadata
        print("\n🔍 Phân tích:")
        print(f"   - Hybrid tìm thấy: {meta_h['source_count']} đoạn văn bản.")
        print(f"   - Graph tìm thấy : {len(meta_g.get('graph_edges', []))} cạnh đồ thị + {len(meta_g.get('vector_sources', []))} đoạn văn bản.")

if __name__ == "__main__":
    main()
