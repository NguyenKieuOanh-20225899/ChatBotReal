# File: scripts/run_cli_chat.py
import sys
import os

# Thêm root project vào sys.path để import được src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.graph_rag_service import GraphRAGService

def main():
    print("🚀 Đang khởi tạo Chatbot GraphRAG (In-memory)...")
    try:
        # Đường dẫn mặc định trỏ tới artifacts và json graph
        bot = GraphRAGService(
            vector_db_path="data/artifacts",
            graph_path="data/knowledge_graph.json"
        )
    except Exception as e:
        print(f"❌ Lỗi khởi tạo: {e}")
        return

    print("✅ Sẵn sàng! Nhập 'exit' để thoát.")
    while True:
        try:
            query = input("\n❓ Nhập câu hỏi pháp luật: ").strip()
            if query.lower() in ["exit", "quit", "thoát"]:
                break
            if not query:
                continue

            # Gọi hàm query
            answer, meta, latency = bot.query(query)

            print("\n=== TRẢ LỜI ===")
            print(answer)
            print(f"\n⏱️ Thời gian: {latency:.2f}s")

            # --- FIX LỖI Ở ĐÂY ---
            # Code cũ: len(meta['graph_edges']) -> Gây lỗi vì key này không còn
            # Code mới: Dùng .get() để lấy giá trị an toàn
            n_graph = meta.get('graph_edges_used', 0)
            n_vector = len(meta.get('vector_sources', []))

            print(f"📊 Metadata: Sử dụng {n_graph} thông tin từ Graph, {n_vector} nguồn từ Vector.")

            # In chi tiết nguồn (Optional)
            if n_vector > 0:
                sources = list(set(meta.get('vector_sources', [])))
                print(f"   (Nguồn: {', '.join(sources[:3])}...)")

        except KeyboardInterrupt:
            print("\nĐã dừng chương trình.")
            break
        except Exception as e:
            print(f"❌ Lỗi xử lý: {e}")

    bot.close()
    print("\nTạm biệt!")

if __name__ == "__main__":
    main()
