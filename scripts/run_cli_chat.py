import sys
import os

# Thêm root project vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.graph_rag_service import GraphRAGService

def main():
    print("🚀 Đang khởi tạo Chatbot GraphRAG...")
    try:
        bot = GraphRAGService(vector_db_path="data/vector_db")
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

            answer, meta, latency = bot.query(query)

            print("\n=== TRẢ LỜI ===")
            print(answer)
            print(f"\n⏱️ Thời gian: {latency:.2f}s")
            print(f"📊 Metadata: {len(meta['graph_edges'])} cạnh đồ thị, {len(meta['vector_sources'])} nguồn vector.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Lỗi xử lý: {e}")

    bot.close()
    print("\nTam biệt!")

if __name__ == "__main__":
    main()
