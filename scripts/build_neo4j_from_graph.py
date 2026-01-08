import json
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# 1. Cấu hình kết nối
load_dotenv()
# Nếu chạy trên Mac/Docker, đôi khi localhost cần đổi, nhưng mặc định cứ để localhost
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USERNAME", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password") # Password bạn set trong lệnh docker
JSON_PATH = "data/knowledge_graph.json" # Đường dẫn file dữ liệu

def build_graph():
    print(f"🔌 Đang kết nối tới Neo4j tại {URI}...")
    try:
        driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
        driver.verify_connectivity()
        print("✅ Kết nối thành công!")
    except Exception as e:
        print(f"❌ Lỗi kết nối: {e}")
        return

    # Kiểm tra file dữ liệu
    if not os.path.exists(JSON_PATH):
        print(f"❌ Lỗi: Không tìm thấy file dữ liệu tại {JSON_PATH}")
        return

    print("📖 Đang đọc file JSON...")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    # Xử lý trường hợp key tên là "relationships" hoặc "edges"
    edges = data.get("relationships", [])
    if not edges and "edges" in data:
        edges = data["edges"]

    print(f"📦 Tìm thấy {len(nodes)} node và {len(edges)} cạnh.")

    with driver.session() as session:
        # 1. Xóa dữ liệu cũ (Reset DB)
        print("🧹 Đang xóa sạch dữ liệu cũ trong Neo4j...")
        session.run("MATCH (n) DETACH DELETE n")

        # 2. Tạo chỉ mục (Index) để tìm nhanh hơn
        print("⚡ Đang tạo Index cho Article ID...")
        try:
            session.run("CREATE CONSTRAINT FOR (a:Article) REQUIRE a.id IS UNIQUE")
        except:
            pass # Bỏ qua nếu đã có

        # 3. Nạp Nodes (Dùng Batch để nạp nhanh)
        print("🚀 Đang nạp Nodes...")
        query_node = """
        UNWIND $batch AS row
        MERGE (a:Article {id: row.id})
        SET a.topic = row.topic,
            a.content = row.content,
            a.source = row.source
        """
        batch_size = 500
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i+batch_size]
            session.run(query_node, batch=batch)
            print(f"   - Đã nạp {min(i+batch_size, len(nodes))}/{len(nodes)} nodes")

        # 4. Nạp Edges
        print("🔗 Đang nạp Relationships...")
        query_edge = """
        UNWIND $batch AS row
        MATCH (source:Article {id: row.source})
        MATCH (target:Article {id: row.target})
        MERGE (source)-[r:RELATED {relation: row.relation}]->(target)
        """
        for i in range(0, len(edges), batch_size):
            batch = edges[i:i+batch_size]
            session.run(query_edge, batch=batch)
            print(f"   - Đã nạp {min(i+batch_size, len(edges))}/{len(edges)} edges")

    driver.close()
    print("✅ HOÀN TẤT! Dữ liệu đã vào Neo4j.")

if __name__ == "__main__":
    build_graph()
