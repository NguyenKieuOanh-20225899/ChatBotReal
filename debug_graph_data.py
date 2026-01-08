import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load cấu hình
load_dotenv()
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))

def debug_neo4j():
    print(f"🕵️‍♂️ Đang soi dữ liệu trong Neo4j tại {URI}...")
    try:
        driver = GraphDatabase.driver(URI, auth=AUTH)
        driver.verify_connectivity()
    except Exception as e:
        print(f"❌ Không kết nối được Neo4j: {e}")
        return

    with driver.session() as sess:
        # 1. Kiểm tra 5 Node đầu tiên xem ID nó trông thế nào
        print("\n=== 1. KIỂM TRA DỮ LIỆU GỐC (Top 5 Nodes) ===")
        res = sess.run("MATCH (n:Article) RETURN n.id, n.topic LIMIT 5")
        nodes = list(res)
        if not nodes:
            print("⚠️ Neo4j TRỐNG RỖNG! Bạn chưa nạp dữ liệu thành công.")
            return

        for record in nodes:
            print(f"   🔹 ID trong DB: '{record['n.id']}' | Topic: '{record['n.topic']}'")

        # 2. Giả lập Chatbot tìm kiếm
        print("\n=== 2. THỬ NGHIỆM TÌM KIẾM CỦA CHATBOT ===")
        # Chatbot thường tìm chuỗi này:
        test_ids = ["Điều 81", "Điều 82", "81", "82"]

        print(f"❓ Chatbot đang thử tìm các ID: {test_ids}")

        query = """
        MATCH (a:Article)-[r]-(b:Article)
        WHERE a.id IN $ids OR a.topic CONTAINS 'nuôi con'
        RETURN a.id, type(r), b.id
        LIMIT 5
        """
        res_search = sess.run(query, ids=test_ids)
        edges = list(res_search)

        if len(edges) == 0:
            print("❌ KẾT QUẢ: Không tìm thấy gì! -> Đây là lý do Chatbot báo 0 cạnh.")
            print("👉 Gợi ý: ID trong DB và ID chatbot tìm không khớp nhau.")
        else:
            print(f"✅ KẾT QUẢ: Tìm thấy {len(edges)} cạnh.")
            for e in edges:
                print(f"   🔗 {e['a.id']} --[{e['type(r)']}]--> {e['b.id']}")

    driver.close()

if __name__ == "__main__":
    debug_neo4j()
