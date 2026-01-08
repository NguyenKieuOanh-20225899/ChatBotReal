# File: scripts/build_knowledge_graph.py
import os
import json
import re
import time
from glob import glob
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load môi trường
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("❌ LỖI: Chưa có GROQ_API_KEY trong file .env")
    exit(1)

# --- CẬP NHẬT MODEL MỚI TẠI ĐÂY ---
# Model cũ 'llama3-70b-8192' đã bị xóa.
# Dùng 'llama-3.3-70b-versatile' (Mạnh nhất) hoặc 'llama-3.1-8b-instant' (Nhanh nhất)
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
    max_retries=3
)

CHUNKS_DIR = "data/chunks"
OUTPUT_FILE = "data/knowledge_graph.json"

def extract_article_id(text):
    """
    Lấy ID: 'Điều 5', 'Điều 13a' từ văn bản.
    """
    match = re.search(r"^(Điều \d+[a-z]*)\b", text, re.IGNORECASE)
    if match:
        raw_id = match.group(1)
        # Chuẩn hóa: "điều 5a" -> "Điều 5a"
        return raw_id.capitalize().replace("điều", "Điều")
    return None

def get_ai_summary(text, retry_count=0):
    """
    Dùng Groq để tóm tắt nội dung điều luật.
    Có cơ chế thử lại thủ công nếu gặp lỗi Rate Limit.
    """
    try:
        prompt = f"""
        Nhiệm vụ: Tóm tắt nội dung chính của văn bản luật dưới đây thành 1 cụm danh từ ngắn gọn (dưới 15 từ).
        Không dùng dấu ngoặc kép. Không giải thích dài dòng.

        Văn bản:
        {text[:800]}

        Tóm tắt:
        """
        response = llm.invoke(prompt)
        return response.content.strip().replace('"', '').replace("Tóm tắt:", "").strip()
    except Exception as e:
        error_msg = str(e)
        # Nếu lỗi do Rate Limit (429), thử đợi và gọi lại
        if "429" in error_msg or "Rate limit" in error_msg:
            if retry_count < 3:
                wait_time = (retry_count + 1) * 5 # Đợi 5s, 10s, 15s
                print(f"⚠️ Quá tải API (Rate Limit), đang đợi {wait_time}s để thử lại...")
                time.sleep(wait_time)
                return get_ai_summary(text, retry_count + 1)

        print(f"❌ Lỗi Groq khi tóm tắt: {error_msg}")
        # Lấy dòng đầu tiên làm fallback
        lines = text.split('\n')
        fallback = lines[0][:50] + "..." if lines else "Nội dung điều luật (Lỗi AI)"
        return fallback

def build_graph():
    nodes = {}
    edges = []

    files = glob(os.path.join(CHUNKS_DIR, "*.json"))
    print(f"🏗️  Đang xây dựng Knowledge Graph từ {len(files)} file...")
    print("⚡ Đang sử dụng Groq API (Llama 3.3) để trích xuất Topic...")

    request_count = 0

    for filepath in tqdm(files):
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        for chunk in chunks:
            # 1. Lấy nội dung và metadata
            if isinstance(chunk, dict):
                content = chunk.get("page_content", "")
                meta = chunk.get("metadata", {})
                source = meta.get("source", os.path.basename(filepath))
            else:
                content = str(chunk)
                source = os.path.basename(filepath)

            # 2. Xác định ID Node (Điều luật)
            node_id = extract_article_id(content)
            if not node_id:
                continue

            # 3. Tạo Node hoặc Cập nhật Node
            should_update_topic = False

            if node_id not in nodes:
                nodes[node_id] = {
                    "id": node_id,
                    "topic": "",
                    "type": "Article",
                    "sources": [source]
                }
                should_update_topic = True
            else:
                if nodes[node_id].get("topic") == "Đang cập nhật":
                    should_update_topic = True
                if source not in nodes[node_id]["sources"]:
                    nodes[node_id]["sources"].append(source)

            # 4. Gọi AI Update Topic (Nếu cần)
            if should_update_topic:
                topic = get_ai_summary(content)
                nodes[node_id]["topic"] = topic

                # Rate Limit thủ công
                request_count += 1
                if request_count % 10 == 0:
                    time.sleep(2)

            # 5. Tạo Edges
            refs = re.findall(r"Điều (\d+[a-z]*)", content, re.IGNORECASE)
            for r in refs:
                target_id = f"Điều {r}"
                if target_id.lower() != node_id.lower():
                    edge = {
                        "from": node_id,
                        "to": target_id,
                        "relation": "dẫn chiếu đến"
                    }
                    if edge not in edges:
                        edges.append(edge)

                    if target_id not in nodes:
                        nodes[target_id] = {
                            "id": target_id,
                            "topic": "Đang cập nhật",
                            "type": "Article",
                            "sources": []
                        }

    # Lưu kết quả
    graph_data = {"nodes": list(nodes.values()), "edges": edges}
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Hoàn tất! Đã lưu tại {OUTPUT_FILE}")
    print(f"   - Nodes: {len(nodes)}")
    print(f"   - Edges: {len(edges)}")

if __name__ == "__main__":
    build_graph()
