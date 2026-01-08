# File: src/services/graph_rag_service.py
import os
import json
import time
from typing import Tuple, List, Dict

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

class GraphRAGService:
    def __init__(self, vector_db_path: str = "data/artifacts", graph_path: str = "data/knowledge_graph.json"):
        load_dotenv()

        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")

        if not self.groq_api_key:
            raise ValueError("❌ Thiếu GROQ_API_KEY trong file .env")

        # 1. KHỞI TẠO LLM
        print("⚡ Đang kết nối tới Groq (Llama-3.1-8b-instant)...")
        self.llm = ChatGroq(
            temperature=0.1,
            model_name="llama-3.1-8b-instant",
            api_key=self.groq_api_key,
            max_retries=2
        )

        # 2. LOAD VECTOR DB (Sửa lỗi quan trọng ở đây)
        print(f"📦 Loading Vector Database từ: {vector_db_path}")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=self.google_api_key
        )
        try:
            # LƯU Ý: Thêm index_name="faiss" để khớp với file faiss.faiss đã tạo
            self.vector_db = FAISS.load_local(
                vector_db_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
                index_name="faiss"  # <--- QUAN TRỌNG: Phải khớp với lúc save
            )
            print("✅ Vector DB loaded thành công.")
        except Exception as e:
            print(f"⚠️ Không load được Vector DB: {e}")
            print("👉 Gợi ý: Hãy chạy 'python scripts/run_pipeline.py' để tạo dữ liệu trước.")
            self.vector_db = None

        # 3. LOAD KNOWLEDGE GRAPH
        print("🕸️ Loading Knowledge Graph...")
        self.graph_nodes = {}
        self.graph_edges = []
        try:
            with open(graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for node in data.get("nodes", []):
                    self.graph_nodes[node["id"]] = node
                self.graph_edges = data.get("edges", [])
            print(f"✅ Graph loaded: {len(self.graph_nodes)} nodes, {len(self.graph_edges)} edges.")
        except Exception as e:
            print(f"⚠️ Không load được Graph JSON: {e}")

    def _find_related_nodes(self, initial_nodes: List[str]) -> List[Dict]:
        """Tìm các node liên quan (bước nhảy 1)"""
        related_info = []
        for edge in self.graph_edges:
            source = edge["from"]
            target = edge["to"]
            relation = edge["relation"]

            if source in initial_nodes:
                target_node = self.graph_nodes.get(target)
                if target_node:
                    topic = target_node.get("topic", "")
                    # Lấy thêm nguồn nếu có
                    src_doc = target_node.get("sources", [])
                    src_str = f" (Nguồn: {src_doc[0]})" if src_doc else ""
                    related_info.append(f"- {source} {relation} {target}: {topic}{src_str}")

        return related_info[:10]

    def query(self, query_text: str, k: int = 4) -> Tuple[str, dict, float]:
        t0 = time.perf_counter()

        # BƯỚC 1: VECTOR SEARCH
        context_parts = []
        found_articles = set()
        vec_sources = []

        if self.vector_db:
            hits = self.vector_db.similarity_search(query_text, k=k)
            for h in hits:
                content = h.page_content
                context_parts.append(content)
                vec_sources.append(h.metadata.get("source", "Unknown"))

                # Tìm ID điều luật trong nội dung tìm được
                for node_id in self.graph_nodes:
                    # Tìm đơn giản: nếu "Điều 5" có trong text
                    if node_id in content:
                        found_articles.add(node_id)

        # BƯỚC 2: GRAPH SEARCH
        graph_context = []
        if found_articles:
            graph_context = self._find_related_nodes(list(found_articles))

        # BƯỚC 3: TẠO PROMPT
        vector_str = "\n\n".join(context_parts)
        graph_str = "\n".join(graph_context) if graph_context else "Không tìm thấy mối liên hệ mở rộng."

        prompt = f"""
Bạn là Trợ lý Luật sư AI. Trả lời câu hỏi dựa trên thông tin sau:

[THÔNG TIN VĂN BẢN - VECTOR]:
{vector_str}

[LIÊN KẾT PHÁP LÝ - GRAPH]:
{graph_str}

[CÂU HỎI]: {query_text}

YÊU CẦU:
1. Trả lời ngắn gọn, chính xác.
2. Trích dẫn điều luật (Ví dụ: Theo Điều 5...).
3. Nếu Graph cung cấp thông tin liên quan, hãy bổ sung.

TRẢ LỜI:
"""
        try:
            response = self.llm.invoke(prompt)
            answer = response.content
        except Exception as e:
            answer = f"Lỗi AI: {e}"

        latency = time.perf_counter() - t0

        meta = {
            "vector_sources": vec_sources,
            "graph_edges_used": len(graph_context)
        }

        return answer, meta, latency

    def close(self):
        pass
