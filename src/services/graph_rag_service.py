import os
import re
import time
from typing import Tuple, Any

# Bỏ qua các cảnh báo thư viện cũ
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from neo4j import GraphDatabase
from dotenv import load_dotenv

class GraphRAGService:
    def __init__(self, vector_db_path: str = "data/vector_db"):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("API Key not found in environment variables.")

        genai.configure(api_key=self.api_key)

        # Init Neo4j
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_pass = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_pass))

        # --- CẤU HÌNH MODEL ---
        # SỬ DỤNG 'gemini-flash-latest' (Alias của 1.5 Flash ổn định)
        self.model_name = "gemini-flash-latest"
        self.model = genai.GenerativeModel(self.model_name)

        print(f"🤖 Đang sử dụng model: {self.model_name} (Bản ổn định)")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=self.api_key
        )

        # Init Vector DB
        try:
            self.vector_db = FAISS.load_local(vector_db_path, self.embeddings, allow_dangerous_deserialization=True)
            print("✅ Vector DB loaded.")
        except Exception as e:
            print(f"⚠️ Warning: Could not load Vector DB at {vector_db_path}. Error: {e}")
            self.vector_db = None

    def close(self):
        if self.driver:
            self.driver.close()

    def query(self, query_text: str, k: int = 3) -> Tuple[str, dict, float]:
        t0 = time.perf_counter()

        # 1. Vector Search
        ctx_vec = ""
        vec_sources = []
        article_ids = []

        if self.vector_db:
            # Lấy top 3 chunk để tiết kiệm token
            hits = self.vector_db.similarity_search(query_text, k=k)
            ctx_vec = "\n\n".join(h.page_content for h in hits)
            vec_sources = [h.metadata.get("source") for h in hits]

            # --- ĐOẠN CODE ĐÃ SỬA: Chuẩn hóa ID để khớp với Neo4j ---
            # Tìm tất cả biến thể (ví dụ: "điều 81", "Điều  81", "đIềU 81")
            found_raw = re.findall(r"Điều\s+\d+", ctx_vec, flags=re.IGNORECASE)

            normalized_ids = set()
            for item in found_raw:
                # Lấy số ra (Ví dụ: "điều  81" -> lấy số "81")
                num_match = re.search(r"\d+", item)
                if num_match:
                    num = num_match.group()
                    # Ép về định dạng chuẩn cứng: "Điều" + cách + số
                    normalized_ids.add(f"Điều {num}")

            article_ids = list(normalized_ids)[:10]
            print(f"🔍 IDs tìm thấy (đã chuẩn hóa): {article_ids}")
            # -----------------------------------------------------

        # 2. Extract Concepts (LLM) - Có Retry
        concepts = []
        try:
            extract_prompt = f"Từ câu hỏi sau, liệt kê tối đa 5 khái niệm pháp lý cốt lõi (mỗi dòng 1 mục, không giải thích):\n{query_text}"
            extract_resp = self._call_llm_with_retry(extract_prompt)
            if extract_resp:
                concepts = [x.strip("-• \n") for x in extract_resp.splitlines() if x.strip()][:5]
        except Exception as e:
            print(f"⚠️ Bỏ qua bước trích xuất concept do lỗi: {e}")

        # 3. Graph Search
        edges = self._query_neo4j(article_ids, concepts)

        ctx_graph = "\n".join(f"{e['from_id']} {e['rel']} {e['to_id']} ({e.get('topic','')})" for e in edges)
        if not ctx_graph:
            ctx_graph = "Không có thông tin từ đồ thị."

        # 4. Generate Answer - Có Retry
        prompt = f"""
Bạn là trợ lý pháp lý Việt Nam. Dựa vào ngữ cảnh dưới đây, trả lời chính xác, có dẫn Điều/khoản nếu có.

[Câu hỏi]
{query_text}

[Đoạn văn pháp luật (Vector)]
{ctx_vec}

[Quan hệ pháp lý (Graph)]
{ctx_graph}
"""
        response_text = "Xin lỗi, không thể tạo câu trả lời lúc này."
        try:
            response_text = self._call_llm_with_retry(prompt)
        except Exception as e:
            response_text = f"Lỗi khi gọi AI (sau nhiều lần thử): {e}"

        latency = time.perf_counter() - t0

        meta = {
            "concepts": concepts,
            "vector_sources": vec_sources,
            "graph_edges": edges,
            "article_ids_from_vector": article_ids
        }

        return response_text, meta, latency

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Hàm gọi LLM với cơ chế chờ thông minh"""
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                error_msg = str(e)
                # Nếu lỗi Quota (429) hoặc Server (5xx)
                if "429" in error_msg or "500" in error_msg or "503" in error_msg:
                    wait_time = 5 * (attempt + 1) # Chờ 5s, 10s, 15s
                    print(f"⏳ Mạng bận/Hết quota, thử lại sau {wait_time}s... (Lần {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise e # Lỗi khác thì dừng luôn
        raise Exception("Đã hết số lần thử lại.")

    def _query_neo4j(self, article_ids: list, concepts: list) -> list:
        edges = []
        try:
            with self.driver.session() as sess:
                if article_ids:
                    res1 = sess.run("""
                        MATCH (a:Article)-[r:RELATED]-(b:Article)
                        WHERE a.id IN $ids
                        RETURN a.id AS from_id, b.id AS to_id, coalesce(r.relation,'RELATED') AS rel, b.topic AS topic
                        LIMIT 50
                    """, ids=article_ids)
                    edges += [dict(r) for r in res1]

                if len(edges) < 5 and concepts:
                    res2 = sess.run("""
                        MATCH (a:Article)-[r:RELATED]->(b:Article)
                        WHERE any(c IN $concepts WHERE toLower(a.topic) CONTAINS toLower(c))
                        OR any(c IN $concepts WHERE toLower(b.topic) CONTAINS toLower(c))
                        RETURN a.id AS from_id, b.id AS to_id, coalesce(r.relation,'RELATED') AS rel, b.topic AS topic
                        LIMIT 25
                    """, concepts=[c.lower() for c in concepts])
                    edges += [dict(r) for r in res2]
        except Exception as e:
            print(f"⚠️ Lỗi truy vấn Neo4j: {e}. Đang bỏ qua Graph.")
        return edges
