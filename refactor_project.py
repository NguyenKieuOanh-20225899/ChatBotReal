import os
import shutil
from pathlib import Path

# --- CẤU HÌNH ---
BASE_DIR = Path(__file__).parent.absolute()

# Định nghĩa các thư mục mới cần tạo
NEW_DIRS = [
    "config",
    "logs",
    "src",
    "src/core",
    "src/services",
    "src/utils",
    "data/artifacts", # Chuyển artifacts vào data
]

# Nội dung cho các file __init__.py
INIT_CONTENT = ""

# --- NỘI DUNG CÁC FILE MỚI (CLEAN CODE) ---

# 1. config/config.yaml (Cập nhật đường dẫn)
CONFIG_CONTENT = """paths:
  chunks_dir: "data/chunks"
  kb_dir: "data/knowledge_base"
  artifacts_dir: "data/artifacts"
  devset_path: "experiments/devset/dev_100.jsonl"

index:
  embedding_model: "intfloat/multilingual-e5-base"
  faiss_nlist: 100
  faiss_nprobe: 10

retrieval:
  bm25_topk: 50
  dense_topk: 50
  rrf_K: 60
  final_topk: 20
  rrf_weights: [2.0, 1.0]

reranker:
  model_name: "BAAI/bge-reranker-v2-m3"
  apply: true
  keep_topk: 5

thresholds:
  answerability_min_score: 0.5
"""

# 2. src/services/retrieval_service.py
RETRIEVAL_SERVICE_CONTENT = """import os
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
"""

# 3. src/services/graph_rag_service.py
GRAPH_RAG_SERVICE_CONTENT = """import os
import re
import time
from typing import Tuple, Any

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

        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_pass = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_pass))

        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=self.api_key
        )

        try:
            self.vector_db = FAISS.load_local(vector_db_path, self.embeddings, allow_dangerous_deserialization=True)
            print("✅ Vector DB loaded.")
        except Exception as e:
            print(f"⚠️ Warning: Could not load Vector DB at {vector_db_path}. Error: {e}")
            self.vector_db = None

    def close(self):
        if self.driver:
            self.driver.close()

    def query(self, query_text: str, k: int = 5) -> Tuple[str, dict, float]:
        t0 = time.perf_counter()

        ctx_vec = ""
        vec_sources = []
        article_ids = []

        if self.vector_db:
            hits = self.vector_db.similarity_search(query_text, k=k)
            ctx_vec = "\\n\\n".join(h.page_content for h in hits)
            vec_sources = [h.metadata.get("source") for h in hits]
            found_ids = re.findall(r"Điều\s+\d+", ctx_vec, flags=re.IGNORECASE)
            article_ids = list({a.strip() for a in found_ids})[:10]

        extract_resp = self.model.generate_content(
            f"Từ câu hỏi sau, liệt kê tối đa 5 khái niệm pháp lý cốt lõi (mỗi dòng 1 mục, không giải thích):\\n{query_text}"
        )
        concepts = [x.strip("-• \\n") for x in extract_resp.text.splitlines() if x.strip()][:5]

        edges = self._query_neo4j(article_ids, concepts)

        ctx_graph = "\\n".join(f"{e['from_id']} {e['rel']} {e['to_id']} ({e.get('topic','')})" for e in edges)
        if not ctx_graph:
            ctx_graph = "Không có thông tin từ đồ thị."

        prompt = f'''
Bạn là trợ lý pháp lý Việt Nam. Dựa vào ngữ cảnh dưới đây, trả lời chính xác, có dẫn Điều/khoản nếu có.

[Câu hỏi]
{query_text}

[Đoạn văn pháp luật (Vector)]
{ctx_vec}

[Quan hệ pháp lý (Graph)]
{ctx_graph}
'''
        response = self.model.generate_content(prompt)
        latency = time.perf_counter() - t0

        meta = {
            "concepts": concepts,
            "vector_sources": vec_sources,
            "graph_edges": edges,
            "article_ids_from_vector": article_ids
        }

        return response.text, meta, latency

    def _query_neo4j(self, article_ids: list, concepts: list) -> list:
        edges = []
        with self.driver.session() as sess:
            if article_ids:
                res1 = sess.run(\"\"\"
                    MATCH (a:Article)-[r:RELATED]-(b:Article)
                    WHERE a.id IN $ids
                    RETURN a.id AS from_id, b.id AS to_id, coalesce(r.relation,'RELATED') AS rel, b.topic AS topic
                    LIMIT 50
                \"\"\", ids=article_ids)
                edges += [dict(r) for r in res1]

            if len(edges) < 5 and concepts:
                res2 = sess.run(\"\"\"
                    MATCH (a:Article)-[r:RELATED]->(b:Article)
                    WHERE any(c IN $concepts WHERE toLower(a.topic) CONTAINS toLower(c))
                       OR any(c IN $concepts WHERE toLower(b.topic) CONTAINS toLower(c))
                    RETURN a.id AS from_id, b.id AS to_id, coalesce(r.relation,'RELATED') AS rel, b.topic AS topic
                    LIMIT 25
                \"\"\", concepts=[c.lower() for c in concepts])
                edges += [dict(r) for r in res2]
        return edges
"""

# 4. scripts/run_cli_chat.py
RUN_CLI_CONTENT = """import sys
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
            query = input("\\n❓ Nhập câu hỏi pháp luật: ").strip()
            if query.lower() in ["exit", "quit", "thoát"]:
                break
            if not query:
                continue

            answer, meta, latency = bot.query(query)

            print("\\n=== TRẢ LỜI ===")
            print(answer)
            print(f"\\n⏱️ Thời gian: {latency:.2f}s")
            print(f"📊 Metadata: {len(meta['graph_edges'])} cạnh đồ thị, {len(meta['vector_sources'])} nguồn vector.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Lỗi xử lý: {e}")

    bot.close()
    print("\\nTam biệt!")

if __name__ == "__main__":
    main()
"""

# --- HÀM HỖ TRỢ ---

def create_directory_structure():
    print("📂 Đang tạo cấu trúc thư mục...")
    for d in NEW_DIRS:
        path = BASE_DIR / d
        path.mkdir(parents=True, exist_ok=True)
        # Tạo __init__.py cho các folder src
        if d.startswith("src"):
            (path / "__init__.py").write_text(INIT_CONTENT)

def move_and_patch_file(src_path: Path, dest_path: Path, replacements: dict):
    """Di chuyển file và thay thế nội dung (imports)"""
    if not src_path.exists():
        print(f"⚠️ Không tìm thấy file gốc: {src_path}. Bỏ qua.")
        return

    print(f"🚚 Di chuyển & Patch: {src_path.name} -> {dest_path}")
    content = src_path.read_text(encoding="utf-8")

    # Thực hiện thay thế các chuỗi import cũ
    for old, new in replacements.items():
        content = content.replace(old, new)

    dest_path.write_text(content, encoding="utf-8")

def write_new_file(path: Path, content: str):
    print(f"📝 Tạo file mới: {path}")
    path.write_text(content, encoding="utf-8")

def move_artifacts():
    """Di chuyển các file trong experiments/artifacts sang data/artifacts"""
    src_dir = BASE_DIR / "experiments" / "artifacts"
    dest_dir = BASE_DIR / "data" / "artifacts"

    if src_dir.exists():
        print("📦 Đang di chuyển Artifacts...")
        for item in src_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, dest_dir / item.name)
        print("✅ Đã di chuyển Artifacts xong.")
    else:
        print("ℹ️ Không tìm thấy thư mục artifacts cũ.")

# --- MAIN EXECUTION ---

def main():
    print("🚀 Bắt đầu Refactor Project...")

    # 1. Tạo thư mục
    create_directory_structure()

    # 2. Tạo Config mới
    write_new_file(BASE_DIR / "config/config.yaml", CONFIG_CONTENT)

    # 3. Move & Patch các file Core từ experiments
    # Map đổi tên import
    core_replacements = {
        "from experiments.text_utils": "from src.utils.text_utils",
        "from experiments.rerank": "from src.core.reranker",
        "from experiments.search_pipeline": "from src.core.search_engine",
        "import experiments.config": "import config"
    }

    # experiments/search_pipeline.py -> src/core/search_engine.py
    move_and_patch_file(
        BASE_DIR / "experiments/search_pipeline.py",
        BASE_DIR / "src/core/search_engine.py",
        core_replacements
    )

    # experiments/rerank.py -> src/core/reranker.py
    move_and_patch_file(
        BASE_DIR / "experiments/rerank.py",
        BASE_DIR / "src/core/reranker.py",
        core_replacements
    )

    # experiments/text_utils.py -> src/utils/text_utils.py
    move_and_patch_file(
        BASE_DIR / "experiments/text_utils.py",
        BASE_DIR / "src/utils/text_utils.py",
        core_replacements
    )

    # 4. Tạo các file Services mới (Code đã refactor hoàn chỉnh)
    write_new_file(BASE_DIR / "src/services/retrieval_service.py", RETRIEVAL_SERVICE_CONTENT)
    write_new_file(BASE_DIR / "src/services/graph_rag_service.py", GRAPH_RAG_SERVICE_CONTENT)

    # 5. Tạo Script chạy
    write_new_file(BASE_DIR / "scripts/run_cli_chat.py", RUN_CLI_CONTENT)

    # 6. Di chuyển dữ liệu artifacts (indexes)
    move_artifacts()

    print("\n🎉 Refactor hoàn tất!")
    print("👉 Bây giờ bạn có thể chạy: python scripts/run_cli_chat.py")
    print("⚠️ Lưu ý: Hãy kiểm tra kỹ lại file .env của bạn.")

if __name__ == "__main__":
    main()
