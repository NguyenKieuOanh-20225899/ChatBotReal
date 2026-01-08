# File: scripts/evaluate_models.py
import sys
import os
import time
import json
import pandas as pd
import yaml
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Setup đường dẫn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Service
from src.core.search_engine import HybridSearcher
try:
    from src.services.graph_rag_service import GraphRAGService
except ImportError:
    GraphRAGService = None

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Khởi tạo AI Judge
judge_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0 # Nhiệt độ 0 để chấm điểm nhất quán
)

def load_config():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# --- HÀM CHẤM ĐIỂM ---
def ai_grade(question, ground_truth, model_answer, mode="essay"):
    """
    Hàm chấm điểm đa năng.
    """
    if not model_answer:
        return 0, "Không trả lời"

    if mode == "mcq":
        # Prompt chấm Trắc nghiệm
        prompt = f"""
        Bạn là máy chấm thi trắc nghiệm.
        [CÂU HỎI]: {question}
        [ĐÁP ÁN ĐÚNG]: {ground_truth}
        [CÂU TRẢ LỜI CỦA AI]: {model_answer}

        YÊU CẦU:
        1. Kiểm tra xem AI có chọn đúng đáp án ({ground_truth}) không.
        2. Nếu đúng -> score: 1, Nếu sai -> score: 0.

        OUTPUT JSON: {{"score": 1, "reason": "Chọn đúng B"}}
        """
    else:
        # Prompt chấm Tự luận
        prompt = f"""
        Bạn là Giám khảo Luật.
        [CÂU HỎI]: {question}
        [ĐÁP ÁN CHUẨN]: {ground_truth}
        [TRẢ LỜI CỦA AI]: {model_answer}

        YÊU CẦU:
        1. Chấm điểm độ chính xác ngữ nghĩa (Thang 0-10).
        2. Không cần đúng từng chữ, chỉ cần đúng ý pháp lý.

        OUTPUT JSON: {{"score": 8.5, "reason": "Đủ ý nhưng thiếu trích dẫn"}}
        """

    try:
        res = judge_llm.invoke(prompt)
        content = res.content.strip()
        # Parse JSON
        start = content.find('{')
        end = content.rfind('}') + 1
        result = json.loads(content[start:end])
        return result.get("score", 0), result.get("reason", "")
    except:
        return 0, "Lỗi chấm điểm"

# --- HÀM CHẠY ĐÁNH GIÁ (Dùng chung) ---
def run_evaluation(test_file, output_file, mode, graph_service):
    if not os.path.exists(test_file):
        print(f"⚠️ Không tìm thấy file: {test_file}. Bỏ qua.")
        return

    print(f"\n⚡ Đang đánh giá: {mode.upper()} (File: {test_file})...")

    with open(test_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    results = []
    total_score = 0

    for idx, case in enumerate(test_cases):
        q = case["question"]
        gt = case["ground_truth"]

        print(f"   🔹 Câu {idx+1}: {q[:50]}...")

        # Query Graph RAG
        ans = "N/A"
        sources = 0
        latency = 0

        if graph_service:
            try:
                # Nếu là trắc nghiệm, nhắc AI chọn A,B,C,D
                query_input = q
                if mode == "mcq":
                    query_input += "\n(Chỉ chọn 1 đáp án đúng nhất A, B, C hoặc D và giải thích ngắn gọn)"

                ans, meta, latency = graph_service.query(query_input)
                sources = len(meta.get("vector_sources", [])) + meta.get("graph_edges_used", 0)
            except Exception as e:
                ans = f"Error: {e}"

        # Chấm điểm
        score, reason = ai_grade(q, gt, ans, mode)
        total_score += score

        # Lưu kết quả
        results.append({
            "Câu hỏi": q,
            "Đáp án chuẩn": gt,
            "AI Trả lời": ans,
            "Điểm": score,
            "Lý do": reason,
            "Nguồn tìm thấy": sources,
            "Thời gian (s)": round(latency, 2)
        })

    # Xuất Excel
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)

    # In báo cáo nhanh
    print(f"   ✅ Đã xong! Kết quả lưu tại: {output_file}")
    if mode == "mcq":
        # Trắc nghiệm tính theo % đúng
        accuracy = (total_score / len(test_cases)) * 100
        print(f"   📊 Độ chính xác (Accuracy): {accuracy:.2f}% ({int(total_score)}/{len(test_cases)} câu đúng)")
    else:
        # Tự luận tính điểm trung bình
        avg_score = total_score / len(test_cases)
        print(f"   📊 Điểm chất lượng TB: {avg_score:.2f}/10")

def main():
    print("🚀 BẮT ĐẦU QUÁ TRÌNH ĐÁNH GIÁ TÁCH BIỆT")

    # Init Graph Service
    graph_service = GraphRAGService() if GraphRAGService else None

    if not graph_service:
        print("❌ Lỗi: Không khởi tạo được GraphRAGService.")
        return

    # 1. Đánh giá Trắc nghiệm (MCQ)
    run_evaluation(
        test_file="data/test_set_mcq.json",
        output_file="ket_qua_trac_nghiem.xlsx",
        mode="mcq",
        graph_service=graph_service
    )

    # 2. Đánh giá Tự luận (Essay)
    run_evaluation(
        test_file="data/test_set_essay.json",
        output_file="ket_qua_tu_luan.xlsx",
        mode="essay",
        graph_service=graph_service
    )

    print("\n🎉 HOÀN TẤT TOÀN BỘ!")

if __name__ == "__main__":
    main()
