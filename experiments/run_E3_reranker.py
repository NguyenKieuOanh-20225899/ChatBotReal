import json
import yaml
import sys
import os
from tqdm import tqdm
from pathlib import Path

# --- 1. Setup đường dẫn (Robust Imports) ---
# Đảm bảo Python tìm thấy các module nằm cùng thư mục (search_pipeline, rerank, eval_metrics)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import các module custom
try:
    from search_pipeline import HybridSearcher
    from rerank import CrossEncoderReranker
    from eval_metrics import recall_at_k, mrr, ndcg_at_k
except ImportError as e:
    print(f"Lỗi import: {e}")
    print("Vui lòng đảm bảo bạn đang có đầy đủ các file: search_pipeline.py, rerank.py, eval_metrics.py trong thư mục experiments/")
    sys.exit(1)

# --- 2. Hàm tiện ích ---
def normalize_id(doc_id):
    """
    Chuẩn hóa ID về format gốc (VD: '01_2004_NQ-HĐTP') để so khớp chính xác.
    """
    s = str(doc_id)
    if "_chunks#" in s: 
        s = s.split("_chunks#")[0]
    if ":" in s: 
        s = s.split(":")[0]
    if s.endswith("_knowledge"): 
        s = s.replace("_knowledge", "")
    return s.strip()

# --- 3. Load cấu hình & Dữ liệu ---
# Tự động tìm config.yaml ở cùng thư mục với script này
base_path = Path(current_dir)
config_path = base_path / "config.yaml"

if not config_path.exists():
    print(f"⚠️ Không tìm thấy config tại: {config_path}")
    print("Hãy đảm bảo bạn đang đứng ở thư mục gốc dự án hoặc file config tồn tại.")
    sys.exit(1)

cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))

# Load Devset (Đường dẫn trong config thường tính từ root project)
# Nếu chạy từ root, cfg["paths"]["devset_path"] là đúng.
devset_path = Path(cfg["paths"]["devset_path"])
if not devset_path.exists():
    # Fallback: Nếu đường dẫn tương đối không thấy, thử tìm tương đối từ script
    devset_path = base_path.parent / cfg["paths"]["devset_path"]

print(f"Loading devset from: {devset_path}")
dev = [json.loads(l) for l in open(devset_path, "r", encoding="utf-8")]

# --- 4. Khởi tạo Pipeline ---
print("Khởi tạo HybridSearcher & Reranker...")
hs = HybridSearcher(cfg)
rr = CrossEncoderReranker(cfg["reranker"]["model_name"])
keep_topk = cfg["reranker"]["keep_topk"]

# --- 5. Chạy đánh giá ---
R10 = []; MRR = []; N10 = []

print(f"🚀 Bắt đầu đánh giá Rerank với model: {cfg['reranker']['model_name']}")
print(f"👉 Chiến lược: Retrieve Top-100 -> Rerank -> Keep Top-{keep_topk}")

for ex in tqdm(dev, desc="Evaluating"):
    q = ex["question"]
    raw_gold = ex["gold"]

    # BƯỚC 1: Hybrid Search mở rộng (Lấy 100 kết quả)
    # Lưu ý: Cần update search_pipeline.py để hàm search nhận tham số k
    cands = hs.search(q, k=100)

    # BƯỚC 2: Rerank (Lọc lại còn top k tốt nhất, vd: 5)
    reranked, smax = rr.rerank(q, cands, keep_topk=keep_topk)

    # BƯỚC 3: Chuẩn hóa ID và tính điểm
    retrieved_ids = [normalize_id(h["meta"].get("chunk_id") or h["meta"].get("stable_id")) for h in reranked]
    gold_ids = [normalize_id(g) for g in raw_gold]

    # Tính toán metrics (Lưu ý: Nếu keep_topk=5 thì Recall@10 thực chất là Recall@5)
    R10.append(recall_at_k(retrieved_ids, gold_ids, k=10))
    MRR.append(mrr(retrieved_ids, gold_ids))
    N10.append(ndcg_at_k(retrieved_ids, gold_ids, k=10))

# --- 6. Kết quả ---
if len(R10) > 0:
    print("\n" + "="*30)
    print("📊 KẾT QUẢ ĐÁNH GIÁ (Reranked)")
    print("="*30)
    print(f"✅ Recall@10 : {sum(R10)/len(R10):.4f}")
    print(f"✅ MRR       : {sum(MRR)/len(MRR):.4f}")
    print(f"✅ nDCG@10   : {sum(N10)/len(N10):.4f}")
    print("="*30)
else:
    print("⚠️ Không có dữ liệu để đánh giá.")