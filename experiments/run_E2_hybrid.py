import json, yaml
from pathlib import Path
from tqdm import tqdm
from search_pipeline import HybridSearcher
from eval_metrics import recall_at_k, mrr, ndcg_at_k

cfg = yaml.safe_load(open("experiments/config.yaml","r",encoding="utf-8"))
dev = [json.loads(l) for l in open(cfg["paths"]["devset_path"],"r",encoding="utf-8")]
hs = HybridSearcher(cfg)

def normalize_id(doc_id):
    """
    Chuẩn hóa ID về tên văn bản gốc để so sánh được với nhau.
    Input 1: 02_2000_NQ-HĐTP_knowledge:Điều 31  -> Output: 02_2000_NQ-HĐTP
    Input 2: 02_2000_NQ-HĐTP_chunks#17          -> Output: 02_2000_NQ-HĐTP
    """
    s = str(doc_id)

    # 1. Xử lý Retrieved ID (Cắt bỏ phần _chunks#...)
    if "_chunks#" in s:
        s = s.split("_chunks#")[0]

    # 2. Xử lý Gold ID (Cắt bỏ phần :Điều...)
    if ":" in s:
        s = s.split(":")[0]

    # 3. Xử lý lệch tên file (Bỏ chữ _knowledge nếu có)
    if s.endswith("_knowledge"):
        s = s.replace("_knowledge", "")

    return s.strip()

R10 = []; MRR = []; N10 = []

print("Đang chạy đánh giá...")
# Debug 3 câu đầu tiên để kiểm tra xem ID đã khớp chưa
debug_count = 0

for ex in tqdm(dev):
    q = ex["question"]
    raw_gold = ex["gold"] # List các ID đáp án gốc

    # 1. Tìm kiếm
    hits = hs.search(q)

    # 2. Lấy ID và CHUẨN HÓA cả 2 bên về cùng định dạng
    retrieved_ids = [normalize_id(h["meta"].get("chunk_id") or h["meta"].get("stable_id")) for h in hits]
    gold_ids = [normalize_id(g) for g in raw_gold]

    # --- DEBUG LOGIC (In ra để kiểm tra) ---
    if debug_count < 3:
        print(f"\n--- DEBUG Query {debug_count+1} ---")
        print(f"Question: {q}")
        # In ra để xem sau khi chuẩn hóa chúng có giống nhau không
        print(f"Gold Norm: {gold_ids}")
        print(f"Retr Norm: {retrieved_ids}")
        debug_count += 1
    # ---------------------------------------

    # 3. Tính điểm
    R10.append(recall_at_k(retrieved_ids, gold_ids, k=10))
    MRR.append(mrr(retrieved_ids, gold_ids))
    N10.append(ndcg_at_k(retrieved_ids, gold_ids, k=10))

# Tránh lỗi chia cho 0 nếu list rỗng
if len(R10) > 0:
    print(f"\nHybrid Result: Recall@10={sum(R10)/len(R10):.3f}  MRR={sum(MRR)/len(MRR):.3f}  nDCG@10={sum(N10)/len(N10):.3f}")
else:
    print("Không có dữ liệu để đánh giá.")