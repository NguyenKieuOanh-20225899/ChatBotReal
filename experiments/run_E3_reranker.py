import json, yaml
from tqdm import tqdm
from search_pipeline import HybridSearcher
from rerank import CrossEncoderReranker
from eval_metrics import recall_at_k, mrr, ndcg_at_k

# Hàm chuẩn hóa ID (copy từ bài sửa E2)
def normalize_id(doc_id):
    s = str(doc_id)
    if "_chunks#" in s: s = s.split("_chunks#")[0]
    if ":" in s: s = s.split(":")[0]
    if s.endswith("_knowledge"): s = s.replace("_knowledge", "")
    return s.strip()

# Load config
cfg = yaml.safe_load(open("experiments/config.yaml","r",encoding="utf-8"))
dev = [json.loads(l) for l in open(cfg["paths"]["devset_path"],"r",encoding="utf-8")]

# Khởi tạo
hs = HybridSearcher(cfg)
rr = CrossEncoderReranker(cfg["reranker"]["model_name"])
keep_topk = cfg["reranker"]["keep_topk"]

R10 = []; MRR = []; N10 = []

print(f"Đang chạy đánh giá Rerank với model: {cfg['reranker']['model_name']}")
for ex in tqdm(dev):
    q = ex["question"]
    raw_gold = ex["gold"]

    # 1. Hybrid Search (Lấy top 50 hoặc config dense_topk)
    # Cần lấy nhiều ứng viên hơn keep_topk để Reranker có cái mà lọc
    cands = hs.search(q)

    # 2. Rerank (Lọc lại còn keep_topk = 5 hoặc 10)
    reranked, smax = rr.rerank(q, cands, keep_topk=keep_topk)

    # 3. Chuẩn hóa ID và tính điểm
    retrieved_ids = [normalize_id(h["meta"].get("chunk_id") or h["meta"].get("stable_id")) for h in reranked]
    gold_ids = [normalize_id(g) for g in raw_gold]

    R10.append(recall_at_k(retrieved_ids, gold_ids, k=10))
    MRR.append(mrr(retrieved_ids, gold_ids))
    N10.append(ndcg_at_k(retrieved_ids, gold_ids, k=10))

if len(R10) > 0:
    print("\n=== KẾT QUẢ SAU KHI RERANK ===")
    print(f"Recall@10 = {sum(R10)/len(R10):.3f}")
    print(f"MRR       = {sum(MRR)/len(MRR):.3f}")
    print(f"nDCG@10   = {sum(N10)/len(N10):.3f}")
else:
    print("Không có dữ liệu.")