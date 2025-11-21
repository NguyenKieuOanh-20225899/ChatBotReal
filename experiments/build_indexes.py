import os, json, yaml, faiss, numpy as np, pickle
from tqdm import tqdm
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
# ... (các import khác giữ nguyên)
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# --- SỬA ĐOẠN NÀY ---
try:
    from experiments.text_utils import tokenize_vn, preprocess_text
except ImportError:
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from text_utils import tokenize_vn, preprocess_text
# --------------------

def load_chunks(chunks_dir):
    docs, metas = [], []
    # Sắp xếp file để thứ tự luôn cố định mỗi lần chạy
    files = sorted(list(Path(chunks_dir).glob("*.json")))
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for i, item in enumerate(data):
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                # Lưu thêm chunk_id gốc nếu có
                meta = {**item, "source_file": p.name, "chunk_id": f"{p.stem}#{i}"}
            else:
                text = str(item)
                meta = {"source_file": p.name, "chunk_id": f"{p.stem}#{i}"}
            
            # Preprocess trước khi thêm
            text = preprocess_text(text)
            if text:
                docs.append(text)
                metas.append(meta)
    return docs, metas

def build_bm25(docs):
    # Sử dụng hàm tokenize đồng nhất
    tokenized = [tokenize_vn(d) for d in tqdm(docs, desc="Tokenizing BM25")]
    return BM25Okapi(tokenized)

def build_faiss(docs, model_name, nlist=4096):
    print(f"Encoding {len(docs)} docs with {model_name}...")
    model = SentenceTransformer(model_name)
    X = model.encode(docs, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    
    d = X.shape[1]
    # Tự động điều chỉnh nlist nếu dữ liệu quá ít
    nlist = min(nlist, int(len(docs) / 30)) if len(docs) > 0 else 1
    
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    
    print("Training FAISS index...")
    index.train(X.astype(np.float32))
    index.add(X.astype(np.float32))
    return index

if __name__ == "__main__":
    cfg = yaml.safe_load(open("experiments/config.yaml", "r", encoding="utf-8"))
    chunks_dir = cfg["paths"]["chunks_dir"]
    arts_dir = Path(cfg["paths"]["artifacts_dir"])
    arts_dir.mkdir(parents=True, exist_ok=True)

    docs, metas = load_chunks(chunks_dir)
    print(f"Loaded {len(docs)} chunks")

    if not docs:
        print("No chunks found! Please run data splitting first.")
        exit()

    # BM25
    bm25 = build_bm25(docs)
    with open(arts_dir/"bm25.pkl", "wb") as f: pickle.dump(bm25, f)
    
    # Save Docs & Metas
    with open(arts_dir/"docs.json", "w", encoding="utf-8") as f: 
        json.dump(docs, f, ensure_ascii=False, indent=2)
    with open(arts_dir/"metas.json", "w", encoding="utf-8") as f: 
        json.dump(metas, f, ensure_ascii=False, indent=2)

    # FAISS
    idx = build_faiss(docs, cfg["index"]["embedding_model"], cfg["index"]["faiss_nlist"])
    faiss.write_index(idx, str(arts_dir/"faiss.index"))
    print(f"Indexes saved to {arts_dir}")