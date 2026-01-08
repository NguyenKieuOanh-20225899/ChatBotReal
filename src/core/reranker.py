from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch, numpy as np

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device=None):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    @torch.no_grad()
    def rerank(self, query, candidates, keep_topk=10, batch_size=32):
        if not candidates:
            return [], 0.0

        pairs = [(query, c["doc"]) for c in candidates]
        scores = []
        for i in range(0, len(pairs), batch_size):
            q, d = zip(*pairs[i:i+batch_size])
            enc = self.tok(list(q), list(d), padding=True, truncation=True, return_tensors="pt").to(self.device)

            out = self.model(**enc).logits

            # --- FIX LỖI Ở ĐÂY ---
            # Nếu model trả về 2 giá trị (Binary Classification), lấy cột thứ 2 (index 1)
            if out.shape[-1] > 1:
                out = out[:, 1]
            else:
                out = out.squeeze(-1)
            # ---------------------

            scores.extend(out.detach().cpu().tolist())

        # Sắp xếp giảm dần
        scores = np.array(scores)
        order = np.argsort(-scores)[:keep_topk]

        reranked = [candidates[i] | {"rerank_score": float(scores[i])} for i in order]
        return reranked, float(np.max(scores)) if len(scores) > 0 else 0.0