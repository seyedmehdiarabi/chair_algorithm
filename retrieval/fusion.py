import numpy as np
from retrieval.reranker import ContextAwareReranker


class HybridRetriever:
    def __init__(self, bm25, semantic):
        """
        ترکیب نتایج BM25 و Semantic با روش RRF + Reranker
        """
        self.bm25 = bm25
        self.semantic = semantic

        # 🔥 اضافه شدن reranker
        self.reranker = ContextAwareReranker()

    def search(self, query, k=5, initial_k=20, rrf_constant=1):
        """
        جستجوی ترکیبی + reranking
        """

        # =========================
        # 1. گرفتن نتایج اولیه
        # =========================
        bm25_results = self.bm25.search(query, k=initial_k)
        semantic_results = self.semantic.search(query, k=initial_k)

        BM25_WEIGHT = 0.6
        SEM_WEIGHT = 0.4

        scores = {}

        # ---------- BM25 ----------
        for rank, item in enumerate(bm25_results):
            idx = item["index"]

            rrf_score = 1.0 / (rank + 60)   # smoothing
            scores[idx] = scores.get(idx, 0) + BM25_WEIGHT * rrf_score

        # ---------- SEMANTIC ----------
        for rank, item in enumerate(semantic_results):
            idx = item["index"]

            rrf_score = 1.0 / (rank + 60)
            scores[idx] = scores.get(idx, 0) + SEM_WEIGHT * rrf_score

        # ---------- OVERLAP BOOST ----------

        for idx in scores:

            in_bm25 = any(
                r["index"] == idx 
                for r in bm25_results
            )

            in_semantic = any(
                r["index"] == idx 
                for r in semantic_results
            )


            if in_bm25 and in_semantic:
                scores[idx] *= 1.3

        # =========================
        # 3. ساخت CANDIDATES
        # =========================
        sorted_indices = sorted(
            scores.keys(),
            key=lambda x: scores[x],
            reverse=True
        )[:initial_k]

        candidates = []

        for idx in sorted_indices:
            item = self.bm25.dataset[idx]

            candidates.append({
                "index": idx,
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "category": item.get("category", ""),
                "fusion_score": scores[idx]
            })

        # =========================
        # 4. RERANKING (مرحله هوشمند)
        # =========================
        reranked_results = []

        for item in candidates:

            doc_text = item["question"] + " " + item["answer"] + " " + item["category"]

            score = self.reranker.score(query, doc_text)

            item["rerank_score"] = score

            reranked_results.append(item)

        # =========================
        # 5. مرتب‌سازی نهایی
        # =========================
        reranked_results.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked_results[:k]

    def search_with_weights(self, query, k=5, alpha=0.5):
        """
        روش قبلی (بدون reranker) - دست نخورده
        """

        bm25_results = self.bm25.search(query, k=20)
        semantic_results = self.semantic.search(query, k=20)

        bm25_scores = np.array([r["score"] for r in bm25_results])

        if bm25_scores.max() != bm25_scores.min():
            bm25_norm = (bm25_scores - bm25_scores.min()) / (
                bm25_scores.max() - bm25_scores.min()
            )
        else:
            bm25_norm = np.full_like(bm25_scores, 0.5)

        semantic_distances = np.array([r["distance"] for r in semantic_results])
        semantic_scores = -semantic_distances

        if semantic_scores.max() != semantic_scores.min():
            semantic_norm = (semantic_scores - semantic_scores.min()) / (
                semantic_scores.max() - semantic_scores.min()
            )
        else:
            semantic_norm = np.full_like(semantic_scores, 0.5)

        combined = {}

        for item, score in zip(bm25_results, bm25_norm):
            idx = item["index"]
            combined[idx] = {
                "item": item,
                "bm25_score": score,
                "semantic_score": 0
            }

        for item, score in zip(semantic_results, semantic_norm):
            idx = item["index"]
            if idx in combined:
                combined[idx]["semantic_score"] = score
            else:
                combined[idx] = {
                    "item": item,
                    "bm25_score": 0,
                    "semantic_score": score
                }

        final_results = []

        for idx, data in combined.items():
            final_score = alpha * data["bm25_score"] + (1 - alpha) * data["semantic_score"]

            item = data["item"]

            final_results.append({
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "category": item.get("category", ""),
                "final_score": final_score,
                "bm25_score": data["bm25_score"],
                "semantic_score": data["semantic_score"],
                "index": idx
            })

        final_results.sort(key=lambda x: x["final_score"], reverse=True)

        return final_results[:k]