import numpy as np
import logging
from utils.error_handler import handle_errors, log_execution_time

logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, bm25, semantic):
        self.bm25 = bm25
        self.semantic = semantic
        self._reranker = None
    
    @property
    def reranker(self):
        if self._reranker is None:
            from retrieval.reranker import ContextAwareReranker
            logger.info("Loading Reranker (lazy)...")
            self._reranker = ContextAwareReranker(self.bm25.dataset)
        return self._reranker
    
    @handle_errors
    @log_execution_time
    def search(self, query, k=5, initial_k=200,   # ← افزایش از ۵۰ به ۲۰۰
               bm25_weight=0.4, sem_weight=0.6,    # ← تغییر وزن به نفع Semantic
               use_reranker=False):
        bm25_results = self.bm25.search(query, k=initial_k)
        semantic_results = self.semantic.search(query, k=initial_k)
        
        scores = {}
        for rank, item in enumerate(bm25_results):
            idx = item.get("index")
            if idx is not None:
                scores[idx] = scores.get(idx, 0) + bm25_weight * (1.0 / (rank + 60))
        for rank, item in enumerate(semantic_results):
            idx = item.get("index")
            if idx is not None:
                scores[idx] = scores.get(idx, 0) + sem_weight * (1.0 / (rank + 60))
        
        bm25_indices = {r["index"] for r in bm25_results if "index" in r}
        sem_indices = {r["index"] for r in semantic_results if "index" in r}
        for idx in scores:
            if idx in bm25_indices and idx in sem_indices:
                scores[idx] *= 1.25
        
        sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:initial_k]
        
        candidates = []
        for idx in sorted_indices:
            item = self.bm25.dataset[idx]
            candidates.append({
                "index": idx,
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "category": item.get("category", ""),
                "specialty": item.get("specialty", ""),
                "context": item.get("context", ""),  # ← اضافه کردن context
                "fusion_score": scores[idx]
            })
        
        if use_reranker:
            reranked = self.reranker.rerank(query, candidates)
            return reranked[:k]
        else:
            candidates.sort(key=lambda x: x["fusion_score"], reverse=True)
            return candidates[:k]
    
    @handle_errors
    @log_execution_time
    def search_with_weights(self, query, k=5, alpha=0.4, initial_k=200):  # ← α کمتر = وزن بیشتر Semantic
        bm25_results = self.bm25.search(query, k=initial_k)
        semantic_results = self.semantic.search(query, k=initial_k)
        
        bm25_scores = np.array([r.get("score", 0) for r in bm25_results])
        semantic_scores = np.array([r.get("score", 0) for r in semantic_results])
        
        def normalize(scores):
            if len(scores) == 0:
                return np.array([])
            if scores.max() == scores.min():
                return np.full_like(scores, 0.5)
            return (scores - scores.min()) / (scores.max() - scores.min())
        
        bm25_norm = normalize(bm25_scores)
        semantic_norm = normalize(semantic_scores)
        
        combined = {}
        for item, score in zip(bm25_results, bm25_norm):
            idx = item.get("index")
            if idx is not None:
                combined[idx] = {"item": item, "bm25_score": float(score), "semantic_score": 0.0}
        
        for item, score in zip(semantic_results, semantic_norm):
            idx = item.get("index")
            if idx is not None:
                if idx in combined:
                    combined[idx]["semantic_score"] = float(score)
                else:
                    combined[idx] = {"item": item, "bm25_score": 0.0, "semantic_score": float(score)}
        
        final_results = []
        for idx, data in combined.items():
            item = data["item"]
            final_score = alpha * data["bm25_score"] + (1 - alpha) * data["semantic_score"]
            final_results.append({
                "index": idx,
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "category": item.get("category", ""),
                "specialty": item.get("specialty", ""),
                "context": item.get("context", ""),  # ← اضافه کردن context
                "final_score": float(final_score),
                "bm25_score": data["bm25_score"],
                "semantic_score": data["semantic_score"]
            })
        
        final_results.sort(key=lambda x: x["final_score"], reverse=True)
        return final_results[:k]