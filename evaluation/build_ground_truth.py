import json
import random
import logging
from retrieval.bm25 import BM25Retriever
from retrieval.semantic import SemanticRetriever
from retrieval.fusion import HybridRetriever

logger = logging.getLogger(__name__)

class GroundTruthBuilder:
    def __init__(self, dataset, sample_size=300, k=10):
        self.dataset = dataset
        self.sample_size = min(sample_size, len(dataset))
        self.k = k
        # ساخت مدل‌ها برای تولید ground truth با روش voting
        self.bm25 = BM25Retriever(dataset)
        self.semantic = SemanticRetriever(dataset)
        self.hybrid = HybridRetriever(self.bm25, self.semantic)

    def build(self):
        """
        ساخت ground truth با استفاده از رأی‌گیری بین سه مدل
        اسنادی که در حداقل دو مدل از سه مدل در نتایج k ظاهر شوند، relevant در نظر گرفته می‌شوند.
        """
        logger.info("Building ground truth using voting among BM25, Semantic, and Hybrid...")
        queries = random.sample(self.dataset, self.sample_size)
        ground_truth = []

        for i, item in enumerate(queries):
            query = item["question"]
            # دریافت نتایج از هر مدل
            bm25_res = {r["index"] for r in self.bm25.search(query, k=self.k)}
            sem_res = {r["index"] for r in self.semantic.search(query, k=self.k)}
            hybrid_res = {r["index"] for r in self.hybrid.search(query, k=self.k)}

            # رأی‌گیری: اسنادی که حداقل در دو مجموعه باشند
            votes = {}
            for idx in bm25_res:
                votes[idx] = votes.get(idx, 0) + 1
            for idx in sem_res:
                votes[idx] = votes.get(idx, 0) + 1
            for idx in hybrid_res:
                votes[idx] = votes.get(idx, 0) + 1

            relevant = [idx for idx, count in votes.items() if count >= 2]
            ground_truth.append({
                "query": query,
                "relevant": relevant
            })

            if (i+1) % 50 == 0:
                logger.info(f"Ground truth progress: {i+1}/{self.sample_size}")

        logger.info(f"Ground truth built with {len(ground_truth)} queries")
        return ground_truth

    def save(self, path="evaluation/ground_truth.json"):
        data = self.build()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Ground truth saved to {path}")
        return data