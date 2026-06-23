from retrieval.bm25 import BM25Retriever
from retrieval.semantic import SemanticRetriever
from retrieval.fusion import HybridRetriever

from evaluation.metrics import precision_at_k, recall_at_k


class EvaluationRunner:

    def __init__(self, dataset):

        self.dataset = dataset

        self.bm25 = BM25Retriever(dataset)
        self.semantic = SemanticRetriever(dataset)
        self.hybrid = HybridRetriever(self.bm25, self.semantic)

    def evaluate_model(self, model, ground_truth, k=10):

        all_relevant = []
        all_retrieved = []

        for item in ground_truth:

            query = item["query"]
            relevant = item["relevant"]

            results = model.search(query, k=k)

            retrieved = [r["index"] for r in results]

            all_relevant.append(relevant)
            all_retrieved.append(retrieved)

        return {
            "precision@k": sum([
                precision_at_k(r, p, k)
                for r, p in zip(all_relevant, all_retrieved)
            ]) / len(ground_truth),

            "recall@k": sum([
                recall_at_k(r, p, k)
                for r, p in zip(all_relevant, all_retrieved)
            ]) / len(ground_truth)
        }