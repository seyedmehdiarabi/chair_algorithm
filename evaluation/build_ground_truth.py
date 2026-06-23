import json
import random
from retrieval.bm25 import BM25Retriever


class GroundTruthBuilder:

    def __init__(self, dataset):
        self.dataset = dataset
        self.bm25 = BM25Retriever(dataset)

    def build(self, sample_size=500, k=10):

        print("Building pseudo ground truth...")

        queries = random.sample(self.dataset, sample_size)

        dataset_size = len(self.dataset)

        ground_truth = []

        for i, item in enumerate(queries):

            query = item["question"]

            results = self.bm25.search(query, k=k)

            relevant_ids = [r["index"] for r in results]

            ground_truth.append({
                "query": query,
                "relevant": relevant_ids
            })

            if i % 50 == 0:
                print(f"Processed: {i}/{sample_size}")

        return ground_truth

    def save(self, path="evaluation/ground_truth.json"):

        data = self.build()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print("Saved ground truth ✔")