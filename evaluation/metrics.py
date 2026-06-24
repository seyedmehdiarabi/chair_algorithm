import numpy as np

def precision_at_k(relevant, retrieved, k):
    if k <= 0:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / k

def recall_at_k(relevant, retrieved, k):
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / len(relevant)

def mean_reciprocal_rank(all_relevant, all_retrieved, k=10):
    rr_sum = 0.0
    for relevant, retrieved in zip(all_relevant, all_retrieved):
        for i, doc in enumerate(retrieved[:k]):
            if doc in relevant:
                rr_sum += 1.0 / (i + 1)
                break
    return rr_sum / len(all_retrieved)

def ndcg_at_k(relevant, retrieved, k):
    def dcg(scores):
        return sum((1 / np.log2(i + 2)) if rel else 0 for i, rel in enumerate(scores))

    rels = [1 if doc in relevant else 0 for doc in retrieved[:k]]
    ideal_rels = sorted(rels, reverse=True)
    dcg_val = dcg(rels)
    idcg_val = dcg(ideal_rels)
    if idcg_val == 0:
        return 0.0
    return dcg_val / idcg_val