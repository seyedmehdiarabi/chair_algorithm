import numpy as np


# =========================
# Precision@K
# =========================
def precision_at_k(relevant, retrieved, k):

    retrieved_k = retrieved[:k]

    if len(retrieved_k) == 0:
        return 0.0

    hits = 0

    for doc in retrieved_k:
        if doc in relevant:
            hits += 1

    return hits / k


# =========================
# Recall@K
# =========================
def recall_at_k(relevant, retrieved, k):

    if len(relevant) == 0:
        return 0.0

    retrieved_k = retrieved[:k]

    hits = 0

    for doc in retrieved_k:
        if doc in relevant:
            hits += 1

    return hits / len(relevant)


# =========================
# MRR
# =========================
def mean_reciprocal_rank(all_relevant_lists, all_retrieved_lists, k=10):

    rr_sum = 0.0

    for relevant, retrieved in zip(all_relevant_lists, all_retrieved_lists):

        rr = 0.0

        for i, doc in enumerate(retrieved[:k]):

            if doc in relevant:
                rr = 1 / (i + 1)
                break

        rr_sum += rr

    return rr_sum / len(all_retrieved_lists)


# =========================
# nDCG@K
# =========================
def ndcg_at_k(relevant, retrieved, k):

    def dcg(scores):
        return sum([
            (1 / np.log2(i + 2)) if rel else 0
            for i, rel in enumerate(scores)
        ])

    rels = [1 if doc in relevant else 0 for doc in retrieved[:k]]

    ideal_rels = sorted(rels, reverse=True)

    dcg_val = dcg(rels)
    idcg_val = dcg(ideal_rels)

    if idcg_val == 0:
        return 0.0

    return dcg_val / idcg_val