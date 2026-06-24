import os
import pickle
import hashlib
import numpy as np
from rank_bm25 import BM25Okapi
from preprocessing.cleaner import clean_text
import logging
from multiprocessing import Pool, cpu_count
from functools import partial

logger = logging.getLogger(__name__)

class BM25Retriever:
    def __init__(self, dataset, cache_dir="cache/bm25"):
        self.dataset = dataset
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        dataset_hash = self._hash_dataset()
        self.corpus_file = os.path.join(cache_dir, f"corpus_{dataset_hash}.pkl")
        self.bm25_file = os.path.join(cache_dir, f"bm25_{dataset_hash}.pkl")

        if self._load_from_cache():
            logger.info("BM25 loaded from cache")
        else:
            logger.info("Building BM25 from scratch...")
            self._build_from_scratch()
            self._save_to_cache()
            logger.info("BM25 built and cached")

    def _hash_dataset(self):
        content = str([(item.get("question",""), item.get("answer","")) for item in self.dataset])
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:8]

    def _load_from_cache(self):
        if all(os.path.exists(f) for f in [self.corpus_file, self.bm25_file]):
            try:
                with open(self.corpus_file, "rb") as f:
                    self.corpus = pickle.load(f)
                with open(self.bm25_file, "rb") as f:
                    self.bm25 = pickle.load(f)
                self.tokenized_corpus = [doc.split() for doc in self.corpus]
                return True
            except Exception as e:
                logger.warning(f"Cache loading failed: {e}")
        return False

    def _save_to_cache(self):
        with open(self.corpus_file, "wb") as f:
            pickle.dump(self.corpus, f)
        with open(self.bm25_file, "wb") as f:
            pickle.dump(self.bm25, f)
        logger.info("BM25 cache saved.")

    def _build_from_scratch(self):
        def process_item(item):
            question = str(item.get("question") or "")
            answer = str(item.get("answer") or "")
            category = str(item.get("category") or "")
            return clean_text(question + " " + answer + " " + category)

        logger.info("Building BM25 corpus with multiprocessing...")
        with Pool(processes=max(1, cpu_count() - 1)) as pool:
            self.corpus = pool.map(process_item, self.dataset)

        self.tokenized_corpus = [doc.split() for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, k=5):
        query = clean_text(query)
        tokens = query.split()
        if not tokens:
            return []
        scores = np.array(self.bm25.get_scores(tokens))
        top_indices = np.argsort(scores)[-k:][::-1]
        results = []
        for idx in top_indices:
            item = self.dataset[idx]
            results.append({
                "index": int(idx),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "category": item.get("category", ""),
                "score": float(scores[idx])
            })
        return results