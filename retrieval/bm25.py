import os
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from preprocessing.cleaner import clean_text


class BM25Retriever:
    def __init__(self, dataset, cache_dir="cache"):
        """
        BM25 Retriever با قابلیت کش

        Args:
            dataset: لیست دیکشنری‌های داده
            cache_dir: مسیر پوشه کش
        """
        print("BM25 INIT START")
        self.dataset = dataset
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # نام فایل‌های کش
        self.corpus_file = os.path.join(cache_dir, "bm25_corpus.pkl")
        self.bm25_file = os.path.join(cache_dir, "bm25_model.pkl")

        # بارگذاری از کش یا ساخت از ابتدا
        if self._load_from_cache():
            print("BM25 loaded from cache")
        else:
            self._build_from_scratch()
            self._save_to_cache()

        print("BM25 READY ✔")

    def _load_from_cache(self):
        """بارگذاری BM25 از کش"""
        if all(os.path.exists(f) for f in [self.corpus_file, self.bm25_file]):
            try:
                with open(self.corpus_file, "rb") as f:
                    self.corpus = pickle.load(f)
                with open(self.bm25_file, "rb") as f:
                    self.bm25 = pickle.load(f)
                self.tokenized_corpus = [doc.split() for doc in self.corpus]
                return True
            except Exception as e:
                print(f"Cache loading failed: {e}")
                return False
        return False

    def _save_to_cache(self):
        """ذخیره BM25 در کش"""
        with open(self.corpus_file, "wb") as f:
            pickle.dump(self.corpus, f)
        with open(self.bm25_file, "wb") as f:
            pickle.dump(self.bm25, f)
        print("BM25 cache saved.")

    def _build_from_scratch(self):
        """ساخت BM25 از ابتدا"""
        print("Building BM25 from scratch...")
        self.corpus = []

        for i, item in enumerate(self.dataset):
            if i % 1000 == 0:
                print(f"Processing: {i}")

            question = str(item.get("question") or "")
            answer = str(item.get("answer") or "")
            category = str(item.get("category") or "")

            # ترکیب سوال، پاسخ و دسته‌بندی
            text = clean_text(question + " " + answer + " " + category)
            self.corpus.append(text)

        print("CORPUS BUILT")
        self.tokenized_corpus = [doc.split() for doc in self.corpus]
        print("TOKENIZATION DONE")
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, k=5):
        """
        جستجو با BM25

        Args:
            query: متن جستجو
            k: تعداد نتایج

        Returns:
            list: لیست نتایج
        """
        query = clean_text(query)
        tokens = query.split()
        scores = np.array(self.bm25.get_scores(tokens))
        top_indices = np.argsort(scores)[-k:][::-1]

        results = []
        for idx in top_indices:
            item = self.dataset[idx]
            results.append({
                "index": idx,
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "category": item.get("category", ""),
                "score": float(scores[idx])
            })
        return results