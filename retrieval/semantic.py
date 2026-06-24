import os
import pickle
import hashlib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
from preprocessing.cleaner import clean_text
import logging

logger = logging.getLogger(__name__)

class SemanticRetriever:
    def __init__(self, dataset, cache_dir="cache/semantic",
                 model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 index_type="hnsw"):  # "hnsw" یا "flat"
        self.dataset = dataset
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.index_type = index_type
        os.makedirs(cache_dir, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

        dataset_hash = self._hash_dataset()
        self.embeddings_file = os.path.join(cache_dir, f"embeddings_{dataset_hash}.npy")
        self.index_file = os.path.join(cache_dir, f"faiss_{dataset_hash}.index")
        self.documents_file = os.path.join(cache_dir, f"documents_{dataset_hash}.pkl")
        self.model_info_file = os.path.join(cache_dir, f"model_{dataset_hash}.txt")
        self.index_type_file = os.path.join(cache_dir, f"index_type_{dataset_hash}.txt")

        if self._load_cache():
            logger.info("Semantic cache loaded")
        else:
            logger.info("Building semantic index...")
            self._build()
            self._save_cache()
            logger.info("Semantic index built and cached")

    def _hash_dataset(self):
        content = str([(item.get("question",""), item.get("answer","")) for item in self.dataset])
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:8]

    def _load_cache(self):
        files = [self.embeddings_file, self.index_file, self.documents_file]
        if not all(os.path.exists(f) for f in files):
            return False
        try:
            self.documents = pickle.load(open(self.documents_file, "rb"))
            self.embeddings = np.load(self.embeddings_file)
            self.index = faiss.read_index(self.index_file)
            # اگر GPU فعال است، ایندکس را به GPU منتقل کنیم
            if self.device == "cuda" and hasattr(faiss, 'index_cpu_to_gpu'):
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            # بررسی نوع ایندکس از فایل ذخیره شده
            if os.path.exists(self.index_type_file):
                with open(self.index_type_file, "r") as f:
                    saved_type = f.read().strip()
                    if saved_type != self.index_type:
                        logger.warning(f"Index type mismatch: cache has {saved_type}, requested {self.index_type}. Rebuilding...")
                        return False
            return True
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            return False

    def _save_cache(self):
        # اگر ایندکس روی GPU است، برای ذخیره به CPU منتقل می‌کنیم
        if self.device == "cuda" and hasattr(self.index, 'get_device'):
            # Faiss GPU index را به CPU تبدیل می‌کنیم
            if self.index.__class__.__name__.startswith('Gpu'):
                self.index = faiss.index_gpu_to_cpu(self.index)
        np.save(self.embeddings_file, self.embeddings)
        faiss.write_index(self.index, self.index_file)
        pickle.dump(self.documents, open(self.documents_file, "wb"))
        with open(self.model_info_file, "w") as f:
            f.write(self.model_name)
        with open(self.index_type_file, "w") as f:
            f.write(self.index_type)

    def _build(self):
        self.documents = []
        for i, item in enumerate(self.dataset):
            if i % 1000 == 0:
                logger.info(f"Semantic preprocessing: {i}/{len(self.dataset)}")
            question = str(item.get("question", ""))
            category = str(item.get("category", ""))
            specialty = str(item.get("specialty", ""))
            text = clean_text(question + " " + category + " " + specialty)
            self.documents.append(text)

        logger.info("Encoding documents with SentenceTransformer...")
        self.embeddings = self.model.encode(
            self.documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=512,          # افزایش یافته برای سرعت بیشتر
            show_progress_bar=True
        )
        dimension = self.embeddings.shape[1]

        # ساخت ایندکس بر اساس نوع انتخاب شده
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(dimension)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 512)
            self.index.train(self.embeddings)
        else:  # hnsw (پیش‌فرض)
            self.index = faiss.IndexHNSWFlat(dimension, 64)   # M=64
            self.index.hnsw.efConstruction = 200              # دقت ساخت بالا

        # انتقال به GPU در صورت وجود و پشتیبانی
        if self.device == "cuda" and hasattr(faiss, 'index_cpu_to_gpu'):
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("FAISS index moved to GPU")
            except Exception as e:
                logger.warning(f"Could not move index to GPU: {e}")

        self.index.add(self.embeddings)

    def search(self, query, k=5, expand_query=True, efSearch=128):
        """
        جستجوی معنایی
        """
        query = clean_text(query)
        if expand_query and len(query.split()) <= 2:
            query += " پزشکی درمان بیماری"
        q_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # تنظیم پارامتر جستجوی HNSW
        if self.index_type == "hnsw" and hasattr(self.index, 'hnsw'):
            self.index.hnsw.efSearch = efSearch

        scores, ids = self.index.search(q_embedding, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self.dataset):
                continue
            item = self.dataset[idx]
            results.append({
                "index": int(idx),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "category": item.get("category", ""),
                "text": self.documents[idx],
                "score": float(score)
            })
        return results