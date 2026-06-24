import os
import pickle
import hashlib
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from preprocessing.cleaner import clean_text
import logging

logger = logging.getLogger(__name__)

class ContextAwareReranker:
    def __init__(self, dataset, cache_dir="cache/reranker"):
        self.dataset = dataset
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
        self.model = AutoModel.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
        self.model.eval()

        # کش کردن embedding اسناد برای سرعت بیشتر
        dataset_hash = self._hash_dataset()
        self.doc_emb_file = os.path.join(cache_dir, f"doc_emb_{dataset_hash}.pkl")
        self.doc_emb = self._load_or_build_embeddings()
        logger.info("Reranker ready")

    def _hash_dataset(self):
        content = str([(item.get("question",""), item.get("answer","")) for item in self.dataset])
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:8]

    def _load_or_build_embeddings(self):
        if os.path.exists(self.doc_emb_file):
            try:
                with open(self.doc_emb_file, "rb") as f:
                    return pickle.load(f)
            except:
                pass
        logger.info("Building document embeddings for reranker...")
        embeddings = []
        for i, item in enumerate(self.dataset):
            if i % 1000 == 0:
                logger.info(f"Reranker embedding: {i}/{len(self.dataset)}")
            doc = clean_text(
                str(item.get("question", "")) + " " +
                str(item.get("answer", "")) + " " +
                str(item.get("category", ""))
            )
            emb = self.encode(doc)
            embeddings.append(emb)
        with open(self.doc_emb_file, "wb") as f:
            pickle.dump(embeddings, f)
        return embeddings

    def encode(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def score(self, query, doc_idx):
        """امتیازدهی به یک سند با ایندکس مشخص"""
        query_emb = self.encode(query)
        doc_emb = self.doc_emb[doc_idx]
        similarity = F.cosine_similarity(query_emb, doc_emb)
        return float(similarity.item())

    def rerank(self, query, candidates):
        """
        candidates: لیست دیکشنری‌های دارای کلید 'index'
        بازگشت: لیست رتبه‌بندی‌شده با کلید 'rerank_score'
        """
        query_emb = self.encode(query)
        reranked = []
        for item in candidates:
            idx = item["index"]
            doc_emb = self.doc_emb[idx]
            sim = F.cosine_similarity(query_emb, doc_emb).item()
            item["rerank_score"] = sim
            reranked.append(item)
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked