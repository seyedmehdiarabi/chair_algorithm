import json
import random
import logging
import torch
from sentence_transformers import CrossEncoder
from retrieval.bm25 import BM25Retriever
from retrieval.semantic import SemanticRetriever
from retrieval.fusion import HybridRetriever

logger = logging.getLogger(__name__)

class GroundTruthBuilder:
    def __init__(self, dataset, sample_size=300, k=10, threshold=0.5):
        self.dataset = dataset
        self.sample_size = min(sample_size, len(dataset))
        self.k = k
        self.threshold = threshold
        
        # مدل‌های اصلی برای استخراج کاندیداها (Pooling)
        self.bm25 = BM25Retriever(dataset)
        self.semantic = SemanticRetriever(dataset)
        self.hybrid = HybridRetriever(self.bm25, self.semantic)
        
        # استفاده از Cross-Encoder به عنوان داور
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cross_encoder = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2',
            max_length=512,
            device=self.device
        )
        logger.info(f"Cross-Encoder loaded successfully on {self.device}")

    def build(self):
        logger.info(f"Building ground truth using Cross-Encoder scoring (Pooling method)...")
        logger.info(f"Sample size: {self.sample_size}, k: {self.k}, threshold: {self.threshold}")
        
        # برای تکرارپذیری
        random.seed(42)
        
        # نمونه‌گیری تصادفی
        queries = random.sample(self.dataset, self.sample_size)
        ground_truth = []

        for i, item in enumerate(queries):
            query = item.get("question", "")
            if not query:
                continue
                
            # 1. Pooling: گرفتن نتایج از همه مدل‌ها
            candidate_indices = set()
            
            try:
                for r in self.bm25.search(query, k=self.k):
                    candidate_indices.add(r["index"])
                for r in self.semantic.search(query, k=self.k):
                    candidate_indices.add(r["index"])
                for r in self.hybrid.search(query, k=self.k):
                    candidate_indices.add(r["index"])
            except Exception as e:
                logger.warning(f"Error retrieving candidates for query {i}: {e}")
                continue

            if not candidate_indices:
                ground_truth.append({"query": query, "relevant": []})
                continue

            # 2. ساخت لیست جفت‌ها برای Cross-Encoder
            pairs = []
            indices_list = list(candidate_indices)
            
            for idx in indices_list:
                doc = self.dataset[idx]
                doc_text = f"{doc.get('question', '')} {doc.get('answer', '')}"
                # محدود کردن طول متن برای جلوگیری از overflow
                if len(doc_text) > 1000:
                    doc_text = doc_text[:1000]
                pairs.append([query, doc_text])

            # 3. امتیازدهی با Cross-Encoder با batch processing
            try:
                scores = self.cross_encoder.predict(pairs, convert_to_numpy=True, batch_size=32)
            except Exception as e:
                logger.warning(f"Cross-Encoder prediction failed for query {i}: {e}")
                continue

            # 4. انتخاب اسناد با امتیاز > threshold
            relevant_indices = []
            for idx, score in zip(indices_list, scores):
                if score > self.threshold:
                    relevant_indices.append(int(idx))

            ground_truth.append({
                "query": query,
                "relevant": relevant_indices
            })

            if (i + 1) % 50 == 0:
                logger.info(f"Ground truth progress: {i+1}/{self.sample_size}")

        logger.info(f"Ground truth built with {len(ground_truth)} queries.")
        return ground_truth

    def save(self, path="evaluation/ground_truth.json"):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = self.build()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Ground truth saved to {path}")
        return data