import os
import pickle
import hashlib
import numpy as np
from rank_bm25 import BM25Okapi
from preprocessing.cleaner import clean_text
from utils.error_handler import handle_errors, log_execution_time, CacheError
from utils.memory import chunk_list
import logging
from multiprocessing import Pool, cpu_count

logger = logging.getLogger(__name__)

def _process_item_for_bm25(item):
    """پردازش یک آیتم برای ساخت corpus BM25 با اضافه کردن specialty و question_type"""
    question = str(item.get("question", ""))
    answer = str(item.get("answer", ""))
    category = str(item.get("category", ""))
    specialty = str(item.get("specialty", ""))          # <-- اضافه شد
    question_type = str(item.get("question_type", ""))  # <-- اضافه شد (اختیاری)
    # ترکیب تمام فیلدهای متنی
    combined = f"{question} {answer} {category} {specialty} {question_type}"
    return clean_text(combined)

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
        content = str([(item.get("question",""), item.get("answer",""), 
                        item.get("specialty",""), item.get("question_type","")) 
                       for item in self.dataset])
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
    
    @log_execution_time
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
        return False
    
    def _save_to_cache(self):
        try:
            with open(self.corpus_file, "wb") as f:
                pickle.dump(self.corpus, f)
            with open(self.bm25_file, "wb") as f:
                pickle.dump(self.bm25, f)
            logger.info("BM25 cache saved.")
        except Exception as e:
            raise CacheError(f"Failed to save BM25 cache: {e}")
    
    @log_execution_time
    def _build_from_scratch(self):
        logger.info("Building BM25 corpus with multiprocessing...")
        num_workers = max(1, cpu_count() - 1)
        chunks = list(chunk_list(self.dataset, max(1000, len(self.dataset)//num_workers)))
        
        corpus = []
        with Pool(processes=num_workers) as pool:
            for chunk_result in pool.map(self._process_chunk, chunks):
                corpus.extend(chunk_result)
        
        self.corpus = corpus
        self.tokenized_corpus = [doc.split() for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    @staticmethod
    def _process_chunk(chunk):
        return [_process_item_for_bm25(item) for item in chunk]
    
    @handle_errors
    @log_execution_time
    def search(self, query, k=5):
        query = clean_text(query)
        tokens = query.split()
        if not tokens:
            logger.warning("Empty query after cleaning")
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
                "specialty": item.get("specialty", ""), 
                "score": float(scores[idx])
            })
        return results