import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from preprocessing.cleaner import clean_text


class SemanticRetriever:
    def __init__(self, dataset, cache_dir="cache", model_name=None):
        """
        Semantic Retriever با قابلیت کش

        Args:
            dataset: لیست دیکشنری‌های داده
            cache_dir: مسیر پوشه کش
            model_name: نام مدل (اختیاری)
        """
        print("SEMANTIC INIT START")
        self.dataset = dataset
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # تنظیم مدل (پیش‌فرض با امکان تغییر به مدل سبکتر)
        self.model_name = model_name or "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        # برای سرعت بیشتر می‌توانید از مدل سبکتر استفاده کنید:
        # self.model_name = "distiluse-base-multilingual-cased-v2"

        # نام فایل‌های کش
        self.embeddings_file = os.path.join(cache_dir, "embeddings.npy")
        self.index_file = os.path.join(cache_dir, "faiss_index.bin")
        self.documents_file = os.path.join(cache_dir, "documents.pkl")
        self.model_info_file = os.path.join(cache_dir, "model_info.txt")

        # بارگذاری از کش یا ساخت از ابتدا
        if self._load_from_cache():
            print("Loaded embeddings and FAISS index from cache")
        else:
            self._build_from_scratch()
            self._save_to_cache()

        # مدل را فقط در صورت نیاز بارگذاری می‌کنیم (در search)
        self.model = None
        print("SEMANTIC READY ✔")

    def _load_from_cache(self):
        """بارگذاری داده‌های کش شده"""
        files_exist = all(os.path.exists(f) for f in [
            self.embeddings_file, self.index_file, self.documents_file
        ])

        if not files_exist:
            return False

        try:
            # بررسی تطابق مدل
            if os.path.exists(self.model_info_file):
                with open(self.model_info_file, "r") as f:
                    cached_model = f.read().strip()
                if cached_model != self.model_name:
                    print(f"Model mismatch: cached={cached_model}, current={self.model_name}")
                    return False

            self.documents = pickle.load(open(self.documents_file, "rb"))
            self.embeddings = np.load(self.embeddings_file)
            self.index = faiss.read_index(self.index_file)
            return True
        except Exception as e:
            print(f"Cache loading failed: {e}")
            return False

    def _save_to_cache(self):
        """ذخیره‌سازی داده‌ها در کش"""
        np.save(self.embeddings_file, self.embeddings)
        faiss.write_index(self.index, self.index_file)
        with open(self.documents_file, "wb") as f:
            pickle.dump(self.documents, f)
        with open(self.model_info_file, "w") as f:
            f.write(self.model_name)
        print("Semantic cache saved.")

    def _build_from_scratch(self):
        """محاسبه از ابتدا (بدون کش)"""
        print(f"Building embeddings from scratch with model: {self.model_name}")
        model = SentenceTransformer(self.model_name)

        # اگر GPU در دسترس است:
        # model = SentenceTransformer(self.model_name, device='cuda')

        self.documents = []
        for i, item in enumerate(self.dataset):
            if i % 1000 == 0:
                print(f"Preparing: {i}")
            question = str(item.get("question") or "")
            answer = str(item.get("answer") or "")
            category = str(item.get("category") or "")
            text = clean_text(question + " " + answer + " " + category)
            self.documents.append(text)

        print("Encoding documents...")
        self.embeddings = model.encode(
            self.documents,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32  # افزایش سرعت با بچ‌های بزرگتر
        )

        print("Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)

        # استفاده از اینکس IP برای سرعت بیشتر در جستجو
        # self.index = faiss.IndexFlatIP(dimension)

        self.index.add(np.array(self.embeddings).astype('float32'))

    def search(self, query, k=5):
        """
        جستجوی معنایی

        Args:
            query: متن جستجو
            k: تعداد نتایج

        Returns:
            list: لیست نتایج
        """
        query = clean_text(query)

        # بارگذاری مدل فقط یک بار (در اولین جستجو)
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
            # اگر GPU در دسترس است:
            # self.model = SentenceTransformer(self.model_name, device='cuda')

        q_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, ids = self.index.search(q_embedding.astype('float32'), k)

        results = []
        for idx, dist in zip(ids[0], distances[0]):
            item = self.dataset[idx]
            results.append({
                "index": idx,
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "category": item.get("category", ""),
                "distance": float(dist)
            })
        return results