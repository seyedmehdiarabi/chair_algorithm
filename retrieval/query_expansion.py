import logging
import re
import os
import numpy as np
from typing import List, Set, Optional
from utils.error_handler import handle_errors

logger = logging.getLogger(__name__)

class QueryExpander:
    """
    Query Expansion خودکار با ترکیب سه روش:
    ۱. مترادف‌های پایه (اختیاری - برای کلمات کلیدی)
    ۲. مدل FastText فارسی (برای کلمات جدید)
    ۳. استخراج از نتایج جستجوی معنایی (تطبیق با دیتاست)
    """
    
    # ===== مترادف‌های پایه (فقط برای کلمات حیاتی - قابل حذف) =====
    BASE_SYNONYMS = {
        'بارداری': ['حاملگی', 'بارداری'],
        'سرطان': ['تومور', 'نئوپلاسم', 'بدخیمی', 'کانسر'],
        'دیابت': ['قند خون', 'بیماری قند'],
        'قلب': ['دل', 'کاردیاک'],
        'فشار خون': ['پرفشاری خون', 'هایپرتانسیون'],
    }
    
    def __init__(self, fasttext_path=None, use_base=True, use_fasttext=True, use_semantic=True):
        """
        Args:
            fasttext_path: مسیر مدل FastText (اگر None باشد، از embedding استفاده نمی‌کند)
            use_base: استفاده از مترادف‌های پایه
            use_fasttext: استفاده از FastText
            use_semantic: استفاده از نتایج جستجوی معنایی
        """
        self.use_base = use_base
        self.use_fasttext = use_fasttext and fasttext_path is not None
        self.use_semantic = use_semantic
        self.fasttext_model = None
        self.cache = {}
        
        # بارگذاری FastText
        if self.use_fasttext and fasttext_path and os.path.exists(fasttext_path):
            try:
                import fasttext
                self.fasttext_model = fasttext.load_model(fasttext_path)
                logger.info(f"✅ FastText loaded from {fasttext_path}")
            except Exception as e:
                logger.warning(f"⚠️ FastText load failed: {e}")
                self.use_fasttext = False
        else:
            if self.use_fasttext:
                logger.warning("⚠️ FastText path not found. Disabling FastText.")
                self.use_fasttext = False
        
        # Stopwords فارسی
        self.stopwords = self._load_stopwords()
        logger.info(f"🚀 QueryExpander ready (fasttext={self.use_fasttext}, base={self.use_base}, semantic={self.use_semantic})")
    
    def _load_stopwords(self) -> Set[str]:
        return {
            'و', 'به', 'از', 'با', 'برای', 'در', 'را', 'که', 'چه', 'چی',
            'چگونه', 'چرا', 'کی', 'کجا', 'آیا', 'خواهم', 'خواهی', 'خواهد',
            'باشم', 'باشی', 'باشد', 'هستم', 'هستی', 'است', 'ایم', 'اید',
            'اند', 'شد', 'شو', 'شده', 'کرد', 'کن', 'کند', 'ده', 'دهد',
            'گیر', 'گیرد', 'آورد', 'آورده', 'مثل', 'مانند', 'بر', 'روی'
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """تقسیم‌بندی هوشمند فارسی"""
        text = re.sub(r'[،؛؟!٪"\'،،\(\)\[\]\{\}]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return [t for t in text.split() if len(t) > 1]
    
    def _get_base_synonyms(self, word: str) -> Set[str]:
        """مترادف‌های پایه"""
        synonyms = set()
        if not self.use_base:
            return synonyms
        
        word_lower = word.lower()
        for key, values in self.BASE_SYNONYMS.items():
            if word_lower in key or key in word_lower:
                synonyms.update(values)
                synonyms.add(key)
                # اضافه کردن مترادف‌های دیگر
                for v in values:
                    if v in self.BASE_SYNONYMS:
                        synonyms.update(self.BASE_SYNONYMS[v])
        return synonyms
    
    def _get_fasttext_synonyms(self, word: str, top_k: int = 5) -> Set[str]:
        """مترادف‌ها از FastText (کاملاً خودکار)"""
        synonyms = set()
        if not self.use_fasttext or self.fasttext_model is None:
            return synonyms
        
        try:
            # FastText روش استاندارد برای یافتن نزدیک‌ترین همسایه‌ها
            # در نسخه‌های مختلف FastText، نام متد ممکن است متفاوت باشد
            if hasattr(self.fasttext_model, 'get_nearest_neighbors'):
                neighbors = self.fasttext_model.get_nearest_neighbors(word, top_k)
                for score, neighbor in neighbors:
                    if score > 0.5 and neighbor != word and len(neighbor) > 2:
                        synonyms.add(neighbor)
            elif hasattr(self.fasttext_model, 'nearest_neighbors'):
                neighbors = self.fasttext_model.nearest_neighbors(word, top_k)
                for neighbor in neighbors:
                    if neighbor != word and len(neighbor) > 2:
                        synonyms.add(neighbor)
            else:
                # روش جایگزین: استفاده از cosine similarity روی بردارها
                vec = self.fasttext_model.get_word_vector(word)
                if np.all(vec == 0):
                    return synonyms
                
                # این بخش نیاز به دیکشنری کامل کلمات دارد که معمولاً در دسترس نیست
                # بنابراین بهتر است از متدهای توکار استفاده شود
                logger.debug(f"FastText nearest neighbors method not available for '{word}'")
        except Exception as e:
            logger.debug(f"FastText error for '{word}': {e}")
        
        return synonyms
    
    def _get_semantic_synonyms(self, word: str, semantic_retriever, top_k: int = 3) -> Set[str]:
        """استخراج کلمات مرتبط از نتایج جستجوی معنایی"""
        synonyms = set()
        if not self.use_semantic or semantic_retriever is None:
            return synonyms
        
        try:
            results = semantic_retriever.search(word, k=top_k)
            for r in results:
                # از فیلدهای مختلف استخراج
                texts = [
                    r.get('question', ''),
                    r.get('answer', ''),
                    r.get('specialty', ''),
                    r.get('category', '')
                ]
                for text in texts:
                    if text:
                        tokens = self._tokenize(text)
                        for token in tokens[:5]:  # محدود کردن
                            if token not in self.stopwords and len(token) > 2 and token != word:
                                synonyms.add(token)
        except Exception as e:
            logger.debug(f"Semantic synonym error: {e}")
        
        return synonyms
    
    def get_synonyms(self, word: str, semantic_retriever=None) -> List[str]:
        """دریافت مترادف‌ها از همه منابع (با کش)"""
        if word in self.cache:
            return self.cache[word]
        
        synonyms = set()
        synonyms.add(word)  # خود کلمه همیشه حضور دارد
        
        # مرحله ۱: مترادف‌های پایه
        synonyms.update(self._get_base_synonyms(word))
        
        # مرحله ۲: FastText
        synonyms.update(self._get_fasttext_synonyms(word))
        
        # مرحله ۳: Semantic
        if semantic_retriever:
            synonyms.update(self._get_semantic_synonyms(word, semantic_retriever))
        
        # فیلتر کردن
        result = [s for s in synonyms if len(s) > 1 and s not in self.stopwords]
        
        # اگر هیچی پیدا نشد، حداقل خود کلمه
        if not result:
            result = [word]
        
        # محدود کردن به ۵ مورد (مرتب‌شده بر اساس طول)
        result = sorted(set(result), key=lambda x: (len(x), x))[:5]
        self.cache[word] = result
        return result
    
    @handle_errors
    def expand(self, query: str, semantic_retriever=None, max_terms: int = 5) -> List[str]:
        """
        گسترش کامل کوئری
        
        Args:
            query: کوئری ورودی
            semantic_retriever: برای استخراج از نتایج معنایی
            max_terms: حداکثر تعداد کوئری‌های خروجی
        
        Returns:
            لیست کوئری‌های گسترش‌یافته
        """
        if not query or len(query.strip()) < 2:
            return [query]
        
        # نرم‌السازی
        query = re.sub(r'[^\w\s]', ' ', query)
        query = re.sub(r'\s+', ' ', query).strip()
        
        tokens = self._tokenize(query)
        if not tokens:
            return [query]
        
        # گرفتن مترادف‌ها برای هر کلمه
        all_synonyms = {}
        for token in tokens[:3]:  # فقط ۳ کلمه اول برای جلوگیری از انفجار
            all_synonyms[token] = self.get_synonyms(token, semantic_retriever)
        
        # تولید کوئری‌های جدید
        expanded_queries = set()
        expanded_queries.add(query)  # کوئری اصلی
        
        # ترکیب مترادف‌ها
        for token, synonyms in all_synonyms.items():
            for syn in synonyms[:2]:  # هر کلمه ۲ مترادف
                if syn != token:
                    new_query = query.replace(token, syn, 1)
                    if new_query != query:
                        expanded_queries.add(new_query)
                    # کوئری فقط با مترادف
                    expanded_queries.add(syn)
        
        # ترکیب دو مترادف با هم
        tokens_list = list(all_synonyms.keys())
        if len(tokens_list) >= 2:
            for i in range(len(tokens_list)):
                for j in range(i+1, len(tokens_list)):
                    syn_i = all_synonyms[tokens_list[i]][:1]
                    syn_j = all_synonyms[tokens_list[j]][:1]
                    if syn_i and syn_j:
                        combined = f"{syn_i[0]} {syn_j[0]}"
                        if len(combined) > 2:
                            expanded_queries.add(combined)
        
        # مرتب‌سازی و محدود کردن
        result = sorted(list(expanded_queries), key=len)[:max_terms]
        
        # حذف کوئری‌های تکراری یا خیلی کوتاه
        final_result = []
        seen = set()
        for q in result:
            q_clean = re.sub(r'\s+', ' ', q).strip()
            if q_clean not in seen and len(q_clean) > 2:
                seen.add(q_clean)
                final_result.append(q_clean)
        
        if not final_result:
            return [query]
        
        logger.info(f"📝 Expanded: '{query}' -> {final_result[:3]}")
        return final_result[:max_terms]


# ===== نمونه برای استفاده =====
# ایجاد یک نمونه با تنظیمات پیش‌فرض
# (مسیر FastText را به‌روز کنید)
default_expander = QueryExpander(
    fasttext_path="cc.fa.300.bin",  # مسیر مدل FastText
    use_base=True,
    use_fasttext=True,
    use_semantic=True
)