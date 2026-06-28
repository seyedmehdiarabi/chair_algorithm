import logging
import re
from typing import List, Set, Optional
from utils.error_handler import handle_errors

logger = logging.getLogger(__name__)


class QueryExpander:
    """
    Query Expansion برای زبان فارسی با:
    - مترادف‌های پایه (پزشکی، عمومی)
    - استخراج از نتایج جستجوی معنایی
    - پشتیبانی از PRF (از طریق ترکیب با prf.py)
    """

    # دیکشنری مترادف‌های فارسی (قابل گسترش)
    SYNONYM_DICT = {
        'بارداری': ['حاملگی', 'بارداری', 'آبستنی', 'حامله'],
        'سرطان': ['تومور', 'نئوپلاسم', 'بدخیمی', 'کانسر', 'کارسینوم'],
        'دیابت': ['قند خون', 'بیماری قند', 'دیابت شیرین'],
        'قلب': ['دل', 'کاردیاک'],
        'فشار خون': ['پرفشاری خون', 'هایپرتانسیون'],
        'سکته': ['استروک', 'سکته مغزی'],
        'عفونت': ['التهاب', 'میکروب', 'باکتری'],
        'درد': ['در', 'ناراحتی', 'سوزش'],
        'درمان': ['مداوا', 'معالجه'],
        'بیماری': ['مرض', 'ناخوشی', 'عارضه'],
        'تیروئید': ['تیرویید', 'غده تیروئید'],
        'کبد': ['جگر'],
        'کلیه': ['گرده'],
        'مغز': ['دماغ'],
        'اعصاب': ['عصب'],
        'پزشک': ['دکتر', 'حکیم'],
        'دارو': ['قرص', 'شربت', 'دواء'],
    }

    def __init__(self, use_wordnet=False):
        self.use_wordnet = use_wordnet
        self.wordnet = None
        self.cache = {}
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self) -> Set[str]:
        return {
            'و', 'به', 'از', 'با', 'برای', 'در', 'را', 'که', 'چه', 'چی',
            'چگونه', 'چرا', 'کی', 'کجا', 'آیا', 'خواهم', 'خواهی', 'خواهد',
            'باشم', 'باشی', 'باشد', 'هستم', 'هستی', 'است', 'ایم', 'اید',
            'اند', 'شد', 'شو', 'شده', 'کرد', 'کن', 'کند', 'ده', 'دهد',
            'گیر', 'گیرد', 'آورد', 'آورده', 'مثل', 'مانند', 'بر', 'روی',
            'هر', 'همه', 'هم', 'نیز', 'همچنین'
        }

    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r'[،؛؟!٪"\'،،\(\)\[\]\{\}]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return [t for t in text.split() if len(t) > 1]

    def _get_synonyms(self, word: str) -> Set[str]:
        """دریافت مترادف‌ها از دیکشنری"""
        if word in self.cache:
            return self.cache[word]

        synonyms = set()
        word_lower = word.lower()

        for key, values in self.SYNONYM_DICT.items():
            if word_lower in key or key in word_lower:
                synonyms.update(values)
                synonyms.add(key)

        self.cache[word] = synonyms
        return synonyms

    @handle_errors
    def expand_with_synonyms(self, query: str, max_terms: int = 5) -> List[str]:
        """گسترش کوئری با مترادف‌های فارسی"""
        if not query or len(query.strip()) < 2:
            return [query]

        tokens = self._tokenize(query)
        expanded_queries = set()
        expanded_queries.add(query)

        # حذف stopwords و گسترش
        for token in tokens:
            if token in self.stopwords:
                continue
            synonyms = self._get_synonyms(token)
            for syn in list(synonyms)[:3]:
                if syn != token and len(syn) > 1:
                    new_query = query.replace(token, syn, 1)
                    if new_query != query:
                        expanded_queries.add(new_query)
                    expanded_queries.add(syn)

        result = list(expanded_queries)[:max_terms]
        result = [q for q in result if len(q.strip().split()) >= 1 and len(q) > 2]

        if not result:
            return [query]

        logger.info(f"Synonym expansion: {query} -> {result[:3]}")
        return result

    @handle_errors
    def expand_with_embeddings(self, query: str, semantic_retriever, top_k: int = 3) -> List[str]:
        """گسترش با استفاده از نتایج جستجوی معنایی"""
        try:
            results = semantic_retriever.search(query, k=top_k)
            terms = []
            for r in results:
                q = r.get('question', '')
                if q:
                    terms.extend(self._tokenize(q)[:5])
            if terms:
                expanded = list(set([query] + terms))[:5]
                logger.info(f"Embedding expansion: {query} -> {expanded[:3]}")
                return expanded
            return [query]
        except Exception as e:
            logger.warning(f"Embedding expansion failed: {e}")
            return [query]

    @handle_errors
    def expand(self, query: str, semantic_retriever=None, max_terms: int = 5) -> List[str]:
        """گسترش ترکیبی با مترادف‌ها و embedding"""
        if not query or len(query.strip()) < 2:
            return [query]

        all_expansions = set()
        all_expansions.add(query)

        # ۱. مترادف‌ها
        syn_exp = self.expand_with_synonyms(query, max_terms)
        all_expansions.update(syn_exp)

        # ۲. Embedding (اگر retriever موجود باشد)
        if semantic_retriever:
            emb_exp = self.expand_with_embeddings(query, semantic_retriever, top_k=3)
            all_expansions.update(emb_exp)

        result = list(all_expansions)[:max_terms]
        result = [q for q in result if len(q.strip()) > 2]

        if not result:
            return [query]

        return result


default_expander = QueryExpander()