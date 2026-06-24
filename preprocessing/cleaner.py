import re
import hazm
from typing import List

class TextCleaner:
    def __init__(self):
        self.normalizer = hazm.Normalizer()
        self.stopwords = set(hazm.stopwords_list())
        # اضافه کردن کلمات سفارشی در صورت نیاز

    def clean_text(self, text: str, remove_stopwords: bool = True, stem: bool = False) -> str:
        """
        پاکسازی و نرمال‌سازی متن
        """
        if not text:
            return ""
        text = str(text)
        # نرمال‌سازی
        text = self.normalizer.normalize(text)
        # حذف کاراکترهای اضافی
        text = re.sub(r'[^\w\s]', ' ', text)  # حذف علائم نگارشی
        text = re.sub(r'\s+', ' ', text).strip()  # حذف فاصله‌های اضافی
        # حذف اعداد (اختیاری)
        # text = re.sub(r'\d+', '', text)
        if remove_stopwords:
            tokens = text.split()
            tokens = [t for t in tokens if t not in self.stopwords]
            text = ' '.join(tokens)
        if stem:
            stemmer = hazm.Stemmer()
            tokens = text.split()
            tokens = [stemmer.stem(t) for t in tokens]
            text = ' '.join(tokens)
        return text

    def tokenize(self, text: str) -> List[str]:
        return self.clean_text(text).split()

# نمونه آبجکت برای استفاده در سایر ماژول‌ها
default_cleaner = TextCleaner()
clean_text = default_cleaner.clean_text
tokenize = default_cleaner.tokenize