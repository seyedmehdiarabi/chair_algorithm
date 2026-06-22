from hazm import Normalizer
import re

normalizer = Normalizer()


def clean_text(text):
    if not text:
        return ""

    # 1. نرمال سازی فارسی (ی، ک، نیم فاصله و ...)
    text = normalizer.normalize(text)

    # 2. کوچک سازی
    text = text.lower()

    # 3. حذف فاصله‌های اضافی
    text = re.sub(r"\s+", " ", text).strip()

    return text