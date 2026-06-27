import re
import hazm
from typing import List, Optional

class TextCleaner:
    def __init__(self, remove_stopwords: bool = True, stem: bool = False):
        self.normalizer = hazm.Normalizer()
        self.stopwords = set(hazm.stopwords_list())
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self.stemmer = hazm.Stemmer() if stem else None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize Persian/English text"""
        if not text:
            return ""
        
        text = str(text)
        # Normalize
        text = self.normalizer.normalize(text)
        # Remove extra spaces and punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords if enabled
        if self.remove_stopwords:
            tokens = text.split()
            tokens = [t for t in tokens if t not in self.stopwords]
            text = ' '.join(tokens)
        
        # Stem if enabled
        if self.stem and self.stemmer:
            tokens = text.split()
            tokens = [self.stemmer.stem(t) for t in tokens]
            text = ' '.join(tokens)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        return self.clean_text(text).split()

# Default instance
default_cleaner = TextCleaner(remove_stopwords=True, stem=False)
clean_text = default_cleaner.clean_text
tokenize = default_cleaner.tokenize