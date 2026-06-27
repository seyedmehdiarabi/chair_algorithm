import logging
from typing import List, Optional
from utils.error_handler import handle_errors

logger = logging.getLogger(__name__)

class QueryExpander:
    """Query expansion using synonyms and embeddings"""
    
    def __init__(self, use_wordnet=True):
        self.use_wordnet = use_wordnet
        self.wordnet = None
        if use_wordnet:
            try:
                import nltk
                nltk.download('wordnet', quiet=True)
                from nltk.corpus import wordnet
                self.wordnet = wordnet
                logger.info("WordNet loaded for query expansion")
            except ImportError:
                logger.warning("NLTK not installed. WordNet expansion disabled.")
                self.use_wordnet = False
    
    @handle_errors
    def expand_with_synonyms(self, query: str, max_terms: int = 5) -> List[str]:
        """Expand query with synonyms from WordNet"""
        if not self.use_wordnet or not self.wordnet:
            return [query]
        
        tokens = query.split()
        expanded = set([query])
        
        for token in tokens:
            synsets = self.wordnet.synsets(token)
            for synset in synsets[:2]:
                for lemma in synset.lemmas()[:3]:
                    if lemma.name() != token:
                        expanded.add(lemma.name().replace('_', ' '))
                        if len(expanded) >= max_terms + 1:
                            return list(expanded)
        
        return list(expanded)
    
    @handle_errors
    def expand_with_embeddings(self, query: str, semantic_retriever, top_k: int = 3) -> List[str]:
        """Expand query using top retrieved documents' content"""
        try:
            results = semantic_retriever.search(query, k=top_k)
            terms = []
            for r in results:
                q = r.get('question', '')
                if q:
                    terms.extend(q.split()[:5])
            if terms:
                return list(set([query] + terms))
            return [query]
        except Exception as e:
            logger.warning(f"Embedding-based expansion failed: {e}")
            return [query]