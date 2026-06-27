import unittest
import json
from retrieval.bm25 import BM25Retriever
from retrieval.semantic import SemanticRetriever
from retrieval.fusion import HybridRetriever

class TestRetrievers(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.dataset = [
            {"question": "What is Python?", "answer": "A programming language", "category": "Programming"},
            {"question": "What is AI?", "answer": "Artificial Intelligence", "category": "Tech"},
            {"question": "What is machine learning?", "answer": "A subset of AI", "category": "Tech"},
        ]
        cls.bm25 = BM25Retriever(cls.dataset, cache_dir="test_cache/bm25")
        cls.semantic = SemanticRetriever(cls.dataset, cache_dir="test_cache/semantic")
        cls.hybrid = HybridRetriever(cls.bm25, cls.semantic)
    
    def test_bm25_search(self):
        results = self.bm25.search("Python", k=2)
        self.assertEqual(len(results), 2)
        self.assertTrue(any("Python" in r['question'] for r in results))
    
    def test_semantic_search(self):
        results = self.semantic.search("programming language", k=2)
        self.assertEqual(len(results), 2)
    
    def test_hybrid_search(self):
        results = self.hybrid.search("AI", k=2)
        self.assertEqual(len(results), 2)
    
    def test_empty_query(self):
        results = self.bm25.search("", k=5)
        self.assertEqual(results, [])
    
    def test_k_larger_than_dataset(self):
        results = self.bm25.search("Python", k=10)
        self.assertLessEqual(len(results), len(self.dataset))

if __name__ == '__main__':
    unittest.main()