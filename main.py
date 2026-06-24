import json
import os
import time
import argparse
import logging
from retrieval.bm25 import BM25Retriever
from retrieval.semantic import SemanticRetriever
from retrieval.fusion import HybridRetriever

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Medical Retrieval System")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after loading")
    parser.add_argument("--data-size", type=int, default=None, help="Number of samples to use (None = all)")
    parser.add_argument("--query", type=str, help="Single query to test (if not in interactive mode)")
    parser.add_argument("--no-interactive", action="store_true", help="Skip interactive query input")
    parser.add_argument("--index-type", choices=["flat", "ivf", "hnsw"], default="hnsw", 
                        help="FAISS index type (hnsw recommended for large datasets)")
    parser.add_argument("--no-reranker", action="store_true", 
                        help="Disable reranker for faster search (recommended for production)")
    parser.add_argument("--use-reranker", action="store_true", 
                        help="Enable reranker (slower but potentially more accurate)")
    args = parser.parse_args()

    # تعیین استفاده از reranker
    use_reranker = args.use_reranker and not args.no_reranker  # در صورت تناقض، no-reranker اولویت دارد

    logger.info("=" * 50)
    logger.info("MAIN STARTED")
    logger.info("=" * 50)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "data", "cahir_train.json")

    logger.info(f"Loading data from: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Full data size: {len(data)}")

    if args.data_size is not None and args.data_size < len(data):
        working_data = data[:args.data_size]
    else:
        working_data = data
    logger.info(f"Working data size: {len(working_data)}")

    start_time = time.time()
    logger.info("Initializing BM25...")
    bm25 = BM25Retriever(working_data, cache_dir="cache/bm25")

    logger.info("Initializing Semantic...")
    semantic = SemanticRetriever(
        working_data,
        cache_dir="cache/semantic",
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        index_type=args.index_type
    )

    logger.info("Initializing Hybrid...")
    hybrid = HybridRetriever(bm25, semantic)

    logger.info(f"Initialization time: {time.time() - start_time:.2f} seconds")

    if args.evaluate:
        from evaluation.compare_models import run_full_evaluation
        run_full_evaluation(working_data, sample_size=min(300, len(working_data)//2))
        if not args.query and args.no_interactive:
            return

    if args.query:
        queries = [args.query]
    elif not args.no_interactive:
        query = input("\nEnter your query (or 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            return
        queries = [query]
    else:
        return

    for query in queries:
        logger.info(f"\nQuery: {query}")
        logger.info("=" * 50)

        # BM25
        logger.info("\n========== BM25 RESULTS ==========")
        start = time.time()
        bm25_res = bm25.search(query, k=5)
        logger.info(f"Search time: {time.time()-start:.4f}s")
        for i, r in enumerate(bm25_res, 1):
            print(f"{i}. Q: {r['question'][:100]}...")
            print(f"   A: {r['answer'][:100]}...")
            print(f"   Cat: {r.get('category','N/A')}  Score: {r['score']:.4f}")

        # Semantic
        logger.info("\n========== SEMANTIC RESULTS ==========")
        start = time.time()
        sem_res = semantic.search(query, k=5)
        logger.info(f"Search time: {time.time()-start:.4f}s")
        for i, r in enumerate(sem_res, 1):
            print(f"{i}. Q: {r['question'][:100]}...")
            print(f"   A: {r['answer'][:100]}...")
            print(f"   Cat: {r.get('category','N/A')}  Score: {r['score']:.4f}")

        # Hybrid RRF (با یا بدون Reranker)
        logger.info(f"\n========== HYBRID (RRF{' with Reranker' if use_reranker else ''}) ==========")
        start = time.time()
        hybrid_res = hybrid.search(query, k=5, use_reranker=use_reranker)
        logger.info(f"Search time: {time.time()-start:.4f}s")
        for i, r in enumerate(hybrid_res, 1):
            score_key = "rerank_score" if use_reranker else "fusion_score"
            print(f"{i}. Q: {r['question'][:100]}...")
            print(f"   A: {r['answer'][:100]}...")
            print(f"   Cat: {r.get('category','N/A')}  Score: {r.get(score_key,0):.4f}")

        # Weighted Hybrid
        logger.info("\n========== HYBRID (Weighted) ==========")
        start = time.time()
        weighted = hybrid.search_with_weights(query, k=5, alpha=0.6)
        logger.info(f"Search time: {time.time()-start:.4f}s")
        for i, r in enumerate(weighted, 1):
            print(f"{i}. Q: {r['question'][:100]}...")
            print(f"   A: {r['answer'][:100]}...")
            print(f"   Cat: {r.get('category','N/A')}")
            print(f"   Final: {r['final_score']:.4f}  BM25: {r['bm25_score']:.4f}  Sem: {r['semantic_score']:.4f}")

    logger.info("\n" + "=" * 50)
    logger.info("PROCESS COMPLETED")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()