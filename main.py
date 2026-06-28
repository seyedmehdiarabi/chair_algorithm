import json
import os
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List
import torch
from retrieval.bm25 import BM25Retriever
from retrieval.semantic import SemanticRetriever
from retrieval.fusion import HybridRetriever
from retrieval.query_expansion import QueryExpander
from retrieval.qa_extractor import PersianQAE
from retrieval.prf import PseudoRelevanceFeedback
from utils.logger import get_result_logger
from utils.data_loader import DatasetLoader
from utils.error_handler import DataError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_dataset_files(data_dir="data"):
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    return sorted([f for f in data_path.glob("*.json") if f.is_file()])


def select_dataset_file(data_dir="data"):
    files = find_dataset_files(data_dir)
    if not files:
        logger.error(f"No JSON files found in '{data_dir}' directory.")
        return None
    if len(files) == 1:
        selected = files[0]
        logger.info(f"Only one dataset found: {selected.name}")
        return str(selected)
    print("\n📂 Available datasets:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f.name}")
    while True:
        try:
            choice = input(f"\nSelect dataset (1-{len(files)}): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(files):
                    return str(files[idx])
            print("Invalid choice. Please try again.")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None


def get_score(item: Dict[str, Any]) -> float:
    """استخراج امتیاز از یک آیتم نتیجه"""
    return item.get("score", item.get("fusion_score", item.get("final_score", 0.0)))


def select_best_answer(results_dict: Dict[str, List[Dict]], query: str) -> Dict[str, Any]:
    """
    انتخاب بهترین پاسخ با اولویت تطابق دقیق یا نزدیک با کوئری.
    اگر سوالی دقیقاً برابر با query باشد، آن را برمی‌گرداند.
    در غیر این صورت، بهترین پاسخ را بر اساس امتیاز روش‌ها (با اولویت hybrid_weighted) انتخاب می‌کند.
    """
    # لیست روش‌ها به ترتیب اولویت
    method_priority = ['hybrid_weighted', 'hybrid_rrf', 'semantic', 'bm25']
    
    # جمع‌آوری همه اسناد از همه روش‌ها
    all_candidates = []
    for method in method_priority:
        if method in results_dict and results_dict[method]:
            for doc in results_dict[method]:
                all_candidates.append({
                    'doc': doc,
                    'method': method,
                    'question': doc.get('question', ''),
                    'answer': doc.get('answer', '')
                })
    
    if not all_candidates:
        return {
            'method': 'none',
            'question': query,
            'answer': 'پاسخی یافت نشد.',
            'score': 0.0,
            'category': 'N/A',
            'index': -1
        }
    
    # ۱. جستجوی تطابق کامل (case-insensitive)
    query_lower = query.lower().strip()
    for cand in all_candidates:
        if cand['question'].lower().strip() == query_lower:
            return {
                'method': cand['method'],
                'question': cand['question'],
                'answer': cand['answer'],
                'score': get_score(cand['doc']),
                'category': cand['doc'].get('category', 'N/A'),
                'index': cand['doc'].get('index', -1)
            }
    
    # ۲. جستجوی تطابق جزئی: بیشترین تعداد کلمات مشترک بین question و query
    query_tokens = set(query_lower.split())
    best_cand = None
    best_match_score = -1
    best_method_priority = -1
    
    for cand in all_candidates:
        if not cand['answer']:
            continue
        question_tokens = set(cand['question'].lower().split())
        # تعداد کلمات مشترک
        common = len(query_tokens.intersection(question_tokens))
        # امتیاز تطابق نرمال‌شده بر اساس طول query
        match_score = common / max(len(query_tokens), 1)
        # اولویت روش را نیز در نظر بگیریم
        method_prio = method_priority.index(cand['method']) if cand['method'] in method_priority else 999
        # ترکیب امتیاز تطابق و اولویت روش: ابتدا تطابق، سپس اولویت
        if match_score > best_match_score or (match_score == best_match_score and method_prio < best_method_priority):
            best_match_score = match_score
            best_method_priority = method_prio
            best_cand = cand
    
    if best_cand and best_match_score > 0:
        return {
            'method': best_cand['method'],
            'question': best_cand['question'],
            'answer': best_cand['answer'],
            'score': get_score(best_cand['doc']),
            'category': best_cand['doc'].get('category', 'N/A'),
            'index': best_cand['doc'].get('index', -1)
        }
    
    # ۳. در غیر این صورت، از روش اول (hybrid_weighted) بهترین سند را انتخاب کن
    for method in method_priority:
        if method in results_dict and results_dict[method]:
            best_doc = max(results_dict[method], key=lambda x: get_score(x))
            if best_doc.get('answer'):
                return {
                    'method': method,
                    'question': best_doc.get('question', ''),
                    'answer': best_doc.get('answer', ''),
                    'score': get_score(best_doc),
                    'category': best_doc.get('category', 'N/A'),
                    'index': best_doc.get('index', -1)
                }
    
    # اگر هیچ پاسخی پیدا نشد
    return {
        'method': 'none',
        'question': query,
        'answer': 'پاسخی یافت نشد.',
        'score': 0.0,
        'category': 'N/A',
        'index': -1
    }


def main():
    parser = argparse.ArgumentParser(description="Intelligent Retrieval System")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after loading")
    parser.add_argument("--data-size", type=int, default=None, help="Number of samples to use (None = all)")
    parser.add_argument("--query", type=str, help="Single query to test (if not in interactive mode)")
    parser.add_argument("--no-interactive", action="store_true", help="Skip interactive query input")
    parser.add_argument("--index-type", choices=["flat", "ivf", "hnsw"], default="hnsw",
                        help="FAISS index type")
    parser.add_argument("--no-reranker", action="store_true", help="Disable reranker")
    parser.add_argument("--use-reranker", action="store_true", help="Enable reranker")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--data-file", type=str, default=None, help="Path to dataset file")
    parser.add_argument("--no-qa", action="store_true", help="Disable extractive QA")
    parser.add_argument("--no-expansion", action="store_true", help="Disable query expansion")
    parser.add_argument("--use-prf", action="store_true", help="Enable Pseudo-Relevance Feedback")
    args = parser.parse_args()

    use_reranker = args.use_reranker and not args.no_reranker
    use_qa = not args.no_qa
    use_expansion = not args.no_expansion
    use_prf = args.use_prf

    logger.info("=" * 50)
    logger.info("INTELLIGENT RETRIEVAL SYSTEM STARTED")
    logger.info("=" * 50)

    result_logger = get_result_logger()
    logger.info(f"Results will be saved to: {result_logger.text_file}")

    # ----- انتخاب فایل دیتاست -----
    data_file = args.data_file
    if data_file is None:
        data_file = select_dataset_file("data")
        if data_file is None:
            return
    else:
        if not Path(data_file).exists():
            logger.error(f"Specified data file not found: {data_file}")
            return

    # ----- بارگذاری دیتاست -----
    try:
        data = DatasetLoader.load(data_file)
        if data is None:
            logger.error(f"Failed to load dataset from {data_file}: loader returned None")
            return
        logger.info(f"Loaded {len(data)} samples from {data_file}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    if args.data_size is not None and args.data_size < len(data):
        data = data[:args.data_size]
    logger.info(f"Working data size: {len(data)}")

    # ----- مقداردهی اولیه مدل‌ها -----
    start_time = time.time()
    try:
        logger.info("Initializing BM25...")
        bm25 = BM25Retriever(data, cache_dir="cache/bm25")

        logger.info("Initializing Semantic Retriever...")
        semantic = SemanticRetriever(
            data,
            cache_dir="cache/semantic",
            model_name="HooshvareLab/bert-fa-base-uncased",
            index_type=args.index_type
        )

        logger.info("Initializing Hybrid Retriever...")
        hybrid = HybridRetriever(bm25, semantic)

        logger.info("Initializing Query Expander...")
        expander = QueryExpander()

        if use_prf:
            logger.info("Initializing PRF...")
            prf = PseudoRelevanceFeedback(bm25, top_k=5, expansion_terms=3)
        else:
            prf = None

        # QA Extractor (تلاش برای بارگذاری، در صورت خطا غیرفعال می‌شود)
        if use_qa:
            logger.info("Attempting to load QA Extractor...")
            try:
                qa_extractor = PersianQAE(
                    model_name="HooshvareLab/bert-fa-base-uncased-pquad",
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                if not qa_extractor.is_available():
                    logger.warning("QA model unavailable. Disabling QA.")
                    use_qa = False
                    qa_extractor = None
                else:
                    logger.info("✅ QA Extractor ready.")
            except Exception as e:
                logger.warning(f"QA loading failed: {e}. Disabling QA.")
                use_qa = False
                qa_extractor = None
        else:
            qa_extractor = None
            logger.info("QA Extractor disabled.")

        logger.info(f"Initialization time: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        return

    # ----- ارزیابی (اختیاری) -----
    if args.evaluate:
        try:
            from evaluation.compare_models import run_full_evaluation
            results = run_full_evaluation(data, sample_size=min(300, len(data)//2))
            logger.info(f"Evaluation results: {results}")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")

    # ----- دریافت کوئری -----
    if args.query:
        queries = [args.query]
    else:
        print("\n" + "=" * 50)
        print("INTERACTIVE MODE - Enter your query (type 'exit' to quit)")
        print("=" * 50)
        try:
            query = input("\nEnter your query: ").strip()
            if not query:
                logger.warning("Empty query received. Exiting...")
                return
            if query.lower() == 'exit':
                logger.info("User requested exit")
                return
            queries = [query]
        except KeyboardInterrupt:
            logger.info("User interrupted")
            return
        except EOFError:
            logger.error("No input available")
            return

    # ----- پردازش هر کوئری -----
    for query in queries:
        logger.info(f"\nQuery: {query}")
        logger.info("=" * 50)

        # ---- ۱. Query Expansion ----
        expanded_queries = [query]
        if use_expansion:
            try:
                expanded_queries = expander.expand(query, semantic_retriever=semantic, max_terms=5)
                logger.info(f"Expanded queries: {expanded_queries}")
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}")
                expanded_queries = [query]

        # ---- ۲. PRF (Pseudo-Relevance Feedback) ----
        if use_prf and prf and expanded_queries:
            # PRF را روی اولین کوئری گسترش‌یافته اعمال می‌کنیم
            try:
                expanded_query = prf.expand_query(expanded_queries[0])
                if expanded_query != expanded_queries[0]:
                    expanded_queries.insert(0, expanded_query)
                logger.info(f"PRF expanded: {expanded_query}")
            except Exception as e:
                logger.warning(f"PRF failed: {e}")

        # ---- ۳. جستجو برای هر کوئری گسترش‌یافته ----
        results_dict = {}
        all_bm25 = []
        all_semantic = []
        all_hybrid_rrf = []
        all_hybrid_weighted = []
        seen_bm25 = set()
        seen_semantic = set()
        seen_rrf = set()
        seen_weighted = set()

        for exp_query in expanded_queries:
            # BM25
            bm25_res = bm25.search(exp_query, k=5)
            for r in bm25_res:
                if r['index'] not in seen_bm25:
                    seen_bm25.add(r['index'])
                    all_bm25.append(r)

            # Semantic
            sem_res = semantic.search(exp_query, k=5)
            for r in sem_res:
                if r['index'] not in seen_semantic:
                    seen_semantic.add(r['index'])
                    all_semantic.append(r)

            # Hybrid RRF
            hybrid_res = hybrid.search(exp_query, k=5, use_reranker=use_reranker)
            for r in hybrid_res:
                if r['index'] not in seen_rrf:
                    seen_rrf.add(r['index'])
                    all_hybrid_rrf.append(r)

            # Weighted Hybrid
            weighted = hybrid.search_with_weights(exp_query, k=5, alpha=0.6)
            for r in weighted:
                if r['index'] not in seen_weighted:
                    seen_weighted.add(r['index'])
                    all_hybrid_weighted.append(r)

        # مرتب‌سازی بر اساس امتیاز و محدود کردن به ۵ نتیجه
        all_bm25.sort(key=lambda x: x['score'], reverse=True)
        all_semantic.sort(key=lambda x: x['score'], reverse=True)
        all_hybrid_rrf.sort(key=lambda x: x.get('fusion_score', 0), reverse=True)
        all_hybrid_weighted.sort(key=lambda x: x.get('final_score', 0), reverse=True)

        results_dict = {
            "bm25": all_bm25[:5],
            "semantic": all_semantic[:5],
            "hybrid_rrf": all_hybrid_rrf[:5],
            "hybrid_weighted": all_hybrid_weighted[:5]
        }

        # ---- ۴. استخراج پاسخ با QA (در صورت فعال بودن) ----
        if use_qa and qa_extractor and qa_extractor.is_available():
            logger.info("Extracting answers with QA...")
            for method in ['hybrid_rrf', 'hybrid_weighted']:
                if method in results_dict and results_dict[method]:
                    for item in results_dict[method]:
                        context = item.get('context', '')
                        if not context:
                            context = item.get('question', '') + " " + item.get('answer', '')
                        if context:
                            ans, score = qa_extractor.extract_answer(query, context)
                            item['extracted_answer'] = ans
                            item['qa_score'] = score

        # ---- ۵. انتخاب بهترین پاسخ ----
        best_answer = select_best_answer(results_dict, query)
        results_dict["best_answer"] = [best_answer]  # ذخیره در خروجی JSON

        # ---- ۶. نمایش نتایج ----
        print("\n" + "=" * 60)
        print("BM25 RESULTS")
        print("=" * 60)
        for i, r in enumerate(results_dict["bm25"], 1):
            print(f"{i}. Q: {r['question'][:80]}...")
            print(f"   A: {r['answer'][:80]}...")
            print(f"   Cat: {r.get('category','N/A')} Score: {r['score']:.4f}\n")

        print("\n" + "=" * 60)
        print("SEMANTIC RESULTS")
        print("=" * 60)
        for i, r in enumerate(results_dict["semantic"], 1):
            print(f"{i}. Q: {r['question'][:80]}...")
            print(f"   A: {r['answer'][:80]}...")
            print(f"   Cat: {r.get('category','N/A')} Score: {r['score']:.4f}\n")

        print("\n" + "=" * 60)
        print(f"HYBRID (RRF{' with Reranker' if use_reranker else ''}) RESULTS")
        print("=" * 60)
        score_key = "rerank_score" if use_reranker else "fusion_score"
        for i, r in enumerate(results_dict["hybrid_rrf"], 1):
            print(f"{i}. Q: {r['question'][:80]}...")
            print(f"   A: {r['answer'][:80]}...")
            print(f"   Cat: {r.get('category','N/A')} Score: {r.get(score_key,0):.4f}")
            if 'extracted_answer' in r and r['extracted_answer']:
                print(f"   ✅ Extracted: {r['extracted_answer'][:80]}... (QA: {r['qa_score']:.4f})")
            print()

        print("\n" + "=" * 60)
        print("HYBRID (Weighted) RESULTS")
        print("=" * 60)
        for i, r in enumerate(results_dict["hybrid_weighted"], 1):
            print(f"{i}. Q: {r['question'][:80]}...")
            print(f"   A: {r['answer'][:80]}...")
            print(f"   Cat: {r.get('category','N/A')}")
            print(f"   Final: {r['final_score']:.4f} BM25: {r['bm25_score']:.4f} Sem: {r['semantic_score']:.4f}")
            if 'extracted_answer' in r and r['extracted_answer']:
                print(f"   ✅ Extracted: {r['extracted_answer'][:80]}... (QA: {r['qa_score']:.4f})")
            print()

        # ---- ۷. نمایش بهترین پاسخ ----
        print("\n" + "=" * 60)
        print("🏆 BEST ANSWER")
        print("=" * 60)
        print(f"✅ Answer: {best_answer['answer']}")
        print(f"   Method: {best_answer['method'].upper().replace('_', ' ')}")
        print(f"   Score: {best_answer['score']:.4f}")
        print(f"   Category: {best_answer['category']}")
        if best_answer['question'] != query:
            print(f"   (from question: {best_answer['question'][:60]}...)")
        print("=" * 60 + "\n")

        # ذخیره نتایج
        result_logger.log_query(query, results_dict)
        logger.info(f"Results saved to: {result_logger.text_file}")

    logger.info("PROCESS COMPLETED")
    logger.info(f"All results saved to: {result_logger.text_file}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()