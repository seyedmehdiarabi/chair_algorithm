from evaluation.runner import EvaluationRunner
from evaluation.build_ground_truth import GroundTruthBuilder
import logging

logger = logging.getLogger(__name__)

def run_full_evaluation(dataset, sample_size=300, k=10):
    """
    اجرای کامل ارزیابی با ساخت ground truth از روش voting
    """
    logger.info("Starting full evaluation...")
    gt_builder = GroundTruthBuilder(dataset, sample_size=sample_size, k=k)
    ground_truth = gt_builder.build()

    evaluator = EvaluationRunner(dataset)

    logger.info("Evaluating BM25...")
    bm25_scores = evaluator.evaluate_model(evaluator.bm25, ground_truth, k=k)

    logger.info("Evaluating Semantic...")
    semantic_scores = evaluator.evaluate_model(evaluator.semantic, ground_truth, k=k)

    logger.info("Evaluating Hybrid...")
    hybrid_scores = evaluator.evaluate_model(evaluator.hybrid, ground_truth, k=k)

    logger.info("===== FINAL RESULTS =====")
    logger.info(f"BM25:     {bm25_scores}")
    logger.info(f"Semantic: {semantic_scores}")
    logger.info(f"Hybrid:   {hybrid_scores}")

    return {
        "BM25": bm25_scores,
        "Semantic": semantic_scores,
        "Hybrid": hybrid_scores
    }