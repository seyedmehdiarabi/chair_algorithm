from evaluation.runner import EvaluationRunner
from evaluation.build_ground_truth import GroundTruthBuilder


def run_full_evaluation(dataset):

    gt_builder = GroundTruthBuilder(dataset)
    ground_truth = gt_builder.build(sample_size=300)

    evaluator = EvaluationRunner(dataset)

    print("\nEvaluating BM25...")
    bm25_scores = evaluator.evaluate_model(evaluator.bm25, ground_truth)

    print("\nEvaluating Semantic...")
    semantic_scores = evaluator.evaluate_model(evaluator.semantic, ground_truth)

    print("\nEvaluating Hybrid...")
    hybrid_scores = evaluator.evaluate_model(evaluator.hybrid, ground_truth)

    print("\n===== FINAL RESULTS =====")
    print("BM25:", bm25_scores)
    print("Semantic:", semantic_scores)
    print("Hybrid:", hybrid_scores)