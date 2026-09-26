"""
Evaluation metrics for the Amazon ML Challenge 2026: Business Entity Resolution.
Evaluates macro-averaged F_0.5 score across all Source 1 entities including singletons.
"""

from typing import Dict, List, Set, Union


def compute_entity_f05(
    predicted_ids: Union[List[str], Set[str]],
    ground_truth_ids: Union[List[str], Set[str]],
) -> float:
    """
    Computes F_0.5 score for a single Source 1 entity.
    
    Formula:
      F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
      
    Singletons:
      - If ground truth is empty:
        - returns 1.0 if predicted is empty
        - returns 0.0 if predicted is non-empty
      - If ground truth is non-empty and predicted is empty:
        - returns 0.0
    """
    pred_set = set(predicted_ids) if not isinstance(predicted_ids, set) else predicted_ids
    gt_set = set(ground_truth_ids) if not isinstance(ground_truth_ids, set) else ground_truth_ids

    # Ground truth has no matches (Singleton)
    if len(gt_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0

    # Ground truth has matches, but model predicted nothing
    if len(pred_set) == 0:
        return 0.0

    tp = len(pred_set & gt_set)
    if tp == 0:
        return 0.0

    precision = tp / len(pred_set)
    recall = tp / len(gt_set)

    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0

    return (1.25 * precision * recall) / denom


def compute_macro_f05(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
) -> float:
    """
    Computes macro-average F_0.5 score across all Source 1 entities.
    
    Args:
      predictions: dict mapping source1_entity_id -> list of matched IDs (S2/S3)
      ground_truth: dict mapping source1_entity_id -> list of true matched IDs (S2/S3)
      
    Returns:
      Macro F_0.5 score (float between 0.0 and 1.0)
    """
    total_score = 0.0
    num_entities = len(ground_truth)

    if num_entities == 0:
        return 0.0

    for s1_id, gt_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, [])
        score = compute_entity_f05(pred_matches, gt_matches)
        total_score += score

    return total_score / num_entities
