"""
Threshold optimization module for Business Entity Resolution.
Directly optimizes probability threshold against macro-averaged F_0.5 scorer.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

_src_dir = Path(__file__).resolve().parent
_code_dir = _src_dir.parents[1]
for _p in [str(_src_dir), str(_code_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from business_entity_resolution.src.scorer import evaluate_predictions
except ImportError:
    from scorer import evaluate_predictions


def optimize_threshold(
    pair_predictions: List[Tuple[str, str, float]],
    ground_truth: Dict[str, Set[str]],
    all_s1_ids: Set[str],
    country_map: Optional[Dict[str, str]] = None,
    threshold_range: Tuple[float, float, float] = (0.20, 0.90, 0.05),
) -> Tuple[float, Dict[str, any], List[Dict[str, float]]]:
    """
    Grid-search probability threshold to maximize macro F_0.5.

    Parameters:
    -----------
    pair_predictions: list of (s1_id, candidate_id, probability)
    ground_truth: dict mapping s1_id -> set of true matched entity IDs
    all_s1_ids: set of all S1 IDs being evaluated (including singletons)
    threshold_range: (min_t, max_t, step)

    Returns:
    --------
    best_threshold, best_metrics, search_history
    """
    # Group candidate predictions by s1_id: s1_id -> list of (cand_id, prob)
    s1_candidates: Dict[str, List[Tuple[str, float]]] = {
        s1_id: [] for s1_id in all_s1_ids
    }
    for s1_id, cand_id, prob in pair_predictions:
        if s1_id in s1_candidates:
            s1_candidates[s1_id].append((cand_id, prob))

    min_t, max_t, step = threshold_range
    thresholds = np.arange(min_t, max_t + 1e-6, step)

    best_threshold = 0.50
    best_f05 = -1.0
    best_metrics = {}
    history = []

    print("\n--- Running Precision-Oriented Threshold Optimization (Macro F_0.5) ---")
    for t in thresholds:
        t_val = round(float(t), 3)
        preds: Dict[str, Set[str]] = {}
        for s1_id, cands in s1_candidates.items():
            matched = {cid for cid, p in cands if p >= t_val}
            preds[s1_id] = matched

        metrics = evaluate_predictions(ground_truth, preds, all_s1_ids, country_map)
        f05 = metrics["macro_f05"]
        prec = metrics["macro_precision"]
        rec = metrics["macro_recall"]

        history.append({
            "threshold": t_val,
            "macro_f05": f05,
            "precision": prec,
            "recall": rec,
            "singleton_f05": metrics["singleton_f05"],
            "non_singleton_f05": metrics["non_singleton_f05"],
        })

        if f05 > best_f05:
            best_f05 = f05
            best_threshold = t_val
            best_metrics = metrics

        print(f"  Threshold {t_val:.2f} -> Macro F_0.5: {f05:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | Sing: {metrics['singleton_f05']:.4f}")

    print(f"\n[OPTIMAL THRESHOLD FOUND]: T* = {best_threshold:.2f} (Macro F_0.5 = {best_f05:.4f})")
    return best_threshold, best_metrics, history
