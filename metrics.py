from typing import NamedTuple, Any, Dict
from collections import OrderedDict

import numpy as np
import pandas as pd
from numpy import ndarray
from mapie.metrics.classification import classification_coverage_score, classification_mean_width_score, \
    classification_ssc_score

from util import transpose_typeddict


class FloatMetrics(NamedTuple):
    """Metrics for classification tasks with mean across multiple replications."""
    coverage: float                              # Classification Coverage Score
    mean_width: float                            # Classification Mean Width Score
    mean_range: float                            # Mean Interval Range (Regression Mean Width Score)
    mean_gaps: float                             # Mean Gaps within Prediction Set
    pred_set_mae: float                          # Mean Absolute Error of Prediction Set (Ordinal Distance from True Class)
    accuracy: float                              # Classification Accuracy
    mae: float                                   # Mean Absolute Error
    qwk: float                                   # Quadratic Weighted Kappa
    non_contiguous_percentage: float             # Percentage of Non-Contiguous Prediction Sets
    size_stratified_coverage: Dict[int, float]   # Coverage per Prediction Set Size
    ssc_score: float                             # Worst-case coverage across bins


class CollectedMetrics(NamedTuple):
    """Metrics for classification tasks collected across multiple replications."""
    coverage: list[float]                              # Classification Coverage Score
    mean_width: list[float]                            # Classification Mean Width Score
    mean_range: list[float]                            # Mean Interval Range (Regression Mean Width Score)
    mean_gaps: list[float]                             # Mean Gaps within Prediction Set
    pred_set_mae: list[float]                          # Mean Absolute Error of Prediction Set (Ordinal Distance from True Class)
    accuracy: list[float]                              # Classification Accuracy
    mae: list[float]                                   # Mean Absolute Error
    qwk: list[float]                                   # Quadratic Weighted Kappa
    non_contiguous_percentage: list[float]             # Percentage of Non-Contiguous Prediction Sets
    size_stratified_coverage: list[Dict[int, float]]   # Coverage per Prediction Set Size
    ssc_score: list[float]                             # Worst-case coverage across bins


def get_metrics(
    y_true: ndarray,
    preds: tuple[ndarray, ndarray],
) -> FloatMetrics:
    """Calculate all metrics for a single prediction result."""
    y_pred, y_pred_set = preds
    pred_set = y_pred_set[:, :, 0]

    return FloatMetrics(
        coverage=float(classification_coverage_score(y_true, pred_set)),
        mean_width=float(classification_mean_width_score(pred_set)),
        mean_range=calc_mean_range(pred_set),
        mean_gaps=calc_mean_gaps(pred_set),
        pred_set_mae=calc_pred_set_mae(y_true, pred_set),
        accuracy=calc_accuracy(y_true, y_pred),
        mae=calc_mae(y_true, y_pred),
        qwk=calc_qwk(y_true, y_pred),
        non_contiguous_percentage=calc_non_contiguous_percentage(pred_set),
        size_stratified_coverage=calc_size_stratified_coverage(y_true, pred_set),
        # ssc_score=float(classification_ssc_score(y_true, y_pred_set)),
        ssc_score=0
    )


def get_metrics_across_reps_collect(
    y_true: ndarray,
    rep_preds: list[tuple[ndarray, ndarray]],
) -> CollectedMetrics:
    """Calculate and collect metrics across multiple replications."""
    rep_metrics = [get_metrics(y_true, preds) for preds in rep_preds]
    return CollectedMetrics(
        coverage=[m.coverage for m in rep_metrics],
        mean_width=[m.mean_width for m in rep_metrics],
        mean_range=[m.mean_range for m in rep_metrics],
        mean_gaps=[m.mean_gaps for m in rep_metrics],
        pred_set_mae=[m.pred_set_mae for m in rep_metrics],
        accuracy=[m.accuracy for m in rep_metrics],
        mae=[m.mae for m in rep_metrics],
        qwk=[m.qwk for m in rep_metrics],
        non_contiguous_percentage=[m.non_contiguous_percentage for m in rep_metrics],
        size_stratified_coverage=[m.size_stratified_coverage for m in rep_metrics],
        ssc_score=[m.ssc_score for m in rep_metrics],
    )


def get_metrics_across_reps(
    y_true: ndarray,
    rep_preds: list[tuple[ndarray, ndarray]],
) -> FloatMetrics:
    """Calculate metrics averaged across multiple replications."""
    rep_metrics = get_metrics_across_reps_collect(y_true, rep_preds)
    
    # Aggregate size-stratified coverage across replications
    all_size_coverage = {}
    for metric in rep_metrics.size_stratified_coverage:
        for size, coverage in metric.items():
            if size not in all_size_coverage:
                all_size_coverage[size] = []
            all_size_coverage[size].append(coverage)
    
    avg_size_coverage = {size: float(np.mean(coverages)) for size, coverages in all_size_coverage.items()}
    
    return FloatMetrics(
        coverage=float(np.mean(rep_metrics.coverage)),
        mean_width=float(np.mean(rep_metrics.mean_width)),
        mean_range=float(np.mean(rep_metrics.mean_range)),
        mean_gaps=float(np.mean(rep_metrics.mean_gaps)),
        pred_set_mae=float(np.mean(rep_metrics.pred_set_mae)),
        accuracy=float(np.mean(rep_metrics.accuracy)),
        mae=float(np.mean(rep_metrics.mae)),
        qwk=float(np.mean(rep_metrics.qwk)),
        non_contiguous_percentage=float(np.mean(rep_metrics.non_contiguous_percentage)),
        size_stratified_coverage=avg_size_coverage,
        ssc_score=float(np.mean(rep_metrics.ssc_score)),
    )


def calc_mean_range(
    y_pred_set: ndarray,
) -> float:
    return float(np.mean([
        np.ptp(np.flatnonzero(row)) if np.any(row) else 0
        for row in y_pred_set
    ]))


def calc_mean_gaps(
    y_pred_set: ndarray,
) -> float:
    return float(np.mean([
        np.sum(np.diff(np.flatnonzero(row)) > 1) if np.sum(row) > 1 else 0
        for row in y_pred_set
    ]))


def calc_pred_set_mae(
    y_true: ndarray,
    y_pred_set: ndarray,
) -> float:
    _, n_classes = y_pred_set.shape
    return float(np.mean([
        np.min(np.abs(np.flatnonzero(row) - y_true[i])) if np.any(row) else n_classes
        for i, row in enumerate(y_pred_set)
    ]))


def calc_accuracy(
    y_true: ndarray,
    y_pred: ndarray,
) -> float:
    """Calculate classification accuracy."""
    return float(np.mean(y_true == y_pred))


def calc_mae(
    y_true: ndarray,
    y_pred: ndarray,
) -> float:
    """Calculate Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def calc_qwk(
    y_true: ndarray,
    y_pred: ndarray,
) -> float:
    """Calculate Quadratic Weighted Kappa."""
    from sklearn.metrics import cohen_kappa_score
    
    # Ensure inputs are integers for kappa calculation
    y_true_int = y_true.astype(int)
    y_pred_int = y_pred.astype(int)
    
    # Calculate quadratic weighted kappa
    qwk = cohen_kappa_score(y_true_int, y_pred_int, weights='quadratic')
    return float(qwk)


def calc_non_contiguous_percentage(
    y_pred_set: ndarray,
) -> float:
    """Calculate percentage of non-contiguous prediction sets (sets with gaps)."""
    non_contiguous_count = 0
    total_count = 0
    
    for row in y_pred_set:
        if np.sum(row) > 1:  # Only consider sets with more than one class
            total_count += 1
            if np.sum(np.diff(np.flatnonzero(row)) > 1) > 0:
                non_contiguous_count += 1
    
    return float(non_contiguous_count / total_count) if total_count > 0 else 0.0


def calc_size_stratified_coverage(
    y_true: ndarray,
    y_pred_set: ndarray,
) -> Dict[int, float]:
    """Calculate coverage for each prediction set size."""
    size_coverage = {}
    
    for i, row in enumerate(y_pred_set):
        set_size = int(np.sum(row))
        if set_size not in size_coverage:
            size_coverage[set_size] = {'correct': 0, 'total': 0}
        
        size_coverage[set_size]['total'] += 1
        if row[y_true[i]]:  # Check if true label is in prediction set
            size_coverage[set_size]['correct'] += 1
    
    # Convert to coverage ratios (0-1) and sort by set size
    coverage_dict = {
        size: float(data['correct'] / data['total']) 
        for size, data in size_coverage.items()
    }
    
    # Return ordered dictionary sorted by set size
    return OrderedDict(sorted(coverage_dict.items()))


def get_results_metrics(results: Dict[str, list[tuple[ndarray, ndarray]]], y_test: ndarray, mode: str = 'mean') -> pd.DataFrame:
    if mode not in ['mean', 'collect']:
        raise ValueError(f"Invalid mode: {mode}. Must be 'mean' or 'collect'.")

    metrics_data = []
    for name, pred_results in results.items():
        # Parse the format: model_loss_fn_score_alg_alphaX.XX
        alpha_part = name.split('_alpha')[-1]
        base_name = name.replace(f'_alpha{alpha_part}', '')
        parts = base_name.split('_', 2 if mode == 'mean' else 3)
        model, loss_fn, score_alg = parts[0], parts[1], parts[2]
        alpha = float(alpha_part)

        if mode == 'mean':
            metrics = get_metrics_across_reps(y_test, pred_results)
            metrics_data.append({
                'name': name,
                'model': model,
                'loss_fn': loss_fn,
                'score_alg': score_alg,
                'alpha': alpha,
                **metrics._asdict()
            })
        elif mode == 'collect':
            metrics = transpose_typeddict(get_metrics_across_reps_collect(y_test, pred_results)._asdict())
            for rep, rep_metrics in enumerate(metrics):
                metrics_data.append({
                    'name': f'{name}_rep{rep}',
                    'model': model,
                    'loss_fn': loss_fn,
                    'score_alg': score_alg,
                    'alpha': alpha,
                    'rep': rep,
                    **rep_metrics
                })

    return pd.DataFrame(metrics_data)
