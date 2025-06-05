from typing import NamedTuple, Any

import numpy as np
from numpy import ndarray, floating


class Metrics(NamedTuple):
    """Metrics for classification tasks."""
    coverage: float
    mean_width: float
    avg_range: floating
    avg_gaps: floating
    avg_ordinal_distance: floating


def get_metrics(
    y_true: ndarray,
    preds: tuple[ndarray, ndarray, float, float],
) -> Metrics:
    _, y_pred_set, coverage, mean_width = preds
    return Metrics(
        coverage,
        mean_width,
        avg_range=calc_avg_range(y_pred_set[:, :, 0]),
        avg_gaps=calc_avg_gaps(y_pred_set[:, :, 0]),
        avg_ordinal_distance=calc_avg_ordinal_distance(y_true, y_pred_set[:, :, 0]),
    )


def get_metrics_across_reps(
    y_true: ndarray,
    rep_preds: list[tuple[ndarray, ndarray, float, float]],
) -> Metrics:
    rep_metrics = [get_metrics(y_true, preds) for preds in rep_preds]
    return Metrics(
        coverage=np.mean([m.coverage for m in rep_metrics]),
        mean_width=np.mean([m.mean_width for m in rep_metrics]),
        avg_range=np.mean([m.avg_range for m in rep_metrics]),
        avg_gaps=np.mean([m.avg_gaps for m in rep_metrics]),
        avg_ordinal_distance=np.mean([m.avg_ordinal_distance for m in rep_metrics]),
    )


def calc_avg_range(
    y_pred_set: ndarray,
) -> floating:
    return np.mean([
        np.ptp(np.flatnonzero(row)) if np.any(row) else 0
        for row in y_pred_set
    ])


def calc_avg_gaps(
    y_pred_set: ndarray,
) -> floating:
    return np.mean([
        np.sum(np.diff(np.flatnonzero(row)) > 1) if np.sum(row) > 1 else 0
        for row in y_pred_set
    ])


def calc_avg_ordinal_distance(
    y_true: ndarray,
    y_pred_set: ndarray,
) -> floating:
    _, n_classes = y_pred_set.shape
    return np.mean([
        np.min(np.abs(np.flatnonzero(row) - y_true[i])) if np.any(row) else n_classes
        for i, row in enumerate(y_pred_set)
    ])
