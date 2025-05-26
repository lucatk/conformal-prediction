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
