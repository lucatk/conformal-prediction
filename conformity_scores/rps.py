from typing import Optional

from mapie._machine_precision import EPSILON
from mapie.conformity_scores.classification import BaseClassificationScore
from mapie.conformity_scores.sets.utils import check_proba_normalized
from mapie.estimator.classifier import EnsembleClassifier
import numpy as np
from numpy.typing import NDArray


class RPSConformityScore(BaseClassificationScore):
    """
    Ranked Probability Score (RPS) based conformity score for ordinal conformity_scores.

    Attributes
    ----------
    classes: Optional[ArrayLike]
        Names of the classes.

    random_state: Optional[Union[int, RandomState]]
        Pseudo random number generator state.

    quantiles_: ArrayLike of shape (n_alpha,)
        The quantiles estimated from ``get_sets`` method.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_conformity_scores(
        self,
        y: NDArray,
        y_pred: NDArray,
        **kwargs
    ) -> NDArray:
        """
        Calculate the RPS conformity score for each prediction.

        Parameters
        ----------
        y: NDArray of shape (n_samples,)
            True target labels (as integers).

        y_pred: NDArray of shape (n_samples, n_classes)
            Predicted probabilities per class.

        Returns
        -------
        NDArray of shape (n_samples,)
            RPS conformity scores.
        """
        n_samples, n_classes = y_pred.shape
        conformity_scores = np.empty(n_samples, dtype="float")

        for i in range(n_samples):
            probs = y_pred[i]
            outcome = np.zeros(n_classes)
            outcome[y[i]] = 1
            cum_probs = np.cumsum(probs)
            cum_outcome = np.cumsum(outcome)
            rps = np.sum((cum_probs - cum_outcome) ** 2) / (n_classes - 1)
            conformity_scores[i] = rps

        return conformity_scores

    def get_predictions(
        self,
        X: NDArray,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        agg_scores: Optional[str] = "mean",
        **kwargs
    ) -> NDArray:
        """
        Predict class probabilities and replicate for each alpha value.

        Parameters
        ----------
        X: NDArray of shape (n_samples, n_features)
        alpha_np: NDArray of shape (n_alpha,)
        estimator: EnsembleClassifier
        agg_scores: Optional[str] Method to aggregate the scores from the base estimators. By default ``"mean"``.

        Returns
        -------
        NDArray of shape (n_samples, n_classes, n_alpha)
        """
        y_pred_proba = estimator.predict(X, agg_scores='mean')
        y_pred_proba = check_proba_normalized(y_pred_proba, axis=1)
        if agg_scores != "crossval":
            y_pred_proba = np.repeat(
                y_pred_proba[:, :, np.newaxis], len(alpha_np), axis=2
            )

        return y_pred_proba

    def get_conformity_score_quantiles(
        self,
        conformity_scores: NDArray,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        **kwargs
    ) -> NDArray:
        """
        Return quantiles of RPS scores to build prediction sets.

        Parameters
        ----------
        conformity_scores: NDArray of shape (n_samples,)
        alpha_np: NDArray of shape (n_alpha,)
        estimator: EnsembleClassifier

        Returns
        -------
        NDArray of shape (n_alpha,)
        """
        return np.quantile(conformity_scores, 1 - alpha_np, axis=0)

    def get_prediction_sets(
        self,
        y_pred_proba: NDArray,
        conformity_scores: NDArray,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        agg_scores: Optional[str] = "mean",
        **kwargs
    ) -> NDArray:
        """
        Generate prediction sets for each sample using the RPS conformity score.

        Parameters
        ----------
        y_pred_proba : NDArray of shape (n_samples, n_classes)
            Predicted class probabilities.
        conformity_scores : NDArray of shape (n_samples,)
            Conformity scores computed from calibration data (not used here directly).
        alpha_np : NDArray of shape (n_alpha,)
            Miscoverage levels (1 - confidence levels).
        estimator : EnsembleClassifier
            Fitted ensemble estimator.
        agg_scores : str, default="mean"
            Not used here, included for compatibility.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        NDArray of shape (n_samples, n_classes, n_alpha)
            Boolean prediction sets.
        """
        n_samples, n_classes, n_alpha = y_pred_proba.shape

        # Thresholds from quantiles
        thresholds = self.quantiles_  # shape: (n_alpha,)
        if thresholds is None:
            raise ValueError("`quantiles_` must be set before calling get_prediction_sets.")

        # Initialize output
        prediction_sets = np.zeros((n_samples, n_classes, n_alpha), dtype=bool)

        # For each class, simulate it being the true label and compute RPS
        for alpha_idx, threshold in enumerate(thresholds):
            for class_idx in range(n_classes):
                # Binary outcomes with 1 at current class position
                outcomes = np.zeros((n_samples, n_classes))
                outcomes[:, class_idx] = 1

                # Compute RPS assuming each class is true
                cum_probs = np.cumsum(y_pred_proba[:, :, alpha_idx], axis=1)
                cum_outcomes = np.cumsum(outcomes, axis=1)
                rps_scores = np.sum((cum_probs - cum_outcomes) ** 2, axis=1) / (n_classes - 1)

                # Compare against thresholds to include in prediction set
                prediction_sets[:, class_idx, alpha_idx] = rps_scores <= threshold + EPSILON

        return prediction_sets
