from typing import Optional, Union, cast

from mapie._machine_precision import EPSILON
from mapie.conformity_scores.classification import BaseClassificationScore
from sklearn.model_selection import BaseCrossValidator
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
            y_enc: Optional[NDArray] = None,
            **kwargs
    ) -> NDArray:
        """
        Calculate the RPS conformity score for each prediction.

        Parameters
        ----------
        y: NDArray of shape (n_samples,)
            Observed target values.

        y_pred: NDArray of shape (n_samples, n_classes)
            Predicted probabilities per class.

        y_enc: NDArray of shape (n_samples,)
            Target values as normalized encodings.

        Returns
        -------
        NDArray of shape (n_samples,)
            RPS conformity scores.
        """
        # Casting
        y_enc = cast(NDArray, y_enc)

        n_samples, n_classes = y_pred.shape
        conformity_scores = np.empty(n_samples, dtype="float")

        for i in range(n_samples):
            probs = y_pred[i]
            outcome = np.zeros(n_classes)
            outcome[y_enc[i]] = 1
            cum_probs = np.cumsum(probs)
            cum_outcome = np.cumsum(outcome)
            rps = np.sum((cum_probs - cum_outcome) ** 2) / (n_classes - 1)
            conformity_scores[i] = rps

        return conformity_scores

    def get_predictions(
            self,
            X: NDArray,
            alpha_np: NDArray,
            y_pred_proba: NDArray,
            cv: Optional[Union[int, str, BaseCrossValidator]],
            agg_scores: Optional[str] = "mean",
            **kwargs
    ) -> NDArray:
        """
        Just processes the passed y_pred_proba.

        Parameters
        ----------
        X: NDArray of shape (n_samples, n_features)
            Observed feature values (not used since predictions are passed).
        alpha_np: NDArray of shape (n_alpha,)
            NDArray of floats between ``0`` and ``1``, represents the
            uncertainty of the confidence interval.
        y_pred_proba: NDArray
            Predicted probabilities from the estimator.
        cv: Optional[Union[int, str, BaseCrossValidator]]
            Cross-validation strategy used by the estimator.
        agg_scores: Optional[str]
            Method to aggregate the scores from the base estimators.
            If "mean", the scores are averaged. If "crossval", the scores are
            obtained from cross-validation.

            By default ``"mean"``.

        Returns
        -------
        NDArray
            Array of predictions.
        """
        if agg_scores != "crossval":
            y_pred_proba = np.repeat(
                y_pred_proba[:, :, np.newaxis], len(alpha_np), axis=2
            )

        return y_pred_proba

    def get_conformity_score_quantiles(
            self,
            conformity_scores: NDArray,
            alpha_np: NDArray,
            cv: Optional[Union[int, str, BaseCrossValidator]],
            agg_scores: Optional[str] = "mean",
            **kwargs
    ) -> NDArray:
        """
        Get the quantiles of the conformity scores for each uncertainty level.

        Parameters
        ----------
        conformity_scores: NDArray of shape (n_samples,)
            Conformity scores for each sample.
        alpha_np: NDArray of shape (n_alpha,)
            NDArray of floats between 0 and 1, representing the uncertainty
            of the confidence interval.
        cv: Optional[Union[int, str, BaseCrossValidator]]
            Cross-validation strategy used by the estimator.
        agg_scores: Optional[str]
            Method to aggregate the scores from the base estimators.
            If "mean", the scores are averaged. If "crossval", the scores are
            obtained from cross-validation.

            By default ``"mean"``.

        Returns
        -------
        NDArray
            Array of quantiles with respect to alpha_np.
        """
        n = len(conformity_scores)

        if cv == "prefit" or agg_scores in ["mean"]:
            quantiles_ = _compute_quantiles(
                conformity_scores,
                alpha_np
            )
        else:
            quantiles_ = (n + 1) * (1 - alpha_np)

        return quantiles_

    def get_prediction_sets(
            self,
            y_pred_proba: NDArray,
            conformity_scores: NDArray,
            alpha_np: NDArray,
            cv: Optional[Union[int, str, BaseCrossValidator]],
            agg_scores: Optional[str] = "mean",
            **kwargs
    ) -> NDArray:
        """
        Generate prediction sets based on the probability predictions,
        the conformity scores and the uncertainty level.

        Parameters
        ----------
        y_pred_proba: NDArray of shape (n_samples, n_classes)
            Target prediction.
        conformity_scores: NDArray of shape (n_samples,)
            Conformity scores for each sample.
        alpha_np: NDArray of shape (n_alpha,)
            NDArray of floats between 0 and 1, representing the uncertainty
            of the confidence interval.
        cv: Optional[Union[int, str, BaseCrossValidator]]
            Cross-validation strategy used by the estimator.
        agg_scores: Optional[str]
            Method to aggregate the scores from the base estimators.
            If "mean", the scores are averaged. If "crossval", the scores are
            obtained from cross-validation.

            By default ``"mean"``.

        Returns
        -------
        NDArray
            Array of quantiles with respect to alpha_np.
        """
        n = len(conformity_scores)

        if agg_scores == "mean":
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
        else:
            # Crossval aggregation not implemented for RPS
            raise NotImplementedError(
                "Crossval aggregation is not implemented for RPS conformity score. "
                "Please use agg_scores='mean' instead."
            )

        return prediction_sets
