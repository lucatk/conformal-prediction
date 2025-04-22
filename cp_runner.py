import time

from dlordinal.losses import OrdinalECOCDistanceLoss
from mapie.classification import MapieClassifier
from mapie.metrics import classification_coverage_score, classification_mean_width_score
from numpy import ndarray
from sklearn.base import ClassifierMixin
from streamlit.elements.progress import ProgressMixin
from torch import nn
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adam
from torchvision import models

from datasets.base_dataset import Dataset
from util import SoftmaxNeuralNetClassifier


class CPRunner:
    progress: float | None = None
    has_run: bool = False

    preds: dict[str, tuple[ndarray, ndarray, float, float]] = {}

    def __init__(self, dataset_name: str, model: str, score_alg: list[str], loss_fn: list[str], alpha: float, device: str):
        self.dataset_name = dataset_name
        self.model = model
        self.score_alg = score_alg
        self.loss_fn = loss_fn
        self.alpha = alpha
        self.device = device

    def run(self, dataset: Dataset, progress_bar: ProgressMixin):
        if self.has_run or self.progress is not None:
            return
        self.set_progress(0, progress_bar)

        self.set_progress(0.1, progress_bar, f'Training...')
        estimators = self._get_estimators(dataset.get_num_classes())
        X_train, y_train = dataset.get_train_data()
        self.set_progress(0.1, progress_bar, f'Training (0/{len(estimators)})...')
        for idx, estimator in enumerate(estimators):
            estimator.fit(X_train, y_train)
            self.set_progress(0.1 + ((idx + 1) / len(estimators)) * 0.3, progress_bar, f'Training ({idx + 1}/{len(estimators)})...')

        self.set_progress(0.4, progress_bar, 'Fitting predictors...')
        predictors = self._get_cp_predictors(estimators)
        X_holdout, y_holdout = dataset.get_hold_out_data()
        self.set_progress(0.1, progress_bar, f'Fitting predictors (0/{len(predictors)})...')
        for idx, (name, predictor) in enumerate(predictors.items()):
            predictor.fit(X_holdout, y_holdout)
            self.set_progress(0.4 + ((idx + 1) / len(predictors)) * 0.3, progress_bar, f'Fitting predictors ({idx + 1}/{len(predictors)})...')

        self.set_progress(0.7, progress_bar, f'Predicting (0/{len(predictors)})...')
        X_test, y_test = dataset.get_test_data()
        for idx, (name, predictor) in enumerate(predictors.items()):
            y_pred, y_pred_set = predictor.predict(X_test, alpha=self.alpha)
            self.preds[name] = (
                y_pred,
                y_pred_set,
                classification_mean_width_score(y_pred_set[:, :, 0]),
                classification_coverage_score(y_test, y_pred_set[:, :, 0]),
            )
            self.set_progress(0.7 + ((idx + 1) / len(predictors)) * 0.3, progress_bar, f'Predicting ({idx + 1}/{len(predictors)})...')

        self.has_run = True

    def set_progress(self, progress, progress_bar: ProgressMixin, text: str = None):
        self.progress = progress
        progress_bar.progress(progress, text=text)

    def _get_model(self):
        if self.model == 'resnet18':
            model = models.resnet18(weights="IMAGENET1K_V1")
        elif self.model == 'resnet50':
            model = models.resnet50(weights="IMAGENET1K_V1")
        else:
            raise ValueError(f"Unknown model: {self.model}")
        return model

    def _get_estimators(self, num_classes: int):
        model = self._get_model()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(self.device)
        loss_fn_set = self._get_loss_fn_set(num_classes)
        return [(
            SoftmaxNeuralNetClassifier(
                module=model,
                criterion=loss_fn.to(self.device),
                optimizer=Adam,
                lr=1e-3,
                max_epochs=25,
                device=self.device,
            )
        ) for loss_fn in loss_fn_set]

    def _get_loss_fn_set(self, num_classes: int):
        loss_fns = []
        for loss_fn in self.loss_fn:
            if loss_fn == 'TriangularCrossEntropy':
                from dlordinal.losses import TriangularCrossEntropyLoss
                loss_fns.append(TriangularCrossEntropyLoss(num_classes=num_classes))
            elif loss_fn == 'CrossEntropy':
                loss_fns.append(CrossEntropyLoss())
            # elif loss_fn == 'OrdinalECOCDistance':
            #     loss_fns.append(OrdinalECOCDistanceLoss(num_classes=num_classes))
            # weighted kappa
            # emd loss
            else:
                raise ValueError(f"Unknown loss function: {loss_fn}")
        return loss_fns

    def _get_score_alg_set(self):
        score_algs = []
        for score_alg in self.score_alg:
            if score_alg == 'LAC':
                from mapie.conformity_scores import LACConformityScore
                score_algs.append(LACConformityScore())
            else:
                raise ValueError(f"Unknown score algorithm: {score_alg}")
        return score_algs

    def _get_cp_predictors(self, estimators: list[ClassifierMixin]):
        score_algs = self._get_score_alg_set()
        return {
            f'{self.loss_fn[loss_fn_idx]}_{self.score_alg[score_alg_idx]}': (
                MapieClassifier(
                    estimator=estimator,
                    conformity_score=score_alg,
                    cv='prefit',
                )
            )
            for loss_fn_idx, estimator in enumerate(estimators)
            for score_alg_idx, score_alg in enumerate(score_algs)
        }

    def get_results(self):
        if not self.has_run:
            raise RuntimeError("CPRunner has not been run yet.")
        return self.preds
