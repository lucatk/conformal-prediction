import copy

from mapie.classification import MapieClassifier
from mapie.metrics import classification_coverage_score, classification_mean_width_score
from numpy import ndarray
from sklearn.base import ClassifierMixin
from skorch.callbacks import Callback
from streamlit.elements.progress import ProgressMixin
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW
from torchvision import models

from datasets.base_dataset import Dataset
from models.resnet18_uni import UnimodalResNet18
from util import SoftmaxNeuralNetClassifier


class CPRunner:
    max_epochs = 25

    progress: float | None = None
    has_run: bool = False
    has_error: bool = False

    dataset: Dataset
    preds: dict[str, tuple[ndarray, ndarray, float, float]] = {}

    def __init__(self, dataset_name: str, model: str, score_alg: list[str], loss_fn: list[str], alpha: float,
                 device: str):
        self.dataset_name = dataset_name
        self.model = model
        self.score_alg = score_alg
        self.loss_fn = loss_fn
        self.alpha = alpha
        self.device = device

    def run(self, dataset: Dataset, progress_bar: ProgressMixin):
        if self.has_run or (self.progress is not None and not self.has_error):
            return
        self.has_error = False
        self.dataset = dataset
        self.set_progress(0, progress_bar)

        try:
            self.set_progress(0, progress_bar, f'Fitting...')
            estimators = self._get_estimators(dataset.get_num_classes())
            predictors = self._get_cp_predictors(estimators)

            X_train, y_train = dataset.get_train_data()
            print("X_train shape:", X_train.shape)
            print("y_train shape:", y_train.shape)
            for idx, (name, predictor) in enumerate(predictors.items()):
                self.set_progress(0.1 + (idx / len(predictors)) * 0.4, progress_bar,
                                  f'Fitting {name} ({idx + 1}/{len(predictors)})...')

                class EpochProgress(Callback):
                    def __init__(self, cp_runner: CPRunner):
                        self.cp_runner = cp_runner

                    def on_epoch_begin(self, net, **kwargs):
                        cur_epoch = len(net.history)
                        self.cp_runner.set_progress(
                            0.1 + ((idx + (cur_epoch/self.cp_runner.max_epochs)) / len(predictors)) * 0.4, progress_bar,
                                  f'Fitting {name} ({idx + 1}/{len(predictors)}) - epoch {cur_epoch}/{self.cp_runner.max_epochs}...')

                predictor.estimator.set_params(callbacks=[('epoch_progress', EpochProgress(self))])
                predictor.fit(X_train, y_train)

            # self.set_progress(0.4, progress_bar, 'Fitting predictors...')
            # X_holdout, y_holdout = dataset.get_hold_out_data()
            # self.set_progress(0.1, progress_bar, f'Fitting predictors (0/{len(predictors)})...')
            # for idx, (name, predictor) in enumerate(predictors.items()):
            #     predictor.fit(X_holdout, y_holdout)
            #     self.set_progress(0.4 + ((idx + 1) / len(predictors)) * 0.3, progress_bar, f'Fitting predictors ({idx + 1}/{len(predictors)})...')

            self.set_progress(0.5, progress_bar, f'Predicting (0/{len(predictors)})...')
            X_test, y_test = dataset.get_test_data()
            for idx, (name, predictor) in enumerate(predictors.items()):
                self.set_progress(0.6 + (idx / len(predictors)) * 0.4, progress_bar,
                                  f'Predicting {name} ({idx + 1}/{len(predictors)})...')
                y_pred, y_pred_set = predictor.predict(X_test, alpha=self.alpha)
                self.preds[name] = (
                    y_pred,
                    y_pred_set,
                    classification_mean_width_score(y_pred_set[:, :, 0]),
                    classification_coverage_score(y_test, y_pred_set[:, :, 0]),
                )
        except:
            self.has_error = True
            raise
        self.has_run = True

    def set_progress(self, progress, progress_bar: ProgressMixin, text: str = None):
        self.progress = progress
        progress_bar.progress(progress, text=text)

    def _get_model(self, num_classes: int):
        if self.model == 'resnet18':
            model = models.resnet18(weights="IMAGENET1K_V1")
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif self.model == 'resnet50':
            model = models.resnet50(weights="IMAGENET1K_V1")
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif self.model == 'resnet18-uni':
            model = UnimodalResNet18(num_classes)
        else:
            raise ValueError(f"Unknown model: {self.model}")
        return model

    def _get_estimators(self, num_classes: int):
        model = self._get_model(num_classes)
        model = model.to(self.device)
        loss_fn_set = self._get_loss_fn_set(num_classes)
        return [(
            SoftmaxNeuralNetClassifier(
                module=model,
                criterion=loss_fn.to(self.device),
                optimizer=AdamW,
                lr=1e-3,
                batch_size=32,
                train_split=None,
                max_epochs=self.max_epochs,
                device=self.device,
            )
        ) for loss_fn in loss_fn_set]

    def _get_loss_fn_set(self, num_classes: int):
        loss_fns = []
        for loss_fn in self.loss_fn:
            if loss_fn == 'CrossEntropy':
                loss_fns.append(CrossEntropyLoss())
            elif loss_fn == 'TriangularCrossEntropy':
                from dlordinal.losses import TriangularLoss
                loss_fns.append(TriangularLoss(base_loss=CrossEntropyLoss(), num_classes=num_classes))
            elif loss_fn == 'WeightedKappa':
                from dlordinal.losses import WKLoss
                loss_fns.append(WKLoss(num_classes=num_classes, use_logits=True))
            elif loss_fn == 'EMD':
                from dlordinal.losses import EMDLoss
                loss_fns.append(EMDLoss(num_classes=num_classes))
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
            elif score_alg == 'RAPS':
                from mapie.conformity_scores import RAPSConformityScore
                score_algs.append(RAPSConformityScore(size_raps=0.2))
            elif score_alg == 'RPS':
                from conformity_scores.rps import RPSConformityScore
                score_algs.append(RPSConformityScore())
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
                    cv='split',
                )
            )
            for loss_fn_idx, estimator in enumerate(estimators)
            for score_alg_idx, score_alg in enumerate(score_algs)
        }

    def get_results(self):
        if not self.has_run:
            raise RuntimeError("CPRunner has not been run yet.")
        return self.preds
