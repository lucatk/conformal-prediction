import pickle
from mapie.classification import MapieClassifier
from numpy import ndarray
from sklearn.base import ClassifierMixin
from skorch.callbacks import Callback
from streamlit.elements.progress import ProgressMixin
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torchvision import models

from datasets.base_dataset import Dataset
from models.resnet18_uni import UnimodalResNet18
from util import SoftmaxNeuralNetClassifier


class CPRunner:
    max_epochs = 25

    def __init__(self, dataset_name: str, model: list[str], score_alg: list[str], loss_fn: list[str], alpha: float,
                 device: str):
        self.progress: float | None = None
        self.has_run: bool = False
        self.has_error: bool = False
        self.preds: dict[str, list[tuple[ndarray, ndarray]]] = {}

        self.dataset_name = dataset_name
        self.model = model
        self.score_alg = score_alg
        self.loss_fn = loss_fn
        self.alpha = alpha
        self.device = device

    def save_results_bytes(self):
        """Save results and metadata as bytes."""
        if not self.has_run:
            raise RuntimeError("CPRunner has not been run yet. No results to save.")
        
        # Save both results and metadata
        save_data = {
            'dataset_name': self.dataset_name,
            'model': self.model,
            'score_alg': self.score_alg,
            'loss_fn': self.loss_fn,
            'alpha': self.alpha,
            'device': self.device,
            'preds': self.preds,
            'has_run': self.has_run,
            'has_error': self.has_error
        }
        
        return pickle.dumps(save_data)

    @staticmethod
    def from_bytes(data_bytes: bytes):
        """Create a new CPRunner instance from saved bytes."""
        save_data = pickle.loads(data_bytes)
        
        # Create new instance with saved metadata
        cp_runner = CPRunner(
            dataset_name=save_data['dataset_name'],
            model=save_data['model'],
            score_alg=save_data['score_alg'],
            loss_fn=save_data['loss_fn'],
            alpha=save_data['alpha'],
            device=save_data['device']
        )
        
        # Restore the results
        cp_runner.preds = save_data['preds']
        cp_runner.has_run = save_data['has_run']
        cp_runner.has_error = save_data['has_error']
        
        return cp_runner

    def run(self, dataset: Dataset, num_replications: int, progress_bar: ProgressMixin):
        if self.has_run or (self.progress is not None and not self.has_error):
            return
        self.has_error = False
        self.dataset = dataset
        self.set_progress(0, progress_bar)

        try:
            for rep in range(num_replications):  # repeat the fitting process for each replication
                self.set_progress(0, progress_bar, f'[Replication {rep+1}] Fitting...')
                estimators = self._get_estimators(dataset.get_num_classes())
                predictors = self._get_cp_predictors(estimators)

                X_train, y_train = dataset.get_train_data()
                print("X_train shape:", X_train.shape)
                print("y_train shape:", y_train.shape)
                for idx, (name, predictor) in enumerate(predictors.items()):
                    self.set_progress(0.1 + (idx / len(predictors)) * 0.4, progress_bar,
                                      f'[Replication {rep+1}] Fitting {name} ({idx + 1}/{len(predictors)})...')

                    class EpochProgress(Callback):
                        def __init__(self, cp_runner: CPRunner):
                            self.cp_runner = cp_runner

                        def on_epoch_begin(self, net, **kwargs):
                            cur_epoch = len(net.history)
                            self.cp_runner.set_progress(
                                0.1 + ((idx + (cur_epoch/self.cp_runner.max_epochs)) / len(predictors)) * 0.4, progress_bar,
                                      f'[Replication {rep+1}] Fitting {name} ({idx + 1}/{len(predictors)}) - epoch {cur_epoch}/{self.cp_runner.max_epochs}...')

                    predictor.estimator.set_params(callbacks=[('epoch_progress', EpochProgress(self))])
                    predictor.fit(X_train, y_train)

                self.set_progress(0.5, progress_bar, f'[Replication {rep+1}] Predicting (0/{len(predictors)})...')
                X_test, y_test = dataset.get_test_data()

                for idx, (name, predictor) in enumerate(predictors.items()):
                    if name not in self.preds:
                        self.preds[name] = []

                    self.set_progress(0.6 + (idx / len(predictors)) * 0.4, progress_bar,
                                      f'[Replication {rep+1}] Predicting {name} ({idx + 1}/{len(predictors)})...')
                    y_pred, y_pred_set = predictor.predict(X_test, alpha=self.alpha)
                    self.preds[name].append((y_pred, y_pred_set))
        except:
            self.has_error = True
            raise
        self.has_run = True

    def set_progress(self, progress, progress_bar: ProgressMixin, text: str = None):
        self.progress = progress
        progress_bar.progress(progress, text=text)

    def _get_model_set(self, num_classes: int):
        models_list = []
        for model_name in self.model:
            if model_name == 'resnet18':
                model = models.resnet18(weights="IMAGENET1K_V1")
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif model_name == 'resnet50':
                model = models.resnet50(weights="IMAGENET1K_V1")
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif model_name == 'resnet18-uni':
                model = UnimodalResNet18(num_classes)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            models_list.append(model)
        return models_list

    def _get_estimators(self, num_classes: int):
        model_set = self._get_model_set(num_classes)
        loss_fn_set = self._get_loss_fn_set(num_classes)
        return [(
            model_idx,
            loss_fn_idx,
            SoftmaxNeuralNetClassifier(
                module=model.to(self.device),
                criterion=loss_fn.to(self.device),
                optimizer=AdamW,
                lr=1e-3,
                batch_size=32,
                train_split=None,
                max_epochs=self.max_epochs,
                device=self.device,
            )
        ) for model_idx, model in enumerate(model_set)
        for loss_fn_idx, loss_fn in enumerate(loss_fn_set)]

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
            elif score_alg == 'APS':
                from mapie.conformity_scores import APSConformityScore
                score_algs.append(APSConformityScore())
            elif score_alg == 'RAPS':
                from mapie.conformity_scores import RAPSConformityScore
                score_algs.append(RAPSConformityScore(size_raps=0.2))
            elif score_alg == 'RPS':
                from conformity_scores.rps import RPSConformityScore
                score_algs.append(RPSConformityScore())
            else:
                raise ValueError(f"Unknown score algorithm: {score_alg}")
        return score_algs

    def _get_cp_predictors(self, estimators: list[tuple[int, int, ClassifierMixin]]):
        score_algs = self._get_score_alg_set()
        return {
            f'{self.model[model_idx]}_{self.loss_fn[loss_fn_idx]}_{self.score_alg[score_alg_idx]}': (
                MapieClassifier(
                    estimator=estimator,
                    conformity_score=score_alg,
                    cv='split',
                )
            )
            for model_idx, loss_fn_idx, estimator in estimators
            for score_alg_idx, score_alg in enumerate(score_algs)
        }

    def get_results(self):
        if not self.has_run:
            raise RuntimeError("CPRunner has not been run yet.")
        return self.preds
