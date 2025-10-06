import os
import pickle
import traceback
from pathlib import Path
from threading import Thread
from typing import Callable

import pandas as pd
import runpod
import torch
from dlordinal.output_layers import COPOC
from mapie.classification import MapieClassifier
from numpy import ndarray
from sklearn.base import ClassifierMixin
from skorch.callbacks import Callback, EarlyStopping
from streamlit.elements.progress import ProgressMixin
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torchvision import models
from tqdm import tqdm

from datasets.base_dataset import Dataset
from util import SoftmaxNeuralNetClassifier

import streamlit as st


runpod_token = os.getenv('RUNPOD_API_KEY')
runpod_pod_id = os.getenv('RUNPOD_POD_ID')
data_root = os.environ['DATA_ROOT'] if 'DATA_ROOT' in os.environ else '.'
if runpod_token is not None:
    runpod.api_key = runpod_token


class CPRunner(Thread):
    lr = 1e-3
    batch_size = 128
    max_epochs = 100
    early_stop_epochs = 40

    def __init__(self, dataset: Dataset, model: list[str], score_alg: list[str], loss_fn: list[str], alpha: list[float],
                 replication: int, device: str):
        super().__init__()
        self.progress: float | None = None
        self.has_run: bool = False
        self.has_error: bool = False
        self.preds: dict[str, list[tuple[ndarray, ndarray]]] = {}

        self.dataset = dataset
        self.model = model
        self.score_alg = score_alg
        self.loss_fn = loss_fn
        self.alpha = alpha
        self.num_replications = replication
        self.device = device

        self.terminate_after_run = False

        self.selected_results = []

    def export_results(self):
        """Save results and metadata as bytes."""
        if not self.has_run:
            raise RuntimeError("CPRunner has not been run yet. No results to save.")
        
        # Save both results and metadata
        save_data = {
            'dataset_name': self.dataset.name,
            'model': self.model,
            'score_alg': self.score_alg,
            'loss_fn': self.loss_fn,
            'alpha': self.alpha,
            'num_replications': self.num_replications,
            'device': self.device,
            'preds': self.preds,
            'has_run': self.has_run,
            'has_error': self.has_error
        }

        score_alg_str = "_".join(self.score_alg)
        loss_fn_str = "_".join(self.loss_fn)
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results_{timestamp}_{self.dataset.name}_{self.model}_{score_alg_str}_{loss_fn_str}_alpha{self.alpha}.pkl"

        return pickle.dumps(save_data), filename

    @staticmethod
    def from_save(save_data: any, dataset: Dataset):
        """Create a new CPRunner instance from saved bytes."""
        
        # Create new instance with saved metadata
        cp_runner = CPRunner(
            dataset=dataset,
            model=save_data['model'],
            score_alg=save_data['score_alg'],
            loss_fn=save_data['loss_fn'],
            alpha=save_data['alpha'],
            replication=save_data['num_replications'],
            device=save_data['device'],
        )
        
        # Restore the results
        cp_runner.preds = save_data['preds']
        cp_runner.has_run = save_data['has_run']
        cp_runner.has_error = save_data['has_error']
        cp_runner.selected_results = [*cp_runner.preds.keys()]
        
        return cp_runner

    def set_terminate_after_run(self, terminate: bool):
        """Set whether to terminate the pod after run."""
        self.terminate_after_run = terminate

    def terminate_pod(self):
        print(f'Run {'failed. T' if self.has_error else 'succeeded. Exporting results and t'}erminating pod...')
        if not self.has_error:
            export_path = str(Path(data_root + '/exports').resolve())
            os.makedirs(export_path, exist_ok=True)
            file_data, filename = self.export_results()
            with open(os.path.join(export_path, filename), 'wb') as f:
                f.write(file_data)
        # runpod.stop_pod(runpod_pod_id)

    def run(self):
        if self.has_run or (self.progress is not None and not self.has_error):
            return
        self.has_error = False

        progress_bar = st.progress(0, text='Evaluation in progress...')
        self.set_progress(0, progress_bar)

        try:
            for rep in (pbar_rep := tqdm(range(self.num_replications), desc='Replications')):  # repeat the fitting process for each replication
                self.set_progress(0, progress_bar, f'[Replication {rep+1}] Fitting...')
                estimators = self._get_estimators(self.dataset.get_num_classes())
                predictors = self._get_cp_predictors(estimators)

                X_train, y_train = self.dataset.get_train_data()
                print("X_train shape:", X_train.shape)
                print("y_train shape:", y_train.shape)
                for idx, (name, (_, predictor)) in enumerate((pbar_fit := tqdm(predictors.items(), leave=False))):
                    desc = f'[Replication {rep+1}] Fitting {name} ({idx + 1}/{len(predictors)})...'
                    self.set_progress(0.1 + (idx / len(predictors)) * 0.4, progress_bar,
                                      desc)
                    pbar_fit.set_description(desc)

                    pbar: tqdm | None = None

                    def on_train_begin(net):
                        nonlocal pbar
                        pbar = tqdm(total=self.max_epochs, leave=False)

                    def on_train_end(net):
                        nonlocal pbar
                        pbar.close()

                    def on_epoch_begin(net):
                        nonlocal pbar
                        cur_epoch = len(net.history)
                        self.set_progress(
                            0.1 + ((idx + (cur_epoch / self.max_epochs)) / len(predictors)) * 0.4, progress_bar,
                            f'[Replication {rep + 1}] Fitting {name} ({idx + 1}/{len(predictors)}) - epoch {cur_epoch}/{self.max_epochs}...')
                        pbar.update()
                        pbar.set_description(f'epoch {cur_epoch}/{self.max_epochs}')

                    predictor.estimator.set_params(callbacks=[
                        EarlyStopping(patience=self.early_stop_epochs, monitor='train_loss'),
                        ('epoch_progress', EpochProgress(on_train_begin, on_epoch_begin, on_train_end))
                    ])
                    predictor.fit(X_train, y_train)
                pbar_fit.close()

                self.set_progress(0.5, progress_bar, f'[Replication {rep+1}] Predicting (0/{len(predictors)})...')

                X_test, y_test = self.dataset.get_test_data()

                for idx, (name, (alpha, predictor)) in enumerate((pbar_pred := tqdm(predictors.items(), desc="Predicting...", leave=False))):
                    if name not in self.preds:
                        self.preds[name] = []

                    desc = f'[Replication {rep+1}] Predicting {name} ({idx + 1}/{len(predictors)})...'
                    self.set_progress(0.6 + (idx / len(predictors)) * 0.4, progress_bar,
                                      desc)
                    pbar_pred.set_description(desc)

                    y_pred, y_pred_set = predictor.predict(X_test, alpha=alpha)
                    self.preds[name].append((y_pred, y_pred_set))
                pbar_pred.close()
        except:
            # pbar_rep.display(msg=f'[Replication {rep+1}] Error occurred.')
            self.has_error = True
            if self.terminate_after_run:
                traceback.print_exc()
                self.terminate_pod()
                return
            else:
                raise
        # pbar_rep.display(msg="Run completed successfully.")
        self.selected_results = [*self.preds.keys()]
        self.has_run = True
        if self.terminate_after_run:
            self.terminate_pod()

    def set_progress(self, progress, progress_bar: ProgressMixin, text: str = None):
        self.progress = progress
        progress_bar.progress(progress, text=text)

    def _get_model_set(self, num_classes: int):
        models_list = []
        for model_name in self.model:
            if model_name == 'resnet18':
                model = models.resnet18(weights="IMAGENET1K_V1")
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif model_name == 'resnet18-uni':
                model = models.resnet18(weights="IMAGENET1K_V1")
                model.fc = nn.Sequential(nn.Linear(model.fc.in_features, num_classes), COPOC())
            elif model_name == 'resnet50':
                model = models.resnet50(weights="IMAGENET1K_V1")
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            if torch.cuda.device_count() > 1:
                print("Using", torch.cuda.device_count(), "GPUs")
                model = nn.DataParallel(model)
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
                lr=self.lr,
                batch_size=self.batch_size,
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
            f'{self.model[model_idx]}_{self.loss_fn[loss_fn_idx]}_{self.score_alg[score_alg_idx]}_alpha{alpha:.2f}': (
                alpha,
                MapieClassifier(
                    estimator=estimator,
                    conformity_score=score_alg,
                    cv='split',
                    random_state=1,
                )
            )
            for model_idx, loss_fn_idx, estimator in estimators
            for score_alg_idx, score_alg in enumerate(score_algs)
            for alpha in self.alpha
        }

    def get_results(self):
        if not self.has_run:
            raise RuntimeError("CPRunner has not been run yet.")
        return self.preds


class EpochProgress(Callback):
    def __init__(
            self,
            _on_train_begin: Callable[[any], None],
            _on_epoch_begin: Callable[[any], None],
            _on_train_end: Callable[[any], None]
    ):
        self._on_train_begin = _on_train_begin
        self._on_epoch_begin = _on_epoch_begin
        self._on_train_end = _on_train_end

    def on_train_begin(self, net, **kwargs):
        self._on_train_begin(net)

    def on_epoch_begin(self, net, **kwargs):
        self._on_epoch_begin(net)

    def on_train_end(self, net, **kwargs):
        self._on_train_end(net)
