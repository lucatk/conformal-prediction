import base64
from io import BytesIO
from typing import Dict, Any, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from highlight_text import fig_text, ax_text
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas.io.formats.style import Styler
from skorch import NeuralNetClassifier
from torch.nn.functional import softmax

import streamlit as st

class SoftmaxNeuralNetClassifier(NeuralNetClassifier):
    def predict_proba(self, X):
        y_pred = self.forward(X, training=False)
        proba = softmax(y_pred, dim=1)
        return proba.detach().cpu().numpy()

    def predict(self, X):
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)


def prepare_sample(idx: int, true_class: Any, class_labels: list[str], y_pred_set: ndarray):
    pred_set = y_pred_set[idx]

    is_correct = False
    try:
        is_correct = pred_set[true_class][0] == True
    except:
        pass
    class_set = [class_labels[iidx] for iidx, p in enumerate(pred_set) if p[0] == True]

    return is_correct, class_set


def plot_image_samples(indices: Iterable[int], X_test: ndarray, y_test: ndarray, y_pred_sets: Dict[str, ndarray], class_labels: list[str]):
    plt.figure(figsize=(16, 8))
    for i, idx in enumerate(indices):
        image = X_test[idx]  # Image corresponding to X_test[idx]
        true_class = y_test[idx]

        pred_samples = [(name, *prepare_sample(idx, true_class, class_labels, y_pred_set)) for name, y_pred_set in y_pred_sets.items()]

        ax = plt.subplot(4, 4, i + 1)
        plt.imshow((np.transpose(image, (1, 2, 0)) * 255).astype("uint8"))  # Adjust if your image format differs
        plt.axis('off')
        ax_text(
            x=0, y=-50,
            s=f'True: {class_labels[true_class]}\n{'\n'.join([line for _, line in pred_samples])}',
            fontsize=9,
            ax=ax,
            va='top'
        )
    plt.tight_layout()
    return plt


def frame_image_samples(X_test: ndarray, y_test: ndarray, y_pred_sets: Dict[str, ndarray], class_labels: list[str]):
    samples = []
    sample_highlight_correct = []
    # sample_highlight_incorrect = []
    for idx, image in enumerate(X_test):
        true_class = y_test[idx]
        pred_samples = [prepare_sample(idx, true_class, class_labels, y_pred_set) for _, y_pred_set in y_pred_sets.items()]

        samples.append([image_tensor_to_base64(image), class_labels[true_class]] + [class_set for _, class_set in pred_samples])
        correct = [pred[0] for pred in pred_samples]
        sample_highlight_correct.append([False, False] + correct)
        # sample_highlight_incorrect.append([False, False] + np.invert(correct))
    df = pd.DataFrame(samples, columns=['image', 'true_class'] + [name for name in y_pred_sets.keys()])
    return (df.style
            .apply(lambda x: np.where(pd.DataFrame(sample_highlight_correct), 'background-color: green', None), axis=None))


def image_tensor_to_base64(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    image = np.transpose(image, (1, 2, 0))
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    im = Image.fromarray(image.astype("uint8"))
    with BytesIO() as buffer:
        im.save(buffer, "png")
        raw_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{raw_base64}"


def st_narrow():
    _, col, _ = st.columns([1, 6, 1])
    return col
