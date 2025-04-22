from skorch import NeuralNetClassifier
from torch.nn.functional import softmax


class SoftmaxNeuralNetClassifier(NeuralNetClassifier):
    def predict_proba(self, X):
        y_pred = self.forward(X, training=False)
        proba = softmax(y_pred, dim=1)
        return proba.detach().cpu().numpy()

    def predict(self, X):
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)
