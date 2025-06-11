import numpy as np
from dlordinal.datasets import Adience
from numpy import ndarray
from torchvision.transforms import Compose, ToTensor

from datasets.base_dataset import Dataset


class AdienceDataset(Dataset):
    """
    Adience dataset.
    """
    train_dataset: Adience
    test_dataset: Adience

    X_train: ndarray
    y_train: ndarray
    X_test: ndarray
    y_test: ndarray

    def __init__(self, root_path: str):
        self.train_dataset = Adience(
            root=root_path,
            train=True,
            target_transform=np.array,
            transform=Compose([ToTensor()]),
        )
        self.test_dataset = Adience(
            root=root_path,
            train=False,
            target_transform=np.array,
            transform=Compose([ToTensor()]),
        )
        self.prepare_data()

    def prepare_data(self):
        X_train, y_train = np.array([x for x, y in iter(self.train_dataset)]), np.array(
            [y for x, y in iter(self.train_dataset)])
        X_test, y_test = np.array([x for x, y in iter(self.test_dataset)]), np.array(
            [y for x, y in iter(self.test_dataset)])

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_train_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the training data.
        """
        return self.X_train, self.y_train

    def get_test_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the test data.
        """
        return self.X_test, self.y_test

    def get_num_classes(self) -> int:
        return len(self.train_dataset.classes)

    def get_class_labels(self) -> list[str]:
        return ['0-2', '4-6', '8-13', '15-20', '25-32', '38-43', '48-53', '60-100']
