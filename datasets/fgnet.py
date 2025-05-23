import numpy as np
from dlordinal.datasets import FGNet
from numpy import ndarray
from sklearn.model_selection import ShuffleSplit
from torchvision.transforms import Compose, ToTensor

from datasets.base_dataset import Dataset


class FGNetDataset(Dataset):
    """
    FGNet dataset.
    """
    train_dataset: FGNet
    test_dataset: FGNet

    X_train: ndarray
    y_train: ndarray
    X_hold_out: ndarray
    y_hold_out: ndarray
    X_test: ndarray
    y_test: ndarray

    def __init__(self, root_path: str, hold_out_size: int):
        self.hold_out_size = hold_out_size
        self.train_dataset = FGNet(
            root=root_path,
            train=True,
            target_transform=np.array,
            transform=Compose([ToTensor()]),
        )
        self.test_dataset = FGNet(
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

        rs = ShuffleSplit(n_splits=1, test_size=self.hold_out_size, random_state=0)
        # train_index, hold_out_index = next(rs.split(X_train))
        self.X_train = X_train
        self.y_train = y_train
        # self.X_hold_out = X_train[hold_out_index]
        # self.y_hold_out = y_train[hold_out_index]
        self.X_test = X_test
        self.y_test = y_test

    def get_train_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the training data.
        """
        return self.X_train, self.y_train

    def get_hold_out_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the hold-out data.
        """
        return self.X_hold_out, self.y_hold_out

    def get_test_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the test data.
        """
        return self.X_test, self.y_test

    def get_num_classes(self) -> int:
        return len(self.train_dataset.classes)

    def get_class_labels(self) -> list[str]:
        return ['0-6', '7-12', '13-19', '20-35', '36-50', '51-99']
