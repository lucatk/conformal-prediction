import numpy as np
from medmnist import RetinaMNIST
from torchvision.transforms import Compose, ToTensor

from datasets.base_dataset import Dataset


class RetinaMNISTDataset(Dataset):
    def __init__(self, root_path: str, hold_out_size: int):
        self.hold_out_size = hold_out_size
        self.train_dataset = RetinaMNIST(
            root=root_path,
            # download=True,
            split='train',
            # target_transform=np.array,
            transform=Compose([ToTensor()]),
        )
        self.test_dataset = RetinaMNIST(
            root=root_path,
            # download=True,
            split='test',
            # target_transform=np.array,
            transform=Compose([ToTensor()]),
        )
        self.prepare_data()

    def prepare_data(self):
        X_train, y_train = np.array([x for x, y in iter(self.train_dataset)]), np.array(
            [y for x, y in iter(self.train_dataset)])
        X_test, y_test = np.array([x for x, y in iter(self.test_dataset)]), np.array(
            [y for x, y in iter(self.test_dataset)])

        # rs = ShuffleSplit(n_splits=1, test_size=self.hold_out_size, random_state=0)
        # train_index, hold_out_index = next(rs.split(X_train))
        self.X_train = X_train
        self.y_train = y_train.ravel()
        # self.X_hold_out = X_train[hold_out_index]
        # self.y_hold_out = y_train[hold_out_index]
        self.X_test = X_test
        self.y_test = y_test.ravel()

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
        return len(self.train_dataset.info['label'])

    def get_class_labels(self) -> list[str]:
        labels_dict = self.train_dataset.info['label']
        return [labels_dict[key] if key in labels_dict.keys() else f'{key}' for key in labels_dict]
