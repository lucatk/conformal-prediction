import numpy as np
from dlordinal.datasets import Adience
from numpy import ndarray
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Lambda

from datasets.base_dataset import Dataset


class AdienceDataset(Dataset):
    """
    Adience dataset.
    """

    name = 'Adience'

    train_dataset: Adience
    test_dataset: Adience

    hold_out_size: float

    X_train: ndarray
    y_train: ndarray
    X_hold_out: ndarray
    y_hold_out: ndarray
    X_test: ndarray
    y_test: ndarray

    def __init__(self, hold_out_size: float, root_path: str):
        self.hold_out_size = hold_out_size
        # Custom resize function that scales by 0.25
        def resize_by_scale(img):
            from PIL import Image
            if isinstance(img, Image.Image):
                width, height = img.size
                new_width = int(width * 0.0625)
                new_height = int(height * 0.0625)
                return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            return img
        self.train_dataset = Adience(
            root=root_path,
            train=True,
            target_transform=np.array,
            transform=Compose([Lambda(resize_by_scale), ToTensor()]),  # Resize by scale
        )
        self.test_dataset = Adience(
            root=root_path,
            train=False,
            target_transform=np.array,
            transform=Compose([Lambda(resize_by_scale), ToTensor()]),  # Resize by scale
        )
        self.prepare_data()

    def prepare_data(self):
        X_train, y_train = np.array([x for x, y in iter(self.train_dataset)]), np.array(
            [y for x, y in iter(self.train_dataset)])
        X_test, y_test = np.array([x for x, y in iter(self.test_dataset)]), np.array(
            [y for x, y in iter(self.test_dataset)])

        X_train, X_hold_out, y_train, y_hold_out = train_test_split(
            X_train, y_train,
            test_size=self.hold_out_size,
            random_state=1,
        )

        self.X_train = X_train
        self.y_train = y_train
        self.X_hold_out = X_hold_out
        self.y_hold_out = y_hold_out
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
        return ['0-2', '4-6', '8-13', '15-20', '25-32', '38-43', '48-53', '60-100']
