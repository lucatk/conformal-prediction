from abc import ABC, abstractmethod

import numpy as np


class Dataset(ABC):
    name: str = None

    @abstractmethod
    def get_train_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the training data.
        """
        pass

    @abstractmethod
    def get_test_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the test data.
        """
        pass

    @abstractmethod
    def get_num_classes(self) -> int:
        """
        Returns the number of classes.
        """
        pass

    @abstractmethod
    def get_class_labels(self) -> list[str]:
        """
        Returns the class labels.
        """
        pass
