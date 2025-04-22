from abc import ABC, abstractmethod

import numpy as np


class Dataset(ABC):
    @abstractmethod
    def get_train_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the training data.
        """
        pass

    @abstractmethod
    def get_hold_out_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the hold-out data.
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