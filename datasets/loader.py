from pathlib import Path

from datasets.base_dataset import Dataset


def load_dataset(dataset: str, hold_out_size: int) -> Dataset:
    """
    Load the dataset.

    Parameters
    ----------
    dataset : str
        The name of the dataset to load.

    Returns
    -------
    Dataset
        The loaded dataset.
    """
    root_path = str(Path('.datasets').resolve())
    if dataset == 'FGNet':
        from datasets.fgnet import FGNetDataset
        return FGNetDataset(root_path, hold_out_size)
    elif dataset == 'RetinaMNIST':
        from datasets.retina_mnist import RetinaMNISTDataset
        return RetinaMNISTDataset(root_path, hold_out_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
