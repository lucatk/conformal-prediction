from pathlib import Path

from datasets.base_dataset import Dataset


def load_dataset(dataset: str) -> Dataset:
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
        return FGNetDataset(root_path)
    elif dataset == 'Adience':
        from datasets.adience import AdienceDataset
        return AdienceDataset(root_path)
    elif dataset == 'RetinaMNIST':
        from datasets.retina_mnist import RetinaMNISTDataset
        return RetinaMNISTDataset(root_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
