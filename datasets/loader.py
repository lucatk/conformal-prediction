from pathlib import Path

from datasets.base_dataset import Dataset


def load_dataset(dataset: str, hold_out_size: float, root_path: str) -> Dataset:
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
    root_path = str(Path(root_path + '/.datasets').resolve())
    if dataset == 'FGNet':
        from datasets.fgnet import FGNetDataset
        return FGNetDataset(hold_out_size, root_path)
    elif dataset == 'Adience':
        from datasets.adience import AdienceDataset
        return AdienceDataset(hold_out_size, root_path)
    elif dataset == 'RetinaMNIST':
        from datasets.retina_mnist import RetinaMNISTDataset
        return RetinaMNISTDataset(hold_out_size, root_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
