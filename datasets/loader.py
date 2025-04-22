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
    if dataset == 'FGNet':
        from datasets.fgnet import FGNetDataset
        return FGNetDataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
