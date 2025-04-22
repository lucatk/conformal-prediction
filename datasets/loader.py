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
    if dataset == 'FGNet':
        from datasets.fgnet import FGNetDataset
        return FGNetDataset(hold_out_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
