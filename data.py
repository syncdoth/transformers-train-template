import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def get_dataloaders(
    batch_size=1,
    eval_batch_size=1,
    **dataset_kwargs,
):
    dataloaders = {}
    for mode in ('train', 'valid', 'test'):
        dataset = SequenceDataset(**dataset_kwargs, mode=mode)
        loader = DataLoader(dataset,
                            batch_size=batch_size if mode == 'train' else eval_batch_size,
                            collate_fn=dataset.make_batch)
        dataloaders[mode] = loader

    return dataloaders


class SequenceDataset(Dataset):

    def __init__(
        self,
        mode='train',
        **kwargs,
    ):
        self.mode = mode
        for kw, arg in kwargs:
            self.__setattr__(kw, arg)
        # TODO: add special actions

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def make_batch(self, batch_indices):
        """collate_fn to be passed to the torch.utils.data.DataLoader"""
        raise NotImplementedError


def train_val_split(data, val_ratio=0.15):
    if val_ratio == 0:
        return np.arange(len(data)), []
    train_idx, val_idx = train_test_split(np.arange(len(data)), test_size=val_ratio, shuffle=False)
    return train_idx, val_idx
