import pandas as pd
import numpy as np
import torch
import os

from collections import defaultdict
from statistics  import mode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import MinMaxScaler
from torch.utils.data        import Dataset, DataLoader


class WindowDataset(Dataset):


    def __init__(self, predictions_fp, truths_fp, cms_fp, valid=False, transform=None, target_transform=None):
        self.predictions = np.transpose(np.load(predictions_fp).astype(np.float32), (1, 0, 2))
        self.truths = np.transpose(np.load(truths_fp).astype(int), (1, 0))
        self.cms = np.load(cms_fp).astype(np.float32)

        self.valid = valid

        if self.valid:
            self.length = int(len(self.predictions) * .5)
        else:
            self.length = int(len(self.predictions) * .5)

        self.transform        = transform
        self.target_transform = target_transform


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        if self.valid:
            idx += int(len(self.predictions) * .5)
        return self.truths[idx], self.predictions[idx], self.cms


def create_training_loaders(data_dir, batch_size=1):
    train_ds = WindowDataset(f'{data_dir}/train_probabilities.npy', f'{data_dir}/train_truths.npy', f'{data_dir}/cms.npy')
    valid_ds = WindowDataset(f'{data_dir}/train_probabilities.npy', f'{data_dir}/train_truths.npy',
                             f'{data_dir}/cms.npy', valid=True)
    print(train_ds.length)
    print(valid_ds.length)

    train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=False, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, drop_last=False, shuffle=True)

    return train_dl, valid_dl


def create_testing_loader(data_dir, batch_size=1):
    test_ds = WindowDataset(f'{data_dir}/test_probabilities.npy', f'{data_dir}/test_truths.npy', f'{data_dir}/cms.npy')
    test_dl = DataLoader(test_ds, batch_size=batch_size, drop_last=False)

    return test_dl


def save_checkpoint(fp, model, optimizer, valid_loss):
    saved = {'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), 'valid_loss': valid_loss}
    torch.save(saved, fp)


def load_checkpoint(fp, model, device, optimizer=None):
    saved = torch.load(fp, map_location=device)
    model.load_state_dict(saved['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(saved['optimizer_state_dict'])

    return saved['valid_loss']


def save_metrics(fp, train_loss_list, valid_loss_list, global_steps_list):
    saved = {'train_loss_list': train_loss_list,
        'valid_loss_list': valid_loss_list, 'global_steps_list': global_steps_list}
    torch.save(saved, fp)


def load_metrics(fp, device):
    saved = torch.load(fp, map_location=device)

    return (saved['train_loss_list'], saved['valid_loss_list'], saved['global_steps_list'])
