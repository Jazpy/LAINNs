import pandas as pd
import numpy as np
import torch
from statistics import mode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import MinMaxScaler
from torch.utils.data        import Dataset, DataLoader

def preprocess_data(fp, data_dir, idx, valid_ratio=0.1):
  df_raw = pd.read_csv(fp, header=None)

  train_ratio = 1.0 - valid_ratio
  df_train, df_valid = train_test_split(df_raw, train_size=train_ratio)

  if data_dir == '':
    data_dir = '.'

  df_train.to_csv(f'{data_dir}/win_{idx}_train.csv', header=False, index=False)
  df_valid.to_csv(f'{data_dir}/win_{idx}_valid.csv', header=False, index=False)

class SNPDataset(Dataset):
  df_snp = None
  #df_pos = None
  anc_lst = None

  @classmethod
  def read_snp_data(cls, snp_fp):
    cls.df_snp = pd.read_csv(snp_fp, header=None).to_numpy(dtype=np.float32)

  #@classmethod
  #def read_pos_data(cls, pos_fp):
    #cls.df_pos = pd.read_csv(pos_fp, header=None).to_numpy(dtype=np.float32)[0]

  @classmethod
  def read_anc_data(cls, anc_fp):
    cls.anc_lst = []
    with open(anc_fp, 'r') as in_f:
      for line in in_f:
        cls.anc_lst.append(mode([int(x) for x in line.split(',')]))

  def __init__(self, win_fp, admixed=False, transform=None, target_transform=None):
    self.df_win = pd.read_csv(win_fp, header=None).to_numpy(dtype=np.float32)
    self.w_size = int(self.df_win[0][3])
    self.length = len(self.df_win)

    self.admixed = admixed

    self.transform        = transform
    self.target_transform = target_transform

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    ind_idx = int(self.df_win[idx][1])
    pos_idx = int(self.df_win[idx][2])
    snp     = SNPDataset.df_snp[ind_idx]#[pos_idx : pos_idx + self.w_size]
    #pos     = SNPDataset.df_pos[pos_idx : pos_idx + self.w_size]

    if not self.admixed:
      lab = int(self.df_win[idx][4])
    else:
      lab = SNPDataset.anc_lst[ind_idx]

    # Scale position
    #scaled_pos = MinMaxScaler().fit_transform(pos.reshape(-1, 1)).flatten()

    #snp_alt = np.full(self.w_size, lab, dtype=np.float32)

    return lab, snp

def create_training_loaders(data_dir, train_fn, valid_fn, idx, batch_size=32):
  SNPDataset.read_snp_data(f'{data_dir}/snp_{idx}.csv')
  #SNPDataset.read_pos_data(f'{data_dir}/pos.csv')
  train_ds = SNPDataset(f'{data_dir}/{train_fn}')
  valid_ds = SNPDataset(f'{data_dir}/{valid_fn}')
  assert train_ds.w_size == valid_ds.w_size

  train_dl = DataLoader(train_ds, batch_size=batch_size,
    drop_last=False, shuffle=True)
  valid_dl = DataLoader(valid_ds, batch_size=batch_size,
    drop_last=False, shuffle=True)

  return train_dl, valid_dl, train_ds.w_size

def create_testing_loader(data_dir, test_fn, idx, admixed=False, batch_size=32):
  SNPDataset.read_snp_data(f'{data_dir}/snp_{idx}.csv')
  if admixed:
    SNPDataset.read_anc_data(f'{data_dir}/anc_{idx}.csv')
  test_ds = SNPDataset(f'{data_dir}/{test_fn}', admixed=admixed)

  test_dl = DataLoader(test_ds, batch_size=batch_size,
    drop_last=False, shuffle=False)

  return test_dl, test_ds.w_size

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

  return (saved['train_loss_list'],
    saved['valid_loss_list'], saved['global_steps_list'])
