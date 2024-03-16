import torch
import argparse
import time
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
import src.model.data_management as data_management
from sklearn.metrics import confusion_matrix
from statistics import mode
from sklearn.utils.extmath import weighted_mode
from src.model.model import BLSTM, Multilayer, Transformer, CNN

def main():
  start_t = time.time()
  print(f'Found device: {"cuda" if torch.cuda.is_available() else "cpu"}')
  device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Argument handling
  args         = handle_args()
  model_dir    = args['model_data']
  model_type   = args['model']
  learning     = args['learning']
  optimizer    = args['optimizer']
  data_dir     = args['directory']
  plot_loss    = args['plot_loss']
  plot_conf    = args['confusion']
  admixed      = args['admixed']
  num_classes  = args['classes']
  start_window = args['start_window']
  num_windows  = args['windows']

  # To condense all prediction results from multiple windows into a single list
  lai_pred = []
  lai_prob = []
  lai_true = []

  for win_index in range(start_window, start_window + num_windows):
    # Data loading
    test_fn = f'win_{win_index}.csv'

    print(f'Creating data loader for {win_index=}...')
    test_dl, window_size = data_management.create_testing_loader(
      data_dir, test_fn, win_index, admixed=admixed, batch_size=64)
    elapsed_t = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_t))
    print(f'Done creating data loader, elapsed time = {elapsed_t}.')

    # Model loading
    model_type = model_type.lower()
    model_id = f'{model_type}_{win_index}_{learning:.0e}_{optimizer}'
    model_fp = f'{model_dir}/{model_id}_model.pt'

    if model_type == 'transformer':
      model = Transformer(window_size, device, num_classes=num_classes)
    elif model_type == 'blstm':
      model = BLSTM(window_size, num_classes=num_classes)
    elif model_type == 'multilayer' or model_type == 'mlp':
      model = Multilayer(window_size, num_classes=num_classes)
    elif model_type == 'cnn':
      model = CNN(window_size, num_classes=num_classes)
    model = model.to(device)

    # Plot loss progress
    if plot_loss:
      plot_metrics_loss(plot_loss, f'{model_id}_loss.png', device)

    # Run evaluation
    if plot_conf:
      data_management.load_checkpoint(model_fp, model, device)
      print(f'Testing... ({model_fp=})')
      evaluate(model, test_dl, device, lai_pred, lai_true, lai_prob)
      elapsed_t = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_t))
      print(f'Done testing, elapsed time = {elapsed_t}.')

  lai_pred = np.array(lai_pred)
  lai_true = np.array(lai_true)
  lai_prob = np.array(lai_prob)
  lai_pred, lai_true = smooth(lai_pred, lai_true, lai_prob, kernel_radius=15)

  if plot_conf:
    sn.set(font_scale=2.0)

    # Plot LAI confusion matrix
    if num_classes == 3:
      lai_classes = ['AFR', 'EUR', 'EAS']
    elif num_classes == 5:
      lai_classes = ['P0', 'P1', 'P2', 'P3', 'P4']

    lai_conf = confusion_matrix(lai_true, lai_pred).astype('float')
    lai_conf = lai_conf / lai_conf.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 7))
    sn.heatmap(lai_conf, annot=True, fmt='.3f',
      xticklabels=lai_classes, yticklabels=lai_classes, vmin=0, vmax=1)
    plt.savefig(f'{model_type}_{learning:.0e}_lai_confusion.png');

    acc = sum(1 for x,y in zip(lai_true, lai_pred) if x == y) / len(lai_true) * 100
    print(f'Overall LAI Acc = {acc}%')


def smooth(pred_mat, truth_mat, probs_mat, kernel_radius=2):
  smooth_pred  = []
  smooth_truth = []

  for win_idx in range(len(pred_mat)):
    left_lim  = max(0, win_idx - kernel_radius)
    right_lim = min(len(pred_mat) - 1, win_idx + kernel_radius) + 1

    for ind_idx in range(len(pred_mat[win_idx])):
      preds = pred_mat[left_lim:right_lim,ind_idx]
      probs = probs_mat[left_lim:right_lim,ind_idx]

      smooth = int(weighted_mode(preds, probs)[0][0])
      smooth_pred.append(smooth)
      smooth_truth.append(truth_mat[win_idx][ind_idx])

  return smooth_pred, smooth_truth


def plot_metrics_loss(metrics_fp, out_fp, device):
  t_loss_l, v_loss_l, steps_l = data_management.load_metrics(
    metrics_fp, device)
  plt.plot(steps_l, t_loss_l, label='Train')
  plt.plot(steps_l, v_loss_l, label='Valid')
  plt.xlabel('Steps')
  plt.ylabel('Loss')
  plt.legend()
  plt.savefig(out_fp);


def evaluate(model, test_i, device, lai_pred, lai_true, lai_prob):
  lai_normalizer = nn.Softmax(dim=1)

  lai_win_pred = []
  lai_win_true = []
  lai_win_prob = []

  model.eval()
  with torch.no_grad():
    for (pop_lab, snp) in test_i:
      snp     = snp.to(device)
      lai_out = model(snp)

      # LAI results
      lai_out = lai_normalizer(lai_out).tolist()
      lai_idx = np.argmax(lai_out, axis=1).tolist()
      lai_win_pred.extend(lai_idx)
      lai_win_true.extend(pop_lab)
      lai_win_prob.extend([x[idx] for x, idx in zip(lai_out, lai_idx)])

  lai_pred.append(lai_win_pred)
  lai_true.append(lai_win_true)
  lai_prob.append(lai_win_prob)

  acc = sum(1 for x,y in zip(lai_win_pred, lai_win_true) if x == y) / len(lai_win_pred) * 100
  print(f'LAI Acc = {acc}%')


def handle_args():
  parser = argparse.ArgumentParser(description='Model evaluation.')
  parser.add_argument('-md','--model-data', required=True,
    help='Data directory with model checkpoints', type=str)
  parser.add_argument('-m','--model', required=True,
    help='Model architecture', type=str)
  parser.add_argument('-l','--learning', required=True,
    help='Model learning rate', type=float)
  parser.add_argument('-o','--optimizer', required=True,
    help='Model optimizer', type=str)
  parser.add_argument('-d','--directory', required=True,
    help='Directory with preprocessed CSV files', type=str)
  parser.add_argument('-c','--classes',
    help='Number of classes to predict', required=True, type=int)
  parser.add_argument('-s','--start-window',
    help='Window index to start evaluation at', default=0, type=int)
  parser.add_argument('-w','--windows',
    help='Number of windows to evaluate', default=1, type=int)
  parser.add_argument('--plot-loss', required=False,
    help='Plot train / valid loss', action=argparse.BooleanOptionalAction)
  parser.add_argument('--confusion',
    help='Plot confusion matrix', action=argparse.BooleanOptionalAction)
  parser.add_argument('-a', '--admixed',
    help='Indicates the windows belong to admixed individuals',
    action=argparse.BooleanOptionalAction)

  return vars(parser.parse_args())


def welcome():
  print(r'''
   _____  _____  _____  _____  _____  _   _   ___   _____  _____ ______
  |_   _||  ___|/  ___||_   _||_   _|| \ | | / _ \ |_   _||  _  || ___ \
    | |  | |__  \ `--.   | |    | |  |  \| |/ /_\ \  | |  | | | || |_/ /
    | |  |  __|  `--. \  | |    | |  | . ` ||  _  |  | |  | | | ||    /
    | |  | |___ /\__/ /  | |   _| |_ | |\  || | | |  | |  \ \_/ /| |\ \
    \_/  \____/ \____/   \_/   \___/ \_| \_/\_| |_/  \_/   \___/ \_| \_|

''')


if __name__ == '__main__':
  welcome()
  main()
