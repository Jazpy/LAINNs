import torch
import argparse
import time
import os
import data_management
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from model import BLSTM, Multilayer, Transformer

def main():
  start_t = time.time()
  print(f'Found device: {"cuda" if torch.cuda.is_available() else "cpu"}')
  device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Argument handling
  args        = handle_args()
  model_fp    = args['model']
  data_dir    = args['directory']
  plot_loss   = args['plot_loss']
  plot_conf   = args['confusion']
  num_classes = args['classes']
  win_index   = args['window_index']

  # Data loading
  if win_index:
    test_fn = f'win_admix_{win_index}.csv'
  else:
    test_fn = f'win_admix.csv'

  print('Creating data loader...')
  test_dl, window_size = data_management.create_testing_loader(
    data_dir, test_fn)
  elapsed_t = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_t))
  print(f'Done creating data loader, elapsed time = {elapsed_t}.')

  # Model loading
  model_id   = os.path.basename(model_fp)
  model_toks = model_id.split('_')
  model_type = model_toks[0]

  if model_type == 'transformer':
    model = Transformer(window_size, device)
  elif model_type == 'blstm':
    model = BLSTM(window_size)
  elif model_type == 'multilayer':
    model = Multilayer(window_size)

  model = model.to(device)

  # Plot loss progress
  if plot_loss:
    plot_metrics_loss(plot_loss, f'{model_id}_loss.png', device)

  # Run evaluation
  if plot_conf:
    data_management.load_checkpoint(model_fp, model, device)
    print(f'Testing... ({model_fp=})')
    evaluate(model, test_dl, device, f'{model_id}_confusion.png')
    elapsed_t = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_t))
    print(f'Done testing, elapsed time = {elapsed_t}.')

def plot_metrics_loss(metrics_fp, out_fp, device):
  t_loss_l, v_loss_l, steps_l = data_management.load_metrics(
    metrics_fp, device)
  plt.plot(steps_l, t_loss_l, label='Train')
  plt.plot(steps_l, v_loss_l, label='Valid')
  plt.xlabel('Steps')
  plt.ylabel('Loss')
  plt.legend()
  plt.savefig(out_fp);

def evaluate(model, test_i, device, out_fp):
  y_pred = []
  y_true = []
  normalizer = nn.Softmax(dim=1)
  pop_labels = [0, 1, 2]

  model.eval()
  with torch.no_grad():
    for (labels, snp) in test_i:
      labels = labels.to(device)
      snp    = snp.to(device)

      output = model(snp)
      output = normalizer(output)
      output = np.argmax(output.data.cpu(), axis=1)

      y_pred.extend(output.tolist())
      y_true.extend(labels.tolist())

  # Plot confusion matrix
  classes     = ['AFR', 'EUR', 'EAS']
  conf_matrix = confusion_matrix(y_true, y_pred).astype('float')
  conf_matrix = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]
  plt.figure(figsize=(12, 7))
  sn.heatmap(conf_matrix, annot=True, fmt='.2f',
    xticklabels=classes, yticklabels=classes)
  plt.savefig(out_fp);

def handle_args():
  parser = argparse.ArgumentParser(description='Model evaluation.')
  parser.add_argument('-m','--model', required=True,
    help='Model checkpoint to evaluate', type=str)
  parser.add_argument('-d','--directory', required=True,
    help='Directory with preprocessed CSV files', type=str)
  parser.add_argument('-l','--plot-loss', required=False,
    help='Model checkpoint to plot loss of', default='', type=str)
  parser.add_argument('-c','--classes',
    help='Number of classes to predict', required=True, type=int)
  parser.add_argument('-w','--window-index',
    help='Window index this model was trained for', default='', type=str)
  parser.add_argument('--confusion',
    help='Plot confusion matrix', action=argparse.BooleanOptionalAction)

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
