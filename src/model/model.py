import torch
import math
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, DataParallel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BLSTM(nn.Module):
  def __init__(self, in_size, num_classes=3, hidden=1024):
    super(BLSTM, self).__init__()

    self.num_features = in_size
    self.hidden_size  = hidden
    self.num_classes  = num_classes

    self.lstm = nn.LSTM(input_size=self.num_features,
      hidden_size=self.hidden_size, num_layers=5,
      batch_first=True, bidirectional=True)

    self.multilayer = nn.Sequential(
      nn.Linear(2 * self.hidden_size, 512), nn.ReLU(),
      nn.Linear(512,                  256), nn.ReLU(),
      nn.Linear(256,                  128), nn.ReLU(),
      nn.Linear(128,                  64),  nn.ReLU(),
      nn.Linear(64,                   self.num_classes))

  def forward(self, snp):
    output, _ = self.lstm(snp)
    pop_pred  = self.multilayer(output)

    return pop_pred

class Multilayer(nn.Module):
  def __init__(self, in_size, num_classes=3):
    super(Multilayer, self).__init__()

    self.num_classes = num_classes

    self.layers = nn.Sequential(
      nn.Linear(in_size, 512), nn.ReLU(),
      nn.Linear(512,     256), nn.ReLU(),
      nn.Linear(256,     128), nn.ReLU(),
      nn.Linear(128,     64),  nn.ReLU(),
      nn.Linear(64,      32),  nn.ReLU(),
      nn.Linear(32,      16),  nn.ReLU(),
      nn.Linear(16,      self.num_classes))

  def forward(self, snp):
    pop_pred = self.layers(snp)

    return pop_pred

class Transformer(nn.Module):
  def __init__(self, in_size, dev, num_heads=8, num_classes=3):
    super(Transformer, self).__init__()

    self.num_classes = num_classes
    self.num_heads   = num_heads
    self.input_size  = in_size
    self.src_mask    = self.__gen_square_subseq_mask(self.input_size, dev)

    self.pos_encoder = self.__PositionalEncoding(self.input_size)
    encoder_layers   = TransformerEncoderLayer(self.input_size,
      self.num_heads)
    self.transformer_encoder = TransformerEncoder(encoder_layers, 6)
    self.decoder_0 = nn.Linear(self.input_size, self.input_size // 2)
    self.decoder_1 = nn.Linear(self.input_size // 2, self.num_classes)

    self.init_weights()

  def init_weights(self):
    initrange = 0.1
    self.decoder_0.bias.data.zero_()
    self.decoder_0.weight.data.uniform_(-initrange, initrange)
    self.decoder_1.bias.data.zero_()
    self.decoder_1.weight.data.uniform_(-initrange, initrange)

  def forward(self, snp):
    src = torch.permute(snp, (1, 0))
    src = torch.unsqueeze(src, dim=-1)
    src = self.pos_encoder(src)
    output = self.transformer_encoder(src, self.src_mask)
    output = self.decoder_0(output)
    output = self.decoder_1(output)
    output = output[-1,:,:]

    return output

  def __gen_square_subseq_mask(self, sz, dev):
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(dev)

  class __PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
      super().__init__()
      self.dropout = nn.Dropout()

      position = torch.arange(max_len).unsqueeze(1)
      div_term = torch.exp(torch.arange(0, d_model, 2) *
        (-math.log(10000.0) / d_model))
      pe = torch.zeros(max_len, 1, d_model)
      pe[:, 0, 0::2] = torch.sin(position * div_term)
      pe[:, 0, 1::2] = torch.cos(position * div_term)
      self.register_buffer('pe', pe)

    def forward(self, x):
      x = x + self.pe[:x.size(0)]
      return self.dropout(x)

# simple CNN test
class CNN(nn.Module):
  def __init__(self, in_size, num_classes=3, hidden=1024):
    super(CNN, self).__init__()

    self.num_features = in_size
    self.num_classes  = num_classes

    self.conv_0 = nn.Sequential(
      nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3),
      nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3),
      nn.MaxPool1d(kernel_size=2, stride=2))
    self.conv_1 = nn.Sequential(
      nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3),
      nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3),
      nn.MaxPool1d(kernel_size=2, stride=2))
    self.conv_2 = nn.Sequential(
      nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3),
      nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3),
      nn.MaxPool1d(kernel_size=2, stride=2))
    self.conv_3 = nn.Sequential(
      nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
      nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
      nn.MaxPool1d(kernel_size=2, stride=2))

    self.fc = nn.Sequential(
      nn.Linear(3840, 512), nn.ReLU(),
      nn.Linear(512,  128), nn.ReLU(),
      nn.Linear(128,  16),  nn.ReLU(),
      nn.Linear(16,   self.num_classes))

  def forward(self, snp):
    src = torch.unsqueeze(snp, dim=-1)
    src = torch.permute(src, (0, 2, 1))

    out = self.conv_0(src)
    out = self.conv_1(out)
    out = self.conv_2(out)
    out = self.conv_3(out)

    out = out.reshape(out.size(0), -1)
    out = self.fc(out)

    return out
