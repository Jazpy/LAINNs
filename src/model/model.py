import torch
import math
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, DataParallel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


#######
# MLP #
#######

class Multilayer(nn.Module):
    def __init__(self, in_size, num_classes=3):
        super(Multilayer, self).__init__()

        in_size += 6

        self.num_classes = num_classes
        growth  = 4
        dropout = 0.3

        self.layers = nn.Sequential(
        nn.Linear(in_size,          in_size * growth), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(in_size * growth, in_size * growth), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(in_size * growth, in_size * growth), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(in_size * growth, self.num_classes))


    def forward(self, snp, aug):
        input    = torch.cat((snp, aug), dim=1)
        pop_pred = self.layers(input)
        return pop_pred


#######
# CNN #
#######

class CNN(nn.Module):
    def __init__(self, in_size, num_classes=3):
        super(CNN, self).__init__()

        self.num_features = in_size
        self.num_classes  = num_classes
        kern = 2
        dropout = 0.35

        self.conv = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3), nn.BatchNorm1d(64), nn.ReLU(),
        nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3), nn.BatchNorm1d(64), nn.ReLU(),
        nn.AvgPool1d(kernel_size=3),
        nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3), nn.BatchNorm1d(128), nn.ReLU(),
        nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3), nn.BatchNorm1d(128), nn.ReLU(),
        nn.AvgPool1d(kernel_size=3),
        nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3), nn.BatchNorm1d(256), nn.ReLU(),
        nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3), nn.BatchNorm1d(256), nn.ReLU(),
        nn.AvgPool1d(kernel_size=3),
        nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3), nn.BatchNorm1d(512), nn.ReLU(),
        nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3), nn.BatchNorm1d(512), nn.ReLU(),
        nn.AvgPool1d(kernel_size=3))

        self.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(5120, 5120), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(5120, 5120), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(5120, self.num_classes))


    def forward(self, snp):
        src = torch.unsqueeze(snp, dim=-1)
        src = torch.permute(src, (0, 2, 1))

        out = self.conv(src)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)

        return out


#########
# BLSTM #
#########

class BLSTM(nn.Module):
    def __init__(self, in_size, num_classes=3, hidden=1024):
        super(BLSTM, self).__init__()

        self.num_features = in_size
        self.hidden_size  = hidden
        self.num_classes  = num_classes
        dropout = 0.2

        self.lstm = nn.LSTM(input_size=self.num_features,
        hidden_size=self.hidden_size, num_layers=3,
        batch_first=True, bidirectional=True)

        self.multilayer = nn.Sequential(
        nn.Linear(self.hidden_size * 2, 2024), nn.ReLU(),
        #nn.Dropout(dropout),
        nn.Linear(2024,                 2024), nn.ReLU(),
        #nn.Dropout(dropout),
        nn.Linear(2024,                 2024), nn.ReLU(),
        #nn.Dropout(dropout),
        nn.Linear(2024,                 self.num_classes))


    def forward(self, snp):
        output, _ = self.lstm(snp)
        pop_pred  = self.multilayer(output)

        return pop_pred


###############
# TRANSFORMER #
###############

class Transformer(nn.Module):
    def __init__(self, in_size, dev, num_heads=4, num_classes=3):
        super(Transformer, self).__init__()

        self.num_classes = num_classes
        self.num_heads   = num_heads
        self.input_size  = in_size
        self.src_mask    = self.__gen_square_subseq_mask(self.input_size, dev)

        self.pos_encoder = self.__PositionalEncoding(self.input_size)
        encoder_layers   = TransformerEncoderLayer(self.input_size,
        self.num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 4)
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

