import torch
import math
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, DataParallel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.functional import relu


#######
# MLP #
#######

class Multilayer(nn.Module):
    def __init__(self, in_size, channels, num_classes=3):
        super(Multilayer, self).__init__()

        in_size *= channels

        self.num_classes = num_classes
        growth  = 2
        dropout = 0.1

        self.layers = nn.Sequential(
        nn.Linear(in_size,          in_size * growth), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(in_size * growth, in_size * growth), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(in_size * growth, in_size * growth), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(in_size * growth, self.num_classes))


    def forward(self, snp, aug):
        if aug.nelement() != 0:
            aug = torch.flatten(aug, start_dim=1)
            snp = torch.cat((snp, aug), dim=1)

        return self.layers(snp)


########
# UNET #
########

class UNet(nn.Module):
    def __init__(self, num_classes=3):
        super(UNet, self).__init__()

        self.num_classes = num_classes
        self.kernel_size = 3
        self.pool_k_size = 2

        self.e00 = nn.Conv1d(1,   16,  kernel_size=self.kernel_size, padding=1)
        self.e01 = nn.Conv1d(16,  16,  kernel_size=self.kernel_size, padding=1)
        self.p0  = nn.MaxPool1d(kernel_size=self.pool_k_size, stride=2)
        self.e10 = nn.Conv1d(16,  32,  kernel_size=self.kernel_size, padding=1)
        self.e11 = nn.Conv1d(32,  32,  kernel_size=self.kernel_size, padding=1)
        self.p1  = nn.MaxPool1d(kernel_size=self.pool_k_size, stride=2)
        self.e20 = nn.Conv1d(32,  64,  kernel_size=self.kernel_size, padding=1)
        self.e21 = nn.Conv1d(64,  64,  kernel_size=self.kernel_size, padding=1)
        self.p2  = nn.MaxPool1d(kernel_size=self.pool_k_size, stride=2)
        self.e30 = nn.Conv1d(64,  128, kernel_size=self.kernel_size, padding=1)
        self.e31 = nn.Conv1d(128, 128, kernel_size=self.kernel_size, padding=1)
        self.p3  = nn.MaxPool1d(kernel_size=self.pool_k_size, stride=2)
        self.e40 = nn.Conv1d(128, 256, kernel_size=self.kernel_size, padding=1)
        self.e41 = nn.Conv1d(256, 256, kernel_size=self.kernel_size, padding=1)

        self.u0  = nn.ConvTranspose1d(256, 128, kernel_size=self.pool_k_size, stride=2)
        self.d00 = nn.Conv1d(256, 128, kernel_size=self.kernel_size, padding=1)
        self.d01 = nn.Conv1d(128, 128, kernel_size=self.kernel_size, padding=1)
        self.u1  = nn.ConvTranspose1d(128, 64, kernel_size=self.pool_k_size, stride=2)
        self.d10 = nn.Conv1d(128, 64, kernel_size=self.kernel_size, padding=1)
        self.d11 = nn.Conv1d(64,  64, kernel_size=self.kernel_size, padding=1)
        self.u2  = nn.ConvTranspose1d(64,  32, kernel_size=self.pool_k_size, stride=2)
        self.d20 = nn.Conv1d(64,  32, kernel_size=self.kernel_size, padding=1)
        self.d21 = nn.Conv1d(32,  32, kernel_size=self.kernel_size, padding=1)
        self.u3  = nn.ConvTranspose1d(32, 16, kernel_size=self.pool_k_size, stride=2)
        self.d30 = nn.Conv1d(32,  16, kernel_size=self.kernel_size, padding=1)
        self.d31 = nn.Conv1d(16,  16, kernel_size=self.kernel_size, padding=1)
        self.oc  = nn.Conv1d(16,  self.num_classes, kernel_size=1)


    def forward(self, snp, aug):
        snp = torch.unsqueeze(snp, 1)

        if aug.nelement() != 0:
            snp = torch.cat((snp, aug), dim=1)

        xe00 = relu(self.e00(snp))
        xe01 = relu(self.e01(xe00))
        xp0 = self.p0(xe01)

        xe10 = relu(self.e10(xp0))
        xe11 = relu(self.e11(xe10))
        xp1 = self.p1(xe11)

        xe20 = relu(self.e20(xp1))
        xe21 = relu(self.e21(xe20))
        xp2 = self.p2(xe21)

        xe30 = relu(self.e30(xp2))
        xe31 = relu(self.e31(xe30))
        xp3 = self.p3(xe31)

        xe40 = relu(self.e40(xp3))
        xe41 = relu(self.e41(xe40))

        # Decoder
        xu0 = self.u0(xe41)
        xu00 = torch.cat([xu0, xe31], dim=1)
        xd00 = relu(self.d00(xu00))
        xd01 = relu(self.d01(xd00))

        xu1 = self.u1(xd01)
        xu11 = torch.cat([xu1, xe21], dim=1)
        xd10 = relu(self.d10(xu11))
        xd11 = relu(self.d11(xd10))

        xu2 = self.u2(xd11)
        xu22 = torch.cat([xu2, xe11], dim=1)
        xd20 = relu(self.d20(xu22))
        xd21 = relu(self.d21(xd20))

        xu3 = self.u3(xd21)
        xu33 = torch.cat([xu3, xe01], dim=1)
        xd30 = relu(self.d30(xu33))
        xd31 = relu(self.d31(xd30))

        # Output layer
        out = self.oc(xd31)

        return out


#######
# CNN #
#######

class CNN(nn.Module):
    def __init__(self, in_size, channels, num_classes=3):
        super(CNN, self).__init__()

        self.num_classes = num_classes
        self.kernel_size = 3
        self.pool_k_size = 3
        dropout = 0.1
        activation = nn.ReLU()

        self.conv = nn.Sequential(
        nn.Conv1d(in_channels=channels, out_channels=8, kernel_size=self.kernel_size), nn.BatchNorm1d(8), activation, nn.Dropout1d(p=dropout),
        nn.Conv1d(in_channels=8, out_channels=8, kernel_size=self.kernel_size), nn.BatchNorm1d(8), activation, nn.Dropout1d(p=dropout),
        nn.AvgPool1d(kernel_size=self.pool_k_size),
        nn.Conv1d(in_channels=8, out_channels=16, kernel_size=self.kernel_size), nn.BatchNorm1d(16), activation, nn.Dropout1d(p=dropout),
        nn.Conv1d(in_channels=16, out_channels=16, kernel_size=self.kernel_size), nn.BatchNorm1d(16), activation, nn.Dropout1d(p=dropout),
        nn.AvgPool1d(kernel_size=self.pool_k_size),
        nn.Conv1d(in_channels=16, out_channels=32, kernel_size=self.kernel_size), nn.BatchNorm1d(32), activation, nn.Dropout1d(p=dropout),
        nn.Conv1d(in_channels=32, out_channels=32, kernel_size=self.kernel_size), nn.BatchNorm1d(32), activation, nn.Dropout1d(p=dropout),
        nn.AvgPool1d(kernel_size=self.pool_k_size),
        nn.Conv1d(in_channels=32, out_channels=64, kernel_size=self.kernel_size), nn.BatchNorm1d(64), activation, nn.Dropout1d(p=dropout),
        nn.Conv1d(in_channels=64, out_channels=64, kernel_size=self.kernel_size), nn.BatchNorm1d(64), activation, nn.Dropout1d(p=dropout),
        nn.AvgPool1d(kernel_size=self.pool_k_size))

        fc_size = self.conv(torch.empty(1, channels, in_size))
        fc_size = fc_size.reshape(fc_size.size(0), -1).shape[1]
        self.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(fc_size, fc_size), activation,
        nn.Dropout(dropout),
        nn.Linear(fc_size, fc_size), activation,
        nn.Dropout(dropout),
        nn.Linear(fc_size, self.num_classes))


    def forward(self, snp, aug):
        snp = torch.unsqueeze(snp, 1)

        if aug.nelement() != 0:
            snp = torch.cat((snp, aug), dim=1)

        out = self.conv(snp)
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
        nn.Dropout(dropout),
        nn.Linear(2024,                 2024), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(2024,                 2024), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(2024,                 self.num_classes))


    def forward(self, snp, aug):
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


    def forward(self, snp, aug):
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
        def __init__(self, d_model, max_len=3000):
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

