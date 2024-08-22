import torch
import torch.nn as nn

from torch.nn.functional import relu


#######
# CNN #
#######

class CNNSmoother(nn.Module):
    def __init__(self, num_classes=3):
        super(CNNSmoother, self).__init__()

        self.num_classes = num_classes
        self.kernel_size = 3
        self.pool_k_size = 2

        self.e00 = nn.Conv1d(num_classes + 1,   16,  kernel_size=self.kernel_size, padding=1)
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

        self.u0  = nn.ConvTranspose1d(128, 64, kernel_size=self.pool_k_size, stride=2)
        self.d00 = nn.Conv1d(128, 64, kernel_size=self.kernel_size, padding=1)
        self.d01 = nn.Conv1d(64,  64, kernel_size=self.kernel_size, padding=1)
        self.u1  = nn.ConvTranspose1d(64,  32, kernel_size=self.pool_k_size, stride=2)
        self.d10 = nn.Conv1d(64,  32, kernel_size=self.kernel_size, padding=1)
        self.d11 = nn.Conv1d(32,  32, kernel_size=self.kernel_size, padding=1)
        self.u2  = nn.ConvTranspose1d(32, 16, kernel_size=self.pool_k_size, stride=2)
        self.d20 = nn.Conv1d(32,  16, kernel_size=self.kernel_size, padding=1)
        self.d21 = nn.Conv1d(16,  16, kernel_size=self.kernel_size, padding=1)
        self.oc  = nn.Conv1d(16,  self.num_classes, kernel_size=1)


    def forward(self, probs, cms):
        probs = probs.permute(0, 2, 1)
        cms = torch.unsqueeze(cms, 1)
        probs = torch.cat((probs, cms), dim=1)

        xe00 = relu(self.e00(probs))
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

        # Decoder
        xu0 = self.u0(xe31)
        xu00 = torch.cat([xu0, xe21], dim=1)
        xd00 = relu(self.d00(xu00))
        xd01 = relu(self.d01(xd00))

        xu1 = self.u1(xd01)
        xu11 = torch.cat([xu1, xe11], dim=1)
        xd10 = relu(self.d10(xu11))
        xd11 = relu(self.d11(xd10))

        xu2 = self.u2(xd11)
        xu22 = torch.cat([xu2, xe01], dim=1)
        xd20 = relu(self.d20(xu22))
        xd21 = relu(self.d21(xd20))

        # Output layer
        out = self.oc(xd21)

        return out
