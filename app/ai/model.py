import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv(nn.Module):
    def __init__(self, out_channels=128, kernel_size=1024, sample_rate=16000):
        super().__init__()
        self.kernel_size = kernel_size + (1 - kernel_size % 2)  # odd 보장
        self.out_channels = out_channels
        self.sample_rate = sample_rate

        # Mel 스케일로 필터 초기화
        f_low, f_high = 30.0, sample_rate / 2 - 30.0
        mel = np.linspace(
            2595 * np.log10(1 + f_low / 700),
            2595 * np.log10(1 + f_high / 700),
            out_channels + 1
        )
        hz = 700 * (10 ** (mel / 2595) - 1)
        self.f_low = nn.Parameter(torch.FloatTensor(hz[:-1]).view(-1, 1))
        self.f_band = nn.Parameter(torch.FloatTensor(np.diff(hz)).view(-1, 1))

        half = self.kernel_size // 2
        n = torch.arange(1, half + 1, dtype=torch.float32)
        self.register_buffer('n_', n.view(1, -1))
        self.register_buffer('window_', (0.54 - 0.46 * torch.cos(np.pi * n / half)).view(1, -1))

    def forward(self, x):
        f1 = 30 + torch.abs(self.f_low)
        f2 = torch.clamp(f1 + torch.abs(self.f_band), min=f1 + 50, max=self.sample_rate / 2)

        def sinc(f):
            return torch.sin(2 * np.pi * f * self.n_ / self.sample_rate) / (np.pi * self.n_)

        right = (sinc(f2) - sinc(f1)) * self.window_   # (out_ch, half)
        center = 2 * (f2 - f1) / self.sample_rate       # (out_ch, 1)
        filters = torch.cat([torch.flip(right, [1]), center, right], dim=1)
        return F.conv1d(x, filters.unsqueeze(1), padding=self.kernel_size // 2)


class FMS(nn.Module):
    """Feature Map Scaling: 채널별 중요도 가중치"""
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Linear(channels, channels)

    def forward(self, x):
        s = torch.sigmoid(self.scale(x.mean(-1)))
        return x * s.unsqueeze(-1) + s.unsqueeze(-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, first=False):
        super().__init__()
        self.first = first
        if not first:
            self.bn0 = nn.BatchNorm1d(in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.pool = nn.MaxPool1d(3)
        self.fms = FMS(out_ch)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else None

    def forward(self, x):
        identity = x
        if not self.first:
            x = F.leaky_relu(self.bn0(x), 0.3)
            identity = x
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.3)
        out = self.conv2(out)
        if self.shortcut:
            identity = self.shortcut(identity)
        out = self.pool(out + identity)
        return self.fms(out)


class RawNet2(nn.Module):
    def __init__(self):
        super().__init__()
        # Stem
        self.sinc = SincConv(128, 1024, 16000)
        self.bn0 = nn.BatchNorm1d(128)

        # Residual blocks
        self.blocks = nn.Sequential(
            ResBlock(128, 128, first=True),
            ResBlock(128, 128),
            ResBlock(128, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
        )

        # Sequence modeling
        self.bn_out = nn.BatchNorm1d(256)
        self.gru = nn.GRU(256, 1024, num_layers=3, batch_first=True)
        self.fc = nn.Linear(1024, 2)

    def forward(self, x):
        # x: (batch, 1, time)
        x = torch.abs(self.sinc(x))
        x = F.leaky_relu(self.bn0(x), 0.3)
        x = F.max_pool1d(x, 3)
        x = self.blocks(x)
        x = F.leaky_relu(self.bn_out(x), 0.3)
        x = x.permute(0, 2, 1)         # (batch, time, channels)
        x, _ = self.gru(x)
        return self.fc(x[:, -1, :])    # 마지막 타임스텝만 사용
