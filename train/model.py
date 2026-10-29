"""PyTorch twin of crates/stemsplit/src/model.rs. Same names, same shapes:
the exported state_dict loads straight into the Rust UNet."""
import torch
from torch import nn

STEMS = 4
BINS = 1024
N_FFT = 2048
HOP = 512


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(2, 16, 3, padding=1)
        self.down1 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.down2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.mid = nn.Conv2d(64, 64, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec1 = nn.Conv2d(64, 32, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.dec2 = nn.Conv2d(32, 16, 3, padding=1)
        self.out = nn.Conv2d(16, STEMS * 2, 1)

    def forward(self, x):  # x: [B, 2, 1024, T] log1p magnitude -> masks [B, 4, 2, 1024, T]
        b, _, f, t = x.shape
        r = torch.relu
        e1 = r(self.enc1(x))
        e2 = r(self.down1(e1))
        e3 = r(self.down2(e2))
        m = r(self.mid(e3))
        d1 = r(self.dec1(torch.cat([r(self.up1(m)), e2], 1)))
        d2 = r(self.dec2(torch.cat([r(self.up2(d1)), e1], 1)))
        return torch.softmax(self.out(d2).reshape(b, STEMS, 2, f, t), 1)


def spec(wave):  # [B, 2, N] -> complex [B, 2, 1025, T], as crates/stemsplit/src/stft.rs
    b, c, n = wave.shape
    s = torch.stft(wave.reshape(b * c, n), N_FFT, HOP, window=torch.hann_window(N_FFT, device=wave.device),
                   center=True, pad_mode="reflect", return_complex=True)
    return s.reshape(b, c, s.shape[-2], s.shape[-1])
