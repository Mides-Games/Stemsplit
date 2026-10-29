"""Train the U-Net on folders written by `stemsplit synth` (or any folder of
<track>/{mixture,drums,bass,vocals,other}.wav at 44.1 kHz) and export
safetensors for the Rust CLI.

    stemsplit synth --out data/train --count 2000 --seed 0
    python train/train.py --data data/train --out models/stemsplit.safetensors

The eval set uses seeds from 100000 up, so it never overlaps training seeds.
Loss: L1 on magnitudes, drums weighted x2 (the stem Overtone needs)."""
import argparse
import pathlib
import random

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from scipy.io import wavfile

from model import BINS, HOP, UNet, spec

STEMS = ["drums", "bass", "vocals", "other"]
WEIGHTS = torch.tensor([2.0, 1.0, 1.0, 1.0])
FRAMES = 256
CROP = (FRAMES - 1) * HOP


def load(p):
    # Memory-mapped: only the crop is read, so RAM does not grow with the dataset.
    # Holding every track in memory got the run killed on the shared server.
    rate, x = wavfile.read(p, mmap=True)
    assert rate == 44100, f"{p}: {rate} Hz"
    return x  # [N, 2] float32, on disk


class Tracks(torch.utils.data.Dataset):
    def __init__(self, root):
        self.dirs = sorted(d for d in pathlib.Path(root).iterdir() if (d / "mixture.wav").exists())

    def __len__(self):
        return len(self.dirs) * 4

    def __getitem__(self, i):
        d = self.dirs[i % len(self.dirs)]
        waves = [load(d / f"{s}.wav") for s in STEMS]
        o = random.randint(0, waves[0].shape[0] - CROP - 1)
        stems = torch.stack([torch.from_numpy(np.array(w[o:o + CROP].T, dtype=np.float32)) for w in waves])
        stems = stems * random.uniform(0.5, 1.2)  # [4, 2, CROP]
        return stems.sum(0), stems  # mixture [2, N], stems [4, 2, N]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/stemsplit.safetensors")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--resume", help="safetensors to start from (a checkpoint of this script)")
    ap.add_argument("--start-epoch", type=int, default=0, help="epochs already done, for the log")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads, 0 = torch default")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    random.seed(0)
    net = UNet()
    if a.resume:
        net.load_state_dict(load_file(a.resume))
    net = net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    dl = torch.utils.data.DataLoader(Tracks(a.data), batch_size=a.batch, shuffle=True, num_workers=0)
    w = WEIGHTS.to(dev).view(1, 4, 1, 1, 1)
    for epoch in range(a.start_epoch, a.epochs):
        total = 0.0
        for mix, stems in dl:
            mix, stems = mix.to(dev), stems.to(dev)
            m = spec(mix)[:, :, :BINS].abs()  # [B, 2, 1024, T]
            b = stems.shape[0]
            target = spec(stems.reshape(b * 4, 2, -1))[:, :, :BINS].abs().reshape(b, 4, 2, BINS, -1)
            masks = net(torch.log1p(m))
            loss = (w * (masks * m.unsqueeze(1) - target).abs()).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch + 1}: loss {total / len(dl):.5f}", flush=True)
        save(net, a.out)  # every epoch, so a stopped run keeps its progress
    print("wrote", a.out)


def save(net, out):
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    tmp = out + ".tmp"
    save_file({k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}, tmp)
    pathlib.Path(tmp).replace(out)


if __name__ == "__main__":
    main()
