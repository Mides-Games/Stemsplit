"""Write models/parity.safetensors: random weights plus one input and the
PyTorch output. `cargo test -- --ignored parity` loads it and checks the Rust
forward pass against it (max abs difference < 1e-4)."""
import torch
from safetensors.torch import save_file

from model import BINS, UNet

torch.manual_seed(1)
net = UNet().eval()
x = torch.rand(1, 2, BINS, 32)
with torch.no_grad():
    y = net(x)
t = {k: v.contiguous() for k, v in net.state_dict().items()}
t["parity.x"], t["parity.y"] = x, y
save_file(t, "models/parity.safetensors")
print("wrote models/parity.safetensors")
