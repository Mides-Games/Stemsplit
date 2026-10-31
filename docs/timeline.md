# Timeline

## 0.1.0 — 2026-10-29

### Changed
- First version: Rust workspace (`stemsplit` library, `stemsplit` CLI with
  `separate`, `synth`, `eval`), our own spectrogram U-Net (candle) and an HPSS
  baseline sharing one mask/iSTFT path; PyTorch twin and trainer in `train/`.

### Decisions
- **Dataset: self-generated synthetic mixes**, not MUSDB18. MUSDB18 (and
  MUSDB18-HQ) is licensed for non-commercial research only, which the owner's
  licence rule excludes. Synthetic stems have exact ground truth and no licence
  question; their limit is realism (no real voices, rooms or mastering), so
  numbers on them do not predict real songs. Next candidates to check, licence
  first: CC-BY / CC0 multitracks (e.g. stems released under CC-BY), or our own
  recorded stems.
- **Architecture: magnitude-mask U-Net**, not a hybrid waveform model. Smallest
  thing that can learn drums vs the rest, trains on CPU or one GPU, and its masks
  sum to the mix by construction. HTDemucs-style hybrid is a later step once the
  data pipeline proves itself.
- **Inference: candle**, pure Rust, no native runtime to ship (ort needs onnxruntime).
  Weights read with `from_buffered_safetensors` (no `unsafe` mmap).
- Resampling is linear interpolation; a band-limited resampler is a follow-up.

### Measured
- Tests: `cargo test --release` 8 passed, 1 ignored (the PyTorch parity test,
  needs `train/parity.py`, which needs torch: not installed locally).
- STFT → iSTFT round trip: max error < 1e-4.
- HPSS baseline, 20 synthetic tracks × 10 s, seeds 100000–100019, plain SDR
  (not BSS-Eval), median dB, mixture-as-estimate in brackets:
  drums 2.72 [0.53], bass −1.14 [−5.08], vocals 0.00 (silent stem) [−8.34],
  other −1.30 [−8.99]. 200 s of audio in 14 s on this machine.
- PyTorch parity (on the server, `train/parity.py` + `--ignored parity`): Rust
  forward pass equals PyTorch's within 1e-4.
- U-Net, training on the owner's server (6 CPU cores, no GPU, shared with other
  services, `nice 19`, 4 threads): 200 synthetic tracks × 10 s, seeds 0–199,
  about 10.5 min per epoch. After **epoch 1 of 30**, same 20 test tracks:
  drums 7.87, bass 2.94, vocals 0.00, other −0.73 (dB). Vocals at exactly 0.00
  means the vocals mask is still near zero; to be re-checked at the end of the run.
  200 s of audio separated in 97 s on this machine.
- After **epoch 3**: drums 11.24, bass 6.48, vocals 0.01, other 0.01 (dB).
- After **epoch 10**: drums 13.30, bass 6.89, vocals 0.01, other 1.21 (dB).
- After **epoch 30** (2026-10-30, loss 0.0695; epochs 21-30 resumed from the
  epoch-20 weights, Adam restarted): drums **13.53**, bass **8.64**, vocals
  **12.27**, other **10.57** (dB), same 20 test tracks. The vocals mask, flat
  through epoch 10, was learned between epochs 10 and 30. Synthetic mixes only:
  real songs will score lower; Overtone's bands stage is the real-audio check.

### Fixed
- The first run died after epoch 3 with nothing in its log, most likely killed
  for memory: the trainer held every track in RAM on a server with 11 GB and no
  swap, beside Overtone's measurement runs (which were cut short at the same
  time). Tracks are now memory-mapped and read per crop (resident ~0.9 GB), and
  `--resume` / `--start-epoch` continue from a checkpoint. Training now waits
  for Overtone's run queue to be empty.
