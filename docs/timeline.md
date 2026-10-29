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
- U-Net: **not measured** — not trained yet (no torch locally; training goes to
  the owner's server).
