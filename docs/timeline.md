# Timeline

## 0.2.0 — 2026-11-02

### Changed
- **The U-Net's convolutions run in our own code** (`crates/stemsplit/src/conv.rs`):
  im2col of one output row and one matrix product per row (candle's gemm), the
  rows in parallel; a stride-2 transposed convolution as four 2x2 convolutions,
  one per output parity. Weights still load through candle's modules, and
  candle's own forward pass stays as `UNet::forward_candle`, the reference the
  tests hold the new one to (within 1e-5). candle 0.9's CPU path ran this network
  at about 11 GFLOP/s: `conv2d` builds 512-pixel im2col tiles from a channels-last
  copy made on one thread, and `conv_transpose2d` is a scalar loop.
- **Only the stems asked for are computed.** `Separator::separate_stems(mix, &[..])`
  builds the masks and runs the iSTFT for those stems alone; `--stems drums` and
  `drums_mono` use it. The masks multiply the spectrogram inside the iSTFT instead
  of in a masked copy, and the two channels' STFTs and iSTFTs run in parallel.
- New dependency: `rayon` (MIT/Apache), already in the tree through candle.

### Measured
Calm Down Juliet (osu! beatmap 1109747, `audio.mp3`, 279.9 s, 44.1 kHz), `stemsplit
separate --stems drums` with the 30-epoch model, on the owner's PC (AMD Ryzen 5
4600G, 6 cores / 12 threads). Peak working set read from outside the process every
200 ms; stage times from a temporarily instrumented build of each version.

| stage                     | 0.1.0     | 0.2.0    |
|---------------------------|-----------|----------|
| decode                    | 0.35 s    | 0.36 s   |
| resample + STFT           | 0.80 s    | 0.43 s   |
| network input             | 0.68 s    | 0.13 s   |
| network (U-Net, softmax)  | 98.1 s    | 12.1 s   |
| masks out, glue           | 1.3 s (4 stems) | 0.6 s (drums) |
| iSTFT + resample          | 2.99 s (4 stems) | 0.47 s (drums) |
| write WAV                 | 0.13 s    | 0.12 s   |
| **wall**                  | **104.7 s** | **14.6 s** |
| CPU time                  | 358 s     | 90 s     |
| peak working set          | 1.94 GB   | 0.93 GB  |

- Wall time over runs: 0.1.0 104.7–111.5 s (3 runs, two instrumented); 0.2.0
  14.6 s (3 runs, one instrumented). 7.2–7.6x faster, half the memory. All four stems (`separate` without `--stems`):
  109.5 s → 18.3 s.
- Network layers, 0.1.0 → 0.2.0 (s over the song's 95 chunks): enc1 2.1 → 1.3,
  down1 12.8 → 0.9, down2 4.8 → 0.6, mid 2.8 → 0.5, up1 6.5 → 1.4, dec1 13.0 → 1.7,
  up2 11.4 → 2.1, dec2 38.2 → 3.0, out 2.0 → 0.5, softmax 4.5 → 0.2. The network
  is 11.9 GFLOP a chunk, 1.13 TFLOP for this song: ~11 GFLOP/s before, ~90 after.
- **Output unchanged within float rounding**: against 0.1.0's output on the same
  song, drums SDR 127.0 dB (max |difference| 6.0e-7 on a peak of 1.01); bass
  130.4, vocals 129.2, other 129.9 dB. Not bit-identical: the sums run in a
  different order.
- Tests: `cargo test --release` 11 passed, 1 ignored (the PyTorch parity test):
  three new ones hold `conv.rs` and the fast forward pass to candle's.
- Not measured: the owner's server (where 0.1.0 took 140 s wall and 2.07 GB on
  this song), and Overtone's stage scores, which this change cannot move beyond
  the 6e-7 above.

### Rejected / tried and dropped
- Bands of several output rows per matrix product (4096 to 16384 columns): 27–34 s
  wall against 14.6 s for one row. A band's im2col (up to 19 MB) falls out of
  cache; one row's (330 KB for dec2) stays in it.
- Turning off gemm's own threading inside the parallel rows
  (`gemm::set_threading_threshold`): no difference at one row per product
  (14.9 s either way, at one or two rows), so the global setting was left alone.

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
