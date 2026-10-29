# Stemsplit

Split a song into **drums, bass, vocals and other** — our own code, in Rust.
Drums come first: Overtone's roadmap step 28.3 feeds a drums-only signal to its
beat tracker on live band songs.

```bash
cargo build --release
target/release/stemsplit separate song.mp3 --out stems            # all four stems as WAV
target/release/stemsplit separate song.mp3 --out stems --stems drums
target/release/stemsplit synth --out data/test --count 20 --seed 100000
target/release/stemsplit eval --data data/test --model hpss        # median SDR per stem
```

## How it works

Decode (Symphonia) → stereo 44.1 kHz → STFT (2048 / 512, Hann) → one soft mask
per stem and channel → iSTFT. Masks sum to one, so the stems always add back to
the mix. Two mask sources:

- **`unet`** — our spectrogram U-Net (`crates/stemsplit/src/model.rs`, inference
  with `candle`), weights trained by `train/train.py`, loaded from
  `models/stemsplit.safetensors`. Written from scratch; no Demucs code or weights.
- **`hpss`** — harmonic/percussive median filtering (no weights). The fallback
  when no model file exists; gives drums, bass (harmonic < 200 Hz) and other, no vocals.

## Library (for Overtone)

```rust
let sep = stemsplit::Separator::load(Path::new("models/stemsplit.safetensors"))
    .unwrap_or(stemsplit::Separator::Hpss);
let drums: Vec<f32> = stemsplit::drums_mono(&sep, &interleaved, channels, rate)?;
let stems = sep.separate(&stemsplit::decode(path)?)?;   // Stems { drums, bass, vocals, other }
```

## Training

Training data is **self-generated** (`stemsplit synth`): procedural drums,
bass, formant voices, pads and plucked strings with exact ground truth. MUSDB18
is non-commercial and is not used. Training is Python/PyTorch only; inference is Rust.

```bash
stemsplit synth --out data/train --count 2000 --seed 0
pip install -r train/requirements.txt
python train/train.py --data data/train --out models/stemsplit.safetensors
python train/parity.py && cargo test --release -- --ignored parity   # Rust == PyTorch
stemsplit eval --data data/test --model models/stemsplit.safetensors
```

## Licences

Our code: MIT OR Apache-2.0. Dependencies: candle, rustfft, hound, clap, anyhow
(MIT/Apache); Symphonia (MPL-2.0, file-level copyleft, commercial use allowed,
unmodified). Training: PyTorch, NumPy, SciPy, safetensors (BSD/Apache). No
non-commercial or AGPL parts, data or weights.

Measured numbers live in [docs/timeline.md](docs/timeline.md).
