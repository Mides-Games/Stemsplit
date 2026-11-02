//! Our separation network: a small spectrogram U-Net that predicts one soft
//! mask per stem and channel. Written from scratch; `train/model.py` is its
//! PyTorch twin and must keep the same parameter names and shapes.
//!
//! Input  `[B, 2, 1024, T]` = log1p(|STFT|) of the mix, bins 0..1024 (Nyquist dropped).
//! Output `[B, 4, 2, 1024, T]` masks, softmax over the stem axis, so stems sum to the mix.
//! `T` must be a multiple of 4.
//!
//! The weights load into candle's modules (the names and shapes `train/` exports);
//! the forward pass runs on [`crate::conv`], which computes the same convolutions
//! about ten times faster on the CPU. [`UNet::forward_candle`] keeps candle's own
//! pass as the reference the tests hold the fast one to.

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    conv2d, conv_transpose2d, Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, Module,
    VarBuilder,
};

use crate::conv::{act_of, Act, Conv, ConvUp};

pub const STEMS: usize = 4;
pub const MODEL_BINS: usize = 1024;

pub struct UNet {
    enc1: Conv2d,
    down1: Conv2d,
    down2: Conv2d,
    mid: Conv2d,
    up1: ConvTranspose2d,
    dec1: Conv2d,
    up2: ConvTranspose2d,
    dec2: Conv2d,
    out: Conv2d,
    fast: Fast,
    pub device: Device,
}

/// The same layers as [`crate::conv`] runs them.
struct Fast {
    enc1: Conv,
    down1: Conv,
    down2: Conv,
    mid: Conv,
    up1: ConvUp,
    dec1: Conv,
    up2: ConvUp,
    dec2: Conv,
    out: Conv,
}

fn same(p: usize) -> Conv2dConfig {
    Conv2dConfig {
        padding: p,
        ..Default::default()
    }
}

fn down() -> Conv2dConfig {
    Conv2dConfig {
        padding: 1,
        stride: 2,
        ..Default::default()
    }
}

fn up() -> ConvTranspose2dConfig {
    ConvTranspose2dConfig {
        padding: 1,
        stride: 2,
        ..Default::default()
    }
}

impl UNet {
    pub fn new(vb: VarBuilder, device: Device) -> Result<Self> {
        let enc1 = conv2d(2, 16, 3, same(1), vb.pp("enc1"))?;
        let down1 = conv2d(16, 32, 4, down(), vb.pp("down1"))?;
        let down2 = conv2d(32, 64, 4, down(), vb.pp("down2"))?;
        let mid = conv2d(64, 64, 3, same(1), vb.pp("mid"))?;
        let up1 = conv_transpose2d(64, 32, 4, up(), vb.pp("up1"))?;
        let dec1 = conv2d(64, 32, 3, same(1), vb.pp("dec1"))?;
        let up2 = conv_transpose2d(32, 16, 4, up(), vb.pp("up2"))?;
        let dec2 = conv2d(32, 16, 3, same(1), vb.pp("dec2"))?;
        let out = conv2d(16, STEMS * 2, 1, same(0), vb.pp("out"))?;
        let fast = Fast {
            enc1: Conv::new(&enc1)?,
            down1: Conv::new(&down1)?,
            down2: Conv::new(&down2)?,
            mid: Conv::new(&mid)?,
            up1: ConvUp::new(&up1)?,
            dec1: Conv::new(&dec1)?,
            up2: ConvUp::new(&up2)?,
            dec2: Conv::new(&dec2)?,
            out: Conv::new(&out)?,
        };
        Ok(UNet {
            enc1,
            down1,
            down2,
            mid,
            up1,
            dec1,
            up2,
            dec2,
            out,
            fast,
            device,
        })
    }

    /// Load weights exported by `train/export.py` (safetensors). Safe, buffered read.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let device = Device::Cpu;
        let vb = VarBuilder::from_buffered_safetensors(bytes, DType::F32, &device)?;
        Self::new(vb, device)
    }

    /// Mask logits of one `[2][f * t]` input: `[stem * 2 + channel][f * t]`,
    /// before the softmax ([`mask_at`]).
    pub fn logits(&self, x: &Act) -> Result<Act> {
        let n = &self.fast;
        let e1 = n.enc1.forward(x, true)?;
        let e2 = n.down1.forward(&e1, true)?;
        let e3 = n.down2.forward(&e2, true)?;
        let m = n.mid.forward(&e3, true)?;
        let u1 = n.up1.forward(&m, true)?;
        let d1 = n.dec1.forward(&Act::cat(u1, &e2), true)?;
        let u2 = n.up2.forward(&d1, true)?;
        let d2 = n.dec2.forward(&Act::cat(u2, &e1), true)?;
        n.out.forward(&d2, false)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _, f, t) = x.dims4()?;
        let plane = f * t;
        let mut items = Vec::with_capacity(b);
        for i in 0..b {
            let l = self.logits(&act_of(x, i)?)?;
            let mut y = vec![0.0f32; STEMS * 2 * plane];
            for s in 0..STEMS {
                for c in 0..2 {
                    for (j, v) in y[(s * 2 + c) * plane..][..plane].iter_mut().enumerate() {
                        *v = mask_at(&l.data, plane, s, c, j);
                    }
                }
            }
            items.push(Tensor::from_vec(y, (STEMS, 2, f, t), &self.device)?);
        }
        Ok(Tensor::stack(&items, 0)?)
    }

    /// candle's own forward pass: the reference for [`UNet::forward`].
    pub fn forward_candle(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _, f, t) = x.dims4()?;
        let e1 = self.enc1.forward(x)?.relu()?;
        let e2 = self.down1.forward(&e1)?.relu()?;
        let e3 = self.down2.forward(&e2)?.relu()?;
        let m = self.mid.forward(&e3)?.relu()?;
        let u1 = self.up1.forward(&m)?.relu()?;
        let d1 = self.dec1.forward(&Tensor::cat(&[&u1, &e2], 1)?)?.relu()?;
        let u2 = self.up2.forward(&d1)?.relu()?;
        let d2 = self.dec2.forward(&Tensor::cat(&[&u2, &e1], 1)?)?.relu()?;
        let o = self.out.forward(&d2)?.reshape((b, STEMS, 2, f, t))?;
        Ok(candle_nn::ops::softmax(&o, 1)?)
    }
}

/// Softmax over the stem axis of `[stem * 2 + c][plane]` logits, at `stem`, `c`, `i`;
/// what candle's `softmax(o, 1)` computes, in the same order.
pub fn mask_at(logits: &[f32], plane: usize, stem: usize, c: usize, i: usize) -> f32 {
    let l = |s: usize| logits[(s * 2 + c) * plane + i];
    let max = (0..STEMS).map(l).fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = (0..STEMS).map(|s| (l(s) - max).exp()).sum();
    (l(stem) - max).exp() / sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn masks_have_the_shape_and_sum_to_one() {
        let device = Device::Cpu;
        let vm = VarMap::new();
        let net = UNet::new(
            VarBuilder::from_varmap(&vm, DType::F32, &device),
            device.clone(),
        )
        .unwrap();
        let x = Tensor::rand(0f32, 1f32, (1, 2, 64, 16), &device).unwrap();
        let y = net.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 4, 2, 64, 16]);
        let s: Vec<f32> = y.sum(1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert!(s.iter().all(|v| (v - 1.0).abs() < 1e-5));
    }

    #[test]
    fn fast_forward_matches_candles() {
        let device = Device::Cpu;
        let vm = VarMap::new();
        let net = UNet::new(
            VarBuilder::from_varmap(&vm, DType::F32, &device),
            device.clone(),
        )
        .unwrap();
        // Two items, frames not a power of two.
        let x = Tensor::rand(0f32, 3f32, (2, 2, 256, 40), &device).unwrap();
        let diff: f32 = (net.forward(&x).unwrap() - net.forward_candle(&x).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff < 1e-5, "max abs difference {diff}");
    }

    /// Needs models/parity.safetensors from `python train/parity.py`.
    #[test]
    #[ignore]
    fn parity_with_pytorch() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/parity.safetensors"
        );
        let bytes = std::fs::read(path).expect("run train/parity.py first");
        let device = Device::Cpu;
        let vb = VarBuilder::from_buffered_safetensors(bytes, DType::F32, &device).unwrap();
        let x = vb.get((1, 2, MODEL_BINS, 32), "parity.x").unwrap();
        let want = vb.get((1, 4, 2, MODEL_BINS, 32), "parity.y").unwrap();
        let net = UNet::new(vb, device).unwrap();
        let diff: f32 = (net.forward(&x).unwrap() - want)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff < 1e-4, "max abs difference {diff}");
    }
}
