//! Our separation network: a small spectrogram U-Net that predicts one soft
//! mask per stem and channel. Written from scratch; `train/model.py` is its
//! PyTorch twin and must keep the same parameter names and shapes.
//!
//! Input  `[B, 2, 1024, T]` = log1p(|STFT|) of the mix, bins 0..1024 (Nyquist dropped).
//! Output `[B, 4, 2, 1024, T]` masks, softmax over the stem axis, so stems sum to the mix.
//! `T` must be a multiple of 4.

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    conv2d, conv_transpose2d, Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, Module,
    VarBuilder,
};

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
    pub device: Device,
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
        Ok(UNet {
            enc1: conv2d(2, 16, 3, same(1), vb.pp("enc1"))?,
            down1: conv2d(16, 32, 4, down(), vb.pp("down1"))?,
            down2: conv2d(32, 64, 4, down(), vb.pp("down2"))?,
            mid: conv2d(64, 64, 3, same(1), vb.pp("mid"))?,
            up1: conv_transpose2d(64, 32, 4, up(), vb.pp("up1"))?,
            dec1: conv2d(64, 32, 3, same(1), vb.pp("dec1"))?,
            up2: conv_transpose2d(32, 16, 4, up(), vb.pp("up2"))?,
            dec2: conv2d(32, 16, 3, same(1), vb.pp("dec2"))?,
            out: conv2d(16, (STEMS * 2) as usize, 1, same(0), vb.pp("out"))?,
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

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
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
