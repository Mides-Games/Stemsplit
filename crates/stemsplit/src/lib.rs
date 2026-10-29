//! Stemsplit: split a song into drums, bass, vocals and other.
//!
//! Two separators share one path (STFT -> per-stem soft masks -> iSTFT):
//! - [`Separator::Network`]: our own U-Net ([`model::UNet`]), weights from `train/`.
//! - [`Separator::Hpss`]: harmonic/percussive median filtering, no weights. A
//!   baseline, and a usable drums estimate when no trained model is present.
//!
//! Masks sum to one in every bin, so the four stems always add back to the mix.
//!
//! For Overtone (roadmap step 28.3) the entry point is [`drums_mono`].

pub mod audio;
pub mod model;
pub mod stft;
pub mod synth;

use std::path::Path;

use anyhow::Result;
use candle_core::Tensor;

pub use audio::{decode, write_wav, Audio, SAMPLE_RATE};
use model::{UNet, MODEL_BINS, STEMS};
use stft::{istft, stft, Spec, BINS};

pub const STEM_NAMES: [&str; STEMS] = ["drums", "bass", "vocals", "other"];

#[derive(Clone, Debug)]
pub struct Stems {
    pub drums: Audio,
    pub bass: Audio,
    pub vocals: Audio,
    pub other: Audio,
}

impl Stems {
    pub fn get(&self, i: usize) -> &Audio {
        [&self.drums, &self.bass, &self.vocals, &self.other][i]
    }
}

pub enum Separator {
    Hpss,
    Network(UNet),
}

impl Separator {
    /// The network if `path` exists, otherwise `None`.
    pub fn load(path: &Path) -> Result<Separator> {
        Ok(Separator::Network(UNet::load(path)?))
    }

    pub fn name(&self) -> &'static str {
        match self {
            Separator::Hpss => "hpss",
            Separator::Network(_) => "unet",
        }
    }

    /// Separate any-rate stereo audio. Stems come back at the input's rate and length.
    pub fn separate(&self, mix: &Audio) -> Result<Stems> {
        let rate = mix.sample_rate;
        let work = mix.resampled(SAMPLE_RATE);
        let specs = [stft(&work.left), stft(&work.right)];
        let masks = match self {
            Separator::Hpss => hpss_masks(&specs),
            Separator::Network(net) => network_masks(net, &specs)?,
        };
        let mut out = Vec::with_capacity(STEMS);
        for stem in masks.iter() {
            let ch: Vec<Vec<f32>> = (0..2)
                .map(|c| {
                    let s = &specs[c];
                    let data = s.data.iter().zip(&stem[c]).map(|(z, m)| z * *m).collect();
                    istft(
                        &Spec {
                            frames: s.frames,
                            data,
                        },
                        work.len(),
                    )
                })
                .collect();
            let a = Audio {
                sample_rate: SAMPLE_RATE,
                left: ch[0].clone(),
                right: ch[1].clone(),
            };
            let mut a = a.resampled(rate);
            a.left.resize(mix.len(), 0.0);
            a.right.resize(mix.len(), 0.0);
            out.push(a);
        }
        let mut it = out.into_iter();
        Ok(Stems {
            drums: it.next().unwrap(),
            bass: it.next().unwrap(),
            vocals: it.next().unwrap(),
            other: it.next().unwrap(),
        })
    }
}

/// Mono drums for a beat tracker: interleaved samples in, mono drums out at the same rate.
pub fn drums_mono(
    separator: &Separator,
    samples: &[f32],
    channels: usize,
    sample_rate: u32,
) -> Result<Vec<f32>> {
    let mix = Audio::from_interleaved(samples, channels, sample_rate);
    Ok(separator.separate(&mix)?.drums.mono())
}

/// `[stem][channel][bin * frames + frame]`
type Masks = Vec<[Vec<f32>; 2]>;

fn median(buf: &mut [f32]) -> f32 {
    let mid = buf.len() / 2;
    *buf.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1
}

/// HPSS (Fitzgerald 2010): median across time keeps sustained partials,
/// median across frequency keeps broadband hits. Percussive -> drums; harmonic
/// below 200 Hz -> bass; the rest -> other. Vocals get nothing: HPSS cannot tell them apart.
fn hpss_masks(specs: &[Spec; 2]) -> Masks {
    const K: usize = 17;
    let cut = (200.0 * stft::N_FFT as f32 / SAMPLE_RATE as f32) as usize;
    let mut masks: Masks = (0..STEMS).map(|_| [Vec::new(), Vec::new()]).collect();
    for (c, s) in specs.iter().enumerate() {
        let t_n = s.frames;
        let mag: Vec<f32> = s.data.iter().map(|z| z.norm()).collect();
        let mut buf = Vec::with_capacity(K);
        let mut d = vec![0.0; mag.len()];
        let mut b = vec![0.0; mag.len()];
        let mut o = vec![0.0; mag.len()];
        let v = vec![0.0; mag.len()];
        for f in 0..BINS {
            for t in 0..t_n {
                buf.clear();
                buf.extend(
                    (t.saturating_sub(K / 2)..(t + K / 2 + 1).min(t_n)).map(|u| mag[f * t_n + u]),
                );
                let h = median(&mut buf);
                buf.clear();
                buf.extend(
                    (f.saturating_sub(K / 2)..(f + K / 2 + 1).min(BINS)).map(|g| mag[g * t_n + t]),
                );
                let p = median(&mut buf);
                let (h2, p2) = (h * h, p * p);
                let mp = if h2 + p2 > 1e-12 { p2 / (h2 + p2) } else { 0.5 };
                let i = f * t_n + t;
                d[i] = mp;
                if f < cut {
                    b[i] = 1.0 - mp
                } else {
                    o[i] = 1.0 - mp
                }
            }
        }
        masks[0][c] = d;
        masks[1][c] = b;
        masks[2][c] = v;
        masks[3][c] = o;
    }
    masks
}

/// Run the U-Net over chunks of frames with context on both sides.
fn network_masks(net: &UNet, specs: &[Spec; 2]) -> Result<Masks> {
    const CORE: usize = 256;
    const CTX: usize = 16;
    const WIN: usize = CORE + 2 * CTX;
    let t_n = specs[0].frames;
    let mut masks: Masks = (0..STEMS)
        .map(|_| [vec![0.0; BINS * t_n], vec![0.0; BINS * t_n]])
        .collect();
    let mut start = 0;
    while start < t_n {
        let mut input = vec![0.0f32; 2 * MODEL_BINS * WIN];
        for c in 0..2 {
            for f in 0..MODEL_BINS {
                for w in 0..WIN {
                    let t = start as isize + w as isize - CTX as isize;
                    if t >= 0 && (t as usize) < t_n {
                        input[(c * MODEL_BINS + f) * WIN + w] =
                            specs[c].at(f, t as usize).norm().ln_1p();
                    }
                }
            }
        }
        let x = Tensor::from_vec(input, (1, 2, MODEL_BINS, WIN), &net.device)?;
        let y: Vec<f32> = net.forward(&x)?.flatten_all()?.to_vec1()?;
        for s in 0..STEMS {
            for c in 0..2 {
                for f in 0..BINS {
                    let fm = f.min(MODEL_BINS - 1); // Nyquist bin borrows the bin below
                    for w in CTX..CTX + CORE {
                        let t = start + w - CTX;
                        if t < t_n {
                            masks[s][c][f * t_n + t] = y[((s * 2 + c) * MODEL_BINS + fm) * WIN + w];
                        }
                    }
                }
            }
        }
        start += CORE;
    }
    Ok(masks)
}

/// Signal-to-distortion ratio in dB over both channels: 10 log10(|s|^2 / |s - ŝ|^2).
/// Plain SDR, not BSS-Eval's (no allowed distortion filter), so it reads lower than museval.
pub fn sdr(reference: &Audio, estimate: &Audio) -> f64 {
    let mut num = 1e-9f64;
    let mut den = 1e-9f64;
    for c in 0..2 {
        for (s, e) in reference.channel(c).iter().zip(estimate.channel(c)) {
            num += (*s as f64).powi(2);
            den += (*s as f64 - *e as f64).powi(2);
        }
    }
    10.0 * (num / den).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdr_of_perfect_estimate_is_high_and_of_silence_is_zero() {
        let a = Audio {
            sample_rate: 44_100,
            left: vec![0.5; 100],
            right: vec![-0.5; 100],
        };
        assert!(sdr(&a, &a) > 80.0);
        assert!(sdr(&a, &Audio::silent(100, 44_100)).abs() < 1e-6);
    }

    #[test]
    fn hpss_stems_add_back_to_the_mix_and_find_the_drums() {
        let t = synth::track(7, 4.0);
        let stems = Separator::Hpss.separate(&t.mix).unwrap();
        let mut sum = Audio::silent(t.mix.len(), t.mix.sample_rate);
        for i in 0..STEMS {
            for c in 0..2 {
                let dst = if c == 0 {
                    &mut sum.left
                } else {
                    &mut sum.right
                };
                for (d, s) in dst.iter_mut().zip(stems.get(i).channel(c)) {
                    *d += s;
                }
            }
        }
        assert!(sdr(&t.mix, &sum) > 60.0, "stems do not sum to the mix");
        // Drums estimate must beat "the whole mix is the drums".
        assert!(sdr(&t.stems.drums, &stems.drums) > sdr(&t.stems.drums, &t.mix));
    }

    #[test]
    fn network_path_runs_over_several_chunks_and_sums_to_the_mix() {
        use candle_core::{DType, Device};
        let vm = candle_nn::VarMap::new();
        let net = UNet::new(
            candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu),
            Device::Cpu,
        )
        .unwrap();
        let t = synth::track(9, 7.0); // 603 frames: three chunks, the last one partial
        let s = Separator::Network(net).separate(&t.mix).unwrap();
        let i = 100_000;
        let sum: f32 = (0..STEMS).map(|k| s.get(k).left[i]).sum();
        assert!((sum - t.mix.left[i]).abs() < 1e-3);
    }

    #[test]
    fn drums_mono_keeps_length_and_rate() {
        let t = synth::track(3, 2.0);
        let inter: Vec<f32> = t.mix.left.iter().map(|x| x * 0.5).collect();
        let rate = 22_050;
        let d = drums_mono(&Separator::Hpss, &inter, 1, rate).unwrap();
        assert_eq!(d.len(), inter.len());
    }
}
