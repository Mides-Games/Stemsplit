//! STFT / iSTFT matching `torch.stft(n_fft=2048, hop_length=512,
//! window=torch.hann_window(2048), center=True, pad_mode="reflect")`, so the
//! training script and this crate see the same spectrogram.

use rustfft::{num_complex::Complex32, FftPlanner};

pub const N_FFT: usize = 2048;
pub const HOP: usize = 512;
pub const BINS: usize = N_FFT / 2 + 1;

/// One channel's complex spectrogram, laid out `[bin * frames + frame]`.
#[derive(Clone, Debug)]
pub struct Spec {
    pub frames: usize,
    pub data: Vec<Complex32>,
}

impl Spec {
    pub fn at(&self, bin: usize, frame: usize) -> Complex32 {
        self.data[bin * self.frames + frame]
    }
}

fn window() -> Vec<f32> {
    // Periodic Hann, as torch.hann_window's default.
    (0..N_FFT)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / N_FFT as f32).cos())
        .collect()
}

fn reflect(x: &[f32], i: isize) -> f32 {
    let n = x.len() as isize;
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return x[0];
    }
    let period = 2 * (n - 1);
    let mut j = i.rem_euclid(period);
    if j >= n {
        j = period - j;
    }
    x[j as usize]
}

pub fn frames_for(len: usize) -> usize {
    1 + len / HOP
}

pub fn stft(x: &[f32]) -> Spec {
    let frames = frames_for(x.len());
    let win = window();
    let fft = FftPlanner::<f32>::new().plan_fft_forward(N_FFT);
    let mut buf = vec![Complex32::new(0.0, 0.0); N_FFT];
    let mut data = vec![Complex32::new(0.0, 0.0); BINS * frames];
    let half = (N_FFT / 2) as isize;
    for t in 0..frames {
        let start = (t * HOP) as isize - half;
        for (n, b) in buf.iter_mut().enumerate() {
            *b = Complex32::new(reflect(x, start + n as isize) * win[n], 0.0);
        }
        fft.process(&mut buf);
        for f in 0..BINS {
            data[f * frames + t] = buf[f];
        }
    }
    Spec { frames, data }
}

pub fn istft(spec: &Spec, len: usize) -> Vec<f32> {
    let win = window();
    let ifft = FftPlanner::<f32>::new().plan_fft_inverse(N_FFT);
    let total = N_FFT + HOP * (spec.frames - 1);
    let mut out = vec![0.0f32; total];
    let mut norm = vec![0.0f32; total];
    let mut buf = vec![Complex32::new(0.0, 0.0); N_FFT];
    for t in 0..spec.frames {
        for f in 0..BINS {
            buf[f] = spec.at(f, t);
        }
        for f in 1..N_FFT / 2 {
            buf[N_FFT - f] = buf[f].conj();
        }
        ifft.process(&mut buf);
        let o = t * HOP;
        for n in 0..N_FFT {
            out[o + n] += buf[n].re / N_FFT as f32 * win[n];
            norm[o + n] += win[n] * win[n];
        }
    }
    let half = N_FFT / 2;
    (0..len)
        .map(|i| {
            let j = i + half;
            if j < total && norm[j] > 1e-8 {
                out[j] / norm[j]
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_is_exact() {
        let x: Vec<f32> = (0..20_000)
            .map(|i| {
                ((i as f32) * 0.013).sin() * 0.5 + ((i * 7919 % 101) as f32 / 101.0 - 0.5) * 0.1
            })
            .collect();
        let y = istft(&stft(&x), x.len());
        let err = x
            .iter()
            .zip(&y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(err < 1e-4, "max error {err}");
    }

    #[test]
    fn frame_count_matches_torch() {
        assert_eq!(stft(&vec![0.0; 44_100]).frames, 87);
    }
}
