//! Self-generated training and evaluation data: procedural drums, bass,
//! "vocals" (formant-shaped harmonic voice) and other (pads and plucked
//! strings), mixed with exact ground truth. No third-party audio, so no
//! licence question (docs/timeline.md, "Dataset"). Deterministic per seed.

use crate::audio::{Audio, SAMPLE_RATE};
use crate::Stems;
use std::f32::consts::TAU;

pub struct Track {
    pub mix: Audio,
    pub stems: Stems,
    pub bpm: f32,
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.f()
    }
    fn chance(&mut self, p: f32) -> bool {
        self.f() < p
    }
    fn noise(&mut self) -> f32 {
        self.f() * 2.0 - 1.0
    }
}

const SR: f32 = SAMPLE_RATE as f32;

fn add(dst: &mut Audio, at: usize, x: &[f32], pan: f32) {
    let (gl, gr) = (((1.0 - pan) * 0.5).sqrt(), ((1.0 + pan) * 0.5).sqrt());
    for (i, v) in x.iter().enumerate() {
        if at + i >= dst.len() {
            break;
        }
        dst.left[at + i] += v * gl;
        dst.right[at + i] += v * gr;
    }
}

fn midi(n: f32) -> f32 {
    440.0 * 2f32.powf((n - 69.0) / 12.0)
}

fn kick(r: &mut Rng) -> Vec<f32> {
    let (top, decay) = (r.range(90.0, 150.0), r.range(6.0, 12.0));
    let mut ph = 0.0;
    (0..(0.35 * SR) as usize)
        .map(|i| {
            let t = i as f32 / SR;
            ph += TAU * (45.0 + top * (-t * 35.0).exp()) / SR;
            ph.sin() * (-t * decay).exp()
        })
        .collect()
}

fn snare(r: &mut Rng) -> Vec<f32> {
    let tone = r.range(160.0, 240.0);
    (0..(0.25 * SR) as usize)
        .map(|i| {
            let t = i as f32 / SR;
            0.6 * r.noise() * (-t * 22.0).exp() + 0.4 * (TAU * tone * t).sin() * (-t * 18.0).exp()
        })
        .collect()
}

fn hihat(r: &mut Rng, open: bool) -> Vec<f32> {
    let decay = if open { 12.0 } else { 70.0 };
    let mut prev = 0.0;
    (0..((if open { 0.3 } else { 0.08 }) * SR) as usize)
        .map(|i| {
            let n = r.noise();
            let hp = n - prev;
            prev = n;
            0.25 * hp * (-(i as f32 / SR) * decay).exp()
        })
        .collect()
}

/// Naive saw through a one-pole low-pass, with a decaying envelope.
fn saw_note(f0: f32, len: usize, cutoff: f32, attack: f32, decay: f32) -> Vec<f32> {
    let a = (-TAU * cutoff / SR).exp();
    let (mut ph, mut y) = (0.0f32, 0.0f32);
    (0..len)
        .map(|i| {
            let t = i as f32 / SR;
            ph = (ph + f0 / SR).fract();
            y = (1.0 - a) * (2.0 * ph - 1.0) + a * y;
            let env = (t / attack).min(1.0) * (-t * decay).exp();
            let release = ((len - i) as f32 / (0.01 * SR)).min(1.0);
            y * env * release
        })
        .collect()
}

/// Karplus-Strong plucked string.
fn pluck(r: &mut Rng, f0: f32, len: usize) -> Vec<f32> {
    let n = (SR / f0).max(2.0) as usize;
    let mut line: Vec<f32> = (0..n).map(|_| r.noise()).collect();
    (0..len)
        .map(|i| {
            let j = i % n;
            let v = line[j];
            line[j] = 0.497 * (v + line[(j + 1) % n]);
            v
        })
        .collect()
}

const VOWELS: [[f32; 3]; 5] = [
    [800.0, 1150.0, 2900.0], // a
    [400.0, 1700.0, 2600.0], // e
    [350.0, 2000.0, 2800.0], // i
    [450.0, 800.0, 2830.0],  // o
    [325.0, 700.0, 2530.0],  // u
];

/// Harmonic voice shaped by three formants, with vibrato and a breath of noise.
fn voice(r: &mut Rng, f0: f32, len: usize) -> Vec<f32> {
    let v = VOWELS[(r.next() % 5) as usize];
    let gain = |f: f32| -> f32 {
        v.iter()
            .enumerate()
            .map(|(k, fc)| (1.0 / (1 + k) as f32) / (1.0 + ((f - fc) / 90.0).powi(2)))
            .sum()
    };
    let harmonics = ((5000.0 / f0) as usize).clamp(1, 40);
    let amps: Vec<f32> = (1..=harmonics)
        .map(|k| gain(k as f32 * f0) / (k as f32).sqrt())
        .collect();
    let (rate, depth) = (r.range(4.5, 6.5), r.range(0.005, 0.02));
    let mut ph = 0.0f32;
    (0..len)
        .map(|i| {
            let t = i as f32 / SR;
            ph = (ph + f0 * (1.0 + depth * (TAU * rate * t).sin()) / SR).fract();
            let s: f32 = amps
                .iter()
                .enumerate()
                .map(|(k, a)| a * (TAU * (k + 1) as f32 * ph).sin())
                .sum();
            let env = (t / 0.04).min(1.0) * ((len - i) as f32 / (0.05 * SR)).min(1.0);
            (s + 0.02 * r.noise()) * env
        })
        .collect()
}

fn rms(a: &Audio) -> f32 {
    let s: f32 = a.left.iter().chain(&a.right).map(|x| x * x).sum();
    (s / (2 * a.len().max(1)) as f32).sqrt()
}

fn scale(a: &mut Audio, g: f32) {
    a.left
        .iter_mut()
        .chain(a.right.iter_mut())
        .for_each(|x| *x *= g);
}

/// One track of `seconds` length. Stem levels vary per seed.
pub fn track(seed: u64, seconds: f32) -> Track {
    let mut r = Rng::new(seed);
    let len = (seconds * SR) as usize;
    let bpm = r.range(80.0, 180.0);
    let step = 60.0 / bpm / 4.0; // sixteenth
    let steps = (seconds / step) as usize;
    let key = r.range(28.0, 40.0).round(); // E1..E2
    let scale_deg: [f32; 7] = [0.0, 2.0, 3.0, 5.0, 7.0, 8.0, 10.0];
    let mut d = Audio::silent(len, SAMPLE_RATE);
    let mut b = d.clone();
    let mut v = d.clone();
    let mut o = d.clone();

    let roots: Vec<f32> = (0..steps / 16 + 1)
        .map(|_| scale_deg[(r.next() % 7) as usize])
        .collect();
    let mut s = 0;
    while s < steps {
        let at = (s as f32 * step * SR) as usize;
        let pos = s % 16;
        let root = key + roots[s / 16];
        if pos == 0 || pos == 8 || (pos == 10 && r.chance(0.4)) || (pos == 3 && r.chance(0.2)) {
            add(&mut d, at, &kick(&mut r), 0.0);
        }
        if pos == 4 || pos == 12 || (pos == 15 && r.chance(0.15)) {
            add(&mut d, at, &snare(&mut r), r.range(-0.1, 0.1));
        }
        if pos % 2 == 0 || r.chance(0.25) {
            let open = pos == 14 && r.chance(0.3);
            add(&mut d, at, &hihat(&mut r, open), 0.35);
        }
        if pos % 2 == 0 && r.chance(0.85) {
            let n = saw_note(
                midi(root + 12.0 * (r.chance(0.2) as u8 as f32)),
                (step * 2.0 * 0.9 * SR) as usize,
                r.range(300.0, 900.0),
                0.004,
                3.0,
            );
            add(&mut b, at, &n, 0.0);
        }
        if pos == 0 {
            for (k, iv) in [0.0, 3.0, 7.0].iter().enumerate() {
                let f = midi(root + 36.0 + iv);
                let pad = saw_note(
                    f * (1.0 + 0.003 * k as f32),
                    (step * 16.0 * SR) as usize,
                    1500.0,
                    0.3,
                    0.2,
                );
                add(&mut o, at, &pad, [-0.6, 0.6, 0.0][k]);
            }
        }
        if pos % 2 == 0 && r.chance(0.3) {
            let iv = [0.0, 3.0, 7.0, 12.0][(r.next() % 4) as usize];
            add(
                &mut o,
                at,
                &pluck(&mut r, midi(root + 48.0 + iv), (0.6 * SR) as usize),
                r.range(-0.8, 0.8),
            );
        }
        if pos % 4 == 0 && r.chance(0.55) {
            let beats = 1 + (r.next() % 2) as usize;
            let note = key
                + 36.0
                + 12.0 * (r.chance(0.4) as u8 as f32)
                + scale_deg[(r.next() % 7) as usize];
            add(
                &mut v,
                at,
                &voice(
                    &mut r,
                    midi(note),
                    (beats as f32 * 4.0 * step * 0.95 * SR) as usize,
                ),
                r.range(-0.15, 0.15),
            );
        }
        s += 1;
    }

    // Levels: drums and vocals loudest, as in most mixes, with per-track spread.
    for (a, target) in [
        (&mut d, r.range(0.12, 0.2)),
        (&mut b, r.range(0.06, 0.14)),
        (&mut v, r.range(0.05, 0.14)),
        (&mut o, r.range(0.04, 0.1)),
    ] {
        let x = rms(a);
        if x > 0.0 {
            scale(a, target / x);
        }
    }
    let mut mix = Audio::silent(len, SAMPLE_RATE);
    for a in [&d, &b, &v, &o] {
        for i in 0..len {
            mix.left[i] += a.left[i];
            mix.right[i] += a.right[i];
        }
    }
    let peak = mix
        .left
        .iter()
        .chain(&mix.right)
        .fold(0.0f32, |m, x| m.max(x.abs()));
    if peak > 0.95 {
        let g = 0.95 / peak;
        for a in [&mut mix, &mut d, &mut b, &mut v, &mut o] {
            scale(a, g);
        }
    }
    Track {
        mix,
        stems: Stems {
            drums: d,
            bass: b,
            vocals: v,
            other: o,
        },
        bpm,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_stems_sum_to_mix() {
        let a = track(11, 3.0);
        let b = track(11, 3.0);
        assert_eq!(a.mix, b.mix);
        let i = 40_000;
        let sum: f32 = (0..4).map(|k| a.stems.get(k).left[i]).sum();
        assert!((sum - a.mix.left[i]).abs() < 1e-5);
        assert!((0..4).all(|k| rms(a.stems.get(k)) > 0.0));
    }
}
