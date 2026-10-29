//! Decode any file Symphonia reads into stereo f32 at 44.1 kHz, and write WAV.

use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

/// The rate the network is trained at; everything is brought to it.
pub const SAMPLE_RATE: u32 = 44_100;

/// Planar stereo audio. Mono input is duplicated into both channels.
#[derive(Clone, Debug, PartialEq)]
pub struct Audio {
    pub sample_rate: u32,
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

impl Audio {
    pub fn silent(len: usize, sample_rate: u32) -> Self {
        Audio {
            sample_rate,
            left: vec![0.0; len],
            right: vec![0.0; len],
        }
    }

    /// Build from interleaved samples with any channel count (extra channels dropped).
    pub fn from_interleaved(samples: &[f32], channels: usize, sample_rate: u32) -> Self {
        let channels = channels.max(1);
        let mut a = Audio::silent(samples.len() / channels, sample_rate);
        for (i, frame) in samples.chunks_exact(channels).enumerate() {
            a.left[i] = frame[0];
            a.right[i] = if channels > 1 { frame[1] } else { frame[0] };
        }
        a
    }

    pub fn len(&self) -> usize {
        self.left.len()
    }

    pub fn is_empty(&self) -> bool {
        self.left.is_empty()
    }

    pub fn channel(&self, c: usize) -> &[f32] {
        if c == 0 {
            &self.left
        } else {
            &self.right
        }
    }

    pub fn mono(&self) -> Vec<f32> {
        self.left
            .iter()
            .zip(&self.right)
            .map(|(l, r)| 0.5 * (l + r))
            .collect()
    }

    /// Linear-interpolation resample. Good enough for analysis input; a
    /// band-limited resampler is a listed follow-up (docs/timeline.md).
    pub fn resampled(&self, rate: u32) -> Audio {
        if rate == self.sample_rate || self.is_empty() {
            return Audio {
                sample_rate: rate,
                ..self.clone()
            };
        }
        let ratio = self.sample_rate as f64 / rate as f64;
        let n = ((self.len() as f64) / ratio).floor() as usize;
        let interp = |x: &[f32]| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let p = i as f64 * ratio;
                    let j = p.floor() as usize;
                    let f = (p - j as f64) as f32;
                    let a = x[j.min(x.len() - 1)];
                    let b = x[(j + 1).min(x.len() - 1)];
                    a + (b - a) * f
                })
                .collect()
        };
        Audio {
            sample_rate: rate,
            left: interp(&self.left),
            right: interp(&self.right),
        }
    }
}

/// Decode a file to stereo at its own rate.
pub fn decode(path: &Path) -> Result<Audio> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let stream = MediaSourceStream::new(Box::new(file), Default::default());
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            stream,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| anyhow!("probe: {e}"))?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| anyhow!("no audio track in {}", path.display()))?;
    let track_id = track.id;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| anyhow!("audio track has no codec parameters"))?
        .clone();
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .map_err(|e| anyhow!("decoder: {e}"))?;

    let mut out = Audio::silent(0, params.sample_rate.unwrap_or(0));
    let mut scratch = Vec::new();
    while let Some(packet) = format.next_packet().map_err(|e| anyhow!("read: {e}"))? {
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(buffer) => {
                let spec = buffer.spec();
                let ch = spec.channels().count().max(1);
                if out.sample_rate == 0 {
                    out.sample_rate = spec.rate();
                }
                scratch.clear();
                buffer.copy_to_vec_interleaved(&mut scratch);
                for frame in scratch.chunks_exact(ch) {
                    out.left.push(frame[0]);
                    out.right.push(if ch > 1 { frame[1] } else { frame[0] });
                }
            }
            // A corrupt packet is skipped rather than losing the file.
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(anyhow!("decode: {e}")),
        }
    }
    if out.is_empty() || out.sample_rate == 0 {
        return Err(anyhow!("decoded no audio from {}", path.display()));
    }
    Ok(out)
}

/// Write 32-bit float stereo WAV (no clipping of stems that exceed 0 dBFS).
pub fn write_wav(path: &Path, audio: &Audio) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: audio.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(path, spec)
        .with_context(|| format!("create {}", path.display()))?;
    for (l, r) in audio.left.iter().zip(&audio.right) {
        w.write_sample(*l)?;
        w.write_sample(*r)?;
    }
    w.finalize()?;
    Ok(())
}
