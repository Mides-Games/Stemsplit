use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use stemsplit::{decode, sdr, synth, write_wav, Separator, STEM_NAMES};

#[derive(Parser)]
#[command(
    name = "stemsplit",
    about = "Split a song into drums, bass, vocals and other"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Separate one file into <out>/{drums,bass,vocals,other}.wav
    Separate {
        input: PathBuf,
        #[arg(long, default_value = "stems")]
        out: PathBuf,
        /// `hpss`, or a .safetensors file from train/. Default: models/stemsplit.safetensors, else hpss.
        #[arg(long)]
        model: Option<String>,
        /// Write only these stems, comma separated (e.g. `drums`).
        #[arg(long)]
        stems: Option<String>,
    },
    /// Write synthetic tracks with ground truth: <out>/<nnnn>/{mixture,drums,bass,vocals,other}.wav
    Synth {
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 100)]
        count: u64,
        #[arg(long, default_value_t = 10.0)]
        seconds: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// SDR per stem over a folder written by `synth` (or any folder of the same shape).
    Eval {
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        model: Option<String>,
    },
}

fn separator(model: Option<&str>) -> Result<Separator> {
    match model {
        Some("hpss") => Ok(Separator::Hpss),
        Some(p) => Separator::load(Path::new(p)),
        None => {
            let default = Path::new("models/stemsplit.safetensors");
            if default.exists() {
                Separator::load(default)
            } else {
                eprintln!(
                    "no models/stemsplit.safetensors: using the HPSS baseline (no vocals stem)"
                );
                Ok(Separator::Hpss)
            }
        }
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    if v.is_empty() {
        f64::NAN
    } else {
        v[v.len() / 2]
    }
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Separate {
            input,
            out,
            model,
            stems,
        } => {
            let sep = separator(model.as_deref())?;
            let mix = decode(&input)?;
            let t = Instant::now();
            let s = sep.separate(&mix)?;
            std::fs::create_dir_all(&out)?;
            let wanted: Vec<String> = stems
                .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
                .unwrap_or_else(|| STEM_NAMES.iter().map(|s| s.to_string()).collect());
            for w in &wanted {
                let Some(i) = STEM_NAMES.iter().position(|n| n == w) else {
                    bail!("unknown stem {w}")
                };
                write_wav(&out.join(format!("{w}.wav")), s.get(i))?;
            }
            eprintln!(
                "{} : {:.1} s of audio in {:.1} s ({})",
                input.display(),
                mix.len() as f64 / mix.sample_rate as f64,
                t.elapsed().as_secs_f64(),
                sep.name()
            );
        }
        Cmd::Synth {
            out,
            count,
            seconds,
            seed,
        } => {
            for k in 0..count {
                let tr = synth::track(seed + k, seconds);
                let dir = out.join(format!("{:04}", seed + k));
                std::fs::create_dir_all(&dir)?;
                write_wav(&dir.join("mixture.wav"), &tr.mix)?;
                for (i, n) in STEM_NAMES.iter().enumerate() {
                    write_wav(&dir.join(format!("{n}.wav")), tr.stems.get(i))?;
                }
            }
            eprintln!("{count} tracks in {}", out.display());
        }
        Cmd::Eval { data, model } => {
            let sep = separator(model.as_deref())?;
            let mut dirs: Vec<PathBuf> = std::fs::read_dir(&data)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.join("mixture.wav").exists())
                .collect();
            dirs.sort();
            let mut got = vec![Vec::new(); 4];
            let mut base = vec![Vec::new(); 4];
            for d in &dirs {
                let mix = decode(&d.join("mixture.wav"))?;
                let s = sep.separate(&mix)?;
                for (i, n) in STEM_NAMES.iter().enumerate() {
                    let r = decode(&d.join(format!("{n}.wav")))?;
                    got[i].push(sdr(&r, s.get(i)));
                    base[i].push(sdr(&r, &mix));
                }
            }
            println!(
                "{} tracks, separator {}; median SDR dB (mixture-as-estimate in brackets)",
                dirs.len(),
                sep.name()
            );
            for (i, n) in STEM_NAMES.iter().enumerate() {
                println!(
                    "{n:>7}: {:6.2}  [{:6.2}]",
                    median(got[i].clone()),
                    median(base[i].clone())
                );
            }
        }
    }
    Ok(())
}
