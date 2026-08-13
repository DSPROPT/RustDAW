//! Runs tempo detection over every processed song in the local song-import
//! store and reports what it found, alongside the median interval between
//! kick-drum hits as an independent cross-check.
//!
//! `ratio` is that kick interval expressed in beats. Values near a simple
//! fraction — 1.00, 2.00, 0.50 — mean the grid agrees with the drummer. A
//! ratio like 1.23 means it does not, and is worth listening to.
//!
//! Reads the song-import store, so it only reports on songs already processed
//! on this machine; it prints nothing on a fresh install.
//!
//! ```bash
//! cargo run --release -p daw-analysis --example tempo-benchmark
//! ```
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use daw_analysis::{beats, onset};
use std::path::{Path, PathBuf};

fn read(path: &Path) -> Option<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path).ok()?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels).max(1);
    let scalar: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(Result::ok).collect(),
        hound::SampleFormat::Int => {
            let scale = 2.0_f32.powi(i32::from(spec.bits_per_sample) - 1);
            reader
                .samples::<i32>()
                .filter_map(Result::ok)
                .map(|v| v as f32 / scale)
                .collect()
        }
    };
    let mono = scalar
        .chunks(channels)
        .map(|f| f.iter().sum::<f32>() / channels as f32)
        .collect();
    Some((mono, spec.sample_rate))
}

/// Median seconds between kick hits, from the isolated kick stem. Only the
/// intervals near the median are kept, so bars where the kick rests do not
/// stretch the estimate.
fn kick_interval(path: &Path) -> Option<f64> {
    let (audio, rate) = read(path)?;
    let envelope = onset::onset_envelope(&audio, rate);
    let peak = envelope.values.iter().copied().fold(0.0_f32, f32::max);
    let floor = peak * 0.25;
    let mut hits: Vec<f64> = Vec::new();
    let mut last = -1.0_f64;
    for (frame, value) in envelope.values.iter().enumerate() {
        if *value > floor {
            let at = envelope.seconds_at(frame);
            if at - last > 0.12 {
                hits.push(at);
                last = at;
            }
        }
    }
    if hits.len() < 8 {
        return None;
    }
    let mut gaps: Vec<f64> = hits.windows(2).map(|p| p[1] - p[0]).collect();
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(gaps[gaps.len() / 2])
}

fn main() {
    let root = dirs_home().join(".local/share/chords-extraction/projects");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&root)
        .map(|dir| dir.filter_map(Result::ok).map(|e| e.path()).collect())
        .unwrap_or_default();
    entries.sort();

    println!(
        "{:<34} {:>8} {:>10} {:>8}",
        "song", "tempo", "kick gap", "ratio"
    );
    let mut count = 0;
    for project in entries {
        let drums = project.join("stems/drums.wav");
        if !drums.exists() {
            continue;
        }
        let title = std::fs::read_to_string(project.join("project.json"))
            .ok()
            .and_then(|text| {
                let start = text.find("\"title\":")? + 8;
                let rest = &text[start..];
                let open = rest.find('"')? + 1;
                let close = rest[open..].find('"')?;
                Some(rest[open..open + close].to_owned())
            })
            .unwrap_or_else(|| "untitled".to_owned());

        let Some((audio, rate)) = read(&drums) else {
            continue;
        };
        let envelope = onset::onset_envelope(&audio, rate);
        let bpm = beats::estimate_tempo(&envelope);

        let gap = kick_interval(&project.join("drumkit/kick.wav"));
        let (gap_text, ratio_text) = match gap {
            Some(seconds) => {
                let beat = 60.0 / bpm;
                (format!("{seconds:.3}s"), format!("{:.2}", seconds / beat))
            }
            None => ("-".to_owned(), "-".to_owned()),
        };
        println!("{title:<34.34} {bpm:>8.1} {gap_text:>10} {ratio_text:>8}");
        count += 1;
    }
    println!("\n{count} songs");
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME").map(PathBuf::from).unwrap_or_default()
}
