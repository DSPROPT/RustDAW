#![allow(clippy::cast_precision_loss)]

//! Detects the chord chart of a separated song.
//!
//! ```text
//! cargo run --release -p daw-analysis --example detect-chords -- <project-dir>
//! ```
//!
//! The project directory is a DSPRO Studio project: tempo comes from the drum
//! stem, harmony from the sum of bass, guitar, piano and other. Drums and
//! vocals are deliberately excluded — percussion has no pitch and a singer's
//! passing notes are not the chord.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    let directory = std::env::args()
        .nth(1)
        .context("usage: detect-chords <project-dir>")?;
    let directory = PathBuf::from(directory);

    let harmonic = sum_stems(
        &directory,
        &["bass.wav", "guitar.wav", "piano.wav", "other.wav"],
    )?;
    let (samples, rate) = harmonic.context("no harmonic stems in that project")?;

    // Beats from the drums when there are any, otherwise from the harmony.
    let drums = directory.join("stems/drums.wav");
    let beat_source = if drums.is_file() {
        daw_analysis::read_wav_mono(&drums)?.0
    } else {
        samples.clone()
    };
    let beats = daw_analysis::analyse_samples(&beat_source, rate, 3.0);
    println!(
        "{:.2} BPM, {} beats, confidence {:.2}",
        beats.bpm(),
        beats.beats.beat_times.len(),
        beats.beats.confidence
    );

    let chromagram = daw_analysis::chroma::chromagram(&samples, rate);
    let (spans, key) = daw_analysis::chords::detect_chords(
        &chromagram,
        &beats.beats.beat_times,
        4,
        beats.beats.downbeat_index,
    );

    println!(
        "key: {}",
        key.map_or_else(|| "unknown".to_owned(), |key| key.name())
    );
    // The chart as a musician would read it: one cell per beat, the chord
    // printed only where it changes.
    let events: Vec<daw_project::ChordEvent> = spans
        .iter()
        .map(|span| daw_project::ChordEvent {
            start_seconds: span.start_seconds,
            end_seconds: span.end_seconds,
            label: span.label(),
            confidence: span.confidence,
        })
        .collect();
    let end = events.last().map_or(0.0, |event| event.end_seconds);
    let chart = daw_project::chord_chart(&events, &beats.tempo_map, 4, end);

    let printed = chart.iter().filter(|beat| beat.is_change()).count();
    println!(
        "\n{} raw span(s) -> {printed} change(s) over {} beats\n",
        spans.len(),
        chart.len()
    );
    println!("{}", daw_project::format_chart(&chart, 4));
    Ok(())
}

/// Mixes several stems into one mono signal.
fn sum_stems(directory: &Path, names: &[&str]) -> Result<Option<(Vec<f32>, u32)>> {
    let mut mixed: Vec<f32> = Vec::new();
    let mut rate = 0;
    for name in names {
        let path = directory.join("stems").join(name);
        if !path.is_file() {
            continue;
        }
        let (samples, sample_rate) = daw_analysis::read_wav_mono(&path)?;
        rate = sample_rate;
        if mixed.len() < samples.len() {
            mixed.resize(samples.len(), 0.0);
        }
        for (slot, value) in mixed.iter_mut().zip(samples) {
            *slot += value;
        }
    }
    Ok((!mixed.is_empty()).then_some((mixed, rate)))
}
