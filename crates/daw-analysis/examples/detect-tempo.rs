//! Detects tempo and beats in a WAV file.
//!
//! ```text
//! cargo run --release -p daw-analysis --example detect-tempo -- drums.wav
//! ```

use anyhow::{Context, Result};
use std::time::Instant;

fn main() -> Result<()> {
    let mut arguments = std::env::args().skip(1);
    let path = arguments.next().context("usage: detect-tempo <file.wav>")?;

    let started = Instant::now();
    let analysis = daw_analysis::analyse_wav(std::path::Path::new(&path), 3.0)?;
    let elapsed = started.elapsed();

    println!(
        "{:>7.2} BPM   confidence {:.2}   {} beat(s)   {} tempo point(s)   {:.1} s audio in {:.2} s",
        analysis.bpm(),
        analysis.beats.confidence,
        analysis.beats.beat_times.len(),
        analysis.tempo_map.points().len(),
        analysis.duration_seconds,
        elapsed.as_secs_f64()
    );
    if !analysis.beats.is_usable() {
        println!("  (no clear pulse; tempo is a fallback)");
        return Ok(());
    }
    println!(
        "  first downbeat at {:.3} s, first beats: {}",
        analysis.beats.first_downbeat(),
        analysis
            .beats
            .beat_times
            .iter()
            .take(6)
            .map(|time| format!("{time:.3}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    for point in analysis.tempo_map.points().iter().take(6) {
        println!("  tempo @ tick {:>8}: {:6.2} BPM", point.tick, point.bpm);
    }
    Ok(())
}
