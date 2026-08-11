//! Prints what `RustDAW` reads from a Standard MIDI File.
//!
//! ```text
//! cargo run -p daw-midi --example dump-midi -- song.mid
//! ```

use anyhow::{Context, Result};

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("usage: dump-midi <file.mid>")?;
    let bytes = std::fs::read(&path).with_context(|| format!("failed to read {path}"))?;
    let file = daw_midi::smf::parse(&bytes)?;

    println!(
        "{}/{} time, {} tempo point(s), first {:.2} BPM",
        file.beats_per_bar,
        file.beat_unit,
        file.tempo_map.points().len(),
        file.tempo_map.bpm_at_tick(0)
    );
    for point in file.tempo_map.points().iter().take(8) {
        println!("  tempo @ tick {:>8}: {:6.2} BPM", point.tick, point.bpm);
    }

    for track in &file.tracks {
        let range = match (
            track.notes.iter().map(|note| note.pitch).min(),
            track.notes.iter().map(|note| note.pitch).max(),
        ) {
            (Some(low), Some(high)) => {
                format!("{} – {}", daw_midi::pitch_name(low), daw_midi::pitch_name(high))
            }
            _ => "—".to_owned(),
        };
        println!(
            "  {:<12} {:>5} notes  {:<14} {}{}",
            if track.name.is_empty() { "(unnamed)" } else { &track.name },
            track.notes.len(),
            range,
            track
                .channel
                .map_or_else(|| "no channel".to_owned(), |channel| format!("ch{}", channel + 1)),
            if track.is_drums() { "  [drums]" } else { "" }
        );
    }

    let end = file.tracks.iter().map(daw_midi::SmfTrack::end_tick).max().unwrap_or(0);
    println!("length: {:.1} s", file.tempo_map.tick_to_seconds(end));
    Ok(())
}
