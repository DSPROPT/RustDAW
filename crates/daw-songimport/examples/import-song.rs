#![allow(clippy::cast_precision_loss)]

//! Imports a song from the command line, without starting the desktop app.
//!
//! ```text
//! cargo run -p daw-songimport --example import-song              # list ready projects
//! cargo run -p daw-songimport --example import-song -- <id>      # import one
//! cargo run -p daw-songimport --example import-song -- <url>     # run the pipeline first
//! ```

use anyhow::Result;
use daw_songimport::{CancelFlag, ImportProgress, ImportSource, IngestOptions};

fn main() -> Result<()> {
    let argument = std::env::args().nth(1);
    let Some(argument) = argument else {
        return list();
    };

    let source = if argument.starts_with("http://") || argument.starts_with("https://") {
        ImportSource::Url(argument)
    } else {
        ImportSource::ExistingProject(argument)
    };

    let options = IngestOptions {
        destination_root: daw_songimport::default_song_root(),
        ..IngestOptions::default()
    };
    let cancel = CancelFlag::new();
    let mut last = String::new();
    let imported = daw_songimport::run_import(&source, &options, 48_000, &cancel, |progress| {
        let line = match progress {
            ImportProgress::Status(message) => message,
            ImportProgress::Pipeline {
                stage,
                percent,
                message,
            } => format!("{stage} {percent:.0}% {message}"),
            ImportProgress::Converting { fraction, stem } => {
                format!("converting {stem} ({:.0}%)", fraction * 100.0)
            }
        };
        if line != last {
            println!("  {line}");
            last = line;
        }
    })?;

    println!("\nSession: {}", imported.session_path.display());
    println!(
        "{} BPM, {}/4, {} track(s):",
        imported.document.tempo,
        imported.document.meter_numerator,
        imported.document.tracks.len()
    );
    for track in &imported.document.tracks {
        if track.kind.is_instrument() {
            let notes: usize = track.midi_clips.iter().map(|clip| clip.notes.len()).sum();
            let end = imported
                .document
                .tempo_map()
                .tick_to_seconds(track.midi_end_tick());
            println!("  {:8} {notes:>5} notes, ends {end:.1} s  [instrument]", track.name);
            continue;
        }
        let frames: u64 = track
            .clips
            .iter()
            .map(|clip| clip.end_frame - clip.start_frame)
            .sum();
        println!(
            "  {:8} {:.1} s",
            track.name,
            frames as f64 / f64::from(imported.document.sample_rate)
        );
    }
    for note in &imported.notes {
        println!("note: {note}");
    }
    Ok(())
}

fn list() -> Result<()> {
    let projects = daw_songimport::list_ready_projects()?;
    if projects.is_empty() {
        println!("No processed songs yet. Pass a YouTube link to make one.");
        return Ok(());
    }
    println!("{} song(s) ready to import:\n", projects.len());
    for project in projects {
        println!(
            "  {:32} {} ({:.0} s)",
            project.id,
            project.label(),
            project.duration.unwrap_or_default()
        );
    }
    Ok(())
}
