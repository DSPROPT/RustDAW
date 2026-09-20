//! Drives ACE-Step from the command line, without starting the desktop app.
//!
//! ```text
//! cargo run -p daw-songimport --example ai-track -- generate "<description>" [seconds]
//! cargo run -p daw-songimport --example ai-track -- rerender <track.wav> "<sound>" [strength]
//! cargo run -p daw-songimport --example ai-track -- extract <mix.wav> <instrument> ["about the song"]
//! cargo run -p daw-songimport --example ai-track -- add <mix.wav> <instrument> ["about the part"]
//! ```
//!
//! `generate` writes a song and imports it as a session, exactly as the
//! IMPORT SONG window does. The other three take one WAV and leave the render
//! under `<wav's folder>/Audio/AI/`, which is what the AI TRACK window does to
//! a loaded session. `<instrument>` is ACE-Step's own vocabulary: `vocals`,
//! `backing_vocals`, `drums`, `percussion`, `bass`, `guitar`, `keyboard`, `synth`, `strings`,
//! `brass`, `woodwinds`, `fx`.

use anyhow::{Context, Result, bail};
use daw_songimport::{
    CancelFlag, GenerateRequest, ImportProgress, ImportSource, IngestOptions, RenderInput,
    RenderMode, RenderRequest, RenderTask,
};
use std::path::PathBuf;

fn main() -> Result<()> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = arguments.first() else {
        bail!("usage: ai-track generate|rerender|extract|add … (see the source for the forms)");
    };
    match command.as_str() {
        "generate" => generate(&arguments[1..]),
        "rerender" | "extract" | "add" => render(command, &arguments[1..]),
        other => bail!("unknown command {other}"),
    }
}

fn generate(arguments: &[String]) -> Result<()> {
    let prompt = arguments
        .first()
        .context("describe the song after `generate`")?
        .clone();
    let duration = arguments
        .get(1)
        .map_or(Ok(60.0), |value| value.parse::<f64>())
        .context("the length is a number of seconds")?;
    let request = GenerateRequest {
        prompt,
        lyrics: std::env::var("LYRICS").unwrap_or_default(),
        duration,
        bpm: std::env::var("BPM")
            .ok()
            .and_then(|value| value.parse().ok()),
        key: std::env::var("KEY").ok().filter(|key| !key.is_empty()),
        time_signature: None,
        language: None,
        title: None,
    };
    let options = IngestOptions {
        destination_root: daw_songimport::default_song_root(),
        ..IngestOptions::default()
    };
    let cancel = CancelFlag::new();
    let imported = daw_songimport::run_import(
        &ImportSource::Generate(request),
        &options,
        48_000,
        &cancel,
        report(),
    )?;
    println!("session: {}", imported.session_path.display());
    for note in imported.notes {
        println!("  note: {note}");
    }
    Ok(())
}

fn render(command: &str, arguments: &[String]) -> Result<()> {
    let source = PathBuf::from(arguments.first().context("give the WAV to work on")?);
    if !source.is_file() {
        bail!("{} is not a file", source.display());
    }
    let mode = match command {
        "rerender" => RenderMode::Rerender,
        "extract" => RenderMode::Extract,
        _ => RenderMode::Add,
    };
    let (track, prompt, strength) = if mode == RenderMode::Rerender {
        (
            String::new(),
            arguments.get(1).cloned().unwrap_or_default(),
            arguments
                .get(2)
                .map_or(Ok(0.9), |value| value.parse::<f32>())
                .context("the strength is a number from 0 to 1")?,
        )
    } else {
        (
            arguments
                .get(1)
                .context("name the instrument after the file")?
                .clone(),
            arguments.get(2).cloned().unwrap_or_default(),
            1.0,
        )
    };
    let frames = u64::from(
        hound::WavReader::open(&source)
            .with_context(|| format!("failed to open {}", source.display()))?
            .duration(),
    );
    let rate = hound::WavReader::open(&source)?.spec().sample_rate;
    let session_dir = source
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_default();
    let input = if mode.takes_mix() {
        RenderInput::Mix(vec![daw_songimport::MixClip {
            path: source.clone(),
            start_frame: 0,
            source_start_frame: 0,
            frames,
        }])
    } else {
        RenderInput::File(source.clone())
    };
    let task = RenderTask {
        session_dir,
        track_name: track.clone(),
        request: RenderRequest {
            mode,
            track,
            prompt,
            strength,
        },
        input,
        target_rate: rate,
        transpose_semitones: 0,
        // A re-render stands in for the file it was given; an extraction for
        // the same-named stem beside the mix, when there is one.
        level_reference: match mode {
            RenderMode::Rerender => Some(source.clone()),
            RenderMode::Extract => {
                let beside = source.with_file_name(format!("{}.wav", task_track_name(arguments)));
                beside.is_file().then_some(beside)
            }
            RenderMode::Add => None,
        },
    };
    let outcome = daw_songimport::render_track(&task, &CancelFlag::new(), report())?;
    println!(
        "render: {} ({} frames)",
        outcome.path.display(),
        outcome.frames
    );
    for note in outcome.notes {
        println!("  note: {note}");
    }
    Ok(())
}

/// The instrument argument, for finding a stem of that name beside the mix.
fn task_track_name(arguments: &[String]) -> String {
    arguments.get(1).cloned().unwrap_or_default()
}

/// Prints each distinct progress line once.
fn report() -> impl FnMut(ImportProgress) {
    let mut last = String::new();
    move |progress| {
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
    }
}
