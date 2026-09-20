//! Re-rendering, extracting or adding one track of a loaded session with
//! ACE-Step.
//!
//! A separated stem is the record with the other instruments filtered out, and
//! the filter leaves marks: bleed, smearing, a guitar that never quite came
//! apart from the keys. ACE-Step offers a different bargain. Given the stem it
//! can play the same part back in a described sound (`cover`); given the mix
//! it can write the named instrument afresh (`extract`), or a part that was
//! never there (`lego`). What comes back is clean and is *not* the record — a
//! performance that resembles it, sometimes closely, sometimes with a note the
//! model preferred. The caller shows that caveat; this module keeps the render
//! honest in the ways it can check: it is exactly as long as what went in, it
//! sits at the session rate, and it is in the key the session is being heard
//! in.
//!
//! Renders land under `Audio/AI/`, and the original files are never touched,
//! so the swap is a clip edit the session can undo.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::client::{RenderMode, RenderRequest, WorkerClient};
use crate::ingest::{convert_exact, convert_pitch, is_percussion, measure_levels, wav_frame_count};
use crate::{CancelFlag, ImportProgress, ensure_acestep, ensure_worker, wait_for_job};

/// One clip's audio as it sits on the timeline, for summing a dry mix.
#[derive(Clone, Debug)]
pub struct MixClip {
    /// The file to read — the unshifted original, so the mix is in the key
    /// the song was recorded in and the render can be shifted like any stem.
    pub path: PathBuf,
    pub start_frame: u64,
    pub source_start_frame: u64,
    /// Frames of the file the clip plays, from `source_start_frame`.
    pub frames: u64,
}

/// What ACE-Step is given to work on.
#[derive(Clone, Debug)]
pub enum RenderInput {
    /// One file, whole. The render replaces it, so every clip reading it
    /// keeps its window.
    File(PathBuf),
    /// A dry sum of these clips at their places on the timeline: the song as
    /// it plays, without effects, gains or the track being written over.
    Mix(Vec<MixClip>),
}

/// One render to run.
#[derive(Clone, Debug)]
pub struct RenderTask {
    /// The session folder holding `Audio/`.
    pub session_dir: PathBuf,
    /// The name of the track the render is for. Decides the file name and,
    /// through [`is_percussion`], whether a key shift applies.
    pub track_name: String,
    pub request: RenderRequest,
    pub input: RenderInput,
    /// The session's sample rate; the render is converted to it.
    pub target_rate: u32,
    /// How far the session sits from its imported key. The render is made
    /// from the originals and moved by the same amount, so it plays in the
    /// key being heard and re-keying later starts from the unshifted file.
    pub transpose_semitones: i32,
    /// The file whose level the render should sit at: the track it replaces.
    /// ACE-Step normalises what it writes to just under full scale, and a stem
    /// handed back 10 dB louder than the one it stands in for is not a swap
    /// but a jump. `None` (a new part) sits a few dB under the mix instead.
    pub level_reference: Option<PathBuf>,
}

/// How far under the mix a new part is placed when nothing says otherwise.
const NEW_PART_UNDER_MIX_DB: f64 = 6.0;
/// Where a render's peak is held to, in dBFS, whatever the match asks for.
const PEAK_CEILING_DB: f64 = -1.0;

/// A finished render, ready to become a clip.
#[derive(Clone, Debug)]
pub struct RenderOutcome {
    /// The file to play: at the session rate, in the session's current key.
    pub path: PathBuf,
    /// The unshifted render, when `path` is a shifted copy of it — what
    /// re-keying reads. `None` when the session is in its original key.
    pub source_path: Option<PathBuf>,
    /// Frames in `path`, which is exactly the length of what went in.
    pub frames: u64,
    /// Where on the timeline the render starts: a file render keeps its
    /// clips' places, a mix render starts where the earliest clip did.
    pub start_frame: u64,
    /// Which model produced it, and anything the user should know.
    pub notes: Vec<String>,
}

/// The instrument ACE-Step should be asked for, given one of `RustDAW`'s
/// track names. Pipeline stems and transcription tracks are named after what
/// they hold; anything unrecognised is offered as a keyboard, the most
/// forgiving of the vocabulary.
#[must_use]
pub fn ace_track_for(track_name: &str) -> &'static str {
    let name = track_name.to_ascii_lowercase();
    let has = |needle: &str| name.contains(needle);
    if has("vocal") || has("voice") || has("sing") {
        if has("back") || has("harmon") || has("choir") {
            "backing_vocals"
        } else {
            "vocals"
        }
    } else if has("drum") || has("kit") || has("kick") || has("snare") || has("cymbal") {
        "drums"
    } else if has("perc") || has("conga") || has("bongo") || has("shaker") || has("tambour") {
        "percussion"
    } else if has("bass") {
        "bass"
    } else if has("guitar") || has("gtr") {
        "guitar"
    } else if has("piano") || has("key") || has("organ") || has("rhodes") || has("epiano") {
        "keyboard"
    } else if has("synth") || has("pad") || has("lead") {
        "synth"
    } else if has("string") || has("violin") || has("cello") || has("viola") {
        "strings"
    } else if has("brass") || has("trumpet") || has("trombone") || has("horn") {
        "brass"
    } else if has("sax") || has("flute") || has("clarinet") || has("wood") {
        "woodwinds"
    } else if has("fx") || has("effect") || has("noise") {
        "fx"
    } else {
        "keyboard"
    }
}

/// Runs one render and returns the file to put on the track.
///
/// Blocking, and slow: a cold ACE-Step loads its models first, and the
/// first run on a fresh install downloads them. Call it from a background
/// thread and show [`ImportProgress`] as it comes.
///
/// # Errors
///
/// Returns an error if the worker or ACE-Step is missing, the job fails or
/// is cancelled, or the result cannot be converted into the session.
pub fn render_track(
    task: &RenderTask,
    cancel: &CancelFlag,
    mut on_progress: impl FnMut(ImportProgress),
) -> Result<RenderOutcome> {
    let client = WorkerClient::new();
    on_progress(ImportProgress::Status(
        "Contacting the song pipeline…".to_owned(),
    ));
    ensure_worker(&client, &mut on_progress)?;
    ensure_acestep(&client)?;

    // What goes in, how long it is, and where it sits.
    let (upload, expected_frames, start_frame, scratch) = match &task.input {
        RenderInput::File(path) => {
            if !path.is_file() {
                bail!("{} is not a file", path.display());
            }
            (path.clone(), wav_frame_count(path)?, 0, None)
        }
        RenderInput::Mix(clips) => {
            on_progress(ImportProgress::Status(
                "Summing the song for ACE-Step…".to_owned(),
            ));
            let mixed = mix_clips(clips, &task.session_dir, task.target_rate)?;
            (
                mixed.path.clone(),
                mixed.frames,
                mixed.start_frame,
                Some(mixed.path),
            )
        }
    };
    if expected_frames == 0 {
        bail!("there is no audio to render");
    }

    let outcome = (|| -> Result<RenderOutcome> {
        on_progress(ImportProgress::Status(
            "Sending the audio to ACE-Step…".to_owned(),
        ));
        let job = client.submit_render(&upload, &task.request)?;
        let job = wait_for_job(
            &client,
            &job.id,
            cancel,
            "render cancelled; ACE-Step finishes the job on its own and the result is discarded",
            &mut on_progress,
        )?;
        let rendered = job
            .result_path
            .context("ACE-Step finished without leaving a file")?;
        if !rendered.is_file() {
            bail!("ACE-Step reported {} but it is missing", rendered.display());
        }

        on_progress(ImportProgress::Status(
            "Converting the render into the session…".to_owned(),
        ));
        settle(task, &rendered, expected_frames, start_frame, &upload)
    })();

    if let Some(scratch) = scratch {
        let _ = std::fs::remove_file(scratch);
    }
    outcome
}

/// Turns the file ACE-Step left into the one the session plays: at the
/// session rate, exactly `expected_frames` long, and in the session's key.
fn settle(
    task: &RenderTask,
    rendered: &Path,
    expected_frames: u64,
    start_frame: u64,
    input: &Path,
) -> Result<RenderOutcome> {
    let ai_dir = task.session_dir.join("Audio").join("AI");
    std::fs::create_dir_all(&ai_dir)
        .with_context(|| format!("failed to create {}", ai_dir.display()))?;
    let unshifted = unique_path(
        &ai_dir,
        &format!(
            "{}-{}",
            sanitize(&task.track_name),
            task.request.mode.wire_name()
        ),
    );
    let (gain_db, level_note) = level_match(task, rendered, input)?;
    let frames = convert_exact(
        rendered,
        &unshifted,
        task.target_rate,
        expected_frames,
        gain_db,
    )?;
    // The worker keeps nothing of a render once it is collected.
    let _ = std::fs::remove_file(rendered);

    let mut notes = vec![format!(
        "{} with ACE-Step ({}).",
        match task.request.mode {
            RenderMode::Rerender => "Re-rendered",
            RenderMode::Extract => "Extracted",
            RenderMode::Add => "Written",
        },
        task.request.mode.wire_name()
    )];
    if let Some(note) = level_note {
        notes.push(note);
    }

    // A kit has no key; everything else follows the session.
    if task.transpose_semitones == 0 || is_percussion(&task.track_name) {
        return Ok(RenderOutcome {
            path: unshifted,
            source_path: None,
            frames,
            start_frame,
            notes,
        });
    }
    let keys = crate::rekey::keys_dir(&task.session_dir, task.transpose_semitones);
    std::fs::create_dir_all(&keys)
        .with_context(|| format!("failed to create {}", keys.display()))?;
    let target = keys.join(
        unshifted
            .file_name()
            .context("the render has no file name")?,
    );
    convert_pitch(&unshifted, &target, task.transpose_semitones)
        .context("failed to move the render into the session's key")?;
    notes.push(format!(
        "Shifted {:+} semitones to match the session.",
        task.transpose_semitones
    ));
    Ok(RenderOutcome {
        path: target,
        source_path: Some(unshifted),
        frames,
        start_frame,
        notes,
    })
}

/// The gain that puts the render at the level of what it stands in for.
///
/// Matched on mean level, because that is what a fader balance is made of;
/// then held under [`PEAK_CEILING_DB`], because a match that clips is worse
/// than one that is a little quiet. A file that cannot be measured leaves the
/// render as ACE-Step wrote it, and the note says so.
fn level_match(task: &RenderTask, rendered: &Path, input: &Path) -> Result<(f64, Option<String>)> {
    let (render_peak, render_mean) = measure_levels(rendered)?;
    let (Some(render_peak), Some(render_mean)) = (render_peak, render_mean) else {
        return Ok((
            0.0,
            Some(
                "Left at the level ACE-Step wrote it; the render could not be measured.".to_owned(),
            ),
        ));
    };
    let (reference, offset, what) = match &task.level_reference {
        Some(reference) => (reference.as_path(), 0.0, "the track it replaces"),
        None => (input, -NEW_PART_UNDER_MIX_DB, "the song, a little under it"),
    };
    let (_, reference_mean) = measure_levels(reference)?;
    let Some(reference_mean) = reference_mean.filter(|mean| mean.is_finite()) else {
        return Ok((
            0.0,
            Some(
                "Left at the level ACE-Step wrote it; the reference could not be measured."
                    .to_owned(),
            ),
        ));
    };
    let wanted = reference_mean + offset - render_mean;
    let ceiling = PEAK_CEILING_DB - render_peak;
    let gain = wanted.min(ceiling);
    let note = if (wanted - gain).abs() > 0.01 {
        format!("Level set {gain:+.1} dB, to match {what} as far as the peak allows.")
    } else {
        format!("Level set {gain:+.1} dB, to match {what}.")
    };
    Ok((gain, Some(note)))
}

/// A dry mixdown written to disk, with where it starts on the timeline.
#[derive(Debug)]
struct Mixed {
    path: PathBuf,
    frames: u64,
    start_frame: u64,
}

/// Sums `clips` at their places on the timeline into one stereo 24-bit WAV.
///
/// No gain, pan or effects: ACE-Step wants the song, not the mix of it. The
/// buffer spans the earliest start to the latest end, and a sum that clips is
/// scaled down whole rather than clipped — a model given a distorted mix
/// renders a distorted part.
fn mix_clips(clips: &[MixClip], session_dir: &Path, target_rate: u32) -> Result<Mixed> {
    let Some(start_frame) = clips.iter().map(|clip| clip.start_frame).min() else {
        bail!("there are no clips to mix");
    };
    let end_frame = clips
        .iter()
        .map(|clip| clip.start_frame.saturating_add(clip.frames))
        .max()
        .unwrap_or(start_frame);
    let total = usize::try_from(end_frame.saturating_sub(start_frame))
        .context("the mix is too long to hold in memory")?;
    if total == 0 {
        bail!("the clips to mix are empty");
    }
    let mut left = vec![0.0_f32; total];
    let mut right = vec![0.0_f32; total];

    for clip in clips {
        let (samples, channels, rate) = read_wav(&clip.path)?;
        if rate != target_rate {
            bail!(
                "{} is at {rate} Hz but the session runs at {target_rate} Hz",
                clip.path.display()
            );
        }
        let offset =
            usize::try_from(clip.start_frame - start_frame).context("clip offset does not fit")?;
        let first = usize::try_from(clip.source_start_frame).context("clip window does not fit")?;
        let wanted = usize::try_from(clip.frames).context("clip length does not fit")?;
        let available = (samples.len() / channels).saturating_sub(first);
        for index in 0..wanted.min(available).min(total.saturating_sub(offset)) {
            let frame = (first + index) * channels;
            let l = samples[frame];
            let r = if channels > 1 { samples[frame + 1] } else { l };
            left[offset + index] += l;
            right[offset + index] += r;
        }
    }

    let peak = left
        .iter()
        .chain(right.iter())
        .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
    let scale = if peak > 1.0 { 1.0 / peak } else { 1.0 };

    let scratch = session_dir.join("Audio").join("AI");
    std::fs::create_dir_all(&scratch)
        .with_context(|| format!("failed to create {}", scratch.display()))?;
    let path = scratch.join(format!("mix-{}.wav", uuid::Uuid::new_v4().simple()));
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: target_rate,
        bits_per_sample: 24,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec)
        .with_context(|| format!("failed to write {}", path.display()))?;
    let full_scale = f32::from(i16::MAX) * 256.0;
    for (l, r) in left.iter().zip(right.iter()) {
        #[allow(clippy::cast_possible_truncation)]
        for sample in [l, r] {
            writer.write_sample(((sample * scale).clamp(-1.0, 1.0) * full_scale) as i32)?;
        }
    }
    writer.finalize()?;
    Ok(Mixed {
        path,
        frames: total as u64,
        start_frame,
    })
}

/// Every sample of a WAV as floats in -1..1, interleaved, with its channel
/// count and rate.
fn read_wav(path: &Path) -> Result<(Vec<f32>, usize, u32)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let spec = reader.spec();
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .with_context(|| format!("failed to read {}", path.display()))?,
        hound::SampleFormat::Int => {
            let scale = 2.0_f32.powi(i32::from(spec.bits_per_sample).saturating_sub(1));
            reader
                .samples::<i32>()
                .map(|sample| {
                    #[allow(clippy::cast_precision_loss)]
                    sample.map(|value| value as f32 / scale)
                })
                .collect::<Result<Vec<_>, _>>()
                .with_context(|| format!("failed to read {}", path.display()))?
        }
    };
    Ok((samples, usize::from(spec.channels).max(1), spec.sample_rate))
}

/// A track name as a file name: letters, digits and dashes.
fn sanitize(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let trimmed = cleaned.trim_matches('-');
    if trimmed.is_empty() {
        "track".to_owned()
    } else {
        trimmed.to_owned()
    }
}

/// `<dir>/<base>.wav`, or `<base>-2.wav` and so on, so a second render of
/// the same track never overwrites one an undo could bring back.
fn unique_path(dir: &Path, base: &str) -> PathBuf {
    let first = dir.join(format!("{base}.wav"));
    if !first.exists() {
        return first;
    }
    (2..10_000)
        .map(|index| dir.join(format!("{base}-{index}.wav")))
        .find(|candidate| !candidate.exists())
        .unwrap_or(first)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_tone(path: &Path, rate: u32, frames: usize, value: f32, channels: u16) {
        let spec = hound::WavSpec {
            channels,
            sample_rate: rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for _ in 0..frames {
            for _ in 0..channels {
                writer.write_sample(value).unwrap();
            }
        }
        writer.finalize().unwrap();
    }

    #[test]
    fn track_names_map_onto_ace_steps_vocabulary() {
        assert_eq!(ace_track_for("Vocals"), "vocals");
        assert_eq!(ace_track_for("Backing Vocals"), "backing_vocals");
        assert_eq!(ace_track_for("Drums"), "drums");
        assert_eq!(ace_track_for("Bass"), "bass");
        assert_eq!(ace_track_for("Electric Guitar (MIDI)"), "guitar");
        assert_eq!(ace_track_for("Piano"), "keyboard");
        assert_eq!(ace_track_for("Other"), "keyboard");
        assert_eq!(ace_track_for("Synth Pad"), "synth");
    }

    #[test]
    fn mix_places_clips_on_the_timeline_and_scales_a_hot_sum() {
        let dir = tempfile_dir("mix");
        let a = dir.join("a.wav");
        let b = dir.join("b.wav");
        write_tone(&a, 48_000, 100, 0.8, 2);
        write_tone(&b, 48_000, 100, 0.8, 1);
        let clips = vec![
            MixClip {
                path: a,
                start_frame: 1000,
                source_start_frame: 0,
                frames: 100,
            },
            MixClip {
                path: b,
                start_frame: 1050,
                source_start_frame: 20,
                frames: 80,
            },
        ];
        let mixed = mix_clips(&clips, &dir, 48_000).unwrap();
        assert_eq!(mixed.start_frame, 1000);
        assert_eq!(mixed.frames, 130);
        let (samples, channels, rate) = read_wav(&mixed.path).unwrap();
        assert_eq!((channels, rate), (2, 48_000));
        assert_eq!(samples.len(), 130 * 2);
        // The overlap summed to 1.6 and was scaled back to full scale; the
        // lone part is half of that, and the tail is b alone.
        assert!(
            (samples[60 * 2] - 1.0).abs() < 0.01,
            "overlap {}",
            samples[60 * 2]
        );
        assert!(
            (samples[10 * 2] - 0.5).abs() < 0.01,
            "a alone {}",
            samples[10 * 2]
        );
        assert!(
            (samples[120 * 2] - 0.5).abs() < 0.01,
            "b alone {}",
            samples[120 * 2]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn mix_refuses_a_file_at_another_rate() {
        let dir = tempfile_dir("rate");
        let a = dir.join("a.wav");
        write_tone(&a, 44_100, 10, 0.1, 2);
        let clips = vec![MixClip {
            path: a,
            start_frame: 0,
            source_start_frame: 0,
            frames: 10,
        }];
        let error = mix_clips(&clips, &dir, 48_000).unwrap_err().to_string();
        assert!(error.contains("44100 Hz"), "{error}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn render_files_never_overwrite_each_other() {
        let dir = tempfile_dir("unique");
        let first = unique_path(&dir, "guitar-extract");
        assert!(first.ends_with("guitar-extract.wav"));
        std::fs::write(&first, b"").unwrap();
        assert!(unique_path(&dir, "guitar-extract").ends_with("guitar-extract-2.wav"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn sanitized_names_are_plain_file_names() {
        assert_eq!(sanitize("Electric Guitar (MIDI)"), "electric-guitar--midi");
        assert_eq!(sanitize("///"), "track");
    }

    fn tempfile_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rustdaw-render-{label}-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
