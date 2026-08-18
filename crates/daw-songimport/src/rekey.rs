//! Moving a session that is already imported into another key.
//!
//! Rehearsing a song a few semitones down is not a reason to separate it again:
//! the stems are already sitting in the session folder, and shifting them is a
//! few seconds of work. This moves the audio, the chord chart and the
//! transcription together and remembers how far the session has been moved, so
//! the next change is worked out from the original recording rather than piled
//! on top of the last shift.
//!
//! Renders are kept under `Audio/Keys/<n>/`, which makes going back to a key
//! that has been heard before immediate: the files are already there.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result};
use daw_project::ProjectDocument;

use crate::ingest::{MAX_TRANSPOSE_SEMITONES, convert_pitch, is_percussion};

/// What a re-key did, for the line the user is shown afterwards.
#[derive(Clone, Debug, Default)]
pub struct Rekeyed {
    /// Where the session now sits, in semitones from its original key.
    pub semitones: i32,
    /// Stems that had to be shifted for this key.
    pub rendered: usize,
    /// Stems this key already had on disk from an earlier visit.
    pub reused: usize,
    /// Anything the user should know, such as notes lost off the keyboard.
    pub notes: Vec<String>,
}

/// Moves `document` into a key `semitones` from the one it was imported in.
///
/// `session_dir` is the folder holding the session's `Audio`. Progress is
/// reported as a 0.0–1.0 fraction and the name of the stem being worked on;
/// it is called from several threads at once, which is why it must be `Sync`.
///
/// # Errors
///
/// Returns an error if a stem cannot be read or ffmpeg cannot convert it. The
/// document is left alone unless every stem succeeded.
#[allow(clippy::too_many_lines)]
pub fn rekey_session(
    document: &mut ProjectDocument,
    session_dir: &Path,
    semitones: i32,
    on_progress: &(impl Fn(f32, &str) + Sync),
) -> Result<Rekeyed> {
    let semitones = semitones.clamp(-MAX_TRANSPOSE_SEMITONES, MAX_TRANSPOSE_SEMITONES);
    let delta = semitones - document.transpose_semitones;
    let mut outcome = Rekeyed {
        semitones,
        ..Rekeyed::default()
    };
    if delta == 0 {
        return Ok(outcome);
    }

    // Work out every render first, so a failure leaves the session as it was
    // rather than half in one key and half in another.
    let mut jobs = Vec::new();
    for track in &document.tracks {
        // A kit has no key: the drums stay as they were played, exactly as they
        // do when a song is imported transposed.
        let percussion = is_percussion(&track.name);
        for clip in &track.clips {
            let original = clip
                .source_path
                .clone()
                .unwrap_or_else(|| clip.path.clone());
            let target = if semitones == 0 || percussion {
                None
            } else {
                let name = original
                    .file_name()
                    .with_context(|| format!("{} has no file name", original.display()))?;
                Some(keys_dir(session_dir, semitones).join(name))
            };
            jobs.push((clip.id, original, target));
        }
    }

    let total = jobs
        .iter()
        .filter(|(_, _, target)| target.is_some())
        .count();
    if let Some(directory) = jobs
        .iter()
        .find_map(|(_, _, target)| target.as_ref().and_then(|path| path.parent()))
    {
        std::fs::create_dir_all(directory)
            .with_context(|| format!("failed to create {}", directory.display()))?;
    }

    let done = AtomicUsize::new(0);
    let reused = AtomicUsize::new(0);
    let report = |name: &str| {
        let finished = done.fetch_add(1, Ordering::Relaxed) + 1;
        #[allow(clippy::cast_precision_loss)]
        on_progress(finished as f32 / total.max(1) as f32, name);
    };

    // ffmpeg is one process per stem and they do not contend, so the whole
    // song is shifted in about as long as its longest stem takes.
    std::thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::new();
        for (_, original, target) in &jobs {
            let Some(target) = target else { continue };
            let report = &report;
            let reused = &reused;
            handles.push(scope.spawn(move || -> Result<()> {
                let name = original
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or("stem");
                // A key that has been heard before is already on disk.
                if target.is_file() {
                    reused.fetch_add(1, Ordering::Relaxed);
                    report(name);
                    return Ok(());
                }
                convert_pitch(original, target, semitones)
                    .with_context(|| format!("failed to move {name} into the new key"))?;
                report(name);
                Ok(())
            }));
        }
        for handle in handles {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("a stem conversion panicked"))??;
        }
        Ok(())
    })?;

    outcome.reused = reused.load(Ordering::Relaxed);
    outcome.rendered = total - outcome.reused;

    // Everything below only rewrites the document, and cannot fail.
    for track in &mut document.tracks {
        for clip in &mut track.clips {
            let Some((_, original, target)) = jobs.iter().find(|(id, _, _)| *id == clip.id) else {
                continue;
            };
            if let Some(path) = target {
                clip.path.clone_from(path);
                clip.source_path = Some(original.clone());
            } else {
                clip.path.clone_from(original);
                clip.source_path = None;
            }
        }
    }

    let dropped = retune_document(document, semitones);
    if dropped > 0 {
        outcome
            .notes
            .push(format!("{dropped} note(s) fell off the keyboard."));
    }
    Ok(outcome)
}

/// Moves everything about a session except its audio files: the transcription,
/// the chord chart and the key it reports. Returns how many notes the shift
/// pushed off the keyboard.
///
/// The distance moved is worked out from where the session already sits, so a
/// song taken to -4 and then to +2 ends up two below the recording rather than
/// two below the shift.
fn retune_document(document: &mut ProjectDocument, semitones: i32) -> usize {
    let delta = semitones - document.transpose_semitones;
    let dropped = transpose_midi(document, delta);
    for chord in &mut document.chords {
        if !chord.is_silent() {
            chord.label = daw_analysis::chords::transpose_label(&chord.label, delta);
        }
    }
    document.key = document
        .key
        .as_deref()
        .map(|key| daw_analysis::chords::transpose_label(key, delta));
    document.transpose_semitones = semitones;
    dropped
}

/// How many bytes of keyed renders the session is holding, other than the key
/// it is in now.
///
/// Keeping them is what makes going back to a key already heard immediate, but
/// a three-minute song is about a quarter of a gigabyte per key, so it is worth
/// being able to see the bill and settle it.
#[must_use]
pub fn other_keys_size(session_dir: &Path, keep: i32) -> u64 {
    let mut total = 0;
    let Ok(entries) = std::fs::read_dir(session_dir.join("Audio").join("Keys")) else {
        return 0;
    };
    for entry in entries.flatten() {
        if entry.path() == keys_dir(session_dir, keep) {
            continue;
        }
        let Ok(files) = std::fs::read_dir(entry.path()) else {
            continue;
        };
        total += files
            .flatten()
            .filter_map(|file| file.metadata().ok())
            .map(|metadata| metadata.len())
            .sum::<u64>();
    }
    total
}

/// Deletes every keyed render except the one the session is playing.
///
/// Only touches `Audio/Keys`, which holds nothing that cannot be made again
/// from the original stems beside it.
///
/// # Errors
///
/// Returns an error if a folder cannot be removed.
pub fn forget_other_keys(session_dir: &Path, keep: i32) -> Result<u64> {
    let freed = other_keys_size(session_dir, keep);
    let keys = session_dir.join("Audio").join("Keys");
    let Ok(entries) = std::fs::read_dir(&keys) else {
        return Ok(0);
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path == keys_dir(session_dir, keep) || !path.is_dir() {
            continue;
        }
        std::fs::remove_dir_all(&path)
            .with_context(|| format!("failed to remove {}", path.display()))?;
    }
    Ok(freed)
}

/// Where renders for one key are kept, e.g. `Audio/Keys/-4`.
fn keys_dir(session_dir: &Path, semitones: i32) -> PathBuf {
    session_dir.join("Audio").join("Keys").join(format!(
        "{}{}",
        if semitones < 0 { "-" } else { "+" },
        semitones.abs()
    ))
}

/// Moves every transcribed note, returning how many fell outside 0–127.
///
/// Drum tracks keep their numbers: on the General MIDI kit note 38 is a snare
/// in every key, so transposing one would swap the drums around.
fn transpose_midi(document: &mut ProjectDocument, delta: i32) -> usize {
    let mut dropped = 0;
    for track in &mut document.tracks {
        if track.drum_kit {
            continue;
        }
        for clip in &mut track.midi_clips {
            let before = clip.notes.len();
            clip.notes
                .retain_mut(|note| match u8::try_from(i32::from(note.pitch) + delta) {
                    Ok(pitch) if pitch <= 127 => {
                        note.pitch = pitch;
                        true
                    }
                    _ => false,
                });
            dropped += before - clip.notes.len();
        }
    }
    dropped
}

#[cfg(test)]
mod tests {
    use super::*;
    use daw_project::{ChordEvent, ProjectClip, ProjectTrack};

    fn song(semitones: i32) -> ProjectDocument {
        let mut drums = ProjectTrack::new("Drums", daw_core::ChannelLayout::Stereo);
        drums.clips.push(clip("drums.wav"));
        let mut bass = ProjectTrack::new("Bass", daw_core::ChannelLayout::Stereo);
        bass.clips.push(clip("bass.wav"));
        let mut piano = ProjectTrack::instrument("Piano", Some(0));
        let mut notes = daw_midi::MidiClip::new("Piano", 0, 0);
        notes.notes = vec![daw_midi::Note::new(60, 100, 0, 480)];
        piano.midi_clips.push(notes);
        let mut kit = ProjectTrack::drum_track("Kit");
        let mut hits = daw_midi::MidiClip::new("Kit", 0, 0);
        hits.notes = vec![daw_midi::Note::new(38, 100, 0, 480)];
        kit.midi_clips.push(hits);
        ProjectDocument {
            tracks: vec![drums, bass, piano, kit],
            chords: vec![
                ChordEvent {
                    start_seconds: 0.0,
                    end_seconds: 1.0,
                    label: "Gm".to_owned(),
                    confidence: 1.0,
                },
                ChordEvent {
                    start_seconds: 1.0,
                    end_seconds: 2.0,
                    label: "N.C.".to_owned(),
                    confidence: 1.0,
                },
            ],
            key: Some("G minor".to_owned()),
            transpose_semitones: semitones,
            ..ProjectDocument::default()
        }
    }

    fn clip(name: &str) -> ProjectClip {
        ProjectClip {
            source_start_frame: 0,
            source_path: None,
            id: uuid::Uuid::new_v4(),
            name: name.to_owned(),
            path: PathBuf::from("Audio").join(name),
            start_frame: 0,
            end_frame: 48_000,
        }
    }

    /// The document half of a re-key. The audio half is ffmpeg, and is covered
    /// by transposing a real song end to end.
    fn rekey_document(document: &mut ProjectDocument, semitones: i32) -> usize {
        super::retune_document(document, semitones)
    }

    #[test]
    fn the_chart_the_key_and_the_transcription_move_together() {
        let mut document = song(0);
        rekey_document(&mut document, -4);
        assert_eq!(document.key.as_deref(), Some("D# minor"));
        assert_eq!(document.chords[0].label, "D#m");
        // "no chord" has no root to move.
        assert_eq!(document.chords[1].label, "N.C.");
        assert_eq!(document.tracks[2].midi_clips[0].notes[0].pitch, 56);
    }

    #[test]
    fn a_kit_keeps_its_numbers_because_they_are_not_pitches() {
        let mut document = song(0);
        rekey_document(&mut document, -4);
        assert_eq!(document.tracks[3].midi_clips[0].notes[0].pitch, 38);
    }

    #[test]
    fn a_second_change_is_measured_from_where_the_song_already_sits() {
        let mut document = song(0);
        rekey_document(&mut document, -4);
        // -4 then +2 is +6 from here, and lands two below the original.
        rekey_document(&mut document, 2);
        assert_eq!(document.transpose_semitones, 2);
        assert_eq!(document.key.as_deref(), Some("A minor"));
        assert_eq!(document.tracks[2].midi_clips[0].notes[0].pitch, 62);

        // …and going home restores exactly what was imported.
        rekey_document(&mut document, 0);
        assert_eq!(document.key.as_deref(), Some("G minor"));
        assert_eq!(document.chords[0].label, "Gm");
        assert_eq!(document.tracks[2].midi_clips[0].notes[0].pitch, 60);
    }

    #[test]
    fn a_render_lands_in_a_folder_named_for_its_key() {
        let session = Path::new("/songs/Take Five");
        assert_eq!(
            keys_dir(session, -4),
            Path::new("/songs/Take Five/Audio/Keys/-4")
        );
        assert_eq!(
            keys_dir(session, 2),
            Path::new("/songs/Take Five/Audio/Keys/+2")
        );
    }

    #[test]
    fn notes_pushed_off_the_keyboard_are_counted() {
        let mut document = song(0);
        document.tracks[2].midi_clips[0].notes = vec![
            daw_midi::Note::new(2, 100, 0, 480),
            daw_midi::Note::new(60, 100, 0, 480),
        ];
        assert_eq!(rekey_document(&mut document, -4), 1);
        assert_eq!(document.tracks[2].midi_clips[0].notes.len(), 1);
    }
}
