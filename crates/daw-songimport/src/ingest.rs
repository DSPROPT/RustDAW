//! Turning a finished DSPRO Studio project into a `RustDAW` session.
//!
//! The two applications disagree about sample rate: the pipeline works at
//! 44.1 kHz because that is what the source material is, while a `RustDAW`
//! session runs at whatever rate the interface opened, normally 48 kHz for a
//! Scarlett. The engine deliberately refuses mismatched media rather than
//! resampling behind the user's back, so conversion happens here, once, at
//! import, into files the session owns.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::process::Command;

use daw_core::ChannelLayout;
use daw_midi::{MidiClip, TempoMap};
use daw_project::{ProjectClip, ProjectDocument, ProjectTrack};
use uuid::Uuid;

use crate::manifest::SongManifest;

/// Below this peak a stem is silence in practice: nothing in it ever moves.
const SILENCE_CEILING_DB: f64 = -60.0;

/// How far below the loudest stem's mean level a stem stops being an instrument
/// and becomes leakage from the ones beside it.
///
/// Demucs always emits all six stems; on a song with no piano the piano stem is
/// what leaked into it and nothing else. Its peak is no help — a leaked snare
/// hit clips as high as a played one, which is why the peak test above passes
/// every stem of every real song — so the level that decides is the mean, taken
/// against the loudest stem so that it scales with the master's own loudness.
/// 35 dB down is inaudible against the rest of the song, and well clear of a
/// genuinely sparse part: a vocal that enters for two phrases sits 25 dB down.
const BLEED_FLOOR_DB: f64 = 35.0;

#[derive(Clone, Debug)]
#[allow(clippy::struct_excessive_bools, reason = "independent import switches")]
pub struct IngestOptions {
    /// Directory that will contain the new session folder.
    pub destination_root: PathBuf,
    /// Also import kick/snare/toms/cymbals. Off by default: they sum to the
    /// drum stem, so importing both plays the drums twice.
    pub include_drumkit: bool,
    /// Delay the song so its first downbeat lands on a bar line of the click.
    pub align_to_bar: bool,
    /// Skip stems that contain no audible signal.
    pub skip_silent: bool,
    /// Detect tempo natively instead of trusting the pipeline's beat grid.
    pub detect_tempo: bool,
    /// Roughly where to expect the tempo. Fast music is reported at half speed
    /// without this, because the audio alone cannot settle a tempo against its
    /// double — see [`daw_analysis::TempoHint`].
    pub tempo_hint: daw_analysis::TempoHint,
    /// Import the transcription as instrument tracks.
    pub import_midi: bool,
    /// Detect the chord chart and the key.
    pub detect_chords: bool,
    /// Semitones to move the song by, for practising in another key. Negative
    /// is down. The stems are pitch-shifted without changing their length, so
    /// the tempo, the beat grid and the bar alignment are unaffected.
    pub transpose_semitones: i32,
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self {
            destination_root: PathBuf::from("Songs"),
            include_drumkit: false,
            align_to_bar: true,
            skip_silent: true,
            detect_tempo: true,
            tempo_hint: daw_analysis::TempoHint::default(),
            import_midi: true,
            detect_chords: true,
            transpose_semitones: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Ingested {
    pub document: ProjectDocument,
    pub session_path: PathBuf,
    pub session_dir: PathBuf,
    /// Human-readable remarks: skipped stems, tempo drift, applied offset.
    /// Shown to the user because each one is a decision they may want to undo.
    pub notes: Vec<String>,
}

/// What a converted stem turned out to be.
#[derive(Clone, Copy, Debug)]
struct ConvertedAudio {
    frames: u64,
    peak_db: Option<f64>,
    mean_db: Option<f64>,
}

impl ConvertedAudio {
    fn is_silent(self) -> bool {
        self.peak_db.is_some_and(|peak| peak <= SILENCE_CEILING_DB)
    }

    /// Whether this stem is leakage, given the loudest mean in the same song.
    ///
    /// A stem whose mean could not be read is never dropped: a change in what
    /// ffmpeg prints must not silently lose part of a song.
    fn is_bleed(self, loudest_mean_db: f64) -> bool {
        self.mean_db
            .is_some_and(|mean| loudest_mean_db - mean >= BLEED_FLOOR_DB)
    }
}

/// Converts a DSPRO project into a saved `RustDAW` session.
///
/// `progress` is called with a 0.0–1.0 fraction and a label for each stem, so
/// a caller on a background thread can drive a progress bar.
///
/// # Errors
///
/// Returns an error if the manifest is unreadable, the project has no stems,
/// ffmpeg is missing or fails, or the session cannot be written.
#[allow(clippy::too_many_lines)]
pub fn ingest_project(
    project_dir: &Path,
    options: &IngestOptions,
    target_rate: u32,
    mut progress: impl FnMut(f32, &str),
) -> Result<Ingested> {
    let manifest = SongManifest::load(project_dir)?;
    if !manifest.has_stems() {
        bail!(
            "this project has no separated stems yet; let the pipeline finish the separate stage first"
        );
    }

    let mut sources = manifest.ordered_stems();
    if options.include_drumkit {
        sources.extend(manifest.ordered_drumkit());
    }

    // The key is part of what this session is: importing the same song at -2
    // and at -4 to rehearse both should give two obviously different folders,
    // not "Song" and "Song 2".
    // Said in words rather than with a sign: "+" is not legal in a file name
    // and sanitising turns it into an underscore, which left "(+4 st)" and
    // "(-4 st)" as "__4 st_" and "_-4 st_" side by side in the folder list.
    let name = match options.transpose_semitones {
        0 => manifest.display_name(),
        semitones => format!(
            "{} ({} st {})",
            manifest.display_name(),
            semitones.abs(),
            if semitones < 0 { "down" } else { "up" }
        ),
    };
    let file_stem = sanitize_file_name(&name);
    let session_dir = unique_directory(&options.destination_root, &file_stem)?;
    let audio_dir = session_dir.join("Audio");
    std::fs::create_dir_all(&audio_dir)
        .with_context(|| format!("failed to create {}", audio_dir.display()))?;

    let mut notes = Vec::new();
    let beats_per_bar = manifest
        .detected_tempo()
        .map_or(4, |detected| detected.beats_per_bar);

    let mut skipped = Vec::new();
    let mut converted = Vec::new();
    let total = sources.len().max(1);
    for (index, (stem_name, relative_path)) in sources.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let fraction = index as f32 / total as f32;
        progress(fraction, stem_name);

        let source = project_dir.join(relative_path);
        if !source.is_file() {
            skipped.push(format!("{stem_name} (file missing)"));
            continue;
        }
        let destination = audio_dir.join(format!("{stem_name}.wav"));
        // Always converted in the key it was recorded in. A transposition is
        // applied afterwards by the same code that re-keys a session later, so
        // the session keeps its original audio and changing key again shifts
        // that rather than shifting a shift.
        let audio = convert_audio(&source, &destination, target_rate, 0)
            .with_context(|| format!("failed to convert the {stem_name} stem"))?;

        if audio.frames == 0 || (options.skip_silent && audio.is_silent()) {
            let _ = std::fs::remove_file(&destination);
            skipped.push(stem_name.clone());
            continue;
        }
        converted.push((stem_name.clone(), destination, audio));
    }

    // Which stems are leakage can only be answered once they have all been
    // measured, because the answer is relative to the loudest one. That also
    // means the test can never empty a session: the loudest stem is not 35 dB
    // below itself, so it always survives.
    let loudest_mean_db = converted
        .iter()
        .filter_map(|(_, _, audio)| audio.mean_db)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut tracks = Vec::new();
    for (stem_name, destination, audio) in converted {
        if options.skip_silent && loudest_mean_db.is_finite() && audio.is_bleed(loudest_mean_db) {
            let _ = std::fs::remove_file(&destination);
            skipped.push(stem_name);
            continue;
        }
        let mut track = ProjectTrack::new(title_case(&stem_name), ChannelLayout::Stereo);
        track.clips.push(ProjectClip {
            source_start_frame: 0,
            id: Uuid::new_v4(),
            name: title_case(&stem_name),
            path: destination,
            start_frame: 0,
            end_frame: audio.frames,
            source_path: None,
        });
        tracks.push(track);
    }
    progress(1.0, "session");

    if tracks.is_empty() {
        bail!("every stem in this project was silent or missing");
    }

    if !skipped.is_empty() {
        notes.push(format!("Skipped silent or missing: {}", skipped.join(", ")));
    }

    progress(1.0, "tempo");
    let detection = detect_tempo(&audio_dir, options, target_rate, &mut notes);
    let tempo_map_beats = detection.beats.clone();
    // The click is a constant integer-BPM grid, so measure everything against
    // that exact tempo: aligning the song to any other value leaves the very
    // first bar off and the drift only grows.
    let click_tempo = detection.click_bpm.round().clamp(20.0, 300.0);

    // Shift the song so its first downbeat lands on bar 1 of the click.
    let offset_seconds = if options.align_to_bar {
        bar_alignment_offset(click_tempo, detection.first_downbeat, beats_per_bar)
    } else {
        0.0
    };
    let offset_frames = seconds_to_frames(offset_seconds, target_rate);
    if offset_frames > 0 {
        for clip in tracks.iter_mut().flat_map(|track| track.clips.iter_mut()) {
            clip.start_frame = offset_frames;
            clip.end_frame = clip.end_frame.saturating_add(offset_frames);
        }
        notes.push(format!(
            "Song delayed {offset_seconds:.2} s so its first downbeat lands on bar 1."
        ));
    }

    let mut document = ProjectDocument {
        name,
        sample_rate: target_rate,
        tempo: 120,
        meter_numerator: beats_per_bar,
        meter_denominator: 4,
        // The song is the reference; a click on top of it is rarely wanted at
        // first and is one keypress away.
        click_enabled: false,
        tracks,
        ..ProjectDocument::default()
    };
    document.set_tempo_map(detection.tempo_map);
    // `set_tempo_map` derives the tempo from the map's first interval, which is
    // one noisy beat; overwrite it with the robust global tempo the click and
    // the bar alignment were both computed from.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        document.tempo = click_tempo as u16;
    }

    if options.detect_chords {
        progress(1.0, "chords");
        match detect_chord_chart(&audio_dir, &tempo_map_beats, offset_seconds) {
            Ok((chords, key)) if !chords.is_empty() => {
                let named = chords.iter().filter(|event| !event.is_silent()).count();
                notes.push(format!(
                    "{named} chord(s) detected{}.",
                    key.as_ref()
                        .map_or_else(String::new, |key| format!(" in {key}"))
                ));
                document.chords = chords;
                document.key = key;
            }
            Ok(_) => notes.push("No chords could be detected.".to_owned()),
            Err(error) => notes.push(format!("Chord detection failed: {error}")),
        }
    }

    if options.import_midi {
        match import_midi_tracks(
            project_dir,
            &document.tempo_map(),
            offset_seconds,
            beats_per_bar,
        ) {
            Ok((instrument_tracks, skipped_midi)) => {
                if !skipped_midi.is_empty() {
                    notes.push(format!("MIDI tracks left out: {}", skipped_midi.join(", ")));
                }
                if instrument_tracks.is_empty() {
                    notes.push("No transcription in this project.".to_owned());
                } else {
                    let count = instrument_tracks.len();
                    document.tracks.extend(instrument_tracks);
                    let backend = manifest
                        .stages
                        .get("transcribe")
                        .and_then(|stage| stage.backend.as_deref())
                        .map_or_else(String::new, |backend| format!(" ({backend})"));
                    notes.push(format!(
                        "Imported {count} instrument track(s) from the transcription{backend}."
                    ));
                }
            }
            Err(error) => notes.push(format!("MIDI could not be imported: {error}")),
        }
    }

    // The song is in the key it was recorded in up to here. Moving it is the
    // same operation as changing key later, so it runs through the same code:
    // the session keeps its original stems and gains a shifted set beside them.
    if options.transpose_semitones != 0 {
        progress(1.0, "key");
        match crate::rekey::rekey_session(
            &mut document,
            &session_dir,
            options.transpose_semitones,
            // The import reports its own stages; the re-key's per-stem
            // progress would need a callback shared across its threads and
            // buys nothing over the "key" step already shown.
            &|_, _| {},
        ) {
            Ok(rekeyed) => {
                notes.push(format!(
                    "Transposed {:+} semitone(s); the drums keep their own pitch.{}",
                    rekeyed.semitones,
                    if has_rubberband() {
                        ""
                    } else {
                        " This ffmpeg has no rubberband filter, so the rougher resampling \
                         fallback was used."
                    }
                ));
                notes.extend(rekeyed.notes);
            }
            Err(error) => notes.push(format!("The song could not be transposed: {error:#}")),
        }
    }

    let session_path = session_dir.join(format!("{file_stem}.rustdaw.json"));
    daw_project::save_atomic(&document, &session_path)?;

    Ok(Ingested {
        document,
        session_path,
        session_dir,
        notes,
    })
}

/// Detects tempo from the converted audio.
///
/// The drum stem is the best evidence available: it carries the pulse without
/// the harmonic content that confuses onset detection. Falling back to the
/// full mix keeps songs with no drums working.
fn detect_tempo(
    audio_dir: &Path,
    options: &IngestOptions,
    target_rate: u32,
    notes: &mut Vec<String>,
) -> TempoDetection {
    if !options.detect_tempo {
        return TempoDetection::fallback(120.0);
    }
    // A hint that is not the default is worth recording: it changes the answer,
    // and someone reading the import notes later should know it was asked for.
    if (options.tempo_hint.centre_bpm() - daw_analysis::TempoHint::default().centre_bpm()).abs()
        > f64::EPSILON
    {
        notes.push(format!(
            "Tempo was detected expecting around {:.0} BPM.",
            options.tempo_hint.centre_bpm()
        ));
    }
    let candidates = ["drums.wav", "other.wav", "bass.wav"];
    let Some(source) = candidates
        .iter()
        .map(|name| audio_dir.join(name))
        .find(|path| path.is_file())
    else {
        notes.push("Nothing to analyse for tempo; left at 120 BPM.".to_owned());
        return TempoDetection::fallback(120.0);
    };

    // A wider tolerance keeps a steady song on one tempo: real recordings jitter
    // by a few BPM per beat, and a tight threshold turns that noise into a string
    // of spurious tempo changes that the click can never follow.
    match daw_analysis::analyse_wav_with(&source, 6.0, options.tempo_hint) {
        Ok(analysis) => {
            if !analysis.beats.is_usable() {
                notes.push(
                    "No clear pulse was found; tempo is a fallback and bar lines are a guess."
                        .to_owned(),
                );
                return TempoDetection::fallback(analysis.bpm());
            }
            let source_name = source
                .file_stem()
                .and_then(|name| name.to_str())
                .unwrap_or("audio");
            notes.push(format!(
                "Tempo {:.2} BPM detected from the {source_name} stem.",
                analysis.bpm()
            ));
            if !analysis.tempo_map.is_constant() {
                notes.push(format!(
                    "The song's tempo moves: {} tempo changes were kept.",
                    analysis.tempo_map.points().len() - 1
                ));
            }
            TempoDetection {
                first_downbeat: analysis.beats.first_downbeat(),
                beats: BeatGrid {
                    beat_times: analysis.beats.beat_times.clone(),
                    downbeat_index: analysis.beats.downbeat_index,
                },
                // The single global tempo drives the click; the map's tick-0
                // value is one noisy interval and drifts against the song.
                click_bpm: analysis.bpm(),
                tempo_map: analysis.tempo_map,
            }
        }
        Err(error) => {
            notes.push(format!(
                "Tempo detection failed ({error}); left at 120 BPM."
            ));
            let _ = target_rate;
            TempoDetection::fallback(120.0)
        }
    }
}

/// Everything tempo detection recovers: the map for MIDI and chords, the phase
/// for bar alignment, and one robust global tempo for the constant click.
struct TempoDetection {
    tempo_map: TempoMap,
    first_downbeat: f64,
    beats: BeatGrid,
    click_bpm: f64,
}

impl TempoDetection {
    fn fallback(bpm: f64) -> Self {
        Self {
            tempo_map: TempoMap::constant(bpm),
            first_downbeat: 0.0,
            beats: BeatGrid::default(),
            click_bpm: bpm,
        }
    }
}

/// The beat grid detection produced, kept for the chord decoder.
#[derive(Clone, Debug, Default)]
struct BeatGrid {
    beat_times: Vec<f64>,
    downbeat_index: usize,
}

/// Detects the chord chart from the harmonic stems.
///
/// Drums have no pitch and a singer's passing notes are not the chord, so both
/// are left out and only bass, guitar, piano and "other" are summed. This is
/// the single biggest thing that makes a chart usable — running the same
/// analysis on a full mix produces a chart that is mostly wrong.
fn detect_chord_chart(
    audio_dir: &Path,
    grid: &BeatGrid,
    offset_seconds: f64,
) -> Result<(Vec<daw_project::ChordEvent>, Option<String>)> {
    if grid.beat_times.len() < 2 {
        return Ok((Vec::new(), None));
    }
    let mut mixed: Vec<f32> = Vec::new();
    let mut rate = 0;
    for name in ["bass.wav", "guitar.wav", "piano.wav", "other.wav"] {
        let path = audio_dir.join(name);
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
    if mixed.is_empty() || rate == 0 {
        return Ok((Vec::new(), None));
    }

    let chromagram = daw_analysis::chroma::chromagram(&mixed, rate);
    let (spans, key) =
        daw_analysis::chords::detect_chords(&chromagram, &grid.beat_times, 4, grid.downbeat_index);
    // The audio was delayed to put its downbeat on bar 1; the chart has to move
    // with it or every chord would sit a fraction of a bar early.
    let chords = spans
        .into_iter()
        .map(|span| daw_project::ChordEvent {
            start_seconds: span.start_seconds + offset_seconds,
            end_seconds: span.end_seconds + offset_seconds,
            label: span.label(),
            confidence: span.confidence,
        })
        .collect();
    Ok((chords, key.map(|key| key.name())))
}

/// Seconds to delay the song so `first_downbeat` lands on a bar line.
/// Seconds of silence to prepend so the first downbeat lands on a bar line of a
/// constant `bpm` click. Measured at the exact tempo the click runs at, so the
/// downbeat and the click's bar one coincide.
fn bar_alignment_offset(bpm: f64, first_downbeat: f64, beats_per_bar: u16) -> f64 {
    if bpm <= 0.0 || first_downbeat <= 0.0 {
        return 0.0;
    }
    let seconds_per_bar = 60.0 / bpm * f64::from(beats_per_bar.max(1));
    if seconds_per_bar <= 0.0 {
        return 0.0;
    }
    let position_in_bar = first_downbeat % seconds_per_bar;
    if position_in_bar <= f64::EPSILON {
        0.0
    } else {
        seconds_per_bar - position_in_bar
    }
}

/// Reads `midi/song.mid` and turns each pitched track into an instrument track.
///
/// Returns the tracks and the names of any drum tracks that were left out.
fn import_midi_tracks(
    project_dir: &Path,
    tempo: &TempoMap,
    offset_seconds: f64,
    beats_per_bar: u16,
) -> Result<(Vec<ProjectTrack>, Vec<String>)> {
    let path = project_dir.join("midi/song.mid");
    if !path.is_file() {
        return Ok((Vec::new(), Vec::new()));
    }
    let bytes =
        std::fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let file = daw_midi::smf::parse(&bytes)?;

    // The file's ticks mean nothing on their own: they are relative to the
    // tempo map the pipeline wrote, which is not the one this session runs at.
    // Every note is therefore rebased through seconds — the only quantity the
    // two maps agree on — or a 120 BPM transcription would play a third too
    // slow against audio detected at 94.
    let rebase = |tick: u64| -> u64 {
        let seconds = file.tempo_map.tick_to_seconds(tick) + offset_seconds.max(0.0);
        tempo.seconds_to_tick(seconds)
    };

    let split_ticks = u64::from(CLIP_SPLIT_BARS)
        * u64::from(beats_per_bar)
        * u64::from(tempo.ticks_per_quarter());

    let mut tracks = Vec::new();
    let skipped = Vec::new();
    for source in file.sounding_tracks() {
        let name = if source.name.is_empty() {
            "MIDI".to_owned()
        } else {
            title_case(&source.name)
        };
        let notes = source
            .notes
            .iter()
            .map(|note| {
                let start = rebase(note.start_tick);
                let end = rebase(note.end_tick()).max(start + 1);
                daw_midi::Note::new(note.pitch, note.velocity, start, end - start)
            })
            .collect();
        // Channel 10 is the General MIDI drum kit, which the synth now plays.
        let mut track = if source.is_drums() {
            ProjectTrack::drum_track(name.clone())
        } else {
            ProjectTrack::instrument(name.clone(), source.program)
        };
        // Instrument tracks start quiet: the stems are the reference and the
        // synth is there to be brought up against them, not to compete.
        track.gain_db = -9.0;
        track.midi_clips = split_into_clips(&name, notes, split_ticks);
        tracks.push(track);
    }
    Ok((tracks, skipped))
}

/// A rest at least this many bars long ends a clip; the next note starts a new
/// one. An instrument that enters at the last chorus used to be drawn as one
/// clip from bar 1 to the end of the song with the notes somewhere inside it,
/// which reads as an empty track. Cutting at the long rests puts a block on the
/// timeline exactly where the instrument is playing and nowhere else. Two bars
/// is past any rest inside a phrase and short enough to separate the sections.
const CLIP_SPLIT_BARS: u16 = 2;

/// Cuts one track's notes into clips at its long rests.
///
/// `notes` are positioned on the session's timeline; each clip returned starts
/// at its own first note and holds its notes relative to that, which is how
/// [`MidiClip`] stores them.
fn split_into_clips(name: &str, mut notes: Vec<daw_midi::Note>, split_ticks: u64) -> Vec<MidiClip> {
    notes.sort_by_key(|note| (note.start_tick, note.pitch));

    let mut clips = Vec::new();
    let mut current: Vec<daw_midi::Note> = Vec::new();
    // The end of the last note still sounding, which is not the end of the last
    // note that started: a held chord under a melody outlasts the notes over it.
    let mut sounding_end = 0;
    for note in notes {
        if current.is_empty() {
            sounding_end = note.end_tick();
        } else if note.start_tick.saturating_sub(sounding_end) >= split_ticks {
            clips.push(gather_clip(name, std::mem::take(&mut current)));
            sounding_end = note.end_tick();
        } else {
            sounding_end = sounding_end.max(note.end_tick());
        }
        current.push(note);
    }
    if !current.is_empty() {
        clips.push(gather_clip(name, current));
    }
    clips
}

/// One clip from notes already known to belong together, rebased onto its start.
fn gather_clip(name: &str, notes: Vec<daw_midi::Note>) -> MidiClip {
    let start = notes.first().map_or(0, |note| note.start_tick);
    let end = notes
        .iter()
        .map(|note| note.end_tick())
        .max()
        .unwrap_or(start);
    let mut clip = MidiClip::new(name.to_owned(), start, end.saturating_sub(start).max(1));
    clip.notes = notes
        .into_iter()
        .map(|note| {
            daw_midi::Note::new(
                note.pitch,
                note.velocity,
                note.start_tick - start,
                note.length_ticks,
            )
        })
        .collect();
    clip
}

/// Resamples one file into the session, reporting its length and peak.
///
/// `volumedetect` rides along in the same pass so the peak costs no extra
/// read; a stem is roughly 50 MB once converted and scanning them twice would
/// double the import's disk traffic for nothing.
/// The platform's usual way to install ffmpeg, for the error hint.
fn ffmpeg_install_hint() -> &'static str {
    if cfg!(target_os = "macos") {
        "brew install ffmpeg"
    } else {
        "sudo apt install ffmpeg"
    }
}

/// Resolves the ffmpeg binary to run.
///
/// A GUI app launched from Finder or Launchpad inherits a minimal `PATH`
/// (`/usr/bin:/bin:…`) that excludes Homebrew, so a bare `ffmpeg` is not found
/// even when it is installed. `FFMPEG` overrides; otherwise the common install
/// locations are checked before falling back to the name on `PATH`.
fn ffmpeg_program() -> std::path::PathBuf {
    if let Some(explicit) = std::env::var_os("FFMPEG").filter(|value| !value.is_empty()) {
        return std::path::PathBuf::from(explicit);
    }
    for candidate in [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
    ] {
        if Path::new(candidate).is_file() {
            return std::path::PathBuf::from(candidate);
        }
    }
    std::path::PathBuf::from("ffmpeg")
}

/// Whether a stem or track is a drum or a piece of a kit, and so has no key to
/// move. Matched on the name, which is the only thing a stem and the track made
/// from it have in common once the session is written.
#[must_use]
pub fn is_percussion(stem_name: &str) -> bool {
    let name = stem_name.to_ascii_lowercase();
    name == "drums"
        || crate::manifest::DRUMKIT_ORDER
            .iter()
            .any(|part| name == *part)
}

/// The widest shift offered. Beyond an octave the fallback filter chain leaves
/// atempo's supported range, and the result stops being music worth practising
/// against anyway.
pub const MAX_TRANSPOSE_SEMITONES: i32 = 12;

/// Whether this ffmpeg was built with librubberband.
///
/// Checked once and remembered: it costs a process launch, and every stem of
/// every import would otherwise ask again.
fn has_rubberband() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        Command::new(ffmpeg_program())
            .args(["-hide_banner", "-filters"])
            .output()
            .is_ok_and(|output| {
                String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .any(|line| line.contains(" rubberband "))
            })
    })
}

/// The ffmpeg filter that moves audio by `semitones` while leaving its length
/// alone, or `None` when there is nothing to shift.
///
/// Rubberband is a phase vocoder written for music and is what this should use.
/// Where ffmpeg was built without it, resampling moves the pitch and `atempo`
/// puts the length back — cruder, audibly so on a full mix, but it means an
/// import still transposes on a machine whose ffmpeg is a plain build.
fn pitch_filter(semitones: i32, rate: u32) -> Option<String> {
    if semitones == 0 {
        return None;
    }
    let clamped = semitones.clamp(-MAX_TRANSPOSE_SEMITONES, MAX_TRANSPOSE_SEMITONES);
    let ratio = (f64::from(clamped) / 12.0).exp2();
    if has_rubberband() {
        return Some(format!("rubberband=pitch={ratio:.9}"));
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let shifted_rate = (f64::from(rate) * ratio).round() as u32;
    Some(format!(
        "asetrate={shifted_rate},aresample={rate},atempo={:.9}",
        1.0 / ratio
    ))
}

/// Shifts one already-converted stem into `destination`, leaving its sample
/// rate and its length exactly as they are.
///
/// Used when a session changes key: unlike [`convert_audio`] there is no
/// resampling or level measurement to do, because the stem is already at the
/// session's rate and was measured when it was imported.
pub(crate) fn convert_pitch(source: &Path, destination: &Path, semitones: i32) -> Result<()> {
    let rate = hound::WavReader::open(source)
        .with_context(|| format!("failed to open {}", source.display()))?
        .spec()
        .sample_rate;
    let Some(filter) = pitch_filter(semitones, rate) else {
        std::fs::copy(source, destination)
            .with_context(|| format!("failed to copy {}", source.display()))?;
        return Ok(());
    };
    let output = Command::new(ffmpeg_program())
        .arg("-nostdin")
        .arg("-y")
        .args(["-v", "error"])
        .arg("-i")
        .arg(source)
        .args(["-af", &filter])
        .args(["-c:a", "pcm_s24le"])
        .arg(destination)
        .output()
        .with_context(|| {
            format!(
                "failed to run ffmpeg; install it with `{}`",
                ffmpeg_install_hint()
            )
        })?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr);
        let reason = detail.lines().last().unwrap_or("ffmpeg failed");
        let _ = std::fs::remove_file(destination);
        bail!("ffmpeg could not shift {}: {reason}", source.display());
    }
    Ok(())
}

fn convert_audio(
    source: &Path,
    destination: &Path,
    target_rate: u32,
    semitones: i32,
) -> Result<ConvertedAudio> {
    let filters = match pitch_filter(semitones, target_rate) {
        // The shift runs before volumedetect so the peak reported is the one
        // actually written, not the level before shifting.
        Some(shift) => format!("{shift},volumedetect"),
        None => "volumedetect".to_owned(),
    };
    let output = Command::new(ffmpeg_program())
        .arg("-nostdin")
        .arg("-y")
        .args(["-v", "info"])
        .arg("-i")
        .arg(source)
        .args(["-af", &filters])
        .args(["-ar", &target_rate.to_string()])
        .args(["-ac", "2"])
        .args(["-c:a", "pcm_s24le"])
        .arg(destination)
        .output()
        .with_context(|| {
            format!(
                "failed to run ffmpeg; install it with `{}`",
                ffmpeg_install_hint()
            )
        })?;

    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr);
        let reason = detail.lines().last().unwrap_or("ffmpeg failed");
        bail!("ffmpeg could not convert {}: {reason}", source.display());
    }

    let frames = wav_frame_count(destination)?;
    let report = String::from_utf8_lossy(&output.stderr);
    Ok(ConvertedAudio {
        frames,
        peak_db: parse_volume(&report, "max_volume:"),
        mean_db: parse_volume(&report, "mean_volume:"),
    })
}

/// Reads the frame count from a WAV header without decoding the audio.
fn wav_frame_count(path: &Path) -> Result<u64> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    Ok(u64::from(reader.duration()))
}

/// Extracts `max_volume: -12.3 dB` or `mean_volume: -22.7 dB` from ffmpeg's
/// `volumedetect` output.
/// Returns `None` when the line is absent or unparseable, which is treated as
/// "assume there is signal" so a parsing change never silently drops a stem.
fn parse_volume(stderr: &str, label: &str) -> Option<f64> {
    let line = stderr.lines().rev().find(|line| line.contains(label))?;
    let value = line.split(label).nth(1)?.trim();
    let number = value.strip_suffix("dB").unwrap_or(value).trim();
    if number.eq_ignore_ascii_case("-inf") {
        return Some(f64::NEG_INFINITY);
    }
    number.parse().ok()
}

/// Picks a session directory that does not exist yet, so importing the same
/// song twice never overwrites the first import's audio or takes.
fn unique_directory(root: &Path, base: &str) -> Result<PathBuf> {
    std::fs::create_dir_all(root)
        .with_context(|| format!("failed to create {}", root.display()))?;
    let first = root.join(base);
    if !first.exists() {
        return Ok(first);
    }
    for suffix in 2..1000_u32 {
        let candidate = root.join(format!("{base} ({suffix})"));
        if !candidate.exists() {
            return Ok(candidate);
        }
    }
    bail!(
        "could not find an unused folder name under {}",
        root.display()
    )
}

fn sanitize_file_name(name: &str) -> String {
    let sanitized = name
        .chars()
        .map(|character| {
            if character.is_alphanumeric() || matches!(character, '-' | '_' | ' ') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    let trimmed = sanitized.trim();
    if trimmed.is_empty() {
        "Imported Song".to_owned()
    } else {
        trimmed.to_owned()
    }
}

fn title_case(name: &str) -> String {
    let mut characters = name.chars();
    match characters.next() {
        Some(first) => first.to_uppercase().collect::<String>() + characters.as_str(),
        None => String::new(),
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn seconds_to_frames(seconds: f64, rate: u32) -> u64 {
    if seconds <= 0.0 {
        return 0;
    }
    (seconds * f64::from(rate)).round() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_levels_are_read_from_ffmpeg_output() {
        let stderr = "[Parsed_volumedetect_0 @ 0x55] n_samples: 100\n\
                      [Parsed_volumedetect_0 @ 0x55] mean_volume: -22.7 dB\n\
                      [Parsed_volumedetect_0 @ 0x55] max_volume: -0.4 dB\n";
        assert_eq!(parse_volume(stderr, "max_volume:"), Some(-0.4));
        assert_eq!(parse_volume(stderr, "mean_volume:"), Some(-22.7));
    }

    #[test]
    fn digital_silence_reports_negative_infinity() {
        let stderr = "[Parsed_volumedetect_0 @ 0x55] max_volume: -inf dB\n";
        assert_eq!(parse_volume(stderr, "max_volume:"), Some(f64::NEG_INFINITY));
        assert!(
            ConvertedAudio {
                frames: 1,
                peak_db: Some(f64::NEG_INFINITY),
                mean_db: Some(f64::NEG_INFINITY),
            }
            .is_silent()
        );
    }

    #[test]
    fn unreadable_output_keeps_the_stem() {
        assert_eq!(parse_volume("ffmpeg version 6.1.1\n", "max_volume:"), None);
        let unmeasured = ConvertedAudio {
            frames: 1,
            peak_db: None,
            mean_db: None,
        };
        assert!(
            !unmeasured.is_silent(),
            "an unparsed peak must not be treated as silence"
        );
        assert!(
            !unmeasured.is_bleed(-10.0),
            "an unparsed mean must not be treated as leakage"
        );
    }

    #[test]
    fn a_quiet_but_audible_stem_is_kept() {
        assert!(
            !ConvertedAudio {
                frames: 1,
                peak_db: Some(-42.0),
                mean_db: Some(-52.0),
            }
            .is_silent()
        );
    }

    #[test]
    fn leakage_is_told_from_a_sparse_part_by_its_mean() {
        // Measured from a real import: the guitar stem of a song with no
        // guitar. Its peak is a leaked transient, so only the mean can tell.
        let leaked = ConvertedAudio {
            frames: 1,
            peak_db: Some(-14.4),
            mean_db: Some(-51.8),
        };
        assert!(!leaked.is_silent(), "the peak test cannot see this");
        assert!(leaked.is_bleed(-14.6));

        // A vocal that only enters for a couple of phrases, from the same set.
        let sparse = ConvertedAudio {
            frames: 1,
            peak_db: Some(-13.3),
            mean_db: Some(-44.3),
        };
        assert!(!sparse.is_bleed(-19.0));
    }

    #[test]
    fn a_track_is_cut_into_clips_at_its_long_rests() {
        let bar = 4 * 960;
        let split = 2 * bar;
        let note = |start: u64| daw_midi::Note::new(60, 100, start, 480);
        let clips = split_into_clips(
            "Piano",
            vec![note(0), note(960), note(10 * bar), note(10 * bar + 960)],
            split,
        );
        assert_eq!(clips.len(), 2);
        assert_eq!(clips[0].start_tick, 0);
        assert_eq!(clips[0].length_ticks, 960 + 480);
        // The second clip starts where the instrument comes back in, not at
        // bar 1, and its notes are relative to that.
        assert_eq!(clips[1].start_tick, 10 * bar);
        assert_eq!(clips[1].notes[0].start_tick, 0);
        assert_eq!(clips[1].notes[1].start_tick, 960);
    }

    #[test]
    fn a_rest_inside_a_phrase_does_not_cut_the_clip() {
        let split = 2 * 4 * 960;
        let note = |start: u64| daw_midi::Note::new(60, 100, start, 480);
        let clips = split_into_clips("Bass", vec![note(0), note(4 * 960)], split);
        assert_eq!(clips.len(), 1);
        assert_eq!(clips[0].notes.len(), 2);
    }

    #[test]
    fn a_note_held_under_the_rest_keeps_the_clip_whole() {
        // The melody stops for three bars but a pad is still sounding, so
        // there is no rest to cut at.
        let bar = 4 * 960;
        let held = daw_midi::Note::new(48, 90, 0, 4 * bar);
        let melody = daw_midi::Note::new(72, 100, 3 * bar, 480);
        let clips = split_into_clips("Pad", vec![held, melody], 2 * bar);
        assert_eq!(clips.len(), 1, "a sounding note is not a rest");
    }

    #[test]
    fn a_track_with_no_rests_stays_one_clip() {
        let note = |start: u64| daw_midi::Note::new(60, 100, start, 480);
        let clips = split_into_clips("Drums", vec![note(0), note(480), note(960)], 2 * 4 * 960);
        assert_eq!(clips.len(), 1);
        assert_eq!(clips[0].start_tick, 0);
        assert_eq!(clips[0].length_ticks, 960 + 480);
    }

    #[test]
    fn a_shift_of_nothing_adds_no_filter() {
        assert!(pitch_filter(0, 48_000).is_none());
    }

    #[test]
    fn a_shift_moves_by_equal_temperament() {
        // Whichever filter this ffmpeg can offer, the ratio in it is the one
        // twelve-tone equal temperament asks for: an octave down is half.
        let down_an_octave = pitch_filter(-12, 48_000).expect("a shift");
        assert!(
            down_an_octave.contains("0.500000000") || down_an_octave.contains("asetrate=24000"),
            "{down_an_octave}"
        );
        let up_an_octave = pitch_filter(12, 48_000).expect("a shift");
        assert!(
            up_an_octave.contains("2.000000000") || up_an_octave.contains("asetrate=96000"),
            "{up_an_octave}"
        );
        // A fifth is 1.4983, not 1.5: tempered, not just intonation.
        let up_a_fifth = pitch_filter(7, 48_000).expect("a shift");
        assert!(
            up_a_fifth.contains("1.498307077") || up_a_fifth.contains("asetrate=71919"),
            "{up_a_fifth}"
        );
    }

    #[test]
    fn the_fallback_stays_inside_atempos_supported_range() {
        // atempo refuses anything outside 0.5-100, which is the reason the
        // shift is capped at an octave.
        for semitones in [-MAX_TRANSPOSE_SEMITONES, MAX_TRANSPOSE_SEMITONES] {
            let ratio = (f64::from(semitones) / 12.0).exp2();
            assert!((0.5..=100.0).contains(&(1.0 / ratio)), "{semitones}");
        }
    }

    #[test]
    fn a_kit_is_recognised_whatever_it_is_called() {
        for name in ["drums", "Drums", "kick", "snare", "toms", "cymbals"] {
            assert!(is_percussion(name), "{name}");
        }
        for name in ["bass", "guitar", "piano", "other", "vocals", "strings"] {
            assert!(!is_percussion(name), "{name}");
        }
    }

    #[test]
    fn file_names_lose_separators_and_keep_spaces() {
        assert_eq!(
            sanitize_file_name("AC/DC - Back in Black"),
            "AC_DC - Back in Black"
        );
        assert_eq!(sanitize_file_name("../../etc"), "______etc");
        assert_eq!(sanitize_file_name("   "), "Imported Song");
    }

    #[test]
    fn seconds_convert_to_frames_at_the_session_rate() {
        assert_eq!(seconds_to_frames(1.5, 48_000), 72_000);
        assert_eq!(seconds_to_frames(-1.0, 48_000), 0);
        assert_eq!(seconds_to_frames(0.0, 48_000), 0);
    }

    #[test]
    fn repeated_imports_get_their_own_folder() {
        let root = std::env::temp_dir().join(format!("rustdaw-ingest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let first = unique_directory(&root, "Song").unwrap();
        std::fs::create_dir_all(&first).unwrap();
        let second = unique_directory(&root, "Song").unwrap();
        assert_eq!(first.file_name().unwrap(), "Song");
        assert_eq!(second.file_name().unwrap(), "Song (2)");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn stem_names_become_track_names() {
        assert_eq!(title_case("drums"), "Drums");
        assert_eq!(title_case(""), "");
    }
}
