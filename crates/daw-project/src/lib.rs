//! Versioned `RustDAW` project documents and atomic persistence.

use anyhow::{Context, Result, bail};
use daw_core::ChannelLayout;
use daw_midi::{MidiClip, TempoMap};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use uuid::Uuid;

pub const CURRENT_PROJECT_VERSION: u32 = 2;
/// Versions this build can open. Version 1 predates MIDI and the tempo map.
pub const SUPPORTED_PROJECT_VERSIONS: [u32; 2] = [1, 2];

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ProjectDocument {
    pub version: u32,
    pub name: String,
    pub sample_rate: u32,
    /// The tempo shown in the transport. For a session with tempo changes this
    /// is the tempo at the start; [`Self::tempo_map`] is the authority.
    pub tempo: u16,
    #[serde(default = "default_meter_numerator")]
    pub meter_numerator: u16,
    #[serde(default = "default_meter_denominator")]
    pub meter_denominator: u16,
    pub click_enabled: bool,
    /// Musical time for the whole session. Absent in version 1 documents, which
    /// migrate to a constant map at their stored tempo.
    #[serde(default)]
    pub tempo_map: Option<TempoMap>,
    /// The detected chord chart, in timeline order.
    #[serde(default)]
    pub chords: Vec<ChordEvent>,
    /// The detected key, e.g. "E minor".
    #[serde(default)]
    pub key: Option<String>,
    pub tracks: Vec<ProjectTrack>,
}

/// One chord held over a stretch of the timeline.
///
/// Stored in seconds rather than ticks: the chart describes the recording, and
/// must not move if the session tempo is later corrected by hand.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ChordEvent {
    pub start_seconds: f64,
    pub end_seconds: f64,
    /// The printed chord, or "N.C." where nothing tonal is playing.
    pub label: String,
    pub confidence: f32,
}

impl ChordEvent {
    #[must_use]
    pub fn is_silent(&self) -> bool {
        self.label == "N.C."
    }
}

impl Default for ProjectDocument {
    fn default() -> Self {
        Self {
            version: CURRENT_PROJECT_VERSION,
            name: "Untitled Session".to_owned(),
            sample_rate: 48_000,
            tempo: 120,
            meter_numerator: default_meter_numerator(),
            meter_denominator: default_meter_denominator(),
            click_enabled: true,
            tempo_map: None,
            chords: Vec::new(),
            key: None,
            tracks: vec![ProjectTrack::new("Audio 1", ChannelLayout::Mono)],
        }
    }
}

impl ProjectDocument {
    /// The session's musical time, falling back to a constant map at the
    /// transport tempo when the document has none.
    #[must_use]
    pub fn tempo_map(&self) -> TempoMap {
        self.tempo_map
            .clone()
            .unwrap_or_else(|| TempoMap::constant(f64::from(self.tempo)))
    }

    /// Replaces the tempo map and keeps the displayed tempo in step with it.
    pub fn set_tempo_map(&mut self, map: TempoMap) {
        // Safe: the map clamps every tempo into 20..=300 on construction.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            self.tempo = map.bpm_at_tick(0).round() as u16;
        }
        self.tempo_map = Some(map);
    }

    /// The chord sounding at a moment, for the chart display.
    #[must_use]
    pub fn chord_at(&self, seconds: f64) -> Option<&ChordEvent> {
        self.chords
            .iter()
            .find(|event| seconds >= event.start_seconds && seconds < event.end_seconds)
    }

    #[must_use]
    pub fn has_midi(&self) -> bool {
        self.tracks.iter().any(|track| !track.midi_clips.is_empty())
    }
}

const fn default_meter_numerator() -> u16 {
    4
}

const fn default_meter_denominator() -> u16 {
    4
}

/// What a track carries. Audio tracks record and play WAV clips; instrument
/// tracks hold notes and are played by the built-in synth.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum TrackKind {
    #[default]
    Audio,
    Instrument,
}

impl TrackKind {
    #[must_use]
    pub const fn is_instrument(self) -> bool {
        matches!(self, Self::Instrument)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ProjectTrack {
    #[serde(default = "Uuid::new_v4")]
    pub id: Uuid,
    pub name: String,
    #[serde(default)]
    pub kind: TrackKind,
    pub layout: ChannelLayout,
    pub input_left: usize,
    pub input_right: usize,
    pub muted: bool,
    #[serde(default)]
    pub solo: bool,
    pub gain_db: f32,
    #[serde(default)]
    pub pan: f32,
    #[serde(default)]
    pub effects: TrackEffects,
    pub clips: Vec<ProjectClip>,
    /// Notes, for instrument tracks. Always empty on audio tracks.
    #[serde(default)]
    pub midi_clips: Vec<MidiClip>,
    /// General MIDI program the synth plays. `None` means program 0.
    #[serde(default)]
    pub program: Option<u8>,
    /// Play this track on the drum kit rather than a pitched instrument, the
    /// way MIDI channel 10 works.
    #[serde(default)]
    pub drum_kit: bool,
}

impl ProjectTrack {
    #[must_use]
    pub fn new(name: impl Into<String>, layout: ChannelLayout) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            kind: TrackKind::Audio,
            layout,
            input_left: usize::from(layout == ChannelLayout::Mono),
            input_right: 1,
            muted: false,
            solo: false,
            gain_db: 0.0,
            pan: 0.0,
            effects: TrackEffects::default(),
            clips: Vec::new(),
            midi_clips: Vec::new(),
            program: None,
            drum_kit: false,
        }
    }

    /// An instrument track, played by the synth rather than from disk.
    #[must_use]
    pub fn instrument(name: impl Into<String>, program: Option<u8>) -> Self {
        Self {
            kind: TrackKind::Instrument,
            program,
            ..Self::new(name, ChannelLayout::Stereo)
        }
    }

    /// An instrument track played by the General MIDI drum kit.
    #[must_use]
    pub fn drum_track(name: impl Into<String>) -> Self {
        Self {
            drum_kit: true,
            ..Self::instrument(name, None)
        }
    }

    /// The General MIDI program this track plays.
    #[must_use]
    pub fn gm_program(&self) -> u8 {
        self.program.unwrap_or(0).min(127)
    }

    /// Highest tick any note on this track reaches.
    #[must_use]
    pub fn midi_end_tick(&self) -> u64 {
        self.midi_clips
            .iter()
            .map(daw_midi::MidiClip::end_tick)
            .max()
            .unwrap_or(0)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct TrackEffects {
    pub eq_enabled: bool,
    pub low_db: f32,
    pub mid_db: f32,
    pub high_db: f32,
    pub compressor_enabled: bool,
    pub compressor_threshold_db: f32,
    pub compressor_ratio: f32,
    pub compressor_attack_ms: f32,
    pub compressor_release_ms: f32,
    pub compressor_makeup_db: f32,
    pub gate_enabled: bool,
    pub gate_threshold_db: f32,
    pub gate_release_ms: f32,
}

impl Default for TrackEffects {
    fn default() -> Self {
        Self {
            eq_enabled: false,
            low_db: 0.0,
            mid_db: 0.0,
            high_db: 0.0,
            compressor_enabled: false,
            compressor_threshold_db: -18.0,
            compressor_ratio: 4.0,
            compressor_attack_ms: 10.0,
            compressor_release_ms: 120.0,
            compressor_makeup_db: 0.0,
            gate_enabled: false,
            gate_threshold_db: -45.0,
            gate_release_ms: 120.0,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ProjectClip {
    #[serde(default = "Uuid::new_v4")]
    pub id: Uuid,
    pub name: String,
    pub path: PathBuf,
    pub start_frame: u64,
    pub end_frame: u64,
}

/// Writes a project via a sibling temporary file and atomic rename.
///
/// # Errors
///
/// Returns an error if serialization, writing, syncing, or replacement fails.
pub fn save_atomic(document: &ProjectDocument, path: &Path) -> Result<()> {
    if document.version != CURRENT_PROJECT_VERSION {
        bail!("refusing to save an unsupported project version");
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let temporary = temporary_path(path);
    let bytes = serde_json::to_vec_pretty(document).context("failed to serialize project")?;
    {
        use std::io::Write;
        let mut file = std::fs::File::create(&temporary)
            .with_context(|| format!("failed to create {}", temporary.display()))?;
        file.write_all(&bytes).context("failed to write project")?;
        file.sync_all().context("failed to sync project")?;
    }
    std::fs::rename(&temporary, path)
        .with_context(|| format!("failed to replace {}", path.display()))?;
    Ok(())
}

/// Loads and validates a project document.
///
/// # Errors
///
/// Returns an error if the file is missing, invalid, or uses an unsupported
/// schema version.
pub fn load(path: &Path) -> Result<ProjectDocument> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read project {}", path.display()))?;
    let mut document: ProjectDocument =
        serde_json::from_slice(&bytes).context("project file is invalid")?;
    if !SUPPORTED_PROJECT_VERSIONS.contains(&document.version) {
        bail!(
            "project version {} is unsupported (this build reads {:?})",
            document.version,
            SUPPORTED_PROJECT_VERSIONS
        );
    }
    migrate(&mut document);
    Ok(document)
}

/// Brings an older document up to [`CURRENT_PROJECT_VERSION`].
///
/// Version 1 had no musical time at all, so its single integer tempo becomes a
/// constant tempo map. Every field added since carries a serde default, which
/// is why the migration has so little to do.
fn migrate(document: &mut ProjectDocument) {
    if document.version < 2 && document.tempo_map.is_none() {
        document.tempo_map = Some(TempoMap::constant(f64::from(document.tempo)));
    }
    document.version = CURRENT_PROJECT_VERSION;
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".tmp");
    PathBuf::from(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_round_trip_preserves_session() {
        let mut project = ProjectDocument::default();
        project.tracks[0].clips.push(ProjectClip {
            id: Uuid::new_v4(),
            name: "Take 1".to_owned(),
            path: PathBuf::from("Audio/Take_1.wav"),
            start_frame: 128,
            end_frame: 48_128,
        });
        let json = serde_json::to_vec(&project).unwrap();
        let restored: ProjectDocument = serde_json::from_slice(&json).unwrap();
        assert_eq!(restored, project);
    }

    #[test]
    fn rejects_unknown_schema_version() {
        let json = br#"{"version":99,"name":"x","sample_rate":48000,"tempo":120,"click_enabled":true,"tracks":[]}"#;
        let project: ProjectDocument = serde_json::from_slice(json).unwrap();
        assert_ne!(project.version, CURRENT_PROJECT_VERSION);
        assert!(!SUPPORTED_PROJECT_VERSIONS.contains(&project.version));
    }

    #[test]
    fn a_version_one_session_migrates_to_a_tempo_map() {
        let json = br#"{
            "version":1,"name":"Old","sample_rate":48000,"tempo":137,
            "click_enabled":true,"tracks":[]
        }"#;
        let mut project: ProjectDocument = serde_json::from_slice(json).unwrap();
        assert!(project.tempo_map.is_none());
        migrate(&mut project);
        assert_eq!(project.version, CURRENT_PROJECT_VERSION);
        let map = project.tempo_map.expect("migration must add musical time");
        assert!((map.bpm_at_tick(0) - 137.0).abs() < f64::EPSILON);
        assert!(map.is_constant());
    }

    #[test]
    fn a_version_one_track_gains_midi_fields_without_losing_audio() {
        let json = br#"{
            "version":1,"name":"Old","sample_rate":48000,"tempo":120,
            "click_enabled":true,"tracks":[{
                "name":"Gtr","layout":"Mono","input_left":0,"input_right":1,
                "muted":false,"gain_db":0.0,"clips":[{
                    "name":"Take","path":"Take.wav","start_frame":0,"end_frame":48000
                }]
            }]
        }"#;
        let project: ProjectDocument = serde_json::from_slice(json).unwrap();
        let track = &project.tracks[0];
        assert_eq!(track.kind, TrackKind::Audio);
        assert!(track.midi_clips.is_empty());
        assert_eq!(track.clips.len(), 1, "audio clips must survive migration");
        assert!(!project.has_midi());
    }

    #[test]
    fn instrument_tracks_round_trip_with_their_notes() {
        use daw_midi::{MidiClip, Note};
        let mut clip = MidiClip::new("Riff", 0, 0);
        clip.insert_note(Note::new(60, 100, 0, 480));
        clip.insert_note(Note::new(67, 90, 480, 480));
        let mut track = ProjectTrack::instrument("Piano", Some(0));
        track.midi_clips.push(clip);

        let mut project = ProjectDocument {
            tracks: vec![track],
            ..ProjectDocument::default()
        };
        project.set_tempo_map(daw_midi::TempoMap::constant(96.0));

        let restored: ProjectDocument =
            serde_json::from_slice(&serde_json::to_vec(&project).unwrap()).unwrap();
        assert_eq!(restored, project);
        assert!(restored.has_midi());
        assert_eq!(restored.tracks[0].kind, TrackKind::Instrument);
        assert_eq!(restored.tracks[0].midi_clips[0].notes.len(), 2);
        assert_eq!(restored.tempo, 96);
    }

    #[test]
    fn setting_a_tempo_map_updates_the_displayed_tempo() {
        let mut project = ProjectDocument::default();
        project.set_tempo_map(daw_midi::TempoMap::constant(143.6));
        assert_eq!(project.tempo, 144);
        assert!((project.tempo_map().bpm_at_tick(0) - 143.6).abs() < 1e-9);
    }

    #[test]
    fn a_document_without_a_map_still_reports_musical_time() {
        let project = ProjectDocument {
            tempo: 100,
            tempo_map: None,
            ..ProjectDocument::default()
        };
        assert!((project.tempo_map().bpm_at_tick(0) - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn legacy_session_without_ids_receives_stable_identifiers() {
        let json = br#"{
            "version":1,"name":"Legacy","sample_rate":48000,"tempo":120,
            "click_enabled":true,"tracks":[{
                "name":"Audio 1","layout":"Mono","input_left":0,"input_right":1,
                "muted":false,"gain_db":0.0,"clips":[{
                    "name":"Take","path":"Take.wav","start_frame":0,"end_frame":48000
                }]
            }]
        }"#;
        let project: ProjectDocument = serde_json::from_slice(json).unwrap();
        assert!(!project.tracks[0].id.is_nil());
        assert!(!project.tracks[0].clips[0].id.is_nil());

        let restored: ProjectDocument =
            serde_json::from_slice(&serde_json::to_vec(&project).unwrap()).unwrap();
        assert_eq!(restored.tracks[0].id, project.tracks[0].id);
        assert_eq!(
            restored.tracks[0].clips[0].id,
            project.tracks[0].clips[0].id
        );
    }

    #[test]
    fn atomic_file_round_trip_replaces_project() {
        let unique = format!(
            "rustdaw-project-test-{}-{}.json",
            std::process::id(),
            std::thread::current().name().unwrap_or("thread")
        );
        let path = std::env::temp_dir().join(unique);
        let mut project = ProjectDocument {
            name: "First".to_owned(),
            ..ProjectDocument::default()
        };
        save_atomic(&project, &path).unwrap();
        project.name = "Second".to_owned();
        save_atomic(&project, &path).unwrap();

        assert_eq!(load(&path).unwrap(), project);
        std::fs::remove_file(path).unwrap();
    }
}
