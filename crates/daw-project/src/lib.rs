//! Versioned `RustDAW` project documents and atomic persistence.

pub mod chart;

pub use chart::{ChartBeat, chord_chart, format_chart};

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
    /// The detected key, e.g. "E minor". Follows [`Self::transpose_semitones`]:
    /// it names the key the session is in now, not the one it was recorded in.
    #[serde(default)]
    pub key: Option<String>,
    /// Semitones this session has been moved from the key it came in at.
    ///
    /// Set by re-keying a song to rehearse it somewhere else. The audio, the
    /// chord chart and the transcription have all been moved by this much
    /// already; it is kept so a further change can be worked out from the
    /// original rather than piled on top of the last one.
    #[serde(default)]
    pub transpose_semitones: i32,
    /// A record the exported mix is matched to, if one has been chosen.
    ///
    /// Additive and optional, so a session written before mastering existed
    /// still opens: absent means the mix is exported as it was mixed.
    #[serde(default)]
    pub master_reference: Option<PathBuf>,
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
            transpose_semitones: 0,
            master_reference: None,
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
    /// Neural Amp Modeler capture used by this guitar track.
    #[serde(default)]
    pub nam_model: Option<PathBuf>,
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
            nam_model: None,
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

// One flag per insert module. Grouping them into an enum would mean modules
// could no longer be switched on independently, which is the whole point.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct TrackEffects {
    #[serde(default)]
    pub nam_enabled: bool,
    #[serde(default)]
    pub nam_input_db: f32,
    #[serde(default)]
    pub nam_output_db: f32,
    #[serde(default = "default_nam_gate_db")]
    pub nam_gate_db: f32,
    #[serde(default)]
    pub nam_tone_enabled: bool,
    #[serde(default = "default_tone_position")]
    pub nam_bass: f32,
    #[serde(default = "default_tone_position")]
    pub nam_middle: f32,
    #[serde(default = "default_tone_position")]
    pub nam_treble: f32,
    #[serde(default)]
    pub nam_normalize: bool,
    #[serde(default)]
    pub delay_enabled: bool,
    #[serde(default = "default_delay_time_ms")]
    pub delay_time_ms: f32,
    #[serde(default = "default_delay_feedback")]
    pub delay_feedback: f32,
    #[serde(default = "default_delay_mix")]
    pub delay_mix: f32,
    #[serde(default)]
    pub reverb_enabled: bool,
    #[serde(default = "default_reverb_size")]
    pub reverb_size: f32,
    #[serde(default = "default_reverb_damping")]
    pub reverb_damping: f32,
    #[serde(default = "default_reverb_mix")]
    pub reverb_mix: f32,
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
            nam_enabled: false,
            nam_input_db: 0.0,
            nam_output_db: 0.0,
            nam_gate_db: default_nam_gate_db(),
            nam_tone_enabled: false,
            nam_bass: default_tone_position(),
            nam_middle: default_tone_position(),
            nam_treble: default_tone_position(),
            nam_normalize: false,
            delay_enabled: false,
            delay_time_ms: default_delay_time_ms(),
            delay_feedback: default_delay_feedback(),
            delay_mix: default_delay_mix(),
            reverb_enabled: false,
            reverb_size: default_reverb_size(),
            reverb_damping: default_reverb_damping(),
            reverb_mix: default_reverb_mix(),
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
    /// Where in the source file this clip begins reading.
    ///
    /// Editing is non-destructive: trimming, splitting and copying only move
    /// this window over a file that is never rewritten. Absent in sessions
    /// written before editing existed, where every clip started at the top of
    /// its file — which is exactly what a default of zero means.
    #[serde(default)]
    pub source_start_frame: u64,
    /// The unshifted file this clip's audio was rendered from, when [`Self::path`]
    /// is a re-keyed render rather than the recording itself.
    ///
    /// Absent means `path` *is* the original, which is what every session
    /// written before re-keying existed holds. Keeping it means changing key
    /// again reads the original each time, so a song moved down four and then
    /// up two has been through one pitch shift, not two.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_path: Option<PathBuf>,
}

impl ProjectClip {
    /// How many frames the clip occupies on the timeline.
    #[must_use]
    pub fn length(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame)
    }

    /// The half-open range of the source file this clip reads.
    #[must_use]
    pub fn source_range(&self) -> std::ops::Range<u64> {
        self.source_start_frame..self.source_start_frame.saturating_add(self.length())
    }

    /// Splits at a timeline frame, returning the part after it.
    ///
    /// `None` when the frame is not strictly inside the clip: splitting at an
    /// edge would make an empty clip, which is not an edit anyone means.
    #[must_use]
    pub fn split_at(&mut self, frame: u64) -> Option<Self> {
        if frame <= self.start_frame || frame >= self.end_frame {
            return None;
        }
        let consumed = frame - self.start_frame;
        let right = Self {
            id: Uuid::new_v4(),
            name: self.name.clone(),
            path: self.path.clone(),
            start_frame: frame,
            end_frame: self.end_frame,
            source_start_frame: self.source_start_frame.saturating_add(consumed),
            // Both halves keep reading the same file, re-keyed or not.
            source_path: self.source_path.clone(),
        };
        self.end_frame = frame;
        Some(right)
    }

    /// Moves the start edge to `frame`, keeping the audio under it still.
    ///
    /// Trimming the front has to walk the source window forward by the same
    /// amount, or the take would slide against the timeline as the edge moved.
    /// Refuses to leave nothing, or to trim back past the head of the file.
    pub fn trim_start(&mut self, frame: u64) -> bool {
        if frame >= self.end_frame {
            return false;
        }
        if frame < self.start_frame {
            // Extending backwards: only as far as the source still has audio.
            let wanted = self.start_frame - frame;
            if wanted > self.source_start_frame {
                return false;
            }
            self.source_start_frame -= wanted;
        } else {
            self.source_start_frame += frame - self.start_frame;
        }
        self.start_frame = frame;
        true
    }

    /// Moves the end edge to `frame`. The source window follows the length.
    pub fn trim_end(&mut self, frame: u64) -> bool {
        if frame <= self.start_frame {
            return false;
        }
        self.end_frame = frame;
        true
    }
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

/// The amp's gate, off by default: an unasked-for gate that eats quiet notes
/// is worse than an unasked-for noise floor.
const fn default_nam_gate_db() -> f32 {
    -95.0
}
/// Tone controls sit at noon.
const fn default_tone_position() -> f32 {
    5.0
}
const fn default_delay_time_ms() -> f32 {
    350.0
}
const fn default_delay_feedback() -> f32 {
    0.35
}
const fn default_delay_mix() -> f32 {
    0.25
}
const fn default_reverb_size() -> f32 {
    0.6
}
const fn default_reverb_damping() -> f32 {
    0.4
}
const fn default_reverb_mix() -> f32 {
    0.2
}

#[cfg(test)]
mod tests {
    /// A clip covering frames 1000..2000 of the timeline, reading its source
    /// from the top.
    fn clip() -> ProjectClip {
        ProjectClip {
            id: Uuid::new_v4(),
            name: "Take".to_owned(),
            path: PathBuf::from("take.wav"),
            start_frame: 1_000,
            end_frame: 2_000,
            source_start_frame: 0,
            source_path: None,
        }
    }

    #[test]
    fn a_clip_written_before_editing_reads_from_the_top_of_its_file() {
        let older = r#"{"id":"00000000-0000-4000-8000-000000000000","name":"Take",
                        "path":"take.wav","start_frame":10,"end_frame":20}"#;
        let restored: ProjectClip = serde_json::from_str(older).expect("older clip");
        assert_eq!(restored.source_start_frame, 0);
        assert_eq!(restored.source_range(), 0..10);
    }

    #[test]
    fn splitting_hands_the_second_half_the_audio_that_follows() {
        let mut left = clip();
        let right = left.split_at(1_400).expect("splits inside");

        assert_eq!((left.start_frame, left.end_frame), (1_000, 1_400));
        assert_eq!((right.start_frame, right.end_frame), (1_400, 2_000));
        // The halves read consecutive windows of one untouched file: nothing
        // is duplicated and nothing is lost.
        assert_eq!(left.source_range(), 0..400);
        assert_eq!(right.source_range(), 400..1_000);
        assert_eq!(left.path, right.path);
        assert_ne!(left.id, right.id, "the halves are separate clips");
    }

    #[test]
    fn splitting_at_an_edge_is_refused() {
        assert!(clip().split_at(1_000).is_none(), "would leave nothing left");
        assert!(
            clip().split_at(2_000).is_none(),
            "would leave nothing right"
        );
        assert!(clip().split_at(500).is_none(), "outside the clip");
    }

    #[test]
    fn trimming_the_front_keeps_the_audio_where_it_was() {
        // The whole point: the take must not slide under the edge.
        let mut clip = clip();
        assert!(clip.trim_start(1_250));
        assert_eq!(clip.start_frame, 1_250);
        assert_eq!(
            clip.source_range(),
            250..1_000,
            "the window walked forward with the edge"
        );
    }

    #[test]
    fn a_trimmed_front_can_be_pulled_back_out_again() {
        let mut clip = clip();
        clip.trim_start(1_250);
        assert!(clip.trim_start(1_100), "there is still source behind it");
        assert_eq!(clip.source_range(), 100..1_000);
    }

    #[test]
    fn the_front_cannot_be_pulled_back_past_the_start_of_the_file() {
        let mut clip = clip();
        assert!(!clip.trim_start(900), "there is no audio before frame zero");
        assert_eq!(clip.start_frame, 1_000, "and nothing moved");
    }

    #[test]
    fn trimming_the_end_shortens_without_moving_the_window() {
        let mut clip = clip();
        assert!(clip.trim_end(1_600));
        assert_eq!(clip.end_frame, 1_600);
        assert_eq!(clip.source_range(), 0..600);
    }

    #[test]
    fn an_edge_cannot_be_dragged_through_the_other_one() {
        let mut clip = clip();
        assert!(!clip.trim_start(2_000));
        assert!(!clip.trim_end(1_000));
        assert_eq!((clip.start_frame, clip.end_frame), (1_000, 2_000));
    }

    #[test]
    fn splitting_a_clip_that_was_already_trimmed_stays_lined_up() {
        // The case that catches off-by-one offsets: edit, then edit again.
        let mut left = clip();
        left.trim_start(1_200);
        let right = left.split_at(1_500).expect("splits");
        assert_eq!(left.source_range(), 200..500);
        assert_eq!(right.source_range(), 500..1_000);
    }

    #[test]
    fn a_session_written_before_the_time_effects_existed_still_opens() {
        // Old sessions have no delay or reverb fields at all. They must load,
        // switched off, with the settings a new track would get rather than
        // with every control at zero.
        let json = r#"{
            "nam_enabled": false,
            "nam_input_db": 0.0,
            "nam_output_db": 0.0,
            "eq_enabled": true,
            "low_db": 3.0,
            "mid_db": 0.0,
            "high_db": -2.0,
            "compressor_enabled": false,
            "compressor_threshold_db": -18.0,
            "compressor_ratio": 4.0,
            "compressor_attack_ms": 10.0,
            "compressor_release_ms": 120.0,
            "compressor_makeup_db": 0.0,
            "gate_enabled": false,
            "gate_threshold_db": -45.0,
            "gate_release_ms": 120.0
        }"#;
        let effects: TrackEffects = serde_json::from_str(json).expect("an old session should load");
        assert!(effects.eq_enabled, "the settings it did have must survive");
        assert!(!effects.delay_enabled);
        assert!(!effects.reverb_enabled);
        let fresh = TrackEffects::default();
        assert!((effects.delay_time_ms - fresh.delay_time_ms).abs() < f32::EPSILON);
        assert!((effects.reverb_mix - fresh.reverb_mix).abs() < f32::EPSILON);
        assert!(
            effects.delay_time_ms > 0.0,
            "an unset time must not be zero"
        );
    }

    #[test]
    fn the_time_effects_survive_a_save_and_reload() {
        let effects = TrackEffects {
            delay_enabled: true,
            delay_time_ms: 420.0,
            reverb_enabled: true,
            reverb_size: 0.85,
            ..TrackEffects::default()
        };
        let json = serde_json::to_string(&effects).expect("serialise");
        let restored: TrackEffects = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(restored, effects);
    }
    use super::*;

    #[test]
    fn json_round_trip_preserves_session() {
        let mut project = ProjectDocument::default();
        project.tracks[0].clips.push(ProjectClip {
            source_start_frame: 0,
            source_path: None,
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
