#![allow(
    clippy::assigning_clones,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]

use crate::piano_roll::{self, PianoRollState};
use crate::theme;
use daw_audio_linux::{
    AudioRuntime, AudioRuntimeConfig, RuntimeSnapshot, RuntimeTransportState,
    enumerate_pipewire_devices,
};
use daw_core::{ChannelLayout, SamplePosition};
use daw_engine::ChannelStripParams;
use daw_midi::{MidiClip, TempoMap};
use daw_project::{
    ProjectClip, ProjectDocument, ProjectTrack, TrackEffects, TrackKind, load, save_atomic,
};
use daw_songimport::{
    CancelFlag, ImportProgress, ImportSource, IngestOptions, Ingested, MAX_TRANSPOSE_SEMITONES,
    ProjectSummary, Rekeyed,
};
use eframe::egui::{
    self, Align, Align2, Color32, FontId, Layout, Pos2, Rect, RichText, Sense, Stroke, StrokeKind,
    Vec2,
};
use std::path::PathBuf;
use std::process::Command;
use std::sync::mpsc::{Receiver, TryRecvError, channel};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

const HEADER_WIDTH: f32 = 265.0;
const TRACK_HEIGHT: f32 = 112.0;
const MIN_RECORDING_SPACE_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Clone)]
struct Clip {
    id: Uuid,
    name: String,
    path: PathBuf,
    start_frame: u64,
    end_frame: u64,
    /// Where in the source file this clip begins reading. Editing only moves
    /// this window; the file itself is never rewritten.
    source_start_frame: u64,
    /// Total frames in the source file. The waveform peaks span the whole
    /// file, so this is what says which part of them this clip is showing.
    source_frames: u64,
    /// The unshifted file this clip was rendered from, when `path` is a
    /// re-keyed render. Carried so changing key again starts from the original.
    source_path: Option<PathBuf>,
    color: Color32,
    waveform: Vec<f32>,
}

impl Clip {
    /// How many frames the clip occupies on the timeline.
    fn length(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame)
    }
}

/// What a drag on a clip will do, decided by where the gesture began on it.
///
/// Pro Tools calls this the Smart Tool: one pointer that becomes the tool the
/// place under it implies, instead of a palette to go and choose from. Near
/// either edge it trims that edge; anywhere else it moves the clip.
///
/// The zone is read from where the pointer was **pressed**, never from where it
/// is now. egui reports no drag until the pointer has travelled `max_click_dist`
/// — six pixels — from the press, so by the time there is a drag to act on the
/// pointer has already left a nine-pixel edge zone. Sampling it then sees the
/// body of the clip and turns every trim into a move.
#[derive(Clone, Copy, Debug, PartialEq)]
enum ClipZone {
    TrimStart,
    Body,
    TrimEnd,
}

/// The furthest the end edge of a clip can be dragged.
///
/// A clip is a window onto a file, and the window cannot show audio the file
/// does not have — dragging the end outwards past the last sample would leave
/// a clip longer than anything it can play, which reads as the take having
/// grown. A source of unknown length is left unbounded, which is the case for
/// the take still being recorded.
fn max_end_frame(start_frame: u64, source_start_frame: u64, source_frames: u64) -> u64 {
    if source_frames == 0 {
        return u64::MAX;
    }
    let available = source_frames.saturating_sub(source_start_frame);
    start_frame.saturating_add(available.max(1))
}

/// The height of the unit toggle sitting in the ruler.
fn ruler_height_inner() -> f32 {
    22.0
}

/// How wide the trim zones are, in pixels. Comfortably wider than egui's
/// six-pixel click threshold, so a press inside one is unambiguous.
const TRIM_ZONE_WIDTH: f32 = 10.0;

impl ClipZone {
    /// Which zone a position is in. Narrow clips give up their trim zones
    /// rather than leave no way to grab the middle.
    fn at(clip_rect: Rect, pointer_x: f32) -> Self {
        let edge = TRIM_ZONE_WIDTH.min(clip_rect.width() / 3.0);
        if pointer_x <= clip_rect.left() + edge {
            Self::TrimStart
        } else if pointer_x >= clip_rect.right() - edge {
            Self::TrimEnd
        } else {
            Self::Body
        }
    }

    fn cursor(self) -> egui::CursorIcon {
        match self {
            Self::Body => egui::CursorIcon::Grab,
            _ => egui::CursorIcon::ResizeHorizontal,
        }
    }

    fn describe(self) -> &'static str {
        match self {
            Self::TrimStart => "start",
            Self::TrimEnd => "end",
            Self::Body => "clip",
        }
    }
}

/// A trim in progress: which edge is moving, and the clip as it was before, so
/// the whole drag becomes one undo entry rather than one per frame.
#[derive(Clone)]
struct TrimDrag {
    track: usize,
    clip: usize,
    edge: ClipZone,
    before: Clip,
}

impl TrimDrag {
    /// Whether this drag belongs to a given clip.
    ///
    /// A gesture keeps the tool it started with: the pointer leaves the edge
    /// zone within a few pixels of moving, and a trim that became a move
    /// part-way through would be unusable.
    fn targets(&self, track: usize, clip: usize) -> bool {
        self.track == track && self.clip == clip
    }
}

#[allow(clippy::struct_excessive_bools)]
struct Track {
    id: Uuid,
    name: String,
    layout: ChannelLayout,
    input_left: usize,
    input_right: usize,
    armed: bool,
    monitoring: bool,
    muted: bool,
    solo: bool,
    gain_db: f32,
    pan: f32,
    effects: TrackEffects,
    nam_model: Option<PathBuf>,
    clips: Vec<Clip>,
    kind: TrackKind,
    midi_clips: Vec<MidiClip>,
    program: Option<u8>,
    drum_kit: bool,
}

#[derive(Clone, Copy)]
struct ClipLocation {
    track_id: Uuid,
    start_frame: u64,
}

/// A clip and the track it belongs to, kept whole so an edit can be undone by
/// putting it back exactly as it was.
#[derive(Clone)]
struct ClipSnapshot {
    track_id: Uuid,
    clip: Clip,
}

#[derive(Clone)]
enum EditCommand {
    MoveClip {
        clip_id: Uuid,
        before: ClipLocation,
        after: ClipLocation,
    },
    /// Any edit that changes which clips exist, described as what it took away
    /// and what it put in place.
    ///
    /// Split, trim, paste, duplicate and delete are all this shape — a split
    /// removes one clip and adds two, a trim removes one and adds one, a paste
    /// removes nothing. Undo puts `removed` back and takes `added` away; redo
    /// does the reverse. One mechanism instead of five, and no edit can be
    /// half-undone because the two lists are applied together.
    ReplaceClips {
        removed: Vec<ClipSnapshot>,
        added: Vec<ClipSnapshot>,
        label: &'static str,
    },
}

/// Messages from the song-import worker thread. Importing runs off the UI
/// thread because a full pipeline run is minutes long; the audio callback is
/// untouched either way, but a frozen window during an import is unacceptable.
enum ImportMessage {
    Progress(ImportProgress),
    Finished(Box<Ingested>),
    Failed(String),
}

/// State of the Import Song window.
#[allow(clippy::struct_excessive_bools)]
struct SongImportState {
    open: bool,
    url: String,
    include_drumkit: bool,
    align_to_bar: bool,
    import_midi: bool,
    /// Index into `TempoHint::PRESETS`. Roughly where to expect the tempo:
    /// fast music imports at half speed without it.
    tempo_hint: usize,
    /// Semitones to move the song by on the way in, for practising in a key
    /// that suits the singer. Negative is down.
    transpose: i32,
    /// Songs the pipeline has already processed; `None` until first loaded.
    catalog: Option<Vec<ProjectSummary>>,
    catalog_error: Option<String>,
    catalog_receiver: Option<Receiver<Result<Vec<ProjectSummary>, String>>>,
    job_receiver: Option<Receiver<ImportMessage>>,
    cancel: Option<CancelFlag>,
    stage: String,
    detail: String,
    /// 0.0–1.0 when known; `None` for steps with no measurable progress.
    fraction: Option<f32>,
    /// Decisions the importer made that the user may want to reverse.
    notes: Vec<String>,
    error: Option<String>,
}

impl Default for SongImportState {
    fn default() -> Self {
        Self {
            open: false,
            url: String::new(),
            include_drumkit: false,
            align_to_bar: true,
            import_midi: true,
            tempo_hint: 0,
            transpose: 0,
            catalog: None,
            catalog_error: None,
            catalog_receiver: None,
            job_receiver: None,
            cancel: None,
            stage: String::new(),
            detail: String::new(),
            fraction: None,
            notes: Vec::new(),
            error: None,
        }
    }
}

impl SongImportState {
    /// The hint the user picked, as the analysis crate wants it.
    fn selected_tempo_hint(&self) -> daw_analysis::TempoHint {
        let (_, centre) = daw_analysis::TempoHint::PRESETS
            .get(self.tempo_hint)
            .copied()
            .unwrap_or(daw_analysis::TempoHint::PRESETS[0]);
        daw_analysis::TempoHint::around(centre)
    }

    fn is_running(&self) -> bool {
        self.job_receiver.is_some()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
struct AudioPreferences {
    input_device: String,
    output_device: String,
    buffer_frames: u32,
    input_labels: [String; 4],
    /// How large the chord chart is drawn, as a multiple of its base size.
    /// A display preference rather than session data: it belongs to the person
    /// and the screen, not to the song.
    chord_lane_scale: f32,
}

impl Default for AudioPreferences {
    fn default() -> Self {
        Self {
            input_device: "Scarlett Solo".to_owned(),
            output_device: "Scarlett Solo".to_owned(),
            buffer_frames: 256,
            input_labels: std::array::from_fn(|index| format!("Input {}", index + 1)),
            chord_lane_scale: 1.0,
        }
    }
}

impl AudioPreferences {
    fn runtime_config(&self) -> AudioRuntimeConfig {
        AudioRuntimeConfig {
            input_name_contains: self.input_device.clone(),
            output_name_contains: self.output_device.clone(),
            buffer_frames: self.buffer_frames,
            ..AudioRuntimeConfig::default()
        }
    }
}

impl Track {
    fn new(index: usize, layout: ChannelLayout) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: format!("Audio {}", index + 1),
            layout,
            input_left: usize::from(layout == ChannelLayout::Mono),
            input_right: 1,
            armed: false,
            monitoring: false,
            muted: false,
            solo: false,
            gain_db: 0.0,
            pan: 0.0,
            effects: TrackEffects::default(),
            nam_model: None,
            clips: Vec::new(),
            kind: TrackKind::Audio,
            midi_clips: Vec::new(),
            program: None,
            drum_kit: false,
        }
    }
}

#[allow(clippy::struct_excessive_bools)]
pub struct RustDawApp {
    runtime: Option<AudioRuntime>,
    audio_error: Option<String>,
    tracks: Vec<Track>,
    session_name: String,
    selected_track: usize,
    tempo: u16,
    meter_numerator: u16,
    meter_denominator: u16,
    click_enabled: bool,
    click_level: f32,
    count_in_enabled: bool,
    recording_start: Option<u64>,
    recording_began: bool,
    current_recording_path: Option<PathBuf>,
    session_path: PathBuf,
    /// The record the exported mix is matched to. `None` exports the mix as it
    /// was mixed.
    master_reference: Option<PathBuf>,
    /// The detected chord chart, in timeline order, and the detected key.
    /// Carried here so saving the session keeps them: they cost real analysis
    /// to produce and were previously discarded on the first save.
    chords: Vec<daw_project::ChordEvent>,
    detected_key: Option<String>,
    /// How far the loaded session has been moved from the key it was imported
    /// in. Kept beside the document so saving does not lose it.
    session_transpose: i32,
    /// Whether the chord lane is shown above the tracks.
    chords_open: bool,
    /// A chord-lane size the wheel has changed but that is not on disk yet.
    /// Written once the wheel stops rather than on every notch.
    chord_scale_unsaved: bool,
    /// The copied clip, kept whole so pasting reproduces its trimmed window.
    clipboard: Option<Clip>,
    /// The loop range in seconds, if one is marked.
    loop_range: Option<(f64, f64)>,
    /// Where a loop drag on the ruler began.
    loop_drag_anchor: Option<f64>,
    /// Whether the ruler counts bars rather than minutes and seconds.
    ruler_shows_bars: bool,
    /// An edge being dragged, if one is.
    trimming: Option<TrimDrag>,
    selected_clip: Option<(usize, usize)>,
    dragged_clip: Option<(usize, usize, u64, Vec2)>,
    dirty: bool,
    confirm_new_session: bool,
    /// The re-key window, open while a key is being chosen.
    transpose_open: bool,
    /// The key being auditioned in that window, in semitones from the original.
    transpose_wanted: i32,
    /// Receives the re-keyed document from the worker thread.
    transpose_receiver: Option<Receiver<Result<(ProjectDocument, Rekeyed), String>>>,
    /// The last export's outcome, until the person exporting dismisses it.
    export_report: Option<ExportReport>,
    status_message: String,
    pixels_per_second: f32,
    /// When set, the timeline scrolls to keep the playhead in view during
    /// playback. Off lets the user scroll freely while the transport runs.
    follow_playhead: bool,
    /// Real-time playback speed (varispeed). 1.0 is normal; changing it while
    /// playing speeds up or slows down the song to audition a different tempo.
    playback_speed: f32,
    /// The timeline's current horizontal scroll offset in pixels, mirrored here
    /// so the ruler above the tracks can be drawn scrolled in step with them.
    timeline_scroll_x: f32,
    /// Recent tap-tempo button presses, for estimating BPM by tapping.
    tap_times: Vec<Instant>,
    free_disk_bytes: Option<u64>,
    last_disk_check: Instant,
    audio_preferences: AudioPreferences,
    available_inputs: Vec<String>,
    available_outputs: Vec<String>,
    audio_settings_open: bool,
    test_input_channel: Option<usize>,
    inserts_open: bool,
    pending_open_session: Option<PathBuf>,
    confirm_open_session: bool,
    session_needs_save_as: bool,
    mixer_open: bool,
    playback_synced: bool,
    pending_delete_track: Option<usize>,
    undo_stack: Vec<EditCommand>,
    redo_stack: Vec<EditCommand>,
    song_import: SongImportState,
    tempo_map: TempoMap,
    piano_roll: PianoRollState,
    tuner: crate::tuner::TunerState,
    /// When the tuner last ran, so the needle's smoothing is in real time
    /// rather than per frame.
    tuner_ticked: Instant,
    /// Amp captures found on disk. Scanned once and on request rather than per
    /// frame: the FX window repaints continuously and the scan touches disk.
    amp_library: Vec<daw_nam::AmpModel>,
    /// An in-flight TONE3000 download. The flow waits on the user's browser,
    /// so it runs on its own thread and reports back here.
    amp_fetch: Option<Receiver<Result<daw_tone3000::FetchedModel, String>>>,
}

impl RustDawApp {
    pub fn new() -> Self {
        let mut audio_preferences = load_audio_preferences().unwrap_or_default();
        if is_output_monitor_name(&audio_preferences.input_device) {
            audio_preferences.input_device = AudioPreferences::default().input_device;
        }
        let runtime = AudioRuntime::open(&audio_preferences.runtime_config());
        let (runtime, audio_error) = match runtime {
            Ok(runtime) => (Some(runtime), None),
            Err(error) => (None, Some(error.to_string())),
        };
        let session_path = default_session_path();
        let document = load(&session_path).unwrap_or_default();
        let session_name = document.name.clone();
        let document_tempo_map = document.tempo_map();
        let tracks = document
            .tracks
            .into_iter()
            .map(track_from_project)
            .collect::<Vec<_>>();
        let (available_inputs, available_outputs) = available_audio_devices();
        let mut app = Self {
            runtime,
            audio_error,
            tracks,
            session_name,
            selected_track: 0,
            tempo: document.tempo,
            meter_numerator: document.meter_numerator,
            meter_denominator: document.meter_denominator,
            click_enabled: document.click_enabled,
            click_level: 0.35,
            count_in_enabled: true,
            recording_start: None,
            recording_began: false,
            current_recording_path: None,
            session_path,
            master_reference: document.master_reference.clone(),
            chords: document.chords.clone(),
            detected_key: document.key.clone(),
            session_transpose: document.transpose_semitones,
            chords_open: true,
            chord_scale_unsaved: false,
            clipboard: None,
            loop_range: None,
            loop_drag_anchor: None,
            ruler_shows_bars: false,
            trimming: None,
            selected_clip: None,
            dragged_clip: None,
            dirty: false,
            confirm_new_session: false,
            transpose_open: false,
            transpose_wanted: 0,
            transpose_receiver: None,
            export_report: None,
            status_message: "Ready".to_owned(),
            pixels_per_second: 84.0,
            follow_playhead: true,
            playback_speed: 1.0,
            timeline_scroll_x: 0.0,
            tap_times: Vec::new(),
            free_disk_bytes: disk_free_bytes(&recording_directory()).ok(),
            last_disk_check: Instant::now(),
            audio_preferences,
            available_inputs,
            available_outputs,
            audio_settings_open: false,
            test_input_channel: None,
            inserts_open: false,
            pending_open_session: None,
            confirm_open_session: false,
            session_needs_save_as: false,
            mixer_open: false,
            playback_synced: false,
            pending_delete_track: None,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            amp_library: daw_nam::discover(),
            amp_fetch: None,
            song_import: SongImportState::default(),
            tempo_map: document_tempo_map,
            piano_roll: PianoRollState::default(),
            tuner: crate::tuner::TunerState::default(),
            tuner_ticked: Instant::now(),
        };
        if app.tracks.is_empty() {
            app.tracks.push(Track::new(0, ChannelLayout::Mono));
        }
        if let Some(runtime) = &app.runtime {
            runtime.set_tempo(app.tempo);
            runtime.set_meter(app.meter_numerator, app.meter_denominator);
            runtime.set_speed(app.playback_speed);
        }
        if let Err(error) = app.sync_playback() {
            app.status_message = format!("Media preload failed: {error}");
        }
        app
    }

    fn apply_audio_settings(&mut self) {
        if matches!(
            self.snapshot().transport,
            RuntimeTransportState::Recording | RuntimeTransportState::CountIn
        ) {
            self.status_message = "Stop recording before changing audio settings".to_owned();
            return;
        }
        if let Some(runtime) = self.runtime.take() {
            runtime.stop();
            drop(runtime);
        }
        match AudioRuntime::open(&self.audio_preferences.runtime_config()) {
            Ok(runtime) => {
                runtime.set_tempo(self.tempo);
                runtime.set_meter(self.meter_numerator, self.meter_denominator);
                runtime.set_speed(self.playback_speed);
                runtime.set_click(self.click_enabled, self.click_level);
                self.runtime = Some(runtime);
                self.playback_synced = false;
                if let Err(error) = self.sync_playback() {
                    self.status_message = format!("Audio restarted; media preload failed: {error}");
                    return;
                }
                self.audio_error = None;
                self.status_message = "Audio engine restarted successfully".to_owned();
                if let Err(error) = save_audio_preferences(&self.audio_preferences) {
                    self.status_message =
                        format!("Audio works, but settings were not saved: {error}");
                }
            }
            Err(error) => {
                self.audio_error = Some(error.to_string());
                self.status_message = format!("Audio settings failed: {error}");
            }
        }
    }

    fn audio_settings(&mut self, context: &egui::Context, snapshot: &RuntimeSnapshot) {
        if !self.audio_settings_open {
            return;
        }
        let mut open = self.audio_settings_open;
        let mut apply = false;
        let mut refresh = false;
        egui::Window::new("Audio Settings")
            .open(&mut open)
            .default_width(620.0)
            .resizable(false)
            .show(context, |ui| {
                ui.heading("Audio Engine");
                ui.label(
                    RichText::new(
                        "Choose PipeWire devices, then identify the Scarlett's physical inputs using the live meters.",
                    )
                    .color(theme::MUTED),
                );
                ui.add_space(8.0);
                egui::Grid::new("audio_devices_grid")
                    .num_columns(2)
                    .spacing([14.0, 8.0])
                    .show(ui, |ui| {
                        ui.label("Input device");
                        egui::ComboBox::from_id_salt("audio_input_device")
                            .selected_text(&self.audio_preferences.input_device)
                            .width(390.0)
                            .show_ui(ui, |ui| {
                                for name in &self.available_inputs {
                                    ui.selectable_value(
                                        &mut self.audio_preferences.input_device,
                                        name.clone(),
                                        name,
                                    );
                                }
                            });
                        ui.end_row();
                        ui.label("Output device");
                        egui::ComboBox::from_id_salt("audio_output_device")
                            .selected_text(&self.audio_preferences.output_device)
                            .width(390.0)
                            .show_ui(ui, |ui| {
                                for name in &self.available_outputs {
                                    ui.selectable_value(
                                        &mut self.audio_preferences.output_device,
                                        name.clone(),
                                        name,
                                    );
                                }
                            });
                        ui.end_row();
                        ui.label("Buffer size");
                        egui::ComboBox::from_id_salt("audio_buffer")
                            .selected_text(format!("{} frames", self.audio_preferences.buffer_frames))
                            .show_ui(ui, |ui| {
                                for frames in [128, 256, 512, 1024] {
                                    ui.selectable_value(
                                        &mut self.audio_preferences.buffer_frames,
                                        frames,
                                        format!("{frames} frames"),
                                    );
                                }
                            });
                        ui.end_row();
                        ui.label("Sample rate");
                        ui.label("48,000 Hz (project/device)");
                        ui.end_row();
                    });
                ui.horizontal(|ui| {
                    if ui.button("Refresh devices").clicked() {
                        refresh = true;
                    }
                    if ui.button("Apply and restart engine").clicked() {
                        apply = true;
                    }
                    if ui.button("Test outputs 1–2").clicked() {
                        if let Some(runtime) = &self.runtime {
                            runtime.trigger_output_test();
                            self.status_message = "Playing a one-second output test tone".to_owned();
                        }
                    }
                });
                ui.separator();
                ui.heading("Input identification");
                ui.label(
                    "Play or tap the connected source. Rename the channel whose meter responds, then use that name on a mono track.",
                );
                let channel_count = self
                    .runtime
                    .as_ref()
                    .map_or(4, |runtime| usize::from(runtime.input_channels()).min(4));
                for channel in 0..channel_count {
                    ui.horizontal(|ui| {
                        ui.label(format!("CH {}", channel + 1));
                        ui.add(
                            egui::TextEdit::singleline(
                                &mut self.audio_preferences.input_labels[channel],
                            )
                            .desired_width(170.0),
                        );
                        ui.allocate_ui_with_layout(
                            Vec2::new(220.0, 18.0),
                            Layout::left_to_right(Align::Center),
                            |ui| meter(ui, snapshot.input_peaks[channel]),
                        );
                        let listening = self.test_input_channel == Some(channel);
                        if ui.selectable_label(listening, "Listen").clicked() {
                            self.test_input_channel = (!listening).then_some(channel);
                            if let Some(runtime) = &self.runtime {
                                runtime.set_monitoring(!listening, channel, channel);
                            }
                        }
                    });
                }
                ui.label(
                    RichText::new(
                        "Listening uses software monitoring. Turn off the Scarlett Direct Monitor first to avoid hearing a doubled signal.",
                    )
                    .small()
                    .color(theme::YELLOW),
                );
                ui.separator();
                if let Some(runtime) = &self.runtime {
                    ui.label(format!(
                        "Active: {} inputs / {} outputs · {} Hz · {} frames",
                        runtime.input_channels(),
                        runtime.output_channels(),
                        runtime.sample_rate().get(),
                        runtime.buffer_frames()
                    ));
                    ui.label(RichText::new(runtime.input_name()).small().color(theme::MUTED));
                    ui.label(RichText::new(runtime.output_name()).small().color(theme::MUTED));
                    // Which instruments the MIDI tracks are playing, so a
                    // missing SoundFont is visible rather than just quieter.
                    let instruments = runtime.soundfont_name().map_or_else(
                        || "Instruments: built-in synth".to_owned(),
                        |name| format!("Instruments: {name}"),
                    );
                    ui.label(RichText::new(instruments).small().color(theme::MUTED));
                } else if let Some(error) = &self.audio_error {
                    ui.colored_label(theme::RED, format!("Offline: {error}"));
                }
            });
        self.audio_settings_open = open;
        if refresh {
            (self.available_inputs, self.available_outputs) = available_audio_devices();
        }
        if apply {
            self.test_input_channel = None;
            self.apply_audio_settings();
        }
        if !open && self.test_input_channel.take().is_some() {
            if let Some(runtime) = &self.runtime {
                runtime.set_monitoring(false, 0, 0);
            }
        }
    }

    /// Registers a tap-tempo press: sets the BPM from the tap spacing and, while
    /// playing, lines the click's downbeat up with the moment of the tap.
    fn tap_tempo(&mut self) {
        let now = Instant::now();
        if self
            .tap_times
            .last()
            .is_some_and(|last| now.duration_since(*last) > Duration::from_secs(2))
        {
            // A long gap means a fresh count-in, not a continuation.
            self.tap_times.clear();
        }
        self.tap_times.push(now);
        if self.tap_times.len() > 8 {
            self.tap_times.remove(0);
        }

        if self.tap_times.len() >= 2 {
            let mut intervals: Vec<f64> = self
                .tap_times
                .windows(2)
                .map(|pair| pair[1].duration_since(pair[0]).as_secs_f64())
                .collect();
            intervals.sort_by(|left, right| {
                left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
            });
            let median = intervals[intervals.len() / 2];
            if median > 0.0 {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let bpm = (60.0 / median).round().clamp(20.0, 300.0) as u16;
                self.tempo = bpm;
                if let Some(runtime) = &self.runtime {
                    runtime.set_tempo(bpm);
                }
                self.dirty = true;
            }
        }

        // Phase-align the click to this tap so bar one falls where you tapped.
        if let Some(runtime) = &self.runtime {
            let snapshot = self.snapshot();
            let running = matches!(
                snapshot.transport,
                RuntimeTransportState::Playing
                    | RuntimeTransportState::Recording
                    | RuntimeTransportState::CountIn
            );
            if running {
                let rate = f64::from(runtime.sample_rate().get());
                let bar_frames = rate * 60.0 * 4.0 * f64::from(self.meter_numerator.max(1))
                    / (f64::from(self.tempo.max(1)) * f64::from(self.meter_denominator.max(1)));
                if bar_frames > 0.0 {
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        clippy::cast_precision_loss
                    )]
                    let offset = (snapshot.position_frames as f64 % bar_frames) as u64;
                    runtime.set_click_offset(offset);
                }
            }
        }
    }

    fn snapshot(&self) -> RuntimeSnapshot {
        self.runtime
            .as_ref()
            .map_or_else(RuntimeSnapshot::default, AudioRuntime::snapshot)
    }

    fn toggle_play(&mut self) {
        let playing = self
            .runtime
            .as_ref()
            .is_some_and(|runtime| runtime.snapshot().transport != RuntimeTransportState::Stopped);
        if playing {
            if let Some(runtime) = &self.runtime {
                runtime.stop();
            }
            self.finish_recording_clip();
            self.status_message = "Stopped".to_owned();
        } else {
            if !self.playback_synced {
                if let Err(error) = self.sync_playback() {
                    self.status_message = error.to_string();
                    return;
                }
            }
            if let Some(runtime) = &self.runtime {
                runtime.play();
                self.status_message = "Playing".to_owned();
            }
        }
    }

    fn toggle_record(&mut self) {
        let snapshot = self.snapshot();
        if matches!(
            snapshot.transport,
            RuntimeTransportState::Recording | RuntimeTransportState::CountIn
        ) {
            if let Some(runtime) = &self.runtime {
                runtime.stop();
            }
            self.finish_recording_clip();
            self.status_message = "Recording saved".to_owned();
            return;
        }

        let Some(track_index) = self.tracks.iter().position(|track| track.armed) else {
            self.status_message = "Arm a track before recording".to_owned();
            return;
        };
        let path = match recording_path(track_index) {
            Ok(path) => path,
            Err(error) => {
                self.status_message = error.to_string();
                return;
            }
        };
        match disk_free_bytes(path.parent().unwrap_or_else(|| std::path::Path::new("."))) {
            Ok(bytes) if bytes < MIN_RECORDING_SPACE_BYTES => {
                self.free_disk_bytes = Some(bytes);
                self.status_message = format!(
                    "Recording blocked: only {} free (256 MiB required)",
                    format_bytes(bytes)
                );
                return;
            }
            Ok(bytes) => self.free_disk_bytes = Some(bytes),
            Err(error) => {
                self.status_message = format!("Recording path preflight failed: {error}");
                return;
            }
        }
        let track = &self.tracks[track_index];
        if let Some(runtime) = &self.runtime {
            match runtime.start_recording(
                path.clone(),
                track.layout,
                track.input_left,
                track.input_right,
                u16::from(self.count_in_enabled),
            ) {
                Ok(record_start) => {
                    self.selected_track = track_index;
                    self.recording_start = Some(record_start);
                    self.recording_began = !self.count_in_enabled;
                    self.current_recording_path = Some(path);
                    self.status_message = if self.count_in_enabled {
                        "Count-in: 4 beats…".to_owned()
                    } else {
                        "Recording…".to_owned()
                    };
                }
                Err(error) => self.status_message = error.to_string(),
            }
        }
    }

    fn finish_recording_clip(&mut self) {
        let Some(start_frame) = self.recording_start.take() else {
            return;
        };
        if !self.recording_began {
            if let Some(path) = self.current_recording_path.take() {
                let _ = std::fs::remove_file(path);
            }
            self.status_message = "Count-in cancelled".to_owned();
            return;
        }
        self.recording_began = false;
        let end_frame = self.snapshot().position_frames.max(start_frame + 1);
        let path = self.current_recording_path.take().unwrap_or_default();
        let name = path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("Take")
            .to_owned();
        let mut added = false;
        let (waveform, source_frames) = analyze_waveform(&path);
        let clip_id = Uuid::new_v4();
        if let Some(track) = self.tracks.get_mut(self.selected_track) {
            track.clips.push(Clip {
                id: clip_id,
                name,
                path,
                start_frame,
                end_frame,
                source_start_frame: 0,
                source_path: None,
                source_frames,
                color: theme::BLUE_DARK,
                waveform,
            });
            self.dirty = true;
            added = true;
        }
        if added {
            self.save_session();
            if let Err(error) = self.sync_playback() {
                self.status_message = format!("Take saved; playback preload failed: {error}");
            }
        }
    }

    fn sync_playback(&mut self) -> anyhow::Result<()> {
        let Some(runtime) = &self.runtime else {
            self.playback_synced = false;
            return Ok(());
        };
        runtime.clear_playback()?;
        let sample_rate = runtime.sample_rate().get();
        let any_solo = self.tracks.iter().any(|track| track.solo);
        for (track_id, track) in self.tracks.iter().enumerate() {
            let audible = !track.muted && (!any_solo || track.solo);
            let gain = 10.0_f32.powf(track.gain_db / 20.0);
            runtime.set_track_effects(track_id, channel_strip_params(track.effects))?;
            runtime.set_track_nam_model(
                track_id,
                track
                    .nam_model
                    .as_deref()
                    .filter(|_| track.effects.nam_enabled),
                channel_strip_params(track.effects),
            )?;
            for clip in &track.clips {
                runtime.add_clip_playback_file(
                    &clip.path,
                    clip.start_frame,
                    clip.source_start_frame,
                    clip.length(),
                    gain,
                    channel_strip_params(track.effects),
                    track.pan,
                    track_id,
                    clip.id.as_u128(),
                    audible,
                )?;
            }
            if !track.midi_clips.is_empty() {
                // Ticks become frames here, on the control thread, so the
                // audio callback never has to consult the tempo map.
                let mut notes = Vec::new();
                for clip in &track.midi_clips {
                    notes.extend(clip.schedule(&self.tempo_map, sample_rate));
                }
                notes.sort_by_key(|note| note.start_frame);
                runtime.add_midi_track(
                    track_id,
                    notes,
                    gain,
                    track.pan,
                    audible,
                    track.program.unwrap_or(0),
                    track.drum_kit,
                )?;
            }
        }
        self.playback_synced = true;
        Ok(())
    }

    fn sync_moved_clip(&self, track_index: usize, clip_index: usize) -> anyhow::Result<()> {
        let Some(runtime) = &self.runtime else {
            return Ok(());
        };
        let track = &self.tracks[track_index];
        let clip = &track.clips[clip_index];
        let any_solo = self.tracks.iter().any(|item| item.solo);
        let audible = !track.muted && (!any_solo || track.solo);
        runtime.move_playback_clip(
            clip.id.as_u128(),
            clip.start_frame,
            track_index,
            10.0_f32.powf(track.gain_db / 20.0),
            track.pan,
            audible,
        )
    }

    fn remember_edit(&mut self, command: EditCommand) {
        const HISTORY_LIMIT: usize = 256;
        if self.undo_stack.len() == HISTORY_LIMIT {
            self.undo_stack.remove(0);
        }
        self.undo_stack.push(command);
        self.redo_stack.clear();
    }

    fn apply_clip_location(
        &mut self,
        clip_id: Uuid,
        destination: ClipLocation,
    ) -> anyhow::Result<()> {
        let source_track = self
            .tracks
            .iter()
            .position(|track| track.clips.iter().any(|clip| clip.id == clip_id))
            .ok_or_else(|| anyhow::anyhow!("clip no longer exists"))?;
        let source_clip = self.tracks[source_track]
            .clips
            .iter()
            .position(|clip| clip.id == clip_id)
            .ok_or_else(|| anyhow::anyhow!("clip no longer exists"))?;
        let target_track = self
            .tracks
            .iter()
            .position(|track| track.id == destination.track_id)
            .ok_or_else(|| anyhow::anyhow!("destination track no longer exists"))?;
        let mut clip = self.tracks[source_track].clips.remove(source_clip);
        let duration = clip.end_frame.saturating_sub(clip.start_frame);
        clip.start_frame = destination.start_frame;
        clip.end_frame = destination.start_frame.saturating_add(duration);
        let target_clip = self.tracks[target_track].clips.len();
        self.tracks[target_track].clips.push(clip);
        self.selected_track = target_track;
        self.selected_clip = Some((target_track, target_clip));
        self.dirty = true;
        self.sync_moved_clip(target_track, target_clip)
    }

    /// Applies a structural edit: takes `remove` away and puts `insert` back.
    ///
    /// Both halves happen together, so an edit is never left half-applied even
    /// if one of its clips has since gone.
    fn apply_replacement(&mut self, remove: &[ClipSnapshot], insert: &[ClipSnapshot]) {
        for snapshot in remove {
            for track in &mut self.tracks {
                track.clips.retain(|clip| clip.id != snapshot.clip.id);
            }
        }
        for snapshot in insert {
            if let Some(track) = self
                .tracks
                .iter_mut()
                .find(|track| track.id == snapshot.track_id)
            {
                track.clips.push(snapshot.clip.clone());
            }
        }
        self.selected_clip = None;
        self.dirty = true;
        self.playback_synced = false;
        if let Err(error) = self.sync_playback() {
            self.status_message = format!("Edit applied; playback preload failed: {error}");
        }
    }

    fn undo(&mut self) {
        let Some(command) = self.undo_stack.pop() else {
            self.status_message = "Nothing to undo".to_owned();
            return;
        };
        match &command {
            EditCommand::MoveClip {
                clip_id, before, ..
            } => match self.apply_clip_location(*clip_id, *before) {
                Ok(()) => {
                    self.status_message = "Undid clip move".to_owned();
                    self.redo_stack.push(command);
                    self.save_session();
                }
                Err(error) => self.status_message = format!("Could not undo: {error}"),
            },
            EditCommand::ReplaceClips {
                removed,
                added,
                label,
            } => {
                self.apply_replacement(added, removed);
                self.status_message = format!("Undid {label}");
                self.save_session();
                self.redo_stack.push(command);
            }
        }
    }

    fn redo(&mut self) {
        let Some(command) = self.redo_stack.pop() else {
            self.status_message = "Nothing to redo".to_owned();
            return;
        };
        match &command {
            EditCommand::MoveClip { clip_id, after, .. } => {
                match self.apply_clip_location(*clip_id, *after) {
                    Ok(()) => {
                        self.status_message = "Redid clip move".to_owned();
                        self.undo_stack.push(command);
                        self.save_session();
                    }
                    Err(error) => self.status_message = format!("Could not redo: {error}"),
                }
            }
            EditCommand::ReplaceClips {
                removed,
                added,
                label,
            } => {
                self.apply_replacement(removed, added);
                self.status_message = format!("Redid {label}");
                self.save_session();
                self.undo_stack.push(command);
            }
        }
    }

    fn sync_track_audibility(&mut self) {
        let Some(runtime) = &self.runtime else {
            return;
        };
        let any_solo = self.tracks.iter().any(|track| track.solo);
        for (track_id, track) in self.tracks.iter().enumerate() {
            let audible = !track.muted && (!any_solo || track.solo);
            if let Err(error) = runtime.set_track_audible(track_id, audible) {
                self.status_message = format!("Could not update mixer state: {error}");
                return;
            }
        }
    }

    fn project_document(&self) -> ProjectDocument {
        ProjectDocument {
            name: self.session_name.clone(),
            sample_rate: self
                .runtime
                .as_ref()
                .map_or(48_000, |runtime| runtime.sample_rate().get()),
            tempo: self.tempo,
            meter_numerator: self.meter_numerator,
            meter_denominator: self.meter_denominator,
            click_enabled: self.click_enabled,
            tempo_map: Some(self.tempo_map.clone()),
            master_reference: self.master_reference.clone(),
            chords: self.chords.clone(),
            key: self.detected_key.clone(),
            transpose_semitones: self.session_transpose,
            tracks: self.tracks.iter().map(track_to_project).collect(),
            ..ProjectDocument::default()
        }
    }

    fn save_session(&mut self) {
        if self.session_needs_save_as {
            self.save_session_as();
            return;
        }
        match save_atomic(&self.project_document(), &self.session_path) {
            Ok(()) => {
                self.dirty = false;
                self.status_message = format!("Saved {}", self.session_path.display());
            }
            Err(error) => self.status_message = error.to_string(),
        }
    }

    fn save_session_as(&mut self) {
        let suggested = format!("{}.rustdaw", sanitize_file_name(&self.session_name));
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("RustDAW Session", &["rustdaw"])
            .set_file_name(&suggested)
            .save_file()
        {
            self.session_path = ensure_extension(path, "rustdaw");
            self.session_needs_save_as = false;
            self.save_session();
        }
    }

    fn choose_session_to_open(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("RustDAW Session", &["rustdaw", "json"])
            .pick_file()
        else {
            return;
        };
        if self.dirty {
            self.pending_open_session = Some(path);
            self.confirm_open_session = true;
        } else {
            self.open_session(path);
        }
    }

    fn open_session(&mut self, path: PathBuf) {
        match load(&path) {
            Ok(document) => {
                if let Some(runtime) = &self.runtime {
                    runtime.stop();
                    let _ = runtime.clear_playback();
                    runtime.seek_to_start();
                    runtime.set_tempo(document.tempo);
                    runtime.set_meter(document.meter_numerator, document.meter_denominator);
                    runtime.set_click(document.click_enabled, self.click_level);
                    runtime.set_click_offset(0);
                }
                self.tap_times.clear();
                self.session_name = document.name.clone();
                self.tempo_map = document.tempo_map();
                self.tempo = document.tempo;
                self.meter_numerator = document.meter_numerator;
                self.meter_denominator = document.meter_denominator;
                self.click_enabled = document.click_enabled;
                self.master_reference = document.master_reference.clone();
                self.chords = document.chords.clone();
                self.detected_key = document.key.clone();
                self.session_transpose = document.transpose_semitones;
                self.tracks = document
                    .tracks
                    .into_iter()
                    .map(track_from_project)
                    .collect();
                if self.tracks.is_empty() {
                    self.tracks.push(Track::new(0, ChannelLayout::Mono));
                }
                self.session_path = path;
                self.session_needs_save_as = false;
                self.selected_track = 0;
                self.selected_clip = None;
                self.recording_start = None;
                self.current_recording_path = None;
                self.undo_stack.clear();
                self.redo_stack.clear();
                self.dirty = false;
                self.status_message = format!("Opened {}", self.session_path.display());
                self.playback_synced = false;
                if let Err(error) = self.sync_playback() {
                    self.status_message = format!("Session opened; media preload failed: {error}");
                }
            }
            Err(error) => self.status_message = format!("Could not open session: {error}"),
        }
    }

    fn choose_audio_to_import(&mut self) {
        if let Some(paths) = rfd::FileDialog::new()
            .add_filter(
                "Audio",
                &[
                    "wav", "wave", "mp3", "flac", "m4a", "aac", "ogg", "opus", "aiff", "aif", "wma",
                ],
            )
            .pick_files()
        {
            self.import_audio_files(paths);
        }
    }

    fn import_audio_files(&mut self, paths: Vec<PathBuf>) {
        let start_frame = self.snapshot().position_frames;
        let expected_rate = self
            .runtime
            .as_ref()
            .map_or(48_000, |runtime| runtime.sample_rate().get());
        let mut imported = 0_usize;
        let mut converted = 0_usize;
        let mut errors = Vec::new();
        for path in paths {
            if !path.is_file() {
                continue;
            }
            // A session-rate mono/stereo WAV plays as it lies and is left where
            // the user keeps it. Everything else — an MP3, a 44.1 kHz WAV, a
            // FLAC — is converted into the session's own Imports folder.
            let prepared = match inspect_import_audio(&path, expected_rate) {
                Ok((layout, frames)) => Ok((path.clone(), layout, frames)),
                Err(_) => convert_import_audio(&path, expected_rate).inspect(|_| converted += 1),
            };
            match prepared {
                Ok((path, layout, frames)) => {
                    let name = path
                        .file_stem()
                        .and_then(|name| name.to_str())
                        .unwrap_or("Imported Audio")
                        .to_owned();
                    let mut track = Track::new(self.tracks.len(), layout);
                    track.name.clone_from(&name);
                    let clip_id = Uuid::new_v4();
                    let (waveform, source_frames) = analyze_waveform(&path);
                    track.clips.push(Clip {
                        id: clip_id,
                        name,
                        waveform,
                        path,
                        start_frame,
                        end_frame: start_frame.saturating_add(frames),
                        source_start_frame: 0,
                        source_path: None,
                        source_frames,
                        color: theme::BLUE_DARK,
                    });
                    self.tracks.push(track);
                    imported += 1;
                }
                Err(error) => errors.push(error.to_string()),
            }
        }
        if imported > 0 {
            self.selected_track = self.tracks.len().saturating_sub(1);
            self.dirty = true;
        }
        let note = match converted {
            0 => String::new(),
            count => format!(" ({count} converted to {expected_rate} Hz)"),
        };
        self.status_message = if errors.is_empty() {
            format!("Imported {imported} audio file(s) at the playhead{note}")
        } else {
            format!(
                "Imported {imported}{note}; {} failed: {}",
                errors.len(),
                errors.join(" · ")
            )
        };
        if imported > 0 {
            self.playback_synced = false;
            if let Err(error) = self.sync_playback() {
                self.status_message = format!("Audio imported; playback preload failed: {error}");
            }
        }
    }

    fn open_song_import(&mut self, context: &egui::Context) {
        self.song_import.open = true;
        self.song_import.error = None;
        if self.song_import.catalog.is_none() {
            self.refresh_song_catalog(context);
        }
    }

    /// Loads the list of already-processed songs on a background thread. The
    /// worker may need starting first, which can take a minute, so this must
    /// never happen on the UI thread.
    fn refresh_song_catalog(&mut self, context: &egui::Context) {
        if self.song_import.catalog_receiver.is_some() {
            return;
        }
        self.song_import.catalog_error = None;
        let (sender, receiver) = channel();
        let repaint = context.clone();
        let spawned = std::thread::Builder::new()
            .name("song-catalog".to_owned())
            .spawn(move || {
                let result =
                    daw_songimport::list_ready_projects().map_err(|error| format!("{error:#}"));
                let _ = sender.send(result);
                repaint.request_repaint();
            });
        match spawned {
            Ok(_) => self.song_import.catalog_receiver = Some(receiver),
            Err(error) => self.song_import.catalog_error = Some(error.to_string()),
        }
    }

    fn start_song_import(&mut self, context: &egui::Context, source: ImportSource) {
        if self.song_import.is_running() {
            return;
        }
        let target_rate = self
            .runtime
            .as_ref()
            .map_or(48_000, |runtime| runtime.sample_rate().get());
        let options = IngestOptions {
            destination_root: daw_songimport::default_song_root(),
            include_drumkit: self.song_import.include_drumkit,
            align_to_bar: self.song_import.align_to_bar,
            skip_silent: true,
            detect_tempo: true,
            tempo_hint: self.song_import.selected_tempo_hint(),
            import_midi: self.song_import.import_midi,
            detect_chords: true,
            transpose_semitones: self.song_import.transpose,
        };
        let cancel = CancelFlag::new();
        let thread_cancel = cancel.clone();
        let (sender, receiver) = channel();
        let repaint = context.clone();
        let spawned = std::thread::Builder::new()
            .name("song-import".to_owned())
            .spawn(move || {
                let progress_sender = sender.clone();
                let progress_repaint = repaint.clone();
                let outcome = daw_songimport::run_import(
                    &source,
                    &options,
                    target_rate,
                    &thread_cancel,
                    |progress| {
                        let _ = progress_sender.send(ImportMessage::Progress(progress));
                        progress_repaint.request_repaint();
                    },
                );
                let message = match outcome {
                    Ok(ingested) => ImportMessage::Finished(Box::new(ingested)),
                    Err(error) => ImportMessage::Failed(format!("{error:#}")),
                };
                let _ = sender.send(message);
                repaint.request_repaint();
            });
        match spawned {
            Ok(_) => {
                self.song_import.job_receiver = Some(receiver);
                self.song_import.cancel = Some(cancel);
                self.song_import.stage = "starting".to_owned();
                self.song_import.detail.clear();
                self.song_import.fraction = None;
                self.song_import.notes.clear();
                self.song_import.error = None;
            }
            Err(error) => self.song_import.error = Some(error.to_string()),
        }
    }

    /// Starts the TONE3000 picker on its own thread.
    ///
    /// The flow waits on the user signing in and browsing, which can take
    /// minutes; the interface has to stay live throughout.
    fn start_amp_fetch(&mut self) {
        if self.amp_fetch.is_some() {
            self.status_message = "Already waiting for TONE3000".to_owned();
            return;
        }
        let client = match daw_tone3000::Client::from_env() {
            Ok(client) => client,
            Err(error) => {
                self.status_message = format!("TONE3000: {error}");
                return;
            }
        };
        let (sender, receiver) = std::sync::mpsc::channel();
        let spawned = std::thread::Builder::new()
            .name("tone3000-fetch".to_owned())
            .spawn(move || {
                let outcome = client
                    .select_tone(open_in_browser)
                    .map_err(|error| error.to_string());
                let _ = sender.send(outcome);
            });
        match spawned {
            Ok(_) => {
                self.amp_fetch = Some(receiver);
                self.status_message = "Pick an amp in your browser — RustDAW is waiting".to_owned();
            }
            Err(error) => self.status_message = format!("TONE3000: {error}"),
        }
    }

    /// Writes a finished download into the amp library and selects it.
    fn poll_amp_fetch(&mut self) {
        let Some(receiver) = self.amp_fetch.take() else {
            return;
        };
        match receiver.try_recv() {
            Ok(Ok(model)) => {
                let directory = daw_nam::amp_dir();
                let destination = directory.join(&model.file_name);
                let written = std::fs::create_dir_all(&directory)
                    .and_then(|()| std::fs::write(&destination, &model.bytes));
                match written {
                    Ok(()) => {
                        self.amp_library = daw_nam::discover();
                        // Put it straight on the selected track, which is why
                        // the user went looking for it.
                        if let Some(track) = self.tracks.get_mut(self.selected_track) {
                            track.nam_model = Some(destination);
                            track.effects.nam_enabled = true;
                            self.dirty = true;
                        }
                        self.status_message = format!("Loaded {} from TONE3000", model.name);
                    }
                    Err(error) => {
                        self.status_message =
                            format!("Could not save {}: {error}", model.file_name);
                    }
                }
            }
            Ok(Err(error)) => self.status_message = format!("TONE3000: {error}"),
            Err(TryRecvError::Empty) => self.amp_fetch = Some(receiver),
            Err(TryRecvError::Disconnected) => {
                self.status_message = "The TONE3000 download stopped unexpectedly".to_owned();
            }
        }
    }

    fn poll_song_import(&mut self, context: &egui::Context) {
        if let Some(receiver) = self.song_import.catalog_receiver.take() {
            match receiver.try_recv() {
                Ok(Ok(projects)) => self.song_import.catalog = Some(projects),
                Ok(Err(error)) => self.song_import.catalog_error = Some(error),
                Err(TryRecvError::Empty) => self.song_import.catalog_receiver = Some(receiver),
                Err(TryRecvError::Disconnected) => {
                    self.song_import.catalog_error =
                        Some("the catalog thread stopped unexpectedly".to_owned());
                }
            }
        }

        let Some(receiver) = self.song_import.job_receiver.take() else {
            return;
        };
        let mut running = true;
        let mut outcome = None;
        loop {
            match receiver.try_recv() {
                Ok(ImportMessage::Progress(progress)) => self.apply_import_progress(progress),
                Ok(ImportMessage::Finished(ingested)) => {
                    outcome = Some(Ok(ingested));
                    running = false;
                    break;
                }
                Ok(ImportMessage::Failed(error)) => {
                    outcome = Some(Err(error));
                    running = false;
                    break;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    outcome = Some(Err("the import thread stopped unexpectedly".to_owned()));
                    running = false;
                    break;
                }
            }
        }
        if running {
            self.song_import.job_receiver = Some(receiver);
            context.request_repaint_after(Duration::from_millis(200));
            return;
        }
        self.song_import.cancel = None;
        self.song_import.stage.clear();
        self.song_import.detail.clear();
        self.song_import.fraction = None;
        match outcome {
            Some(Ok(ingested)) => self.finish_song_import(context, *ingested),
            Some(Err(error)) => {
                self.song_import.error = Some(error.clone());
                self.status_message = format!("Song import failed: {error}");
            }
            None => {}
        }
    }

    fn apply_import_progress(&mut self, progress: ImportProgress) {
        match progress {
            ImportProgress::Status(message) => {
                self.song_import.stage = "pipeline".to_owned();
                self.song_import.detail = message;
                self.song_import.fraction = None;
            }
            ImportProgress::Pipeline {
                stage,
                percent,
                message,
            } => {
                self.song_import.stage = stage;
                self.song_import.detail = message;
                self.song_import.fraction = Some((percent / 100.0).clamp(0.0, 1.0));
            }
            ImportProgress::Converting { fraction, stem } => {
                self.song_import.stage = "converting".to_owned();
                self.song_import.detail = format!("{stem} → {} kHz", self.session_rate_khz());
                self.song_import.fraction = Some(fraction.clamp(0.0, 1.0));
            }
        }
    }

    fn session_rate_khz(&self) -> String {
        let rate = self
            .runtime
            .as_ref()
            .map_or(48_000, |runtime| runtime.sample_rate().get());
        format!("{:.1}", f64::from(rate) / 1000.0)
    }

    fn finish_song_import(&mut self, context: &egui::Context, ingested: Ingested) {
        let track_count = ingested.document.tracks.len();
        let name = ingested.document.name.clone();
        self.song_import.notes = ingested.notes;
        // A fresh pipeline run added an entry, so reload rather than just
        // dropping the list — a cleared catalog with no request in flight
        // would leave the window with nothing to show.
        self.song_import.catalog = None;
        self.refresh_song_catalog(context);
        if self.dirty {
            self.pending_open_session = Some(ingested.session_path);
            self.confirm_open_session = true;
            self.status_message =
                format!("Imported {name}; confirm before replacing the current session");
        } else {
            self.open_session(ingested.session_path);
            self.status_message = format!("Imported {name} as {track_count} instrument track(s)");
        }
    }

    fn song_import_window(&mut self, context: &egui::Context) {
        if !self.song_import.open {
            return;
        }
        let mut open = self.song_import.open;
        let mut requested = None;
        let mut cancel_requested = false;
        let mut refresh_requested = false;

        egui::Window::new("IMPORT SONG")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_width(600.0)
            .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
            .show(context, |ui| {
                ui.label(
                    RichText::new(
                        "Separates a song into instrument tracks on the GPU so you can play \
                         along. Everything runs on this machine.",
                    )
                    .small()
                    .color(theme::MUTED),
                );
                ui.separator();

                if self.song_import.is_running() {
                    ui.horizontal(|ui| {
                        ui.add(egui::Spinner::new());
                        ui.label(
                            RichText::new(self.song_import.stage.to_uppercase())
                                .strong()
                                .color(theme::BLUE),
                        );
                    });
                    if !self.song_import.detail.is_empty() {
                        ui.label(RichText::new(&self.song_import.detail).small());
                    }
                    match self.song_import.fraction {
                        Some(fraction) => {
                            ui.add(egui::ProgressBar::new(fraction).show_percentage());
                        }
                        None => {
                            ui.label(RichText::new("Working…").small().color(theme::MUTED));
                        }
                    }
                    ui.add_space(4.0);
                    if ui.button("CANCEL").clicked() {
                        cancel_requested = true;
                    }
                    ui.label(
                        RichText::new(
                            "Cancelling stops RustDAW waiting. The pipeline keeps running and \
                             the song appears in the list below when it finishes.",
                        )
                        .small()
                        .color(theme::MUTED),
                    );
                    return;
                }

                ui.label(RichText::new("NEW SONG").strong());
                ui.horizontal(|ui| {
                    ui.label("Link");
                    let field = ui.add(
                        egui::TextEdit::singleline(&mut self.song_import.url)
                            .hint_text("https://www.youtube.com/watch?v=…")
                            .desired_width(340.0),
                    );
                    let submitted =
                        field.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
                    let usable = self.song_import.url.trim().starts_with("http");
                    // A disabled button never reports a click, so `usable` is
                    // already implied on that side.
                    let clicked = ui
                        .add_enabled(usable, egui::Button::new("SEPARATE & IMPORT"))
                        .clicked();
                    if clicked || (submitted && usable) {
                        requested = Some(ImportSource::Url(self.song_import.url.trim().to_owned()));
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("File");
                    if ui
                        .button("CHOOSE FILE & IMPORT")
                        .on_hover_text(
                            "Separates a song you already have. Any format ffmpeg reads works.",
                        )
                        .clicked()
                        && let Some(path) = rfd::FileDialog::new()
                            .set_title("Choose a song to separate")
                            .add_filter(
                                "Audio",
                                &[
                                    "mp3", "wav", "wave", "flac", "m4a", "aac", "ogg", "opus",
                                    "wma", "aiff", "aif", "mp4", "webm", "mkv",
                                ],
                            )
                            // The pipeline decodes with ffmpeg, which reads far
                            // more than the list above; this is the way out for
                            // anything it can read that is not named here.
                            .add_filter("Any file (*.*)", &["*"])
                            .pick_file()
                    {
                        requested = Some(ImportSource::LocalFile(path));
                    }
                    ui.label(
                        RichText::new("…or separate a song from this disk")
                            .small()
                            .color(theme::MUTED),
                    );
                });
                ui.label(
                    RichText::new(
                        "A new song takes a few minutes: download, separation, then MIDI.",
                    )
                    .small()
                    .color(theme::MUTED),
                );

                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.checkbox(
                        &mut self.song_import.align_to_bar,
                        "Align first downbeat to bar 1",
                    )
                    .on_hover_text(
                        "Delays the song by less than a bar so the click lines up with the music.",
                    );
                    ui.checkbox(&mut self.song_import.import_midi, "Import MIDI")
                        .on_hover_text(
                            "Adds the transcription as instrument tracks you can edit in the \
                             piano roll.",
                        );
                    ui.checkbox(
                        &mut self.song_import.include_drumkit,
                        "Separate drum kit parts",
                    )
                    .on_hover_text(
                        "Adds kick, snare, toms and cymbals as their own tracks. They sum to the \
                         drum stem, so mute the Drums track if you enable this.",
                    );
                });

                // A tempo and its double fit the audio equally well, so fast
                // music is read at half speed unless someone says otherwise.
                // The person importing the song knows what it is.
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label("Expected tempo");
                    let presets = daw_analysis::TempoHint::PRESETS;
                    let selected = presets
                        .get(self.song_import.tempo_hint)
                        .copied()
                        .unwrap_or(presets[0]);
                    egui::ComboBox::from_id_salt("tempo-hint")
                        .selected_text(selected.0)
                        .width(240.0)
                        .show_ui(ui, |ui| {
                            for (index, (label, centre)) in presets.iter().enumerate() {
                                ui.selectable_value(
                                    &mut self.song_import.tempo_hint,
                                    index,
                                    if index == 0 {
                                        (*label).to_owned()
                                    } else {
                                        format!("{label}  (~{centre:.0} BPM)")
                                    },
                                );
                            }
                        })
                        .response
                        .on_hover_text(
                            "Which tempo to prefer when a song fits two readings equally — \
                             174 BPM and 87 BPM describe the same drum and bass track.\nThe \
                             tempo is still measured from the audio; this only settles the tie, \
                             so a slow song picked as \"very fast\" still reports its own tempo.",
                        );
                });

                // Singers ask for a lower key at the last minute, so this sits
                // with the other import options and applies to the songs below
                // as well: re-importing an already-separated song at another
                // transposition is a few seconds of conversion.
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label("Transpose");
                    ui.add(
                        egui::DragValue::new(&mut self.song_import.transpose)
                            .range(-MAX_TRANSPOSE_SEMITONES..=MAX_TRANSPOSE_SEMITONES)
                            .speed(0.1)
                            .suffix(" st"),
                    )
                    .on_hover_text(
                        "Moves the song into another key without changing its tempo. The drums \
                         are left alone — a kit has no key — and the chord chart and \
                         transcription move with the audio.\nApplies to already-processed songs \
                         too, so the same song can be imported at two keys to rehearse both.",
                    );
                    // Both directions: a singer asking for a higher key is as
                    // ordinary as one asking for a lower one, and the box beside
                    // these goes the whole octave either way.
                    for semitones in [-4, -2, -1, 0, 1, 2, 4] {
                        let label = if semitones == 0 {
                            "0".to_owned()
                        } else {
                            format!("{semitones:+}")
                        };
                        if ui
                            .selectable_label(self.song_import.transpose == semitones, label)
                            .clicked()
                        {
                            self.song_import.transpose = semitones;
                        }
                    }
                    ui.label(
                        RichText::new(transpose_description(self.song_import.transpose))
                            .small()
                            .color(theme::MUTED),
                    );
                });

                ui.separator();
                ui.horizontal(|ui| {
                    ui.label(RichText::new("ALREADY PROCESSED").strong());
                    if ui.small_button("Refresh").clicked() {
                        refresh_requested = true;
                    }
                });

                if self.song_import.catalog_receiver.is_some() {
                    // Only a request actually in flight may show a spinner.
                    ui.horizontal(|ui| {
                        ui.add(egui::Spinner::new());
                        ui.label(RichText::new("Looking for the pipeline…").small());
                    });
                } else if let Some(error) = &self.song_import.catalog_error {
                    ui.label(RichText::new(error).color(theme::RED));
                } else if let Some(catalog) = &self.song_import.catalog {
                    if catalog.is_empty() {
                        ui.label(
                            RichText::new("No processed songs yet.")
                                .small()
                                .color(theme::MUTED),
                        );
                    } else {
                        egui::ScrollArea::vertical()
                            .max_height(220.0)
                            .show(ui, |ui| {
                                for project in catalog {
                                    ui.horizontal(|ui| {
                                        if ui.button("IMPORT").clicked() {
                                            requested = Some(ImportSource::ExistingProject(
                                                project.id.clone(),
                                            ));
                                        }
                                        ui.label(project.label());
                                        if let Some(duration) = project.duration {
                                            ui.label(
                                                RichText::new(format!(
                                                    "{:.0}:{:02.0}",
                                                    (duration / 60.0).floor(),
                                                    duration % 60.0
                                                ))
                                                .small()
                                                .color(theme::MUTED),
                                            );
                                        }
                                    });
                                }
                            });
                    }
                } else {
                    ui.label(
                        RichText::new("Press Refresh to list processed songs.")
                            .small()
                            .color(theme::MUTED),
                    );
                }

                if let Some(error) = &self.song_import.error {
                    ui.separator();
                    ui.label(RichText::new(error).color(theme::RED));
                }
                if !self.song_import.notes.is_empty() {
                    ui.separator();
                    ui.label(RichText::new("LAST IMPORT").strong().color(theme::GREEN));
                    for note in &self.song_import.notes {
                        ui.label(RichText::new(note).small());
                    }
                }
            });

        self.song_import.open = open;
        if cancel_requested {
            if let Some(cancel) = &self.song_import.cancel {
                cancel.cancel();
            }
            self.status_message = "Cancelling song import…".to_owned();
        }
        if refresh_requested {
            self.song_import.catalog = None;
            self.refresh_song_catalog(context);
        }
        if let Some(source) = requested {
            self.start_song_import(context, source);
        }
    }

    /// The tick under the playhead, for drawing the piano roll's cursor.
    fn playhead_tick(&self) -> u64 {
        let sample_rate = self
            .runtime
            .as_ref()
            .map_or(48_000, |runtime| runtime.sample_rate().get());
        self.tempo_map
            .frame_to_tick(self.snapshot().position_frames, sample_rate)
    }

    /// Opens the piano roll on the first instrument track that has notes.
    fn open_first_midi_clip(&mut self) {
        let found =
            self.tracks.iter().enumerate().find_map(|(index, track)| {
                track.midi_clips.first().map(|clip| (index, clip.clone()))
            });
        if let Some((index, clip)) = found {
            self.piano_roll.open_clip(index, 0, &clip);
            self.selected_track = index;
        } else {
            self.status_message = "No instrument tracks in this session".to_owned();
        }
    }

    fn piano_roll_window(&mut self, context: &egui::Context) {
        if !self.piano_roll.open {
            return;
        }
        let (track_index, clip_index) = (self.piano_roll.track, self.piano_roll.clip);
        let Some(mut clip) = self
            .tracks
            .get(track_index)
            .and_then(|track| track.midi_clips.get(clip_index))
            .cloned()
        else {
            self.piano_roll.open = false;
            return;
        };
        let track_name = self.tracks[track_index].name.clone();
        let playhead_tick = self.playhead_tick();
        let beats_per_bar = self.meter_numerator;

        let mut open = true;
        let mut result = None;
        egui::Window::new(format!("PIANO ROLL — {track_name}"))
            .open(&mut open)
            .default_size([1_000.0, 460.0])
            .resizable(true)
            .show(context, |ui| {
                result = Some(piano_roll::show(
                    &mut self.piano_roll,
                    ui,
                    &mut clip,
                    &self.tempo_map,
                    beats_per_bar,
                    playhead_tick,
                ));
            });
        self.piano_roll.open = open;

        let Some(result) = result else {
            return;
        };
        if let Some(tick) = result.seek_to_tick {
            let sample_rate = self
                .runtime
                .as_ref()
                .map_or(48_000, |runtime| runtime.sample_rate().get());
            let frame = self.tempo_map.tick_to_frame(tick, sample_rate);
            if let Some(runtime) = &self.runtime {
                runtime.seek(SamplePosition::new(frame));
            }
        }
        if result.edited {
            self.tracks[track_index].midi_clips[clip_index] = clip;
            self.dirty = true;
            self.playback_synced = false;
            if let Err(error) = self.sync_playback() {
                self.status_message = format!("Note edited; playback update failed: {error}");
            }
        }
    }

    /// The chord chart, drawn as a lane of marks under the ruler.
    ///
    /// One cell per beat, on the same grid and the same scroll offset as the
    /// tracks below, so a chord sits over the audio it belongs to. A chord is
    /// printed where it changes and dotted where it is held — the way a chart
    /// is written, and the way the eye reads one, which is by looking for the
    /// changes.
    /// The smallest and largest the chord chart can be drawn, as a multiple of
    /// its base size. The bottom is where the text stops being legible; the top
    /// is where the lane starts eating the timeline.
    const CHORD_SCALE_RANGE: std::ops::RangeInclusive<f32> = 0.7..=3.0;

    /// Seconds under an x position **on the ruler**.
    ///
    /// The ruler is drawn outside the timeline's scroll area and offsets itself
    /// by hand, so reading a position off it has to add that offset back.
    fn ruler_seconds_at(&self, ruler_left: f32, x: f32) -> f64 {
        f64::from(((x - ruler_left + self.timeline_scroll_x) / self.pixels_per_second).max(0.0))
    }

    /// Seconds under an x position **inside a track row**.
    ///
    /// Track rows live inside the scroll area, which has already translated
    /// them — [`draw_clip`] places a clip at `track_rect.left() + start` with no
    /// offset of its own. Adding the scroll here as well would count it twice,
    /// which is what made a trim at the end of a scrolled timeline stretch the
    /// clip by however far the view had been scrolled.
    fn track_seconds_at(&self, track_left: f32, x: f32) -> f64 {
        f64::from(((x - track_left) / self.pixels_per_second).max(0.0))
    }

    /// The timeline ruler: where you are, where you are going, and what is
    /// looping.
    ///
    /// Clicking moves the playhead, because waiting for playback to reach the
    /// chorus to hear the chorus is not a workflow. Dragging marks a loop, so
    /// a passage can be played round until it is right.
    fn timeline_ruler(
        &mut self,
        ui: &mut egui::Ui,
        ruler: Rect,
        response: &egui::Response,
        sample_rate: u32,
    ) {
        let painter = ui.painter_at(ruler);
        painter.rect_filled(ruler, 0.0, theme::PANEL_2);

        let to_frame = |seconds: f64| (seconds * f64::from(sample_rate)) as u64;
        let x_of = |seconds: f64| {
            ruler.left() + (seconds as f32) * self.pixels_per_second - self.timeline_scroll_x
        };

        // Dragging marks a loop; a plain click drops the playhead.
        if response.drag_started() {
            if let Some(pointer) = response.interact_pointer_pos() {
                self.loop_drag_anchor = Some(self.ruler_seconds_at(ruler.left(), pointer.x));
            }
        }
        if response.dragged() {
            if let (Some(anchor), Some(pointer)) =
                (self.loop_drag_anchor, response.interact_pointer_pos())
            {
                let here = self.ruler_seconds_at(ruler.left(), pointer.x);
                let (from, to) = if here < anchor {
                    (here, anchor)
                } else {
                    (anchor, here)
                };
                // A flick of a drag is a click that wobbled, not a loop.
                if to - from > 0.05 {
                    self.loop_range = Some((from, to));
                }
            }
        }
        if response.drag_stopped() {
            self.loop_drag_anchor = None;
            if let (Some(runtime), Some((from, to))) = (&self.runtime, self.loop_range) {
                runtime.set_loop(to_frame(from), to_frame(to));
                runtime.seek(daw_core::SamplePosition::new(to_frame(from)));
                let playing = matches!(
                    self.snapshot().transport,
                    RuntimeTransportState::Playing | RuntimeTransportState::Recording
                );
                self.status_message = if playing {
                    format!("Looping {from:.2}s – {to:.2}s. Click the ruler to clear.")
                } else {
                    format!(
                        "Loop set {from:.2}s – {to:.2}s — press Play. Click the ruler to clear."
                    )
                };
            }
        }
        if response.clicked() {
            if let Some(pointer) = response.interact_pointer_pos() {
                let seconds = self.ruler_seconds_at(ruler.left(), pointer.x);
                if let Some(runtime) = &self.runtime {
                    runtime.seek(daw_core::SamplePosition::new(to_frame(seconds)));
                    // A click outside the loop means the loop is no longer what
                    // is wanted; one inside it is just moving within it.
                    let inside = self
                        .loop_range
                        .is_some_and(|(from, to)| seconds >= from && seconds <= to);
                    if !inside && self.loop_range.take().is_some() {
                        runtime.clear_loop();
                        self.status_message = "Loop cleared".to_owned();
                    }
                }
            }
        }

        // The loop, behind the marks.
        if let Some((from, to)) = self.loop_range {
            let band = Rect::from_min_max(
                Pos2::new(x_of(from).max(ruler.left()), ruler.top()),
                Pos2::new(x_of(to).min(ruler.right()), ruler.bottom()),
            );
            if band.width() > 0.0 {
                painter.rect_filled(band, 0.0, theme::BLUE_DARK);
                for edge in [band.left(), band.right()] {
                    painter.line_segment(
                        [
                            Pos2::new(edge, ruler.top()),
                            Pos2::new(edge, ruler.bottom()),
                        ],
                        Stroke::new(1.5_f32, theme::BLUE),
                    );
                }
            }
        }

        if self.ruler_shows_bars {
            self.draw_bar_marks(&painter, ruler);
        } else {
            self.draw_second_marks(&painter, ruler);
        }

        // The unit switch, parked at the left where it cannot be scrolled away.
        let toggle = Rect::from_min_size(
            Pos2::new(ruler.left() + 4.0, ruler.top() + 4.0),
            Vec2::new(46.0, ruler_height_inner()),
        );
        let toggle_response = ui.interact(toggle, ui.id().with("ruler-units"), Sense::click());
        painter.rect_filled(toggle, 3.0, theme::PANEL);
        painter.text(
            toggle.center(),
            Align2::CENTER_CENTER,
            if self.ruler_shows_bars {
                "BARS"
            } else {
                "TIME"
            },
            FontId::monospace(10.0),
            theme::BLUE,
        );
        if toggle_response.clicked() {
            self.ruler_shows_bars = !self.ruler_shows_bars;
        }
        toggle_response
            .on_hover_text("Show the ruler in bars and beats, or in minutes and seconds");

        response.clone().on_hover_text(
            "Click to move the playhead.\nDrag to mark a loop; click outside it to clear.",
        );
    }

    /// Minutes and seconds, labelled every five.
    fn draw_second_marks(&self, painter: &egui::Painter, ruler: Rect) {
        for second in 0..=3_600 {
            let x = ruler.left() + second as f32 * self.pixels_per_second - self.timeline_scroll_x;
            if x > ruler.right() {
                break;
            }
            if x < ruler.left() - 4.0 {
                continue;
            }
            if second % 5 == 0 {
                painter.line_segment(
                    [
                        Pos2::new(x, ruler.bottom() - 5.0),
                        Pos2::new(x, ruler.bottom()),
                    ],
                    Stroke::new(1.0_f32, theme::BORDER),
                );
                painter.text(
                    Pos2::new(x + 4.0, ruler.center().y),
                    Align2::LEFT_CENTER,
                    format!("{:02}:{:02}", second / 60, second % 60),
                    FontId::monospace(11.0),
                    theme::MUTED,
                );
            }
        }
    }

    /// Bars and beats, from the session's tempo map, so the ruler agrees with
    /// the click and with the chord chart above the tracks.
    fn draw_bar_marks(&self, painter: &egui::Painter, ruler: Rect) {
        let per_beat = u64::from(daw_midi::TICKS_PER_QUARTER);
        let beats_per_bar = u64::from(self.meter_numerator.max(1));
        // Label every bar while they are far enough apart to read, then every
        // fourth, then every sixteenth.
        let seconds_per_beat = 60.0 / f64::from(self.tempo.max(1));
        let pixels_per_bar =
            (seconds_per_beat * beats_per_bar as f64) as f32 * self.pixels_per_second;
        let label_every = if pixels_per_bar > 48.0 {
            1
        } else if pixels_per_bar > 14.0 {
            4
        } else {
            16
        };

        for bar in 0..10_000_u64 {
            let tick = bar * beats_per_bar * per_beat;
            let seconds = self.tempo_map.tick_to_seconds(tick);
            let x =
                ruler.left() + (seconds as f32) * self.pixels_per_second - self.timeline_scroll_x;
            if x > ruler.right() {
                break;
            }
            if x < ruler.left() - 4.0 {
                continue;
            }
            let labelled = bar % label_every == 0;
            painter.line_segment(
                [
                    Pos2::new(x, ruler.bottom() - if labelled { 7.0 } else { 4.0 }),
                    Pos2::new(x, ruler.bottom()),
                ],
                Stroke::new(1.0_f32, theme::BORDER),
            );
            if labelled {
                painter.text(
                    Pos2::new(x + 4.0, ruler.center().y),
                    Align2::LEFT_CENTER,
                    format!("{}", bar + 1),
                    FontId::monospace(11.0),
                    theme::MUTED,
                );
            }
        }
    }

    /// The chord chart, drawn as a lane of marks under the ruler.
    ///
    /// One cell per beat, on the same grid and the same scroll offset as the
    /// tracks below, so a chord sits over the audio it belongs to. A chord is
    /// printed where it changes and dotted where it is held — the way a chart
    /// is written, and the way the eye reads one, which is by looking for the
    /// changes.
    ///
    /// Scrolling over the lane resizes it. Reading a chart is something people
    /// do at a glance from a distance, sometimes with a guitar in their hands,
    /// so how big it wants to be is a matter of the room and the eyes rather
    /// than anything this can pick correctly on their behalf.
    fn chord_lane(&mut self, ui: &mut egui::Ui, left: f32, width: f32) {
        if !self.chords_open || self.chords.is_empty() {
            return;
        }

        let scale = self.audio_preferences.chord_lane_scale.clamp(
            *Self::CHORD_SCALE_RANGE.start(),
            *Self::CHORD_SCALE_RANGE.end(),
        );
        let text_size = 12.0 * scale;
        let lane_height = (text_size + 14.0).max(22.0);

        let (lane, response) =
            ui.allocate_exact_size(Vec2::new(width, lane_height), Sense::click_and_drag());

        // Resize by scrolling over the lane. Multiplying rather than adding
        // keeps each notch the same proportional step at every size.
        if response.hovered() {
            let wheel = ui.input(|input| input.smooth_scroll_delta.y);
            if wheel.abs() > 0.1 {
                let adjusted = (scale * (1.0 + wheel * 0.0015)).clamp(
                    *Self::CHORD_SCALE_RANGE.start(),
                    *Self::CHORD_SCALE_RANGE.end(),
                );
                if (adjusted - scale).abs() > f32::EPSILON {
                    self.audio_preferences.chord_lane_scale = adjusted;
                    // Written once the wheel stops, not on every notch.
                    self.chord_scale_unsaved = true;
                    self.status_message = format!("Chord chart at {:.0}%", adjusted * 100.0);
                }
            }
        }

        let painter = ui.painter_at(lane);
        painter.rect_filled(lane, 0.0, theme::PANEL);

        let visible_start = f64::from(self.timeline_scroll_x / self.pixels_per_second);
        let visible_end = visible_start + f64::from(width / self.pixels_per_second);
        let chart = daw_project::chord_chart(
            &self.chords,
            &self.tempo_map,
            self.meter_numerator,
            visible_end + 1.0,
        );

        // Below this, beats are closer together than their labels are wide and
        // the lane becomes unreadable; only the changes are drawn. The bigger
        // the text, the sooner that happens.
        let beat_width = self.pixels_per_second * 60.0 / f32::from(self.tempo.max(1));
        let crowded = beat_width < text_size * 2.2;

        for beat in &chart {
            if beat.seconds < visible_start - 1.0 || beat.seconds > visible_end {
                continue;
            }
            #[allow(clippy::cast_possible_truncation)]
            let x = left + beat.seconds as f32 * self.pixels_per_second - self.timeline_scroll_x;
            if x < left - 40.0 * scale || x > lane.right() {
                continue;
            }

            // Bar lines, so the chart is countable.
            if beat.is_downbeat() {
                painter.line_segment(
                    [Pos2::new(x, lane.top()), Pos2::new(x, lane.bottom())],
                    Stroke::new(1.0_f32, theme::BORDER),
                );
            }

            match &beat.label {
                Some(label) => {
                    // A change: the chord itself, and a tick marking the beat
                    // it lands on.
                    painter.line_segment(
                        [
                            Pos2::new(x, lane.bottom() - 4.0 * scale),
                            Pos2::new(x, lane.bottom()),
                        ],
                        Stroke::new(1.5_f32 * scale, theme::BLUE),
                    );
                    let faded = beat.confidence < 0.4;
                    painter.text(
                        Pos2::new(x + 3.0 * scale, lane.center().y),
                        Align2::LEFT_CENTER,
                        label,
                        FontId::proportional(text_size),
                        if faded { theme::MUTED } else { theme::TEXT },
                    );
                }
                None if !crowded => {
                    // Held: a dot, meaning "still the last one".
                    painter.circle_filled(
                        Pos2::new(x + 6.0 * scale, lane.center().y + 1.0),
                        1.2 * scale,
                        theme::MUTED,
                    );
                }
                None => {}
            }
        }

        let mut hint = "The detected chord chart. A chord is shown where it changes; a dot means \
                        it is still playing.\n\nScroll here to resize it."
            .to_owned();
        if let Some(key) = &self.detected_key {
            hint = format!("Key: {key}.\n{hint}");
        }
        response.on_hover_text(hint);
    }

    fn export_mix(&mut self) {
        if matches!(
            self.snapshot().transport,
            RuntimeTransportState::Recording | RuntimeTransportState::CountIn
        ) {
            self.status_message = "Stop recording before exporting".to_owned();
            self.export_report = Some(ExportReport::failed(
                "Still recording",
                "Stop the transport, then export.".to_owned(),
            ));
            return;
        }
        if self.tracks.iter().all(|track| track.clips.is_empty()) {
            self.status_message = "Nothing to export yet".to_owned();
            self.export_report = Some(ExportReport::failed(
                "Nothing to export",
                "This session has no audio clips yet.".to_owned(),
            ));
            return;
        }

        // The export used to go straight to Exports/Current Mix.wav under the
        // media root, which is the working directory — for an app started from
        // the desktop that is the home folder, so the mix landed somewhere the
        // person exporting it never saw. Now they say where it goes.
        let directory = daw_core::media_dir("Exports");
        let _ = std::fs::create_dir_all(&directory);
        let suggested = format!("{}.wav", sanitize_file_name(&self.session_name));
        let Some(chosen) = rfd::FileDialog::new()
            .set_title("Export the mix as")
            .add_filter("WAV audio", &["wav"])
            .set_directory(&directory)
            .set_file_name(&suggested)
            .save_file()
        else {
            self.status_message = "Export cancelled".to_owned();
            return;
        };
        let destination = ensure_extension(chosen, "wav");

        let mastered = if self.master_reference.is_some() {
            " (mastered)"
        } else {
            ""
        };
        match daw_render::export_stereo(&self.project_document(), &destination) {
            Ok(frames) => {
                let rate = self
                    .runtime
                    .as_ref()
                    .map_or(48_000, |runtime| runtime.sample_rate().get());
                let seconds = frames as f64 / f64::from(rate);
                self.status_message = format!(
                    "Exported {seconds:.2} s{mastered} to {}",
                    destination.display()
                );
                self.export_report = Some(ExportReport::exported(
                    format!(
                        "{}, {seconds:.1} s{mastered}.",
                        destination
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("The mix")
                    ),
                    destination,
                ));
            }
            Err(error) => {
                self.status_message = format!("{error:#}");
                self.export_report =
                    Some(ExportReport::failed("Export failed", format!("{error:#}")));
            }
        }
    }

    /// Starts moving the loaded session into another key on a worker thread.
    ///
    /// The stems are re-rendered from the originals the session already holds,
    /// so this is a few seconds rather than another import — and a key that has
    /// been heard before is already on disk and comes back at once.
    fn start_rekey(&mut self, context: &egui::Context, semitones: i32) {
        if self.transpose_receiver.is_some() {
            return;
        }
        let Some(session_dir) = self.session_path.parent().map(std::path::Path::to_path_buf) else {
            self.status_message = "This session has no folder to write the new key into".to_owned();
            return;
        };
        let mut document = self.project_document();
        let (sender, receiver) = channel();
        let repaint = context.clone();
        let spawned = std::thread::Builder::new()
            .name("rekey".to_owned())
            .spawn(move || {
                let outcome = daw_songimport::rekey_session(
                    &mut document,
                    &session_dir,
                    semitones,
                    &|_, _| {},
                )
                .map(|rekeyed| (document, rekeyed))
                .map_err(|error| format!("{error:#}"));
                let _ = sender.send(outcome);
                repaint.request_repaint();
            });
        match spawned {
            Ok(_) => {
                self.transpose_receiver = Some(receiver);
                self.status_message = format!("Moving the song to {semitones:+} semitones…");
            }
            Err(error) => self.status_message = format!("Could not start the re-key: {error}"),
        }
    }

    /// Moves the re-keyed document into the loaded session.
    ///
    /// Only what a key change touches is copied across — the files the clips
    /// read, the transcription and the chart — so the waveforms already drawn,
    /// the selection and the undo history all survive it. A shifted stem has
    /// the same length and envelope as the one it came from, so its peaks are
    /// still the right picture.
    fn apply_rekeyed(&mut self, document: &ProjectDocument) {
        for project_track in &document.tracks {
            let Some(track) = self
                .tracks
                .iter_mut()
                .find(|track| track.id == project_track.id)
            else {
                continue;
            };
            for project_clip in &project_track.clips {
                if let Some(clip) = track
                    .clips
                    .iter_mut()
                    .find(|clip| clip.id == project_clip.id)
                {
                    clip.path.clone_from(&project_clip.path);
                    clip.source_path.clone_from(&project_clip.source_path);
                }
            }
            track.midi_clips.clone_from(&project_track.midi_clips);
        }
        self.chords.clone_from(&document.chords);
        self.detected_key.clone_from(&document.key);
        self.session_transpose = document.transpose_semitones;
    }

    /// Takes the re-keyed session from the worker thread once it is ready.
    fn poll_rekey(&mut self) {
        let Some(receiver) = self.transpose_receiver.take() else {
            return;
        };
        match receiver.try_recv() {
            Ok(Ok((document, rekeyed))) => {
                // The playhead is left where it was: the whole point is to hear
                // the same passage again in the new key.
                self.apply_rekeyed(&document);
                self.dirty = true;
                self.save_session();
                let source = if rekeyed.rendered == 0 {
                    "already rendered".to_owned()
                } else {
                    format!("{} stem(s) shifted", rekeyed.rendered)
                };
                let key = self
                    .detected_key
                    .as_deref()
                    .map_or_else(String::new, |key| format!(" — now in {key}"));
                self.status_message = format!(
                    "Song moved to {:+} semitones ({source}){key}. {}",
                    rekeyed.semitones,
                    rekeyed.notes.join(" ")
                );
                if let Err(error) = self.sync_playback() {
                    self.status_message = format!("Key changed; playback preload failed: {error}");
                }
            }
            Ok(Err(error)) => {
                self.status_message = format!("The key could not be changed: {error}");
            }
            Err(TryRecvError::Empty) => self.transpose_receiver = Some(receiver),
            Err(TryRecvError::Disconnected) => {
                self.status_message = "The re-key stopped before it finished".to_owned();
            }
        }
    }

    /// The window that chooses the key the loaded session plays in.
    fn transpose_window(&mut self, context: &egui::Context) {
        if !self.transpose_open {
            return;
        }
        let mut open = self.transpose_open;
        let mut wanted = None;
        let mut forget = false;
        let running = self.transpose_receiver.is_some();
        let current = self.session_transpose;
        egui::Window::new("SONG KEY")
            .open(&mut open)
            .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
            .collapsible(false)
            .resizable(false)
            .show(context, |ui| {
                ui.label(
                    RichText::new(
                        "Moves the whole song into another key to rehearse in: the stems, the \
                         chord chart and the transcription together. The drums are left alone.",
                    )
                    .small()
                    .color(theme::MUTED),
                );
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Now playing in");
                    ui.label(
                        RichText::new(match self.detected_key.as_deref() {
                            Some(key) => key.to_owned(),
                            None => "an undetected key".to_owned(),
                        })
                        .strong()
                        .color(theme::BLUE),
                    );
                    if current != 0 {
                        ui.label(
                            RichText::new(format!("({current:+} from the original)"))
                                .small()
                                .color(theme::MUTED),
                        );
                    }
                });
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label("Move to");
                    ui.add_enabled(
                        !running,
                        egui::DragValue::new(&mut self.transpose_wanted)
                            .range(-MAX_TRANSPOSE_SEMITONES..=MAX_TRANSPOSE_SEMITONES)
                            .speed(0.1)
                            .suffix(" st"),
                    );
                    for semitones in [-4, -2, -1, 0, 1, 2, 4] {
                        let label = if semitones == 0 {
                            "0".to_owned()
                        } else {
                            format!("{semitones:+}")
                        };
                        if ui
                            .add_enabled(
                                !running,
                                egui::Button::selectable(self.transpose_wanted == semitones, label),
                            )
                            .clicked()
                        {
                            self.transpose_wanted = semitones;
                        }
                    }
                    ui.label(
                        RichText::new(transpose_description(self.transpose_wanted))
                            .small()
                            .color(theme::MUTED),
                    );
                });
                ui.add_space(6.0);
                if running {
                    ui.horizontal(|ui| {
                        ui.add(egui::Spinner::new());
                        ui.label(RichText::new("Rendering the new key…").small());
                    });
                    return;
                }
                let changed = self.transpose_wanted != current;
                if ui
                    .add_enabled(changed, egui::Button::new("CHANGE KEY"))
                    .on_hover_text(
                        "A few seconds the first time. A key you have already heard comes back \
                         at once, because its render is still on disk.",
                    )
                    .clicked()
                {
                    wanted = Some(self.transpose_wanted);
                }
                if !changed {
                    ui.label(
                        RichText::new("The song is already in this key.")
                            .small()
                            .color(theme::MUTED),
                    );
                }
                // Every key heard is kept so it comes back at once, which is a
                // quarter of a gigabyte a time on a three-minute song. Say what
                // that is costing, and offer to hand it back.
                if let Some(session_dir) = self.session_path.parent() {
                    let held = daw_songimport::other_keys_size(session_dir, current);
                    if held > 0 {
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new(format!(
                                    "{:.1} GB of other keys kept for instant recall",
                                    held as f64 / 1_000_000_000.0
                                ))
                                .small()
                                .color(theme::MUTED),
                            );
                            if ui.small_button("FORGET THEM").clicked() {
                                forget = true;
                            }
                        });
                    }
                }
            });
        self.transpose_open = open;
        if forget {
            let freed = self
                .session_path
                .parent()
                .map(|directory| daw_songimport::forget_other_keys(directory, current));
            self.status_message = match freed {
                Some(Ok(bytes)) => format!(
                    "Freed {:.1} GB; those keys will render again when asked for",
                    bytes as f64 / 1_000_000_000.0
                ),
                Some(Err(error)) => format!("Could not remove the other keys: {error:#}"),
                None => "This session has no folder".to_owned(),
            };
        }
        if let Some(semitones) = wanted {
            self.start_rekey(context, semitones);
        }
    }

    /// Exports every audible track as its own WAV, into a folder of stems.
    fn export_stems(&mut self) {
        if matches!(
            self.snapshot().transport,
            RuntimeTransportState::Recording | RuntimeTransportState::CountIn
        ) {
            self.status_message = "Stop recording before exporting".to_owned();
            self.export_report = Some(ExportReport::failed(
                "Still recording",
                "Stop the transport, then export.".to_owned(),
            ));
            return;
        }
        let any_solo = self.tracks.iter().any(|track| track.solo);
        let exportable = self
            .tracks
            .iter()
            .filter(|track| !track.clips.is_empty() && !track.muted && (!any_solo || track.solo))
            .count();
        if exportable == 0 {
            self.status_message = "Nothing to export yet".to_owned();
            self.export_report = Some(ExportReport::failed(
                "Nothing to export",
                "No audible track in this session has audio clips.".to_owned(),
            ));
            return;
        }

        let directory = daw_core::media_dir("Exports");
        let _ = std::fs::create_dir_all(&directory);
        let Some(chosen) = rfd::FileDialog::new()
            .set_title("Choose where to put the stems")
            .set_directory(&directory)
            .pick_folder()
        else {
            self.status_message = "Export cancelled".to_owned();
            return;
        };
        // Its own folder inside the chosen one: a stem export is a dozen files,
        // and dropping them loose into Music or Desktop would be rude.
        let target = unique_directory(
            &chosen,
            &format!("{} Stems", sanitize_file_name(&self.session_name)),
        );

        match daw_render::export_stems(&self.project_document(), &target) {
            Ok(stems) => {
                // Everything the session has that a stem could not be made of,
                // said plainly rather than left as a silent difference in count.
                let mut notes = Vec::new();
                let instruments = self
                    .tracks
                    .iter()
                    .filter(|track| track.kind.is_instrument())
                    .count();
                if instruments > 0 {
                    notes.push(format!(
                        "{instruments} instrument track(s) skipped — MIDI is not rendered offline"
                    ));
                }
                let silenced = self
                    .tracks
                    .iter()
                    .filter(|track| {
                        !track.clips.is_empty() && (track.muted || (any_solo && !track.solo))
                    })
                    .count();
                if silenced > 0 {
                    notes.push(format!("{silenced} muted track(s) left out"));
                }
                if self.master_reference.is_some() {
                    notes.push("mastering applies to the mix, not to stems".to_owned());
                }
                // A track over full scale on its own clips in its stem even
                // when the mix sounds clean, so it has to be said out loud.
                let clipped: Vec<&str> = stems
                    .iter()
                    .filter(|stem| stem.clipped())
                    .map(|stem| stem.track.as_str())
                    .collect();
                if !clipped.is_empty() {
                    notes.push(format!(
                        "clipped on its own, lower the fader: {}",
                        clipped.join(", ")
                    ));
                }
                self.status_message =
                    format!("Exported {} stem(s) to {}", stems.len(), target.display());
                self.export_report = Some(ExportReport::exported(
                    if notes.is_empty() {
                        format!("{} stem(s) written.", stems.len())
                    } else {
                        format!("{} stem(s) written. {}.", stems.len(), notes.join("; "))
                    },
                    target,
                ));
            }
            Err(error) => {
                self.status_message = format!("{error:#}");
                self.export_report = Some(ExportReport::failed(
                    "Stem export failed",
                    format!("{error:#}"),
                ));
            }
        }
    }

    /// Chooses the record the exported mix is matched to.
    ///
    /// The reference has to be at the session rate for the same reason every
    /// other file does: the engine does not resample, and a reference
    /// converted on the way in would be measured through whatever the
    /// converter did to it.
    fn choose_master_reference(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .set_title("Choose a reference track to master against")
            .add_filter("WAV audio", &["wav"])
            .pick_file()
        else {
            return;
        };

        let rate = self
            .runtime
            .as_ref()
            .map_or(48_000, |runtime| runtime.sample_rate().get());
        match daw_master::load_reference(&path, rate) {
            Ok(_) => {
                let name = path.file_name().map_or_else(
                    || path.display().to_string(),
                    |name| name.to_string_lossy().into_owned(),
                );
                self.master_reference = Some(path);
                self.dirty = true;
                self.status_message = format!("Mastering against {name}");
            }
            Err(error) => self.status_message = format!("{error:#}"),
        }
    }

    fn new_session(&mut self) {
        if let Some(runtime) = &self.runtime {
            runtime.stop();
            let _ = runtime.clear_playback();
            runtime.seek_to_start();
            runtime.set_tempo(120);
            runtime.set_meter(4, 4);
            runtime.set_speed(1.0);
            runtime.set_click_offset(0);
        }
        self.playback_speed = 1.0;
        self.tap_times.clear();
        self.tracks = vec![Track::new(0, ChannelLayout::Mono)];
        self.session_name = "Untitled Session".to_owned();
        self.selected_track = 0;
        self.selected_clip = None;
        self.recording_start = None;
        self.current_recording_path = None;
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.tempo = 120;
        self.tempo_map = TempoMap::constant(120.0);
        self.piano_roll.open = false;
        self.meter_numerator = 4;
        self.meter_denominator = 4;
        self.click_enabled = true;
        self.master_reference = None;
        self.chords.clear();
        self.detected_key = None;
        self.playback_synced = true;
        self.session_needs_save_as = true;
        self.dirty = true;
        self.status_message = "New session — press Ctrl+S to save".to_owned();
    }

    /// The playhead, snapped to the nearest beat unless Alt is held.
    ///
    /// Snapping is the default because most sessions here start life as an
    /// imported song, where everything already sits on the grid and an edit a
    /// few milliseconds off the beat is never what was meant. Alt gives the
    /// sample-accurate position back for material that was not played to a
    /// click.
    fn edit_frame(&self, context: &egui::Context) -> u64 {
        let frame = self.snapshot().position_frames;
        if context.input(|input| input.modifiers.alt) {
            return frame;
        }
        self.snap_to_beat(frame)
    }

    /// The nearest beat to a timeline frame.
    fn snap_to_beat(&self, frame: u64) -> u64 {
        let rate = self
            .runtime
            .as_ref()
            .map_or(48_000, |runtime| runtime.sample_rate().get());
        if rate == 0 {
            return frame;
        }
        let seconds = frame as f64 / f64::from(rate);
        let tick = self.tempo_map.seconds_to_tick(seconds);
        let per_beat = u64::from(daw_midi::TICKS_PER_QUARTER);
        let nearest = ((tick + per_beat / 2) / per_beat) * per_beat;
        let snapped = self.tempo_map.tick_to_seconds(nearest) * f64::from(rate);
        if snapped < 0.0 {
            0
        } else {
            snapped.round() as u64
        }
    }

    /// Splits every clip under the playhead, on the selected track or on all
    /// tracks when nothing is selected.
    fn split_at_playhead(&mut self, context: &egui::Context) {
        let frame = self.edit_frame(context);
        let mut removed = Vec::new();
        let mut added = Vec::new();

        for track in &mut self.tracks {
            let track_id = track.id;
            let mut produced = Vec::new();
            for clip in &mut track.clips {
                if frame <= clip.start_frame || frame >= clip.end_frame {
                    continue;
                }
                removed.push(ClipSnapshot {
                    track_id,
                    clip: clip.clone(),
                });
                // Both halves read one decoded source through the cache, so a
                // split costs no memory and no reload.
                let consumed = frame - clip.start_frame;
                let mut right = clip.clone();
                right.id = Uuid::new_v4();
                right.start_frame = frame;
                right.source_start_frame = clip.source_start_frame.saturating_add(consumed);
                clip.end_frame = frame;
                produced.push(right);
            }
            for clip in produced {
                added.push(ClipSnapshot {
                    track_id,
                    clip: clip.clone(),
                });
                track.clips.push(clip);
            }
        }

        if removed.is_empty() {
            self.status_message = "Nothing under the playhead to split".to_owned();
            return;
        }

        // The left halves changed too, so they are recorded as replacements of
        // themselves.
        for snapshot in &removed {
            if let Some(clip) = self
                .tracks
                .iter()
                .flat_map(|track| &track.clips)
                .find(|clip| clip.id == snapshot.clip.id)
            {
                added.push(ClipSnapshot {
                    track_id: snapshot.track_id,
                    clip: clip.clone(),
                });
            }
        }

        let count = removed.len();
        self.remember_edit(EditCommand::ReplaceClips {
            removed,
            added,
            label: "split",
        });
        self.selected_clip = None;
        self.dirty = true;
        self.playback_synced = false;
        if let Err(error) = self.sync_playback() {
            self.status_message = format!("Split; playback preload failed: {error}");
        } else {
            self.status_message = format!("Split {count} clip(s) at the playhead");
        }
        self.save_session();
    }

    /// Moves the edge being dragged to `frame`, live.
    ///
    /// Applied straight to the clip so the waveform follows the pointer; the
    /// undo entry is not written until the drag ends, so a whole trim is one
    /// step rather than one per frame.
    fn apply_trim(&mut self, frame: u64) {
        let Some(drag) = self.trimming.clone() else {
            return;
        };
        let Some(clip) = self
            .tracks
            .get_mut(drag.track)
            .and_then(|track| track.clips.get_mut(drag.clip))
        else {
            return;
        };

        match drag.edge {
            ClipZone::TrimStart => {
                if frame >= clip.end_frame {
                    return;
                }
                if frame < clip.start_frame {
                    // Pulling the edge back out again, but only as far as the
                    // source still has audio behind it.
                    let wanted = clip.start_frame - frame;
                    if wanted > clip.source_start_frame {
                        return;
                    }
                    clip.source_start_frame -= wanted;
                } else {
                    // The window walks forward with the edge, so the take does
                    // not slide under the pointer.
                    clip.source_start_frame += frame - clip.start_frame;
                }
                clip.start_frame = frame;
            }
            ClipZone::TrimEnd => {
                if frame <= clip.start_frame {
                    return;
                }
                clip.end_frame = frame.min(max_end_frame(
                    clip.start_frame,
                    clip.source_start_frame,
                    clip.source_frames,
                ));
            }
            ClipZone::Body => {}
        }
        self.dirty = true;
    }

    /// Ends a trim: records one undo entry for the whole drag and reloads the
    /// engine with the new window.
    fn commit_trim(&mut self) {
        let Some(drag) = self.trimming.take() else {
            return;
        };
        let Some(after) = self
            .tracks
            .get(drag.track)
            .and_then(|track| track.clips.get(drag.clip))
            .cloned()
        else {
            return;
        };
        if after.start_frame == drag.before.start_frame && after.end_frame == drag.before.end_frame
        {
            return;
        }
        let Some(track_id) = self.tracks.get(drag.track).map(|track| track.id) else {
            return;
        };

        self.remember_edit(EditCommand::ReplaceClips {
            removed: vec![ClipSnapshot {
                track_id,
                clip: drag.before,
            }],
            added: vec![ClipSnapshot {
                track_id,
                clip: after,
            }],
            label: "trim",
        });
        self.playback_synced = false;
        if let Err(error) = self.sync_playback() {
            self.status_message = format!("Trimmed; playback preload failed: {error}");
        } else {
            self.status_message = "Trimmed clip (audio file preserved)".to_owned();
        }
        self.save_session();
    }

    /// Copies the selected clip to the clipboard.
    fn copy_selected_clip(&mut self) {
        let Some((track_index, clip_index)) = self.selected_clip else {
            self.status_message = "Select a clip to copy".to_owned();
            return;
        };
        let Some(clip) = self
            .tracks
            .get(track_index)
            .and_then(|track| track.clips.get(clip_index))
        else {
            return;
        };
        self.clipboard = Some(clip.clone());
        self.status_message = format!("Copied {}", clip.name);
    }

    /// Pastes the clipboard onto the selected track at the playhead.
    fn paste_clip(&mut self, context: &egui::Context) {
        let Some(source) = self.clipboard.clone() else {
            self.status_message = "Nothing to paste".to_owned();
            return;
        };
        let frame = self.edit_frame(context);
        let Some(track) = self.tracks.get_mut(self.selected_track) else {
            return;
        };

        // A pasted clip keeps its source window, so pasting a trimmed region
        // gives back exactly that region.
        let mut clip = source;
        clip.id = Uuid::new_v4();
        let length = clip.length();
        clip.start_frame = frame;
        clip.end_frame = frame.saturating_add(length);

        let snapshot = ClipSnapshot {
            track_id: track.id,
            clip: clip.clone(),
        };
        let name = clip.name.clone();
        track.clips.push(clip);

        self.remember_edit(EditCommand::ReplaceClips {
            removed: Vec::new(),
            added: vec![snapshot],
            label: "paste",
        });
        self.dirty = true;
        self.playback_synced = false;
        if let Err(error) = self.sync_playback() {
            self.status_message = format!("Pasted; playback preload failed: {error}");
        } else {
            self.status_message = format!("Pasted {name}");
        }
        self.save_session();
    }

    /// Duplicates the selected clip immediately after itself.
    fn duplicate_selected_clip(&mut self) {
        let Some((track_index, clip_index)) = self.selected_clip else {
            self.status_message = "Select a clip to duplicate".to_owned();
            return;
        };
        let Some(track) = self.tracks.get_mut(track_index) else {
            return;
        };
        let Some(source) = track.clips.get(clip_index) else {
            return;
        };

        let mut clip = source.clone();
        let length = clip.length();
        clip.id = Uuid::new_v4();
        clip.start_frame = source.end_frame;
        clip.end_frame = source.end_frame.saturating_add(length);

        let snapshot = ClipSnapshot {
            track_id: track.id,
            clip: clip.clone(),
        };
        let name = clip.name.clone();
        track.clips.push(clip);

        self.remember_edit(EditCommand::ReplaceClips {
            removed: Vec::new(),
            added: vec![snapshot],
            label: "duplicate",
        });
        self.dirty = true;
        self.playback_synced = false;
        if let Err(error) = self.sync_playback() {
            self.status_message = format!("Duplicated; playback preload failed: {error}");
        } else {
            self.status_message = format!("Duplicated {name}");
        }
        self.save_session();
    }

    fn delete_selected_clip(&mut self) {
        let Some((track_index, clip_index)) = self.selected_clip.take() else {
            return;
        };
        let Some(track) = self.tracks.get_mut(track_index) else {
            return;
        };
        if clip_index >= track.clips.len() {
            return;
        }
        let clip = track.clips.remove(clip_index);
        let snapshot = ClipSnapshot {
            track_id: track.id,
            clip,
        };

        self.remember_edit(EditCommand::ReplaceClips {
            removed: vec![snapshot],
            added: Vec::new(),
            label: "delete",
        });
        self.dirty = true;
        self.status_message = "Clip removed (audio file preserved)".to_owned();
        self.playback_synced = false;
        if let Err(error) = self.sync_playback() {
            self.status_message = format!("Clip removed; playback preload failed: {error}");
        }
        self.save_session();
    }

    fn delete_track(&mut self, index: usize) {
        if index >= self.tracks.len() {
            return;
        }
        if self.tracks[index].monitoring {
            if let Some(runtime) = &self.runtime {
                runtime.set_monitoring(false, 0, 0);
            }
        }
        let name = self.tracks[index].name.clone();
        self.tracks.remove(index);
        if self.tracks.is_empty() {
            self.tracks.push(Track::new(0, ChannelLayout::Mono));
        }
        self.selected_track = index.min(self.tracks.len().saturating_sub(1));
        self.selected_clip = None;
        self.dirty = true;
        self.playback_synced = false;
        if let Err(error) = self.sync_playback() {
            self.status_message = format!("Track deleted; playback preload failed: {error}");
            return;
        }
        self.save_session();
        self.status_message = format!("Deleted track ‘{name}’ (audio files preserved)");
    }

    fn nudge_selected_clip(&mut self, delta_frames: i64) {
        let Some((track_index, clip_index)) = self.selected_clip else {
            return;
        };
        let Some(original) = self
            .tracks
            .get(track_index)
            .and_then(|track| track.clips.get(clip_index))
            .map(|clip| (clip.id, clip.start_frame))
        else {
            return;
        };
        let track_id = self.tracks[track_index].id;
        let Some(clip) = self
            .tracks
            .get_mut(track_index)
            .and_then(|track| track.clips.get_mut(clip_index))
        else {
            return;
        };
        let duration = clip.end_frame.saturating_sub(clip.start_frame);
        let start = if delta_frames.is_negative() {
            clip.start_frame.saturating_sub(delta_frames.unsigned_abs())
        } else {
            clip.start_frame.saturating_add(delta_frames as u64)
        };
        clip.start_frame = start;
        clip.end_frame = start.saturating_add(duration);
        self.dirty = true;
        if let Err(error) = self.sync_moved_clip(track_index, clip_index) {
            self.status_message = format!("Clip moved; audio update failed: {error}");
        } else {
            self.remember_edit(EditCommand::MoveClip {
                clip_id: original.0,
                before: ClipLocation {
                    track_id,
                    start_frame: original.1,
                },
                after: ClipLocation {
                    track_id,
                    start_frame: start,
                },
            });
            self.status_message = format!("Nudged {} samples", delta_frames.abs());
        }
    }

    fn handle_shortcuts(&mut self, context: &egui::Context) {
        if context.input(|input| {
            input.modifiers.ctrl && !input.modifiers.shift && input.key_pressed(egui::Key::Z)
        }) {
            self.undo();
        }
        if context.input(|input| {
            input.modifiers.ctrl
                && (input.key_pressed(egui::Key::Y)
                    || (input.modifiers.shift && input.key_pressed(egui::Key::Z)))
        }) {
            self.redo();
        }
        if context.input(|input| input.key_pressed(egui::Key::Space)) {
            self.toggle_play();
        }
        if context.input(|input| input.key_pressed(egui::Key::R) && !input.modifiers.ctrl) {
            if let Some(track) = self.tracks.get_mut(self.selected_track) {
                track.armed = !track.armed;
            }
        }
        // Plain C toggles the click, so the edit shortcuts take the modifier
        // forms and B — which is where Pro Tools puts separate-at-selection.
        if context.input(|input| input.key_pressed(egui::Key::C) && !input.modifiers.ctrl) {
            self.click_enabled = !self.click_enabled;
        }
        if context.input(|input| input.key_pressed(egui::Key::B) && !input.modifiers.ctrl) {
            self.split_at_playhead(context);
        }
        if context.input(|input| input.modifiers.ctrl && input.key_pressed(egui::Key::C)) {
            self.copy_selected_clip();
        }
        if context.input(|input| input.modifiers.ctrl && input.key_pressed(egui::Key::X)) {
            self.copy_selected_clip();
            self.delete_selected_clip();
        }
        if context.input(|input| input.modifiers.ctrl && input.key_pressed(egui::Key::V)) {
            self.paste_clip(context);
        }
        if context.input(|input| input.modifiers.ctrl && input.key_pressed(egui::Key::D)) {
            self.duplicate_selected_clip();
        }
        if context.input(|input| input.key_pressed(egui::Key::Home)) {
            if let Some(runtime) = &self.runtime {
                runtime.seek_to_start();
            }
        }
        // Enter returns the playhead to the start, unless a text field has focus
        // (where Enter confirms the edit instead of jumping the transport).
        if !context.wants_keyboard_input()
            && context.input(|input| input.key_pressed(egui::Key::Enter))
        {
            if let Some(runtime) = &self.runtime {
                runtime.seek_to_start();
            }
        }
        if context.input(|input| input.modifiers.ctrl && input.key_pressed(egui::Key::Space)) {
            self.toggle_record();
        }
        if context.input(|input| input.modifiers.ctrl && input.key_pressed(egui::Key::S)) {
            if context.input(|input| input.modifiers.shift) {
                self.save_session_as();
            } else {
                self.save_session();
            }
        }
        if context.input(|input| input.modifiers.ctrl && input.key_pressed(egui::Key::O)) {
            self.choose_session_to_open();
        }
        if context.input(|input| input.modifiers.ctrl && input.key_pressed(egui::Key::M)) {
            self.mixer_open = !self.mixer_open;
        }
        if context.input(|input| input.key_pressed(egui::Key::Delete)) {
            self.delete_selected_clip();
        }
        if context.input(|input| input.key_pressed(egui::Key::ArrowLeft)) {
            let amount = context.input(|input| if input.modifiers.shift { 48_000 } else { 480 });
            self.nudge_selected_clip(-amount);
        }
        if context.input(|input| input.key_pressed(egui::Key::ArrowRight)) {
            let amount = context.input(|input| if input.modifiers.shift { 48_000 } else { 480 });
            self.nudge_selected_clip(amount);
        }
    }

    /// Feeds the tuner and draws it.
    ///
    /// The tap on the input is opened only while the window is, so a closed
    /// tuner costs the audio thread nothing but an atomic load.
    fn run_tuner(&mut self, context: &egui::Context) {
        let Some(runtime) = &self.runtime else {
            self.tuner.open = false;
            return;
        };
        runtime.set_tuning(self.tuner.open);
        if !self.tuner.open {
            return;
        }
        runtime.drain_tuner(self.tuner.window_mut(), crate::tuner::WINDOW_FRAMES);
        let elapsed = self.tuner_ticked.elapsed().as_secs_f32();
        self.tuner_ticked = Instant::now();
        #[allow(clippy::cast_precision_loss)]
        let rate = runtime.sample_rate().get() as f32;
        self.tuner.analyse(rate, elapsed);
        crate::tuner::window(context, &mut self.tuner);
        // A needle has to move between input events, not only when the mouse does.
        context.request_repaint_after(std::time::Duration::from_millis(33));
    }

    fn mixer_window(&mut self, context: &egui::Context, snapshot: &RuntimeSnapshot) {
        if !self.mixer_open {
            return;
        }
        let mut open = self.mixer_open;
        let mut audibility_changed = false;
        let mut mix_changes = Vec::new();
        let mut open_fx = None;
        egui::Window::new(format!("Mix — {}", self.session_name))
            .open(&mut open)
            .default_size(Vec2::new(1_250.0, 690.0))
            .min_size(Vec2::new(720.0, 520.0))
            .show(context, |ui| {
                ui.horizontal(|ui| {
                    ui.heading(RichText::new("RUSTDAW MIX").color(theme::BLUE));
                    ui.label(RichText::new("TRACKS").small().color(theme::MUTED));
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.label(RichText::new("OUT 1–2 · 48 kHz").monospace().small());
                    });
                });
                ui.separator();
                egui::ScrollArea::horizontal()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.horizontal_top(|ui| {
                            for (index, track) in self.tracks.iter_mut().enumerate() {
                                let peak = snapshot.track_peaks.get(index).copied().unwrap_or(0.0);
                                let action = mixer_channel(ui, index, track, peak);
                                audibility_changed |= action.audibility_changed;
                                if action.mix_changed {
                                    mix_changes.push(index);
                                }
                                if action.open_fx {
                                    open_fx = Some(index);
                                }
                            }
                        });
                    });
            });
        self.mixer_open = open;
        if audibility_changed {
            self.dirty = true;
            self.sync_track_audibility();
        }
        for index in mix_changes {
            self.dirty = true;
            if let (Some(runtime), Some(track)) = (&self.runtime, self.tracks.get(index)) {
                let gain = 10.0_f32.powf(track.gain_db / 20.0);
                let _ = runtime.set_track_mix(index, gain, track.pan);
            }
        }
        if let Some(index) = open_fx {
            self.selected_track = index;
            self.inserts_open = true;
        }
    }

    fn inserts_window(&mut self, context: &egui::Context, snapshot: &RuntimeSnapshot) {
        if !self.inserts_open || self.selected_track >= self.tracks.len() {
            return;
        }
        let mut open = self.inserts_open;
        let track_index = self.selected_track;
        // Disjoint fields: the amp list is read while the track is edited.
        // Set inside the window and acted on once its borrow has ended.
        let mut rescan_amps = false;
        let mut fetch_amp = false;
        let amp_library = &self.amp_library;
        let track = &mut self.tracks[track_index];
        let before = track.effects;
        let before_nam_model = track.nam_model.clone();
        let input_peak = if track.layout == ChannelLayout::Mono {
            snapshot.input_peaks[track.input_left.min(3)]
        } else {
            snapshot.input_peaks[0].max(snapshot.input_peaks[1])
        };
        egui::Window::new(format!("Channel Strip — {}", track.name))
            .open(&mut open)
            .default_width(1_540.0)
            .default_height(390.0)
            .max_height(470.0)
            .resizable(false)
            .show(context, |ui| {
                egui::Frame::new()
                    .fill(Color32::from_rgb(45, 47, 49))
                    .stroke(Stroke::new(2.0_f32, theme::BLUE))
                    .corner_radius(4.0)
                    .inner_margin(12.0)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.heading(RichText::new("RUSTDAW").color(theme::BLUE));
                            ui.label(RichText::new("ANALOG CHANNEL STRIP").strong());
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                ui.label(RichText::new("v1 · 48 kHz").small().color(theme::MUTED));
                            });
                        });
                        ui.separator();
                        ui.horizontal_top(|ui| {
                            // Laid out as the Neural Amp Modeler plugin is:
                            // one row of controls, then the model it is
                            // playing. The tone stack belongs to the amp, not
                            // to the channel EQ beside it.
                            channel_module(
                                ui,
                                "NEURAL AMP MODELER",
                                track.effects.nam_enabled,
                                430.0,
                                |ui| {
                                    ui.horizontal(|ui| {
                                        rotary_knob(
                                            ui,
                                            "INPUT",
                                            &mut track.effects.nam_input_db,
                                            -24.0,
                                            24.0,
                                            "dB",
                                            theme::GREEN,
                                        );
                                        rotary_knob(
                                            ui,
                                            "GATE",
                                            &mut track.effects.nam_gate_db,
                                            GATE_OPEN_DB,
                                            -20.0,
                                            "dB",
                                            theme::GREEN,
                                        );
                                        rotary_knob(
                                            ui,
                                            "BASS",
                                            &mut track.effects.nam_bass,
                                            0.0,
                                            10.0,
                                            "",
                                            theme::BLUE,
                                        );
                                        rotary_knob(
                                            ui,
                                            "MIDDLE",
                                            &mut track.effects.nam_middle,
                                            0.0,
                                            10.0,
                                            "",
                                            theme::BLUE,
                                        );
                                        rotary_knob(
                                            ui,
                                            "TREBLE",
                                            &mut track.effects.nam_treble,
                                            0.0,
                                            10.0,
                                            "",
                                            theme::BLUE,
                                        );
                                        rotary_knob(
                                            ui,
                                            "OUTPUT",
                                            &mut track.effects.nam_output_db,
                                            -24.0,
                                            12.0,
                                            "dB",
                                            theme::GREEN,
                                        );
                                    });
                                    ui.horizontal(|ui| {
                                        illuminated_toggle(
                                            ui,
                                            "AMP IN",
                                            &mut track.effects.nam_enabled,
                                            theme::GREEN,
                                        );
                                        illuminated_toggle(
                                            ui,
                                            "EQ",
                                            &mut track.effects.nam_tone_enabled,
                                            theme::BLUE,
                                        );
                                        let normalize = ui.add(egui::Button::selectable(
                                            track.effects.nam_normalize,
                                            "NORMALIZE",
                                        ));
                                        if normalize.clicked() {
                                            track.effects.nam_normalize =
                                                !track.effects.nam_normalize;
                                        }
                                        normalize.on_hover_text(
                                            "Level every capture against its own measured \
                                             loudness, so swapping amps is a change of amp \
                                             rather than a change of volume",
                                        );
                                    });
                                    ui.separator();
                                    // The model slot: step through the library
                                    // with the arrows, or pick from the list.
                                    ui.horizontal(|ui| {
                                        let position =
                                            track.nam_model.as_deref().and_then(|path| {
                                                amp_library
                                                    .iter()
                                                    .position(|model| model.path == path)
                                            });
                                        let count = amp_library.len();
                                        // Wraps at both ends, so stepping
                                        // through a library is a loop rather
                                        // than something to run off the end of.
                                        let step = |forward: bool| -> Option<PathBuf> {
                                            let next = match position {
                                                None => 0,
                                                Some(index) if forward => {
                                                    (index + 1) % count.max(1)
                                                }
                                                Some(index) => (index + count - 1) % count.max(1),
                                            };
                                            amp_library.get(next).map(|model| model.path.clone())
                                        };
                                        if ui.button("‹").on_hover_text("Previous amp").clicked()
                                        {
                                            if let Some(path) = step(false) {
                                                track.nam_model = Some(path);
                                                track.effects.nam_enabled = true;
                                            }
                                        }
                                        if ui.button("›").on_hover_text("Next amp").clicked() {
                                            if let Some(path) = step(true) {
                                                track.nam_model = Some(path);
                                                track.effects.nam_enabled = true;
                                            }
                                        }
                                        let selected = track
                                            .nam_model
                                            .as_deref()
                                            .and_then(std::path::Path::file_stem)
                                            .and_then(|name| name.to_str())
                                            .unwrap_or("Select amp...")
                                            .to_owned();
                                        egui::ComboBox::from_id_salt(("nam_model", track_index))
                                            .selected_text(RichText::new(selected).small())
                                            .width(228.0)
                                            .show_ui(ui, |ui| {
                                                for model in amp_library {
                                                    let chosen = track.nam_model.as_deref()
                                                        == Some(model.path.as_path());
                                                    if ui
                                                        .selectable_label(chosen, &model.name)
                                                        .clicked()
                                                    {
                                                        track.nam_model = Some(model.path.clone());
                                                        track.effects.nam_enabled = true;
                                                    }
                                                }
                                                if amp_library.is_empty() {
                                                    ui.label(
                                                        RichText::new("No captures found")
                                                            .small()
                                                            .color(theme::MUTED),
                                                    );
                                                }
                                            });
                                        if ui.button("✕").on_hover_text("Clear").clicked() {
                                            track.nam_model = None;
                                            track.effects.nam_enabled = false;
                                        }
                                    });
                                    ui.horizontal(|ui| {
                                        let linked = daw_tone3000::publishable_key().is_some();
                                        let hint = if linked {
                                            "Pick an amp on TONE3000 and load it straight onto \
                                             this track"
                                                .to_owned()
                                        } else {
                                            format!(
                                                "Browse free amp captures on tone3000.com, then \
                                                 save the .nam files into\n{}",
                                                daw_nam::amp_dir().display()
                                            )
                                        };
                                        if ui.button("GET AMPS").on_hover_text(hint).clicked() {
                                            let _ = std::fs::create_dir_all(daw_nam::amp_dir());
                                            if linked {
                                                fetch_amp = true;
                                            } else {
                                                open_in_browser(TONE3000_URL);
                                            }
                                        }
                                        if ui
                                            .button("RESCAN")
                                            .on_hover_text(amp_library_hint())
                                            .clicked()
                                        {
                                            rescan_amps = true;
                                        }
                                        if ui
                                            .button("BROWSE")
                                            .on_hover_text("Load a capture from anywhere on disk")
                                            .clicked()
                                        {
                                            if let Some(path) = rfd::FileDialog::new()
                                                .add_filter("NAM model", &["nam"])
                                                .set_directory(daw_nam::amp_dir())
                                                .pick_file()
                                            {
                                                track.nam_model = Some(path);
                                                track.effects.nam_enabled = true;
                                            }
                                        }
                                    });
                                },
                            );

                            channel_module(
                                ui,
                                "3-BAND EQ",
                                track.effects.eq_enabled,
                                225.0,
                                |ui| {
                                    illuminated_toggle(
                                        ui,
                                        "EQ IN",
                                        &mut track.effects.eq_enabled,
                                        theme::YELLOW,
                                    );
                                    ui.add_space(8.0);
                                    ui.horizontal(|ui| {
                                        rotary_knob(
                                            ui,
                                            "HIGH",
                                            &mut track.effects.high_db,
                                            -12.0,
                                            12.0,
                                            "dB",
                                            Color32::from_rgb(188, 70, 62),
                                        );
                                        rotary_knob(
                                            ui,
                                            "MID",
                                            &mut track.effects.mid_db,
                                            -12.0,
                                            12.0,
                                            "dB",
                                            Color32::from_rgb(77, 153, 80),
                                        );
                                        rotary_knob(
                                            ui,
                                            "LOW",
                                            &mut track.effects.low_db,
                                            -12.0,
                                            12.0,
                                            "dB",
                                            Color32::from_rgb(61, 119, 184),
                                        );
                                    });
                                    ui.add_space(8.0);
                                    eq_curve_display(ui, track.effects);
                                },
                            );

                            channel_module(ui, "INPUT / OUTPUT", true, 145.0, |ui| {
                                ui.horizontal(|ui| {
                                    vertical_level_meter(ui, input_peak, "INPUT");
                                    ui.vertical_centered(|ui| {
                                        rotary_knob(
                                            ui,
                                            "TRIM",
                                            &mut track.gain_db,
                                            -60.0,
                                            12.0,
                                            "dB",
                                            theme::BLUE,
                                        );
                                        ui.add_space(8.0);
                                        ui.label(
                                            RichText::new(match track.layout {
                                                ChannelLayout::Mono => "MONO",
                                                ChannelLayout::Stereo => "STEREO",
                                            })
                                            .small()
                                            .color(theme::MUTED),
                                        );
                                    });
                                });
                            });

                            channel_module(
                                ui,
                                "DYNAMICS",
                                track.effects.compressor_enabled || track.effects.gate_enabled,
                                330.0,
                                |ui| {
                                    ui.horizontal(|ui| {
                                        illuminated_toggle(
                                            ui,
                                            "COMP IN",
                                            &mut track.effects.compressor_enabled,
                                            theme::YELLOW,
                                        );
                                        illuminated_toggle(
                                            ui,
                                            "GATE IN",
                                            &mut track.effects.gate_enabled,
                                            theme::GREEN,
                                        );
                                    });
                                    ui.separator();
                                    ui.label(
                                        RichText::new("COMPRESSOR").strong().color(theme::YELLOW),
                                    );
                                    ui.horizontal(|ui| {
                                        rotary_knob(
                                            ui,
                                            "THRESH",
                                            &mut track.effects.compressor_threshold_db,
                                            -60.0,
                                            0.0,
                                            "dB",
                                            theme::YELLOW,
                                        );
                                        rotary_knob(
                                            ui,
                                            "RATIO",
                                            &mut track.effects.compressor_ratio,
                                            1.0,
                                            20.0,
                                            ":1",
                                            theme::YELLOW,
                                        );
                                        rotary_knob(
                                            ui,
                                            "ATTACK",
                                            &mut track.effects.compressor_attack_ms,
                                            0.5,
                                            100.0,
                                            "ms",
                                            theme::YELLOW,
                                        );
                                        rotary_knob(
                                            ui,
                                            "RELEASE",
                                            &mut track.effects.compressor_release_ms,
                                            10.0,
                                            1_000.0,
                                            "ms",
                                            theme::YELLOW,
                                        );
                                    });
                                    ui.horizontal(|ui| {
                                        rotary_knob(
                                            ui,
                                            "MAKEUP",
                                            &mut track.effects.compressor_makeup_db,
                                            0.0,
                                            24.0,
                                            "dB",
                                            theme::YELLOW,
                                        );
                                        dynamics_activity(
                                            ui,
                                            input_peak,
                                            track.effects.compressor_threshold_db,
                                            track.effects.gate_threshold_db,
                                        );
                                    });
                                    ui.separator();
                                    ui.label(
                                        RichText::new("NOISE GATE").strong().color(theme::GREEN),
                                    );
                                    ui.horizontal(|ui| {
                                        rotary_knob(
                                            ui,
                                            "THRESH",
                                            &mut track.effects.gate_threshold_db,
                                            -80.0,
                                            -5.0,
                                            "dB",
                                            theme::GREEN,
                                        );
                                        rotary_knob(
                                            ui,
                                            "RELEASE",
                                            &mut track.effects.gate_release_ms,
                                            10.0,
                                            1_000.0,
                                            "ms",
                                            theme::GREEN,
                                        );
                                    });
                                },
                            );

                            // Time-based modules last in the rack, as they are
                            // last in the signal: echoes and a room around the
                            // finished tone, not around what it was before the
                            // amp and the compressor shaped it.
                            channel_module(
                                ui,
                                "DELAY / REVERB",
                                track.effects.delay_enabled || track.effects.reverb_enabled,
                                300.0,
                                |ui| {
                                    illuminated_toggle(
                                        ui,
                                        "DLY IN",
                                        &mut track.effects.delay_enabled,
                                        theme::BLUE,
                                    );
                                    ui.horizontal(|ui| {
                                        rotary_knob(
                                            ui,
                                            "TIME",
                                            &mut track.effects.delay_time_ms,
                                            20.0,
                                            MAX_DELAY_MS,
                                            "ms",
                                            theme::BLUE,
                                        );
                                        rotary_knob(
                                            ui,
                                            "FEEDBACK",
                                            &mut track.effects.delay_feedback,
                                            0.0,
                                            0.95,
                                            "",
                                            theme::BLUE,
                                        );
                                        rotary_knob(
                                            ui,
                                            "MIX",
                                            &mut track.effects.delay_mix,
                                            0.0,
                                            1.0,
                                            "",
                                            theme::BLUE,
                                        );
                                    });
                                    ui.separator();
                                    illuminated_toggle(
                                        ui,
                                        "VERB IN",
                                        &mut track.effects.reverb_enabled,
                                        theme::YELLOW,
                                    );
                                    ui.horizontal(|ui| {
                                        rotary_knob(
                                            ui,
                                            "SIZE",
                                            &mut track.effects.reverb_size,
                                            0.0,
                                            1.0,
                                            "",
                                            theme::YELLOW,
                                        );
                                        rotary_knob(
                                            ui,
                                            "DAMP",
                                            &mut track.effects.reverb_damping,
                                            0.0,
                                            1.0,
                                            "",
                                            theme::YELLOW,
                                        );
                                        rotary_knob(
                                            ui,
                                            "MIX",
                                            &mut track.effects.reverb_mix,
                                            0.0,
                                            1.0,
                                            "",
                                            theme::YELLOW,
                                        );
                                    });
                                },
                            );
                        });
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("GATE").color(theme::GREEN));
                            ui.label("→");
                            ui.label(RichText::new("AMP").color(theme::GREEN));
                            ui.label("→");
                            ui.label(RichText::new("TONE").color(theme::BLUE));
                            ui.label("→");
                            ui.label(RichText::new("EQ").color(theme::BLUE));
                            ui.label("→");
                            ui.label(RichText::new("COMPRESSOR").color(theme::YELLOW));
                            ui.label("→");
                            ui.label(RichText::new("DELAY").color(theme::BLUE));
                            ui.label("→");
                            ui.label(RichText::new("REVERB").color(theme::YELLOW));
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                ui.label(
                                    RichText::new("NON-DESTRUCTIVE · SOURCE WAV DRY")
                                        .small()
                                        .color(theme::MUTED),
                                );
                            });
                        });
                    });
            });
        self.inserts_open = open;
        if before != track.effects || before_nam_model != track.nam_model {
            self.dirty = true;
            if let Some(runtime) = &self.runtime {
                let _ = runtime.set_track_effects(track_index, channel_strip_params(track.effects));
                if before_nam_model != track.nam_model
                    || before.nam_enabled != track.effects.nam_enabled
                {
                    match runtime.set_track_nam_model(
                        track_index,
                        track
                            .nam_model
                            .as_deref()
                            .filter(|_| track.effects.nam_enabled),
                        channel_strip_params(track.effects),
                    ) {
                        Ok(()) => {
                            self.status_message = if track.effects.nam_enabled {
                                "NAM guitar amp loaded".to_owned()
                            } else {
                                "NAM guitar amp bypassed".to_owned()
                            }
                        }
                        Err(error) => {
                            track.effects.nam_enabled = false;
                            self.status_message = format!("NAM model failed: {error}");
                        }
                    }
                }
                if track.monitoring {
                    let _ = runtime.set_monitor_effects(channel_strip_params(track.effects));
                    if before_nam_model != track.nam_model
                        || before.nam_enabled != track.effects.nam_enabled
                    {
                        let _ = runtime.set_monitor_nam_model(
                            track
                                .nam_model
                                .as_deref()
                                .filter(|_| track.effects.nam_enabled),
                            channel_strip_params(track.effects),
                        );
                    }
                }
            }
        }
        if fetch_amp {
            self.start_amp_fetch();
        }
        if rescan_amps {
            self.amp_library = daw_nam::discover();
            self.status_message = match self.amp_library.len() {
                0 => format!("No amp captures in {}", daw_nam::amp_dir().display()),
                1 => "Found 1 amp capture".to_owned(),
                found => format!("Found {found} amp captures"),
            };
        }
    }

    fn transport(&mut self, context: &egui::Context, snapshot: &RuntimeSnapshot) {
        egui::TopBottomPanel::top("transport")
            .exact_height(82.0)
            .frame(
                egui::Frame::new()
                    .fill(theme::PANEL)
                    .inner_margin(10.0)
                    .stroke(Stroke::new(1.0_f32, theme::BORDER)),
            )
            .show(context, |ui| {
                ui.horizontal(|ui| {
                    ui.heading(RichText::new("RUSTDAW").color(theme::BLUE));
                    ui.label(RichText::new("EDIT").small().color(theme::MUTED));
                    if ui
                        .add(
                            egui::TextEdit::singleline(&mut self.session_name).desired_width(140.0),
                        )
                        .changed()
                    {
                        self.dirty = true;
                    }
                    ui.separator();

                    if transport_button(ui, "|◀", false).clicked() {
                        if let Some(runtime) = &self.runtime {
                            runtime.seek_to_start();
                        }
                    }
                    if transport_button(ui, "■", false).clicked() {
                        if let Some(runtime) = &self.runtime {
                            runtime.stop();
                        }
                        self.finish_recording_clip();
                    }
                    if transport_button(
                        ui,
                        "▶",
                        snapshot.transport == RuntimeTransportState::Playing,
                    )
                    .clicked()
                    {
                        self.toggle_play();
                    }
                    if transport_button(
                        ui,
                        "●",
                        matches!(
                            snapshot.transport,
                            RuntimeTransportState::Recording | RuntimeTransportState::CountIn
                        ),
                    )
                    .clicked()
                    {
                        self.toggle_record();
                    }

                    ui.add_space(12.0);
                    let seconds = snapshot.position_frames as f64 / 48_000.0;
                    let minutes = (seconds / 60.0).floor() as u64;
                    let remainder = seconds - minutes as f64 * 60.0;
                    egui::Frame::new()
                        .fill(theme::BG)
                        .stroke(Stroke::new(1.0_f32, theme::BORDER))
                        .inner_margin(egui::Margin::symmetric(14, 7))
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new(format!("{minutes:02}:{remainder:06.3}"))
                                    .monospace()
                                    .size(21.0)
                                    .color(theme::GREEN),
                            );
                        });

                    ui.add_space(12.0);
                    ui.label("TEMPO");
                    if ui
                        .add(
                            egui::DragValue::new(&mut self.tempo)
                                .range(20..=300)
                                .speed(0.25),
                        )
                        .changed()
                    {
                        if let Some(runtime) = &self.runtime {
                            runtime.set_tempo(self.tempo);
                        }
                        self.dirty = true;
                    }
                    ui.label("BPM");
                    if ui
                        .button("TAP")
                        .on_hover_text(
                            "Tap in time with the song to set the tempo. Tapping while it plays \
                             also lines the click's downbeat up with your taps.",
                        )
                        .clicked()
                    {
                        self.tap_tempo();
                    }
                    let mut meter = (self.meter_numerator, self.meter_denominator);
                    let meter_response = egui::ComboBox::from_id_salt("meter")
                        .selected_text(format!("{}/{}", meter.0, meter.1))
                        .width(54.0)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut meter, (3, 4), "3/4");
                            ui.selectable_value(&mut meter, (4, 4), "4/4");
                            ui.selectable_value(&mut meter, (6, 8), "6/8");
                        });
                    if meter_response.response.changed() {
                        (self.meter_numerator, self.meter_denominator) = meter;
                        if let Some(runtime) = &self.runtime {
                            runtime.set_meter(meter.0, meter.1);
                        }
                        self.dirty = true;
                    }
                    ui.separator();
                    // Pitch-preserving playback tempo: a time-stretch multiplier
                    // on the session tempo, shown as the tempo you actually hear.
                    ui.label("PLAY");
                    let base_bpm = f32::from(self.tempo.max(1));
                    let speed_response = ui
                        .add(
                            egui::Slider::new(&mut self.playback_speed, 0.5..=2.0)
                                .show_value(false)
                                .custom_formatter(move |value, _| {
                                    format!("{:.0} BPM", base_bpm * value as f32)
                                }),
                        )
                        .on_hover_text(
                            "Playback tempo without changing pitch. Drag to audition the song \
                             faster or slower; double-click to return to the session tempo.",
                        );
                    if speed_response.changed() {
                        if let Some(runtime) = &self.runtime {
                            runtime.set_speed(self.playback_speed);
                        }
                    }
                    if speed_response.double_clicked() {
                        self.playback_speed = 1.0;
                        if let Some(runtime) = &self.runtime {
                            runtime.set_speed(1.0);
                        }
                    }
                    ui.label(
                        RichText::new(format!("{:.0} BPM", base_bpm * self.playback_speed))
                            .small()
                            .color(theme::MUTED),
                    );
                    ui.separator();
                    // Read back from the engine rather than from this side, so
                    // what is shown is what the audio callback will actually
                    // act on.
                    let armed = self.runtime.as_ref().and_then(AudioRuntime::loop_range);
                    let rate = self
                        .runtime
                        .as_ref()
                        .map_or(48_000, |runtime| runtime.sample_rate().get());
                    let label = armed.map_or_else(
                        || "LOOP —".to_owned(),
                        |(start, end)| {
                            format!(
                                "LOOP {:.1}s–{:.1}s",
                                start as f64 / f64::from(rate),
                                end as f64 / f64::from(rate)
                            )
                        },
                    );
                    if ui
                        .selectable_label(armed.is_some(), label)
                        .on_hover_text(
                            "The loop the audio engine is holding.\nDrag the ruler to set one; \
                             click here to clear it.",
                        )
                        .clicked()
                    {
                        if let Some(runtime) = &self.runtime {
                            runtime.clear_loop();
                        }
                        self.loop_range = None;
                        self.status_message = "Loop cleared".to_owned();
                    }
                    ui.separator();
                    if ui.selectable_label(self.click_enabled, "CLICK").clicked() {
                        self.click_enabled = !self.click_enabled;
                        self.dirty = true;
                    }
                    if ui
                        .selectable_label(self.count_in_enabled, "COUNT-IN")
                        .clicked()
                    {
                        self.count_in_enabled = !self.count_in_enabled;
                    }
                    ui.add(
                        egui::Slider::new(&mut self.click_level, 0.0..=0.8)
                            .show_value(false)
                            .text("level"),
                    );
                    if let Some(runtime) = &self.runtime {
                        runtime.set_click(self.click_enabled, self.click_level);
                    }

                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let status_color = if snapshot.transport == RuntimeTransportState::Recording
                        {
                            theme::RED
                        } else if snapshot.transport == RuntimeTransportState::CountIn
                            || self.audio_error.is_some()
                        {
                            theme::YELLOW
                        } else {
                            theme::GREEN
                        };
                        ui.label(RichText::new("●").color(status_color));
                        ui.label(if self.audio_error.is_some() {
                            "AUDIO OFFLINE"
                        } else {
                            "SCARLETT · 48 kHz · 256"
                        });
                    });
                });
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!(
                            "Bars|Beats  1|1|000  ·  {}/{}",
                            self.meter_numerator, self.meter_denominator
                        ))
                        .monospace()
                        .color(theme::MUTED),
                    );
                    ui.separator();
                    ui.label(RichText::new(&self.status_message).color(theme::TEXT));
                    if snapshot.transport == RuntimeTransportState::CountIn {
                        if let Some(start) = self.recording_start {
                            let frames_per_beat = 48_000_u64.saturating_mul(60).saturating_mul(4)
                                / u64::from(self.tempo.max(1))
                                    .saturating_mul(u64::from(self.meter_denominator));
                            let remaining = start.saturating_sub(snapshot.position_frames);
                            let beats = remaining.div_ceil(frames_per_beat.max(1));
                            ui.label(
                                RichText::new(format!("COUNT-IN {beats}"))
                                    .strong()
                                    .color(theme::YELLOW),
                            );
                        }
                    }
                    if snapshot.xruns > 0 {
                        ui.label(
                            RichText::new(format!("XRUN {}", snapshot.xruns)).color(theme::RED),
                        );
                    }
                    if snapshot.dropped_record_frames > 0 {
                        ui.label(
                            RichText::new(format!("DROPPED {}", snapshot.dropped_record_frames))
                                .color(theme::RED),
                        );
                    }
                    if snapshot.monitor_amp_faults > 0 {
                        // The amp is on but not being heard; the dry signal is
                        // going through instead. Without this the only symptom
                        // is a guitar that sounds unplugged.
                        ui.label(RichText::new("AMP BYPASSED").color(theme::RED))
                            .on_hover_text(format!(
                                "The monitoring amp failed on {} block(s) and passed the dry \
                             signal through. Check the model matches the session sample rate.",
                                snapshot.monitor_amp_faults
                            ));
                    }
                    if snapshot.disk_error {
                        ui.label(RichText::new("DISK WRITE ERROR").color(theme::RED));
                    }
                });
            });
    }

    fn track_controls(&mut self, ui: &mut egui::Ui, index: usize, snapshot: &RuntimeSnapshot) {
        let selected = self.selected_track == index;
        let track = &mut self.tracks[index];
        let before = (
            track.name.clone(),
            track.input_left,
            track.armed,
            track.monitoring,
            track.muted,
            track.solo,
            track.gain_db.to_bits(),
        );
        let fill = if selected {
            Color32::from_rgb(43, 52, 59)
        } else {
            theme::PANEL
        };
        egui::Frame::new()
            .fill(fill)
            .stroke(Stroke::new(1.0_f32, theme::BORDER))
            .inner_margin(8.0)
            .show(ui, |ui| {
                ui.set_min_size(Vec2::new(HEADER_WIDTH - 18.0, TRACK_HEIGHT - 18.0));
                if ui
                    .add(egui::TextEdit::singleline(&mut track.name).desired_width(122.0))
                    .clicked()
                {
                    self.selected_track = index;
                }
                ui.horizontal(|ui| {
                    let record = ui.add(
                        egui::Button::new(RichText::new("R").color(if track.armed {
                            Color32::WHITE
                        } else {
                            theme::MUTED
                        }))
                        .fill(if track.armed {
                            theme::RED
                        } else {
                            theme::PANEL_2
                        }),
                    );
                    if record.clicked() {
                        track.armed = !track.armed;
                        self.selected_track = index;
                    }
                    // "I" said nothing about what it did. This is the only way
                    // to hear the live input — and the only way to hear an amp
                    // while playing — so it has to name itself.
                    if ui
                        .selectable_label(track.monitoring, "MON")
                        .on_hover_text(
                            "Input monitoring: hear this track's live input, through its FX \
                             and amp.\nTurn DIRECT MONITOR off on the interface first, or the \
                             dry signal is mixed in on top.",
                        )
                        .clicked()
                    {
                        track.monitoring = !track.monitoring;
                        if let Some(runtime) = &self.runtime {
                            let right = if track.layout == ChannelLayout::Mono {
                                track.input_left
                            } else {
                                track.input_right
                            };
                            runtime.set_monitoring(track.monitoring, track.input_left, right);
                            let _ =
                                runtime.set_monitor_effects(channel_strip_params(track.effects));
                            let _ = runtime.set_monitor_nam_model(
                                track
                                    .nam_model
                                    .as_deref()
                                    .filter(|_| track.effects.nam_enabled),
                                channel_strip_params(track.effects),
                            );
                        }
                    }
                    if ui.selectable_label(track.muted, "M").clicked() {
                        track.muted = !track.muted;
                    }
                    if ui.selectable_label(track.solo, "S").clicked() {
                        track.solo = !track.solo;
                    }
                    let fx_active = track.effects.eq_enabled
                        || track.effects.compressor_enabled
                        || track.effects.gate_enabled
                        || track.effects.nam_enabled;
                    if ui
                        .add(
                            egui::Button::new(RichText::new("FX").color(if fx_active {
                                theme::GREEN
                            } else {
                                theme::MUTED
                            }))
                            .fill(if fx_active {
                                theme::BLUE_DARK
                            } else {
                                theme::PANEL_2
                            }),
                        )
                        .clicked()
                    {
                        self.selected_track = index;
                        self.inserts_open = true;
                    }
                    if ui
                        .add(egui::Button::new(RichText::new("×").color(theme::RED)))
                        .on_hover_text("Delete track")
                        .clicked()
                    {
                        self.pending_delete_track = Some(index);
                    }
                    ui.label(
                        RichText::new(match track.layout {
                            ChannelLayout::Mono => "MONO",
                            ChannelLayout::Stereo => "STEREO",
                        })
                        .small()
                        .color(theme::MUTED),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("IN");
                    egui::ComboBox::from_id_salt(("input", index))
                        .selected_text(match track.layout {
                            ChannelLayout::Mono => self
                                .audio_preferences
                                .input_labels
                                .get(track.input_left)
                                .cloned()
                                .unwrap_or_else(|| format!("Input {}", track.input_left + 1)),
                            ChannelLayout::Stereo => "Inputs 1–2".to_owned(),
                        })
                        .width(104.0)
                        .show_ui(ui, |ui| {
                            if track.layout == ChannelLayout::Mono {
                                let input_count = self.runtime.as_ref().map_or(4, |runtime| {
                                    usize::from(runtime.input_channels()).min(4)
                                });
                                for channel in 0..input_count {
                                    ui.selectable_value(
                                        &mut track.input_left,
                                        channel,
                                        &self.audio_preferences.input_labels[channel],
                                    );
                                }
                            } else {
                                ui.label("Inputs 1–2");
                            }
                        });
                    ui.add(
                        egui::DragValue::new(&mut track.gain_db)
                            .range(-60.0..=12.0)
                            .suffix(" dB")
                            .speed(0.2),
                    );
                });
                let peak = if track.layout == ChannelLayout::Mono {
                    snapshot.input_peaks[track.input_left.min(3)]
                } else {
                    snapshot.input_peaks[0].max(snapshot.input_peaks[1])
                };
                meter(ui, peak);
            });
        let after = (
            track.name.clone(),
            track.input_left,
            track.armed,
            track.monitoring,
            track.muted,
            track.solo,
            track.gain_db.to_bits(),
        );
        if before.1 != after.1 && track.monitoring {
            if let Some(runtime) = &self.runtime {
                let right = if track.layout == ChannelLayout::Mono {
                    track.input_left
                } else {
                    track.input_right
                };
                runtime.set_monitoring(true, track.input_left, right);
                // The amp goes with it: changing which socket is being listened
                // to must not drop the sound the player is monitoring through.
                let _ = runtime.set_monitor_effects(channel_strip_params(track.effects));
                let _ = runtime.set_monitor_nam_model(
                    track
                        .nam_model
                        .as_deref()
                        .filter(|_| track.effects.nam_enabled),
                    channel_strip_params(track.effects),
                );
            }
        }
        if before != after {
            self.dirty = true;
        }
        let current_gain_db = track.gain_db;
        let current_pan = track.pan;
        if before.4 != after.4 || before.5 != after.5 {
            self.sync_track_audibility();
        }
        if before.6 != after.6 {
            if let Some(runtime) = &self.runtime {
                let gain = 10.0_f32.powf(current_gain_db / 20.0);
                let _ = runtime.set_track_mix(index, gain, current_pan);
            }
        }
    }

    /// The furthest point any clip reaches, in seconds — the timeline's length.
    fn timeline_length_seconds(&self, sample_rate: u32) -> f32 {
        let rate = sample_rate.max(1) as f32;
        let audio = self
            .tracks
            .iter()
            .flat_map(|track| &track.clips)
            .map(|clip| clip.end_frame as f32 / rate)
            .fold(0.0_f32, f32::max);
        let midi = self
            .tracks
            .iter()
            .flat_map(|track| &track.midi_clips)
            .map(|clip| self.tempo_map.tick_to_seconds(clip.end_tick()) as f32)
            .fold(0.0_f32, f32::max);
        audio.max(midi)
    }

    /// The timeline content width in pixels: long enough for the whole song plus
    /// a little tail to scroll into, but never narrower than the viewport.
    fn timeline_content_width(&self, viewport_width: f32, sample_rate: u32) -> f32 {
        let song = (self.timeline_length_seconds(sample_rate) + 4.0) * self.pixels_per_second;
        song.max(viewport_width).max(900.0)
    }

    fn timeline_track(&mut self, ui: &mut egui::Ui, index: usize, snapshot: &RuntimeSnapshot) {
        let sample_rate = self
            .runtime
            .as_ref()
            .map_or(48_000, |runtime| runtime.sample_rate().get());
        let width = self.timeline_content_width(ui.available_width(), sample_rate);
        let desired = Vec2::new(width, TRACK_HEIGHT);
        let (rect, track_response) = ui.allocate_exact_size(desired, Sense::click());
        let painter = ui.painter_at(rect);
        painter.rect_filled(
            rect,
            0.0,
            if index % 2 == 0 {
                theme::BG
            } else {
                Color32::from_rgb(25, 29, 32)
            },
        );
        for second in 0..=120 {
            let x = rect.left() + second as f32 * self.pixels_per_second;
            if x > rect.right() {
                break;
            }
            let strong = second % 5 == 0;
            painter.line_segment(
                [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
                Stroke::new(
                    if strong { 1.0_f32 } else { 0.5_f32 },
                    if strong {
                        theme::BORDER
                    } else {
                        Color32::from_rgb(39, 44, 48)
                    },
                ),
            );
        }
        let mut selected_request = None;
        let mut drag_start_request = None;
        let mut drag_progress_request = None;
        let mut drag_destination = None;
        let mut trim_start_request: Option<TrimDrag> = None;
        let mut trim_to_frame: Option<u64> = None;
        let mut trim_commit = false;
        let mut clip_interacted = false;
        let mut open_piano_roll = None;
        for (clip_index, clip) in self.tracks[index].midi_clips.iter().enumerate() {
            let clip_rect = draw_midi_clip(
                &painter,
                rect,
                clip,
                &self.tempo_map,
                sample_rate,
                self.pixels_per_second,
            );
            let response = ui.interact(
                clip_rect,
                ui.id().with(("midi_clip", index, clip_index)),
                Sense::click(),
            );
            if response.double_clicked() {
                open_piano_roll = Some(clip_index);
                clip_interacted = true;
            }
            response.on_hover_text("Double-click to edit in the piano roll");
        }
        if let Some(clip_index) = open_piano_roll {
            let clip = self.tracks[index].midi_clips[clip_index].clone();
            self.piano_roll.open_clip(index, clip_index, &clip);
            self.selected_track = index;
        }
        for (clip_index, clip) in self.tracks[index].clips.iter().enumerate() {
            let selected = self.selected_clip == Some((index, clip_index));
            let clip_rect = draw_clip(
                &painter,
                rect,
                clip,
                sample_rate,
                self.pixels_per_second,
                selected,
            );
            let response = ui.interact(
                clip_rect,
                ui.id().with(("clip", index, clip_index)),
                Sense::click_and_drag(),
            );

            let pointer_x = response
                .hover_pos()
                .or_else(|| response.interact_pointer_pos())
                .map(|position| position.x);
            let hover_zone = pointer_x.map_or(ClipZone::Body, |x| ClipZone::at(clip_rect, x));

            if response.clicked() {
                selected_request = Some((index, clip_index));
                clip_interacted = true;
            }
            if response.drag_started() {
                selected_request = Some((index, clip_index));
                clip_interacted = true;
                // Where the gesture began, not where the pointer has run on to.
                let press_zone = ui
                    .ctx()
                    .input(|input| input.pointer.press_origin())
                    .map_or(hover_zone, |origin| ClipZone::at(clip_rect, origin.x));
                if press_zone == ClipZone::Body {
                    drag_start_request = Some((index, clip_index, clip.start_frame, Vec2::ZERO));
                } else {
                    trim_start_request = Some(TrimDrag {
                        track: index,
                        clip: clip_index,
                        edge: press_zone,
                        before: clip.clone(),
                    });
                }
            }

            let trimming_this = self
                .trimming
                .as_ref()
                .or(trim_start_request.as_ref())
                .is_some_and(|drag| drag.targets(index, clip_index));

            // The cursor shows the latched tool while a gesture is running, so
            // it does not flip to the grab hand as the pointer leaves the edge.
            if response.hovered() || trimming_this {
                let showing = if trimming_this {
                    self.trimming.as_ref().map_or(hover_zone, |drag| drag.edge)
                } else {
                    hover_zone
                };
                ui.ctx().set_cursor_icon(showing.cursor());
            }

            if trimming_this && response.dragged() {
                clip_interacted = true;
                // The edge follows the pointer rather than a delta, so a trim
                // cannot drift away from the cursor over a long drag.
                if let Some(pointer) = response.interact_pointer_pos() {
                    let seconds = self.track_seconds_at(rect.left(), pointer.x);
                    let raw = (seconds * f64::from(sample_rate)) as u64;
                    let snapped = if ui.ctx().input(|input| input.modifiers.alt) {
                        raw
                    } else {
                        self.snap_to_beat(raw)
                    };
                    trim_to_frame = Some(snapped);
                }
            }
            if trimming_this && response.drag_stopped() {
                trim_commit = true;
                clip_interacted = true;
            }
            if response.hovered() && !trimming_this {
                response.clone().on_hover_text(
                    "Drag the middle to move, or between tracks.\nDrag either edge to trim \
                     it.\nEdits snap to the beat; hold Alt for sample-accurate placement.",
                );
            }
            if !trimming_this && response.dragged() {
                clip_interacted = true;
                let total_delta = ui
                    .ctx()
                    .input(|input| input.pointer.total_drag_delta())
                    .unwrap_or_else(|| {
                        self.dragged_clip
                            .filter(|(track, item, _, _)| *track == index && *item == clip_index)
                            .map_or(response.drag_delta(), |(_, _, _, accumulated)| {
                                accumulated + response.drag_delta()
                            })
                    });
                drag_progress_request = Some((index, clip_index, clip.start_frame, total_delta));
                let ghost = clip_rect.translate(total_delta);
                ui.ctx().layer_painter(ui.layer_id()).rect_stroke(
                    ghost,
                    3.0,
                    Stroke::new(2.0_f32, theme::YELLOW),
                    StrokeKind::Inside,
                );
            }
            if !trimming_this && response.drag_stopped() {
                let origin = self
                    .dragged_clip
                    .filter(|(track, item, _, _)| *track == index && *item == clip_index)
                    .map_or(clip.start_frame, |(_, _, start, _)| start);
                let total_delta = self
                    .dragged_clip
                    .filter(|(track, item, _, _)| *track == index && *item == clip_index)
                    .map_or(Vec2::ZERO, |(_, _, _, accumulated)| accumulated);
                let frame_delta =
                    (total_delta.x / self.pixels_per_second * sample_rate as f32) as i64;
                let new_start = moved_start_frame(origin, frame_delta);
                let target = response.interact_pointer_pos().map_or(index, |pointer| {
                    let row_delta = ((pointer.y - rect.top()) / TRACK_HEIGHT).floor() as i64;
                    let rows = usize::try_from(row_delta.unsigned_abs()).unwrap_or(usize::MAX);
                    if row_delta.is_negative() {
                        index.saturating_sub(rows)
                    } else {
                        index
                            .saturating_add(rows)
                            .min(self.tracks.len().saturating_sub(1))
                    }
                });
                drag_destination = Some((clip_index, target, new_start));
                clip_interacted = true;
            }
        }
        if let Some(selection) = selected_request {
            self.selected_track = index;
            self.selected_clip = Some(selection);
        }
        if let Some(drag) = trim_start_request {
            self.status_message = format!("Trimming the {} edge…", drag.edge.describe());
            self.trimming = Some(drag);
        }
        if let Some(frame) = trim_to_frame {
            self.apply_trim(frame);
        }
        if trim_commit {
            self.commit_trim();
        }
        if let Some(drag) = drag_start_request {
            self.dragged_clip = Some(drag);
        }
        if let Some(progress) = drag_progress_request {
            self.dragged_clip = Some(progress);
        }
        if let Some((clip_index, target_track, new_start)) = drag_destination {
            self.dragged_clip = None;
            let edit_before = self.tracks[index].clips.get(clip_index).map(|clip| {
                (
                    clip.id,
                    ClipLocation {
                        track_id: self.tracks[index].id,
                        start_frame: clip.start_frame,
                    },
                )
            });
            if let Some(clip) = self.tracks[index].clips.get_mut(clip_index) {
                let duration = clip.end_frame.saturating_sub(clip.start_frame);
                clip.start_frame = new_start;
                clip.end_frame = new_start.saturating_add(duration);
                self.dirty = true;
            }
            let mut moved_clip = (index, clip_index);
            if target_track != index && clip_index < self.tracks[index].clips.len() {
                let clip = self.tracks[index].clips.remove(clip_index);
                let target_clip = self.tracks[target_track].clips.len();
                self.tracks[target_track].clips.push(clip);
                self.selected_track = target_track;
                self.selected_clip = Some((target_track, target_clip));
                moved_clip = (target_track, target_clip);
                self.status_message = format!("Moved clip to {}", self.tracks[target_track].name);
            }
            if let Err(error) = self.sync_moved_clip(moved_clip.0, moved_clip.1) {
                self.status_message = format!("Clip moved; audio update failed: {error}");
            }
            if let Some((clip_id, before)) = edit_before {
                let after = ClipLocation {
                    track_id: self.tracks[moved_clip.0].id,
                    start_frame: new_start,
                };
                if before.track_id != after.track_id || before.start_frame != after.start_frame {
                    self.remember_edit(EditCommand::MoveClip {
                        clip_id,
                        before,
                        after,
                    });
                }
            }
            self.save_session();
        }
        if track_response.clicked() && !clip_interacted {
            if let Some(pointer) = track_response.interact_pointer_pos() {
                let seconds = ((pointer.x - rect.left()) / self.pixels_per_second).max(0.0);
                let frame = (seconds * sample_rate as f32) as u64;
                if let Some(runtime) = &self.runtime {
                    runtime.seek(SamplePosition::new(frame));
                }
                self.selected_track = index;
                self.selected_clip = None;
            }
        }
        if snapshot.transport == RuntimeTransportState::Recording && self.selected_track == index {
            if let Some(start) = self.recording_start {
                let clip = Clip {
                    id: Uuid::nil(),
                    name: "Recording…".to_owned(),
                    path: PathBuf::new(),
                    start_frame: start,
                    end_frame: snapshot.position_frames.max(start + 1),
                    source_start_frame: 0,
                    source_path: None,
                    source_frames: 0,
                    color: Color32::from_rgb(92, 42, 45),
                    waveform: Vec::new(),
                };
                draw_clip(
                    &painter,
                    rect,
                    &clip,
                    sample_rate,
                    self.pixels_per_second,
                    false,
                );
            }
        }
        let playhead_x = rect.left()
            + snapshot.position_frames as f32 / sample_rate as f32 * self.pixels_per_second;
        painter.line_segment(
            [
                Pos2::new(playhead_x, rect.top()),
                Pos2::new(playhead_x, rect.bottom()),
            ],
            Stroke::new(1.5_f32, theme::RED),
        );
    }
}

impl eframe::App for RustDawApp {
    fn update(&mut self, context: &egui::Context, _frame: &mut eframe::Frame) {
        context.request_repaint_after(Duration::from_millis(33));

        // The chord-lane size is written once the wheel comes to rest, so a
        // long adjustment is one file write rather than one per notch.
        if self.chord_scale_unsaved
            && context.input(|input| input.smooth_scroll_delta.y.abs() < 0.1)
        {
            self.chord_scale_unsaved = false;
            if let Err(error) = save_audio_preferences(&self.audio_preferences) {
                self.status_message = format!("Chord chart size was not saved: {error}");
            }
        }
        let dropped_audio = context.input(|input| {
            input
                .raw
                .dropped_files
                .iter()
                .filter_map(|file| file.path.clone())
                .collect::<Vec<_>>()
        });
        if !dropped_audio.is_empty() {
            self.import_audio_files(dropped_audio);
        }
        if self.last_disk_check.elapsed() >= Duration::from_secs(2) {
            self.free_disk_bytes = disk_free_bytes(&recording_directory()).ok();
            self.last_disk_check = Instant::now();
        }
        self.handle_shortcuts(context);
        let snapshot = self.snapshot();
        if snapshot.transport == RuntimeTransportState::Recording && self.recording_start.is_some()
        {
            self.recording_began = true;
        }
        self.transport(context, &snapshot);

        egui::TopBottomPanel::bottom("status")
            .exact_height(27.0)
            .frame(
                egui::Frame::new()
                    .fill(theme::PANEL_2)
                    .inner_margin(egui::Margin::symmetric(8, 4)),
            )
            .show(context, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!(
                            "{} track{}",
                            self.tracks.len(),
                            if self.tracks.len() == 1 { "" } else { "s" }
                        ))
                        .small(),
                    );
                    if let Some(bytes) = self.free_disk_bytes {
                        ui.separator();
                        ui.label(
                            RichText::new(format!("Recordings · {} free", format_bytes(bytes)))
                                .small()
                                .color(theme::MUTED),
                        );
                    }
                    ui.separator();
                    ui.label(RichText::new("ZOOM").small().color(theme::MUTED));
                    ui.add(
                        egui::Slider::new(&mut self.pixels_per_second, 20.0..=320.0)
                            .show_value(false)
                            .logarithmic(true)
                            .custom_formatter(|value, _| format!("{value:.0} px/s")),
                    );
                    ui.checkbox(&mut self.follow_playhead, "FOLLOW");
                    ui.separator();
                    if ui
                        .add_enabled(!self.undo_stack.is_empty(), egui::Button::new("UNDO"))
                        .on_hover_text("Undo last clip move (Ctrl+Z)")
                        .clicked()
                    {
                        self.undo();
                    }
                    if ui
                        .add_enabled(!self.redo_stack.is_empty(), egui::Button::new("REDO"))
                        .on_hover_text("Redo clip move (Ctrl+Shift+Z)")
                        .clicked()
                    {
                        self.redo();
                    }
                    ui.separator();
                    if ui.button("NEW").clicked() {
                        if self.dirty {
                            self.confirm_new_session = true;
                        } else {
                            self.new_session();
                        }
                    }
                    if ui
                        .button(if self.dirty { "SAVE *" } else { "SAVE" })
                        .clicked()
                    {
                        self.save_session();
                    }
                    if ui.button("SAVE AS").clicked() {
                        self.save_session_as();
                    }
                    if ui.button("OPEN SESSION").clicked() {
                        self.choose_session_to_open();
                    }
                    if ui.button("IMPORT AUDIO").clicked() {
                        self.choose_audio_to_import();
                    }
                    if ui
                        .button("IMPORT SONG")
                        .on_hover_text("Separate a song into instrument tracks to play along with")
                        .clicked()
                    {
                        self.open_song_import(context);
                    }
                    let key_label = if self.session_transpose == 0 {
                        "SONG KEY".to_owned()
                    } else {
                        format!("SONG KEY ({:+})", self.session_transpose)
                    };
                    if ui
                        .add_enabled(
                            self.tracks.iter().any(|track| !track.clips.is_empty()),
                            egui::Button::new(key_label),
                        )
                        .on_hover_text(
                            "Move the loaded song into another key to rehearse in, without \
                             importing it again",
                        )
                        .clicked()
                    {
                        self.transpose_wanted = self.session_transpose;
                        self.transpose_open = true;
                    }
                    if ui
                        .add_enabled(
                            self.tracks.iter().any(|track| !track.midi_clips.is_empty()),
                            egui::Button::new("PIANO ROLL"),
                        )
                        .on_hover_text("Edit the notes of an instrument track")
                        .clicked()
                    {
                        self.open_first_midi_clip();
                    }
                    if ui.button("MIX").clicked() {
                        self.mixer_open = true;
                    }
                    if ui
                        .selectable_label(self.tuner.open, "TUNER")
                        .on_hover_text(
                            "Tune the instrument on the armed track's input.\nListens whether \
                             or not the transport is running.",
                        )
                        .clicked()
                    {
                        self.tuner.open = !self.tuner.open;
                    }
                    if ui
                        .button("EXPORT MIX")
                        .on_hover_text("Render the session to one stereo WAV")
                        .clicked()
                    {
                        self.export_mix();
                    }
                    if ui
                        .button("EXPORT STEMS")
                        .on_hover_text(
                            "Render every audible track to its own WAV, all the same length so \
                             they line up. They add back up to the mix.",
                        )
                        .clicked()
                    {
                        self.export_stems();
                    }
                    // The mastering reference sits next to EXPORT MIX because
                    // that is the only thing it affects: it is a property of
                    // the bounce, not of the mix.
                    let reference_name = self.master_reference.as_ref().map(|path| {
                        path.file_stem().map_or_else(
                            || path.display().to_string(),
                            |stem| stem.to_string_lossy().into_owned(),
                        )
                    });
                    let (label, hint) = match &reference_name {
                        Some(name) => (
                            format!("MASTER: {name}"),
                            "The exported mix is matched to this record's loudness, tone and \
                             stereo width.\nClick to choose another; right-click to export dry."
                                .to_owned(),
                        ),
                        None => (
                            "MASTER…".to_owned(),
                            "Match the exported mix to a reference record — a track you want \
                             yours to sound like.\nThe reference must be a WAV at the session \
                             sample rate."
                                .to_owned(),
                        ),
                    };
                    let master_button = ui
                        .selectable_label(reference_name.is_some(), label)
                        .on_hover_text(hint);
                    if master_button.clicked() {
                        self.choose_master_reference();
                    }
                    if master_button.secondary_clicked() && self.master_reference.is_some() {
                        self.master_reference = None;
                        self.dirty = true;
                        self.status_message = "Mastering off — exports the mix as mixed".to_owned();
                    }
                    if ui.button("AUDIO SETTINGS").clicked() {
                        self.audio_settings_open = true;
                    }
                    ui.separator();
                    ui.label(
                        RichText::new("48 kHz / 24-bit WAV")
                            .small()
                            .color(theme::MUTED),
                    );
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.label(
                            RichText::new("RustDAW Recording MVP")
                                .small()
                                .color(theme::MUTED),
                        );
                    });
                });
            });

        egui::SidePanel::left("track_headers")
            .exact_width(HEADER_WIDTH)
            .resizable(false)
            .frame(
                egui::Frame::new()
                    .fill(theme::PANEL)
                    .stroke(Stroke::new(1.0_f32, theme::BORDER)),
            )
            .show(context, |ui| {
                ui.horizontal(|ui| {
                    ui.menu_button("+ AUDIO TRACK", |ui| {
                        if ui.button("Mono audio track").clicked() {
                            self.tracks
                                .push(Track::new(self.tracks.len(), ChannelLayout::Mono));
                            self.dirty = true;
                            ui.close();
                        }
                        if ui.button("Stereo audio track").clicked() {
                            self.tracks
                                .push(Track::new(self.tracks.len(), ChannelLayout::Stereo));
                            self.dirty = true;
                            ui.close();
                        }
                    });
                    ui.label(RichText::new("TRACKS").small().color(theme::MUTED));
                });
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for index in 0..self.tracks.len() {
                        self.track_controls(ui, index, &snapshot);
                    }
                });
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(theme::BG))
            .show(context, |ui| {
                let ruler_height = 30.0;
                let (ruler, ruler_response) = ui.allocate_exact_size(
                    Vec2::new(ui.available_width(), ruler_height),
                    Sense::click_and_drag(),
                );
                let ruler_rate = self
                    .runtime
                    .as_ref()
                    .map_or(48_000, |runtime| runtime.sample_rate().get());
                self.timeline_ruler(ui, ruler, &ruler_response, ruler_rate);
                self.chord_lane(ui, ruler.left(), ruler.width());

                // While the transport runs, keep the playhead in view by
                // scrolling the timeline so it sits near the centre. The user
                // can turn this off, or it yields as soon as playback stops.
                let is_running = matches!(
                    snapshot.transport,
                    RuntimeTransportState::Playing
                        | RuntimeTransportState::Recording
                        | RuntimeTransportState::CountIn
                );
                let mut timeline = egui::ScrollArea::both().auto_shrink([false, false]);
                if self.follow_playhead && is_running {
                    let sample_rate = self
                        .runtime
                        .as_ref()
                        .map_or(48_000, |runtime| runtime.sample_rate().get());
                    let playhead_x = snapshot.position_frames as f32 / sample_rate as f32
                        * self.pixels_per_second;
                    let target = (playhead_x - ui.available_width() * 0.5).max(0.0);
                    timeline = timeline.horizontal_scroll_offset(target);
                }
                let output = timeline.show(ui, |ui| {
                    for index in 0..self.tracks.len() {
                        self.timeline_track(ui, index, &snapshot);
                    }
                });
                // Mirror the actual (clamped) offset so the ruler tracks it next
                // frame, whether the scroll came from following or the user.
                self.timeline_scroll_x = output.state.offset.x;
            });

        if let Some(error) = &self.audio_error {
            egui::Window::new("Audio engine offline").anchor(Align2::CENTER_TOP, [0.0, 105.0]).collapsible(false).resizable(false).show(context, |ui| {
                ui.label("The editor is available, but the Scarlett stream could not open.");
                ui.colored_label(theme::YELLOW, error);
                ui.label("Close other exclusive audio applications or reconnect the interface, then restart RustDAW.");
            });
        }

        if let Some(report) = &self.export_report {
            let mut dismissed = false;
            let mut reveal = None;
            egui::Window::new(&report.heading)
                .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
                .collapsible(false)
                .resizable(false)
                .show(context, |ui| {
                    ui.label(&report.detail);
                    if let Some(path) = &report.path {
                        ui.label(
                            RichText::new(path.display().to_string())
                                .small()
                                .color(theme::MUTED),
                        );
                    }
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        if ui.button("OK").clicked() {
                            dismissed = true;
                        }
                        if let Some(directory) = report.path.as_deref().and_then(|p| p.parent())
                            && ui.button("Show in folder").clicked()
                        {
                            reveal = Some(directory.to_owned());
                            dismissed = true;
                        }
                    });
                });
            if let Some(directory) = reveal {
                reveal_in_file_manager(&directory);
            }
            if dismissed {
                self.export_report = None;
            }
        }
        if self.confirm_new_session {
            egui::Window::new("Start a new session?")
                .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
                .collapsible(false)
                .resizable(false)
                .show(context, |ui| {
                    ui.label("Unsaved edit changes will be replaced.");
                    ui.label("Recorded WAV files will not be deleted.");
                    ui.horizontal(|ui| {
                        if ui.button("Cancel").clicked() {
                            self.confirm_new_session = false;
                        }
                        if ui.button("Save and create").clicked() {
                            self.save_session();
                            self.new_session();
                            self.confirm_new_session = false;
                        }
                        if ui.button("Discard edits").clicked() {
                            self.new_session();
                            self.confirm_new_session = false;
                        }
                    });
                });
        }
        if self.confirm_open_session {
            egui::Window::new("Open another session?")
                .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
                .collapsible(false)
                .resizable(false)
                .show(context, |ui| {
                    ui.label("The current session has unsaved changes.");
                    ui.horizontal(|ui| {
                        if ui.button("Cancel").clicked() {
                            self.pending_open_session = None;
                            self.confirm_open_session = false;
                        }
                        if ui.button("Save and open").clicked() {
                            self.save_session();
                            if !self.dirty {
                                if let Some(path) = self.pending_open_session.take() {
                                    self.open_session(path);
                                }
                                self.confirm_open_session = false;
                            }
                        }
                        if ui.button("Discard changes and open").clicked() {
                            if let Some(path) = self.pending_open_session.take() {
                                self.open_session(path);
                            }
                            self.confirm_open_session = false;
                        }
                    });
                });
        }
        if let Some(index) = self.pending_delete_track {
            let name = self
                .tracks
                .get(index)
                .map_or("Track", |track| track.name.as_str())
                .to_owned();
            egui::Window::new("Delete audio track?")
                .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
                .collapsible(false)
                .resizable(false)
                .show(context, |ui| {
                    ui.label(format!(
                        "Remove ‘{name}’ and all its clips from this session?"
                    ));
                    ui.label(
                        RichText::new("The source WAV files will not be deleted.")
                            .color(theme::GREEN),
                    );
                    ui.horizontal(|ui| {
                        if ui.button("Cancel").clicked() {
                            self.pending_delete_track = None;
                        }
                        if ui
                            .add(egui::Button::new("Delete Track").fill(theme::RED))
                            .clicked()
                        {
                            self.pending_delete_track = None;
                            self.delete_track(index);
                        }
                    });
                });
        }
        if context.input(|input| !input.raw.hovered_files.is_empty()) {
            egui::Area::new(egui::Id::new("audio_drop_overlay"))
                .order(egui::Order::Foreground)
                .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
                .show(context, |ui| {
                    egui::Frame::new()
                        .fill(Color32::from_rgba_unmultiplied(20, 66, 86, 245))
                        .stroke(Stroke::new(2.0_f32, theme::BLUE))
                        .corner_radius(8.0)
                        .inner_margin(24.0)
                        .show(ui, |ui| {
                            ui.heading(RichText::new("DROP AUDIO TO IMPORT").color(Color32::WHITE));
                            ui.label(
                                "Each file becomes a new track at the playhead — MP3, WAV, FLAC \
                                 and anything else ffmpeg reads.",
                            );
                        });
                });
        }
        self.audio_settings(context, &snapshot);
        self.mixer_window(context, &snapshot);
        self.inserts_window(context, &snapshot);
        self.run_tuner(context);
        self.poll_song_import(context);
        self.poll_rekey();
        self.poll_amp_fetch();
        self.song_import_window(context);
        self.transpose_window(context);
        self.piano_roll_window(context);
    }
}

#[derive(Default)]
struct MixerChannelAction {
    audibility_changed: bool,
    mix_changed: bool,
    open_fx: bool,
}

#[allow(clippy::too_many_lines)]
fn mixer_channel(
    ui: &mut egui::Ui,
    index: usize,
    track: &mut Track,
    peak: f32,
) -> MixerChannelAction {
    let before_mute = track.muted;
    let before_solo = track.solo;
    let before_gain = track.gain_db.to_bits();
    let before_pan = track.pan.to_bits();
    let mut open_fx = false;
    egui::Frame::new()
        .fill(Color32::from_rgb(37, 40, 43))
        .stroke(Stroke::new(1.0_f32, theme::BORDER))
        .inner_margin(7.0)
        .show(ui, |ui| {
            ui.set_min_size(Vec2::new(138.0, 585.0));
            ui.set_max_width(138.0);
            ui.with_layout(Layout::top_down(Align::Min), |ui| {
                ui.label(
                    RichText::new(format!("{:02}  {}", index + 1, track.name))
                        .strong()
                        .color(theme::TEXT),
                );
                ui.label(
                    RichText::new(match track.layout {
                        ChannelLayout::Mono => "MONO AUDIO",
                        ChannelLayout::Stereo => "STEREO AUDIO",
                    })
                    .monospace()
                    .small()
                    .color(theme::MUTED),
                );
                ui.separator();
                ui.label(RichText::new("INSERTS A–E").monospace().small());
                let inserts = [
                    ("A  EQ", track.effects.eq_enabled, theme::BLUE),
                    ("B  COMP", track.effects.compressor_enabled, theme::YELLOW),
                    ("C  GATE", track.effects.gate_enabled, theme::GREEN),
                ];
                for (label, active, color) in inserts {
                    if ui
                        .add_sized(
                            [124.0, 20.0],
                            egui::Button::new(
                                RichText::new(label).monospace().small().color(if active {
                                    Color32::WHITE
                                } else {
                                    theme::MUTED
                                }),
                            )
                            .fill(if active {
                                color.gamma_multiply(0.45)
                            } else {
                                theme::PANEL
                            }),
                        )
                        .clicked()
                    {
                        open_fx = true;
                    }
                }
                for label in ["D  —", "E  —"] {
                    ui.add_enabled(
                        false,
                        egui::Button::new(label).min_size(Vec2::new(124.0, 18.0)),
                    );
                }
                ui.add_space(4.0);
                ui.label(RichText::new("SENDS A–E").monospace().small());
                for label in ["A  —", "B  —", "C  —"] {
                    ui.add_enabled(
                        false,
                        egui::Button::new(label).min_size(Vec2::new(124.0, 17.0)),
                    );
                }
                ui.separator();
                ui.label(RichText::new("I / O").monospace().small());
                ui.label(
                    RichText::new(match track.layout {
                        ChannelLayout::Mono => format!("Input {}", track.input_left + 1),
                        ChannelLayout::Stereo => "Inputs 1–2".to_owned(),
                    })
                    .small(),
                );
                ui.label(RichText::new("Output 1–2").small().color(theme::GREEN));
                ui.separator();
                ui.horizontal(|ui| {
                    ui.add_space(27.0);
                    rotary_knob(ui, "PAN", &mut track.pan, -1.0, 1.0, "", theme::BLUE);
                });
                ui.horizontal(|ui| {
                    mixer_toggle(ui, "S", &mut track.solo, theme::YELLOW);
                    mixer_toggle(ui, "M", &mut track.muted, theme::RED);
                });
                ui.horizontal_top(|ui| {
                    ui.vertical_centered(|ui| {
                        ui.label(
                            RichText::new("FADER")
                                .monospace()
                                .small()
                                .color(theme::MUTED),
                        );
                        ui.add_sized(
                            [30.0, 220.0],
                            egui::Slider::new(&mut track.gain_db, -60.0..=12.0)
                                .vertical()
                                .show_value(false),
                        );
                    });
                    mixer_meter(ui, peak);
                });
                ui.label(
                    RichText::new(format!("{:+.1} dB", track.gain_db))
                        .monospace()
                        .color(theme::GREEN),
                );
            });
        });
    MixerChannelAction {
        audibility_changed: before_mute != track.muted || before_solo != track.solo,
        mix_changed: before_gain != track.gain_db.to_bits() || before_pan != track.pan.to_bits(),
        open_fx,
    }
}

fn mixer_toggle(ui: &mut egui::Ui, label: &str, value: &mut bool, color: Color32) {
    if ui
        .add(
            egui::Button::new(RichText::new(label).strong().color(if *value {
                Color32::WHITE
            } else {
                theme::MUTED
            }))
            .fill(if *value { color } else { theme::PANEL_2 })
            .min_size(Vec2::new(55.0, 24.0)),
        )
        .clicked()
    {
        *value = !*value;
    }
}

fn mixer_meter(ui: &mut egui::Ui, peak: f32) {
    ui.vertical_centered(|ui| {
        ui.label(
            RichText::new("METER")
                .monospace()
                .small()
                .color(theme::MUTED),
        );
        let (rect, _) = ui.allocate_exact_size(Vec2::new(24.0, 220.0), Sense::hover());
        ui.painter()
            .rect_filled(rect, 1.0, Color32::from_rgb(15, 17, 18));
        let level = peak.clamp(0.0, 1.0).sqrt();
        for segment in 0..28 {
            let fraction = segment as f32 / 28.0;
            let segment_rect = Rect::from_min_max(
                Pos2::new(
                    rect.left() + 3.0,
                    rect.bottom() - (segment + 1) as f32 * 7.6,
                ),
                Pos2::new(
                    rect.right() - 3.0,
                    rect.bottom() - segment as f32 * 7.6 - 2.0,
                ),
            );
            let color = if fraction > 0.9 {
                theme::RED
            } else if fraction > 0.72 {
                theme::YELLOW
            } else {
                theme::GREEN
            };
            ui.painter().rect_filled(
                segment_rect,
                0.5,
                if fraction < level {
                    color
                } else {
                    color.gamma_multiply(0.13)
                },
            );
        }
    });
}

fn channel_module(
    ui: &mut egui::Ui,
    title: &str,
    active: bool,
    width: f32,
    controls: impl FnOnce(&mut egui::Ui),
) {
    egui::Frame::new()
        .fill(Color32::from_rgb(52, 55, 58))
        .stroke(Stroke::new(
            1.0_f32,
            if active { theme::BLUE } else { theme::BORDER },
        ))
        .corner_radius(3.0)
        .inner_margin(10.0)
        .show(ui, |ui| {
            ui.set_min_width(width);
            ui.set_max_width(width);
            ui.with_layout(Layout::top_down(Align::Min), |ui| {
                ui.label(RichText::new(title).monospace().strong().color(if active {
                    theme::TEXT
                } else {
                    theme::MUTED
                }));
                ui.separator();
                controls(ui);
            });
        });
}

fn illuminated_toggle(ui: &mut egui::Ui, label: &str, enabled: &mut bool, color: Color32) {
    let response = ui.add(
        egui::Button::new(RichText::new(label).monospace().color(if *enabled {
            Color32::BLACK
        } else {
            theme::MUTED
        }))
        .fill(if *enabled { color } else { theme::PANEL })
        .stroke(Stroke::new(
            1.0_f32,
            if *enabled { color } else { theme::BORDER },
        )),
    );
    if response.clicked() {
        *enabled = !*enabled;
    }
}

#[allow(clippy::too_many_arguments)]
fn rotary_knob(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    minimum: f32,
    maximum: f32,
    unit: &str,
    color: Color32,
) {
    let desired = Vec2::new(72.0, 92.0);
    let (rect, response) = ui.allocate_exact_size(desired, Sense::click_and_drag());
    if response.dragged() {
        let delta = ui.input(|input| input.pointer.delta());
        let fine = ui.input(|input| input.modifiers.shift);
        let speed = (maximum - minimum) / if fine { 1_200.0 } else { 280.0 };
        *value = (*value + (delta.x - delta.y) * speed).clamp(minimum, maximum);
        ui.ctx().request_repaint();
    }
    let painter = ui.painter_at(rect);
    let center = Pos2::new(rect.center().x, rect.top() + 39.0);
    let radius = 24.0;
    painter.circle_filled(center + Vec2::new(1.5, 2.0), radius + 2.0, Color32::BLACK);
    painter.circle_filled(center, radius, Color32::from_rgb(70, 72, 74));
    painter.circle_stroke(
        center,
        radius,
        Stroke::new(2.0_f32, Color32::from_rgb(25, 26, 27)),
    );
    let fraction = ((*value - minimum) / (maximum - minimum)).clamp(0.0, 1.0);
    let angle = -std::f32::consts::PI * 0.75 + fraction * std::f32::consts::PI * 1.5;
    let direction = Vec2::new(angle.cos(), angle.sin());
    painter.line_segment(
        [center + direction * 7.0, center + direction * 20.0],
        Stroke::new(3.0_f32, color),
    );
    for tick in 0..=10 {
        let tick_fraction = tick as f32 / 10.0;
        let tick_angle = -std::f32::consts::PI * 0.75 + tick_fraction * std::f32::consts::PI * 1.5;
        let tick_direction = Vec2::new(tick_angle.cos(), tick_angle.sin());
        painter.line_segment(
            [
                center + tick_direction * 29.0,
                center + tick_direction * 32.0,
            ],
            Stroke::new(1.0_f32, theme::MUTED),
        );
    }
    painter.text(
        Pos2::new(rect.center().x, rect.top() + 2.0),
        Align2::CENTER_TOP,
        label,
        FontId::monospace(11.0),
        theme::MUTED,
    );
    let decimals = usize::from((maximum - minimum) < 100.0);
    painter.text(
        Pos2::new(rect.center().x, rect.bottom() - 3.0),
        Align2::CENTER_BOTTOM,
        format!("{:.*} {unit}", decimals, *value),
        FontId::monospace(11.0),
        theme::TEXT,
    );
    response.on_hover_text("Drag vertically or horizontally · hold Shift for fine adjustment");
}

fn vertical_level_meter(ui: &mut egui::Ui, peak: f32, label: &str) {
    ui.vertical_centered(|ui| {
        ui.label(RichText::new(label).monospace().small().color(theme::MUTED));
        let (rect, _) = ui.allocate_exact_size(Vec2::new(28.0, 210.0), Sense::hover());
        ui.painter()
            .rect_filled(rect, 2.0, Color32::from_rgb(18, 20, 21));
        let level = peak.clamp(0.0, 1.0).sqrt();
        for segment in 0..24 {
            let bottom_fraction = segment as f32 / 24.0;
            let segment_rect = Rect::from_min_max(
                Pos2::new(
                    rect.left() + 4.0,
                    rect.bottom() - (segment + 1) as f32 * 8.3,
                ),
                Pos2::new(
                    rect.right() - 4.0,
                    rect.bottom() - segment as f32 * 8.3 - 2.0,
                ),
            );
            let lit = bottom_fraction < level;
            let color = if bottom_fraction > 0.88 {
                theme::RED
            } else if bottom_fraction > 0.68 {
                theme::YELLOW
            } else {
                theme::GREEN
            };
            ui.painter().rect_filled(
                segment_rect,
                1.0,
                if lit {
                    color
                } else {
                    color.gamma_multiply(0.16)
                },
            );
        }
    });
}

fn eq_curve_display(ui: &mut egui::Ui, effects: TrackEffects) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(205.0, 82.0), Sense::hover());
    ui.painter()
        .rect_filled(rect, 2.0, Color32::from_rgb(18, 22, 23));
    for division in 1..4 {
        let x = egui::lerp(rect.x_range(), division as f32 / 4.0);
        ui.painter().line_segment(
            [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
            Stroke::new(0.5_f32, theme::BORDER),
        );
    }
    ui.painter().line_segment(
        [
            Pos2::new(rect.left(), rect.center().y),
            Pos2::new(rect.right(), rect.center().y),
        ],
        Stroke::new(1.0_f32, theme::BORDER),
    );
    let points = (0..64)
        .map(|index| {
            let t = index as f32 / 63.0;
            let gain = effects.low_db * (1.0 - t).powi(2)
                + effects.mid_db * (std::f32::consts::PI * t).sin()
                + effects.high_db * t.powi(2);
            Pos2::new(
                egui::lerp(rect.x_range(), t),
                rect.center().y - gain / 12.0 * rect.height() * 0.42,
            )
        })
        .collect::<Vec<_>>();
    ui.painter()
        .add(egui::Shape::line(points, Stroke::new(2.0_f32, theme::BLUE)));
}

fn dynamics_activity(ui: &mut egui::Ui, peak: f32, compressor_threshold: f32, gate_threshold: f32) {
    let peak_db = 20.0 * peak.max(0.000_001).log10();
    ui.vertical(|ui| {
        ui.label(
            RichText::new("ACTIVITY")
                .monospace()
                .small()
                .color(theme::MUTED),
        );
        activity_led(ui, "COMP", peak_db > compressor_threshold, theme::YELLOW);
        activity_led(ui, "GATE OPEN", peak_db > gate_threshold, theme::GREEN);
    });
}

fn activity_led(ui: &mut egui::Ui, label: &str, active: bool, color: Color32) {
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
        ui.painter().circle_filled(
            rect.center(),
            4.0,
            if active {
                color
            } else {
                color.gamma_multiply(0.15)
            },
        );
        ui.label(RichText::new(label).monospace().small());
    });
}

fn transport_button(ui: &mut egui::Ui, label: &str, active: bool) -> egui::Response {
    ui.add(
        egui::Button::new(RichText::new(label).size(17.0).color(if active {
            Color32::WHITE
        } else {
            theme::TEXT
        }))
        .fill(if active {
            if label == "●" {
                theme::RED
            } else {
                theme::BLUE_DARK
            }
        } else {
            theme::PANEL_2
        })
        .min_size(Vec2::new(39.0, 32.0)),
    )
}

fn meter(ui: &mut egui::Ui, peak: f32) {
    let desired = Vec2::new(ui.available_width(), 7.0);
    let (rect, _) = ui.allocate_exact_size(desired, Sense::hover());
    ui.painter().rect_filled(rect, 1.0, theme::BG);
    let fraction = peak.clamp(0.0, 1.0).sqrt();
    let filled = Rect::from_min_size(rect.min, Vec2::new(rect.width() * fraction, rect.height()));
    let color = if peak > 0.95 {
        theme::RED
    } else if peak > 0.7 {
        theme::YELLOW
    } else {
        theme::GREEN
    };
    ui.painter().rect_filled(filled, 1.0, color);
    ui.painter().rect_stroke(
        rect,
        1.0,
        Stroke::new(1.0_f32, theme::BORDER),
        StrokeKind::Inside,
    );
}

fn draw_clip(
    painter: &egui::Painter,
    track_rect: Rect,
    clip: &Clip,
    sample_rate: u32,
    pixels_per_second: f32,
    selected: bool,
) -> Rect {
    let start = clip.start_frame as f32 / sample_rate as f32 * pixels_per_second;
    let end = clip.end_frame as f32 / sample_rate as f32 * pixels_per_second;
    let rect = Rect::from_min_max(
        Pos2::new(track_rect.left() + start, track_rect.top() + 7.0),
        Pos2::new(
            track_rect.left() + end.max(start + 8.0),
            track_rect.bottom() - 7.0,
        ),
    );
    painter.rect_filled(rect, 3.0, clip.color);
    painter.rect_stroke(
        rect,
        3.0,
        Stroke::new(
            if selected { 2.0_f32 } else { 1.0_f32 },
            if selected { theme::YELLOW } else { theme::BLUE },
        ),
        StrokeKind::Inside,
    );
    painter.text(
        rect.left_top() + Vec2::new(6.0, 5.0),
        Align2::LEFT_TOP,
        &clip.name,
        FontId::proportional(11.0),
        Color32::WHITE,
    );
    let center = rect.center().y + 6.0;
    let mut x = rect.left() + 5.0;
    // Which slice of the file this clip is showing. The peaks span the whole
    // source, so a trimmed clip has to read the matching part of them —
    // drawing them across the clip's width regardless is what made a trim look
    // like the audio had moved rather than been trimmed.
    let (window_start, window_span) = if clip.source_frames > 0 {
        let total = clip.source_frames as f32;
        let start = clip.source_start_frame as f32 / total;
        let span = (clip.length() as f32 / total).max(f32::EPSILON);
        (start, span)
    } else {
        (0.0, 1.0)
    };
    while x < rect.right() - 3.0 {
        let across = ((x - rect.left()) / rect.width()).clamp(0.0, 1.0);
        let fraction = (window_start + across * window_span).clamp(0.0, 1.0);
        let peak_index = ((clip.waveform.len().saturating_sub(1)) as f32 * fraction) as usize;
        let peak = clip.waveform.get(peak_index).copied().unwrap_or(0.0);
        let normalized = peak.sqrt() * (rect.height() * 0.34);
        painter.line_segment(
            [
                Pos2::new(x, center - normalized),
                Pos2::new(x, center + normalized),
            ],
            Stroke::new(0.7_f32, Color32::from_rgba_unmultiplied(184, 220, 240, 150)),
        );
        x += 2.0;
    }
    rect
}

/// Draws a MIDI clip as a block with a miniature of its notes, so an
/// instrument track reads at a glance like the waveform lanes beside it.
fn draw_midi_clip(
    painter: &egui::Painter,
    track_rect: Rect,
    clip: &MidiClip,
    tempo: &TempoMap,
    sample_rate: u32,
    pixels_per_second: f32,
) -> Rect {
    let start_seconds = tempo.tick_to_seconds(clip.start_tick);
    let end_seconds = tempo.tick_to_seconds(clip.end_tick());
    let x0 = track_rect.left() + start_seconds as f32 * pixels_per_second;
    let x1 = (track_rect.left() + end_seconds as f32 * pixels_per_second).max(x0 + 3.0);
    let rect = Rect::from_min_max(
        Pos2::new(x0, track_rect.top() + 6.0),
        Pos2::new(x1.min(track_rect.right()), track_rect.bottom() - 6.0),
    );
    painter.rect_filled(rect, 3.0, Color32::from_rgb(46, 74, 96));
    painter.rect_stroke(
        rect,
        3.0,
        Stroke::new(1.0_f32, theme::BLUE),
        StrokeKind::Inside,
    );

    let (Some(low), Some(high)) = (clip.lowest_pitch(), clip.highest_pitch()) else {
        return rect;
    };
    let span = f32::from(high.saturating_sub(low)).max(1.0);
    let inner = rect.shrink(3.0);
    for note in &clip.notes {
        let note_start = tempo.tick_to_seconds(clip.start_tick + note.start_tick);
        let note_end = tempo.tick_to_seconds(clip.start_tick + note.end_tick());
        let nx0 = track_rect.left() + note_start as f32 * pixels_per_second;
        let nx1 = (track_rect.left() + note_end as f32 * pixels_per_second).max(nx0 + 1.0);
        if nx1 < inner.left() || nx0 > inner.right() {
            continue;
        }
        let level = f32::from(note.pitch - low) / span;
        let y = inner.bottom() - level * inner.height();
        painter.line_segment(
            [
                Pos2::new(nx0.max(inner.left()), y),
                Pos2::new(nx1.min(inner.right()), y),
            ],
            Stroke::new(1.5_f32, Color32::from_rgb(150, 205, 245)),
        );
    }
    let _ = sample_rate;
    rect
}

/// Reads a WAV's layout and length, if it is one this session can play as it
/// stands: the engine does not resample, so anything else has to be converted
/// by [`convert_import_audio`] first.
/// Names a transposition the way a musician would say it, so the number in the
/// box is not the only thing to go on.
fn transpose_description(semitones: i32) -> String {
    const INTERVALS: [&str; 13] = [
        "unison",
        "minor 2nd",
        "major 2nd",
        "minor 3rd",
        "major 3rd",
        "perfect 4th",
        "tritone",
        "perfect 5th",
        "minor 6th",
        "major 6th",
        "minor 7th",
        "major 7th",
        "octave",
    ];
    if semitones == 0 {
        return "original key".to_owned();
    }
    let direction = if semitones < 0 { "down" } else { "up" };
    let name = INTERVALS
        .get(semitones.unsigned_abs() as usize)
        .copied()
        .unwrap_or("more than an octave");
    format!("{direction} a {name}")
}

/// What the last export did, shown in a dialog until it is dismissed.
///
/// The status bar alone was not enough: it is one line at the bottom of a wide
/// window, and an export that took a second to render looked like a button that
/// did nothing at all.
struct ExportReport {
    heading: String,
    detail: String,
    /// The written file, when there is one to reveal.
    path: Option<PathBuf>,
}

impl ExportReport {
    fn exported(detail: String, path: PathBuf) -> Self {
        Self {
            heading: "Mix exported".to_owned(),
            detail,
            path: Some(path),
        }
    }

    fn failed(heading: &str, detail: String) -> Self {
        Self {
            heading: heading.to_owned(),
            detail,
            path: None,
        }
    }
}

/// Opens a folder in the desktop's file manager.
///
/// Failure is ignored for the same reason [`open_in_browser`] ignores it: a
/// machine with no handler makes this a button that does nothing, which is a
/// disappointment rather than an error worth interrupting the session for.
fn reveal_in_file_manager(directory: &std::path::Path) {
    let _ = Command::new("xdg-open")
        .arg(directory)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
}

fn inspect_import_audio(
    path: &std::path::Path,
    expected_rate: u32,
) -> anyhow::Result<(ChannelLayout, u64)> {
    let reader = hound::WavReader::open(path)
        .map_err(|error| anyhow::anyhow!("{}: {error}", path.display()))?;
    let spec = reader.spec();
    anyhow::ensure!(
        spec.sample_rate == expected_rate,
        "{} is {} Hz; session is {} Hz",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("audio"),
        spec.sample_rate,
        expected_rate
    );
    let layout = match spec.channels {
        1 => ChannelLayout::Mono,
        2 => ChannelLayout::Stereo,
        channels => anyhow::bail!(
            "{} has {channels} channels; only mono/stereo WAV is supported",
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("audio")
        ),
    };
    Ok((layout, u64::from(reader.duration())))
}

/// Where converted imports are kept, beside `Recordings` and `Songs`.
///
/// They have to outlive the import: a clip refers to its file by path, so a
/// session saved today reopens next week by reading the same converted WAV.
fn import_dir() -> PathBuf {
    daw_core::media_dir("Imports")
}

/// A folder in `parent` that does not exist yet, so exporting stems twice keeps
/// both sets rather than writing the second over the first.
fn unique_directory(parent: &std::path::Path, name: &str) -> PathBuf {
    let candidate = parent.join(name);
    if !candidate.exists() {
        return candidate;
    }
    for index in 2..1000 {
        let candidate = parent.join(format!("{name} {index}"));
        if !candidate.exists() {
            return candidate;
        }
    }
    parent.join(format!("{name} {}", Uuid::new_v4().simple()))
}

/// A name in `directory` that is not taken yet, so importing the same song
/// twice does not overwrite the copy the first clip is still playing.
fn unique_import_path(directory: &std::path::Path, stem: &str) -> PathBuf {
    let candidate = directory.join(format!("{stem}.wav"));
    if !candidate.exists() {
        return candidate;
    }
    for index in 2..1000 {
        let candidate = directory.join(format!("{stem}-{index}.wav"));
        if !candidate.exists() {
            return candidate;
        }
    }
    directory.join(format!("{stem}-{}.wav", Uuid::new_v4().simple()))
}

/// Converts any audio file ffmpeg can read into a session-rate WAV under
/// [`import_dir`], and returns it with its layout and length.
///
/// Stereo is the output for everything except a mono source, which stays mono:
/// an MP3 dropped on the timeline should arrive as the stereo track it is,
/// while a mono take should not be doubled into two identical channels. Sources
/// with more channels than two are downmixed by ffmpeg rather than refused.
fn convert_import_audio(
    source: &std::path::Path,
    expected_rate: u32,
) -> anyhow::Result<(PathBuf, ChannelLayout, u64)> {
    convert_import_audio_into(&import_dir(), source, expected_rate)
}

/// [`convert_import_audio`] with the destination folder named, so a test can
/// convert into a temporary directory instead of the session's own.
fn convert_import_audio_into(
    directory: &std::path::Path,
    source: &std::path::Path,
    expected_rate: u32,
) -> anyhow::Result<(PathBuf, ChannelLayout, u64)> {
    std::fs::create_dir_all(directory)
        .map_err(|error| anyhow::anyhow!("could not create {}: {error}", directory.display()))?;
    let stem = source
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("Imported Audio");
    let destination = unique_import_path(directory, stem);

    // Only a WAV can be inspected without decoding, and only its channel count
    // is needed here; everything else becomes stereo.
    let layout = match hound::WavReader::open(source).map(|reader| reader.spec().channels) {
        Ok(1) => ChannelLayout::Mono,
        _ => ChannelLayout::Stereo,
    };
    let output = Command::new(ffmpeg_program())
        .arg("-nostdin")
        .arg("-y")
        .args(["-v", "error"])
        .arg("-i")
        .arg(source)
        .arg("-vn")
        .args(["-ar", &expected_rate.to_string()])
        .args([
            "-ac",
            if layout == ChannelLayout::Mono {
                "1"
            } else {
                "2"
            },
        ])
        .args(["-c:a", "pcm_s24le"])
        .arg(&destination)
        .output()
        .map_err(|error| {
            anyhow::anyhow!(
                "could not run ffmpeg to convert {}: {error}; install it with `{}`",
                source.display(),
                ffmpeg_install_hint()
            )
        })?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr);
        let reason = detail.lines().last().unwrap_or("ffmpeg failed");
        let _ = std::fs::remove_file(&destination);
        anyhow::bail!(
            "{}: {reason}",
            source
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("audio")
        );
    }
    let (layout, frames) = inspect_import_audio(&destination, expected_rate)?;
    Ok((destination, layout, frames))
}

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
/// A GUI app launched from the desktop inherits a minimal `PATH` that excludes
/// Homebrew, so a bare `ffmpeg` is not found even when it is installed.
/// `FFMPEG` overrides; otherwise the common install locations are checked
/// before falling back to the name on `PATH`.
fn ffmpeg_program() -> PathBuf {
    if let Some(explicit) = std::env::var_os("FFMPEG").filter(|value| !value.is_empty()) {
        return PathBuf::from(explicit);
    }
    for candidate in [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
    ] {
        if std::path::Path::new(candidate).is_file() {
            return PathBuf::from(candidate);
        }
    }
    PathBuf::from("ffmpeg")
}

fn ensure_extension(mut path: PathBuf, extension: &str) -> PathBuf {
    if path.extension().is_none() {
        path.set_extension(extension);
    }
    path
}

/// The bottom of the amp gate's travel, where it passes everything.
const GATE_OPEN_DB: f32 = -95.0;

/// The delay's longest setting, for the TIME control's range.
const MAX_DELAY_MS: f32 = 1_000.0;

/// Where free amp captures come from. The catalogue is browsed on the site
/// itself: the files are downloaded there and read back out of [`daw_nam::amp_dir`].
const TONE3000_URL: &str = "https://www.tone3000.com/";

/// Opens a URL in the user's browser.
///
/// Failure is ignored on purpose. A machine with no browser handler makes this
/// a button that does nothing, which is a disappointment rather than an error
/// worth interrupting the session for.
fn open_in_browser(url: &str) {
    let _ = std::process::Command::new("xdg-open")
        .arg(url)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
}

/// Names the folder captures are read from, so the rescan button says where to
/// put the files rather than leaving the user to guess.
fn amp_library_hint() -> String {
    format!("Look again in {}", daw_nam::amp_dir().display())
}

fn sanitize_file_name(name: &str) -> String {
    let sanitized = name
        .chars()
        .map(|character| {
            if character.is_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if sanitized.is_empty() {
        "Untitled_Session".to_owned()
    } else {
        sanitized
    }
}

fn recording_path(track_index: usize) -> anyhow::Result<PathBuf> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let directory = recording_directory();
    std::fs::create_dir_all(&directory)?;
    Ok(directory.join(format!("Audio_{:02}_{timestamp}.wav", track_index + 1)))
}

fn recording_directory() -> PathBuf {
    daw_core::media_dir("Recordings")
}

fn disk_free_bytes(path: &std::path::Path) -> anyhow::Result<u64> {
    std::fs::create_dir_all(path)?;
    let output = Command::new("df").arg("-Pk").arg(path).output()?;
    anyhow::ensure!(
        output.status.success(),
        "df could not inspect {}",
        path.display()
    );
    let text = String::from_utf8(output.stdout)?;
    let available_kib = text
        .lines()
        .last()
        .and_then(|line| line.split_whitespace().nth(3))
        .ok_or_else(|| anyhow::anyhow!("unexpected df output"))?
        .parse::<u64>()?;
    Ok(available_kib.saturating_mul(1024))
}

fn format_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GiB", bytes as f64 / GIB)
    } else {
        format!("{:.0} MiB", bytes as f64 / MIB)
    }
}

fn available_audio_devices() -> (Vec<String>, Vec<String>) {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    if let Ok(devices) = enumerate_pipewire_devices() {
        for device in devices {
            if (!device.input_ranges.is_empty() || device.default_input.is_some())
                && !is_output_monitor_name(&device.name)
            {
                inputs.push(device.name.clone());
            }
            if !device.output_ranges.is_empty() || device.default_output.is_some() {
                outputs.push(device.name);
            }
        }
    }
    inputs.sort();
    inputs.dedup();
    outputs.sort();
    outputs.dedup();
    (inputs, outputs)
}

fn is_output_monitor_name(name: &str) -> bool {
    let lowercase = name.to_ascii_lowercase();
    lowercase.starts_with("monitor of ") || lowercase.ends_with(".monitor")
}

fn audio_preferences_path() -> anyhow::Result<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
        .ok_or_else(|| anyhow::anyhow!("cannot resolve the user configuration directory"))?;
    Ok(base.join("rustdaw").join("audio.json"))
}

fn load_audio_preferences() -> anyhow::Result<AudioPreferences> {
    let bytes = std::fs::read(audio_preferences_path()?)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn save_audio_preferences(preferences: &AudioPreferences) -> anyhow::Result<()> {
    let path = audio_preferences_path()?;
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("invalid audio preferences path"))?;
    std::fs::create_dir_all(parent)?;
    let temporary = path.with_extension("json.tmp");
    std::fs::write(&temporary, serde_json::to_vec_pretty(preferences)?)?;
    std::fs::rename(temporary, path)?;
    Ok(())
}

fn default_session_path() -> PathBuf {
    daw_core::media_dir("Sessions").join("Current.rustdaw.json")
}

/// Peak buckets for a whole file, and how many frames it holds.
///
/// The peaks cover the entire file rather than any one clip's window, so
/// several clips split from one take share the same measurement and a trim
/// redraws without touching the disk.
fn analyze_waveform(path: &std::path::Path) -> (Vec<f32>, u64) {
    const PEAK_BUCKETS: usize = 512;
    let Ok(mut reader) = hound::WavReader::open(path) else {
        return (Vec::new(), 0);
    };
    let spec = reader.spec();
    let values = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .filter_map(Result::ok)
            .map(f32::abs)
            .collect::<Vec<_>>(),
        hound::SampleFormat::Int => {
            let scale = 2.0_f32.powi(i32::from(spec.bits_per_sample).saturating_sub(1));
            reader
                .samples::<i32>()
                .filter_map(Result::ok)
                .map(|sample| {
                    #[allow(clippy::cast_precision_loss)]
                    {
                        (sample as f32 / scale).abs()
                    }
                })
                .collect::<Vec<_>>()
        }
    };
    let channels = usize::from(spec.channels).max(1);
    let frames = values.len() / channels;
    if frames == 0 {
        return (Vec::new(), 0);
    }
    let frames_per_bucket = frames.div_ceil(PEAK_BUCKETS).max(1);
    let peaks = values
        .chunks(frames_per_bucket.saturating_mul(channels))
        .map(|bucket| bucket.iter().copied().fold(0.0_f32, f32::max))
        .collect();
    (peaks, frames as u64)
}

fn track_from_project(track: ProjectTrack) -> Track {
    Track {
        id: track.id,
        name: track.name,
        layout: track.layout,
        input_left: track.input_left,
        input_right: track.input_right,
        armed: false,
        monitoring: false,
        muted: track.muted,
        solo: track.solo,
        gain_db: track.gain_db,
        pan: track.pan,
        effects: track.effects,
        nam_model: track.nam_model,
        clips: track
            .clips
            .into_iter()
            .map(|clip| {
                let (waveform, source_frames) = analyze_waveform(&clip.path);
                Clip {
                    id: clip.id,
                    waveform,
                    source_frames,
                    name: clip.name,
                    path: clip.path,
                    start_frame: clip.start_frame,
                    end_frame: clip.end_frame,
                    source_start_frame: clip.source_start_frame,
                    source_path: clip.source_path,
                    color: theme::BLUE_DARK,
                }
            })
            .collect(),
        kind: track.kind,
        midi_clips: track.midi_clips,
        program: track.program,
        drum_kit: track.drum_kit,
    }
}

fn track_to_project(track: &Track) -> ProjectTrack {
    ProjectTrack {
        id: track.id,
        name: track.name.clone(),
        layout: track.layout,
        input_left: track.input_left,
        input_right: track.input_right,
        muted: track.muted,
        solo: track.solo,
        gain_db: track.gain_db,
        pan: track.pan,
        effects: track.effects,
        nam_model: track.nam_model.clone(),
        clips: track
            .clips
            .iter()
            .map(|clip| ProjectClip {
                id: clip.id,
                name: clip.name.clone(),
                path: clip.path.clone(),
                start_frame: clip.start_frame,
                end_frame: clip.end_frame,
                source_start_frame: clip.source_start_frame,
                source_path: clip.source_path.clone(),
            })
            .collect(),
        kind: track.kind,
        midi_clips: track.midi_clips.clone(),
        program: track.program,
        drum_kit: track.drum_kit,
    }
}

fn channel_strip_params(effects: TrackEffects) -> ChannelStripParams {
    ChannelStripParams {
        nam_enabled: effects.nam_enabled,
        nam_input_db: effects.nam_input_db,
        nam_output_db: effects.nam_output_db,
        nam_gate_db: effects.nam_gate_db,
        nam_tone_enabled: effects.nam_tone_enabled,
        nam_bass: effects.nam_bass,
        nam_middle: effects.nam_middle,
        nam_treble: effects.nam_treble,
        nam_normalize: effects.nam_normalize,
        delay_enabled: effects.delay_enabled,
        delay_time_ms: effects.delay_time_ms,
        delay_feedback: effects.delay_feedback,
        delay_mix: effects.delay_mix,
        reverb_enabled: effects.reverb_enabled,
        reverb_size: effects.reverb_size,
        reverb_damping: effects.reverb_damping,
        reverb_mix: effects.reverb_mix,
        eq_enabled: effects.eq_enabled,
        low_db: effects.low_db,
        mid_db: effects.mid_db,
        high_db: effects.high_db,
        compressor_enabled: effects.compressor_enabled,
        compressor_threshold_db: effects.compressor_threshold_db,
        compressor_ratio: effects.compressor_ratio,
        compressor_attack_ms: effects.compressor_attack_ms,
        compressor_release_ms: effects.compressor_release_ms,
        compressor_makeup_db: effects.compressor_makeup_db,
        gate_enabled: effects.gate_enabled,
        gate_threshold_db: effects.gate_threshold_db,
        gate_release_ms: effects.gate_release_ms,
    }
}

fn moved_start_frame(origin: u64, frame_delta: i64) -> u64 {
    if frame_delta.is_negative() {
        origin.saturating_sub(frame_delta.unsigned_abs())
    } else {
        origin.saturating_add(frame_delta as u64)
    }
}

#[cfg(test)]
mod tests {
    fn test_clip(start: u64, end: u64) -> Clip {
        Clip {
            id: Uuid::new_v4(),
            name: "Take".to_owned(),
            path: PathBuf::from("take.wav"),
            start_frame: start,
            end_frame: end,
            source_start_frame: 0,
            source_path: None,
            source_frames: 0,
            color: theme::BLUE_DARK,
            waveform: Vec::new(),
        }
    }

    #[test]
    fn the_end_edge_cannot_reveal_audio_the_file_does_not_have() {
        // A ten-second take, untrimmed, dropped at frame 1000.
        assert_eq!(max_end_frame(1_000, 0, 480_000), 481_000);
        // Already trimmed 2 s off the front: only 8 s remain to show.
        assert_eq!(max_end_frame(1_000, 96_000, 480_000), 385_000);
        // A source still being recorded has no known length yet.
        assert_eq!(max_end_frame(1_000, 0, 0), u64::MAX);
        // A window starting past the end still leaves a usable clip.
        assert_eq!(max_end_frame(1_000, 999_999, 480_000), 1_001);
    }

    #[test]
    fn the_pointer_becomes_a_trimmer_near_either_edge() {
        let rect = Rect::from_min_max(Pos2::new(100.0, 0.0), Pos2::new(400.0, 40.0));
        assert_eq!(ClipZone::at(rect, 102.0), ClipZone::TrimStart);
        assert_eq!(ClipZone::at(rect, 398.0), ClipZone::TrimEnd);
        assert_eq!(ClipZone::at(rect, 250.0), ClipZone::Body);
    }

    #[test]
    fn a_narrow_clip_keeps_a_body_to_grab() {
        // Edge zones covering a whole clip would leave no way to move it.
        let rect = Rect::from_min_max(Pos2::new(100.0, 0.0), Pos2::new(115.0, 40.0));
        assert_eq!(ClipZone::at(rect, 107.0), ClipZone::Body);
    }

    #[test]
    fn a_press_anywhere_in_the_zone_is_a_trim() {
        // egui reports no drag until the pointer has moved six pixels from the
        // press, so the zone is read from the press origin. This walks the
        // whole zone to show every part of it decides a trim — the live
        // position, six pixels further in, would not.
        let rect = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(600.0, 40.0));
        let mut offset = 0.0_f32;
        while offset <= TRIM_ZONE_WIDTH {
            assert_eq!(
                ClipZone::at(rect, rect.left() + offset),
                ClipZone::TrimStart,
                "a press {offset} px in must trim"
            );
            offset += 1.0;
        }
        assert_eq!(
            ClipZone::at(rect, rect.left() + TRIM_ZONE_WIDTH + 1.0),
            ClipZone::Body,
            "and past the zone it moves"
        );
    }

    #[test]
    fn the_trim_cursor_differs_from_the_move_cursor() {
        assert_eq!(ClipZone::Body.cursor(), egui::CursorIcon::Grab);
        assert_eq!(
            ClipZone::TrimStart.cursor(),
            egui::CursorIcon::ResizeHorizontal
        );
        assert_eq!(
            ClipZone::TrimEnd.cursor(),
            egui::CursorIcon::ResizeHorizontal
        );
    }

    #[test]
    fn a_clip_reports_the_length_it_occupies() {
        assert_eq!(test_clip(1_000, 2_500).length(), 1_500);
        // A degenerate clip must not underflow into an enormous length.
        let mut backwards = test_clip(2_000, 2_000);
        backwards.end_frame = 1_000;
        assert_eq!(backwards.length(), 0);
    }

    #[test]
    fn a_settings_file_written_before_the_chord_lane_still_loads() {
        // `#[serde(default)]` on the struct is what makes this true; without
        // it an older config would fail to parse and the user would silently
        // lose their device choices.
        let older = r#"{"input_device":"Scarlett Solo","output_device":"Scarlett Solo",
                        "buffer_frames":256,"input_labels":["a","b","c","d"]}"#;
        let preferences: AudioPreferences = serde_json::from_str(older).expect("older config");
        assert_eq!(preferences.buffer_frames, 256);
        assert!(
            (preferences.chord_lane_scale - 1.0).abs() < f32::EPSILON,
            "a missing size falls back to the base size"
        );
    }

    #[test]
    fn the_chord_lane_size_round_trips() {
        let preferences = AudioPreferences {
            chord_lane_scale: 1.8,
            ..AudioPreferences::default()
        };
        let text = serde_json::to_string(&preferences).expect("serialises");
        let back: AudioPreferences = serde_json::from_str(&text).expect("parses");
        assert!((back.chord_lane_scale - 1.8).abs() < f32::EPSILON);
    }

    use super::*;

    #[test]
    fn import_inspection_detects_layout_and_rate_mismatch() {
        let stem = format!("rustdaw-import-test-{}", std::process::id());
        let mono = std::env::temp_dir().join(format!("{stem}-mono.wav"));
        let stereo_wrong_rate = std::env::temp_dir().join(format!("{stem}-stereo-wrong-rate.wav"));
        write_test_wav(&mono, 1, 48_000);
        write_test_wav(&stereo_wrong_rate, 2, 44_100);

        assert_eq!(
            inspect_import_audio(&mono, 48_000).unwrap(),
            (ChannelLayout::Mono, 64)
        );
        assert!(inspect_import_audio(&stereo_wrong_rate, 48_000).is_err());

        std::fs::remove_file(mono).unwrap();
        std::fs::remove_file(stereo_wrong_rate).unwrap();
    }

    /// The conversion tests drive the real ffmpeg, the same one an import
    /// uses. Where it is absent the import itself cannot work either, so they
    /// pass rather than fail on a machine that is missing it.
    fn ffmpeg_missing() -> bool {
        Command::new(ffmpeg_program())
            .arg("-version")
            .output()
            .is_err()
    }

    #[test]
    fn a_file_at_the_wrong_rate_is_converted_to_the_session_rate() {
        if ffmpeg_missing() {
            return;
        }
        let directory =
            std::env::temp_dir().join(format!("rustdaw-convert-{}", std::process::id()));
        let source =
            std::env::temp_dir().join(format!("rustdaw-convert-{}.wav", std::process::id()));
        write_test_wav(&source, 2, 44_100);

        let (converted, layout, frames) =
            convert_import_audio_into(&directory, &source, 48_000).unwrap();
        assert_eq!(layout, ChannelLayout::Stereo);
        // 64 frames of 44.1 kHz is 1.45 ms, which is ~70 frames at 48 kHz.
        assert!((60..=80).contains(&frames), "{frames} frames");
        // Playable as it stands now: that is what the engine will demand.
        assert_eq!(
            inspect_import_audio(&converted, 48_000).unwrap(),
            (ChannelLayout::Stereo, frames)
        );

        std::fs::remove_file(source).unwrap();
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn a_mono_source_is_not_widened_into_two_identical_channels() {
        if ffmpeg_missing() {
            return;
        }
        let directory =
            std::env::temp_dir().join(format!("rustdaw-convert-mono-{}", std::process::id()));
        let source =
            std::env::temp_dir().join(format!("rustdaw-convert-mono-{}.wav", std::process::id()));
        write_test_wav(&source, 1, 44_100);

        let (_, layout, _) = convert_import_audio_into(&directory, &source, 48_000).unwrap();
        assert_eq!(layout, ChannelLayout::Mono);

        std::fs::remove_file(source).unwrap();
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn importing_the_same_name_twice_does_not_overwrite_the_first_copy() {
        let directory = std::env::temp_dir().join(format!("rustdaw-unique-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();

        let first = unique_import_path(&directory, "Song");
        assert_eq!(first, directory.join("Song.wav"));
        std::fs::write(&first, b"taken").unwrap();
        let second = unique_import_path(&directory, "Song");
        assert_eq!(second, directory.join("Song-2.wav"));

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn session_extension_is_added_only_when_missing() {
        assert_eq!(
            ensure_extension(PathBuf::from("Song"), "rustdaw"),
            PathBuf::from("Song.rustdaw")
        );
        assert_eq!(
            ensure_extension(PathBuf::from("Song.session"), "rustdaw"),
            PathBuf::from("Song.session")
        );
    }

    #[test]
    fn drag_release_commits_cumulative_offset_once() {
        assert_eq!(moved_start_frame(48_000, 24_000), 72_000);
        assert_eq!(moved_start_frame(48_000, -12_000), 36_000);
        assert_eq!(moved_start_frame(1_000, -2_000), 0);
    }

    fn write_test_wav(path: &std::path::Path, channels: u16, sample_rate: u32) {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 24,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for _ in 0..64 {
            for _ in 0..channels {
                writer.write_sample(0_i32).unwrap();
            }
        }
        writer.finalize().unwrap();
    }
}
