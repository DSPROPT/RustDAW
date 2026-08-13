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
    CancelFlag, ImportProgress, ImportSource, IngestOptions, Ingested, ProjectSummary,
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
    color: Color32,
    waveform: Vec<f32>,
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

#[derive(Clone, Copy)]
enum EditCommand {
    MoveClip {
        clip_id: Uuid,
        before: ClipLocation,
        after: ClipLocation,
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
}

impl Default for AudioPreferences {
    fn default() -> Self {
        Self {
            input_device: "Scarlett Solo".to_owned(),
            output_device: "Scarlett Solo".to_owned(),
            buffer_frames: 256,
            input_labels: std::array::from_fn(|index| format!("Input {}", index + 1)),
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
    selected_clip: Option<(usize, usize)>,
    dragged_clip: Option<(usize, usize, u64, Vec2)>,
    dirty: bool,
    confirm_new_session: bool,
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
            selected_clip: None,
            dragged_clip: None,
            dirty: false,
            confirm_new_session: false,
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
        let waveform = analyze_waveform(&path);
        let clip_id = Uuid::new_v4();
        if let Some(track) = self.tracks.get_mut(self.selected_track) {
            track.clips.push(Clip {
                id: clip_id,
                name,
                path,
                start_frame,
                end_frame,
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
                runtime.add_identified_track_playback_file(
                    &clip.path,
                    clip.start_frame,
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

    fn undo(&mut self) {
        let Some(command) = self.undo_stack.pop() else {
            self.status_message = "Nothing to undo".to_owned();
            return;
        };
        let EditCommand::MoveClip {
            clip_id, before, ..
        } = command;
        match self.apply_clip_location(clip_id, before) {
            Ok(()) => {
                self.redo_stack.push(command);
                self.save_session();
                self.status_message = "Undid clip move".to_owned();
            }
            Err(error) => self.status_message = format!("Could not undo: {error}"),
        }
    }

    fn redo(&mut self) {
        let Some(command) = self.redo_stack.pop() else {
            self.status_message = "Nothing to redo".to_owned();
            return;
        };
        let EditCommand::MoveClip { clip_id, after, .. } = command;
        match self.apply_clip_location(clip_id, after) {
            Ok(()) => {
                self.undo_stack.push(command);
                self.save_session();
                self.status_message = "Redid clip move".to_owned();
            }
            Err(error) => self.status_message = format!("Could not redo: {error}"),
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
            .add_filter("WAV Audio", &["wav", "wave"])
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
        let mut errors = Vec::new();
        for path in paths {
            if !path.is_file() {
                continue;
            }
            match inspect_import_audio(&path, expected_rate) {
                Ok((layout, frames)) => {
                    let name = path
                        .file_stem()
                        .and_then(|name| name.to_str())
                        .unwrap_or("Imported Audio")
                        .to_owned();
                    let mut track = Track::new(self.tracks.len(), layout);
                    track.name.clone_from(&name);
                    let clip_id = Uuid::new_v4();
                    track.clips.push(Clip {
                        id: clip_id,
                        name,
                        waveform: analyze_waveform(&path),
                        path,
                        start_frame,
                        end_frame: start_frame.saturating_add(frames),
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
        self.status_message = if errors.is_empty() {
            format!("Imported {imported} audio file(s) at the playhead")
        } else {
            format!(
                "Imported {imported}; {} failed: {}",
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

    fn export_mix(&mut self) {
        if matches!(
            self.snapshot().transport,
            RuntimeTransportState::Recording | RuntimeTransportState::CountIn
        ) {
            self.status_message = "Stop recording before exporting".to_owned();
            return;
        }
        let destination = daw_core::media_dir("Exports").join("Current Mix.wav");
        if self.master_reference.is_some() {
            // Mastering analyses the whole mix twice over and convolves it, so
            // on a long song this is seconds rather than instant.
            self.status_message = "Exporting and mastering…".to_owned();
        }
        match daw_render::export_stereo(&self.project_document(), &destination) {
            Ok(frames) => {
                let mastered = if self.master_reference.is_some() {
                    " (mastered)"
                } else {
                    ""
                };
                self.status_message = format!(
                    "Exported {:.2} s{mastered} to {}",
                    frames as f64 / 48_000.0,
                    destination.display()
                );
            }
            Err(error) => self.status_message = format!("{error:#}"),
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
        self.playback_synced = true;
        self.session_needs_save_as = true;
        self.dirty = true;
        self.status_message = "New session — press Ctrl+S to save".to_owned();
    }

    fn delete_selected_clip(&mut self) {
        let Some((track_index, clip_index)) = self.selected_clip.take() else {
            return;
        };
        if let Some(track) = self.tracks.get_mut(track_index) {
            if clip_index < track.clips.len() {
                track.clips.remove(clip_index);
                self.dirty = true;
                self.status_message = "Clip removed (audio file preserved)".to_owned();
                self.save_session();
                self.playback_synced = false;
                if let Err(error) = self.sync_playback() {
                    self.status_message = format!("Clip removed; playback preload failed: {error}");
                }
            }
        }
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
        if context.input(|input| input.key_pressed(egui::Key::C)) {
            self.click_enabled = !self.click_enabled;
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
            if response.clicked() {
                selected_request = Some((index, clip_index));
                clip_interacted = true;
            }
            if response.drag_started() {
                drag_start_request = Some((index, clip_index, clip.start_frame, Vec2::ZERO));
                selected_request = Some((index, clip_index));
                clip_interacted = true;
            }
            if response.dragged() {
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
            if response.drag_stopped() {
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
                    if ui.button("EXPORT MIX").clicked() {
                        self.export_mix();
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
                let (ruler, _) = ui.allocate_exact_size(
                    Vec2::new(ui.available_width(), ruler_height),
                    Sense::hover(),
                );
                let painter = ui.painter_at(ruler);
                painter.rect_filled(ruler, 0.0, theme::PANEL_2);
                // The ruler is drawn scrolled by the same offset as the tracks
                // below it, so its time labels always sit over the right frames.
                for second in 0..=600 {
                    let x = ruler.left() + second as f32 * self.pixels_per_second
                        - self.timeline_scroll_x;
                    if x > ruler.right() {
                        break;
                    }
                    if x < ruler.left() - 4.0 {
                        continue;
                    }
                    if second % 5 == 0 {
                        painter.text(
                            Pos2::new(x + 4.0, ruler.center().y),
                            Align2::LEFT_CENTER,
                            format!("{:02}:{:02}", second / 60, second % 60),
                            FontId::monospace(11.0),
                            theme::MUTED,
                        );
                    }
                }
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
                            ui.label("Each mono/stereo WAV becomes a new track at the playhead.");
                        });
                });
        }
        self.audio_settings(context, &snapshot);
        self.mixer_window(context, &snapshot);
        self.inserts_window(context, &snapshot);
        self.run_tuner(context);
        self.poll_song_import(context);
        self.poll_amp_fetch();
        self.song_import_window(context);
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
    while x < rect.right() - 3.0 {
        let fraction = ((x - rect.left()) / rect.width()).clamp(0.0, 1.0);
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

fn analyze_waveform(path: &std::path::Path) -> Vec<f32> {
    const PEAK_BUCKETS: usize = 512;
    let Ok(mut reader) = hound::WavReader::open(path) else {
        return Vec::new();
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
        return Vec::new();
    }
    let frames_per_bucket = frames.div_ceil(PEAK_BUCKETS).max(1);
    values
        .chunks(frames_per_bucket.saturating_mul(channels))
        .map(|bucket| bucket.iter().copied().fold(0.0_f32, f32::max))
        .collect()
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
            .map(|clip| Clip {
                id: clip.id,
                waveform: analyze_waveform(&clip.path),
                name: clip.name,
                path: clip.path,
                start_frame: clip.start_frame,
                end_frame: clip.end_frame,
                color: theme::BLUE_DARK,
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
