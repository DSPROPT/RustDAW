use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};
use crossbeam_queue::ArrayQueue;
use daw_core::{ChannelLayout, SamplePosition, SampleRate};
use daw_engine::{
    ChannelStrip, ChannelStripParams, GmBank, Metronome, NoiseGate, Reverb, SampledSynth,
    SoundFontBank, Synth, ToneStack,
};
use daw_midi::ScheduledNote;
use daw_nam::NamProcessor;

use crate::time_stretch::TimeStretcher;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const RECORD_QUEUE_SECONDS: usize = 8;
const CLICK_SCRATCH_FRAMES: usize = 2_048;
const PLAYBACK_COMMAND_CAPACITY: usize = 128;
const PLAYBACK_SLOT_CAPACITY: usize = 64;
const MONITOR_QUEUE_FRAMES: usize = 8_192;
/// The shortest monitor backlog worth keeping. Below a block or so the two
/// streams' ordinary jitter starts causing dropouts instead of latency.
const MONITOR_MIN_QUEUE_FRAMES: usize = 64;
const RETIRED_PLAYBACK_CAPACITY: usize = 256;
const MIXER_TRACK_CAPACITY: usize = 64;
/// How long the instrument reverb keeps running after the last note was sent
/// to it. Long enough to cover the tail, short enough that a session of nothing
/// but audio clips stops paying for an empty room.
const REVERB_TAIL_SECONDS: usize = 6;
/// The loudness every normalised capture is brought to, in dB. The value NAM's
/// own trainer measures against, so a normalised model here sits where it does
/// in the reference plugin.
const NORMALIZE_TARGET_DB: f64 = -18.0;
/// One slot per instrument track.
const MIDI_SLOT_CAPACITY: usize = 64;
const RETIRED_NAM_CAPACITY: usize = PLAYBACK_COMMAND_CAPACITY;
/// Playback-speed limits for the real-time time-stretch control.
const MIN_SPEED: f32 = 0.5;
const MAX_SPEED: f32 = 2.0;

fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db.clamp(-60.0, 24.0) / 20.0)
}

#[derive(Clone, Debug)]
pub struct AudioRuntimeConfig {
    pub input_name_contains: String,
    pub output_name_contains: String,
    pub sample_rate: SampleRate,
    pub buffer_frames: u32,
    pub instruments: InstrumentSource,
}

impl Default for AudioRuntimeConfig {
    fn default() -> Self {
        Self {
            input_name_contains: "Scarlett Solo".to_owned(),
            output_name_contains: "Scarlett Solo".to_owned(),
            sample_rate: SampleRate::DEFAULT,
            buffer_frames: 256,
            instruments: InstrumentSource::default(),
        }
    }
}

/// What instrument tracks are played by.
// "SoundFont" here is the file format, not a type name.
#[allow(clippy::doc_markdown)]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum InstrumentSource {
    /// Play a SoundFont if one can be found, and the synthesised bank if not.
    ///
    /// Recorded instruments sound better than anything synthesis will manage,
    /// so they win when they are there; a machine without one still plays.
    #[default]
    Auto,
    /// Play this SoundFont. Failing to load it fails to open the engine, since
    /// the user asked for this file specifically and silently ignoring that
    /// would leave them wondering why nothing changed.
    SoundFont(PathBuf),
    /// Always play the synthesised bank, whatever is installed.
    Synthesised,
}

#[derive(Clone, Copy, Debug)]
pub struct RuntimeSnapshot {
    pub position_frames: u64,
    pub input_peaks: [f32; 4],
    pub track_peaks: [f32; MIXER_TRACK_CAPACITY],
    pub transport: RuntimeTransportState,
    pub xruns: u64,
    pub dropped_record_frames: u64,
    /// Input frames dropped to keep monitoring latency down. A one-off jump at
    /// the start of monitoring is normal; steady growth means the input stream
    /// is outrunning the output.
    pub trimmed_monitor_frames: u32,
    /// Blocks where the monitoring amp failed and the dry signal was heard
    /// instead. Non-zero means the amp is switched on but not working.
    pub monitor_amp_faults: u32,
    pub disk_error: bool,
}

impl Default for RuntimeSnapshot {
    fn default() -> Self {
        Self {
            position_frames: 0,
            input_peaks: [0.0; 4],
            track_peaks: [0.0; MIXER_TRACK_CAPACITY],
            transport: RuntimeTransportState::Stopped,
            xruns: 0,
            dropped_record_frames: 0,
            trimmed_monitor_frames: 0,
            monitor_amp_faults: 0,
            disk_error: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RuntimeTransportState {
    #[default]
    Stopped,
    Playing,
    CountIn,
    Recording,
}

struct Shared {
    playing: AtomicBool,
    recording: AtomicBool,
    record_pending: AtomicBool,
    record_start_frame: AtomicU64,
    position: AtomicU64,
    tempo: AtomicU16,
    meter_numerator: AtomicU16,
    meter_denominator: AtomicU16,
    click_enabled: AtomicBool,
    click_level_bits: AtomicU32,
    input_peaks: [AtomicU32; 4],
    track_peaks: [AtomicU32; MIXER_TRACK_CAPACITY],
    record_left: AtomicUsize,
    record_right: AtomicUsize,
    xruns: AtomicU64,
    dropped_record_frames: AtomicU64,
    /// Input frames discarded to keep monitoring latency down. Steady growth
    /// means the input stream is running ahead of the output.
    trimmed_monitor_frames: AtomicU32,
    /// Blocks the monitoring amp failed to process and passed through dry.
    /// Non-zero means the amp is not being heard even though it is switched on.
    monitor_amp_faults: AtomicU32,
    writer_ready: AtomicBool,
    disk_error: AtomicBool,
    monitoring: AtomicBool,
    monitor_left: AtomicUsize,
    monitor_right: AtomicUsize,
    output_test_frames: AtomicU64,
    /// Playback speed as f32 bits. 1.0 is normal; other values resample the mix
    /// in real time, which changes pitch as well as tempo (varispeed).
    speed_bits: AtomicU32,
    /// Phase offset of the click grid, in frames. Shifts where bar one falls so
    /// the click can be lined up with a song whose first beat is not at frame 0.
    click_offset_frames: AtomicU64,
}

impl Shared {
    fn new() -> Self {
        Self {
            playing: AtomicBool::new(false),
            recording: AtomicBool::new(false),
            record_pending: AtomicBool::new(false),
            record_start_frame: AtomicU64::new(0),
            position: AtomicU64::new(0),
            tempo: AtomicU16::new(120),
            meter_numerator: AtomicU16::new(4),
            meter_denominator: AtomicU16::new(4),
            click_enabled: AtomicBool::new(true),
            click_level_bits: AtomicU32::new(0.35_f32.to_bits()),
            input_peaks: std::array::from_fn(|_| AtomicU32::new(0)),
            track_peaks: std::array::from_fn(|_| AtomicU32::new(0)),
            record_left: AtomicUsize::new(0),
            record_right: AtomicUsize::new(1),
            xruns: AtomicU64::new(0),
            dropped_record_frames: AtomicU64::new(0),
            trimmed_monitor_frames: AtomicU32::new(0),
            monitor_amp_faults: AtomicU32::new(0),
            writer_ready: AtomicBool::new(false),
            disk_error: AtomicBool::new(false),
            monitoring: AtomicBool::new(false),
            monitor_left: AtomicUsize::new(0),
            monitor_right: AtomicUsize::new(1),
            output_test_frames: AtomicU64::new(0),
            speed_bits: AtomicU32::new(1.0_f32.to_bits()),
            click_offset_frames: AtomicU64::new(0),
        }
    }
}

enum WriterCommand {
    Start { path: PathBuf, channels: u16 },
    Stop,
    Shutdown,
}

struct PlaybackClip {
    clip_id: u128,
    track_id: usize,
    audible: bool,
    start_frame: u64,
    samples: Arc<Vec<[f32; 2]>>,
    gain: f32,
    pan: f32,
}

/// One track's amplifier: a gate ahead of the model, the model, its tone
/// stack, and the levelling that lets one capture be swapped for another.
struct ActiveNam {
    processor: NamProcessor,
    enabled: bool,
    input_gain: f32,
    output_gain: f32,
    /// Gain that brings this capture to a common loudness, or `1.0` for a
    /// model that does not know its own.
    normalize_gain: f32,
    normalize: bool,
    gate: NoiseGate,
    gate_db: f32,
    tone: ToneStack,
    tone_enabled: bool,
    bass: f32,
    middle: f32,
    treble: f32,
    mono: Vec<f32>,
    /// The tone stack works in stereo; the amp itself is mono.
    stereo: Vec<[f32; 2]>,
}

impl ActiveNam {
    /// Builds the amp around a loaded model. Allocates; control thread only.
    fn new(processor: NamProcessor, params: ChannelStripParams, sample_rate: SampleRate) -> Self {
        // Captures vary by tens of decibels; levelling each against its own
        // measured loudness is what makes swapping one for another a change of
        // amp rather than a change of volume.
        let normalize_gain = processor.loudness().map_or(1.0, |loudness| {
            #[allow(clippy::cast_possible_truncation)]
            let difference = (NORMALIZE_TARGET_DB - loudness) as f32;
            db_to_linear(difference.clamp(-24.0, 24.0))
        });
        let mut amp = Self {
            processor,
            enabled: false,
            input_gain: 1.0,
            output_gain: 1.0,
            normalize_gain,
            normalize: false,
            gate: NoiseGate::new(sample_rate),
            gate_db: daw_engine::OPEN_THRESHOLD_DB,
            tone: ToneStack::new(sample_rate),
            tone_enabled: false,
            bass: 5.0,
            middle: 5.0,
            treble: 5.0,
            mono: vec![0.0; CLICK_SCRATCH_FRAMES],
            stereo: vec![[0.0; 2]; CLICK_SCRATCH_FRAMES],
        };
        amp.set_params(params);
        amp
    }

    /// Applies the parameters that can change while playing. No allocation:
    /// this runs on the audio thread.
    fn set_params(&mut self, params: ChannelStripParams) {
        self.enabled = params.nam_enabled;
        self.input_gain = db_to_linear(params.nam_input_db);
        self.output_gain = db_to_linear(params.nam_output_db);
        self.gate_db = params.nam_gate_db;
        self.tone_enabled = params.nam_tone_enabled;
        self.bass = params.nam_bass;
        self.middle = params.nam_middle;
        self.treble = params.nam_treble;
        self.normalize = params.nam_normalize;
    }

    /// The whole amp, in the order a real one is wired: gate, then the model,
    /// then its tone stack, then the output level.
    fn process_stereo(&mut self, frames: &mut [[f32; 2]]) {
        if !self.enabled {
            return;
        }
        let count = frames.len().min(self.mono.len()).min(self.stereo.len());
        for (sample, frame) in self.mono[..count].iter_mut().zip(frames.iter()) {
            *sample = (frame[0] + frame[1]) * 0.5 * self.input_gain;
        }
        // Ahead of the model, so the hiss between notes never reaches the gain.
        self.gate.process(&mut self.mono[..count], self.gate_db);

        let amped = self.processor.process(&mut self.mono[..count]).is_ok();
        let level = if amped {
            let normalize = if self.normalize { self.normalize_gain } else { 1.0 };
            self.output_gain * normalize
        } else {
            // Never silence: the player has to hear themselves whatever failed.
            if self.input_gain > 1e-6 { 1.0 / self.input_gain } else { 1.0 }
        };

        for (frame, sample) in self.stereo[..count].iter_mut().zip(&self.mono[..count]) {
            let output = *sample * level;
            *frame = [output, output];
        }
        if amped && self.tone_enabled {
            self.tone
                .process(&mut self.stereo[..count], self.bass, self.middle, self.treble);
        }
        frames[..count].copy_from_slice(&self.stereo[..count]);
    }
}

/// One instrument track's notes, already converted to absolute frames on the
/// control thread. The audio thread never sees ticks or tempo.
struct MidiPart {
    track_id: usize,
    audible: bool,
    /// Sorted by `start_frame`; the synth relies on that ordering.
    notes: Arc<Vec<ScheduledNote>>,
    gain: f32,
    pan: f32,
    /// General MIDI program this track plays.
    program: u8,
    /// True for a channel-10 style track, played by the drum kit.
    is_drum_kit: bool,
}

enum PlaybackCommand {
    Clear,
    Add(PlaybackClip),
    AddMidi(MidiPart),
    SetMonitorEffects(ChannelStripParams),
    SetTrackAudible {
        track_id: usize,
        audible: bool,
    },
    SetTrackMix {
        track_id: usize,
        gain: f32,
        pan: f32,
    },
    SetTrackEffects {
        track_id: usize,
        params: ChannelStripParams,
    },
    SetTrackNam {
        track_id: usize,
        model: Option<ActiveNam>,
    },
    SetMonitorNam {
        model: Option<ActiveNam>,
    },
    MoveClip {
        clip_id: u128,
        start_frame: u64,
        track_id: usize,
        gain: f32,
        pan: f32,
        audible: bool,
    },
}

struct OutputQueues {
    playback: Arc<ArrayQueue<PlaybackCommand>>,
    monitor: Arc<ArrayQueue<[f32; 2]>>,
    retired: Arc<ArrayQueue<PlaybackClip>>,
    retired_midi: Arc<ArrayQueue<MidiPart>>,
    retired_nam: Arc<ArrayQueue<ActiveNam>>,
}

pub struct AudioRuntime {
    shared: Arc<Shared>,
    writer_tx: Sender<WriterCommand>,
    writer_thread: Option<JoinHandle<()>>,
    playback_commands: Arc<ArrayQueue<PlaybackCommand>>,
    retired_playback: Arc<ArrayQueue<PlaybackClip>>,
    retired_midi: Arc<ArrayQueue<MidiPart>>,
    retired_nam: Arc<ArrayQueue<ActiveNam>>,
    playback_cache: Mutex<HashMap<PathBuf, Arc<Vec<[f32; 2]>>>>,
    _input_stream: Stream,
    _output_stream: Stream,
    sample_rate: SampleRate,
    input_name: String,
    output_name: String,
    input_channels: u16,
    output_channels: u16,
    buffer_frames: u32,
    /// The sound font instrument tracks are playing from, if any. Kept for
    /// display: the samples themselves live in the players on the audio thread.
    soundfont_name: Option<String>,
}

impl AudioRuntime {
    /// Opens the configured input/output streams and starts the disk writer.
    ///
    /// # Errors
    ///
    /// Returns an error when PulseAudio/PipeWire is unavailable, a matching
    /// device is missing, or either stream cannot be opened.
    #[allow(clippy::too_many_lines)]
    pub fn open(config: &AudioRuntimeConfig) -> Result<Self> {
        let host = select_host()?;
        let input = find_device(&host, &config.input_name_contains, true)?;
        let output = find_device(&host, &config.output_name_contains, false)?;
        let input_default = input
            .default_input_config()
            .context("selected input has no usable format")?;
        let output_default = output
            .default_output_config()
            .context("selected output has no usable format")?;
        let input_name = input
            .description()
            .context("selected input has no readable name")?
            .name()
            .to_owned();
        let output_name = output
            .description()
            .context("selected output has no readable name")?
            .name()
            .to_owned();

        // The engine clock follows the output device, since that is what drives
        // synthesis and playback. A single interface like a Scarlett reports the
        // same rate on both sides; a laptop's built-in mic and speakers often do
        // not, so the input stream is allowed to run at its own rate rather than
        // refusing to open. (Recording through a mismatched-rate input is out of
        // scope here — this path exists so playback works on any machine.)
        let sample_rate = SampleRate::new(output_default.sample_rate())
            .context("audio backend returned a zero output sample rate")?;
        let input_rate =
            SampleRate::new(input_default.sample_rate()).unwrap_or(sample_rate);

        let shared = Arc::new(Shared::new());
        let playback_commands = Arc::new(ArrayQueue::new(PLAYBACK_COMMAND_CAPACITY));
        let retired_playback = Arc::new(ArrayQueue::new(RETIRED_PLAYBACK_CAPACITY));
        let retired_midi = Arc::new(ArrayQueue::new(RETIRED_PLAYBACK_CAPACITY));
        let retired_nam = Arc::new(ArrayQueue::new(RETIRED_NAM_CAPACITY));
        let monitor_queue = Arc::new(ArrayQueue::new(MONITOR_QUEUE_FRAMES));
        let queue_capacity = usize::try_from(sample_rate.get())
            .unwrap_or(48_000)
            .saturating_mul(RECORD_QUEUE_SECONDS);
        let record_queue = Arc::new(ArrayQueue::new(queue_capacity));
        let (writer_tx, writer_rx) = mpsc::channel();
        let writer_queue = Arc::clone(&record_queue);
        let writer_shared = Arc::clone(&shared);
        let writer_thread = thread::Builder::new()
            .name("rustdaw-disk-writer".to_owned())
            .spawn(move || writer_loop(writer_rx, writer_queue, sample_rate, writer_shared))
            .context("failed to start recording writer")?;

        let input_config = StreamConfig {
            channels: input_default.channels(),
            sample_rate: input_rate.get(),
            buffer_size: cpal::BufferSize::Fixed(config.buffer_frames),
        };
        let output_config = StreamConfig {
            channels: output_default.channels(),
            sample_rate: sample_rate.get(),
            buffer_size: cpal::BufferSize::Fixed(config.buffer_frames),
        };

        let input_stream = build_input(
            &input,
            input_config,
            input_default.sample_format(),
            Arc::clone(&shared),
            record_queue,
            Arc::clone(&monitor_queue),
        )?;
        // Loading a sound font reads tens of megabytes off disk and allocates
        // heavily, so it happens here rather than anywhere near the callback.
        let soundfont = match &config.instruments {
            InstrumentSource::Synthesised => None,
            InstrumentSource::Auto => SoundFontBank::discover(),
            InstrumentSource::SoundFont(path) => Some(
                SoundFontBank::load(path)
                    .with_context(|| format!("failed to load {}", path.display()))?,
            ),
        };
        let soundfont_name = soundfont.as_ref().map(|bank| bank.name().to_owned());
        let output_stream = build_output(
            &output,
            output_config,
            output_default.sample_format(),
            Arc::clone(&shared),
            sample_rate,
            soundfont.as_ref(),
            OutputQueues {
                playback: Arc::clone(&playback_commands),
                monitor: monitor_queue,
                retired: Arc::clone(&retired_playback),
                retired_midi: Arc::clone(&retired_midi),
                retired_nam: Arc::clone(&retired_nam),
            },
        )?;
        input_stream
            .play()
            .context("failed to start input stream")?;
        output_stream
            .play()
            .context("failed to start output stream")?;

        Ok(Self {
            shared,
            writer_tx,
            writer_thread: Some(writer_thread),
            playback_commands,
            retired_playback,
            retired_midi,
            retired_nam,
            playback_cache: Mutex::new(HashMap::new()),
            _input_stream: input_stream,
            _output_stream: output_stream,
            sample_rate,
            input_name,
            output_name,
            input_channels: input_default.channels(),
            output_channels: output_default.channels(),
            buffer_frames: config.buffer_frames,
            soundfont_name,
        })
    }

    #[must_use]
    pub const fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }

    /// The sound font instrument tracks are playing from, or `None` when they
    /// are playing the synthesised bank.
    #[must_use]
    pub fn soundfont_name(&self) -> Option<&str> {
        self.soundfont_name.as_deref()
    }

    #[must_use]
    pub fn input_name(&self) -> &str {
        &self.input_name
    }

    #[must_use]
    pub fn output_name(&self) -> &str {
        &self.output_name
    }

    #[must_use]
    pub const fn input_channels(&self) -> u16 {
        self.input_channels
    }

    #[must_use]
    pub const fn output_channels(&self) -> u16 {
        self.output_channels
    }

    #[must_use]
    pub const fn buffer_frames(&self) -> u32 {
        self.buffer_frames
    }

    pub fn trigger_output_test(&self) {
        self.shared
            .output_test_frames
            .store(u64::from(self.sample_rate.get()), Ordering::Release);
    }

    pub fn play(&self) {
        self.shared.playing.store(true, Ordering::Release);
    }

    pub fn stop(&self) {
        let was_recording = self.shared.recording.swap(false, Ordering::AcqRel);
        let was_pending = self.shared.record_pending.swap(false, Ordering::AcqRel);
        self.shared.playing.store(false, Ordering::Release);
        let _ = self.writer_tx.send(WriterCommand::Stop);
        if was_recording || was_pending {
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while self.shared.writer_ready.load(Ordering::Acquire)
                && std::time::Instant::now() < deadline
            {
                thread::sleep(Duration::from_millis(1));
            }
        }
    }

    pub fn seek_to_start(&self) {
        self.shared.position.store(0, Ordering::Release);
    }

    pub fn seek(&self, position: SamplePosition) {
        self.shared
            .position
            .store(position.get(), Ordering::Release);
    }

    /// Removes all currently scheduled playback clips.
    ///
    /// # Errors
    ///
    /// Returns an error when the bounded real-time command queue is full.
    pub fn clear_playback(&self) -> Result<()> {
        self.collect_retired_playback();
        self.playback_commands
            .push(PlaybackCommand::Clear)
            .map_err(|_| anyhow::anyhow!("playback command queue is full"))
    }

    /// Loads a WAV file on the control thread and schedules it for playback.
    ///
    /// # Errors
    ///
    /// Returns an error when the WAV cannot be read, its sample rate/layout is
    /// unsupported, or the bounded real-time command queue is full.
    pub fn add_playback_file(
        &self,
        path: &std::path::Path,
        start_frame: u64,
        gain: f32,
    ) -> Result<()> {
        self.collect_retired_playback();
        self.add_playback_file_with_effects(path, start_frame, gain, ChannelStripParams::default())
    }

    /// Loads, processes, and schedules a WAV file for playback.
    ///
    /// # Errors
    ///
    /// Returns an error when media loading or command scheduling fails.
    pub fn add_playback_file_with_effects(
        &self,
        path: &std::path::Path,
        start_frame: u64,
        gain: f32,
        effects: ChannelStripParams,
    ) -> Result<()> {
        self.add_track_playback_file(path, start_frame, gain, effects, 0.0, 0, true)
    }

    /// Loads and schedules a WAV associated with a specific mixer track.
    ///
    /// # Errors
    ///
    /// Returns an error when media loading or command scheduling fails.
    #[allow(clippy::too_many_arguments)]
    pub fn add_track_playback_file(
        &self,
        path: &std::path::Path,
        start_frame: u64,
        gain: f32,
        effects: ChannelStripParams,
        pan: f32,
        track_id: usize,
        audible: bool,
    ) -> Result<()> {
        self.add_identified_track_playback_file(
            path,
            start_frame,
            gain,
            effects,
            pan,
            track_id,
            track_id as u128,
            audible,
        )
    }

    /// Loads and schedules a WAV with a stable clip identifier.
    ///
    /// # Errors
    ///
    /// Returns an error when media loading or command scheduling fails.
    #[allow(clippy::too_many_arguments)]
    pub fn add_identified_track_playback_file(
        &self,
        path: &std::path::Path,
        start_frame: u64,
        gain: f32,
        _effects: ChannelStripParams,
        pan: f32,
        track_id: usize,
        clip_id: u128,
        audible: bool,
    ) -> Result<()> {
        self.collect_retired_playback();
        let source = {
            let cached = self
                .playback_cache
                .lock()
                .map_err(|_| anyhow::anyhow!("playback cache is unavailable"))?
                .get(path)
                .cloned();
            if let Some(cached) = cached {
                cached
            } else {
                let loaded = Arc::new(load_wav(path, self.sample_rate)?);
                self.playback_cache
                    .lock()
                    .map_err(|_| anyhow::anyhow!("playback cache is unavailable"))?
                    .insert(path.to_path_buf(), Arc::clone(&loaded));
                loaded
            }
        };
        let samples = source;
        self.playback_commands
            .push(PlaybackCommand::Add(PlaybackClip {
                clip_id,
                track_id,
                audible,
                start_frame,
                samples,
                gain: gain.clamp(0.0, 4.0),
                pan: pan.clamp(-1.0, 1.0),
            }))
            .map_err(|_| anyhow::anyhow!("playback command queue is full"))
    }

    /// Moves an already loaded clip without reopening or reprocessing its WAV.
    ///
    /// # Errors
    ///
    /// Returns an error when the real-time command queue is full.
    #[allow(clippy::too_many_arguments)]
    pub fn move_playback_clip(
        &self,
        clip_id: u128,
        start_frame: u64,
        track_id: usize,
        gain: f32,
        pan: f32,
        audible: bool,
    ) -> Result<()> {
        self.playback_commands
            .push(PlaybackCommand::MoveClip {
                clip_id,
                start_frame,
                track_id,
                gain: gain.clamp(0.0, 4.0),
                pan: pan.clamp(-1.0, 1.0),
                audible,
            })
            .map_err(|_| anyhow::anyhow!("audio command queue is full"))
    }

    /// Changes one track's playback audibility without reloading its media.
    ///
    /// # Errors
    ///
    /// Returns an error when the real-time command queue is full.
    /// Schedules one instrument track's notes.
    ///
    /// `notes` must already be in absolute frames and sorted by start; use
    /// [`daw_midi::MidiClip::schedule`] to convert from musical time. The
    /// conversion stays on the control thread so the audio thread never
    /// consults a tempo map.
    ///
    /// # Errors
    ///
    /// Returns an error when the real-time command queue is full.
    #[allow(clippy::too_many_arguments)]
    pub fn add_midi_track(
        &self,
        track_id: usize,
        notes: Vec<ScheduledNote>,
        gain: f32,
        pan: f32,
        audible: bool,
        program: u8,
        is_drum_kit: bool,
    ) -> Result<()> {
        self.collect_retired_playback();
        debug_assert!(
            notes.windows(2).all(|pair| pair[0].start_frame <= pair[1].start_frame),
            "notes must be sorted before reaching the audio thread"
        );
        self.playback_commands
            .push(PlaybackCommand::AddMidi(MidiPart {
                track_id,
                audible,
                notes: Arc::new(notes),
                gain: gain.clamp(0.0, 4.0),
                pan: pan.clamp(-1.0, 1.0),
                program: program.min(127),
                is_drum_kit,
            }))
            .map_err(|_| anyhow::anyhow!("audio command queue is full"))
    }

    /// Updates one track's audibility for both its clips and its notes.
    ///
    /// # Errors
    ///
    /// Returns an error when the real-time command queue is full.
    pub fn set_track_audible(&self, track_id: usize, audible: bool) -> Result<()> {
        self.playback_commands
            .push(PlaybackCommand::SetTrackAudible { track_id, audible })
            .map_err(|_| anyhow::anyhow!("audio command queue is full"))
    }

    /// Updates one track's gain and pan without reloading its media.
    ///
    /// # Errors
    ///
    /// Returns an error when the real-time command queue is full.
    pub fn set_track_mix(&self, track_id: usize, gain: f32, pan: f32) -> Result<()> {
        self.playback_commands
            .push(PlaybackCommand::SetTrackMix {
                track_id,
                gain: gain.clamp(0.0, 4.0),
                pan: pan.clamp(-1.0, 1.0),
            })
            .map_err(|_| anyhow::anyhow!("audio command queue is full"))
    }

    /// Updates one track's channel-strip parameters without rebuilding audio.
    ///
    /// # Errors
    ///
    /// Returns an error when the real-time command queue is full.
    pub fn set_track_effects(&self, track_id: usize, params: ChannelStripParams) -> Result<()> {
        self.playback_commands
            .push(PlaybackCommand::SetTrackEffects { track_id, params })
            .map_err(|_| anyhow::anyhow!("audio command queue is full"))
    }

    /// Loads and activates a NAM capture for one guitar track. Model parsing and
    /// prewarming happen here, never in the audio callback.
    pub fn set_track_nam_model(
        &self,
        track_id: usize,
        path: Option<&std::path::Path>,
        params: ChannelStripParams,
    ) -> Result<()> {
        self.collect_retired_playback();
        let model = path
            // Prepared for the largest block the render path can hand it, which
            // is the callback chunk size and not the requested buffer size: the
            // backend is free to deliver more frames than were asked for, and a
            // model prepared for fewer refuses the block and is bypassed in
            // silence. `mono` is sized to match.
            .map(|path| NamProcessor::load(path, self.sample_rate.get(), CLICK_SCRATCH_FRAMES))
            .transpose()
            .map_err(anyhow::Error::msg)?
            .map(|processor| ActiveNam::new(processor, params, self.sample_rate));
        self.playback_commands
            .push(PlaybackCommand::SetTrackNam { track_id, model })
            .map_err(|_| anyhow::anyhow!("audio command queue is full"))
    }

    /// Loads the NAM capture used by the currently monitored guitar input.
    pub fn set_monitor_nam_model(
        &self,
        path: Option<&std::path::Path>,
        params: ChannelStripParams,
    ) -> Result<()> {
        self.collect_retired_playback();
        let model = path
            // Prepared for the largest block the render path can hand it, which
            // is the callback chunk size and not the requested buffer size: the
            // backend is free to deliver more frames than were asked for, and a
            // model prepared for fewer refuses the block and is bypassed in
            // silence. `mono` is sized to match.
            .map(|path| NamProcessor::load(path, self.sample_rate.get(), CLICK_SCRATCH_FRAMES))
            .transpose()
            .map_err(anyhow::Error::msg)?
            .map(|processor| ActiveNam::new(processor, params, self.sample_rate));
        self.playback_commands
            .push(PlaybackCommand::SetMonitorNam { model })
            .map_err(|_| anyhow::anyhow!("audio command queue is full"))
    }

    /// Updates the non-destructive channel strip used for software monitoring.
    ///
    /// # Errors
    ///
    /// Returns an error when the real-time command queue is full.
    pub fn set_monitor_effects(&self, effects: ChannelStripParams) -> Result<()> {
        self.playback_commands
            .push(PlaybackCommand::SetMonitorEffects(effects))
            .map_err(|_| anyhow::anyhow!("audio command queue is full"))
    }

    fn collect_retired_playback(&self) {
        while self.retired_playback.pop().is_some() {}
        while self.retired_midi.pop().is_some() {}
        while self.retired_nam.pop().is_some() {}
    }

    pub fn set_tempo(&self, bpm: u16) {
        self.shared
            .tempo
            .store(bpm.clamp(20, 300), Ordering::Release);
    }

    /// Sets the real-time playback speed. 1.0 is normal; higher is faster (and
    /// higher-pitched), lower is slower. Clamped to a sane range.
    pub fn set_speed(&self, speed: f32) {
        let clamped = if speed.is_finite() {
            speed.clamp(MIN_SPEED, MAX_SPEED)
        } else {
            1.0
        };
        self.shared
            .speed_bits
            .store(clamped.to_bits(), Ordering::Release);
    }

    #[must_use]
    pub fn speed(&self) -> f32 {
        f32::from_bits(self.shared.speed_bits.load(Ordering::Acquire))
    }

    pub fn set_meter(&self, numerator: u16, denominator: u16) {
        self.shared
            .meter_numerator
            .store(numerator.max(1), Ordering::Release);
        self.shared
            .meter_denominator
            .store(denominator.max(1), Ordering::Release);
    }

    pub fn set_click(&self, enabled: bool, level: f32) {
        self.shared.click_enabled.store(enabled, Ordering::Release);
        self.shared
            .click_level_bits
            .store(level.clamp(0.0, 1.0).to_bits(), Ordering::Release);
    }

    /// Shifts the click grid so bar one lands `frames` later, lining the click
    /// up with a song whose first beat is not at frame zero.
    pub fn set_click_offset(&self, frames: u64) {
        self.shared
            .click_offset_frames
            .store(frames, Ordering::Release);
    }

    pub fn set_monitoring(&self, enabled: bool, left_channel: usize, right_channel: usize) {
        self.shared
            .monitor_left
            .store(left_channel, Ordering::Release);
        self.shared
            .monitor_right
            .store(right_channel, Ordering::Release);
        self.shared.monitoring.store(enabled, Ordering::Release);
    }

    /// Begins recording the chosen input route to a WAV file.
    ///
    /// # Errors
    ///
    /// Returns the exact timeline frame at which capture is scheduled.
    ///
    /// # Errors
    ///
    /// Returns an error if the disk writer has terminated or cannot open the
    /// destination file.
    pub fn start_recording(
        &self,
        path: PathBuf,
        layout: ChannelLayout,
        left_channel: usize,
        right_channel: usize,
        count_in_bars: u16,
    ) -> Result<u64> {
        let channels = u16::try_from(layout.channel_count()).unwrap_or(1);
        self.shared.writer_ready.store(false, Ordering::Release);
        self.shared.disk_error.store(false, Ordering::Release);
        self.shared
            .record_left
            .store(left_channel, Ordering::Release);
        self.shared
            .record_right
            .store(right_channel, Ordering::Release);
        self.writer_tx
            .send(WriterCommand::Start { path, channels })
            .context("recording writer is unavailable")?;
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while !self.shared.writer_ready.load(Ordering::Acquire) {
            if self.shared.disk_error.load(Ordering::Acquire) {
                bail!("failed to create the recording WAV file");
            }
            if std::time::Instant::now() >= deadline {
                bail!("recording writer did not become ready");
            }
            thread::sleep(Duration::from_millis(1));
        }
        let tempo = u64::from(self.shared.tempo.load(Ordering::Acquire));
        let count_in_frames = calculate_count_in_frames(
            self.sample_rate,
            tempo,
            self.shared.meter_numerator.load(Ordering::Acquire),
            self.shared.meter_denominator.load(Ordering::Acquire),
            count_in_bars,
        );
        let start_frame = self
            .shared
            .position
            .load(Ordering::Acquire)
            .saturating_add(count_in_frames);
        self.shared
            .record_start_frame
            .store(start_frame, Ordering::Release);
        self.shared.playing.store(true, Ordering::Release);
        if count_in_bars == 0 {
            self.shared.recording.store(true, Ordering::Release);
            self.shared.record_pending.store(false, Ordering::Release);
        } else {
            self.shared.recording.store(false, Ordering::Release);
            self.shared.record_pending.store(true, Ordering::Release);
        }
        Ok(start_frame)
    }

    #[must_use]
    pub fn snapshot(&self) -> RuntimeSnapshot {
        let transport = if self.shared.recording.load(Ordering::Acquire) {
            RuntimeTransportState::Recording
        } else if self.shared.record_pending.load(Ordering::Acquire) {
            RuntimeTransportState::CountIn
        } else if self.shared.playing.load(Ordering::Acquire) {
            RuntimeTransportState::Playing
        } else {
            RuntimeTransportState::Stopped
        };
        RuntimeSnapshot {
            position_frames: self.shared.position.load(Ordering::Acquire),
            input_peaks: std::array::from_fn(|index| {
                f32::from_bits(self.shared.input_peaks[index].load(Ordering::Relaxed))
            }),
            track_peaks: std::array::from_fn(|index| {
                f32::from_bits(self.shared.track_peaks[index].load(Ordering::Relaxed))
            }),
            transport,
            xruns: self.shared.xruns.load(Ordering::Relaxed),
            dropped_record_frames: self.shared.dropped_record_frames.load(Ordering::Relaxed),
            trimmed_monitor_frames: self
                .shared
                .trimmed_monitor_frames
                .load(Ordering::Relaxed),
            monitor_amp_faults: self.shared.monitor_amp_faults.load(Ordering::Relaxed),
            disk_error: self.shared.disk_error.load(Ordering::Acquire),
        }
    }
}

impl Drop for AudioRuntime {
    fn drop(&mut self) {
        self.shared.recording.store(false, Ordering::Release);
        self.shared.record_pending.store(false, Ordering::Release);
        let _ = self.writer_tx.send(WriterCommand::Shutdown);
        if let Some(thread) = self.writer_thread.take() {
            let _ = thread.join();
        }
    }
}

/// Chooses the audio host to open streams on.
///
/// Linux drives `PipeWire` through its `PulseAudio` compatibility host, which is
/// what this backend was built for. On macOS and Windows that host does not
/// exist, so fall back to the platform default (`CoreAudio` / WASAPI) and the
/// app runs against the built-in interface instead.
fn select_host() -> Result<cpal::Host> {
    if let Some(id) = cpal::available_hosts()
        .into_iter()
        .find(|id| id.name() == "PulseAudio")
    {
        return cpal::host_from_id(id).context("failed to connect to PipeWire audio");
    }
    Ok(cpal::default_host())
}

fn find_device(host: &cpal::Host, pattern: &str, input: bool) -> Result<cpal::Device> {
    let devices = if input {
        host.input_devices()
            .context("failed to enumerate input devices")?
    } else {
        host.output_devices()
            .context("failed to enumerate output devices")?
    };
    let mut matches = devices.filter_map(|device| {
        let name = device.description().ok()?.name().to_owned();
        if input && is_output_monitor_name(&name) {
            return None;
        }
        name.contains(pattern).then_some((name, device))
    });
    let mut fallback = None;
    let selected = matches.find_map(|(name, device)| {
        if name == pattern {
            Some(device)
        } else {
            if fallback.is_none() {
                fallback = Some(device);
            }
            None
        }
    });
    if let Some(device) = selected.or(fallback) {
        return Ok(device);
    }
    // Nothing matched the preferred name — e.g. no Scarlett on a laptop — so use
    // the host's default device, which is the machine's built-in interface.
    let default = if input {
        host.default_input_device()
    } else {
        host.default_output_device()
    };
    default.with_context(|| {
        format!("no audio device contains ‘{pattern}’ and the host has no default device")
    })
}

fn is_output_monitor_name(name: &str) -> bool {
    let lowercase = name.to_ascii_lowercase();
    lowercase.starts_with("monitor of ") || lowercase.ends_with(".monitor")
}

fn build_input(
    device: &cpal::Device,
    config: StreamConfig,
    format: SampleFormat,
    shared: Arc<Shared>,
    queue: Arc<ArrayQueue<[f32; 2]>>,
    monitor_queue: Arc<ArrayQueue<[f32; 2]>>,
) -> Result<Stream> {
    let channels = usize::from(config.channels);
    let stream = match format {
        SampleFormat::F32 => input_stream(
            device,
            config,
            shared,
            queue,
            monitor_queue,
            channels,
            |value: f32| value,
        ),
        SampleFormat::I16 => input_stream(
            device,
            config,
            shared,
            queue,
            monitor_queue,
            channels,
            |value: i16| f32::from(value) / f32::from(i16::MAX),
        ),
        SampleFormat::I32 => input_stream(
            device,
            config,
            shared,
            queue,
            monitor_queue,
            channels,
            |value: i32| {
                #[allow(clippy::cast_precision_loss)]
                {
                    value as f32 / i32::MAX as f32
                }
            },
        ),
        unsupported => bail!("unsupported input sample format {unsupported}"),
    }?;
    Ok(stream)
}

fn input_stream<T: cpal::SizedSample + Copy + Send + 'static>(
    device: &cpal::Device,
    config: StreamConfig,
    shared: Arc<Shared>,
    queue: Arc<ArrayQueue<[f32; 2]>>,
    monitor_queue: Arc<ArrayQueue<[f32; 2]>>,
    channels: usize,
    convert: fn(T) -> f32,
) -> Result<Stream> {
    let error_shared = Arc::clone(&shared);
    device
        .build_input_stream::<T, _, _>(
            config,
            move |data, _| {
                capture_input(data, channels, convert, &shared, &queue, &monitor_queue);
            },
            move |_| {
                error_shared.xruns.fetch_add(1, Ordering::Relaxed);
            },
            Some(Duration::from_secs(2)),
        )
        .context("failed to build input stream")
}

fn capture_input<T: Copy>(
    data: &[T],
    channels: usize,
    convert: fn(T) -> f32,
    shared: &Shared,
    queue: &ArrayQueue<[f32; 2]>,
    monitor_queue: &ArrayQueue<[f32; 2]>,
) {
    if channels == 0 {
        return;
    }
    let mut peaks = [0.0_f32; 4];
    let left = shared.record_left.load(Ordering::Relaxed).min(channels - 1);
    let right = shared
        .record_right
        .load(Ordering::Relaxed)
        .min(channels - 1);
    let recording = shared.recording.load(Ordering::Acquire);
    let monitoring = shared.monitoring.load(Ordering::Acquire);
    let monitor_left = shared
        .monitor_left
        .load(Ordering::Relaxed)
        .min(channels - 1);
    let monitor_right = shared
        .monitor_right
        .load(Ordering::Relaxed)
        .min(channels - 1);

    for frame in data.chunks_exact(channels) {
        for (index, sample) in frame.iter().take(4).enumerate() {
            peaks[index] = peaks[index].max(convert(*sample).abs());
        }
        if recording {
            let captured = [convert(frame[left]), convert(frame[right])];
            if queue.push(captured).is_err() {
                shared.dropped_record_frames.fetch_add(1, Ordering::Relaxed);
            }
        }
        if monitoring {
            let _ =
                monitor_queue.push([convert(frame[monitor_left]), convert(frame[monitor_right])]);
        }
    }
    for (atom, peak) in shared.input_peaks.iter().zip(peaks) {
        let old = f32::from_bits(atom.load(Ordering::Relaxed));
        atom.store((old * 0.82).max(peak).to_bits(), Ordering::Relaxed);
    }
}

fn build_output(
    device: &cpal::Device,
    config: StreamConfig,
    format: SampleFormat,
    shared: Arc<Shared>,
    sample_rate: SampleRate,
    soundfont: Option<&SoundFontBank>,
    queues: OutputQueues,
) -> Result<Stream> {
    let channels = usize::from(config.channels);
    let stream = match format {
        SampleFormat::F32 => output_stream(
            device,
            config,
            shared,
            channels,
            sample_rate,
            soundfont,
            queues,
            |value| value,
        ),
        SampleFormat::I16 => output_stream(
            device,
            config,
            shared,
            channels,
            sample_rate,
            soundfont,
            queues,
            |value| {
                #[allow(clippy::cast_possible_truncation)]
                {
                    (value.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16
                }
            },
        ),
        SampleFormat::I32 => output_stream(
            device,
            config,
            shared,
            channels,
            sample_rate,
            soundfont,
            queues,
            |value| {
                #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
                {
                    (value.clamp(-1.0, 1.0) * i32::MAX as f32) as i32
                }
            },
        ),
        unsupported => bail!("unsupported output sample format {unsupported}"),
    }?;
    Ok(stream)
}

#[allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
#[allow(clippy::too_many_arguments)]
fn output_stream<T: cpal::SizedSample + Copy + Send + 'static>(
    device: &cpal::Device,
    config: StreamConfig,
    shared: Arc<Shared>,
    channels: usize,
    sample_rate: SampleRate,
    soundfont: Option<&SoundFontBank>,
    queues: OutputQueues,
    convert: fn(f32) -> T,
) -> Result<Stream> {
    let error_shared = Arc::clone(&shared);
    let mut scratch_left = [0.0_f32; CLICK_SCRATCH_FRAMES];
    let mut scratch_right = [0.0_f32; CLICK_SCRATCH_FRAMES];
    let mut playback_slots: [Option<PlaybackClip>; PLAYBACK_SLOT_CAPACITY] =
        std::array::from_fn(|_| None);
    let mut midi_slots: [Option<MidiPart>; MIDI_SLOT_CAPACITY] = std::array::from_fn(|_| None);
    let mut mixer = TrackMixer::new(sample_rate, soundfont)?;
    let mut monitor_effects = ChannelStrip::new(sample_rate, ChannelStripParams::default());
    let mut monitor_nam: Option<ActiveNam> = None;
    let mut was_playing = false;
    let mut test_phase = 0.0_f32;
    // Pitch-preserving time-stretcher, carried across callbacks so its overlap
    // buffers stay continuous. Engaged only when the speed is not 1.0.
    let mut stretcher = TimeStretcher::new();
    device
        .build_output_stream::<T, _, _>(
            config,
            move |data, _| {
                while let Some(command) = queues.playback.pop() {
                    match command {
                        PlaybackCommand::Clear => {
                            for slot in &mut playback_slots {
                                if let Some(clip) = slot.take() {
                                    if let Err(unretired) = queues.retired.push(clip) {
                                        *slot = Some(unretired);
                                    }
                                }
                            }
                            for slot in &mut midi_slots {
                                if let Some(part) = slot.take() {
                                    if let Err(unretired) = queues.retired_midi.push(part) {
                                        *slot = Some(unretired);
                                    }
                                }
                            }
                            mixer.silence_tracks();
                        }
                        PlaybackCommand::Add(clip) => {
                            if let Some(slot) =
                                playback_slots.iter_mut().find(|slot| slot.is_none())
                            {
                                *slot = Some(clip);
                            }
                        }
                        PlaybackCommand::AddMidi(part) => {
                            if let Some(slot) = midi_slots.iter_mut().find(|slot| slot.is_none()) {
                                *slot = Some(part);
                            }
                        }
                        PlaybackCommand::SetMonitorEffects(params) => {
                            monitor_effects.set_params(params);
                            if let Some(nam) = &mut monitor_nam {
                                nam.set_params(params);
                            }
                        }
                        PlaybackCommand::SetTrackAudible { track_id, audible } => {
                            set_track_audible_in_slots(&mut playback_slots, track_id, audible);
                            for part in midi_slots.iter_mut().flatten() {
                                if part.track_id == track_id {
                                    part.audible = audible;
                                }
                            }
                        }
                        PlaybackCommand::SetTrackMix {
                            track_id,
                            gain,
                            pan,
                        } => {
                            for clip in playback_slots.iter_mut().flatten() {
                                if clip.track_id == track_id {
                                    clip.gain = gain;
                                    clip.pan = pan;
                                }
                            }
                            for part in midi_slots.iter_mut().flatten() {
                                if part.track_id == track_id {
                                    part.gain = gain;
                                    part.pan = pan;
                                }
                            }
                        }
                        PlaybackCommand::SetTrackEffects { track_id, params } => {
                            mixer.set_track_effects(track_id, params);
                        }
                        PlaybackCommand::SetTrackNam {
                            track_id,
                            model,
                        } => mixer.set_track_nam(
                            track_id,
                            model,
                            &queues.retired_nam,
                        ),
                        PlaybackCommand::SetMonitorNam { model } => {
                            if let Some(old) = monitor_nam.take() {
                                let _ = queues.retired_nam.push(old);
                            }
                            monitor_nam = model;
                        }
                        PlaybackCommand::MoveClip {
                            clip_id,
                            start_frame,
                            track_id,
                            gain,
                            pan,
                            audible,
                        } => move_clip_in_slots(
                            &mut playback_slots,
                            clip_id,
                            start_frame,
                            track_id,
                            gain,
                            pan,
                            audible,
                        ),
                    }
                }
                data.fill(convert(0.0));
                let playing = shared.playing.load(Ordering::Acquire);
                if was_playing != playing {
                    if !playing {
                        // Stopping must not leave a held note ringing.
                        mixer.silence_tracks();
                    }
                    // The stretcher's carried history is meaningless across a
                    // start/stop; reset it so playback resumes cleanly.
                    stretcher.reset();
                }
                was_playing = playing;
                // Monitoring has to work with the transport stopped: that is
                // how anyone practises, sets a level, or dials in an amp. Only
                // an idle engine with nothing to hear may skip the work.
                let monitoring = shared.monitoring.load(Ordering::Acquire);
                if channels == 0
                    || (!playing
                        && !monitoring
                        && shared.output_test_frames.load(Ordering::Acquire) == 0)
                {
                    return;
                }
                // Speed only bends live playback; a stopped output test stays 1:1.
                let speed = if playing {
                    f32::from_bits(shared.speed_bits.load(Ordering::Relaxed))
                        .clamp(MIN_SPEED, MAX_SPEED)
                } else {
                    1.0
                };
                let mut position = shared.position.load(Ordering::Relaxed);
                activate_recording_if_due(&shared, position);
                let tempo = shared.tempo.load(Ordering::Relaxed);
                let meter_numerator = shared.meter_numerator.load(Ordering::Relaxed);
                let meter_denominator = shared.meter_denominator.load(Ordering::Relaxed);
                let click_enabled = shared.click_enabled.load(Ordering::Relaxed);
                let click_level = f32::from_bits(shared.click_level_bits.load(Ordering::Relaxed));

                for output_chunk in data.chunks_mut(channels * CLICK_SCRATCH_FRAMES) {
                    let frames = output_chunk.len() / channels;
                    if !playing || (speed - 1.0).abs() < 1e-4 {
                        // Normal 1:1 path — unchanged, and the only path that can
                        // sound the stopped output-test tone.
                        render_source_block(
                            position, frames, playing, click_enabled, click_level, tempo,
                            meter_numerator, meter_denominator, sample_rate, &mut scratch_left,
                            &mut scratch_right, &mut mixer, &playback_slots, &midi_slots, &shared,
                            &queues.monitor, &mut monitor_effects, &mut monitor_nam,
                        );
                        let test_frames = shared.output_test_frames.load(Ordering::Relaxed);
                        let rendered_test_frames =
                            usize::try_from(test_frames.min(frames as u64)).unwrap_or(frames);
                        for (left, right) in scratch_left[..rendered_test_frames]
                            .iter_mut()
                            .zip(&mut scratch_right[..rendered_test_frames])
                        {
                            let sample = test_phase.sin() * 0.15;
                            *left += sample;
                            *right += sample;
                            #[allow(clippy::cast_precision_loss)]
                            {
                                test_phase +=
                                    std::f32::consts::TAU * 440.0 / sample_rate.get() as f32;
                            }
                            if test_phase >= std::f32::consts::TAU {
                                test_phase -= std::f32::consts::TAU;
                            }
                        }
                        if rendered_test_frames > 0 {
                            shared.output_test_frames.store(
                                test_frames.saturating_sub(rendered_test_frames as u64),
                                Ordering::Relaxed,
                            );
                        }
                        for ((frame, left), right) in output_chunk
                            .chunks_exact_mut(channels)
                            .zip(&scratch_left[..frames])
                            .zip(&scratch_right[..frames])
                        {
                            if let Some(sample) = frame.first_mut() {
                                *sample = convert(*left);
                            }
                            if let Some(sample) = frame.get_mut(1) {
                                *sample = convert(*right);
                            }
                        }
                        if playing {
                            position = position.saturating_add(frames as u64);
                        }
                    } else {
                        // Pitch-preserving time-stretch. The stretcher pulls the
                        // mix forward through `render`, which advances the
                        // transport by exactly the source it consumes so the
                        // playhead still tracks the audio. The click is left out
                        // here so it need not be re-timed against the stretch.
                        let ratio = f64::from(speed);
                        stretcher.process(
                            &mut scratch_left[..frames],
                            &mut scratch_right[..frames],
                            ratio,
                            |count, source_left, source_right| {
                                render_source_block(
                                    position, count, true, false, click_level, tempo,
                                    meter_numerator, meter_denominator, sample_rate, source_left,
                                    source_right, &mut mixer, &playback_slots, &midi_slots, &shared,
                                    &queues.monitor, &mut monitor_effects, &mut monitor_nam,
                                );
                                position = position.saturating_add(count as u64);
                            },
                        );
                        for ((frame, left), right) in output_chunk
                            .chunks_exact_mut(channels)
                            .zip(&scratch_left[..frames])
                            .zip(&scratch_right[..frames])
                        {
                            if let Some(sample) = frame.first_mut() {
                                *sample = convert(*left);
                            }
                            if let Some(sample) = frame.get_mut(1) {
                                *sample = convert(*right);
                            }
                        }
                    }
                }
                shared.position.store(position, Ordering::Release);
            },
            move |_| {
                error_shared.xruns.fetch_add(1, Ordering::Relaxed);
            },
            Some(Duration::from_secs(2)),
        )
        .context("failed to build output stream")
}

/// Renders `frames` of the mix at `position` into the scratch buffers: the
/// click, every track through the mixer, and software monitoring. Shared by the
/// normal 1:1 path and the varispeed resampler, so both produce the same mix.
#[allow(clippy::too_many_arguments)]
fn render_source_block(
    position: u64,
    frames: usize,
    playing: bool,
    click_enabled: bool,
    click_level: f32,
    tempo: u16,
    meter_numerator: u16,
    meter_denominator: u16,
    sample_rate: SampleRate,
    scratch_left: &mut [f32],
    scratch_right: &mut [f32],
    mixer: &mut TrackMixer,
    playback_slots: &[Option<PlaybackClip>; PLAYBACK_SLOT_CAPACITY],
    midi_slots: &[Option<MidiPart>; MIDI_SLOT_CAPACITY],
    shared: &Shared,
    monitor_queue: &ArrayQueue<[f32; 2]>,
    monitor_effects: &mut ChannelStrip,
    monitor_nam: &mut Option<ActiveNam>,
) {
    scratch_left[..frames].fill(0.0);
    scratch_right[..frames].fill(0.0);
    if click_enabled && playing {
        if let Ok(mut metronome) =
            Metronome::with_meter(sample_rate, tempo, meter_numerator, meter_denominator)
        {
            metronome.set_level(click_level);
            // Shift the grid by the phase offset so bar one can be lined up with
            // the song's first beat rather than always sitting at frame zero.
            let click_offset = shared.click_offset_frames.load(Ordering::Relaxed);
            metronome.render_mono(
                SamplePosition::new(position.saturating_sub(click_offset)),
                &mut scratch_left[..frames],
            );
            scratch_right[..frames].copy_from_slice(&scratch_left[..frames]);
        }
    }
    let mut block_track_peaks = [0.0_f32; MIXER_TRACK_CAPACITY];
    // Only while the transport runs. Stopped, the position does not advance,
    // so rendering the timeline would emit the same block over and over — a
    // stuck fragment of whatever the playhead happens to be sitting on, and a
    // note retriggered on every callback. Monitoring below is unaffected: it
    // reads the live input, which has nothing to do with the playhead.
    if playing {
        mixer.render(
            playback_slots,
            midi_slots,
            position,
            &mut scratch_left[..frames],
            &mut scratch_right[..frames],
            &mut block_track_peaks,
        );
    }
    for (atomic, peak) in shared.track_peaks.iter().zip(block_track_peaks) {
        let previous = f32::from_bits(atomic.load(Ordering::Relaxed));
        atomic.store((previous * 0.86).max(peak).to_bits(), Ordering::Relaxed);
    }
    mix_monitoring(
        shared,
        monitor_queue,
        &mut scratch_left[..frames],
        &mut scratch_right[..frames],
        monitor_effects,
        monitor_nam,
    );
}

fn mix_monitoring(
    shared: &Shared,
    queue: &ArrayQueue<[f32; 2]>,
    output_left: &mut [f32],
    output_right: &mut [f32],
    effects: &mut ChannelStrip,
    nam: &mut Option<ActiveNam>,
) {
    if !shared.monitoring.load(Ordering::Acquire) {
        while queue.pop().is_some() {}
        return;
    }
    trim_monitor_queue(queue, output_left.len(), shared);
    if let Some(nam) = nam.as_mut().filter(|nam| nam.enabled) {
        let count = output_left.len().min(nam.mono.len());
        for sample in &mut nam.mono[..count] {
            *sample = queue
                .pop()
                .map_or(0.0, |frame| (frame[0] + frame[1]) * 0.5 * nam.input_gain);
        }
        // A failed amp must not mean silence. Whatever went wrong — a block
        // bigger than the model was prepared for, a model that throws — the
        // player still has to hear themselves, or the only symptom of a
        // recoverable fault is a guitar that appears to be unplugged.
        let gain = if nam.processor.process(&mut nam.mono[..count]).is_ok() {
            nam.output_gain
        } else {
            shared.monitor_amp_faults.fetch_add(1, Ordering::Relaxed);
            // Undo the input trim so the dry fallback sits at unity rather
            // than at whatever was dialled in for the amp's input stage.
            if nam.input_gain > 1e-6 {
                1.0 / nam.input_gain
            } else {
                1.0
            }
        };
        for ((left, right), sample) in output_left
            .iter_mut()
            .zip(output_right)
            .zip(&nam.mono[..count])
        {
            let mut processed = [[*sample * gain; 2]];
            effects.process_stereo(&mut processed);
            *left += processed[0][0];
            *right += processed[0][1];
        }
    } else {
        for (left, right) in output_left.iter_mut().zip(output_right) {
            if let Some(frame) = queue.pop() {
                let mut processed = [frame];
                effects.process_stereo(&mut processed);
                *left += processed[0][0];
                *right += processed[0][1];
            }
        }
    }
}

/// Drops input that has queued up ahead of the monitor mix.
///
/// Every frame sitting in the queue is delay between the string being struck
/// and the player hearing it. The input and output callbacks are separate and
/// are not phase-locked, so the depth that accumulates at the start of a stream
/// — or after the first xrun — is arbitrary, and nothing else would ever remove
/// it: the queue is drained at exactly the rate it is filled, so a backlog is
/// carried for the rest of the session. That is the difference between playing
/// through an amp and playing through a delay pedal.
///
/// Trimming costs a fraction of a millisecond of input, once, when the backlog
/// appears. Keeping it would cost that much on every note.
fn trim_monitor_queue(queue: &ArrayQueue<[f32; 2]>, frames: usize, shared: &Shared) {
    // One block being consumed now, plus one in hand for the next callback, so
    // ordinary jitter between the two streams does not cause a dropout.
    let target = frames.saturating_mul(2).max(MONITOR_MIN_QUEUE_FRAMES);
    let mut dropped = 0_u32;
    while queue.len() > target {
        if queue.pop().is_none() {
            break;
        }
        dropped += 1;
    }
    if dropped > 0 {
        shared
            .trimmed_monitor_frames
            .fetch_add(dropped, Ordering::Relaxed);
    }
}

fn calculate_count_in_frames(
    sample_rate: SampleRate,
    tempo: u64,
    numerator: u16,
    denominator: u16,
    bars: u16,
) -> u64 {
    u64::from(sample_rate.get())
        .saturating_mul(60)
        .saturating_mul(4)
        .saturating_mul(u64::from(numerator))
        .saturating_mul(u64::from(bars))
        / tempo.max(1).saturating_mul(u64::from(denominator).max(1))
}

fn activate_recording_if_due(shared: &Shared, position: u64) {
    if shared.record_pending.load(Ordering::Acquire)
        && position >= shared.record_start_frame.load(Ordering::Acquire)
    {
        shared.record_pending.store(false, Ordering::Release);
        shared.recording.store(true, Ordering::Release);
    }
}

#[cfg(test)]
fn mix_playback(
    clips: &[Option<PlaybackClip>; PLAYBACK_SLOT_CAPACITY],
    block_start: u64,
    output_left: &mut [f32],
    output_right: &mut [f32],
    track_peaks: &mut [f32; MIXER_TRACK_CAPACITY],
) {
    for clip in clips.iter().flatten() {
        if !clip.audible {
            continue;
        }
        let block_end = block_start.saturating_add(output_left.len() as u64);
        let clip_end = clip.start_frame.saturating_add(clip.samples.len() as u64);
        if clip.start_frame >= block_end || clip_end <= block_start {
            continue;
        }
        let overlap_start = block_start.max(clip.start_frame);
        let overlap_end = block_end.min(clip_end);
        for timeline_frame in overlap_start..overlap_end {
            let output_index = usize::try_from(timeline_frame - block_start).unwrap_or(0);
            let clip_index = usize::try_from(timeline_frame - clip.start_frame).unwrap_or(0);
            let frame = clip.samples[clip_index];
            let left_pan_gain = if clip.pan > 0.0 { 1.0 - clip.pan } else { 1.0 };
            let right_pan_gain = if clip.pan < 0.0 { 1.0 + clip.pan } else { 1.0 };
            let left = frame[0] * clip.gain * left_pan_gain;
            let right = frame[1] * clip.gain * right_pan_gain;
            output_left[output_index] += left;
            output_right[output_index] += right;
            if let Some(peak) = track_peaks.get_mut(clip.track_id) {
                *peak = peak.max(left.abs()).max(right.abs());
            }
        }
    }
}

/// Everything the audio thread needs to turn clips and notes into a mix.
///
/// Bundled into one owner so audio clips and instrument tracks share the same
/// per-track bus: an instrument track gets the same inserts, gain, pan and
/// meter as a recorded one, because as far as the mixer is concerned the only
/// difference is where the samples came from.
struct TrackMixer {
    /// On the heap, not inline. The mixer is captured by the audio callback,
    /// and a strip now carries a delay line and a reverb — over a kilobyte
    /// each, so sixty-four of them inline put seventy kilobytes into the
    /// closure and overflowed the callback thread's stack before the window
    /// ever opened. Everything else here is already a `Vec` for the same
    /// reason.
    effects: Vec<ChannelStrip>,
    nam: Vec<Option<ActiveNam>>,
    scratch: Vec<Vec<[f32; 2]>>,
    synths: Vec<Synth>,
    /// One sound font player per track, when the session has a font. Present or
    /// absent for the whole engine: the two renderers are interchangeable from
    /// here, so the mixer only has to know which it is holding.
    sampled: Option<Vec<SampledSynth>>,
    synth_left: Vec<f32>,
    synth_right: Vec<f32>,
    /// A shared room for the instrument tracks. Recorded audio was played in a
    /// room already; synthesised notes never were, and dry is the one cue no
    /// amount of work on the voices removes.
    reverb: Reverb,
    reverb_send: Vec<[f32; 2]>,
    /// Frames of tail still worth computing. The reverb is skipped entirely
    /// once it has run out, so a session of nothing but audio clips does not
    /// pay for an empty room every block.
    reverb_tail: usize,
    reverb_tail_frames: usize,
}

impl TrackMixer {
    /// Builds the mixer, and with it a renderer per track.
    ///
    /// The synthesised bank is always built: it costs a few megabytes and a few
    /// milliseconds, and having it there means a sound font that fails to load
    /// later, or a track the font has no preset for, still makes a sound.
    fn new(sample_rate: SampleRate, soundfont: Option<&SoundFontBank>) -> Result<Self> {
        // One bank for every track: the wavetables are identical, and building
        // 128 sets of them per track would cost megabytes each for nothing.
        let bank = Arc::new(GmBank::new(sample_rate));
        let sampled = soundfont
            .map(|font| {
                (0..MIXER_TRACK_CAPACITY)
                    .map(|_| font.player(sample_rate))
                    .collect::<Result<Vec<_>, _>>()
                    .with_context(|| format!("failed to prepare {}", font.name()))
            })
            .transpose()?;
        Ok(Self {
            effects: (0..MIXER_TRACK_CAPACITY)
                .map(|_| ChannelStrip::new(sample_rate, ChannelStripParams::default()))
                .collect(),
            nam: (0..MIXER_TRACK_CAPACITY).map(|_| None).collect(),
            scratch: (0..MIXER_TRACK_CAPACITY)
                .map(|_| vec![[0.0_f32; 2]; CLICK_SCRATCH_FRAMES])
                .collect(),
            synths: (0..MIXER_TRACK_CAPACITY)
                .map(|_| Synth::new(sample_rate, Arc::clone(&bank)))
                .collect(),
            sampled,
            synth_left: vec![0.0; CLICK_SCRATCH_FRAMES],
            synth_right: vec![0.0; CLICK_SCRATCH_FRAMES],
            reverb: Reverb::new(sample_rate),
            reverb_send: vec![[0.0_f32; 2]; CLICK_SCRATCH_FRAMES],
            reverb_tail: 0,
            reverb_tail_frames: sample_rate.get() as usize * REVERB_TAIL_SECONDS,
        })
    }

    fn set_track_effects(&mut self, track_id: usize, params: ChannelStripParams) {
        if let Some(strip) = self.effects.get_mut(track_id) {
            strip.set_params(params);
        }
        if let Some(Some(nam)) = self.nam.get_mut(track_id) {
            nam.set_params(params);
        }
    }

    fn set_track_nam(
        &mut self,
        track_id: usize,
        model: Option<ActiveNam>,
        retired: &ArrayQueue<ActiveNam>,
    ) {
        let Some(slot) = self.nam.get_mut(track_id) else {
            return;
        };
        if let Some(old) = slot.take() {
            let _ = retired.push(old);
        }
        *slot = model;
    }

    /// Silences every track, so stopping never leaves a note hanging.
    ///
    /// The tails go with them — the shared reverb, and each strip's own delay
    /// and reverb. Letting the old position's echoes ring on over the new one
    /// is heard as a smear across the edit. Monitoring is deliberately not
    /// touched: the player is still playing.
    fn silence_tracks(&mut self) {
        for synth in &mut self.synths {
            synth.reset();
        }
        if let Some(players) = self.sampled.as_mut() {
            for player in players {
                player.reset();
            }
        }
        for strip in &mut self.effects {
            strip.clear_tails();
        }
        self.reverb.clear();
        self.reverb_tail = 0;
    }

    /// Silences one track's instrument, whichever renderer is playing it.
    fn silence_track(&mut self, track_id: usize) {
        if let Some(player) = self.sampled.as_mut().and_then(|p| p.get_mut(track_id)) {
            player.reset();
        }
        if let Some(synth) = self.synths.get_mut(track_id) {
            synth.reset();
        }
    }

    /// Renders one instrument track into the synth scratch buffers and returns
    /// how much of it belongs in the reverb.
    fn render_part(&mut self, part: &MidiPart, block_start: u64, frame_count: usize) -> f32 {
        let left = &mut self.synth_left[..frame_count];
        let right = &mut self.synth_right[..frame_count];
        left.fill(0.0);
        right.fill(0.0);
        if let Some(player) = self
            .sampled
            .as_mut()
            .and_then(|players| players.get_mut(part.track_id))
        {
            player.set_drum_kit(part.is_drum_kit);
            player.set_program(part.program);
            player.render(&part.notes, block_start, left, right);
            return player.reverb_send();
        }
        let Some(synth) = self.synths.get_mut(part.track_id) else {
            return 0.0;
        };
        synth.set_drum_kit(part.is_drum_kit);
        synth.set_program(part.program);
        synth.render(&part.notes, block_start, left, right);
        synth.reverb_send()
    }

    fn render(
        &mut self,
        clips: &[Option<PlaybackClip>; PLAYBACK_SLOT_CAPACITY],
        midi: &[Option<MidiPart>; MIDI_SLOT_CAPACITY],
        block_start: u64,
        output_left: &mut [f32],
        output_right: &mut [f32],
        track_peaks: &mut [f32; MIXER_TRACK_CAPACITY],
    ) {
        let frame_count = output_left.len();
        let mut track_used = [false; MIXER_TRACK_CAPACITY];
        for scratch in &mut self.scratch {
            scratch[..frame_count].fill([0.0, 0.0]);
        }
        self.reverb_send[..frame_count].fill([0.0, 0.0]);

        self.render_notes(midi, block_start, frame_count, &mut track_used);
        mix_clips(clips, block_start, frame_count, &mut self.scratch, &mut track_used);

        for track_id in 0..MIXER_TRACK_CAPACITY {
            if !track_used[track_id] {
                track_peaks[track_id] = 0.0;
                continue;
            }
            let scratch = &mut self.scratch[track_id];
            if let Some(nam) = &mut self.nam[track_id] {
                nam.process_stereo(&mut scratch[..frame_count]);
            }
            let strip = &mut self.effects[track_id];
            strip.process_stereo(&mut scratch[..frame_count]);
            let mut peak = 0.0_f32;
            for ((output_l, output_r), frame) in output_left
                .iter_mut()
                .zip(output_right.iter_mut())
                .zip(&scratch[..frame_count])
            {
                *output_l += frame[0];
                *output_r += frame[1];
                peak = peak.max(frame[0].abs()).max(frame[1].abs());
            }
            track_peaks[track_id] = peak;
        }

        // The reverb is a bus, not an insert: it takes what the instrument
        // tracks send it and goes straight to the master, past their strips.
        if self.reverb_send[..frame_count]
            .iter()
            .any(|frame| frame[0] != 0.0 || frame[1] != 0.0)
        {
            self.reverb_tail = self.reverb_tail_frames;
        }
        if self.reverb_tail > 0 {
            self.reverb
                .process(&self.reverb_send[..frame_count], output_left, output_right);
            self.reverb_tail = self.reverb_tail.saturating_sub(frame_count);
        }
    }

    fn render_notes(
        &mut self,
        midi: &[Option<MidiPart>; MIDI_SLOT_CAPACITY],
        block_start: u64,
        frame_count: usize,
        track_used: &mut [bool; MIXER_TRACK_CAPACITY],
    ) {
        for part in midi.iter().flatten() {
            if part.track_id >= MIXER_TRACK_CAPACITY {
                continue;
            }
            if !part.audible {
                // A muted instrument must go quiet immediately, not finish the
                // notes already sounding.
                self.silence_track(part.track_id);
                continue;
            }
            // How far into the room this instrument sits: a hall for strings,
            // a small room for a piano, next to nothing for a bass.
            let send = self.render_part(part, block_start, frame_count);
            let Some(scratch) = self.scratch.get_mut(part.track_id) else {
                continue;
            };

            let left_pan_gain = if part.pan > 0.0 { 1.0 - part.pan } else { 1.0 };
            let right_pan_gain = if part.pan < 0.0 { 1.0 + part.pan } else { 1.0 };
            let mut sounded = false;
            for (index, frame) in scratch[..frame_count].iter_mut().enumerate() {
                let (left, right) = (self.synth_left[index], self.synth_right[index]);
                if left != 0.0 || right != 0.0 {
                    sounded = true;
                }
                let (left, right) = (
                    left * part.gain * left_pan_gain,
                    right * part.gain * right_pan_gain,
                );
                frame[0] += left;
                frame[1] += right;
                self.reverb_send[index][0] += left * send;
                self.reverb_send[index][1] += right * send;
            }
            if sounded {
                track_used[part.track_id] = true;
            }
        }
    }
}

fn mix_clips(
    clips: &[Option<PlaybackClip>; PLAYBACK_SLOT_CAPACITY],
    block_start: u64,
    frame_count: usize,
    track_scratch: &mut [Vec<[f32; 2]>],
    track_used: &mut [bool; MIXER_TRACK_CAPACITY],
) {
    let block_end = block_start.saturating_add(frame_count as u64);
    for clip in clips.iter().flatten().filter(|clip| clip.audible) {
        let Some(scratch) = track_scratch.get_mut(clip.track_id) else {
            continue;
        };
        let clip_end = clip.start_frame.saturating_add(clip.samples.len() as u64);
        if clip.start_frame >= block_end || clip_end <= block_start {
            continue;
        }
        track_used[clip.track_id] = true;
        let overlap_start = block_start.max(clip.start_frame);
        let overlap_end = block_end.min(clip_end);
        let left_pan_gain = if clip.pan > 0.0 { 1.0 - clip.pan } else { 1.0 };
        let right_pan_gain = if clip.pan < 0.0 { 1.0 + clip.pan } else { 1.0 };
        for timeline_frame in overlap_start..overlap_end {
            let output_index = usize::try_from(timeline_frame - block_start).unwrap_or(0);
            let clip_index = usize::try_from(timeline_frame - clip.start_frame).unwrap_or(0);
            let frame = clip.samples[clip_index];
            scratch[output_index][0] += frame[0] * clip.gain * left_pan_gain;
            scratch[output_index][1] += frame[1] * clip.gain * right_pan_gain;
        }
    }
}

fn set_track_audible_in_slots(
    clips: &mut [Option<PlaybackClip>; PLAYBACK_SLOT_CAPACITY],
    track_id: usize,
    audible: bool,
) {
    for clip in clips.iter_mut().flatten() {
        if clip.track_id == track_id {
            clip.audible = audible;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn move_clip_in_slots(
    clips: &mut [Option<PlaybackClip>; PLAYBACK_SLOT_CAPACITY],
    clip_id: u128,
    start_frame: u64,
    track_id: usize,
    gain: f32,
    pan: f32,
    audible: bool,
) {
    if let Some(clip) = clips
        .iter_mut()
        .flatten()
        .find(|clip| clip.clip_id == clip_id)
    {
        clip.start_frame = start_frame;
        clip.track_id = track_id;
        clip.gain = gain;
        clip.pan = pan;
        clip.audible = audible;
    }
}

fn load_wav(path: &std::path::Path, expected_rate: SampleRate) -> Result<Vec<[f32; 2]>> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open playback file {}", path.display()))?;
    let spec = reader.spec();
    if spec.sample_rate != expected_rate.get() {
        bail!(
            "{} uses {} Hz; this session requires {} Hz",
            path.display(),
            spec.sample_rate,
            expected_rate.get()
        );
    }
    if !(1..=2).contains(&spec.channels) {
        bail!("{} has an unsupported channel count", path.display());
    }

    let values = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 2.0_f32.powi(i32::from(spec.bits_per_sample).saturating_sub(1));
            reader
                .samples::<i32>()
                .map(|sample| {
                    sample.map(|value| {
                        #[allow(clippy::cast_precision_loss)]
                        {
                            value as f32 / scale
                        }
                    })
                })
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
    };

    let channels = usize::from(spec.channels);
    Ok(values
        .chunks_exact(channels)
        .map(|frame| {
            if channels == 1 {
                [frame[0], frame[0]]
            } else {
                [frame[0], frame[1]]
            }
        })
        .collect())
}

#[allow(clippy::needless_pass_by_value)]
fn writer_loop(
    receiver: Receiver<WriterCommand>,
    queue: Arc<ArrayQueue<[f32; 2]>>,
    sample_rate: SampleRate,
    shared: Arc<Shared>,
) {
    let mut writer: Option<hound::WavWriter<std::io::BufWriter<std::fs::File>>> = None;
    let mut channels = 1_u16;
    let mut frames_since_flush = 0_usize;

    loop {
        match receiver.recv_timeout(Duration::from_millis(5)) {
            Ok(WriterCommand::Start {
                path,
                channels: requested,
            }) => {
                drain_queue(&queue);
                channels = requested;
                frames_since_flush = 0;
                let spec = hound::WavSpec {
                    channels,
                    sample_rate: sample_rate.get(),
                    bits_per_sample: 24,
                    sample_format: hound::SampleFormat::Int,
                };
                writer = if let Ok(created) = hound::WavWriter::create(path, spec) {
                    Some(created)
                } else {
                    shared.disk_error.store(true, Ordering::Release);
                    None
                };
                shared
                    .writer_ready
                    .store(writer.is_some(), Ordering::Release);
            }
            Ok(WriterCommand::Stop) => {
                let _ = write_available(&queue, writer.as_mut(), channels, &shared);
                if let Some(active) = writer.take() {
                    if active.finalize().is_err() {
                        shared.disk_error.store(true, Ordering::Release);
                    }
                }
                shared.writer_ready.store(false, Ordering::Release);
                frames_since_flush = 0;
            }
            Ok(WriterCommand::Shutdown) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                shared.writer_ready.store(false, Ordering::Release);
                let _ = write_available(&queue, writer.as_mut(), channels, &shared);
                if let Some(active) = writer.take() {
                    if active.finalize().is_err() {
                        shared.disk_error.store(true, Ordering::Release);
                    }
                }
                break;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
        frames_since_flush = frames_since_flush.saturating_add(write_available(
            &queue,
            writer.as_mut(),
            channels,
            &shared,
        ));
        if frames_since_flush >= usize::try_from(sample_rate.get()).unwrap_or(48_000) {
            if let Some(active) = writer.as_mut() {
                if active.flush().is_err() {
                    shared.disk_error.store(true, Ordering::Release);
                }
            }
            frames_since_flush = 0;
        }
    }
}

fn write_available(
    queue: &ArrayQueue<[f32; 2]>,
    mut writer: Option<&mut hound::WavWriter<std::io::BufWriter<std::fs::File>>>,
    channels: u16,
    shared: &Shared,
) -> usize {
    let mut frames_written = 0;
    while let Some(frame) = queue.pop() {
        if let Some(active) = writer.as_deref_mut() {
            if active.write_sample(float_to_i24(frame[0])).is_err() {
                shared.disk_error.store(true, Ordering::Release);
            }
            if channels == 2 && active.write_sample(float_to_i24(frame[1])).is_err() {
                shared.disk_error.store(true, Ordering::Release);
            }
            frames_written += 1;
        }
    }
    frames_written
}

fn drain_queue(queue: &ArrayQueue<[f32; 2]>) {
    while queue.pop().is_some() {}
}

fn float_to_i24(sample: f32) -> i32 {
    #[allow(clippy::cast_possible_truncation)]
    {
        (sample.clamp(-1.0, 1.0) * 8_388_607.0).round() as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn playback_is_mixed_at_its_absolute_position_in_stereo() {
        let clip = PlaybackClip {
            clip_id: 1,
            track_id: 0,
            audible: true,
            start_frame: 102,
            samples: Arc::new(vec![[0.25, -0.5], [0.5, -0.25]]),
            gain: 2.0,
            pan: 0.0,
        };
        let mut slots: [Option<PlaybackClip>; PLAYBACK_SLOT_CAPACITY] =
            std::array::from_fn(|_| None);
        slots[0] = Some(clip);
        let mut left = [0.0; 8];
        let mut right = [0.0; 8];

        let mut peaks = [0.0; MIXER_TRACK_CAPACITY];
        mix_playback(&slots, 100, &mut left, &mut right, &mut peaks);

        assert!((left[2] - 0.5).abs() < f32::EPSILON);
        assert!((right[2] + 1.0).abs() < f32::EPSILON);
        assert!((left[3] - 1.0).abs() < f32::EPSILON);
        assert!((right[3] + 0.5).abs() < f32::EPSILON);
        assert!(left[..2].iter().all(|sample| sample.abs() < f32::EPSILON));
        assert!(left[4..].iter().all(|sample| sample.abs() < f32::EPSILON));
        assert!((peaks[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn track_pan_and_meter_are_applied_independently() {
        let mut slots: [Option<PlaybackClip>; PLAYBACK_SLOT_CAPACITY] =
            std::array::from_fn(|_| None);
        slots[0] = Some(PlaybackClip {
            clip_id: 1,
            track_id: 4,
            audible: true,
            start_frame: 0,
            samples: Arc::new(vec![[0.5, 0.5]]),
            gain: 1.0,
            pan: -1.0,
        });
        let mut left = [0.0];
        let mut right = [0.0];
        let mut peaks = [0.0; MIXER_TRACK_CAPACITY];
        mix_playback(&slots, 0, &mut left, &mut right, &mut peaks);
        assert!((left[0] - 0.5).abs() < f32::EPSILON);
        assert!(right[0].abs() < f32::EPSILON);
        assert!((peaks[4] - 0.5).abs() < f32::EPSILON);
        assert!(peaks[3].abs() < f32::EPSILON);
    }

    #[test]
    fn playback_overlap_adds_clips() {
        let mut slots: [Option<PlaybackClip>; PLAYBACK_SLOT_CAPACITY] =
            std::array::from_fn(|_| None);
        slots[0] = Some(PlaybackClip {
            clip_id: 1,
            track_id: 0,
            audible: true,
            start_frame: 0,
            samples: Arc::new(vec![[0.2, 0.3]]),
            gain: 1.0,
            pan: 0.0,
        });
        slots[1] = Some(PlaybackClip {
            clip_id: 2,
            track_id: 1,
            audible: true,
            start_frame: 0,
            samples: Arc::new(vec![[0.4, 0.1]]),
            gain: 1.0,
            pan: 0.0,
        });
        let mut left = [0.0];
        let mut right = [0.0];

        let mut peaks = [0.0; MIXER_TRACK_CAPACITY];
        mix_playback(&slots, 0, &mut left, &mut right, &mut peaks);

        assert!((left[0] - 0.6).abs() < f32::EPSILON);
        assert!((right[0] - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn inaudible_track_does_not_enter_the_mix() {
        let mut slots: [Option<PlaybackClip>; PLAYBACK_SLOT_CAPACITY] =
            std::array::from_fn(|_| None);
        slots[0] = Some(PlaybackClip {
            clip_id: 1,
            track_id: 3,
            audible: false,
            start_frame: 0,
            samples: Arc::new(vec![[0.8, -0.8]]),
            gain: 1.0,
            pan: 0.0,
        });
        let mut left = [0.0];
        let mut right = [0.0];
        let mut peaks = [0.0; MIXER_TRACK_CAPACITY];
        mix_playback(&slots, 0, &mut left, &mut right, &mut peaks);
        assert!(left[0].abs() < f32::EPSILON);
        assert!(right[0].abs() < f32::EPSILON);
    }

    #[test]
    fn audibility_command_changes_only_the_target_track() {
        let mut slots: [Option<PlaybackClip>; PLAYBACK_SLOT_CAPACITY] =
            std::array::from_fn(|_| None);
        for (slot, track_id) in [0, 1].into_iter().enumerate() {
            slots[slot] = Some(PlaybackClip {
                clip_id: slot as u128,
                track_id,
                audible: true,
                start_frame: 0,
                samples: Arc::new(vec![[0.2, 0.2]]),
                gain: 1.0,
                pan: 0.0,
            });
        }
        set_track_audible_in_slots(&mut slots, 1, false);
        assert!(slots[0].as_ref().unwrap().audible);
        assert!(!slots[1].as_ref().unwrap().audible);
    }

    #[test]
    fn clip_move_updates_routing_without_reloading_samples() {
        let samples = Arc::new(vec![[0.2, 0.2]]);
        let mut slots: [Option<PlaybackClip>; PLAYBACK_SLOT_CAPACITY] =
            std::array::from_fn(|_| None);
        slots[0] = Some(PlaybackClip {
            clip_id: 42,
            track_id: 0,
            audible: true,
            start_frame: 0,
            samples: Arc::clone(&samples),
            gain: 1.0,
            pan: 0.0,
        });

        move_clip_in_slots(&mut slots, 42, 96_000, 3, 0.5, -0.25, false);

        let moved = slots[0].as_ref().unwrap();
        assert_eq!(moved.start_frame, 96_000);
        assert_eq!(moved.track_id, 3);
        assert!((moved.gain - 0.5).abs() < f32::EPSILON);
        assert!((moved.pan + 0.25).abs() < f32::EPSILON);
        assert!(!moved.audible);
        assert!(Arc::ptr_eq(&moved.samples, &samples));
    }

    #[test]
    fn realtime_track_effects_process_cached_source_without_mutating_it() {
        let source = Arc::new(vec![[1.0, 1.0]; 2_048]);
        let mut slots: [Option<PlaybackClip>; PLAYBACK_SLOT_CAPACITY] =
            std::array::from_fn(|_| None);
        slots[0] = Some(PlaybackClip {
            clip_id: 7,
            track_id: 2,
            audible: true,
            start_frame: 0,
            samples: Arc::clone(&source),
            gain: 1.0,
            pan: 0.0,
        });
        let params = ChannelStripParams {
            compressor_enabled: true,
            compressor_threshold_db: -20.0,
            compressor_ratio: 10.0,
            compressor_attack_ms: 1.0,
            ..ChannelStripParams::default()
        };
        let mut mixer = TrackMixer::new(SampleRate::DEFAULT, None).expect("the synthesised bank always builds");
        mixer.set_track_effects(2, params);
        let empty_midi: [Option<MidiPart>; MIDI_SLOT_CAPACITY] = std::array::from_fn(|_| None);
        let mut left = [0.0_f32; CLICK_SCRATCH_FRAMES];
        let mut right = [0.0_f32; CLICK_SCRATCH_FRAMES];
        let mut peaks = [0.0_f32; MIXER_TRACK_CAPACITY];

        mixer.render(&slots, &empty_midi, 0, &mut left, &mut right, &mut peaks);

        assert!(left[2_047] < 0.2);
        assert!((source[2_047][0] - 1.0).abs() < f32::EPSILON);
        assert!((source[2_047][1] - 1.0).abs() < f32::EPSILON);
        assert!(peaks[2] > 0.0);
        assert!(peaks[1].abs() < f32::EPSILON);
    }

    #[test]
    fn instrument_tracks_reach_the_mix_through_their_own_bus() {
        let mut mixer = TrackMixer::new(SampleRate::DEFAULT, None).expect("the synthesised bank always builds");
        let mut midi: [Option<MidiPart>; MIDI_SLOT_CAPACITY] = std::array::from_fn(|_| None);
        midi[0] = Some(MidiPart {
            track_id: 3,
            audible: true,
            notes: Arc::new(vec![ScheduledNote {
                start_frame: 0,
                end_frame: 24_000,
                pitch: 60,
                velocity: 100,
            }]),
            gain: 1.0,
            pan: 0.0,
            program: 16,
            is_drum_kit: false,
        });
        let slots: [Option<PlaybackClip>; PLAYBACK_SLOT_CAPACITY] = std::array::from_fn(|_| None);
        let mut left = [0.0_f32; CLICK_SCRATCH_FRAMES];
        let mut right = [0.0_f32; CLICK_SCRATCH_FRAMES];
        let mut peaks = [0.0_f32; MIXER_TRACK_CAPACITY];

        mixer.render(&slots, &midi, 0, &mut left, &mut right, &mut peaks);

        let energy: f32 = left.iter().map(|value| value.abs()).sum();
        assert!(energy > 1.0, "the synth did not reach the mix");
        assert!(peaks[3] > 0.0, "the instrument track did not meter");
        assert!(peaks[2].abs() < f32::EPSILON, "it bled into another track");
    }

    /// Writes the fixture font to a temporary file and loads it, so this can
    /// run on a machine with no sound font installed.
    fn fixture_soundfont() -> (SoundFontBank, PathBuf) {
        let path = std::env::temp_dir().join(format!(
            "rustdaw-mixer-{}-{:?}.sf2",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, daw_engine::soundfont::fixture::soundfont())
            .expect("write the fixture font");
        let bank = SoundFontBank::load(&path).expect("the fixture font should load");
        (bank, path)
    }

    fn one_instrument_part(program: u8) -> [Option<MidiPart>; MIDI_SLOT_CAPACITY] {
        let mut midi: [Option<MidiPart>; MIDI_SLOT_CAPACITY] = std::array::from_fn(|_| None);
        midi[0] = Some(MidiPart {
            track_id: 3,
            audible: true,
            notes: Arc::new(vec![ScheduledNote {
                start_frame: 0,
                end_frame: 24_000,
                pitch: 60,
                velocity: 100,
            }]),
            gain: 1.0,
            pan: 0.0,
            program,
            is_drum_kit: false,
        });
        midi
    }

    #[test]
    fn a_sound_font_takes_over_the_instrument_tracks() {
        // With a font loaded the tracks play from it instead of the synthesised
        // bank, still through their own per-track bus and meter.
        let (font, path) = fixture_soundfont();
        let mut mixer = TrackMixer::new(SampleRate::DEFAULT, Some(&font))
            .expect("the fixture font should build players");
        assert!(mixer.sampled.is_some(), "the font was not picked up");

        let midi = one_instrument_part(16);
        let slots: [Option<PlaybackClip>; PLAYBACK_SLOT_CAPACITY] = std::array::from_fn(|_| None);
        let mut left = [0.0_f32; CLICK_SCRATCH_FRAMES];
        let mut right = [0.0_f32; CLICK_SCRATCH_FRAMES];
        let mut peaks = [0.0_f32; MIXER_TRACK_CAPACITY];
        mixer.render(&slots, &midi, 0, &mut left, &mut right, &mut peaks);

        let energy: f32 = left.iter().map(|value| value.abs()).sum();
        assert!(energy > 0.0, "the sampled instrument did not reach the mix");
        assert!(peaks[3] > 0.0, "the instrument track did not meter");
        assert!(peaks[2].abs() < f32::EPSILON, "it bled into another track");

        // And muting still silences it at once, on the sampled path too. The
        // track's own meter is what to check: the mix still carries the reverb
        // tail of what it played a moment ago, as it should.
        let mut midi = midi;
        if let Some(part) = midi[0].as_mut() {
            part.audible = false;
        }
        let mut left = [0.0_f32; CLICK_SCRATCH_FRAMES];
        let mut right = [0.0_f32; CLICK_SCRATCH_FRAMES];
        mixer.render(&slots, &midi, 0, &mut left, &mut right, &mut peaks);
        assert!(
            peaks[3].abs() < f32::EPSILON,
            "a muted sampled track kept sounding"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn without_a_sound_font_the_synthesised_bank_plays() {
        let mixer = TrackMixer::new(SampleRate::DEFAULT, None)
            .expect("the synthesised bank always builds");
        assert!(mixer.sampled.is_none());
    }

    #[test]
    fn instrument_tracks_ring_on_in_the_shared_reverb() {
        // The reverb is a bus, so its tail has to survive after the notes have
        // stopped and after the tracks that fed it have gone quiet.
        let mut mixer = TrackMixer::new(SampleRate::DEFAULT, None)
            .expect("the synthesised bank always builds");
        let mut midi: [Option<MidiPart>; MIDI_SLOT_CAPACITY] = std::array::from_fn(|_| None);
        midi[0] = Some(MidiPart {
            track_id: 3,
            audible: true,
            notes: Arc::new(vec![ScheduledNote {
                start_frame: 0,
                end_frame: 2_000,
                pitch: 60,
                velocity: 110,
            }]),
            gain: 1.0,
            pan: 0.0,
            // Strings, which are sent furthest into the room.
            program: 48,
            is_drum_kit: false,
        });
        let slots: [Option<PlaybackClip>; PLAYBACK_SLOT_CAPACITY] = std::array::from_fn(|_| None);
        let mut peaks = [0.0_f32; MIXER_TRACK_CAPACITY];
        let mut frame = 0_u64;
        // A second of playback: past the note, past its release, and well
        // inside the reverb's own tail.
        for _ in 0..24 {
            let mut left = [0.0_f32; CLICK_SCRATCH_FRAMES];
            let mut right = [0.0_f32; CLICK_SCRATCH_FRAMES];
            mixer.render(&slots, &midi, frame, &mut left, &mut right, &mut peaks);
            frame += CLICK_SCRATCH_FRAMES as u64;
        }

        // Well past the note and its release, and still ringing.
        let mut left = [0.0_f32; CLICK_SCRATCH_FRAMES];
        let mut right = [0.0_f32; CLICK_SCRATCH_FRAMES];
        mixer.render(&slots, &midi, frame, &mut left, &mut right, &mut peaks);
        let tail: f32 = left.iter().map(|value| value.abs()).sum();
        assert!(
            peaks[3].abs() < f32::EPSILON,
            "the note itself should have stopped by now"
        );
        assert!(tail > 0.0, "the reverb tail did not reach the mix");

        // A stop drops the tail rather than smearing it over the new position.
        mixer.silence_tracks();
        let mut left = [0.0_f32; CLICK_SCRATCH_FRAMES];
        let mut right = [0.0_f32; CLICK_SCRATCH_FRAMES];
        let empty: [Option<MidiPart>; MIDI_SLOT_CAPACITY] = std::array::from_fn(|_| None);
        mixer.render(&slots, &empty, frame, &mut left, &mut right, &mut peaks);
        let after: f32 = left.iter().map(|value| value.abs()).sum();
        assert!(after < f32::EPSILON, "the tail survived a stop");
    }

    #[test]
    fn muting_an_instrument_track_silences_it_immediately() {
        let mut mixer = TrackMixer::new(SampleRate::DEFAULT, None).expect("the synthesised bank always builds");
        let mut midi: [Option<MidiPart>; MIDI_SLOT_CAPACITY] = std::array::from_fn(|_| None);
        midi[0] = Some(MidiPart {
            track_id: 1,
            audible: false,
            notes: Arc::new(vec![ScheduledNote {
                start_frame: 0,
                end_frame: 480_000,
                pitch: 64,
                velocity: 127,
            }]),
            gain: 1.0,
            pan: 0.0,
            program: 16,
            is_drum_kit: false,
        });
        let slots: [Option<PlaybackClip>; PLAYBACK_SLOT_CAPACITY] = std::array::from_fn(|_| None);
        let mut left = [0.0_f32; CLICK_SCRATCH_FRAMES];
        let mut right = [0.0_f32; CLICK_SCRATCH_FRAMES];
        let mut peaks = [0.0_f32; MIXER_TRACK_CAPACITY];

        mixer.render(&slots, &midi, 0, &mut left, &mut right, &mut peaks);
        assert!(left.iter().all(|value| value.abs() < f32::EPSILON));
        assert!(peaks[1].abs() < f32::EPSILON);
    }

    #[test]
    fn software_monitoring_routes_stereo_frames() {
        let shared = Shared::new();
        shared.monitoring.store(true, Ordering::Release);
        let queue = ArrayQueue::new(4);
        queue.push([0.25, -0.5]).unwrap();
        queue.push([0.5, -0.25]).unwrap();
        let mut left = [0.1, 0.1];
        let mut right = [0.1, 0.1];

        let mut effects = ChannelStrip::new(SampleRate::DEFAULT, ChannelStripParams::default());
        mix_monitoring(
            &shared,
            &queue,
            &mut left,
            &mut right,
            &mut effects,
            &mut None,
        );

        assert!((left[0] - 0.35).abs() < f32::EPSILON);
        assert!((right[0] + 0.4).abs() < f32::EPSILON);
        assert!((left[1] - 0.6).abs() < f32::EPSILON);
        assert!((right[1] + 0.15).abs() < f32::EPSILON);
    }

    #[test]
    fn monitoring_plays_the_newest_input_rather_than_a_backlog() {
        // The two streams are not phase-locked, so a backlog builds up at the
        // start. Nothing drains it faster than it fills, so without trimming
        // the player hears every note late for the rest of the session.
        let shared = Shared::new();
        shared.monitoring.store(true, Ordering::Release);
        let queue = ArrayQueue::new(MONITOR_QUEUE_FRAMES);
        // Nearly a second of backlog, then the cushion the trim is allowed to
        // keep — one block being consumed now and one in hand for the next.
        for _ in 0..4_000 {
            queue.push([-1.0, -1.0]).expect("push stale input");
        }
        for _ in 0..512 {
            queue.push([1.0, 1.0]).expect("push live input");
        }
        let deep = queue.len();

        let mut left = [0.0_f32; 256];
        let mut right = [0.0_f32; 256];
        let mut effects = ChannelStrip::new(SampleRate::DEFAULT, ChannelStripParams::default());
        mix_monitoring(&shared, &queue, &mut left, &mut right, &mut effects, &mut None);

        assert!(queue.len() < deep, "the backlog was not trimmed");
        assert!(
            queue.len() <= 512,
            "still {} frames of delay queued",
            queue.len()
        );
        // What came out is recent input, not the second-old backlog.
        assert!(
            left.iter().all(|value| *value > 0.0),
            "the monitor played the backlog instead of the live input"
        );
        assert!(
            shared.trimmed_monitor_frames.load(Ordering::Relaxed) > 0,
            "trimming should be reported"
        );
    }

    #[test]
    fn a_short_queue_is_left_alone() {
        // Trimming a queue that is already short would cause dropouts for no
        // latency gain.
        let shared = Shared::new();
        shared.monitoring.store(true, Ordering::Release);
        let queue = ArrayQueue::new(MONITOR_QUEUE_FRAMES);
        for _ in 0..64 {
            queue.push([0.5, 0.5]).expect("push");
        }
        let mut left = [0.0_f32; 32];
        let mut right = [0.0_f32; 32];
        let mut effects = ChannelStrip::new(SampleRate::DEFAULT, ChannelStripParams::default());
        mix_monitoring(&shared, &queue, &mut left, &mut right, &mut effects, &mut None);

        assert_eq!(
            shared.trimmed_monitor_frames.load(Ordering::Relaxed),
            0,
            "a short queue must not be trimmed"
        );
        assert_eq!(queue.len(), 32, "only the consumed frames should be gone");
    }

    #[test]
    fn monitoring_through_an_amp_is_not_the_dry_signal() {
        // The whole point of monitoring through NAM: what comes back must be
        // the amp, not the guitar that went in.
        let (bank, path) = fixture_soundfont();
        drop(bank);
        let _ = std::fs::remove_file(path);

        let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../third_party/NeuralAmpModelerCore/example_models/lstm.nam");
        let Ok(processor) =
            daw_nam::NamProcessor::load(&model, SampleRate::DEFAULT.get(), CLICK_SCRATCH_FRAMES)
        else {
            return; // The submodule is not checked out; nothing to assert.
        };

        let shared = Shared::new();
        shared.monitoring.store(true, Ordering::Release);
        let queue = ArrayQueue::new(MONITOR_QUEUE_FRAMES);
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        for sample in &input {
            queue.push([*sample, *sample]).expect("push");
        }

        let mut nam = Some(ActiveNam::new(
            processor,
            ChannelStripParams {
                nam_enabled: true,
                ..ChannelStripParams::default()
            },
            SampleRate::DEFAULT,
        ));
        let mut left = [0.0_f32; 256];
        let mut right = [0.0_f32; 256];
        let mut effects = ChannelStrip::new(SampleRate::DEFAULT, ChannelStripParams::default());
        mix_monitoring(&shared, &queue, &mut left, &mut right, &mut effects, &mut nam);

        assert!(left.iter().all(|value| value.is_finite()));
        let difference: f32 = left
            .iter()
            .zip(&input)
            .map(|(heard, played)| (heard - played).abs())
            .sum();
        assert!(
            difference > 0.1,
            "the monitor passed the dry guitar through unchanged"
        );
    }

    #[test]
    fn a_failing_amp_falls_back_to_the_dry_signal_rather_than_silence() {
        // A player who cannot hear themselves has no way to tell a broken amp
        // from a broken cable. Whatever the fault, the guitar must come back.
        let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../third_party/NeuralAmpModelerCore/example_models/lstm.nam");
        // Deliberately prepared for a block far smaller than the one it is
        // given, which is the failure this fallback exists for.
        let Ok(processor) = daw_nam::NamProcessor::load(&model, SampleRate::DEFAULT.get(), 32)
        else {
            return; // The submodule is not checked out; nothing to assert.
        };

        let shared = Shared::new();
        shared.monitoring.store(true, Ordering::Release);
        let queue = ArrayQueue::new(MONITOR_QUEUE_FRAMES);
        for _ in 0..256 {
            queue.push([0.4, 0.4]).expect("push");
        }
        let mut nam = Some(ActiveNam::new(
            processor,
            ChannelStripParams {
                nam_enabled: true,
                nam_input_db: 6.0206, // ×2
                ..ChannelStripParams::default()
            },
            SampleRate::DEFAULT,
        ));
        let mut left = [0.0_f32; 256];
        let mut right = [0.0_f32; 256];
        let mut effects = ChannelStrip::new(SampleRate::DEFAULT, ChannelStripParams::default());
        mix_monitoring(&shared, &queue, &mut left, &mut right, &mut effects, &mut nam);

        assert!(
            shared.monitor_amp_faults.load(Ordering::Relaxed) > 0,
            "the fault should be recorded so it can be reported"
        );
        // The guitar is audible, and at the level it went in rather than
        // multiplied by the amp's input trim.
        assert!(
            left.iter().all(|value| (value - 0.4).abs() < 1e-3),
            "expected the dry signal at unity, got {:?}",
            &left[..4]
        );
    }

    /// Renders one block through the whole source path at a given transport
    /// state, with one audio clip sitting under the playhead and one frame of
    /// live input waiting to be monitored.
    fn render_with_transport(playing: bool) -> (Vec<f32>, ArrayQueue<[f32; 2]>) {
        let shared = Shared::new();
        shared.monitoring.store(true, Ordering::Release);
        let mut mixer = TrackMixer::new(SampleRate::DEFAULT, None)
            .expect("the synthesised bank always builds");
        let mut playback_slots: [Option<PlaybackClip>; PLAYBACK_SLOT_CAPACITY] =
            std::array::from_fn(|_| None);
        playback_slots[0] = Some(PlaybackClip {
            clip_id: 1,
            track_id: 0,
            start_frame: 0,
            samples: Arc::new(vec![[0.5_f32; 2]; 4_096]),
            gain: 1.0,
            pan: 0.0,
            audible: true,
        });
        let midi_slots: [Option<MidiPart>; MIDI_SLOT_CAPACITY] = std::array::from_fn(|_| None);

        let queue = ArrayQueue::new(MONITOR_QUEUE_FRAMES);
        for _ in 0..256 {
            queue.push([0.2, 0.2]).expect("push live input");
        }
        let mut left = vec![0.0_f32; 128];
        let mut right = vec![0.0_f32; 128];
        let mut effects = ChannelStrip::new(SampleRate::DEFAULT, ChannelStripParams::default());
        render_source_block(
            0,
            128,
            playing,
            false,
            0.0,
            120,
            4,
            4,
            SampleRate::DEFAULT,
            &mut left,
            &mut right,
            &mut mixer,
            &playback_slots,
            &midi_slots,
            &shared,
            &queue,
            &mut effects,
            &mut None,
        );
        (left, queue)
    }

    #[test]
    fn the_mixer_stays_small_enough_for_the_callback_thread() {
        // The audio callback closure captures the mixer by value, and the
        // thread it runs on is created by the audio backend with a stack far
        // smaller than a Rust thread's. Growing a per-track field is what
        // pushed this over once already: a strip gained a delay line and a
        // reverb, sixty-four of them inline came to seventy kilobytes, and the
        // application aborted before its window opened with nothing but
        // "thread '<unknown>' has overflowed its stack" to go on.
        //
        // Per-track state belongs on the heap. This is a tripwire for the next
        // field, since the failure it catches is invisible until it is fatal.
        const BUDGET: usize = 8 * 1_024;
        let size = std::mem::size_of::<TrackMixer>();
        assert!(
            size <= BUDGET,
            "TrackMixer is {size} bytes, past the {BUDGET} the callback thread can hold; \
             put the new per-track state behind a Vec or a Box rather than inline"
        );
    }

    #[test]
    fn monitoring_is_heard_with_the_transport_stopped() {
        // Practising, setting a level and dialling in an amp all happen with
        // the transport stopped. Requiring playback to hear yourself makes the
        // feature useless for the thing it is most wanted for.
        let (left, _) = render_with_transport(false);
        assert!(
            left.iter().any(|value| value.abs() > 0.01),
            "nothing was heard while stopped"
        );
    }

    #[test]
    fn a_stopped_transport_does_not_replay_the_playhead() {
        // Stopped, the position does not advance, so rendering the timeline
        // would emit the same fragment on every callback.
        let (stopped, _) = render_with_transport(false);
        let (running, _) = render_with_transport(true);
        let level = |buffer: &[f32]| buffer.iter().map(|value| value.abs()).sum::<f32>();
        // The clip under the playhead is loud; only the quiet live input should
        // be present while stopped.
        assert!(
            level(&running) > level(&stopped) * 2.0,
            "the stopped mix ({}) carried the timeline as well as the input ({})",
            level(&stopped),
            level(&running)
        );
    }

    #[test]
    fn one_bar_count_in_respects_the_meter() {
        assert_eq!(
            calculate_count_in_frames(SampleRate::DEFAULT, 120, 4, 4, 1),
            96_000
        );
        assert_eq!(
            calculate_count_in_frames(SampleRate::DEFAULT, 120, 6, 8, 1),
            72_000
        );
    }

    #[test]
    fn pending_recording_activates_only_at_scheduled_frame() {
        let shared = Shared::new();
        shared.record_pending.store(true, Ordering::Release);
        shared.record_start_frame.store(1_000, Ordering::Release);

        activate_recording_if_due(&shared, 999);
        assert!(shared.record_pending.load(Ordering::Acquire));
        assert!(!shared.recording.load(Ordering::Acquire));

        activate_recording_if_due(&shared, 1_000);
        assert!(!shared.record_pending.load(Ordering::Acquire));
        assert!(shared.recording.load(Ordering::Acquire));
    }

    #[test]
    fn output_monitors_are_not_capture_devices() {
        assert!(is_output_monitor_name(
            "Monitor of Scarlett Solo 4th Gen Analog Stereo"
        ));
        assert!(is_output_monitor_name("alsa_output.usb-focusrite.monitor"));
        assert!(!is_output_monitor_name(
            "Scarlett Solo 4th Gen Analog Surround 4.0"
        ));
    }
}
