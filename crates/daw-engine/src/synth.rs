#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

//! A polyphonic General MIDI synthesiser for instrument tracks.
//!
//! Real-time contract: [`Synth::render`] allocates nothing, locks nothing and
//! does bounded work per block. Voices live in a fixed array and every
//! wavetable is built once, before the stream opens, into a [`GmBank`] shared
//! by every track.
//!
//! A voice is three detuned oscillators reading a band-limited table, plus
//! noise, through a resonant low-pass under an exponential envelope. Four
//! things do most of the work of sounding like an instrument rather than a
//! synthesiser:
//!
//! - **Exponential envelopes.** Every physical resonator decays by a constant
//!   fraction per unit time. A linear ramp to zero is audibly synthetic.
//! - **Key-tracked decay.** A piano's bottom string rings for half a minute
//!   and its top one for well under a second.
//! - **An onset transient.** The hammer, the pick, the breath — a short burst
//!   of noise before the tone speaks. Ears identify instruments largely from
//!   their first few milliseconds.
//! - **Never repeating exactly.** Tuning, level and timbre move a little from
//!   note to note, derived from the note itself so a render is still
//!   reproducible bit for bit.
//!
//! Channel-10 tracks take a different path: sine partials that sweep downwards
//! in pitch over filtered noise, which is what a drum kit actually is.

use std::sync::Arc;

use daw_core::SampleRate;
use daw_midi::ScheduledNote;

use crate::gm::{self, DrumVoice, GmBank, Patch, Wavetable};

/// Simultaneous notes. Transcribed piano and guitar parts rarely exceed a
/// dozen; the rest is headroom for pedalled chords and cymbal tails.
pub const MAX_VOICES: usize = 48;

/// Oscillators per voice: a centre, and a detuned pair placed left and right.
const UNISON: usize = 3;
/// Envelope level below which a voice is inaudible and its slot can be reused.
const SILENCE: f32 = 1e-4;
/// Decibels a patch's decay and release times are measured over.
const DECAY_DB: f32 = 60.0;
/// How much of a drum track goes to the shared reverb. A kit wants a room, not
/// a hall, or the transients turn to mush.
const DRUM_REVERB_SEND: f32 = 0.12;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum Stage {
    #[default]
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

#[derive(Clone, Copy, Debug, Default)]
struct Voice {
    stage: Stage,
    /// Phase of each oscillator, in cycles, and how far it moves per frame.
    /// A zero increment means the oscillator is unused.
    phase: [f32; UNISON],
    increment: [f32; UNISON],
    /// Each oscillator's gain into the left and right channels. Detune heard
    /// in mono is beating; spread across the stereo field it is width.
    osc_gain: [[f32; 2]; UNISON],
    /// Where the breath, bow or drum noise sits, which is the voice's own
    /// position rather than any one oscillator's.
    noise_gain: [f32; 2],
    /// Index into the bank, resolved once at note-on: which band-limited table
    /// this voice's pitch reads from.
    table: usize,
    amplitude: f32,
    envelope: f32,
    end_frame: u64,

    attack_step: f32,
    /// Envelopes fall by a constant fraction per frame, not a constant amount.
    decay_coeff: f32,
    sustain: f32,
    release_coeff: f32,

    /// Steady breath or bow noise, as a fraction of the tone.
    noise_mix: f32,
    /// The onset transient — hammer, pick or chiff — and its fast decay.
    burst: f32,
    burst_env: f32,
    burst_coeff: f32,
    /// An extra decay on the tone alone, so a drum's head can stop while its
    /// wires rattle on.
    tone_env: f32,
    tone_coeff: f32,

    /// The low-pass sweeps from `filter_coeff_peak` down to `filter_coeff_base`
    /// as `filter_env` falls, brightening the onset. Drums pin both
    /// coefficients to the same value, so their filter is fixed.
    resonance: f32,
    filter_coeff_base: f32,
    filter_coeff_peak: f32,
    filter_env: f32,
    filter_env_coeff: f32,
    /// Two cascaded one-pole stages with a feedback path make a resonant
    /// -12 dB/octave low-pass, per channel.
    filter_state: [[f32; 2]; 2],
    /// One-pole high-pass on the noise, which is the difference between a
    /// hi-hat and a hiss.
    highpass_coeff: f32,
    highpass_state: f32,

    /// Vibrato as a quadrature oscillator: `(lfo_sin, lfo_cos)` is rotated by a
    /// fixed angle each frame, so pitch is modulated without a `sin` call per
    /// sample. `vibrato_depth` is the peak deviation as a frequency ratio, and
    /// `vibrato_ramp` swells it in from silence over the note's onset.
    lfo_sin: f32,
    lfo_cos: f32,
    lfo_rot_sin: f32,
    lfo_rot_cos: f32,
    vibrato_depth: f32,
    vibrato_ramp: f32,
    vibrato_ramp_step: f32,

    /// Drum voices sweep in pitch and ignore how long the note is held.
    is_drum: bool,
    pitch_multiplier: f32,
    pitch_target: f32,
    pitch_coeff: f32,
}

impl Voice {
    fn level(&self) -> f32 {
        self.envelope * self.amplitude
    }

    fn is_active(&self) -> bool {
        self.stage != Stage::Idle
    }
}

pub struct Synth {
    sample_rate: SampleRate,
    bank: Arc<GmBank>,
    program: u8,
    is_drum_kit: bool,
    voices: [Voice; MAX_VOICES],
    /// Index of the next note to consider, valid only while playing forward.
    cursor: usize,
    /// Frame the next block is expected to start at; a mismatch means a seek.
    expected_frame: u64,
    level: f32,
    /// Xorshift state for the noise source. A voice cannot allocate an RNG, and
    /// noise only has to be uncorrelated, not cryptographic.
    noise_state: u32,
}

impl Synth {
    #[must_use]
    pub fn new(sample_rate: SampleRate, bank: Arc<GmBank>) -> Self {
        Self {
            sample_rate,
            bank,
            program: 0,
            is_drum_kit: false,
            voices: [Voice::default(); MAX_VOICES],
            cursor: 0,
            expected_frame: u64::MAX,
            level: 0.5,
            noise_state: 0x2545_F491,
        }
    }

    /// Selects the General MIDI program this track plays.
    pub fn set_program(&mut self, program: u8) {
        if self.program != program.min(127) {
            self.program = program.min(127);
            self.reset();
        }
    }

    /// Switches the track between pitched instrument and drum kit.
    pub fn set_drum_kit(&mut self, is_drum_kit: bool) {
        if self.is_drum_kit != is_drum_kit {
            self.is_drum_kit = is_drum_kit;
            self.reset();
        }
    }

    #[must_use]
    pub const fn program(&self) -> u8 {
        self.program
    }

    pub fn set_level(&mut self, level: f32) {
        self.level = level.clamp(0.0, 2.0);
    }

    /// How much of this track belongs in the shared reverb. A hall for strings,
    /// a small room for a piano, next to nothing for a bass.
    #[must_use]
    pub fn reverb_send(&self) -> f32 {
        if self.is_drum_kit {
            DRUM_REVERB_SEND
        } else {
            self.bank.patch(self.program).reverb_send
        }
    }

    /// Silences every voice, for a stop or a track becoming inaudible.
    pub fn reset(&mut self) {
        self.voices = [Voice::default(); MAX_VOICES];
        self.cursor = 0;
        self.expected_frame = u64::MAX;
    }

    #[must_use]
    pub fn active_voices(&self) -> usize {
        self.voices.iter().filter(|voice| voice.is_active()).count()
    }

    /// Renders one block, adding into `left` and `right`.
    ///
    /// `notes` must be sorted by `start_frame`. The synth keeps a cursor into
    /// it so ordinary playback costs one comparison per block, and rebuilds the
    /// cursor by binary search only when the transport has jumped.
    pub fn render(
        &mut self,
        notes: &[ScheduledNote],
        block_start: u64,
        left: &mut [f32],
        right: &mut [f32],
    ) {
        if block_start != self.expected_frame {
            self.seek(notes, block_start);
        }
        let frames = left.len().min(right.len());

        // Split the borrow so the shared bank can be read while the voices are
        // written. Cloning the Arc instead would put an atomic in the callback
        // for no reason.
        let Self {
            bank,
            program,
            is_drum_kit,
            voices,
            cursor,
            level,
            noise_state,
            sample_rate,
            ..
        } = self;
        let patch = bank.patch(*program);
        let rate = sample_rate.get().max(1) as f32;

        for offset in 0..frames {
            let frame = block_start.saturating_add(offset as u64);

            while let Some(note) = notes.get(*cursor) {
                if note.start_frame > frame {
                    break;
                }
                if note.end_frame > frame {
                    let slot = free_voice(voices);
                    voices[slot] = new_voice(note, *program, patch, *is_drum_kit, rate);
                }
                *cursor += 1;
            }

            let mut sample = [0.0_f32; 2];
            for voice in voices.iter_mut() {
                if !voice.is_active() {
                    continue;
                }
                // Drums are one-shots: their length is the decay, not the note.
                if !voice.is_drum && voice.stage != Stage::Release && frame >= voice.end_frame {
                    voice.stage = Stage::Release;
                }
                let table = if voice.is_drum {
                    bank.sine()
                } else {
                    bank.table_at(voice.table)
                };
                let value = advance(voice, table, next_noise(noise_state));
                sample[0] += value[0];
                sample[1] += value[1];
            }

            left[offset] += sample[0] * *level;
            right[offset] += sample[1] * *level;
        }

        self.expected_frame = block_start.saturating_add(frames as u64);
    }

    fn start_voice(&mut self, note: &ScheduledNote) {
        let slot = free_voice(&self.voices);
        let rate = self.sample_rate.get().max(1) as f32;
        self.voices[slot] = new_voice(
            note,
            self.program,
            self.bank.patch(self.program),
            self.is_drum_kit,
            rate,
        );
    }

    /// Rebuilds state after the transport jumped.
    ///
    /// Notes already sounding at the new position are restarted so scrubbing
    /// into the middle of a chord still plays it, rather than leaving silence
    /// until the next note begins.
    fn seek(&mut self, notes: &[ScheduledNote], frame: u64) {
        self.voices = [Voice::default(); MAX_VOICES];
        // Strictly before: a note starting exactly on the seek position has not
        // happened yet and must stay pending, or a drum hit landing on the
        // playhead would be swallowed by the seek instead of sounding.
        self.cursor = notes.partition_point(|note| note.start_frame < frame);
        // Drums are instantaneous; restarting them at a seek would fire every
        // hit that had already passed, all at once.
        if self.is_drum_kit {
            return;
        }
        for note in notes[..self.cursor].iter().rev().take(MAX_VOICES) {
            if note.end_frame > frame {
                self.start_voice(note);
            }
        }
    }
}

/// Xorshift32. Deterministic, so rendering the same session twice produces
/// the same file twice.
fn next_noise(state: &mut u32) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    (*state >> 8) as f32 / 8_388_608.0 - 1.0
}

/// How far one note strays from the written one.
///
/// Derived by hashing the note rather than drawing from the running noise
/// source, so the same note varies the same way however the block is split and
/// wherever the transport was before it — an offline render and a live pass
/// come out identical.
#[derive(Clone, Copy, Debug)]
struct Variation {
    tuning: f32,
    level: f32,
    timbre: f32,
}

fn variation(note: &ScheduledNote) -> Variation {
    let mut hash = (note.start_frame as u32)
        ^ ((note.start_frame >> 32) as u32).wrapping_mul(0x9E37_79B9)
        ^ u32::from(note.pitch).wrapping_mul(0x0085_EBCA)
        ^ u32::from(note.velocity).wrapping_mul(0x00C2_B2AE);
    hash = (hash ^ (hash >> 16)).wrapping_mul(0x21F0_AAAD);
    hash = (hash ^ (hash >> 15)).wrapping_mul(0x735A_2D97);
    hash ^= hash >> 15;
    // Disjoint ten-bit slices, each mapped to -1..1.
    let unit = |shift: u32| ((hash >> shift) & 0x3FF) as f32 / 512.0 - 1.0;
    Variation {
        tuning: unit(0),
        level: unit(10),
        timbre: unit(20),
    }
}

fn new_voice(
    note: &ScheduledNote,
    program: u8,
    patch: &Patch,
    is_drum_kit: bool,
    rate: f32,
) -> Voice {
    let amplitude = f32::from(note.velocity) / 127.0;
    let variation = variation(note);
    if is_drum_kit {
        drum_voice_state(gm::drum_voice(note.pitch), rate, amplitude, variation)
    } else {
        pitched_voice_state(patch, program, note, rate, amplitude, variation)
    }
}

/// Picks a free voice, or steals the quietest one when all are busy. Stealing
/// the quietest is what makes a dense transcription degrade gracefully instead
/// of dropping whichever note happened to be newest.
fn free_voice(voices: &[Voice; MAX_VOICES]) -> usize {
    if let Some(index) = voices.iter().position(|voice| !voice.is_active()) {
        return index;
    }
    voices
        .iter()
        .enumerate()
        .min_by(|left, right| left.1.level().total_cmp(&right.1.level()))
        .map_or(0, |(index, _)| index)
}

fn pitched_voice_state(
    patch: &Patch,
    program: u8,
    note: &ScheduledNote,
    rate: f32,
    amplitude: f32,
    variation: Variation,
) -> Voice {
    let detune = variation.tuning * patch.humanise_cents;
    let frequency = midi_to_frequency(note.pitch) * cents_ratio(detune);
    let increment = frequency / rate;

    // Brightness is a multiple of the note's own pitch, so high notes stay
    // proportionally as bright as low ones instead of turning into sine waves.
    // A softly played note is darker as well as quieter: velocity scales the
    // cutoff down as `velocity_brightness` approaches one.
    let ceiling = rate * 0.45;
    let velocity_scale = 1.0 - patch.velocity_brightness * (1.0 - amplitude);
    // The onset reaches the patch's full brightness; the tone then darkens to a
    // fraction of that as it rings, which is where the filter can actually be
    // heard — the band-limited tables have little energy above the nominal
    // cutoff, so the sweep has to move down into the harmonics, not up past
    // them, to change the timbre.
    let peak_cutoff =
        (frequency * patch.brightness * velocity_scale * (1.0 + variation.timbre * 0.06))
            .clamp(60.0, ceiling);
    let base_cutoff = (peak_cutoff / (1.0 + patch.filter_env)).clamp(60.0, ceiling);

    let vibrato_depth = cents_ratio(patch.vibrato_cents) - 1.0;
    let vibrato_angle = std::f32::consts::TAU * patch.vibrato_hz / rate;

    // Where the note sits between the ends of the keyboard, -1 to 1, so a
    // piano is laid out under the listener the way it is recorded.
    let keyboard_position = (f32::from(note.pitch) - 60.0) / 42.0;
    let voice_pan = (keyboard_position * patch.keyboard_spread).clamp(-1.0, 1.0);

    let mut voice = Voice {
        stage: Stage::Attack,
        table: gm::table_index(program, note.pitch),
        amplitude: amplitude * patch.level * (1.0 + variation.level * 0.05),
        envelope: 0.0,
        end_frame: note.end_frame,
        attack_step: 1.0 / (patch.attack_seconds * rate).max(1.0),
        decay_coeff: decay_coefficient(patch.decay_seconds_at(note.pitch), rate),
        sustain: patch.sustain,
        release_coeff: decay_coefficient(patch.release_seconds, rate),
        noise_mix: patch.noise,
        burst: patch.attack_noise,
        burst_env: 1.0,
        burst_coeff: decay_coefficient(patch.attack_noise_seconds, rate),
        tone_env: 1.0,
        tone_coeff: 1.0,
        resonance: patch.resonance.clamp(0.0, 1.0),
        filter_coeff_base: one_pole_coefficient(base_cutoff, rate),
        filter_coeff_peak: one_pole_coefficient(peak_cutoff, rate),
        filter_env: 1.0,
        filter_env_coeff: decay_coefficient(patch.filter_decay_seconds, rate),
        highpass_coeff: 0.0,
        lfo_cos: 1.0,
        lfo_rot_sin: vibrato_angle.sin(),
        lfo_rot_cos: vibrato_angle.cos(),
        vibrato_depth,
        vibrato_ramp_step: 1.0 / (patch.vibrato_delay_seconds * rate).max(1.0),
        pitch_multiplier: 1.0,
        pitch_target: 1.0,
        pitch_coeff: 1.0,
        ..Voice::default()
    };

    voice.increment[0] = increment;
    voice.osc_gain[0] = pan_gains(voice_pan);
    voice.noise_gain = voice.osc_gain[0];
    if patch.detune_cents > 0.0 {
        // The centre keeps the pitch; the pair straddles it and is pushed out
        // to the sides. Gains sum to one so a detuned patch is no louder than
        // a single oscillator.
        let spread = patch.stereo_spread.clamp(0.0, 1.0);
        // Deliberately lopsided. Detuned by equal amounts either side, the two
        // would beat against the centre in step and periodically cancel it
        // outright, which is heard as the note swelling and dropping away.
        // Nothing tunes that evenly; unequal offsets beat at unequal rates and
        // never all line up.
        voice.increment[1] = increment / cents_ratio(patch.detune_cents);
        voice.increment[2] = increment * cents_ratio(patch.detune_cents * 0.62);
        // Start them apart. Three strings struck in perfect phase agreement
        // would be loudest at the onset and thin out as they drifted, which is
        // both wrong and audible as a swell backwards; no two strings on a real
        // instrument agree on where their cycle begins.
        voice.phase[1] = (0.37 + variation.tuning * 0.5).fract().abs();
        voice.phase[2] = (0.71 + variation.timbre * 0.5).fract().abs();
        for (index, side) in [(1_usize, -1.0_f32), (2, 1.0)] {
            let pan = (voice_pan + side * spread).clamp(-1.0, 1.0);
            let gains = pan_gains(pan);
            voice.osc_gain[index] = [gains[0] * 0.3, gains[1] * 0.3];
        }
        voice.osc_gain[0] = [voice.osc_gain[0][0] * 0.4, voice.osc_gain[0][1] * 0.4];
    }
    voice
}

fn drum_voice_state(drum: DrumVoice, rate: f32, amplitude: f32, variation: Variation) -> Voice {
    // No two hits on a real kit are the same. Without this a hi-hat pattern is
    // instantly recognisable as a machine.
    let decay_seconds = (drum.decay_seconds * (1.0 + variation.level * 0.12)).max(0.005);
    let tone_seconds = (drum.tone_decay() * (1.0 + variation.level * 0.12)).max(0.005);
    let frequency = drum.frequency * (1.0 + variation.tuning * 0.03);
    let ceiling = rate * 0.45;
    // A hard hit is brighter as well as louder, on a drum more than anything.
    let cutoff = (drum.noise_cutoff * (0.55 + 0.45 * amplitude) * (1.0 + variation.timbre * 0.08))
        .clamp(200.0, ceiling);

    let mut voice = Voice {
        stage: Stage::Decay,
        amplitude: amplitude * drum.level * (1.0 + variation.level * 0.08),
        envelope: 1.0,
        end_frame: u64::MAX,
        attack_step: 1.0,
        decay_coeff: decay_coefficient(decay_seconds, rate),
        sustain: 0.0,
        release_coeff: decay_coefficient(decay_seconds, rate),
        noise_mix: drum.noise,
        burst: drum.click,
        burst_env: 1.0,
        burst_coeff: decay_coefficient(0.004, rate),
        tone_env: 1.0,
        tone_coeff: decay_coefficient(tone_seconds, rate),
        resonance: if drum.frequency > 0.0 { 0.15 } else { 0.0 },
        filter_coeff_base: one_pole_coefficient(cutoff, rate),
        filter_coeff_peak: one_pole_coefficient(cutoff, rate),
        filter_env: 0.0,
        filter_env_coeff: 0.0,
        highpass_coeff: if drum.noise_highpass > 0.0 {
            one_pole_coefficient(drum.noise_highpass.min(ceiling), rate)
        } else {
            0.0
        },
        lfo_cos: 1.0,
        lfo_rot_cos: 1.0,
        vibrato_ramp_step: 1.0,
        is_drum: true,
        pitch_multiplier: 1.0,
        pitch_target: drum.pitch_drop.max(0.05),
        pitch_coeff: decay_coefficient(drum.pitch_drop_seconds, rate),
        ..Voice::default()
    };

    let gains = pan_gains(drum.pan);
    voice.noise_gain = gains;
    if frequency > 0.0 {
        voice.increment[0] = frequency / rate;
        voice.osc_gain[0] = gains;
        if drum.partial_ratio > 0.0 && drum.partial_level > 0.0 {
            voice.increment[1] = frequency * drum.partial_ratio / rate;
            voice.osc_gain[1] = [gains[0] * drum.partial_level, gains[1] * drum.partial_level];
        }
    }
    voice
}

/// Falls to `-DECAY_DB` over `seconds`, as a per-frame multiplier.
fn decay_coefficient(seconds: f32, rate: f32) -> f32 {
    let frames = (seconds * rate).max(1.0);
    (-DECAY_DB / 20.0 * std::f32::consts::LN_10 / frames).exp()
}

/// A frequency ratio from an interval in cents.
fn cents_ratio(cents: f32) -> f32 {
    2.0_f32.powf(cents / 1_200.0)
}

/// Constant-power pan: centre is unity in both channels, hard over is `√2` in
/// one and silence in the other, and the total power is the same throughout.
fn pan_gains(pan: f32) -> [f32; 2] {
    let angle = (pan.clamp(-1.0, 1.0) + 1.0) * std::f32::consts::FRAC_PI_4;
    [
        angle.cos() * std::f32::consts::SQRT_2,
        angle.sin() * std::f32::consts::SQRT_2,
    ]
}

/// One-pole low-pass coefficient for a cutoff in Hz.
fn one_pole_coefficient(cutoff: f32, rate: f32) -> f32 {
    let normalised = (cutoff / rate).clamp(1e-5, 0.45);
    1.0 - (-std::f32::consts::TAU * normalised).exp()
}

/// Advances one voice by a frame and returns its stereo contribution.
fn advance(voice: &mut Voice, table: &Wavetable, noise: f32) -> [f32; 2] {
    match voice.stage {
        Stage::Idle => return [0.0; 2],
        Stage::Attack => {
            voice.envelope += voice.attack_step;
            if voice.envelope >= 1.0 {
                voice.envelope = 1.0;
                voice.stage = Stage::Decay;
            }
        }
        Stage::Decay => {
            // Towards the sustain level, by a constant fraction per frame. A
            // patch that does not sustain therefore approaches silence rather
            // than crossing it, and is retired at the audibility floor.
            voice.envelope = voice.sustain + (voice.envelope - voice.sustain) * voice.decay_coeff;
            if voice.sustain > SILENCE {
                if voice.envelope - voice.sustain <= SILENCE {
                    voice.envelope = voice.sustain;
                    voice.stage = Stage::Sustain;
                }
            } else if voice.envelope <= SILENCE {
                *voice = Voice::default();
                return [0.0; 2];
            }
        }
        Stage::Sustain => {}
        Stage::Release => {
            voice.envelope *= voice.release_coeff;
            if voice.envelope <= SILENCE {
                *voice = Voice::default();
                return [0.0; 2];
            }
        }
    }

    // Pitched voices bend with the vibrato; drums fall towards their target.
    let pitch = if voice.is_drum {
        let multiplier = voice.pitch_multiplier;
        voice.pitch_multiplier =
            voice.pitch_target + (voice.pitch_multiplier - voice.pitch_target) * voice.pitch_coeff;
        multiplier
    } else {
        // Rotate the quadrature LFO one step; a zero-Hz vibrato leaves it at
        // (0, 1), so a patch without vibrato costs nothing and never drifts.
        let sin = voice.lfo_sin * voice.lfo_rot_cos + voice.lfo_cos * voice.lfo_rot_sin;
        let cos = voice.lfo_cos * voice.lfo_rot_cos - voice.lfo_sin * voice.lfo_rot_sin;
        voice.lfo_sin = sin;
        voice.lfo_cos = cos;
        voice.vibrato_ramp = (voice.vibrato_ramp + voice.vibrato_ramp_step).min(1.0);
        1.0 + voice.vibrato_depth * voice.vibrato_ramp * sin
    };

    let mut tone = [0.0_f32; 2];
    for index in 0..UNISON {
        let increment = voice.increment[index];
        if increment <= 0.0 {
            continue;
        }
        let value = table.sample(voice.phase[index]) * voice.tone_env;
        tone[0] += value * voice.osc_gain[index][0];
        tone[1] += value * voice.osc_gain[index][1];
        let mut phase = voice.phase[index] + increment * pitch;
        if phase >= 1.0 {
            phase -= phase.floor();
        }
        voice.phase[index] = phase;
    }
    if voice.tone_coeff < 1.0 {
        voice.tone_env *= voice.tone_coeff;
    }

    // A hi-hat is noise with everything below a few kilohertz taken out; the
    // same filter keeps a breath from thickening the low end of a flute.
    let shaped = if voice.highpass_coeff > 0.0 {
        voice.highpass_state += voice.highpass_coeff * (noise - voice.highpass_state);
        noise - voice.highpass_state
    } else {
        noise
    };
    let noise_amount = voice.noise_mix + voice.burst * voice.burst_env;
    voice.burst_env *= voice.burst_coeff;
    let noise_gain = voice.noise_gain;
    let tone_gain = (1.0 - voice.noise_mix).max(0.0);

    // The filter cutoff sweeps from its bright onset down to the settled tone.
    let coefficient = voice.filter_coeff_base
        + (voice.filter_coeff_peak - voice.filter_coeff_base) * voice.filter_env;
    voice.filter_env *= voice.filter_env_coeff;

    // Feeding the second stage's output back into the first puts a resonant
    // peak at the corner, which is what makes the sweep read as a body rather
    // than a blanket. The make-up gain restores what the feedback takes off.
    let make_up = 1.0 + voice.resonance;
    let level = voice.level();
    let mut output = [0.0_f32; 2];
    for channel in 0..2 {
        let state = &mut voice.filter_state[channel];
        let mixed = tone[channel] * tone_gain + shaped * noise_amount * noise_gain[channel]
            - voice.resonance * state[1];
        state[0] += coefficient * (mixed - state[0]);
        state[1] += coefficient * (state[0] - state[1]);
        output[channel] = state[1] * make_up * level;
    }
    output
}

/// Equal-temperament frequency for a MIDI pitch, A4 = 440 Hz.
#[must_use]
pub fn midi_to_frequency(pitch: u8) -> f32 {
    440.0 * 2.0_f32.powf((f32::from(pitch) - 69.0) / 12.0)
}

#[cfg(test)]
mod tests {
    // Comparing a voice's vibrato depth to exactly zero is the assertion:
    // a struck instrument must carry no vibrato at all, not merely little.
    #![allow(clippy::float_cmp)]
    use super::*;

    /// Built once for the whole test binary: the tables cost real work to
    /// build, and a test that rebuilds them per note takes minutes.
    fn bank() -> Arc<GmBank> {
        static BANK: std::sync::OnceLock<Arc<GmBank>> = std::sync::OnceLock::new();
        Arc::clone(BANK.get_or_init(|| Arc::new(GmBank::new(SampleRate::DEFAULT))))
    }

    fn synth() -> Synth {
        Synth::new(SampleRate::DEFAULT, bank())
    }

    fn note(start: u64, end: u64, pitch: u8) -> ScheduledNote {
        ScheduledNote {
            start_frame: start,
            end_frame: end,
            pitch,
            velocity: 100,
        }
    }

    fn energy(buffer: &[f32]) -> f32 {
        buffer.iter().map(|value| value.abs()).sum()
    }

    /// Renders one note held for the whole block and returns the left channel.
    fn render_note(program: u8, drums: bool, pitch: u8, frames: usize) -> Vec<f32> {
        render_stereo(program, drums, pitch, 100, frames).0
    }

    fn render_stereo(
        program: u8,
        drums: bool,
        pitch: u8,
        velocity: u8,
        frames: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut synth = synth();
        synth.set_program(program);
        synth.set_drum_kit(drums);
        let notes = [ScheduledNote {
            start_frame: 0,
            end_frame: frames as u64,
            pitch,
            velocity,
        }];
        let mut left = vec![0.0; frames];
        let mut right = vec![0.0; frames];
        synth.render(&notes, 0, &mut left, &mut right);
        (left, right)
    }

    #[test]
    fn concert_a_is_440_hertz() {
        assert!((midi_to_frequency(69) - 440.0).abs() < 1e-3);
        assert!((midi_to_frequency(81) - 880.0).abs() < 1e-2);
    }

    #[test]
    fn a_note_sounds_only_while_it_lasts() {
        let mut synth = synth();
        // An organ sustains, so this tests the note bounds and not the decay.
        synth.set_program(16);
        let notes = [note(1_000, 5_000, 60)];
        let (mut left, mut right) = ([0.0; 512], [0.0; 512]);
        synth.render(&notes, 0, &mut left, &mut right);
        assert!(energy(&left) < 1e-6, "sound before the note starts");

        let (mut left, mut right) = ([0.0; 512], [0.0; 512]);
        synth.render(&notes, 1_024, &mut left, &mut right);
        assert!(energy(&left) > 0.5, "the note did not sound");

        let (mut left, mut right) = ([0.0; 512], [0.0; 512]);
        synth.render(&notes, 40_000, &mut left, &mut right);
        assert!(energy(&left) < 1e-6, "the note outlived its release");
    }

    #[test]
    fn different_programs_produce_different_sound() {
        let piano = render_note(0, false, 60, 4_096);
        let organ = render_note(16, false, 60, 4_096);
        let flute = render_note(73, false, 60, 4_096);
        let difference = |left: &[f32], right: &[f32]| -> f32 {
            left.iter().zip(right).map(|(a, b)| (a - b).abs()).sum()
        };
        assert!(
            difference(&piano, &organ) > 1.0,
            "piano and organ sound alike"
        );
        assert!(
            difference(&piano, &flute) > 1.0,
            "piano and flute sound alike"
        );
        assert!(
            difference(&organ, &flute) > 1.0,
            "organ and flute sound alike"
        );
    }

    #[test]
    fn a_piano_decays_by_itself_and_an_organ_holds() {
        let piano = render_note(0, false, 60, 96_000);
        let organ = render_note(16, false, 60, 96_000);
        let early = |buffer: &[f32]| energy(&buffer[4_000..8_000]);
        let late = |buffer: &[f32]| energy(&buffer[80_000..84_000]);
        assert!(
            late(&piano) < early(&piano) * 0.6,
            "the piano did not decay"
        );
        assert!(
            late(&organ) > early(&organ) * 0.7,
            "the organ should have held"
        );
    }

    #[test]
    fn a_low_string_rings_far_longer_than_a_high_one() {
        // Key-tracked decay. Both notes are held for the whole buffer, so any
        // difference is the instrument's own, not the note length's.
        let low = render_note(0, false, 33, 192_000);
        let high = render_note(0, false, 93, 192_000);
        let tail = |buffer: &[f32]| energy(&buffer[96_000..]);
        let onset = |buffer: &[f32]| energy(&buffer[..8_000]);
        let low_ratio = tail(&low) / onset(&low);
        let high_ratio = tail(&high) / onset(&high);
        assert!(
            low_ratio > high_ratio * 4.0,
            "the bottom of the keyboard ({low_ratio}) died as fast as the top ({high_ratio})"
        );
    }

    #[test]
    fn a_voice_reads_the_table_band_limited_for_its_own_pitch() {
        // The mip levels exist so a bass note can keep the harmonics that make
        // it an instrument rather than a hum, while a treble note stays under
        // Nyquist. A voice has to pick the one for its own pitch.
        let bank = bank();
        let rate = SampleRate::DEFAULT.get() as f32;
        let voice = |pitch: u8| new_voice(&note(0, 48_000, pitch), 0, bank.patch(0), false, rate);
        let bass = voice(33).table;
        let treble = voice(105).table;
        assert_ne!(bass, treble, "both ends of the keyboard read one table");
        assert!(
            bank.table_at(bass).len() > bank.table_at(treble).len(),
            "the bass note is as band-limited as the treble one"
        );
    }

    #[test]
    fn drum_notes_play_a_kit_rather_than_pitches() {
        let kick = render_note(0, true, 36, 24_000);
        let snare = render_note(0, true, 38, 24_000);
        let hat = render_note(0, true, 42, 24_000);
        for (name, buffer) in [("kick", &kick), ("snare", &snare), ("hat", &hat)] {
            assert!(energy(buffer) > 0.5, "the {name} made no sound");
        }
        let tail = |buffer: &[f32]| energy(&buffer[12_000..]);
        assert!(
            tail(&hat) < tail(&kick),
            "the hi-hat rang longer than the kick"
        );
    }

    #[test]
    fn a_hi_hat_is_brighter_than_a_kick() {
        // The high-pass on the noise is what separates a cymbal from a hiss and
        // a hiss from a thud.
        let kick = render_stereo(0, true, 36, 100, 8_000).0;
        let hat = render_stereo(0, true, 42, 100, 8_000).0;
        assert!(
            brightness(&hat[..2_000]) > brightness(&kick[..2_000]) * 4.0,
            "the hi-hat is not sitting above the kick"
        );
    }

    #[test]
    fn repeated_drum_hits_are_not_identical() {
        // A pattern of bit-identical hits is the machine-gun effect, and is the
        // fastest way to hear that a kit is programmed rather than played.
        let mut synth = synth();
        synth.set_drum_kit(true);
        let notes: Vec<ScheduledNote> = (0..2)
            .map(|index| note(index * 8_000, index * 8_000 + 100, 42))
            .collect();
        let mut left = vec![0.0; 16_000];
        let mut right = vec![0.0; 16_000];
        synth.render(&notes, 0, &mut left, &mut right);
        let first = energy(&left[..4_000]);
        let second = energy(&left[8_000..12_000]);
        assert!(first > 0.0 && second > 0.0, "a hit was silent");
        assert!(
            (first - second).abs() / first > 0.005,
            "two hits came out identical: {first} and {second}"
        );
    }

    #[test]
    fn a_drum_hit_ignores_how_long_the_note_is_held() {
        // A one-frame note must not cut a crash cymbal short.
        let mut synth = synth();
        synth.set_drum_kit(true);
        let notes = [note(0, 1, 49)];
        let mut left = vec![0.0; 24_000];
        let mut right = vec![0.0; 24_000];
        synth.render(&notes, 0, &mut left, &mut right);
        assert!(
            energy(&left[8_000..]) > 0.1,
            "the crash was cut off by the note length"
        );
    }

    #[test]
    fn a_kick_drops_in_pitch_well_before_it_stops_sounding() {
        // The pitch envelope is far shorter than the amplitude one; stretched
        // over the whole decay it is a slide whistle, not a kick.
        let rate = SampleRate::DEFAULT.get() as f32;
        let mut voice = drum_voice_state(
            gm::drum_voice(36),
            rate,
            1.0,
            Variation {
                tuning: 0.0,
                level: 0.0,
                timbre: 0.0,
            },
        );
        let bank = bank();
        for _ in 0..(rate * 0.1) as usize {
            advance(&mut voice, bank.sine(), 0.0);
        }
        assert!(
            voice.pitch_multiplier < voice.pitch_target * 1.05,
            "the kick was still sliding a tenth of a second in"
        );
        assert!(
            voice.envelope > 0.15,
            "the kick had already stopped sounding"
        );
    }

    #[test]
    fn a_patch_without_width_stays_centred() {
        // A flute is one player in one place.
        let (left, right) = render_stereo(73, false, 64, 100, 4_096);
        assert!(
            left.iter()
                .zip(right.iter())
                .all(|(l, r)| (l - r).abs() < 1e-6)
        );
    }

    #[test]
    fn an_ensemble_is_wider_than_a_soloist() {
        // Detune heard in mono is beating; spread across the channels it is
        // width, which is the whole difference between a section and a player.
        let width = |(left, right): (Vec<f32>, Vec<f32>)| -> f32 {
            let side: f32 = left.iter().zip(&right).map(|(l, r)| (l - r).abs()).sum();
            let mid: f32 = left.iter().zip(&right).map(|(l, r)| (l + r).abs()).sum();
            side / (mid + 1e-9)
        };
        let flute = width(render_stereo(73, false, 64, 100, 8_192));
        let strings = width(render_stereo(48, false, 64, 100, 8_192));
        assert!(
            strings > flute + 0.1,
            "the string section ({strings}) is no wider than the flute ({flute})"
        );
    }

    #[test]
    fn the_drum_kit_is_spread_across_the_stereo_field() {
        let (hat_left, hat_right) = render_stereo(0, true, 42, 100, 8_000);
        let (crash_left, crash_right) = render_stereo(0, true, 49, 100, 8_000);
        let bias = |left: &[f32], right: &[f32]| energy(right) - energy(left);
        assert!(
            bias(&hat_left, &hat_right) > 0.0,
            "the hats should sit to one side"
        );
        assert!(
            bias(&crash_left, &crash_right) < 0.0,
            "the crash should sit to the other"
        );
    }

    #[test]
    fn rendering_adds_to_the_buffer_rather_than_replacing_it() {
        let mut synth = synth();
        let (mut left, mut right) = ([0.25_f32; 128], [0.25_f32; 128]);
        synth.render(&[], 0, &mut left, &mut right);
        assert!(
            left.iter()
                .all(|value| (*value - 0.25).abs() < f32::EPSILON)
        );
    }

    #[test]
    fn a_chord_plays_every_note() {
        let mut synth = synth();
        synth.set_program(16);
        let notes = [
            note(0, 24_000, 60),
            note(0, 24_000, 64),
            note(0, 24_000, 67),
        ];
        let (mut left, mut right) = ([0.0; 256], [0.0; 256]);
        synth.render(&notes, 0, &mut left, &mut right);
        assert_eq!(synth.active_voices(), 3);
    }

    #[test]
    fn seeking_into_a_held_note_still_plays_it() {
        let mut synth = synth();
        synth.set_program(16);
        let notes = [note(0, 480_000, 60)];
        let (mut left, mut right) = ([0.0; 512], [0.0; 512]);
        synth.render(&notes, 240_000, &mut left, &mut right);
        assert_eq!(synth.active_voices(), 1);
        assert!(energy(&left) > 0.5);
    }

    #[test]
    fn seeking_does_not_replay_drum_hits_that_already_passed() {
        let mut synth = synth();
        synth.set_drum_kit(true);
        let notes: Vec<ScheduledNote> = (0..20)
            .map(|index| note(index * 1_000, index * 1_000 + 10, 36))
            .collect();
        let (mut left, mut right) = ([0.0; 256], [0.0; 256]);
        synth.render(&notes, 19_000, &mut left, &mut right);
        assert!(
            synth.active_voices() <= 1,
            "seeking fired {} old hits at once",
            synth.active_voices()
        );
    }

    #[test]
    fn block_size_does_not_change_the_output() {
        let notes: Vec<ScheduledNote> = (0..8)
            .map(|index| note(index * 4_000, index * 4_000 + 3_000, 60 + index as u8))
            .collect();
        let bank = bank();

        let mut whole = Synth::new(SampleRate::DEFAULT, Arc::clone(&bank));
        let (mut left_whole, mut right_whole) = (vec![0.0; 8_192], vec![0.0; 8_192]);
        whole.render(&notes, 0, &mut left_whole, &mut right_whole);

        let mut split = Synth::new(SampleRate::DEFAULT, bank);
        let (mut left_split, mut right_split) = (vec![0.0; 8_192], vec![0.0; 8_192]);
        for start in (0..8_192).step_by(128) {
            let (left, right) = (
                &mut left_split[start..start + 128],
                &mut right_split[start..start + 128],
            );
            split.render(&notes, start as u64, left, right);
        }

        for (index, (whole, split)) in left_whole.iter().zip(left_split.iter()).enumerate() {
            assert!(
                (whole - split).abs() < 1e-6,
                "frame {index} differs between block sizes: {whole} vs {split}"
            );
        }
    }

    #[test]
    fn the_same_note_varies_but_the_same_render_does_not() {
        // Humanisation must come from the note, not from a running counter, or
        // rendering twice would give two different files.
        let first = render_note(0, false, 60, 4_096);
        let second = render_note(0, false, 60, 4_096);
        assert_eq!(first, second, "the same render came out differently");

        let one = variation(&note(0, 1_000, 60));
        let other = variation(&note(48_000, 49_000, 60));
        assert!(
            (one.tuning - other.tuning).abs() > 1e-6,
            "two notes were tuned identically"
        );
    }

    #[test]
    fn more_notes_than_voices_does_not_overflow() {
        let mut synth = synth();
        synth.set_program(16);
        let notes: Vec<ScheduledNote> = (0..MAX_VOICES * 3)
            .map(|index| note(0, 48_000, 30 + (index % 60) as u8))
            .collect();
        let (mut left, mut right) = ([0.0; 256], [0.0; 256]);
        synth.render(&notes, 0, &mut left, &mut right);
        assert_eq!(synth.active_voices(), MAX_VOICES);
        assert!(left.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn output_stays_within_a_sane_range() {
        for program in 0..=127_u8 {
            let mut synth = synth();
            synth.set_program(program);
            let notes: Vec<ScheduledNote> = (0..MAX_VOICES)
                .map(|index| ScheduledNote {
                    start_frame: 0,
                    end_frame: 96_000,
                    pitch: 40 + index as u8,
                    velocity: 127,
                })
                .collect();
            let (mut left, mut right) = (vec![0.0; 4_096], vec![0.0; 4_096]);
            synth.render(&notes, 0, &mut left, &mut right);
            let peak = left
                .iter()
                .fold(0.0_f32, |peak, value| peak.max(value.abs()));
            assert!(
                peak.is_finite() && peak < 40.0,
                "{} peaked at {peak}",
                gm::program_name(program)
            );
        }
    }

    #[test]
    fn a_resonant_filter_does_not_run_away() {
        // The feedback path is only conditionally stable; every patch must sit
        // well inside the stable region at every pitch.
        for program in 0..=127_u8 {
            for pitch in [24_u8, 60, 108] {
                let buffer = render_stereo(program, false, pitch, 127, 8_000).0;
                let peak = buffer
                    .iter()
                    .fold(0.0_f32, |peak, value| peak.max(value.abs()));
                assert!(
                    peak.is_finite() && peak < 4.0,
                    "{} at pitch {pitch} peaked at {peak}",
                    gm::program_name(program)
                );
            }
        }
    }

    #[test]
    fn zero_length_and_empty_input_are_safe() {
        let mut synth = synth();
        synth.render(&[], 0, &mut [], &mut []);
        let notes = [note(100, 100, 60)];
        let (mut left, mut right) = ([0.0; 64], [0.0; 64]);
        synth.render(&notes, 0, &mut left, &mut right);
        assert!(left.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn reset_silences_everything() {
        let mut synth = synth();
        synth.set_program(16);
        let notes = [note(0, 480_000, 60)];
        let (mut left, mut right) = ([0.0; 256], [0.0; 256]);
        synth.render(&notes, 0, &mut left, &mut right);
        assert_eq!(synth.active_voices(), 1);
        synth.reset();
        assert_eq!(synth.active_voices(), 0);
    }

    #[test]
    fn changing_program_silences_stale_voices() {
        let mut synth = synth();
        synth.set_program(16);
        let notes = [note(0, 480_000, 60)];
        let (mut left, mut right) = ([0.0; 256], [0.0; 256]);
        synth.render(&notes, 0, &mut left, &mut right);
        assert!(synth.active_voices() > 0);
        synth.set_program(42);
        assert_eq!(
            synth.active_voices(),
            0,
            "a voice survived a program change"
        );
    }

    #[test]
    fn no_instrument_is_wildly_louder_than_another() {
        // A transcription with a flute part and a guitar part has to balance
        // without reaching for the faders, so every patch carries a level trim
        // measured against the rest of the bank. Loudness is taken over the
        // first half second, which is what the ear weights for a note that
        // decays away rather than holding.
        let head = SampleRate::DEFAULT.get() as usize / 2;
        let mut quietest = (f32::MAX, 0_u8);
        let mut loudest = (0.0_f32, 0_u8);
        for program in 0..=127_u8 {
            let rendered = render_stereo(program, false, 60, 100, head).0;
            let rms =
                (rendered.iter().map(|value| value * value).sum::<f32>() / head as f32).sqrt();
            assert!(rms > 0.0, "{} is silent", gm::program_name(program));
            if rms < quietest.0 {
                quietest = (rms, program);
            }
            if rms > loudest.0 {
                loudest = (rms, program);
            }
        }
        assert!(
            loudest.0 < quietest.0 * 3.0,
            "{} ({}) drowns out {} ({})",
            gm::program_name(loudest.1),
            loudest.0,
            gm::program_name(quietest.1),
            quietest.0
        );
    }

    #[test]
    fn a_hall_instrument_is_sent_further_into_the_reverb_than_a_bass() {
        let mut synth = synth();
        synth.set_program(48);
        let strings = synth.reverb_send();
        synth.set_program(33);
        let bass = synth.reverb_send();
        assert!(
            strings > bass * 2.0,
            "strings {strings} against bass {bass}"
        );
    }

    /// A spectral-tilt measure: energy of the sample-to-sample change over the
    /// signal's own energy. This is the normalised spectral centroid — high
    /// harmonics carry quadratic weight, so it tracks how bright a tone is
    /// independent of how loud the window is, without being swamped by the
    /// fundamental the way a first-difference-over-level ratio is.
    fn brightness(buffer: &[f32]) -> f32 {
        let change: f32 = buffer
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).powi(2))
            .sum();
        let energy: f32 = buffer.iter().map(|value| value * value).sum();
        change / (energy + 1e-12)
    }

    #[test]
    fn a_struck_note_is_brightest_at_its_attack() {
        // The filter envelope opens the piano's tone at the onset and lets it
        // darken as the string rings, the way a real one does.
        let piano = render_note(0, false, 60, 96_000);
        let onset = brightness(&piano[500..4_500]);
        let ring = brightness(&piano[60_000..64_000]);
        assert!(
            onset > ring * 1.1,
            "the attack ({onset}) was not brighter than the tail ({ring})"
        );
    }

    #[test]
    fn a_harder_hit_is_brighter_not_just_louder() {
        // Velocity must change timbre, not only level: a hard piano note is
        // brighter than a soft one over the same window.
        let soft = render_stereo(0, false, 60, 30, 8_000).0;
        let hard = render_stereo(0, false, 60, 120, 8_000).0;
        let soft_bright = brightness(&soft[500..4_500]);
        let hard_bright = brightness(&hard[500..4_500]);
        assert!(
            hard_bright > soft_bright * 1.1,
            "a hard hit ({hard_bright}) was no brighter than a soft one ({soft_bright})"
        );
    }

    #[test]
    fn a_struck_note_opens_with_a_transient_before_the_tone() {
        // The hammer, then the string. Without the burst the onset is a pure
        // tone fading up, which no acoustic instrument does. Rendering the same
        // voice with and without it isolates the transient from the filter
        // envelope, which brightens the onset for its own reasons.
        let bank = bank();
        let rate = SampleRate::DEFAULT.get() as f32;
        let struck = note(0, 48_000, 60);
        let render = |mut voice: Voice| -> Vec<f32> {
            let table = bank.table_at(voice.table);
            let mut noise = 0x2545_F491_u32;
            (0..2_400)
                .map(|_| advance(&mut voice, table, next_noise(&mut noise))[0])
                .collect()
        };
        let hammered = render(new_voice(&struck, 0, bank.patch(0), false, rate));
        let mut without = new_voice(&struck, 0, bank.patch(0), false, rate);
        without.burst = 0.0;
        let string_alone = render(without);

        let difference = |range: std::ops::Range<usize>| -> f32 {
            hammered[range.clone()]
                .iter()
                .zip(&string_alone[range])
                .map(|(with, out)| (with - out).abs())
                .sum()
        };
        let onset = difference(0..480);
        let body = difference(1_920..2_400);
        assert!(onset > 0.0, "the hammer made no sound at all");
        assert!(
            body < onset * 0.05,
            "the transient ({onset}) was still going in the body ({body})"
        );
    }

    #[test]
    fn sustained_instruments_waver_and_struck_ones_hold_steady() {
        let rate = SampleRate::DEFAULT.get().max(1) as f32;
        let bank = bank();
        let depth = |program: u8| {
            new_voice(
                &note(0, 48_000, 60),
                program,
                bank.patch(program),
                false,
                rate,
            )
            .vibrato_depth
        };
        // Strings and flute carry a vibrato; a piano and a pluck must not.
        assert!(depth(48) > 0.0, "strings should waver");
        assert!(depth(73) > 0.0, "a flute should waver");
        assert_eq!(depth(0), 0.0, "a piano must not waver");
        assert_eq!(depth(45), 0.0, "a plucked string must not waver");
    }

    #[test]
    fn the_vibrato_swells_in_rather_than_starting_at_full_depth() {
        let rate = SampleRate::DEFAULT.get().max(1) as f32;
        let bank = bank();
        let table = bank.table(48, 60);
        let mut voice = new_voice(&note(0, 96_000, 60), 48, bank.patch(48), false, rate);
        // Immediately after the onset the LFO has barely moved off its start;
        // a second later it is oscillating with real depth.
        for _ in 0..64 {
            advance(&mut voice, table, 0.0);
        }
        let early = voice.lfo_sin.abs() * voice.vibrato_ramp;
        for _ in 0..48_000 {
            advance(&mut voice, table, 0.0);
        }
        assert!(
            voice.vibrato_ramp > 0.9,
            "the vibrato never reached full depth"
        );
        assert!(early < 0.1, "the vibrato did not delay its onset");
    }
}
