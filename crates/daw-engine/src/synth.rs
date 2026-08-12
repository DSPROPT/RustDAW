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
//! Each voice is two detuned oscillators plus noise through a one-pole
//! low-pass, under an ADSR — enough for a piano to sound like a piano next to
//! a flute. Channel-10 tracks take a different path entirely: a pitch-swept
//! tone and filtered noise, which is what a drum kit actually is.

use std::sync::Arc;

use daw_core::SampleRate;
use daw_midi::ScheduledNote;

use crate::gm::{self, DrumVoice, GmBank, Patch, TABLE_SIZE};

/// Simultaneous notes. Transcribed piano and guitar parts rarely exceed a
/// dozen; the rest is headroom for pedalled chords and cymbal tails.
pub const MAX_VOICES: usize = 48;

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
    /// Phase of the two oscillators, in cycles.
    phase: f32,
    phase_detuned: f32,
    increment: f32,
    increment_detuned: f32,
    amplitude: f32,
    envelope: f32,
    end_frame: u64,

    attack_step: f32,
    decay_step: f32,
    sustain: f32,
    release_step: f32,

    noise_level: f32,
    /// The low-pass sweeps from `filter_coeff_peak` down to `filter_coeff_base`
    /// as `filter_env` falls from 1 to 0, brightening the onset. Drums pin both
    /// coefficients to the same value, so their filter is fixed.
    filter_coeff_base: f32,
    filter_coeff_peak: f32,
    filter_env: f32,
    filter_env_step: f32,
    /// Two cascaded one-pole stages make a -12 dB/octave low-pass, steep enough
    /// for the sweep to be heard against the band-limited tables.
    filter_state: f32,
    filter_state2: f32,

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
    pitch_step: f32,
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
        let table = bank.table(*program);
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
                    voices[slot] = new_voice(note, patch, *is_drum_kit, rate);
                }
                *cursor += 1;
            }

            let mut sample = 0.0_f32;
            for voice in voices.iter_mut() {
                if !voice.is_active() {
                    continue;
                }
                // Drums are one-shots: their length is the decay, not the note.
                if !voice.is_drum && voice.stage != Stage::Release && frame >= voice.end_frame {
                    voice.stage = Stage::Release;
                }
                sample += advance(voice, table, next_noise(noise_state));
            }

            let value = sample * *level;
            left[offset] += value;
            right[offset] += value;
        }

        self.expected_frame = block_start.saturating_add(frames as u64);
    }

    fn start_voice(&mut self, note: &ScheduledNote) {
        let slot = free_voice(&self.voices);
        let rate = self.sample_rate.get().max(1) as f32;
        self.voices[slot] = new_voice(note, self.bank.patch(self.program), self.is_drum_kit, rate);
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

fn new_voice(note: &ScheduledNote, patch: &Patch, is_drum_kit: bool, rate: f32) -> Voice {
    let amplitude = f32::from(note.velocity) / 127.0;
    if is_drum_kit {
        drum_voice_state(gm::drum_voice(note.pitch), rate, amplitude)
    } else {
        pitched_voice_state(
            patch,
            midi_to_frequency(note.pitch),
            rate,
            amplitude,
            note.end_frame,
        )
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
    frequency: f32,
    rate: f32,
    amplitude: f32,
    end_frame: u64,
) -> Voice {
    let increment = frequency / rate;
    let detune = 2.0_f32.powf(patch.detune_cents / 1_200.0);
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
    let peak_cutoff = (frequency * patch.brightness * velocity_scale).clamp(60.0, ceiling);
    let base_cutoff = (peak_cutoff / (1.0 + patch.filter_env)).clamp(60.0, ceiling);
    let vibrato_depth = 2.0_f32.powf(patch.vibrato_cents / 1_200.0) - 1.0;
    let vibrato_angle = std::f32::consts::TAU * patch.vibrato_hz / rate;
    Voice {
        stage: Stage::Attack,
        phase: 0.0,
        phase_detuned: 0.0,
        increment,
        increment_detuned: if patch.detune_cents > 0.0 {
            increment * detune
        } else {
            0.0
        },
        amplitude: amplitude * patch.level,
        envelope: 0.0,
        end_frame,
        attack_step: 1.0 / (patch.attack_seconds * rate).max(1.0),
        decay_step: 1.0 / (patch.decay_seconds * rate).max(1.0),
        sustain: patch.sustain,
        release_step: 1.0 / (patch.release_seconds * rate).max(1.0),
        noise_level: patch.noise,
        filter_coeff_base: one_pole_coefficient(base_cutoff, rate),
        filter_coeff_peak: one_pole_coefficient(peak_cutoff, rate),
        filter_env: 1.0,
        filter_env_step: 1.0 / (patch.filter_decay_seconds * rate).max(1.0),
        filter_state: 0.0,
        filter_state2: 0.0,
        lfo_sin: 0.0,
        lfo_cos: 1.0,
        lfo_rot_sin: vibrato_angle.sin(),
        lfo_rot_cos: vibrato_angle.cos(),
        vibrato_depth,
        vibrato_ramp: 0.0,
        vibrato_ramp_step: 1.0 / (patch.vibrato_delay_seconds * rate).max(1.0),
        is_drum: false,
        pitch_multiplier: 1.0,
        pitch_step: 0.0,
    }
}

fn drum_voice_state(voice: DrumVoice, rate: f32, amplitude: f32) -> Voice {
    let decay_frames = (voice.decay_seconds * rate).max(1.0);
    let increment = if voice.frequency > 0.0 {
        voice.frequency / rate
    } else {
        0.0
    };
    Voice {
        stage: Stage::Decay,
        phase: 0.0,
        phase_detuned: 0.0,
        increment,
        increment_detuned: 0.0,
        amplitude: amplitude * voice.level,
        envelope: 1.0,
        end_frame: u64::MAX,
        attack_step: 1.0,
        decay_step: 1.0 / decay_frames,
        sustain: 0.0,
        release_step: 1.0 / decay_frames,
        noise_level: voice.noise,
        filter_coeff_base: one_pole_coefficient(voice.noise_cutoff.min(rate * 0.45), rate),
        filter_coeff_peak: one_pole_coefficient(voice.noise_cutoff.min(rate * 0.45), rate),
        filter_env: 0.0,
        filter_env_step: 1.0,
        filter_state: 0.0,
        filter_state2: 0.0,
        lfo_sin: 0.0,
        lfo_cos: 1.0,
        lfo_rot_sin: 0.0,
        lfo_rot_cos: 1.0,
        vibrato_depth: 0.0,
        vibrato_ramp: 0.0,
        vibrato_ramp_step: 1.0,
        is_drum: true,
        pitch_multiplier: 1.0,
        pitch_step: (voice.pitch_drop - 1.0) / decay_frames,
    }
}

/// One-pole low-pass coefficient for a cutoff in Hz.
fn one_pole_coefficient(cutoff: f32, rate: f32) -> f32 {
    let normalised = (cutoff / rate).clamp(1e-5, 0.49);
    1.0 - (-std::f32::consts::TAU * normalised).exp()
}

/// Advances one voice by a frame and returns its contribution.
fn advance(voice: &mut Voice, table: &[f32; TABLE_SIZE], noise: f32) -> f32 {
    match voice.stage {
        Stage::Idle => return 0.0,
        Stage::Attack => {
            voice.envelope += voice.attack_step;
            if voice.envelope >= 1.0 {
                voice.envelope = 1.0;
                voice.stage = Stage::Decay;
            }
        }
        Stage::Decay => {
            voice.envelope -= voice.decay_step;
            if voice.envelope <= voice.sustain {
                if voice.sustain > 0.0 {
                    voice.envelope = voice.sustain;
                    voice.stage = Stage::Sustain;
                } else if voice.envelope <= 0.0 {
                    *voice = Voice::default();
                    return 0.0;
                }
            }
        }
        Stage::Sustain => {}
        Stage::Release => {
            voice.envelope -= voice.release_step;
            if voice.envelope <= 0.0 {
                *voice = Voice::default();
                return 0.0;
            }
        }
    }

    let tone = if voice.increment > 0.0 {
        let first = sample_table(table, voice.phase);
        if voice.increment_detuned > 0.0 {
            // Two oscillators at half level each keeps a detuned patch the
            // same loudness as a single one.
            (first + sample_table(table, voice.phase_detuned)) * 0.5
        } else {
            first
        }
    } else {
        0.0
    };

    let mixed = tone * (1.0 - voice.noise_level) + noise * voice.noise_level;
    // The filter cutoff sweeps from its bright onset down to the settled tone.
    let coefficient =
        voice.filter_coeff_base + (voice.filter_coeff_peak - voice.filter_coeff_base) * voice.filter_env;
    voice.filter_state += coefficient * (mixed - voice.filter_state);
    voice.filter_state2 += coefficient * (voice.filter_state - voice.filter_state2);
    if voice.filter_env > 0.0 {
        voice.filter_env = (voice.filter_env - voice.filter_env_step).max(0.0);
    }

    // Pitched voices bend with the vibrato; drums keep their downward sweep.
    let pitch = if voice.is_drum {
        let multiplier = voice.pitch_multiplier;
        voice.pitch_multiplier = (voice.pitch_multiplier + voice.pitch_step).max(0.05);
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

    voice.phase += voice.increment * pitch;
    if voice.phase >= 1.0 {
        voice.phase -= voice.phase.floor();
    }
    voice.phase_detuned += voice.increment_detuned * pitch;
    if voice.phase_detuned >= 1.0 {
        voice.phase_detuned -= voice.phase_detuned.floor();
    }

    voice.filter_state2 * voice.level()
}

fn sample_table(table: &[f32; TABLE_SIZE], phase: f32) -> f32 {
    let index = phase * TABLE_SIZE as f32;
    let lower = (index as usize) % TABLE_SIZE;
    let upper = (lower + 1) % TABLE_SIZE;
    let fraction = index - index.floor();
    table[lower] + (table[upper] - table[lower]) * fraction
}

/// Equal-temperament frequency for a MIDI pitch, A4 = 440 Hz.
#[must_use]
pub fn midi_to_frequency(pitch: u8) -> f32 {
    440.0 * 2.0_f32.powf((f32::from(pitch) - 69.0) / 12.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth() -> Synth {
        Synth::new(SampleRate::DEFAULT, Arc::new(GmBank::new()))
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
        let mut synth = synth();
        synth.set_program(program);
        synth.set_drum_kit(drums);
        let notes = [note(0, frames as u64, pitch)];
        let mut left = vec![0.0; frames];
        let mut right = vec![0.0; frames];
        synth.render(&notes, 0, &mut left, &mut right);
        left
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
        assert!(difference(&piano, &organ) > 1.0, "piano and organ sound alike");
        assert!(difference(&piano, &flute) > 1.0, "piano and flute sound alike");
        assert!(difference(&organ, &flute) > 1.0, "organ and flute sound alike");
    }

    #[test]
    fn a_piano_decays_by_itself_and_an_organ_holds() {
        let piano = render_note(0, false, 60, 96_000);
        let organ = render_note(16, false, 60, 96_000);
        let early = |buffer: &[f32]| energy(&buffer[4_000..8_000]);
        let late = |buffer: &[f32]| energy(&buffer[80_000..84_000]);
        assert!(late(&piano) < early(&piano) * 0.6, "the piano did not decay");
        assert!(late(&organ) > early(&organ) * 0.7, "the organ should have held");
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
        assert!(tail(&hat) < tail(&kick), "the hi-hat rang longer than the kick");
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
    fn the_two_output_channels_match() {
        let mut synth = synth();
        let notes = [note(0, 24_000, 64)];
        let (mut left, mut right) = ([0.0; 256], [0.0; 256]);
        synth.render(&notes, 0, &mut left, &mut right);
        assert!(
            left.iter()
                .zip(right.iter())
                .all(|(l, r)| (l - r).abs() < f32::EPSILON)
        );
    }

    #[test]
    fn rendering_adds_to_the_buffer_rather_than_replacing_it() {
        let mut synth = synth();
        let (mut left, mut right) = ([0.25_f32; 128], [0.25_f32; 128]);
        synth.render(&[], 0, &mut left, &mut right);
        assert!(left.iter().all(|value| (*value - 0.25).abs() < f32::EPSILON));
    }

    #[test]
    fn a_chord_plays_every_note() {
        let mut synth = synth();
        synth.set_program(16);
        let notes = [note(0, 24_000, 60), note(0, 24_000, 64), note(0, 24_000, 67)];
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
        let bank = Arc::new(GmBank::new());

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
        let mut synth = synth();
        synth.set_program(48);
        let notes: Vec<ScheduledNote> = (0..MAX_VOICES)
            .map(|index| ScheduledNote {
                start_frame: 0,
                end_frame: 96_000,
                pitch: 40 + index as u8,
                velocity: 127,
            })
            .collect();
        let (mut left, mut right) = (vec![0.0; 8_192], vec![0.0; 8_192]);
        synth.render(&notes, 0, &mut left, &mut right);
        let peak = left.iter().fold(0.0_f32, |peak, value| peak.max(value.abs()));
        assert!(peak.is_finite() && peak < 40.0, "peak was {peak}");
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
        assert_eq!(synth.active_voices(), 0, "a voice survived a program change");
    }

    #[test]
    fn noise_is_deterministic_across_runs() {
        let first = render_note(73, false, 60, 2_048);
        let second = render_note(73, false, 60, 2_048);
        assert_eq!(first, second, "a breathy patch must render identically twice");
    }

    /// A spectral-tilt measure: energy of the sample-to-sample change over the
    /// signal's own energy. This is the normalised spectral centroid — high
    /// harmonics carry quadratic weight, so it tracks how bright a tone is
    /// independent of how loud the window is, without being swamped by the
    /// fundamental the way a first-difference-over-level ratio is.
    fn brightness(buffer: &[f32]) -> f32 {
        let change: f32 = buffer.windows(2).map(|pair| (pair[1] - pair[0]).powi(2)).sum();
        let energy: f32 = buffer.iter().map(|value| value * value).sum();
        change / (energy + 1e-12)
    }

    fn render_note_at(program: u8, pitch: u8, velocity: u8, frames: usize) -> Vec<f32> {
        let mut synth = synth();
        synth.set_program(program);
        let notes = [ScheduledNote {
            start_frame: 0,
            end_frame: frames as u64,
            pitch,
            velocity,
        }];
        let mut left = vec![0.0; frames];
        let mut right = vec![0.0; frames];
        synth.render(&notes, 0, &mut left, &mut right);
        left
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
        let soft = render_note_at(0, 60, 30, 8_000);
        let hard = render_note_at(0, 60, 120, 8_000);
        let soft_bright = brightness(&soft[500..4_500]);
        let hard_bright = brightness(&hard[500..4_500]);
        assert!(
            hard_bright > soft_bright * 1.1,
            "a hard hit ({hard_bright}) was no brighter than a soft one ({soft_bright})"
        );
    }

    #[test]
    fn sustained_instruments_waver_and_struck_ones_hold_steady() {
        let rate = SampleRate::DEFAULT.get().max(1) as f32;
        let bank = GmBank::new();
        let depth = |program: u8| new_voice(&note(0, 48_000, 60), bank.patch(program), false, rate).vibrato_depth;
        // Strings and flute carry a vibrato; a piano and a pluck must not.
        assert!(depth(48) > 0.0, "strings should waver");
        assert!(depth(73) > 0.0, "a flute should waver");
        assert_eq!(depth(0), 0.0, "a piano must not waver");
        assert_eq!(depth(45), 0.0, "a plucked string must not waver");
    }

    #[test]
    fn the_vibrato_swells_in_rather_than_starting_at_full_depth() {
        let rate = SampleRate::DEFAULT.get().max(1) as f32;
        let bank = GmBank::new();
        let table = bank.table(48);
        let mut voice = new_voice(&note(0, 96_000, 60), bank.patch(48), false, rate);
        // Immediately after the onset the LFO has barely moved off its start;
        // a second later it is oscillating with real depth.
        for _ in 0..64 {
            advance(&mut voice, table, 0.0);
        }
        let early = voice.lfo_sin.abs() * voice.vibrato_ramp;
        for _ in 0..48_000 {
            advance(&mut voice, table, 0.0);
        }
        assert!(voice.vibrato_ramp > 0.9, "the vibrato never reached full depth");
        assert!(early < 0.1, "the vibrato did not delay its onset");
    }
}
