#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

//! A General MIDI instrument bank, synthesised rather than sampled.
//!
//! Every one of the 128 GM programs and the channel-10 drum kit is described by
//! a handful of synthesis parameters: a harmonic recipe, an envelope, a
//! brightness and a little noise. That is enough to tell a piano from a bowed
//! string from a trumpet, which is what a transcription needs — you are
//! listening to check whether the notes are right, against the actual recording
//! playing beside them.
//!
//! Sampled instruments sound better still, which is what the `soundfont`
//! module is for. This bank is a few megabytes of tables built at startup,
//! needs no asset file, and is what plays when no sound font is loaded.
//!
//! # Band limiting
//!
//! A patch's spectrum runs to [`MAX_HARMONICS`], far past what fits under
//! Nyquist for a high note. Each program therefore holds [`MIP_LEVELS`]
//! wavetables, one per octave of the keyboard, each built with only the
//! harmonics that octave can carry. Without this a bass note has to be cut to
//! the same handful of harmonics as a top note, which is what makes a naive
//! wavetable bass sound like a muffled hum.

use daw_core::SampleRate;

use crate::synth::midi_to_frequency;

/// Harmonics named explicitly by a patch. These set the character; everything
/// above them follows the patch's rolloff.
pub const SEED_HARMONICS: usize = 8;
/// Ceiling on harmonics per table, reached only by the bottom octaves.
pub const MAX_HARMONICS: usize = 48;
/// One band-limited table per octave of MIDI pitch, so 0–127 needs eleven.
pub const MIP_LEVELS: usize = 11;
/// Points in the longest wavetable. Sixteen points per harmonic keeps linear
/// interpolation's error well below the noise the patches deliberately add.
pub const MAX_TABLE_SIZE: usize = 2_048;
/// Points in the shortest, used by the top octaves that carry few harmonics.
pub const MIN_TABLE_SIZE: usize = 256;
pub const PROGRAM_COUNT: usize = 128;
/// Target RMS every table is normalised to. Holding RMS rather than peak
/// constant keeps loudness even across the keyboard: a top note has almost no
/// harmonics and would otherwise jump in level as it crossed a mip boundary.
const TARGET_RMS: f32 = 0.25;

/// The sixteen General MIDI instrument families, in program order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Family {
    Piano,
    ChromaticPercussion,
    Organ,
    Guitar,
    Bass,
    Strings,
    Ensemble,
    Brass,
    Reed,
    Pipe,
    SynthLead,
    SynthPad,
    SynthEffects,
    Ethnic,
    Percussive,
    SoundEffects,
}

impl Family {
    /// The family a GM program belongs to. GM groups programs in eights.
    #[must_use]
    pub const fn of_program(program: u8) -> Self {
        match program / 8 {
            0 => Self::Piano,
            1 => Self::ChromaticPercussion,
            2 => Self::Organ,
            3 => Self::Guitar,
            4 => Self::Bass,
            5 => Self::Strings,
            6 => Self::Ensemble,
            7 => Self::Brass,
            8 => Self::Reed,
            9 => Self::Pipe,
            10 => Self::SynthLead,
            11 => Self::SynthPad,
            12 => Self::SynthEffects,
            13 => Self::Ethnic,
            14 => Self::Percussive,
            _ => Self::SoundEffects,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Piano => "Piano",
            Self::ChromaticPercussion => "Chromatic Percussion",
            Self::Organ => "Organ",
            Self::Guitar => "Guitar",
            Self::Bass => "Bass",
            Self::Strings => "Strings",
            Self::Ensemble => "Ensemble",
            Self::Brass => "Brass",
            Self::Reed => "Reed",
            Self::Pipe => "Pipe",
            Self::SynthLead => "Synth Lead",
            Self::SynthPad => "Synth Pad",
            Self::SynthEffects => "Synth Effects",
            Self::Ethnic => "Ethnic",
            Self::Percussive => "Percussive",
            Self::SoundEffects => "Sound Effects",
        }
    }
}

/// How a patch behaves over time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Patch {
    /// Amplitude of the first [`SEED_HARMONICS`] harmonics, fundamental first.
    pub harmonics: [f32; SEED_HARMONICS],
    /// Harmonics above the seed continue from the last one as `n^-rolloff`.
    /// Low values keep a spectrum rich and buzzy up to [`MAX_HARMONICS`]; high
    /// values fade it to near a sine. Zero in the last seed slot ends the
    /// spectrum there, which is how a flute stays pure.
    pub rolloff: f32,
    /// Thins the extended even harmonics, `0` to `1`. A cylindrical bore — a
    /// clarinet — keeps the odd partials and loses the even ones.
    pub odd_bias: f32,
    pub attack_seconds: f32,
    /// Time to fall 60 dB towards `sustain`. The fall is exponential, as every
    /// resonator's is; a linear ramp is the sound of a synthesiser.
    pub decay_seconds: f32,
    /// Level held while the key is down. Zero means the note always decays
    /// away on its own, like a plucked or struck string.
    pub sustain: f32,
    pub release_seconds: f32,
    /// How much shorter the decay gets as the note rises, in halvings per
    /// octave above middle C. A piano's bottom string rings for half a minute
    /// and its top one for well under a second; one decay time for the whole
    /// keyboard is the loudest tell that a piano patch is synthetic.
    pub decay_key_track: f32,
    /// Low-pass cutoff as a multiple of the note's own frequency. Low values
    /// darken high notes the way a real instrument's body does.
    pub brightness: f32,
    /// How far the tone darkens as it rings, as a fraction of the onset cutoff:
    /// the low-pass starts at the full `brightness` and settles to
    /// `brightness / (1 + filter_env)` over `filter_decay_seconds`. This bright
    /// transient that mellows as the note sustains is what tells a struck string
    /// from a sine — the single biggest step towards a realistic tone.
    pub filter_env: f32,
    pub filter_decay_seconds: f32,
    /// Emphasis at the filter's corner, `0` to `1`. A little resonance is what
    /// makes the filter sweep audible as a body rather than a blanket.
    pub resonance: f32,
    /// How much velocity brightens the tone, beyond making it louder. Zero
    /// means a soft note and a hard one differ only in level; higher values
    /// make a soft note as dark as it is quiet, the way a real instrument
    /// responds to how hard it is played.
    pub velocity_brightness: f32,
    /// Breath or bow noise mixed in for the whole note.
    pub noise: f32,
    /// A burst of noise at the onset, over and above `noise`: the hammer on the
    /// string, the pick on the fret, the breath before the flute speaks. Ears
    /// identify an instrument largely from its first few milliseconds, so this
    /// buys more realism per parameter than anything else here.
    pub attack_noise: f32,
    pub attack_noise_seconds: f32,
    /// Detune between the stacked oscillators, in cents. Gives ensembles and
    /// pads their width, and a piano the slow beating of its unison strings.
    pub detune_cents: f32,
    /// How far apart the detuned oscillators sit in the stereo field, `0` to
    /// `1`. Detune heard in mono is beating; spread across the stereo field it
    /// is width.
    pub stereo_spread: f32,
    /// Panning by keyboard position, `0` to `1`: bass to the left, treble to
    /// the right, as a piano is recorded.
    pub keyboard_spread: f32,
    /// Pitch vibrato: depth in cents, rate in Hz, and how long the note waits
    /// before the vibrato swells in. Zero depth for anything struck or plucked;
    /// a delayed, gentle vibrato is what brings bowed and blown notes to life.
    pub vibrato_cents: f32,
    pub vibrato_hz: f32,
    pub vibrato_delay_seconds: f32,
    /// Note-to-note variation in tuning, in cents, with level and timbre moved
    /// in proportion. No player repeats a note exactly; a sequencer that does
    /// is heard as a machine within about three notes.
    pub humanise_cents: f32,
    /// How much of this instrument is sent to the shared reverb. A concert
    /// hall for strings, a small room for a piano, almost nothing for a bass.
    pub reverb_send: f32,
    pub level: f32,
}

impl Default for Patch {
    fn default() -> Self {
        Self {
            harmonics: [1.0, 0.5, 0.25, 0.12, 0.06, 0.03, 0.015, 0.008],
            rolloff: 1.6,
            odd_bias: 0.0,
            attack_seconds: 0.005,
            decay_seconds: 0.4,
            sustain: 0.6,
            release_seconds: 0.12,
            decay_key_track: 0.0,
            brightness: 8.0,
            filter_env: 1.2,
            filter_decay_seconds: 0.3,
            resonance: 0.2,
            velocity_brightness: 0.5,
            noise: 0.0,
            attack_noise: 0.0,
            attack_noise_seconds: 0.02,
            detune_cents: 0.0,
            stereo_spread: 0.0,
            keyboard_spread: 0.0,
            vibrato_cents: 0.0,
            vibrato_hz: 5.0,
            vibrato_delay_seconds: 0.3,
            humanise_cents: 1.5,
            reverb_send: 0.18,
            level: 1.0,
        }
    }
}

impl Patch {
    /// True when the note dies away by itself, as a struck or plucked string
    /// does. Such patches ignore how long the key is held.
    #[must_use]
    pub fn is_percussive(&self) -> bool {
        self.sustain <= f32::EPSILON
    }

    /// The decay time for one pitch, after key tracking. Middle C plays the
    /// patch's nominal decay; every octave up halves it `decay_key_track`
    /// times over.
    #[must_use]
    pub fn decay_seconds_at(&self, pitch: u8) -> f32 {
        if self.decay_key_track <= 0.0 {
            return self.decay_seconds;
        }
        let octaves_above_middle_c = (f32::from(pitch) - 60.0) / 12.0;
        let scale = 2.0_f32.powf(-octaves_above_middle_c * self.decay_key_track);
        (self.decay_seconds * scale).clamp(0.03, 30.0)
    }

    /// Gain of the `n`-th harmonic, one-based.
    #[must_use]
    pub fn harmonic_gain(&self, harmonic: usize) -> f32 {
        if harmonic == 0 {
            return 0.0;
        }
        if harmonic <= SEED_HARMONICS {
            return self.harmonics[harmonic - 1];
        }
        // The seed's last gain anchors the tail, so a patch that ends its seed
        // at zero has no tail at all.
        let anchor = self.harmonics[SEED_HARMONICS - 1];
        if anchor <= 0.0 {
            return 0.0;
        }
        let ratio = harmonic as f32 / SEED_HARMONICS as f32;
        let gain = anchor * ratio.powf(-self.rolloff);
        if harmonic % 2 == 0 {
            gain * (1.0 - self.odd_bias.clamp(0.0, 1.0))
        } else {
            gain
        }
    }
}

/// The patch for a GM program number.
///
/// Families set the character; the notable programs within each family are
/// then adjusted, because "Acoustic Grand" and "Honky-tonk" being audibly
/// different is most of the point of a bank.
#[must_use]
#[allow(clippy::match_same_arms, clippy::too_many_lines)]
pub fn patch_for_program(program: u8) -> Patch {
    let program = program.min(127);
    let base = match Family::of_program(program) {
        Family::Piano => Patch {
            harmonics: [1.0, 0.62, 0.44, 0.3, 0.21, 0.15, 0.1, 0.07],
            rolloff: 1.5,
            attack_seconds: 0.002,
            decay_seconds: 2.4,
            sustain: 0.0,
            release_seconds: 0.22,
            decay_key_track: 0.75,
            brightness: 9.0,
            filter_env: 2.4,
            filter_decay_seconds: 0.45,
            resonance: 0.25,
            velocity_brightness: 0.7,
            // The hammer felt striking the string, before the string speaks.
            attack_noise: 0.22,
            attack_noise_seconds: 0.012,
            // Three strings per note, never quite in tune with each other.
            detune_cents: 1.6,
            stereo_spread: 0.45,
            keyboard_spread: 0.35,
            humanise_cents: 1.2,
            reverb_send: 0.16,
            level: 1.17,
            ..Patch::default()
        },
        Family::ChromaticPercussion => Patch {
            harmonics: [1.0, 0.06, 0.42, 0.03, 0.16, 0.0, 0.05, 0.02],
            rolloff: 2.2,
            attack_seconds: 0.001,
            decay_seconds: 1.4,
            sustain: 0.0,
            release_seconds: 0.14,
            decay_key_track: 0.6,
            brightness: 14.0,
            filter_env: 2.5,
            filter_decay_seconds: 0.18,
            resonance: 0.3,
            velocity_brightness: 0.6,
            attack_noise: 0.3,
            attack_noise_seconds: 0.005,
            keyboard_spread: 0.3,
            reverb_send: 0.22,
            level: 0.92,
            ..Patch::default()
        },
        Family::Organ => Patch {
            // Drawbar-like: strong octaves and a fifth, no decay at all.
            harmonics: [1.0, 0.7, 0.5, 0.35, 0.0, 0.22, 0.0, 0.12],
            rolloff: 2.0,
            attack_seconds: 0.012,
            decay_seconds: 0.05,
            sustain: 0.95,
            release_seconds: 0.06,
            brightness: 10.0,
            // Electronic and near-static: barely any transient, level-flat.
            filter_env: 0.25,
            filter_decay_seconds: 0.1,
            resonance: 0.1,
            velocity_brightness: 0.15,
            // The drawbar key click, which every real organ has.
            attack_noise: 0.1,
            attack_noise_seconds: 0.006,
            detune_cents: 1.0,
            stereo_spread: 0.3,
            humanise_cents: 0.4,
            reverb_send: 0.3,
            level: 0.77,
            ..Patch::default()
        },
        Family::Guitar => Patch {
            harmonics: [1.0, 0.55, 0.4, 0.26, 0.18, 0.12, 0.08, 0.055],
            rolloff: 1.3,
            attack_seconds: 0.003,
            decay_seconds: 1.6,
            sustain: 0.0,
            release_seconds: 0.18,
            decay_key_track: 0.8,
            brightness: 7.0,
            filter_env: 2.0,
            filter_decay_seconds: 0.3,
            // The guitar's body, ringing around the filter corner.
            resonance: 0.45,
            velocity_brightness: 0.7,
            // The pick, which is half of what makes a guitar a guitar.
            attack_noise: 0.32,
            attack_noise_seconds: 0.009,
            detune_cents: 0.8,
            stereo_spread: 0.25,
            keyboard_spread: 0.15,
            humanise_cents: 2.5,
            reverb_send: 0.18,
            level: 2.06,
            ..Patch::default()
        },
        Family::Bass => Patch {
            harmonics: [1.0, 0.45, 0.24, 0.14, 0.08, 0.05, 0.03, 0.02],
            rolloff: 1.5,
            attack_seconds: 0.006,
            decay_seconds: 1.4,
            sustain: 0.25,
            release_seconds: 0.1,
            decay_key_track: 0.5,
            brightness: 5.0,
            filter_env: 1.6,
            filter_decay_seconds: 0.22,
            resonance: 0.4,
            velocity_brightness: 0.65,
            attack_noise: 0.22,
            attack_noise_seconds: 0.008,
            humanise_cents: 1.5,
            // A bass belongs at the front of the mix, dry and central.
            reverb_send: 0.05,
            level: 0.82,
            ..Patch::default()
        },
        Family::Strings => Patch {
            harmonics: [1.0, 0.72, 0.52, 0.4, 0.3, 0.23, 0.17, 0.13],
            // A bowed string is rich a long way up; this is most of its sound.
            rolloff: 0.95,
            attack_seconds: 0.09,
            decay_seconds: 0.3,
            sustain: 0.85,
            release_seconds: 0.32,
            brightness: 6.5,
            // Bowed: soft, gradual onset, so little filter transient — the life
            // comes from the vibrato and the bow instead.
            filter_env: 0.4,
            resonance: 0.35,
            velocity_brightness: 0.4,
            noise: 0.02,
            // Rosin on the string as the bow catches.
            attack_noise: 0.14,
            attack_noise_seconds: 0.06,
            detune_cents: 4.0,
            stereo_spread: 0.4,
            vibrato_cents: 8.0,
            vibrato_hz: 5.5,
            vibrato_delay_seconds: 0.35,
            humanise_cents: 3.0,
            reverb_send: 0.35,
            level: 0.99,
            ..Patch::default()
        },
        Family::Ensemble => Patch {
            harmonics: [1.0, 0.68, 0.48, 0.34, 0.25, 0.18, 0.13, 0.1],
            rolloff: 0.95,
            attack_seconds: 0.12,
            decay_seconds: 0.3,
            sustain: 0.88,
            release_seconds: 0.4,
            brightness: 6.0,
            filter_env: 0.3,
            resonance: 0.3,
            velocity_brightness: 0.35,
            noise: 0.025,
            attack_noise: 0.12,
            attack_noise_seconds: 0.08,
            detune_cents: 12.0,
            stereo_spread: 0.85,
            // Many players never quite agree on the vibrato, so it is slower and
            // shallower than a soloist's.
            vibrato_cents: 5.0,
            vibrato_hz: 4.8,
            vibrato_delay_seconds: 0.45,
            humanise_cents: 4.0,
            reverb_send: 0.4,
            level: 0.95,
            ..Patch::default()
        },
        Family::Brass => Patch {
            harmonics: [1.0, 0.85, 0.7, 0.56, 0.44, 0.34, 0.26, 0.2],
            // Brass is the brightest thing in the orchestra: a slow rolloff.
            rolloff: 0.8,
            attack_seconds: 0.045,
            decay_seconds: 0.25,
            sustain: 0.8,
            release_seconds: 0.14,
            brightness: 9.0,
            // Brass blares open when pushed: a strong onset and a big velocity
            // response are most of what makes it read as brass.
            filter_env: 1.3,
            filter_decay_seconds: 0.12,
            resonance: 0.4,
            velocity_brightness: 0.85,
            noise: 0.012,
            attack_noise: 0.16,
            attack_noise_seconds: 0.03,
            detune_cents: 3.0,
            stereo_spread: 0.3,
            vibrato_cents: 4.0,
            vibrato_hz: 5.5,
            vibrato_delay_seconds: 0.3,
            humanise_cents: 3.0,
            reverb_send: 0.28,
            level: 0.92,
            ..Patch::default()
        },
        Family::Reed => Patch {
            // Odd harmonics dominate, as in a cylindrical bore.
            harmonics: [1.0, 0.18, 0.62, 0.12, 0.34, 0.08, 0.18, 0.06],
            rolloff: 1.1,
            odd_bias: 0.65,
            attack_seconds: 0.03,
            decay_seconds: 0.2,
            sustain: 0.82,
            release_seconds: 0.12,
            brightness: 8.0,
            filter_env: 0.8,
            filter_decay_seconds: 0.15,
            resonance: 0.35,
            velocity_brightness: 0.6,
            noise: 0.035,
            attack_noise: 0.22,
            attack_noise_seconds: 0.035,
            vibrato_cents: 7.0,
            vibrato_hz: 5.0,
            vibrato_delay_seconds: 0.3,
            humanise_cents: 2.5,
            reverb_send: 0.24,
            level: 0.53,
            ..Patch::default()
        },
        Family::Pipe => Patch {
            harmonics: [1.0, 0.12, 0.06, 0.03, 0.015, 0.008, 0.004, 0.002],
            rolloff: 2.6,
            attack_seconds: 0.055,
            decay_seconds: 0.2,
            sustain: 0.9,
            release_seconds: 0.13,
            brightness: 12.0,
            filter_env: 0.5,
            resonance: 0.25,
            velocity_brightness: 0.4,
            noise: 0.09,
            // The chiff: a flute is breath before it is a pitch.
            attack_noise: 0.45,
            attack_noise_seconds: 0.05,
            vibrato_cents: 9.0,
            vibrato_hz: 5.0,
            vibrato_delay_seconds: 0.25,
            humanise_cents: 2.5,
            reverb_send: 0.3,
            level: 0.49,
            ..Patch::default()
        },
        Family::SynthLead => Patch {
            harmonics: [1.0, 0.5, 0.34, 0.26, 0.21, 0.17, 0.15, 0.13],
            rolloff: 1.0,
            attack_seconds: 0.008,
            decay_seconds: 0.3,
            sustain: 0.75,
            release_seconds: 0.1,
            brightness: 7.0,
            filter_env: 1.4,
            filter_decay_seconds: 0.25,
            resonance: 0.55,
            detune_cents: 7.0,
            stereo_spread: 0.5,
            vibrato_cents: 8.0,
            vibrato_hz: 6.0,
            vibrato_delay_seconds: 0.2,
            humanise_cents: 0.5,
            reverb_send: 0.2,
            level: 0.88,
            ..Patch::default()
        },
        Family::SynthPad => Patch {
            harmonics: [1.0, 0.55, 0.36, 0.26, 0.18, 0.13, 0.09, 0.07],
            rolloff: 1.2,
            attack_seconds: 0.35,
            decay_seconds: 0.6,
            sustain: 0.8,
            release_seconds: 0.7,
            brightness: 4.5,
            filter_env: 0.6,
            filter_decay_seconds: 0.9,
            resonance: 0.4,
            velocity_brightness: 0.3,
            detune_cents: 14.0,
            stereo_spread: 0.9,
            humanise_cents: 1.0,
            reverb_send: 0.45,
            level: 1.08,
            ..Patch::default()
        },
        Family::SynthEffects => Patch {
            harmonics: [1.0, 0.3, 0.5, 0.2, 0.35, 0.15, 0.25, 0.12],
            rolloff: 1.0,
            attack_seconds: 0.2,
            decay_seconds: 0.8,
            sustain: 0.5,
            release_seconds: 0.5,
            brightness: 6.0,
            resonance: 0.5,
            noise: 0.05,
            detune_cents: 18.0,
            stereo_spread: 0.8,
            reverb_send: 0.45,
            level: 1.37,
            ..Patch::default()
        },
        Family::Ethnic => Patch {
            harmonics: [1.0, 0.48, 0.32, 0.22, 0.15, 0.1, 0.07, 0.05],
            rolloff: 1.3,
            attack_seconds: 0.006,
            decay_seconds: 1.1,
            sustain: 0.15,
            release_seconds: 0.14,
            decay_key_track: 0.6,
            brightness: 8.0,
            resonance: 0.4,
            attack_noise: 0.3,
            attack_noise_seconds: 0.01,
            detune_cents: 2.5,
            stereo_spread: 0.3,
            humanise_cents: 3.0,
            reverb_send: 0.25,
            level: 1.64,
            ..Patch::default()
        },
        Family::Percussive => Patch {
            harmonics: [1.0, 0.1, 0.35, 0.05, 0.2, 0.0, 0.08, 0.03],
            rolloff: 2.0,
            attack_seconds: 0.001,
            decay_seconds: 0.7,
            sustain: 0.0,
            release_seconds: 0.1,
            decay_key_track: 0.5,
            brightness: 12.0,
            filter_env: 2.2,
            filter_decay_seconds: 0.15,
            attack_noise: 0.35,
            attack_noise_seconds: 0.006,
            keyboard_spread: 0.25,
            reverb_send: 0.22,
            level: 1.32,
            ..Patch::default()
        },
        Family::SoundEffects => Patch {
            harmonics: [1.0, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 0.03],
            rolloff: 1.2,
            attack_seconds: 0.05,
            decay_seconds: 0.6,
            sustain: 0.3,
            release_seconds: 0.3,
            brightness: 6.0,
            noise: 0.3,
            reverb_send: 0.3,
            level: 1.24,
            ..Patch::default()
        },
    };

    refine(program, base)
}

/// Per-program adjustments within a family.
#[allow(clippy::too_many_lines)]
fn refine(program: u8, mut patch: Patch) -> Patch {
    match program {
        // Bright and honky-tonk pianos, electric pianos, harpsichord, clav.
        1 => {
            patch.brightness = 13.0;
            patch.rolloff = 1.3;
        }
        3 => patch.detune_cents = 12.0,
        4 | 5 => {
            // A tine, not a string: a bell-like odd partial and no hammer felt.
            patch.harmonics = [1.0, 0.28, 0.5, 0.12, 0.2, 0.06, 0.09, 0.04];
            patch.rolloff = 1.8;
            patch.decay_seconds = 3.0;
            patch.decay_key_track = 0.5;
            patch.brightness = 10.0;
            patch.attack_noise = 0.12;
            patch.detune_cents = 0.0;
            patch.level *= 0.61;
        }
        6 => {
            // Harpsichord: plucked, bright, and the same at every velocity —
            // the instrument has no way to play louder.
            patch.harmonics = [1.0, 0.5, 0.7, 0.35, 0.4, 0.22, 0.18, 0.1];
            patch.rolloff = 1.0;
            patch.decay_seconds = 1.2;
            patch.velocity_brightness = 0.05;
            patch.level *= 1.89;
            patch.attack_noise = 0.4;
            patch.attack_noise_seconds = 0.006;
        }
        7 => {
            patch.brightness = 15.0;
            patch.rolloff = 1.0;
        }
        // Celesta through tubular bells: longer, purer rings.
        8 | 14 => {
            patch.decay_seconds = 3.4;
            patch.decay_key_track = 0.4;
            patch.level *= 0.71;
        }
        9 | 11 => {
            patch.decay_seconds = 0.7;
            patch.level *= 1.47;
        }
        // Percussive organ and rock organ.
        17 => {
            patch.attack_seconds = 0.002;
            patch.attack_noise = 0.25;
        }
        18 => {
            patch.harmonics = [1.0, 0.62, 0.55, 0.3, 0.24, 0.3, 0.12, 0.22];
            patch.rolloff = 1.2;
        }
        // Church organ fills the room it is built into.
        19 => patch.reverb_send = 0.55,
        // Reed and pipe organs breathe a little.
        20 | 21 => patch.noise = 0.03,
        // Accordion and harmonica.
        22 | 23 => {
            patch.harmonics = [1.0, 0.6, 0.45, 0.3, 0.24, 0.17, 0.12, 0.08];
            patch.rolloff = 1.2;
            patch.detune_cents = 9.0;
            patch.stereo_spread = 0.35;
        }
        // Nylon and steel acoustic guitars.
        24 => {
            patch.brightness = 5.5;
            patch.attack_noise = 0.2;
        }
        25 => {
            patch.brightness = 9.0;
            patch.rolloff = 1.15;
            patch.attack_noise = 0.4;
        }
        // Overdriven and distorted guitar: dense harmonics, long sustain, and
        // a compressed onset — distortion flattens the pick attack.
        29 | 30 => {
            patch.harmonics = [1.0, 0.8, 0.72, 0.62, 0.54, 0.47, 0.4, 0.35];
            patch.rolloff = 0.45;
            patch.sustain = 0.55;
            patch.decay_seconds = 0.7;
            patch.decay_key_track = 0.3;
            patch.attack_noise = 0.12;
            patch.resonance = 0.5;
            patch.velocity_brightness = 0.35;
        }
        // Acoustic and fingered bass are darker than picked or synth bass.
        32 | 33 => {
            patch.brightness = 4.0;
            patch.attack_noise = 0.15;
        }
        34 => patch.attack_noise = 0.35,
        // Slap bass is all attack.
        36 | 37 => {
            patch.brightness = 8.0;
            patch.attack_noise = 0.5;
            patch.filter_env = 2.6;
            patch.resonance = 0.7;
        }
        38 | 39 => {
            patch.harmonics = [1.0, 0.6, 0.44, 0.32, 0.22, 0.15, 0.1, 0.07];
            patch.rolloff = 1.2;
            patch.brightness = 6.5;
            patch.attack_noise = 0.0;
        }
        // Tremolo strings: the bow reverses several times a second.
        44 => {
            patch.vibrato_cents = 3.0;
            patch.attack_noise = 0.3;
            patch.attack_noise_seconds = 0.02;
        }
        // Pizzicato strings and harp are plucked, not bowed: a struck
        // transient, and none of the family's vibrato.
        45 | 46 => {
            patch.attack_seconds = 0.002;
            patch.sustain = 0.0;
            patch.decay_seconds = 1.0;
            patch.decay_key_track = 0.7;
            patch.noise = 0.0;
            patch.attack_noise = 0.3;
            patch.attack_noise_seconds = 0.008;
            patch.filter_env = 1.8;
            patch.filter_decay_seconds = 0.25;
            patch.vibrato_cents = 0.0;
            patch.level *= 2.38;
        }
        // Timpani.
        47 => {
            patch.attack_seconds = 0.001;
            patch.sustain = 0.0;
            patch.decay_seconds = 1.8;
            patch.decay_key_track = 0.0;
            patch.brightness = 4.0;
            patch.filter_env = 1.2;
            patch.attack_noise = 0.4;
            patch.attack_noise_seconds = 0.01;
            patch.vibrato_cents = 0.0;
            patch.level *= 1.76;
            patch.reverb_send = 0.3;
        }
        // Choir and voice: formant-ish, breathy, slow, with a human vibrato.
        52..=54 => {
            patch.harmonics = [1.0, 0.5, 0.62, 0.28, 0.18, 0.11, 0.06, 0.03];
            patch.rolloff = 1.8;
            patch.noise = 0.05;
            patch.attack_seconds = 0.14;
            patch.attack_noise = 0.1;
            patch.attack_noise_seconds = 0.09;
            patch.vibrato_cents = 6.0;
            patch.vibrato_hz = 4.5;
            patch.vibrato_delay_seconds = 0.4;
            patch.reverb_send = 0.45;
        }
        // An orchestra hit is one stab, not a held chord.
        55 => {
            patch.attack_seconds = 0.004;
            patch.sustain = 0.0;
            patch.decay_seconds = 0.5;
            patch.attack_noise = 0.35;
            patch.vibrato_cents = 0.0;
            patch.level *= 2.48;
        }
        // Muted trumpet is thin and buzzy, with a tighter onset.
        59 => {
            patch.harmonics = [1.0, 0.4, 0.8, 0.3, 0.5, 0.22, 0.32, 0.16];
            patch.rolloff = 0.9;
            patch.brightness = 11.0;
            patch.filter_env = 0.9;
            patch.attack_seconds = 0.02;
        }
        // French horn is the mellowest of the brass.
        60 => {
            patch.rolloff = 1.4;
            patch.brightness = 6.5;
            patch.attack_seconds = 0.07;
            patch.reverb_send = 0.4;
        }
        61 => patch.stereo_spread = 0.6,
        62 | 63 => {
            patch.detune_cents = 10.0;
            patch.stereo_spread = 0.5;
        }
        // Oboe and bassoon are double reeds: nasal, not hollow like a clarinet.
        68..=70 => patch.odd_bias = 0.15,
        // Clarinet, the cylindrical bore this family is shaped around.
        71 => patch.odd_bias = 0.8,
        // Flutes and whistles are nearly pure with breath on top.
        72..=79 => {
            patch.harmonics = [1.0, 0.09, 0.045, 0.02, 0.01, 0.0, 0.0, 0.0];
            patch.rolloff = 3.0;
            patch.noise = 0.12;
        }
        // Kalimba and steel drums ring metallically.
        108 | 114 => {
            patch.harmonics = [1.0, 0.1, 0.45, 0.06, 0.25, 0.04, 0.12, 0.05];
            patch.rolloff = 1.8;
            patch.decay_seconds = 1.2;
            patch.sustain = 0.0;
        }
        _ => {}
    }
    patch
}

/// A drum voice: how one percussion note is made.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrumVoice {
    /// Pitch the tonal part starts at, in Hz. Zero for pure noise.
    pub frequency: f32,
    /// Fraction of `frequency` the pitch falls to. A kick's pitch drop is what
    /// makes it a kick rather than a low beep.
    pub pitch_drop: f32,
    /// How long the pitch takes to fall. Far shorter than the decay: a kick's
    /// pitch is down within 50 ms while the note rings for 300, and stretching
    /// the drop over the whole decay is what turns a kick into a slide whistle.
    pub pitch_drop_seconds: f32,
    /// Second tonal partial, as a ratio of `frequency`. Zero for none. A snare
    /// head has two modes, a tom has an overtone above its fundamental.
    pub partial_ratio: f32,
    pub partial_level: f32,
    /// How long the tone rings, when that differs from the noise. A snare's
    /// shell thuds and stops while the wires underneath rattle on.
    pub tone_decay_seconds: f32,
    pub noise: f32,
    pub decay_seconds: f32,
    /// Low-pass cutoff for the noise, in Hz. Separates a snare from a cymbal.
    pub noise_cutoff: f32,
    /// High-pass corner for the noise, in Hz. Without it a hi-hat is white
    /// noise with a low-pass on it, which sounds like a hiss and not a cymbal.
    pub noise_highpass: f32,
    /// A short bright click at the very onset: the beater hitting the head.
    pub click: f32,
    /// Where the drum sits in the kit, `-1` left to `1` right, as heard from
    /// behind the kit.
    pub pan: f32,
    pub level: f32,
}

impl Default for DrumVoice {
    fn default() -> Self {
        Self {
            frequency: 200.0,
            pitch_drop: 1.0,
            pitch_drop_seconds: 0.04,
            partial_ratio: 0.0,
            partial_level: 0.0,
            tone_decay_seconds: 0.0,
            noise: 0.5,
            decay_seconds: 0.3,
            noise_cutoff: 6_000.0,
            noise_highpass: 0.0,
            click: 0.15,
            pan: 0.0,
            level: 1.0,
        }
    }
}

impl DrumVoice {
    /// How long the tone rings; falls back to the voice's overall decay.
    #[must_use]
    pub fn tone_decay(&self) -> f32 {
        if self.tone_decay_seconds > 0.0 {
            self.tone_decay_seconds
        } else {
            self.decay_seconds
        }
    }
}

/// The GM percussion voice for a channel-10 note number.
///
/// Notes outside the standard 35–81 range fall back to a neutral hit rather
/// than going silent, so an unusual transcription still plays.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn drum_voice(note: u8) -> DrumVoice {
    match note {
        // Kicks: a low tone whose pitch collapses in the first few tens of
        // milliseconds, over a click of beater.
        35 | 36 => DrumVoice {
            frequency: 115.0,
            pitch_drop: 0.42,
            pitch_drop_seconds: 0.045,
            noise: 0.05,
            decay_seconds: 0.45,
            noise_cutoff: 1_800.0,
            click: 0.4,
            level: 1.25,
            ..DrumVoice::default()
        },
        // Sticks and rim.
        37 | 39 => DrumVoice {
            frequency: 420.0,
            pitch_drop: 0.8,
            pitch_drop_seconds: 0.01,
            noise: 0.75,
            decay_seconds: 0.08,
            noise_cutoff: 9_000.0,
            noise_highpass: 900.0,
            click: 0.5,
            pan: -0.15,
            level: 0.7,
            ..DrumVoice::default()
        },
        // Snares: two head modes over the wires, and the wires outlast the head.
        38 | 40 => DrumVoice {
            frequency: 185.0,
            pitch_drop: 0.8,
            pitch_drop_seconds: 0.03,
            partial_ratio: 1.6,
            partial_level: 0.6,
            tone_decay_seconds: 0.14,
            noise: 0.72,
            decay_seconds: 0.3,
            noise_cutoff: 8_500.0,
            noise_highpass: 500.0,
            click: 0.3,
            pan: -0.1,
            level: 1.0,
        },
        // Toms, low to high, laid out left to right across the kit.
        41 | 43 => DrumVoice {
            frequency: 90.0,
            pitch_drop: 0.62,
            pitch_drop_seconds: 0.09,
            partial_ratio: 1.5,
            partial_level: 0.3,
            noise: 0.12,
            decay_seconds: 0.65,
            noise_cutoff: 3_000.0,
            click: 0.25,
            pan: 0.35,
            ..DrumVoice::default()
        },
        45 | 47 => DrumVoice {
            frequency: 130.0,
            pitch_drop: 0.62,
            pitch_drop_seconds: 0.08,
            partial_ratio: 1.5,
            partial_level: 0.3,
            noise: 0.12,
            decay_seconds: 0.55,
            noise_cutoff: 3_300.0,
            click: 0.25,
            pan: 0.1,
            ..DrumVoice::default()
        },
        48 | 50 => DrumVoice {
            frequency: 180.0,
            pitch_drop: 0.64,
            pitch_drop_seconds: 0.06,
            partial_ratio: 1.5,
            partial_level: 0.3,
            noise: 0.12,
            decay_seconds: 0.45,
            noise_cutoff: 3_800.0,
            click: 0.25,
            pan: -0.2,
            ..DrumVoice::default()
        },
        // Closed hi-hat: noise with everything below 6 kHz taken out, which is
        // what makes it a cymbal rather than a burst of hiss.
        42 | 44 => DrumVoice {
            frequency: 0.0,
            noise: 1.0,
            decay_seconds: 0.075,
            noise_cutoff: 16_000.0,
            noise_highpass: 6_500.0,
            click: 0.0,
            pan: 0.3,
            level: 0.55,
            ..DrumVoice::default()
        },
        // Open hi-hat.
        46 => DrumVoice {
            frequency: 0.0,
            noise: 1.0,
            decay_seconds: 0.45,
            noise_cutoff: 15_000.0,
            noise_highpass: 5_500.0,
            click: 0.0,
            pan: 0.3,
            level: 0.5,
            ..DrumVoice::default()
        },
        // Crashes and splash: wide, long, and above the kit.
        49 | 52 | 55 | 57 => DrumVoice {
            frequency: 0.0,
            noise: 1.0,
            decay_seconds: 2.2,
            noise_cutoff: 14_000.0,
            noise_highpass: 3_000.0,
            click: 0.0,
            pan: -0.45,
            level: 0.6,
            ..DrumVoice::default()
        },
        // Rides: a struck bell over a wash.
        51 | 53 | 59 => DrumVoice {
            frequency: 640.0,
            partial_ratio: 2.4,
            partial_level: 0.5,
            tone_decay_seconds: 0.4,
            noise: 0.5,
            decay_seconds: 1.4,
            noise_cutoff: 14_000.0,
            noise_highpass: 3_500.0,
            click: 0.35,
            pan: 0.45,
            level: 0.55,
            ..DrumVoice::default()
        },
        // Hand percussion: bongos, congas, timbales.
        60..=66 => DrumVoice {
            frequency: 260.0 + f32::from(note - 60) * 22.0,
            pitch_drop: 0.75,
            pitch_drop_seconds: 0.03,
            noise: 0.22,
            decay_seconds: 0.28,
            noise_cutoff: 5_500.0,
            noise_highpass: 300.0,
            click: 0.3,
            pan: -0.3,
            level: 0.8,
            ..DrumVoice::default()
        },
        // Shakers, cabasa, maracas, guiro.
        69..=74 => DrumVoice {
            frequency: 0.0,
            noise: 1.0,
            decay_seconds: 0.11,
            noise_cutoff: 16_000.0,
            noise_highpass: 4_000.0,
            click: 0.0,
            pan: 0.5,
            level: 0.45,
            ..DrumVoice::default()
        },
        // Woodblocks, claves, cuica, triangle.
        75..=81 => DrumVoice {
            frequency: 1_150.0,
            pitch_drop: 0.95,
            pitch_drop_seconds: 0.01,
            partial_ratio: 2.7,
            partial_level: 0.35,
            noise: 0.15,
            decay_seconds: 0.2,
            noise_cutoff: 14_000.0,
            noise_highpass: 800.0,
            click: 0.3,
            pan: -0.5,
            level: 0.6,
            ..DrumVoice::default()
        },
        _ => DrumVoice::default(),
    }
}

/// One band-limited cycle of a patch's waveform.
///
/// The length is a power of two so wrapping is a mask rather than a remainder,
/// and is chosen from how many harmonics the table actually carries — the top
/// octaves need a few hundred points, not a few thousand.
pub struct Wavetable {
    samples: Vec<f32>,
    mask: usize,
}

impl Wavetable {
    /// Reads the table at a phase in cycles, `0.0` to `1.0`, interpolating
    /// linearly between the two nearest points.
    #[must_use]
    pub fn sample(&self, phase: f32) -> f32 {
        let scaled = phase * self.samples.len() as f32;
        let lower = scaled as usize & self.mask;
        let upper = (lower + 1) & self.mask;
        let fraction = scaled - scaled.floor();
        let low = self.samples[lower];
        low + (self.samples[upper] - low) * fraction
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// The mip level a pitch reads from: one per octave.
#[must_use]
pub fn mip_level(pitch: u8) -> usize {
    (usize::from(pitch) / 12).min(MIP_LEVELS - 1)
}

/// Wavetables for every GM program, built once and shared by every voice.
///
/// One bank for the whole engine rather than one per track: the tables run to
/// a few megabytes, and duplicating that per instrument track would buy
/// nothing.
pub struct GmBank {
    /// `PROGRAM_COUNT * MIP_LEVELS` tables, program-major.
    tables: Vec<Wavetable>,
    /// One cycle of a sine, which is what the drum kit's tonal parts are.
    sine: Wavetable,
    patches: Vec<Patch>,
    sample_rate: SampleRate,
}

/// Index of the table a program plays at a pitch, for [`GmBank::table_at`].
///
/// A voice resolves this once when it starts, so the per-sample path is an
/// index rather than two clamps and a division.
#[must_use]
pub fn table_index(program: u8, pitch: u8) -> usize {
    usize::from(program.min(127)) * MIP_LEVELS + mip_level(pitch)
}

impl GmBank {
    /// Builds every table for one sample rate.
    ///
    /// Band limiting depends on the rate, so a bank belongs to the stream it
    /// was built for. This allocates and does real work; call it before the
    /// audio stream opens, never from the callback.
    #[must_use]
    pub fn new(sample_rate: SampleRate) -> Self {
        let patches: Vec<Patch> = (0..PROGRAM_COUNT)
            .map(|program| patch_for_program(program as u8))
            .collect();
        let sine = sine_table();
        let rate = sample_rate.get().max(1) as f32;
        let mut tables = Vec::with_capacity(PROGRAM_COUNT * MIP_LEVELS);
        for patch in &patches {
            for level in 0..MIP_LEVELS {
                tables.push(build_table(patch, harmonic_limit(level, rate), &sine));
            }
        }
        let mut fundamental = [0.0; SEED_HARMONICS];
        fundamental[0] = 1.0;
        let sine_patch = Patch {
            harmonics: fundamental,
            ..Patch::default()
        };
        Self {
            sine: build_table(&sine_patch, 1, &sine),
            tables,
            patches,
            sample_rate,
        }
    }

    #[must_use]
    pub const fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }

    #[must_use]
    pub fn patch(&self, program: u8) -> &Patch {
        &self.patches[usize::from(program.min(127))]
    }

    /// The band-limited table for one program at one pitch.
    #[must_use]
    pub fn table(&self, program: u8, pitch: u8) -> &Wavetable {
        self.table_at(table_index(program, pitch))
    }

    /// The table an index from [`table_index`] refers to. Out-of-range indices
    /// are clamped: a voice must never be able to panic the audio thread.
    #[must_use]
    pub fn table_at(&self, index: usize) -> &Wavetable {
        &self.tables[index.min(self.tables.len() - 1)]
    }

    /// A plain sine, which is what a kick drum's body and a snare's head modes
    /// are made of.
    #[must_use]
    pub const fn sine(&self) -> &Wavetable {
        &self.sine
    }
}

/// Harmonics that fit under Nyquist for the top note of one octave band.
///
/// The band's top note sets the limit, so every note in the band is safe; the
/// lowest note of a band gives up at most an octave of harmonics it could have
/// carried, which is the ordinary cost of mip-mapping by octave.
fn harmonic_limit(level: usize, sample_rate: f32) -> usize {
    let top_pitch = (level * 12 + 11).min(127) as u8;
    let top_frequency = midi_to_frequency(top_pitch).max(1.0);
    let limit = (sample_rate * 0.45 / top_frequency) as usize;
    limit.clamp(1, MAX_HARMONICS)
}

/// Points to give a table carrying `harmonics` partials.
fn table_size(harmonics: usize) -> usize {
    (harmonics * 16)
        .next_power_of_two()
        .clamp(MIN_TABLE_SIZE, MAX_TABLE_SIZE)
}

/// One cycle of a sine, used to build every other table without paying for a
/// `sin` call per point per harmonic.
fn sine_table() -> Vec<f32> {
    (0..MAX_TABLE_SIZE)
        .map(|index| (index as f32 / MAX_TABLE_SIZE as f32 * std::f32::consts::TAU).sin())
        .collect()
}

/// Renders one cycle of a patch's spectrum, truncated to `harmonics`.
///
/// `cycle` is one cycle of a sine at [`MAX_TABLE_SIZE`] points, read at a
/// stride rather than called: a `sin` per harmonic per point would make
/// building the bank take seconds.
fn build_table(patch: &Patch, harmonics: usize, cycle: &[f32]) -> Wavetable {
    let size = table_size(harmonics);
    let stride = MAX_TABLE_SIZE / size;
    let mask = MAX_TABLE_SIZE - 1;
    let gains: Vec<f32> = (1..=harmonics).map(|n| patch.harmonic_gain(n)).collect();

    let mut samples = vec![0.0_f32; size];
    for (index, slot) in samples.iter_mut().enumerate() {
        let step = index * stride;
        *slot = gains
            .iter()
            .enumerate()
            .map(|(harmonic, gain)| cycle[(step * (harmonic + 1)) & mask] * gain)
            .sum();
    }

    // Match every table's RMS so the keyboard stays even, but never let a rich
    // spectrum's peak run past unity.
    let sum_of_squares: f32 = samples.iter().map(|value| value * value).sum();
    let rms = (sum_of_squares / size as f32).sqrt();
    let peak = samples
        .iter()
        .fold(0.0_f32, |peak, value| peak.max(value.abs()));
    let scale = if rms > 0.0 && peak > 0.0 {
        (TARGET_RMS / rms).min(1.0 / peak)
    } else {
        0.0
    };
    for value in &mut samples {
        *value *= scale;
    }

    Wavetable {
        samples,
        mask: size - 1,
    }
}

/// The GM program name, for display.
// One line per General MIDI program; the table is the function.
#[allow(clippy::too_many_lines)]
#[must_use]
pub fn program_name(program: u8) -> &'static str {
    const NAMES: [&str; PROGRAM_COUNT] = [
        "Acoustic Grand Piano",
        "Bright Acoustic Piano",
        "Electric Grand Piano",
        "Honky-tonk Piano",
        "Electric Piano 1",
        "Electric Piano 2",
        "Harpsichord",
        "Clavinet",
        "Celesta",
        "Glockenspiel",
        "Music Box",
        "Vibraphone",
        "Marimba",
        "Xylophone",
        "Tubular Bells",
        "Dulcimer",
        "Drawbar Organ",
        "Percussive Organ",
        "Rock Organ",
        "Church Organ",
        "Reed Organ",
        "Accordion",
        "Harmonica",
        "Tango Accordion",
        "Acoustic Guitar (nylon)",
        "Acoustic Guitar (steel)",
        "Electric Guitar (jazz)",
        "Electric Guitar (clean)",
        "Electric Guitar (muted)",
        "Overdriven Guitar",
        "Distortion Guitar",
        "Guitar Harmonics",
        "Acoustic Bass",
        "Electric Bass (finger)",
        "Electric Bass (pick)",
        "Fretless Bass",
        "Slap Bass 1",
        "Slap Bass 2",
        "Synth Bass 1",
        "Synth Bass 2",
        "Violin",
        "Viola",
        "Cello",
        "Contrabass",
        "Tremolo Strings",
        "Pizzicato Strings",
        "Orchestral Harp",
        "Timpani",
        "String Ensemble 1",
        "String Ensemble 2",
        "Synth Strings 1",
        "Synth Strings 2",
        "Choir Aahs",
        "Voice Oohs",
        "Synth Voice",
        "Orchestra Hit",
        "Trumpet",
        "Trombone",
        "Tuba",
        "Muted Trumpet",
        "French Horn",
        "Brass Section",
        "Synth Brass 1",
        "Synth Brass 2",
        "Soprano Sax",
        "Alto Sax",
        "Tenor Sax",
        "Baritone Sax",
        "Oboe",
        "English Horn",
        "Bassoon",
        "Clarinet",
        "Piccolo",
        "Flute",
        "Recorder",
        "Pan Flute",
        "Blown Bottle",
        "Shakuhachi",
        "Whistle",
        "Ocarina",
        "Lead 1 (square)",
        "Lead 2 (sawtooth)",
        "Lead 3 (calliope)",
        "Lead 4 (chiff)",
        "Lead 5 (charang)",
        "Lead 6 (voice)",
        "Lead 7 (fifths)",
        "Lead 8 (bass + lead)",
        "Pad 1 (new age)",
        "Pad 2 (warm)",
        "Pad 3 (polysynth)",
        "Pad 4 (choir)",
        "Pad 5 (bowed)",
        "Pad 6 (metallic)",
        "Pad 7 (halo)",
        "Pad 8 (sweep)",
        "FX 1 (rain)",
        "FX 2 (soundtrack)",
        "FX 3 (crystal)",
        "FX 4 (atmosphere)",
        "FX 5 (brightness)",
        "FX 6 (goblins)",
        "FX 7 (echoes)",
        "FX 8 (sci-fi)",
        "Sitar",
        "Banjo",
        "Shamisen",
        "Koto",
        "Kalimba",
        "Bag pipe",
        "Fiddle",
        "Shanai",
        "Tinkle Bell",
        "Agogo",
        "Steel Drums",
        "Woodblock",
        "Taiko Drum",
        "Melodic Tom",
        "Synth Drum",
        "Reverse Cymbal",
        "Guitar Fret Noise",
        "Breath Noise",
        "Seashore",
        "Bird Tweet",
        "Telephone Ring",
        "Helicopter",
        "Applause",
        "Gunshot",
    ];
    NAMES[usize::from(program.min(127))]
}

#[cfg(test)]
mod tests {
    // Comparing a patch's vibrato depth to exactly zero is the assertion:
    // a struck instrument must carry no vibrato at all, not merely little.
    #![allow(clippy::float_cmp)]
    use super::*;

    #[test]
    fn families_follow_the_general_midi_grouping() {
        assert_eq!(Family::of_program(0), Family::Piano);
        assert_eq!(Family::of_program(24), Family::Guitar);
        assert_eq!(Family::of_program(33), Family::Bass);
        assert_eq!(Family::of_program(48), Family::Ensemble);
        assert_eq!(Family::of_program(127), Family::SoundEffects);
    }

    #[test]
    fn every_program_has_a_usable_patch() {
        for program in 0..=127_u8 {
            let patch = patch_for_program(program);
            assert!(
                patch.harmonics.iter().any(|gain| *gain > 0.0),
                "program {program} is silent"
            );
            assert!(
                patch.attack_seconds > 0.0,
                "program {program} has no attack"
            );
            assert!(patch.decay_seconds > 0.0, "program {program} has no decay");
            assert!((0.0..=1.0).contains(&patch.sustain));
            assert!(patch.release_seconds > 0.0);
            assert!(patch.brightness > 0.0);
            assert!(patch.harmonics.iter().all(|gain| gain.is_finite()));
        }
    }

    #[test]
    fn every_program_has_sane_expression_parameters() {
        for program in 0..=127_u8 {
            let patch = patch_for_program(program);
            assert!(patch.filter_env >= 0.0 && patch.filter_env.is_finite());
            assert!(patch.filter_decay_seconds > 0.0);
            assert!((0.0..=1.0).contains(&patch.velocity_brightness));
            assert!((0.0..=1.0).contains(&patch.resonance), "program {program}");
            assert!(
                (0.0..=1.0).contains(&patch.attack_noise),
                "program {program}"
            );
            assert!(patch.attack_noise_seconds > 0.0);
            assert!((0.0..=1.0).contains(&patch.stereo_spread));
            assert!((0.0..=1.0).contains(&patch.keyboard_spread));
            assert!(
                (0.0..=1.0).contains(&patch.reverb_send),
                "program {program}"
            );
            assert!(patch.rolloff > 0.0, "program {program} has no rolloff");
            assert!((0.0..=1.0).contains(&patch.odd_bias));
            assert!(patch.humanise_cents >= 0.0);
            assert!(patch.vibrato_cents >= 0.0 && patch.vibrato_cents.is_finite());
            assert!(
                patch.vibrato_hz > 0.0,
                "program {program} has a zero vibrato rate"
            );
            assert!(patch.vibrato_delay_seconds > 0.0);
        }
    }

    #[test]
    fn bowed_and_blown_instruments_waver_while_struck_ones_do_not() {
        // Strings, a flute and a trumpet carry vibrato; a piano, a plucked
        // string and a drum-like mallet do not.
        for program in [40_u8, 48, 56, 73] {
            assert!(
                patch_for_program(program).vibrato_cents > 0.0,
                "{} should have vibrato",
                program_name(program)
            );
        }
        for program in [0_u8, 24, 12, 45] {
            assert_eq!(
                patch_for_program(program).vibrato_cents,
                0.0,
                "{} should not waver",
                program_name(program)
            );
        }
    }

    #[test]
    fn plucked_and_struck_instruments_do_not_sustain() {
        // A piano, a guitar and a marimba must die away on their own.
        for program in [0_u8, 24, 12, 45] {
            assert!(
                patch_for_program(program).is_percussive(),
                "{} should decay by itself",
                program_name(program)
            );
        }
        // An organ, strings and a flute hold as long as the key is down.
        for program in [16_u8, 48, 73] {
            assert!(
                !patch_for_program(program).is_percussive(),
                "{} should sustain",
                program_name(program)
            );
        }
    }

    #[test]
    fn a_struck_string_rings_far_longer_low_than_high() {
        // The bottom of a piano sustains for many seconds; the top is almost a
        // click. One decay time for the whole keyboard is the giveaway.
        let piano = patch_for_program(0);
        let low = piano.decay_seconds_at(28);
        let high = piano.decay_seconds_at(96);
        assert!(
            low > high * 6.0,
            "low notes ({low}s) barely outlast high ones ({high}s)"
        );
        // An organ pipe does not care which key it is.
        let organ = patch_for_program(16);
        assert!(
            (organ.decay_seconds_at(28) - organ.decay_seconds_at(96)).abs() < 1e-6,
            "an organ should not key-track its decay"
        );
    }

    #[test]
    fn struck_and_plucked_instruments_have_an_onset_transient() {
        // The hammer, the pick and the breath. Without these the onset is a
        // pure tone fading up, which no acoustic instrument does.
        for program in [0_u8, 24, 25, 73] {
            assert!(
                patch_for_program(program).attack_noise > 0.05,
                "{} has no attack transient",
                program_name(program)
            );
        }
    }

    #[test]
    fn a_spectrum_reaches_far_past_its_named_harmonics() {
        // A bass note needs harmonics into the dozens or it is a muffled hum.
        let piano = patch_for_program(0);
        assert!(
            piano.harmonic_gain(24) > 0.001,
            "the piano's spectrum stops short"
        );
        assert!(
            piano.harmonic_gain(24) < piano.harmonic_gain(12),
            "the spectrum must fall as it rises"
        );
        // A flute is nearly a sine and must stay one.
        let flute = patch_for_program(73);
        assert!(flute.harmonic_gain(24) < 1e-4, "the flute grew a tail");
    }

    #[test]
    fn a_clarinet_keeps_its_odd_harmonics_and_loses_its_even_ones() {
        let clarinet = patch_for_program(71);
        assert!(
            clarinet.harmonic_gain(11) > clarinet.harmonic_gain(12) * 2.0,
            "a cylindrical bore should thin its even partials"
        );
    }

    #[test]
    fn families_are_audibly_different_from_each_other() {
        // Compare harmonic recipes: a piano, an organ and a flute must not
        // collapse to the same table, or the bank is decoration.
        let piano = patch_for_program(0).harmonics;
        let organ = patch_for_program(16).harmonics;
        let flute = patch_for_program(73).harmonics;
        let distance = |left: [f32; SEED_HARMONICS], right: [f32; SEED_HARMONICS]| -> f32 {
            left.iter()
                .zip(right.iter())
                .map(|(a, b)| (a - b).abs())
                .sum()
        };
        assert!(distance(piano, organ) > 0.5);
        assert!(distance(piano, flute) > 0.5);
        assert!(distance(organ, flute) > 0.5);
    }

    #[test]
    fn the_bank_builds_a_sane_table_for_every_program_and_octave() {
        let bank = GmBank::new(SampleRate::DEFAULT);
        for program in 0..=127_u8 {
            for pitch in (0..=127_u8).step_by(12) {
                let table = bank.table(program, pitch);
                assert!(table.len().is_power_of_two(), "table length must mask");
                let peak = table
                    .samples
                    .iter()
                    .fold(0.0_f32, |peak, value| peak.max(value.abs()));
                assert!(
                    peak > 0.0 && peak <= 1.0 + 1e-5,
                    "program {program} at pitch {pitch} peaks at {peak}"
                );
                assert!(table.samples.iter().all(|value| value.is_finite()));
            }
        }
    }

    #[test]
    fn low_octaves_carry_far_more_harmonics_than_high_ones() {
        // This is the whole point of the mip levels: without them a bass note
        // is cut to the same few harmonics as a piccolo.
        let rate = 48_000.0;
        assert_eq!(harmonic_limit(2, rate), MAX_HARMONICS);
        let top = harmonic_limit(MIP_LEVELS - 1, rate);
        assert!(
            top < 4,
            "the top octave kept {top} harmonics and will alias"
        );
        for level in 1..MIP_LEVELS {
            assert!(
                harmonic_limit(level, rate) <= harmonic_limit(level - 1, rate),
                "level {level} carries more harmonics than the octave below it"
            );
        }
    }

    #[test]
    fn no_table_carries_a_harmonic_past_nyquist() {
        let rate = 48_000.0_f32;
        for pitch in 0..=127_u8 {
            let level = mip_level(pitch);
            let harmonics = harmonic_limit(level, rate);
            let top = midi_to_frequency(pitch) * harmonics as f32;
            assert!(
                top < rate * 0.5,
                "pitch {pitch} reaches {top} Hz against a {rate} Hz rate"
            );
        }
    }

    #[test]
    fn tables_hold_an_even_loudness_across_the_keyboard() {
        let bank = GmBank::new(SampleRate::DEFAULT);
        let rms = |table: &Wavetable| -> f32 {
            let sum: f32 = table.samples.iter().map(|value| value * value).sum();
            (sum / table.len() as f32).sqrt()
        };
        for program in [0_u8, 48, 56, 73] {
            let low = rms(bank.table(program, 36));
            let high = rms(bank.table(program, 96));
            assert!(
                (low / high) < 1.6 && (high / low) < 1.6,
                "{} jumps from {low} to {high} across the keyboard",
                program_name(program)
            );
        }
    }

    #[test]
    fn a_table_interpolates_and_wraps() {
        let bank = GmBank::new(SampleRate::DEFAULT);
        let table = bank.table(0, 60);
        // A phase of exactly 1.0 wraps to the start rather than reading past.
        assert!((table.sample(1.0) - table.sample(0.0)).abs() < 1e-6);
        for step in 0..1_000 {
            assert!(table.sample(step as f32 / 1_000.0).is_finite());
        }
    }

    #[test]
    fn out_of_range_programs_are_clamped_rather_than_panicking() {
        let bank = GmBank::new(SampleRate::DEFAULT);
        assert_eq!(bank.patch(255), bank.patch(127));
        assert_eq!(program_name(255), program_name(127));
        assert!(bank.table(255, 255).sample(0.5).is_finite());
    }

    #[test]
    fn drum_voices_cover_the_general_midi_kit() {
        for note in 35..=81_u8 {
            let voice = drum_voice(note);
            assert!(voice.decay_seconds > 0.0, "drum note {note} has no decay");
            assert!(voice.tone_decay() > 0.0);
            assert!(voice.pitch_drop_seconds > 0.0);
            assert!(voice.level > 0.0);
            assert!((-1.0..=1.0).contains(&voice.pan));
            assert!(
                voice.noise > 0.0 || voice.frequency > 0.0,
                "drum note {note} makes no sound"
            );
        }
    }

    #[test]
    fn a_kick_is_low_and_falls_in_pitch_while_a_hat_is_pure_noise() {
        let kick = drum_voice(36);
        assert!(kick.frequency < 150.0);
        assert!(kick.pitch_drop < 0.6, "a kick must sweep downwards");
        assert!(
            kick.pitch_drop_seconds < kick.decay_seconds * 0.3,
            "a kick's pitch must fall well before the note ends"
        );
        assert!(kick.noise < 0.2);

        let hat = drum_voice(42);
        assert!(hat.frequency.abs() < f32::EPSILON, "a hi-hat has no pitch");
        assert!(hat.noise > 0.9);
        assert!(hat.decay_seconds < 0.1, "a closed hat must be short");
        assert!(
            hat.noise_highpass > 4_000.0,
            "a hi-hat without a high-pass is just hiss"
        );

        let crash = drum_voice(49);
        assert!(crash.decay_seconds > hat.decay_seconds * 5.0);
    }

    #[test]
    fn a_snare_rattles_longer_than_its_head_rings() {
        let snare = drum_voice(38);
        assert!(snare.partial_ratio > 1.0, "a snare head has two modes");
        assert!(
            snare.tone_decay() < snare.decay_seconds,
            "the wires should outlast the shell"
        );
    }

    #[test]
    fn the_kit_is_spread_across_the_stereo_field() {
        // A kit heard entirely from one point is the sound of a drum machine.
        let hat = drum_voice(42);
        let crash = drum_voice(49);
        assert!(
            (hat.pan - crash.pan).abs() > 0.5,
            "the hats and the crash sit on top of each other"
        );
    }

    #[test]
    fn unknown_drum_notes_still_make_a_sound() {
        let voice = drum_voice(20);
        assert!(voice.decay_seconds > 0.0 && voice.level > 0.0);
    }

    #[test]
    fn program_names_match_general_midi() {
        assert_eq!(program_name(0), "Acoustic Grand Piano");
        assert_eq!(program_name(32), "Acoustic Bass");
        assert_eq!(program_name(56), "Trumpet");
        assert_eq!(program_name(73), "Flute");
        assert_eq!(program_name(127), "Gunshot");
    }
}
