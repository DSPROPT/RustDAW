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
//! Sampled instruments would sound better and cost a 140 MB dependency that
//! sessions would then break without. This bank is a few kilobytes of tables
//! and always works.

/// Harmonics per patch. Eight reaches the fourth octave above the fundamental,
/// past which a wavetable of this size would alias on high notes anyway.
pub const HARMONICS: usize = 8;
/// Points per wavetable cycle.
pub const TABLE_SIZE: usize = 2_048;
pub const PROGRAM_COUNT: usize = 128;

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
    /// Amplitude of each harmonic, fundamental first.
    pub harmonics: [f32; HARMONICS],
    pub attack_seconds: f32,
    pub decay_seconds: f32,
    /// Level held while the key is down. Zero means the note always decays
    /// away on its own, like a plucked or struck string.
    pub sustain: f32,
    pub release_seconds: f32,
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
    /// How much velocity brightens the tone, beyond making it louder. Zero
    /// means a soft note and a hard one differ only in level; higher values
    /// make a soft note as dark as it is quiet, the way a real instrument
    /// responds to how hard it is played.
    pub velocity_brightness: f32,
    /// Breath or bow noise mixed in.
    pub noise: f32,
    /// Detune between two stacked oscillators, in cents. Gives ensembles and
    /// pads their width; zero for anything that should sound like one player.
    pub detune_cents: f32,
    /// Pitch vibrato: depth in cents, rate in Hz, and how long the note waits
    /// before the vibrato swells in. Zero depth for anything struck or plucked;
    /// a delayed, gentle vibrato is what brings bowed and blown notes to life.
    pub vibrato_cents: f32,
    pub vibrato_hz: f32,
    pub vibrato_delay_seconds: f32,
    pub level: f32,
}

impl Default for Patch {
    fn default() -> Self {
        Self {
            harmonics: [1.0, 0.5, 0.25, 0.12, 0.06, 0.03, 0.0, 0.0],
            attack_seconds: 0.005,
            decay_seconds: 0.4,
            sustain: 0.6,
            release_seconds: 0.12,
            brightness: 8.0,
            filter_env: 1.2,
            filter_decay_seconds: 0.3,
            velocity_brightness: 0.5,
            noise: 0.0,
            detune_cents: 0.0,
            vibrato_cents: 0.0,
            vibrato_hz: 5.0,
            vibrato_delay_seconds: 0.3,
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
            harmonics: [1.0, 0.62, 0.31, 0.18, 0.09, 0.05, 0.02, 0.01],
            attack_seconds: 0.002,
            decay_seconds: 1.8,
            sustain: 0.0,
            release_seconds: 0.18,
            brightness: 9.0,
            filter_env: 2.2,
            filter_decay_seconds: 0.35,
            velocity_brightness: 0.7,
            ..Patch::default()
        },
        Family::ChromaticPercussion => Patch {
            harmonics: [1.0, 0.06, 0.42, 0.03, 0.16, 0.0, 0.05, 0.0],
            attack_seconds: 0.001,
            decay_seconds: 1.1,
            sustain: 0.0,
            release_seconds: 0.14,
            brightness: 14.0,
            filter_env: 2.5,
            filter_decay_seconds: 0.18,
            velocity_brightness: 0.6,
            ..Patch::default()
        },
        Family::Organ => Patch {
            // Drawbar-like: strong octaves and a fifth, no decay at all.
            harmonics: [1.0, 0.7, 0.5, 0.35, 0.0, 0.22, 0.0, 0.12],
            attack_seconds: 0.012,
            decay_seconds: 0.05,
            sustain: 0.95,
            release_seconds: 0.06,
            brightness: 10.0,
            // Electronic and near-static: barely any transient, level-flat.
            filter_env: 0.25,
            filter_decay_seconds: 0.1,
            velocity_brightness: 0.15,
            ..Patch::default()
        },
        Family::Guitar => Patch {
            harmonics: [1.0, 0.55, 0.38, 0.22, 0.14, 0.08, 0.04, 0.02],
            attack_seconds: 0.003,
            decay_seconds: 1.3,
            sustain: 0.0,
            release_seconds: 0.16,
            brightness: 7.0,
            filter_env: 2.0,
            filter_decay_seconds: 0.3,
            velocity_brightness: 0.7,
            ..Patch::default()
        },
        Family::Bass => Patch {
            harmonics: [1.0, 0.45, 0.2, 0.1, 0.05, 0.02, 0.0, 0.0],
            attack_seconds: 0.006,
            decay_seconds: 1.0,
            sustain: 0.25,
            release_seconds: 0.1,
            brightness: 5.0,
            filter_env: 1.6,
            filter_decay_seconds: 0.22,
            velocity_brightness: 0.65,
            ..Patch::default()
        },
        Family::Strings => Patch {
            harmonics: [1.0, 0.72, 0.5, 0.36, 0.26, 0.18, 0.12, 0.08],
            attack_seconds: 0.09,
            decay_seconds: 0.3,
            sustain: 0.85,
            release_seconds: 0.32,
            brightness: 6.5,
            // Bowed: soft, gradual onset, so little filter transient — the life
            // comes from the vibrato instead.
            filter_env: 0.4,
            velocity_brightness: 0.4,
            noise: 0.015,
            detune_cents: 4.0,
            vibrato_cents: 8.0,
            vibrato_hz: 5.5,
            vibrato_delay_seconds: 0.35,
            ..Patch::default()
        },
        Family::Ensemble => Patch {
            harmonics: [1.0, 0.68, 0.46, 0.3, 0.2, 0.13, 0.08, 0.05],
            attack_seconds: 0.12,
            decay_seconds: 0.3,
            sustain: 0.88,
            release_seconds: 0.4,
            brightness: 6.0,
            filter_env: 0.3,
            velocity_brightness: 0.35,
            noise: 0.02,
            detune_cents: 11.0,
            // Many players never quite agree on the vibrato, so it is slower and
            // shallower than a soloist's.
            vibrato_cents: 5.0,
            vibrato_hz: 4.8,
            vibrato_delay_seconds: 0.45,
            ..Patch::default()
        },
        Family::Brass => Patch {
            harmonics: [1.0, 0.85, 0.68, 0.52, 0.38, 0.26, 0.16, 0.09],
            attack_seconds: 0.045,
            decay_seconds: 0.25,
            sustain: 0.8,
            release_seconds: 0.14,
            brightness: 9.0,
            // Brass blares open when pushed: a strong onset and a big velocity
            // response are most of what makes it read as brass.
            filter_env: 1.3,
            filter_decay_seconds: 0.12,
            velocity_brightness: 0.85,
            noise: 0.01,
            detune_cents: 3.0,
            vibrato_cents: 4.0,
            vibrato_hz: 5.5,
            vibrato_delay_seconds: 0.3,
            ..Patch::default()
        },
        Family::Reed => Patch {
            // Odd harmonics dominate, as in a cylindrical bore.
            harmonics: [1.0, 0.18, 0.62, 0.12, 0.34, 0.08, 0.16, 0.05],
            attack_seconds: 0.03,
            decay_seconds: 0.2,
            sustain: 0.82,
            release_seconds: 0.12,
            brightness: 8.0,
            filter_env: 0.8,
            filter_decay_seconds: 0.15,
            velocity_brightness: 0.6,
            noise: 0.035,
            vibrato_cents: 7.0,
            vibrato_hz: 5.0,
            vibrato_delay_seconds: 0.3,
            ..Patch::default()
        },
        Family::Pipe => Patch {
            harmonics: [1.0, 0.12, 0.06, 0.03, 0.015, 0.0, 0.0, 0.0],
            attack_seconds: 0.055,
            decay_seconds: 0.2,
            sustain: 0.9,
            release_seconds: 0.13,
            brightness: 12.0,
            filter_env: 0.5,
            velocity_brightness: 0.4,
            noise: 0.09,
            vibrato_cents: 9.0,
            vibrato_hz: 5.0,
            vibrato_delay_seconds: 0.25,
            ..Patch::default()
        },
        Family::SynthLead => Patch {
            harmonics: [1.0, 0.5, 0.33, 0.25, 0.2, 0.16, 0.14, 0.12],
            attack_seconds: 0.008,
            decay_seconds: 0.3,
            sustain: 0.75,
            release_seconds: 0.1,
            brightness: 7.0,
            filter_env: 1.4,
            filter_decay_seconds: 0.25,
            detune_cents: 7.0,
            vibrato_cents: 8.0,
            vibrato_hz: 6.0,
            vibrato_delay_seconds: 0.2,
            ..Patch::default()
        },
        Family::SynthPad => Patch {
            harmonics: [1.0, 0.55, 0.34, 0.24, 0.16, 0.11, 0.07, 0.05],
            attack_seconds: 0.35,
            decay_seconds: 0.6,
            sustain: 0.8,
            release_seconds: 0.7,
            brightness: 4.5,
            filter_env: 0.6,
            filter_decay_seconds: 0.9,
            velocity_brightness: 0.3,
            detune_cents: 14.0,
            ..Patch::default()
        },
        Family::SynthEffects => Patch {
            harmonics: [1.0, 0.3, 0.5, 0.2, 0.35, 0.15, 0.25, 0.1],
            attack_seconds: 0.2,
            decay_seconds: 0.8,
            sustain: 0.5,
            release_seconds: 0.5,
            brightness: 6.0,
            noise: 0.05,
            detune_cents: 18.0,
            ..Patch::default()
        },
        Family::Ethnic => Patch {
            harmonics: [1.0, 0.48, 0.3, 0.2, 0.12, 0.07, 0.04, 0.02],
            attack_seconds: 0.006,
            decay_seconds: 0.9,
            sustain: 0.15,
            release_seconds: 0.14,
            brightness: 8.0,
            ..Patch::default()
        },
        Family::Percussive => Patch {
            harmonics: [1.0, 0.1, 0.35, 0.05, 0.2, 0.0, 0.08, 0.0],
            attack_seconds: 0.001,
            decay_seconds: 0.55,
            sustain: 0.0,
            release_seconds: 0.1,
            brightness: 12.0,
            ..Patch::default()
        },
        Family::SoundEffects => Patch {
            harmonics: [1.0, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 0.02],
            attack_seconds: 0.05,
            decay_seconds: 0.6,
            sustain: 0.3,
            release_seconds: 0.3,
            brightness: 6.0,
            noise: 0.3,
            ..Patch::default()
        },
    };

    refine(program, base)
}

/// Per-program adjustments within a family.
fn refine(program: u8, mut patch: Patch) -> Patch {
    match program {
        // Bright and honky-tonk pianos, electric pianos, harpsichord, clav.
        1 => patch.brightness = 13.0,
        3 => patch.detune_cents = 12.0,
        4 | 5 => {
            patch.harmonics = [1.0, 0.28, 0.5, 0.12, 0.2, 0.06, 0.08, 0.02];
            patch.decay_seconds = 2.4;
            patch.brightness = 10.0;
        }
        6 => {
            patch.harmonics = [1.0, 0.5, 0.7, 0.35, 0.4, 0.2, 0.16, 0.08];
            patch.decay_seconds = 1.0;
        }
        7 => patch.brightness = 15.0,
        // Celesta through tubular bells: longer, purer rings.
        8 | 14 => patch.decay_seconds = 2.6,
        9 | 11 => patch.decay_seconds = 0.55,
        // Percussive organ and rock organ.
        17 => patch.attack_seconds = 0.002,
        18 => patch.harmonics = [1.0, 0.62, 0.55, 0.3, 0.24, 0.3, 0.1, 0.2],
        // Reed and pipe organs breathe a little.
        20 | 21 => patch.noise = 0.03,
        // Accordion and harmonica.
        22 | 23 => {
            patch.harmonics = [1.0, 0.6, 0.45, 0.3, 0.24, 0.16, 0.1, 0.06];
            patch.detune_cents = 9.0;
        }
        // Nylon and steel acoustic guitars.
        24 => patch.brightness = 5.5,
        25 => patch.brightness = 9.0,
        // Overdriven and distorted guitar: dense harmonics, long sustain.
        29 | 30 => {
            patch.harmonics = [1.0, 0.8, 0.72, 0.6, 0.5, 0.42, 0.34, 0.28];
            patch.sustain = 0.55;
            patch.decay_seconds = 0.5;
        }
        // Acoustic and fingered bass are darker than picked or synth bass.
        32 | 33 => patch.brightness = 4.0,
        38 | 39 => {
            patch.harmonics = [1.0, 0.6, 0.42, 0.3, 0.2, 0.12, 0.07, 0.04];
            patch.brightness = 6.5;
        }
        // Pizzicato strings and harp are plucked, not bowed: a struck
        // transient, and none of the family's vibrato.
        45 | 46 => {
            patch.attack_seconds = 0.002;
            patch.sustain = 0.0;
            patch.decay_seconds = 0.8;
            patch.noise = 0.0;
            patch.filter_env = 1.8;
            patch.filter_decay_seconds = 0.25;
            patch.vibrato_cents = 0.0;
        }
        // Timpani.
        47 => {
            patch.attack_seconds = 0.001;
            patch.sustain = 0.0;
            patch.decay_seconds = 1.4;
            patch.brightness = 4.0;
            patch.filter_env = 1.2;
            patch.vibrato_cents = 0.0;
        }
        // Choir and voice: formant-ish, breathy, slow, with a human vibrato.
        52..=54 => {
            patch.harmonics = [1.0, 0.5, 0.62, 0.28, 0.18, 0.1, 0.05, 0.02];
            patch.noise = 0.05;
            patch.attack_seconds = 0.14;
            patch.vibrato_cents = 6.0;
            patch.vibrato_hz = 4.5;
            patch.vibrato_delay_seconds = 0.4;
        }
        // Muted trumpet is thin and buzzy, with a tighter onset.
        59 => {
            patch.harmonics = [1.0, 0.4, 0.8, 0.3, 0.5, 0.2, 0.3, 0.12];
            patch.brightness = 11.0;
            patch.filter_env = 0.9;
        }
        61..=63 => patch.detune_cents = 10.0,
        // Flutes and whistles are nearly pure with breath on top.
        73..=79 => {
            patch.harmonics = [1.0, 0.08, 0.04, 0.02, 0.0, 0.0, 0.0, 0.0];
            patch.noise = 0.12;
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
    /// Fraction of `frequency` the pitch falls to over the decay. A kick's
    /// pitch drop is what makes it a kick rather than a low beep.
    pub pitch_drop: f32,
    pub noise: f32,
    pub decay_seconds: f32,
    /// Low-pass cutoff for the noise, in Hz. Separates a hi-hat from a snare.
    pub noise_cutoff: f32,
    pub level: f32,
}

impl Default for DrumVoice {
    fn default() -> Self {
        Self {
            frequency: 200.0,
            pitch_drop: 1.0,
            noise: 0.5,
            decay_seconds: 0.2,
            noise_cutoff: 6_000.0,
            level: 1.0,
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
        // Kicks.
        35 | 36 => DrumVoice {
            frequency: 105.0,
            pitch_drop: 0.42,
            noise: 0.06,
            decay_seconds: 0.34,
            noise_cutoff: 1_200.0,
            level: 1.25,
        },
        // Sticks and rim.
        37 | 39 => DrumVoice {
            frequency: 420.0,
            pitch_drop: 0.8,
            noise: 0.8,
            decay_seconds: 0.06,
            noise_cutoff: 8_000.0,
            level: 0.7,
        },
        // Snares.
        38 | 40 => DrumVoice {
            frequency: 190.0,
            pitch_drop: 0.75,
            noise: 0.85,
            decay_seconds: 0.19,
            noise_cutoff: 7_500.0,
            level: 1.0,
        },
        // Toms, low to high.
        41 | 43 => DrumVoice {
            frequency: 90.0,
            pitch_drop: 0.6,
            noise: 0.1,
            decay_seconds: 0.42,
            noise_cutoff: 2_500.0,
            ..DrumVoice::default()
        },
        45 | 47 => DrumVoice {
            frequency: 130.0,
            pitch_drop: 0.6,
            noise: 0.1,
            decay_seconds: 0.36,
            noise_cutoff: 2_800.0,
            ..DrumVoice::default()
        },
        48 | 50 => DrumVoice {
            frequency: 180.0,
            pitch_drop: 0.62,
            noise: 0.1,
            decay_seconds: 0.3,
            noise_cutoff: 3_200.0,
            ..DrumVoice::default()
        },
        // Closed hi-hat.
        42 | 44 => DrumVoice {
            frequency: 0.0,
            pitch_drop: 1.0,
            noise: 1.0,
            decay_seconds: 0.05,
            noise_cutoff: 12_000.0,
            level: 0.6,
        },
        // Open hi-hat.
        46 => DrumVoice {
            frequency: 0.0,
            pitch_drop: 1.0,
            noise: 1.0,
            decay_seconds: 0.34,
            noise_cutoff: 11_000.0,
            level: 0.55,
        },
        // Crashes and splash.
        49 | 52 | 55 | 57 => DrumVoice {
            frequency: 0.0,
            pitch_drop: 1.0,
            noise: 1.0,
            decay_seconds: 1.1,
            noise_cutoff: 9_500.0,
            level: 0.65,
        },
        // Rides.
        51 | 53 | 59 => DrumVoice {
            frequency: 640.0,
            pitch_drop: 1.0,
            noise: 0.55,
            decay_seconds: 0.7,
            noise_cutoff: 10_000.0,
            level: 0.55,
        },
        // Hand percussion: bongos, congas, timbales.
        60..=66 => DrumVoice {
            frequency: 260.0 + f32::from(note - 60) * 22.0,
            pitch_drop: 0.72,
            noise: 0.22,
            decay_seconds: 0.16,
            noise_cutoff: 5_000.0,
            level: 0.8,
        },
        // Shakers, cabasa, maracas, guiro.
        69..=74 => DrumVoice {
            frequency: 0.0,
            pitch_drop: 1.0,
            noise: 1.0,
            decay_seconds: 0.09,
            noise_cutoff: 13_000.0,
            level: 0.5,
        },
        // Woodblocks, claves, cuica, triangle.
        75..=81 => DrumVoice {
            frequency: 1_150.0,
            pitch_drop: 0.95,
            noise: 0.2,
            decay_seconds: 0.13,
            noise_cutoff: 12_000.0,
            level: 0.6,
        },
        _ => DrumVoice::default(),
    }
}

/// Wavetables for every GM program, built once and shared by every voice.
///
/// One bank for the whole engine rather than one per track: 128 tables is a
/// megabyte, and duplicating that per instrument track would be a megabyte
/// each for no benefit.
pub struct GmBank {
    tables: Vec<[f32; TABLE_SIZE]>,
    patches: Vec<Patch>,
}

impl Default for GmBank {
    fn default() -> Self {
        Self::new()
    }
}

impl GmBank {
    #[must_use]
    pub fn new() -> Self {
        let patches: Vec<Patch> = (0..PROGRAM_COUNT)
            .map(|program| patch_for_program(program as u8))
            .collect();
        let tables = patches.iter().map(build_table).collect();
        Self { tables, patches }
    }

    #[must_use]
    pub fn patch(&self, program: u8) -> &Patch {
        &self.patches[usize::from(program.min(127))]
    }

    #[must_use]
    pub fn table(&self, program: u8) -> &[f32; TABLE_SIZE] {
        &self.tables[usize::from(program.min(127))]
    }
}

/// Renders one cycle of a patch's harmonic recipe, normalised to unit peak.
fn build_table(patch: &Patch) -> [f32; TABLE_SIZE] {
    let mut table = [0.0_f32; TABLE_SIZE];
    for (index, slot) in table.iter_mut().enumerate() {
        let phase = index as f32 / TABLE_SIZE as f32 * std::f32::consts::TAU;
        *slot = patch
            .harmonics
            .iter()
            .enumerate()
            .map(|(harmonic, gain)| (phase * (harmonic + 1) as f32).sin() * gain)
            .sum();
    }
    let peak = table.iter().fold(0.0_f32, |peak, value| peak.max(value.abs()));
    if peak > 0.0 {
        for value in &mut table {
            *value /= peak;
        }
    }
    table
}

/// The GM program name, for display.
#[must_use]
pub fn program_name(program: u8) -> &'static str {
    const NAMES: [&str; PROGRAM_COUNT] = [
        "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano", "Honky-tonk Piano",
        "Electric Piano 1", "Electric Piano 2", "Harpsichord", "Clavinet",
        "Celesta", "Glockenspiel", "Music Box", "Vibraphone",
        "Marimba", "Xylophone", "Tubular Bells", "Dulcimer",
        "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ",
        "Reed Organ", "Accordion", "Harmonica", "Tango Accordion",
        "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)", "Electric Guitar (jazz)",
        "Electric Guitar (clean)", "Electric Guitar (muted)", "Overdriven Guitar",
        "Distortion Guitar", "Guitar Harmonics",
        "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)", "Fretless Bass",
        "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2",
        "Violin", "Viola", "Cello", "Contrabass",
        "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani",
        "String Ensemble 1", "String Ensemble 2", "Synth Strings 1", "Synth Strings 2",
        "Choir Aahs", "Voice Oohs", "Synth Voice", "Orchestra Hit",
        "Trumpet", "Trombone", "Tuba", "Muted Trumpet",
        "French Horn", "Brass Section", "Synth Brass 1", "Synth Brass 2",
        "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax",
        "Oboe", "English Horn", "Bassoon", "Clarinet",
        "Piccolo", "Flute", "Recorder", "Pan Flute",
        "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina",
        "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)", "Lead 4 (chiff)",
        "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)", "Lead 8 (bass + lead)",
        "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)", "Pad 4 (choir)",
        "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)",
        "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)",
        "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)",
        "Sitar", "Banjo", "Shamisen", "Koto",
        "Kalimba", "Bag pipe", "Fiddle", "Shanai",
        "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock",
        "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal",
        "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet",
        "Telephone Ring", "Helicopter", "Applause", "Gunshot",
    ];
    NAMES[usize::from(program.min(127))]
}

#[cfg(test)]
mod tests {
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
            assert!(patch.attack_seconds > 0.0, "program {program} has no attack");
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
            assert!(patch.vibrato_cents >= 0.0 && patch.vibrato_cents.is_finite());
            assert!(patch.vibrato_hz > 0.0, "program {program} has a zero vibrato rate");
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
    fn families_are_audibly_different_from_each_other() {
        // Compare harmonic recipes: a piano, an organ and a flute must not
        // collapse to the same table, or the bank is decoration.
        let piano = patch_for_program(0).harmonics;
        let organ = patch_for_program(16).harmonics;
        let flute = patch_for_program(73).harmonics;
        let distance = |left: [f32; HARMONICS], right: [f32; HARMONICS]| -> f32 {
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
    fn the_bank_builds_normalised_tables_for_every_program() {
        let bank = GmBank::new();
        for program in 0..=127_u8 {
            let table = bank.table(program);
            let peak = table.iter().fold(0.0_f32, |peak, value| peak.max(value.abs()));
            assert!(
                (peak - 1.0).abs() < 1e-5,
                "program {program} peaks at {peak}, not 1.0"
            );
            assert!(table.iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn out_of_range_programs_are_clamped_rather_than_panicking() {
        let bank = GmBank::new();
        assert_eq!(bank.patch(255), bank.patch(127));
        assert_eq!(program_name(255), program_name(127));
    }

    #[test]
    fn drum_voices_cover_the_general_midi_kit() {
        for note in 35..=81_u8 {
            let voice = drum_voice(note);
            assert!(voice.decay_seconds > 0.0, "drum note {note} has no decay");
            assert!(voice.level > 0.0);
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
        assert!(kick.noise < 0.2);

        let hat = drum_voice(42);
        assert!(hat.frequency.abs() < f32::EPSILON, "a hi-hat has no pitch");
        assert!(hat.noise > 0.9);
        assert!(hat.decay_seconds < 0.1, "a closed hat must be short");

        let crash = drum_voice(49);
        assert!(crash.decay_seconds > hat.decay_seconds * 5.0);
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
