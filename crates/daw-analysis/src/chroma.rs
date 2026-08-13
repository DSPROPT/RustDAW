#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

//! Chromagrams: how much of each of the twelve pitch classes is sounding.
//!
//! Every chord decision rests on this, so it is worth doing carefully. Three
//! things matter more than the choice of transform:
//!
//! 1. **Harmonics lie.** A single C3 puts energy on C, G and E as well, which
//!    is a C major triad that nobody played. Energy is therefore divided down
//!    the harmonic series before it is binned.
//! 2. **Loudness is not importance.** A chorus is not more harmonic than a
//!    verse, so each frame is normalised before it is averaged.
//! 3. **Bass is not the chord.** The lowest register is collected separately so
//!    inversions can be named without the bass note dominating the triad.

use crate::fft::{hann_window, magnitude_spectrum};

pub const PITCH_CLASSES: usize = 12;
/// Analysis window. Long, because distinguishing adjacent semitones in the bass
/// needs frequency resolution far more than it needs timing: at 48 kHz this is
/// 340 ms and about 3 Hz per bin — fine enough that a semitone is wider than a
/// bin down to about 50 Hz, which is what keeps the low register from folding
/// neighbouring notes onto one bin. Chords are averaged over whole beats anyway,
/// so the timing that is given up costs nothing.
pub const WINDOW: usize = 16_384;
pub const HOP: usize = 2_048;

/// Lowest and highest frequencies considered, roughly C1 to C8.
const MIN_HZ: f32 = 32.7;
const MAX_HZ: f32 = 4_186.0;
/// Bass register, used only for the separate bass chroma.
const BASS_MAX_HZ: f32 = 261.6;
/// How many harmonics of each detected partial are discounted.
const HARMONICS: usize = 4;

/// One frame of harmonic content.
#[derive(Clone, Copy, Debug, Default)]
pub struct Chroma {
    /// Energy per pitch class, C first, normalised so the strongest is 1.
    pub pitches: [f32; PITCH_CLASSES],
    /// The same for the bass register alone.
    pub bass: [f32; PITCH_CLASSES],
    /// Raw level of the frame, before normalisation.
    pub energy: f32,
    /// How concentrated the spectrum is in a few pitch classes. Sustained
    /// chords score high; drums, applause and noise score low, which is what
    /// separates "no chord" from "a chord I cannot name".
    pub tonalness: f32,
}

impl Chroma {
    /// Combines frames by averaging, for collapsing a beat's worth into one.
    #[must_use]
    pub fn average(frames: &[Self]) -> Self {
        if frames.is_empty() {
            return Self::default();
        }
        let mut result = Self::default();
        for frame in frames {
            for index in 0..PITCH_CLASSES {
                result.pitches[index] += frame.pitches[index];
                result.bass[index] += frame.bass[index];
            }
            result.energy += frame.energy;
            result.tonalness += frame.tonalness;
        }
        let count = frames.len() as f32;
        for index in 0..PITCH_CLASSES {
            result.pitches[index] /= count;
            result.bass[index] /= count;
        }
        result.energy /= count;
        result.tonalness /= count;
        result.normalise();
        result
    }

    fn normalise(&mut self) {
        normalise_in_place(&mut self.pitches);
        normalise_in_place(&mut self.bass);
    }

    /// The strongest pitch class in the bass register, when there is one.
    #[must_use]
    pub fn bass_pitch_class(&self) -> Option<u8> {
        let (index, value) = self
            .bass
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))?;
        (*value > 0.55).then_some(index as u8)
    }
}

fn normalise_in_place(values: &mut [f32; PITCH_CLASSES]) {
    let peak = values.iter().fold(0.0_f32, |peak, value| peak.max(*value));
    if peak > 1e-9 {
        for value in values.iter_mut() {
            *value /= peak;
        }
    }
}

/// A chromagram over time.
#[derive(Clone, Debug)]
pub struct Chromagram {
    pub frames: Vec<Chroma>,
    pub frames_per_second: f64,
}

impl Chromagram {
    #[must_use]
    pub fn frame_at(&self, seconds: f64) -> usize {
        (seconds * self.frames_per_second).max(0.0) as usize
    }

    /// Averages the frames falling between two moments in seconds.
    #[must_use]
    pub fn between(&self, start_seconds: f64, end_seconds: f64) -> Chroma {
        if self.frames.is_empty() {
            return Chroma::default();
        }
        let start = self.frame_at(start_seconds);
        if start >= self.frames.len() {
            return Chroma::default();
        }
        let end = self
            .frame_at(end_seconds)
            .max(start + 1)
            .min(self.frames.len());
        Chroma::average(&self.frames[start..end])
    }
}

/// Computes a chromagram from mono samples.
#[must_use]
pub fn chromagram(samples: &[f32], sample_rate: u32) -> Chromagram {
    let frames_per_second = f64::from(sample_rate) / HOP as f64;
    if samples.len() < WINDOW || sample_rate == 0 {
        return Chromagram {
            frames: Vec::new(),
            frames_per_second,
        };
    }

    let window = hann_window(WINDOW);
    let bands = pitch_bands(sample_rate);

    let mut windowed = vec![0.0_f32; WINDOW];
    let (mut scratch_real, mut scratch_imag) = (Vec::new(), Vec::new());
    let mut spectrum = Vec::new();
    let mut frames = Vec::with_capacity(samples.len() / HOP);

    let mut start = 0;
    while start + WINDOW <= samples.len() {
        for (index, slot) in windowed.iter_mut().enumerate() {
            *slot = samples[start + index] * window[index];
        }
        magnitude_spectrum(
            &windowed,
            &mut scratch_real,
            &mut scratch_imag,
            &mut spectrum,
        );
        frames.push(frame_chroma(&spectrum, &bands));
        start += HOP;
    }

    Chromagram {
        frames,
        frames_per_second,
    }
}

/// The FFT bins belonging to one semitone.
struct PitchBand {
    pitch_class: usize,
    is_bass: bool,
    first_bin: usize,
    last_bin: usize,
}

/// Works out which bins belong to each semitone, once.
///
/// Assigning bins to pitches rather than pitches to bins is what makes the low
/// register usable: at 65 Hz a semitone is narrower than an FFT bin, so a
/// nearest-bin rule is the only way every bass note gets a share of the
/// spectrum instead of falling between the cracks.
fn pitch_bands(sample_rate: u32) -> Vec<PitchBand> {
    let bins = WINDOW / 2 + 1;
    let bin_hz = sample_rate as f32 / WINDOW as f32;
    let semitone = 2.0_f32.powf(0.5 / 12.0);

    let mut bands = Vec::new();
    for midi in 24..=108_u32 {
        let centre = 440.0 * 2.0_f32.powf((midi as f32 - 69.0) / 12.0);
        if !(MIN_HZ..=MAX_HZ).contains(&centre) {
            continue;
        }
        let low = centre / semitone;
        let high = centre * semitone;
        let mut first = (low / bin_hz).ceil() as usize;
        let mut last = (high / bin_hz).floor() as usize;
        if first > last || last >= bins {
            // The band is narrower than one bin: take the closest bin instead.
            let nearest = (centre / bin_hz).round() as usize;
            if nearest >= bins {
                continue;
            }
            first = nearest;
            last = nearest;
        }
        bands.push(PitchBand {
            pitch_class: (midi as usize) % PITCH_CLASSES,
            is_bass: centre <= BASS_MAX_HZ,
            first_bin: first,
            last_bin: last.min(bins - 1),
        });
    }
    bands
}

fn frame_chroma(spectrum: &[f32], bands: &[PitchBand]) -> Chroma {
    let mut pitches = [0.0_f32; PITCH_CLASSES];
    let mut bass = [0.0_f32; PITCH_CLASSES];
    let mut energy = 0.0_f32;

    for band in bands {
        let Some(slice) = spectrum.get(band.first_bin..=band.last_bin) else {
            continue;
        };
        // The peak, not the sum: a wide band must not out-score a narrow one
        // simply for containing more bins.
        let magnitude = slice.iter().fold(0.0_f32, |peak, value| peak.max(*value));
        if magnitude <= 0.0 {
            continue;
        }
        energy += magnitude;
        pitches[band.pitch_class] += magnitude;
        if band.is_bass {
            bass[band.pitch_class] += magnitude;
        }
    }

    // Harmonic discount. A partial at 3f is mostly the harmonic of f, not a
    // note of its own, so its energy is subtracted from the class it landed in
    // in proportion to how strong the fundamental below it is.
    let mut cleaned = pitches;
    for (class, discounted) in cleaned.iter_mut().enumerate() {
        for harmonic in 2..=HARMONICS {
            // Semitones between a fundamental and its nth harmonic.
            let offset = (12.0 * (harmonic as f32).log2()).round() as usize;
            let source = (class + PITCH_CLASSES - offset % PITCH_CLASSES) % PITCH_CLASSES;
            let weight = 1.0 / harmonic as f32;
            *discounted -= pitches[source] * weight * 0.34;
        }
        *discounted = discounted.max(0.0);
    }

    let total: f32 = cleaned.iter().sum();
    let tonalness = if total > 1e-9 {
        // Share of the energy held by the strongest four classes. A triad puts
        // nearly everything there; noise spreads it evenly.
        let mut sorted = cleaned;
        sorted.sort_by(|left, right| right.total_cmp(left));
        sorted[..4].iter().sum::<f32>() / total
    } else {
        0.0
    };

    let mut chroma = Chroma {
        pitches: cleaned,
        bass,
        energy,
        tonalness,
    };
    chroma.normalise();
    chroma
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    const RATE: u32 = 48_000;

    /// A sine at a MIDI pitch, with optional harmonics as a real instrument has.
    fn tone(pitch: u8, seconds: f32, harmonics: usize) -> Vec<f32> {
        let frequency = 440.0 * 2.0_f32.powf((f32::from(pitch) - 69.0) / 12.0);
        let count = (seconds * RATE as f32) as usize;
        (0..count)
            .map(|index| {
                let time = index as f32 / RATE as f32;
                (1..=harmonics)
                    .map(|harmonic| {
                        (TAU * frequency * harmonic as f32 * time).sin() / harmonic as f32
                    })
                    .sum::<f32>()
                    * 0.3
            })
            .collect()
    }

    fn mix(parts: &[Vec<f32>]) -> Vec<f32> {
        let length = parts.iter().map(Vec::len).max().unwrap_or(0);
        (0..length)
            .map(|index| parts.iter().filter_map(|part| part.get(index)).sum())
            .collect()
    }

    fn strongest(chroma: &Chroma) -> usize {
        chroma
            .pitches
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map_or(0, |(index, _)| index)
    }

    #[test]
    fn a_single_note_lands_on_its_own_pitch_class() {
        // A4 = MIDI 69 = pitch class 9.
        let chromagram = chromagram(&tone(69, 1.0, 1), RATE);
        let chroma = Chroma::average(&chromagram.frames);
        assert_eq!(strongest(&chroma), 9);
    }

    #[test]
    fn every_pitch_class_is_found_correctly() {
        for pitch in 60..72_u8 {
            let chromagram = chromagram(&tone(pitch, 0.5, 1), RATE);
            let chroma = Chroma::average(&chromagram.frames);
            assert_eq!(
                strongest(&chroma),
                usize::from(pitch) % 12,
                "pitch {pitch} was misclassified"
            );
        }
    }

    #[test]
    fn harmonics_do_not_invent_a_chord() {
        // One C3 with a full harmonic series must still read as mostly C, not
        // as a C major triad conjured from its third and fifth harmonics.
        let chromagram = chromagram(&tone(48, 1.0, 6), RATE);
        let chroma = Chroma::average(&chromagram.frames);
        assert_eq!(strongest(&chroma), 0, "the fundamental should dominate");
        // G is the third harmonic; it must be well below the fundamental.
        assert!(
            chroma.pitches[7] < 0.7,
            "the fifth read as {:.2} of the root — harmonics were not discounted",
            chroma.pitches[7]
        );
    }

    #[test]
    fn a_triad_shows_its_three_notes_above_the_rest() {
        // C major: C4, E4, G4.
        let chromagram = chromagram(
            &mix(&[tone(60, 1.0, 3), tone(64, 1.0, 3), tone(67, 1.0, 3)]),
            RATE,
        );
        let chroma = Chroma::average(&chromagram.frames);
        let chord = [0_usize, 4, 7];
        let weakest_chord_tone = chord
            .iter()
            .map(|class| chroma.pitches[*class])
            .fold(f32::MAX, f32::min);
        let strongest_other = (0..PITCH_CLASSES)
            .filter(|class| !chord.contains(class))
            .map(|class| chroma.pitches[class])
            .fold(0.0_f32, f32::max);
        assert!(
            weakest_chord_tone > strongest_other,
            "chord tones {weakest_chord_tone:.2} did not beat the rest {strongest_other:.2}"
        );
    }

    #[test]
    fn the_bass_register_is_collected_separately() {
        // A low C with a high E above it: the bass must read C, not E.
        let samples = mix(&[tone(36, 1.0, 3), tone(76, 1.0, 3)]);
        let chromagram = chromagram(&samples, RATE);
        let chroma = Chroma::average(&chromagram.frames);
        assert_eq!(
            chroma.bass_pitch_class(),
            Some(0),
            "the bass note was missed"
        );
    }

    #[test]
    fn tonal_material_scores_higher_than_noise() {
        let chord = chromagram(
            &mix(&[tone(60, 1.0, 3), tone(64, 1.0, 3), tone(67, 1.0, 3)]),
            RATE,
        );
        let tonal = Chroma::average(&chord.frames).tonalness;

        let mut state = 12_345_u32;
        let noise: Vec<f32> = (0..RATE as usize)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / 8_388_608.0 - 1.0
            })
            .collect();
        let unpitched = Chroma::average(&chromagram(&noise, RATE).frames).tonalness;
        assert!(
            tonal > unpitched,
            "a chord ({tonal:.3}) must read as more tonal than noise ({unpitched:.3})"
        );
    }

    #[test]
    fn silence_produces_no_energy_and_no_bass_note() {
        let chromagram = chromagram(&vec![0.0; RATE as usize], RATE);
        let chroma = Chroma::average(&chromagram.frames);
        assert!(chroma.energy < 1e-6);
        assert_eq!(chroma.bass_pitch_class(), None);
    }

    #[test]
    fn short_and_empty_input_is_handled() {
        assert!(chromagram(&[0.1, 0.2], RATE).frames.is_empty());
        assert!(chromagram(&[], RATE).frames.is_empty());
        assert!(chromagram(&vec![0.5; 8_192], 0).frames.is_empty());
        assert!(Chroma::average(&[]).energy.abs() < f32::EPSILON);
    }

    #[test]
    fn averaging_between_two_moments_uses_the_right_frames() {
        let chromagram = chromagram(&tone(69, 2.0, 1), RATE);
        let chroma = chromagram.between(0.5, 1.5);
        assert_eq!(strongest(&chroma), 9);
        // A range past the end must not panic.
        let _ = chromagram.between(90.0, 120.0);
    }
}
