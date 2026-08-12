//! Monophonic pitch detection, for tuning an instrument.
//!
//! Not an FFT. A guitar's low E is 82 Hz and its drop-D is 73, and telling one
//! cent from the next there means resolving about 0.05 Hz — a transform would
//! need a window of twenty seconds to do that from bin spacing alone. Tuners
//! work in the time domain instead, and this is YIN: the difference function,
//! normalised so its own average sets the threshold, with the winning dip
//! interpolated to a fraction of a sample.
//!
//! What that buys is accuracy at the bottom of the range, and immunity to the
//! octave errors a spectral peak-picker makes on a plucked string, where the
//! second harmonic is routinely louder than the fundamental.
//!
//! See de Cheveigné and Kawahara, "YIN, a fundamental frequency estimator for
//! speech and music" (2002).

/// Below this the difference function is considered to have found a period.
/// YIN's paper suggests 0.1; a plucked string decays through its own harmonics
/// and does better with a little more tolerance.
const THRESHOLD: f32 = 0.15;
/// Signals quieter than this are silence rather than a quiet note, and
/// reporting a pitch for them makes a tuner's needle dance at nothing.
const SILENCE_RMS: f32 = 0.002;

/// A detected pitch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pitch {
    pub hertz: f32,
    /// How confident the estimate is, `0` to `1`. Derived from how deep the
    /// winning dip in the difference function went.
    pub confidence: f32,
}

/// Finds the fundamental of a monophonic signal.
///
/// `lowest` and `highest` bound the search, in Hz — narrowing them is what
/// keeps a bass guitar's fundamental from being mistaken for a rumble and a
/// harmonic from being mistaken for the note. Returns `None` for silence, for
/// a window too short to hold two periods of `lowest`, or when nothing in the
/// range looks periodic.
#[must_use]
pub fn detect(samples: &[f32], sample_rate: f32, lowest: f32, highest: f32) -> Option<Pitch> {
    if sample_rate <= 0.0 || lowest <= 0.0 || highest <= lowest {
        return None;
    }
    let min_lag = (sample_rate / highest).floor().max(2.0) as usize;
    let max_lag = (sample_rate / lowest).ceil() as usize;
    // Two full periods of the lowest note asked for, or the difference
    // function has nothing to compare the second one against.
    if samples.len() < max_lag * 2 || max_lag <= min_lag {
        return None;
    }

    let energy: f32 = samples.iter().map(|sample| sample * sample).sum();
    let rms = (energy / samples.len() as f32).sqrt();
    if rms < SILENCE_RMS {
        return None;
    }

    // The window used for comparison. Everything past `max_lag` is the tail
    // each shifted copy is compared against.
    let window = samples.len() - max_lag;
    let mut difference = vec![0.0_f32; max_lag + 1];
    for (lag, slot) in difference.iter_mut().enumerate().skip(1) {
        let mut total = 0.0;
        for index in 0..window {
            let delta = samples[index] - samples[index + lag];
            total += delta * delta;
        }
        *slot = total;
    }

    // Cumulative mean normalisation: divide each value by the running mean of
    // everything before it. This is what stops lag zero — always a perfect
    // match — from winning, and what makes one fixed threshold work across
    // signals of any level.
    let mut normalised = vec![1.0_f32; max_lag + 1];
    let mut running = 0.0_f32;
    for lag in 1..=max_lag {
        running += difference[lag];
        normalised[lag] = if running > 0.0 {
            difference[lag] * lag as f32 / running
        } else {
            1.0
        };
    }

    // The first dip under the threshold, not the deepest: the deepest is
    // usually an octave down, since two periods match as well as one.
    let mut chosen = None;
    for lag in min_lag..=max_lag {
        if normalised[lag] < THRESHOLD {
            // Walk to the bottom of this dip before accepting it.
            let mut best = lag;
            while best < max_lag && normalised[best + 1] < normalised[best] {
                best += 1;
            }
            chosen = Some(best);
            break;
        }
    }
    // Nothing crossed the threshold: fall back to the shallowest dip in range,
    // which is what a quiet or decaying note leaves behind.
    let lag = chosen.or_else(|| {
        (min_lag..=max_lag)
            .min_by(|left, right| normalised[*left].total_cmp(&normalised[*right]))
            .filter(|lag| normalised[*lag] < 0.5)
    })?;

    // A sample is a coarse unit at these frequencies — one sample either way at
    // 82 Hz is nearly twenty cents — so the true minimum is interpolated from
    // the parabola through the winning point and its neighbours.
    let refined = interpolate(&normalised, lag);
    let hertz = sample_rate / refined;
    if !hertz.is_finite() || hertz < lowest || hertz > highest {
        return None;
    }
    Some(Pitch {
        hertz,
        confidence: (1.0 - normalised[lag]).clamp(0.0, 1.0),
    })
}

/// The vertex of the parabola through `lag` and its neighbours.
fn interpolate(values: &[f32], lag: usize) -> f32 {
    if lag == 0 || lag + 1 >= values.len() {
        return lag as f32;
    }
    let (previous, current, next) = (values[lag - 1], values[lag], values[lag + 1]);
    let denominator = 2.0 * (2.0 * current - previous - next);
    if denominator.abs() < 1e-12 {
        return lag as f32;
    }
    lag as f32 + (next - previous) / denominator
}

/// The twelve pitch classes, sharps rather than flats, matching the chord
/// detector so one note is spelled one way throughout.
pub const NOTE_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// Concert pitch, and what the reference control moves away from.
pub const DEFAULT_REFERENCE_HZ: f32 = 440.0;

/// A pitch expressed the way a tuner shows it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reading {
    pub hertz: f32,
    /// The nearest note, as a MIDI number.
    pub midi: i32,
    /// How far the pitch is from that note, in cents. Negative is flat.
    pub cents: f32,
    pub confidence: f32,
}

impl Reading {
    #[must_use]
    pub fn note_name(&self) -> &'static str {
        NOTE_NAMES[self.midi.rem_euclid(12) as usize]
    }

    /// Scientific pitch notation, where middle C is C4 and a guitar's low E
    /// is E2 — the convention the rest of the application uses.
    #[must_use]
    pub fn octave(&self) -> i32 {
        self.midi.div_euclid(12) - 1
    }

    #[must_use]
    pub fn label(&self) -> String {
        format!("{}{}", self.note_name(), self.octave())
    }

    /// Whether this is close enough to call it in tune. Three cents is under
    /// what anyone hears on a plucked string, and is what hardware tuners use.
    #[must_use]
    pub fn in_tune(&self) -> bool {
        self.cents.abs() <= 3.0
    }
}

/// Names a frequency against a reference for A4.
///
/// Returns `None` for a frequency at or below zero, which has no note.
#[must_use]
pub fn reading(pitch: Pitch, reference_hz: f32) -> Option<Reading> {
    if pitch.hertz <= 0.0 || reference_hz <= 0.0 {
        return None;
    }
    // A4 is MIDI 69; every other note is a twelfth of an octave from it.
    let exact = 69.0 + 12.0 * (pitch.hertz / reference_hz).log2();
    if !exact.is_finite() {
        return None;
    }
    let midi = exact.round();
    Some(Reading {
        hertz: pitch.hertz,
        midi: midi as i32,
        cents: (exact - midi) * 100.0,
        confidence: pitch.confidence,
    })
}

/// The open strings of a guitar in standard tuning, low to high, as MIDI
/// numbers: E2 A2 D3 G3 B3 E4.
pub const GUITAR_STRINGS: [i32; 6] = [40, 45, 50, 55, 59, 64];
/// The open strings of a four-string bass, low to high: E1 A1 D2 G2.
pub const BASS_STRINGS: [i32; 4] = [28, 33, 38, 43];

/// Which open string a reading is nearest, as an index into `strings`.
///
/// For showing the player which peg to turn. Returns `None` when the note is
/// more than a whole tone from any of them, where naming a string would be a
/// guess rather than a help.
#[must_use]
pub fn nearest_string(reading: &Reading, strings: &[i32]) -> Option<usize> {
    strings
        .iter()
        .enumerate()
        .min_by_key(|(_, note)| (**note - reading.midi).abs())
        .filter(|(_, note)| (**note - reading.midi).abs() <= 2)
        .map(|(index, _)| index)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: f32 = 48_000.0;

    fn sine(hertz: f32, frames: usize) -> Vec<f32> {
        (0..frames)
            .map(|index| {
                (index as f32 / RATE * hertz * std::f32::consts::TAU).sin() * 0.3
            })
            .collect()
    }

    /// A plucked string: a fundamental with harmonics above it, the second
    /// louder than the first, which is what fools a spectral peak-picker.
    fn plucked(hertz: f32, frames: usize) -> Vec<f32> {
        (0..frames)
            .map(|index| {
                let phase = index as f32 / RATE * hertz * std::f32::consts::TAU;
                (phase.sin() * 0.3 + (phase * 2.0).sin() * 0.5 + (phase * 3.0).sin() * 0.25) * 0.4
            })
            .collect()
    }


    /// Genuine pseudo-random noise. An indexed expression like
    /// `index * k % n` repeats, and a repeating signal has a pitch — which is
    /// the opposite of what a noise test needs.
    fn noise(frames: usize, amplitude: f32) -> Vec<f32> {
        let mut state = 0x2545_F491_u32;
        (0..frames)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                ((state >> 8) as f32 / 8_388_608.0 - 1.0) * amplitude
            })
            .collect()
    }

    fn cents_between(measured: f32, expected: f32) -> f32 {
        1_200.0 * (measured / expected).log2()
    }

    #[test]
    fn every_open_guitar_string_is_found_to_within_a_cent() {
        // Standard tuning, low to high. A tuner that cannot resolve a cent is
        // not a tuner.
        for expected in [82.407, 110.0, 146.832, 195.998, 246.942, 329.628] {
            let samples = sine(expected, 8_192);
            let pitch = detect(&samples, RATE, 60.0, 1_400.0)
                .unwrap_or_else(|| panic!("{expected} Hz was not detected"));
            let error = cents_between(pitch.hertz, expected).abs();
            assert!(
                error < 1.0,
                "{expected} Hz came back as {} Hz, {error:.2} cents out",
                pitch.hertz
            );
        }
    }

    #[test]
    fn a_plucked_string_is_not_heard_an_octave_up() {
        // The second harmonic being the loudest partial is normal on a guitar,
        // and is exactly what makes naive detectors report the octave.
        for expected in [82.407_f32, 110.0, 146.832] {
            let samples = plucked(expected, 8_192);
            let pitch = detect(&samples, RATE, 60.0, 1_400.0)
                .unwrap_or_else(|| panic!("{expected} Hz was not detected"));
            let error = cents_between(pitch.hertz, expected).abs();
            assert!(
                error < 2.0,
                "{expected} Hz came back as {} Hz — {error:.0} cents, an octave error?",
                pitch.hertz
            );
        }
    }

    #[test]
    fn a_string_a_few_cents_out_reads_as_a_few_cents_out() {
        // The whole job: not "which note" but "how far off".
        let reference = 110.0_f32;
        for offset in [-25.0_f32, -8.0, -1.0, 1.0, 8.0, 25.0] {
            let detuned = reference * 2.0_f32.powf(offset / 1_200.0);
            let samples = plucked(detuned, 8_192);
            let pitch = detect(&samples, RATE, 60.0, 1_400.0).expect("detected");
            let measured = cents_between(pitch.hertz, reference);
            assert!(
                (measured - offset).abs() < 1.0,
                "{offset:+} cents read as {measured:+.2}"
            );
        }
    }

    #[test]
    fn silence_reports_nothing_rather_than_a_number() {
        // A needle that dances at a silent input is worse than no needle.
        assert!(detect(&vec![0.0; 8_192], RATE, 60.0, 1_400.0).is_none());
        assert!(detect(&noise(8_192, 1e-6), RATE, 60.0, 1_400.0).is_none());
    }

    #[test]
    fn a_window_too_short_for_the_lowest_note_is_refused() {
        // Rather than reporting a confident wrong answer.
        let samples = sine(82.407, 512);
        assert!(detect(&samples, RATE, 60.0, 1_400.0).is_none());
    }

    #[test]
    fn a_note_outside_the_range_is_not_dragged_into_it() {
        // A 40 Hz rumble must not be reported as the bottom of the range.
        let samples = sine(40.0, 8_192);
        let found = detect(&samples, RATE, 70.0, 1_400.0);
        if let Some(pitch) = found {
            assert!(
                cents_between(pitch.hertz, 40.0).abs() > 50.0,
                "a 40 Hz tone was reported as {} Hz inside a 70 Hz floor",
                pitch.hertz
            );
        }
    }

    #[test]
    fn a_clean_note_is_more_confident_than_noise_over_one() {
        let clean = detect(&sine(196.0, 8_192), RATE, 60.0, 1_400.0).expect("detected");
        let hiss = noise(8_192, 0.35);
        let mut noisy = sine(196.0, 8_192);
        for (sample, hiss) in noisy.iter_mut().zip(&hiss) {
            *sample += hiss;
        }
        let noisy = detect(&noisy, RATE, 60.0, 1_400.0).expect("detected");
        assert!(
            clean.confidence > noisy.confidence,
            "clean {} was no more confident than noisy {}",
            clean.confidence,
            noisy.confidence
        );
    }

    #[test]
    fn a_bass_low_b_is_within_reach() {
        // Five-string bass, 30.9 Hz, which needs a long window and is where a
        // spectral detector gives up entirely.
        let samples = plucked(30.868, 16_384);
        let pitch = detect(&samples, RATE, 28.0, 1_400.0).expect("detected");
        assert!(
            cents_between(pitch.hertz, 30.868).abs() < 3.0,
            "a low B came back as {} Hz",
            pitch.hertz
        );
    }

    #[test]
    fn a_pitch_is_named_the_way_the_rest_of_the_application_names_notes() {
        // Scientific pitch: middle C is C4, a guitar's low E is E2.
        let named = |hertz: f32| {
            reading(Pitch { hertz, confidence: 1.0 }, DEFAULT_REFERENCE_HZ)
                .expect("named")
                .label()
        };
        assert_eq!(named(440.0), "A4");
        assert_eq!(named(261.626), "C4");
        assert_eq!(named(82.407), "E2");
        assert_eq!(named(329.628), "E4");
        assert_eq!(named(30.868), "B0");
    }

    #[test]
    fn cents_are_signed_the_way_a_needle_leans() {
        let flat = reading(Pitch { hertz: 438.0, confidence: 1.0 }, 440.0).expect("named");
        let sharp = reading(Pitch { hertz: 442.0, confidence: 1.0 }, 440.0).expect("named");
        assert!(flat.cents < 0.0, "flat should read negative: {}", flat.cents);
        assert!(sharp.cents > 0.0, "sharp should read positive: {}", sharp.cents);
        assert_eq!(flat.note_name(), "A");
        assert_eq!(sharp.note_name(), "A");
        assert!(!flat.in_tune() && !sharp.in_tune());
        let spot_on = reading(Pitch { hertz: 440.0, confidence: 1.0 }, 440.0).expect("named");
        assert!(spot_on.in_tune());
        assert!(spot_on.cents.abs() < 1e-3);
    }

    #[test]
    fn moving_the_reference_moves_every_note_with_it() {
        // Tuning to A=432 must make 432 read as a dead-on A, not as a flat one.
        let at_432 = reading(Pitch { hertz: 432.0, confidence: 1.0 }, 432.0).expect("named");
        assert_eq!(at_432.label(), "A4");
        assert!(at_432.in_tune());
        let at_440 = reading(Pitch { hertz: 432.0, confidence: 1.0 }, 440.0).expect("named");
        assert!(at_440.cents < -30.0, "{} cents", at_440.cents);
    }

    #[test]
    fn a_reading_points_at_the_string_being_tuned() {
        let string_of = |hertz: f32| {
            let reading = reading(Pitch { hertz, confidence: 1.0 }, DEFAULT_REFERENCE_HZ)
                .expect("named");
            nearest_string(&reading, &GUITAR_STRINGS)
        };
        assert_eq!(string_of(82.407), Some(0), "low E");
        assert_eq!(string_of(329.628), Some(5), "high E");
        assert_eq!(string_of(196.0), Some(3), "G");
        // A semitone flat still points at the string it is closest to.
        assert_eq!(string_of(77.78), Some(0), "low E, a semitone down");
        // Inside the range every note is within a whole tone of some string —
        // the widest gap between two of them is five semitones — so naming one
        // is always a help. Outside it, naming one would be a guess.
        assert_eq!(string_of(30.868), None, "a bass low B is not a guitar string");
        assert_eq!(string_of(1_046.5), None, "two octaves above the top string");
    }

    #[test]
    fn nonsense_arguments_are_refused_rather_than_panicking() {
        let samples = sine(220.0, 8_192);
        assert!(detect(&samples, 0.0, 60.0, 1_400.0).is_none());
        assert!(detect(&samples, RATE, 0.0, 1_400.0).is_none());
        assert!(detect(&samples, RATE, 1_400.0, 60.0).is_none());
        assert!(detect(&[], RATE, 60.0, 1_400.0).is_none());
    }
}
