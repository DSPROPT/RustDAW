#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

//! Chord recognition.
//!
//! Chords are decided **per beat, not per frame**. A chord is a musical object
//! that lasts a beat or a bar, so averaging the chromagram across each beat
//! removes passing notes and arpeggiation before anything is decided, and
//! guarantees the answer changes only where a chord could actually change.
//! That is why the beat tracker had to be right first.
//!
//! The sequence is then chosen by Viterbi rather than beat by beat. Picking the
//! best template for each beat independently produces a chart that flickers
//! between relative majors and minors on every other beat; a transition cost
//! makes the decoder prefer an explanation that holds still, and a key estimate
//! makes it prefer chords that belong together.

use crate::chroma::{Chroma, Chromagram, PITCH_CLASSES};

/// A chord type: its printed suffix and the semitones above the root.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quality {
    pub suffix: &'static str,
    intervals: &'static [u8],
    /// Multiplies this quality's score. Rarer chords need better evidence, or
    /// every major triad with a passing note becomes a 6/9.
    prior: f32,
}

/// Recognised chord types, commonest first.
pub const QUALITIES: [Quality; 10] = [
    Quality {
        suffix: "",
        intervals: &[0, 4, 7],
        prior: 1.0,
    },
    Quality {
        suffix: "m",
        intervals: &[0, 3, 7],
        prior: 1.0,
    },
    Quality {
        suffix: "7",
        intervals: &[0, 4, 7, 10],
        prior: 0.92,
    },
    Quality {
        suffix: "m7",
        intervals: &[0, 3, 7, 10],
        prior: 0.92,
    },
    Quality {
        suffix: "maj7",
        intervals: &[0, 4, 7, 11],
        prior: 0.88,
    },
    Quality {
        suffix: "sus4",
        intervals: &[0, 5, 7],
        prior: 0.84,
    },
    Quality {
        suffix: "sus2",
        intervals: &[0, 2, 7],
        prior: 0.8,
    },
    Quality {
        suffix: "6",
        intervals: &[0, 4, 7, 9],
        prior: 0.8,
    },
    Quality {
        suffix: "dim",
        intervals: &[0, 3, 6],
        prior: 0.78,
    },
    Quality {
        suffix: "aug",
        intervals: &[0, 4, 8],
        prior: 0.72,
    },
];

/// Number of decoder states: every root/quality pair, plus "no chord".
pub const CHORD_STATES: usize = PITCH_CLASSES * QUALITIES.len();
pub const NO_CHORD: usize = CHORD_STATES;

const NOTE_NAMES: [&str; PITCH_CLASSES] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// Scales template scores before decoding. Cosine similarities between a
/// chroma and the ten qualities all sit in a narrow band, so the gap between
/// the best chord and the runner-up is a few hundredths. Left at that scale it
/// is dwarfed by the change penalty and the decoder holds one chord for the
/// whole song; multiplied up, evidence and smoothing are finally in the same
/// units and the penalty means what it says.
const EMISSION_GAIN: f32 = 8.0;
/// Cost of changing chord between beats. Higher holds chords longer.
const CHANGE_PENALTY: f32 = 1.6;
/// The change penalty is reduced by this much on a downbeat, where chords
/// actually tend to change.
const DOWNBEAT_DISCOUNT: f32 = 0.55;
/// Emission for "no chord" on a beat that has tonal content.
const NO_CHORD_EMISSION: f32 = 0.25;
/// Below this tonalness a beat is drums or noise, not harmony.
const MIN_TONALNESS: f32 = 0.12;
/// Bonus for chords belonging to the detected key.
const IN_KEY_BONUS: f32 = 0.06;
/// Bonus when the bass is playing the chord's root. The bass player names the
/// chord more reliably than the middle of the texture does, where a suspension
/// or a passing note can outweigh the root.
const BASS_ROOT_BONUS: f32 = 0.09;
/// Decoder steps per beat. Chords change on half beats often enough that
/// deciding once per beat misses them, and the chroma still averages over
/// enough frames at this resolution to be stable.
const UNITS_PER_BEAT: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Chord {
    /// Pitch class of the root, C = 0.
    pub root: u8,
    /// Index into [`QUALITIES`].
    pub quality: usize,
    /// Pitch class in the bass when it is not the root, giving a slash chord.
    pub bass: Option<u8>,
}

impl Chord {
    #[must_use]
    pub fn name(&self) -> String {
        let root = NOTE_NAMES[usize::from(self.root) % PITCH_CLASSES];
        let suffix = QUALITIES[self.quality.min(QUALITIES.len() - 1)].suffix;
        match self.bass {
            Some(bass) if bass != self.root => {
                format!(
                    "{root}{suffix}/{}",
                    NOTE_NAMES[usize::from(bass) % PITCH_CLASSES]
                )
            }
            _ => format!("{root}{suffix}"),
        }
    }

    /// The chord without its inversion, for comparing two charts.
    #[must_use]
    pub fn root_name(&self) -> String {
        format!(
            "{}{}",
            NOTE_NAMES[usize::from(self.root) % PITCH_CLASSES],
            QUALITIES[self.quality.min(QUALITIES.len() - 1)].suffix
        )
    }
}

/// One chord held over a stretch of the song.
#[derive(Clone, Debug, PartialEq)]
pub struct ChordSpan {
    pub start_seconds: f64,
    pub end_seconds: f64,
    /// `None` means no chord was playing — an intro, a drum break, silence.
    pub chord: Option<Chord>,
    /// How far the chosen chord beat the runner-up, 0.0–1.0.
    pub confidence: f32,
}

impl ChordSpan {
    #[must_use]
    pub fn label(&self) -> String {
        self.chord
            .map_or_else(|| "N.C.".to_owned(), |chord| chord.name())
    }
}

/// A detected key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Key {
    pub tonic: u8,
    pub is_minor: bool,
}

impl Key {
    #[must_use]
    pub fn name(&self) -> String {
        format!(
            "{} {}",
            NOTE_NAMES[usize::from(self.tonic) % PITCH_CLASSES],
            if self.is_minor { "minor" } else { "major" }
        )
    }

    /// Pitch classes of the key's scale.
    #[must_use]
    pub fn scale(&self) -> [bool; PITCH_CLASSES] {
        const MAJOR: [u8; 7] = [0, 2, 4, 5, 7, 9, 11];
        const MINOR: [u8; 7] = [0, 2, 3, 5, 7, 8, 10];
        let steps = if self.is_minor { MINOR } else { MAJOR };
        let mut scale = [false; PITCH_CLASSES];
        for step in steps {
            scale[usize::from((self.tonic + step) % 12)] = true;
        }
        scale
    }
}

/// Krumhansl–Schmuckler key profiles: how much each scale degree is used in a
/// piece in that key. Correlating an average chromagram against all
/// twenty-four rotations is the standard way to name a key.
const MAJOR_PROFILE: [f32; PITCH_CLASSES] = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];
const MINOR_PROFILE: [f32; PITCH_CLASSES] = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];

/// Estimates the key from an average chromagram.
#[must_use]
pub fn detect_key(average: &[f32; PITCH_CLASSES]) -> Option<Key> {
    if average.iter().sum::<f32>() <= 1e-6 {
        return None;
    }
    let mut best = None;
    let mut best_score = f32::NEG_INFINITY;
    for tonic in 0..PITCH_CLASSES {
        for is_minor in [false, true] {
            let profile = if is_minor {
                MINOR_PROFILE
            } else {
                MAJOR_PROFILE
            };
            let score = correlation(average, &profile, tonic);
            if score > best_score {
                best_score = score;
                best = Some(Key {
                    tonic: tonic as u8,
                    is_minor,
                });
            }
        }
    }
    best
}

/// Pearson correlation of a chroma against a profile rotated to `tonic`.
fn correlation(chroma: &[f32; PITCH_CLASSES], profile: &[f32; PITCH_CLASSES], tonic: usize) -> f32 {
    let rotated: Vec<f32> = (0..PITCH_CLASSES)
        .map(|index| profile[(index + PITCH_CLASSES - tonic) % PITCH_CLASSES])
        .collect();
    let mean_chroma = chroma.iter().sum::<f32>() / PITCH_CLASSES as f32;
    let mean_profile = rotated.iter().sum::<f32>() / PITCH_CLASSES as f32;
    let mut covariance = 0.0;
    let mut variance_chroma = 0.0;
    let mut variance_profile = 0.0;
    for index in 0..PITCH_CLASSES {
        let a = chroma[index] - mean_chroma;
        let b = rotated[index] - mean_profile;
        covariance += a * b;
        variance_chroma += a * a;
        variance_profile += b * b;
    }
    let denominator = (variance_chroma * variance_profile).sqrt();
    if denominator > 1e-9 {
        covariance / denominator
    } else {
        0.0
    }
}

/// How well a chroma matches one root/quality template, as a cosine similarity.
fn template_score(chroma: &[f32; PITCH_CLASSES], root: usize, quality: &Quality) -> f32 {
    let mut template = [0.0_f32; PITCH_CLASSES];
    for interval in quality.intervals {
        template[(root + usize::from(*interval)) % PITCH_CLASSES] = 1.0;
    }
    let dot: f32 = chroma
        .iter()
        .zip(template.iter())
        .map(|(value, weight)| value * weight)
        .sum();
    let norm_chroma = chroma.iter().map(|value| value * value).sum::<f32>().sqrt();
    let norm_template = (quality.intervals.len() as f32).sqrt();
    if norm_chroma <= 1e-9 {
        return 0.0;
    }
    dot / (norm_chroma * norm_template) * quality.prior
}

/// Detects the chord on every beat, then merges equal neighbours into spans.
///
/// `beat_times` comes from the beat tracker; `beats_per_bar` and
/// `downbeat_index` let the decoder charge less for a change on a downbeat.
#[must_use]
pub fn detect_chords(
    chromagram: &Chromagram,
    beat_times: &[f64],
    beats_per_bar: u16,
    downbeat_index: usize,
) -> (Vec<ChordSpan>, Option<Key>) {
    if beat_times.len() < 2 || chromagram.frames.is_empty() {
        return (Vec::new(), None);
    }

    // One chroma per decoder unit, which is a fraction of a beat.
    let mut units: Vec<f64> = Vec::with_capacity(beat_times.len() * UNITS_PER_BEAT);
    for pair in beat_times.windows(2) {
        let step = (pair[1] - pair[0]) / UNITS_PER_BEAT as f64;
        for unit in 0..UNITS_PER_BEAT {
            units.push(step.mul_add(unit as f64, pair[0]));
        }
    }
    units.push(*beat_times.last().unwrap_or(&0.0));

    // The chromagram stops one window short of the end of the audio. Beats past
    // that point have no evidence behind them and would be reported as "no
    // chord", inventing a silence that is really just the edge of the analysis.
    let analysable_end = chromagram.frames.len() as f64 / chromagram.frames_per_second;
    units.retain(|time| *time <= analysable_end);
    if units.len() < 2 {
        return (Vec::new(), None);
    }

    let beats: Vec<Chroma> = units
        .windows(2)
        .map(|pair| chromagram.between(pair[0], pair[1]))
        .collect();

    let mut average = [0.0_f32; PITCH_CLASSES];
    for beat in &beats {
        for (slot, value) in average.iter_mut().zip(beat.pitches.iter()) {
            *slot += value;
        }
    }
    let key = detect_key(&average);
    let scale = key.map(|key| key.scale());

    // Emissions.
    let mut emissions = vec![[0.0_f32; NO_CHORD + 1]; beats.len()];
    for (beat_index, beat) in beats.iter().enumerate() {
        let row = &mut emissions[beat_index];
        if beat.energy <= 1e-6 || beat.tonalness < MIN_TONALNESS {
            // Nothing tonal here; every chord is a bad explanation.
            row.fill(0.35);
            row[NO_CHORD] = 1.0;
            continue;
        }
        let bass = beat.bass_pitch_class();
        for root in 0..PITCH_CLASSES {
            for (quality_index, quality) in QUALITIES.iter().enumerate() {
                let mut score = template_score(&beat.pitches, root, quality);
                if bass == Some(root as u8) {
                    score += BASS_ROOT_BONUS;
                }
                if let Some(scale) = scale {
                    let in_key = quality
                        .intervals
                        .iter()
                        .all(|interval| scale[(root + usize::from(*interval)) % PITCH_CLASSES]);
                    if in_key {
                        score += IN_KEY_BONUS;
                    }
                }
                row[root * QUALITIES.len() + quality_index] = score;
            }
        }
        row[NO_CHORD] = NO_CHORD_EMISSION;
    }

    for row in &mut emissions {
        for value in row.iter_mut() {
            *value *= EMISSION_GAIN;
        }
    }

    let states = decode(&emissions, beats_per_bar, downbeat_index);
    let confidences = margins(&emissions, &states);
    build_spans(&states, &confidences, &beats, &units)
        .map_or((Vec::new(), key), |spans| (spans, key))
}

/// Viterbi over the beat sequence.
fn decode(
    emissions: &[[f32; NO_CHORD + 1]],
    beats_per_bar: u16,
    downbeat_index: usize,
) -> Vec<usize> {
    let beats = emissions.len();
    let states = NO_CHORD + 1;
    let mut score = vec![f32::NEG_INFINITY; beats * states];
    let mut back = vec![0_usize; beats * states];

    score[..states].copy_from_slice(&emissions[0]);

    for beat in 1..beats {
        // A change costs less where a bar begins.
        let units_per_bar = usize::from(beats_per_bar.max(1)) * UNITS_PER_BEAT;
        let offset = downbeat_index * UNITS_PER_BEAT % units_per_bar;
        let on_downbeat = (beat + units_per_bar - offset) % units_per_bar == 0;
        let penalty = if on_downbeat {
            CHANGE_PENALTY * DOWNBEAT_DISCOUNT
        } else {
            CHANGE_PENALTY
        };

        // The best predecessor is either "stay" or the overall best minus the
        // change penalty, so the whole step is O(states) rather than O(states²).
        let (earlier, current) = score.split_at_mut(beat * states);
        let previous = &earlier[(beat - 1) * states..];
        let (best_previous, best_value) = previous
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map_or((0, f32::NEG_INFINITY), |(index, value)| (index, *value));

        for state in 0..states {
            let stay = previous[state];
            let switch = best_value - penalty;
            let (from, value) = if stay >= switch {
                (state, stay)
            } else {
                (best_previous, switch)
            };
            current[state] = value + emissions[beat][state];
            back[beat * states + state] = from;
        }
    }

    let last = (beats - 1) * states;
    let mut cursor = score[last..last + states]
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map_or(NO_CHORD, |(index, _)| index);

    let mut path = vec![0_usize; beats];
    for beat in (0..beats).rev() {
        path[beat] = cursor;
        cursor = back[beat * states + cursor];
    }
    path
}

/// How far each chosen state beat the best alternative on its beat.
fn margins(emissions: &[[f32; NO_CHORD + 1]], states: &[usize]) -> Vec<f32> {
    emissions
        .iter()
        .zip(states.iter())
        .map(|(row, chosen)| {
            let picked = row[*chosen];
            let best_other = row
                .iter()
                .enumerate()
                .filter(|(index, _)| index != chosen)
                .map(|(_, value)| *value)
                .fold(f32::NEG_INFINITY, f32::max);
            (picked - best_other).clamp(0.0, 1.0)
        })
        .collect()
}

/// Merges equal neighbouring beats into spans and names the bass note.
fn build_spans(
    states: &[usize],
    confidences: &[f32],
    beats: &[Chroma],
    unit_times: &[f64],
) -> Option<Vec<ChordSpan>> {
    let mut spans: Vec<ChordSpan> = Vec::new();
    for (index, state) in states.iter().enumerate() {
        let start = *unit_times.get(index)?;
        let end = *unit_times.get(index + 1)?;
        let chord = (*state != NO_CHORD).then(|| {
            let root = (state / QUALITIES.len()) as u8;
            let quality = state % QUALITIES.len();
            // A bass note that is not the root makes this an inversion.
            let bass = beats
                .get(index)
                .and_then(Chroma::bass_pitch_class)
                .filter(|bass| *bass != root);
            Chord {
                root,
                quality,
                bass,
            }
        });

        match spans.last_mut() {
            // Extend the run when the chord itself is unchanged. The bass may
            // differ between beats of one chord; the first reading wins rather
            // than splitting a held chord into inversions of itself.
            Some(last)
                if last.chord.map(|chord| (chord.root, chord.quality))
                    == chord.map(|chord| (chord.root, chord.quality)) =>
            {
                last.end_seconds = end;
                last.confidence = last
                    .confidence
                    .max(confidences.get(index).copied().unwrap_or(0.0));
            }
            _ => spans.push(ChordSpan {
                start_seconds: start,
                end_seconds: end,
                chord,
                confidence: confidences.get(index).copied().unwrap_or(0.0),
            }),
        }
    }
    Some(spans)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chroma::chromagram;
    use std::f32::consts::TAU;

    const RATE: u32 = 48_000;

    fn tone(pitch: u8, seconds: f32) -> Vec<f32> {
        let frequency = 440.0 * 2.0_f32.powf((f32::from(pitch) - 69.0) / 12.0);
        let count = (seconds * RATE as f32) as usize;
        (0..count)
            .map(|index| {
                let time = index as f32 / RATE as f32;
                (1..=3)
                    .map(|harmonic| {
                        (TAU * frequency * harmonic as f32 * time).sin() / harmonic as f32
                    })
                    .sum::<f32>()
                    * 0.25
            })
            .collect()
    }

    /// Renders a sequence of chords, each `seconds` long.
    fn play(chords: &[&[u8]], seconds: f32) -> Vec<f32> {
        let mut output = Vec::new();
        for notes in chords {
            let parts: Vec<Vec<f32>> = notes.iter().map(|pitch| tone(*pitch, seconds)).collect();
            let length = parts.iter().map(Vec::len).max().unwrap_or(0);
            for index in 0..length {
                output.push(parts.iter().filter_map(|part| part.get(index)).sum());
            }
        }
        output
    }

    fn beats_over(total_seconds: f64, beat_seconds: f64) -> Vec<f64> {
        let mut times = Vec::new();
        let mut time = 0.0;
        while time <= total_seconds {
            times.push(time);
            time += beat_seconds;
        }
        times
    }

    fn labels(spans: &[ChordSpan]) -> Vec<String> {
        spans
            .iter()
            .map(|span| {
                span.chord
                    .map_or_else(|| "N.C.".to_owned(), |chord| chord.root_name())
            })
            .collect()
    }

    #[test]
    fn a_major_triad_is_named_correctly() {
        let samples = play(&[&[60, 64, 67]], 4.0);
        let (spans, _) = detect_chords(&chromagram(&samples, RATE), &beats_over(4.0, 0.5), 4, 0);
        assert!(!spans.is_empty());
        assert_eq!(spans[0].chord.unwrap().root_name(), "C");
    }

    #[test]
    fn major_and_minor_are_told_apart() {
        let major = play(&[&[60, 64, 67]], 3.0);
        let minor = play(&[&[60, 63, 67]], 3.0);
        let beats = beats_over(3.0, 0.5);
        let (major_spans, _) = detect_chords(&chromagram(&major, RATE), &beats, 4, 0);
        let (minor_spans, _) = detect_chords(&chromagram(&minor, RATE), &beats, 4, 0);
        assert_eq!(major_spans[0].chord.unwrap().root_name(), "C");
        assert_eq!(minor_spans[0].chord.unwrap().root_name(), "Cm");
    }

    #[test]
    fn a_seventh_is_recognised() {
        let samples = play(&[&[67, 71, 74, 77]], 3.0);
        let (spans, _) = detect_chords(&chromagram(&samples, RATE), &beats_over(3.0, 0.5), 4, 0);
        assert_eq!(spans[0].chord.unwrap().root_name(), "G7");
    }

    #[test]
    fn a_progression_is_followed() {
        // C – Am – F – G, two seconds each.
        let samples = play(
            &[&[60, 64, 67], &[57, 60, 64], &[53, 57, 60], &[55, 59, 62]],
            2.0,
        );
        let (spans, _) = detect_chords(&chromagram(&samples, RATE), &beats_over(8.0, 0.5), 4, 0);
        let found = labels(&spans);
        assert_eq!(
            found,
            ["C", "Am", "F", "G"],
            "progression came out as {found:?}"
        );
    }

    #[test]
    fn a_held_chord_is_one_span_not_one_per_beat() {
        let samples = play(&[&[60, 64, 67]], 6.0);
        let (spans, _) = detect_chords(&chromagram(&samples, RATE), &beats_over(6.0, 0.5), 4, 0);
        assert_eq!(
            spans.len(),
            1,
            "a held chord split into {} spans",
            spans.len()
        );
    }

    #[test]
    fn an_inversion_is_named_over_its_bass() {
        // C major with E in the bass.
        let samples = play(&[&[52, 60, 67, 72]], 4.0);
        let (spans, _) = detect_chords(&chromagram(&samples, RATE), &beats_over(4.0, 0.5), 4, 0);
        let chord = spans[0].chord.unwrap();
        assert_eq!(chord.root_name(), "C");
        assert_eq!(chord.name(), "C/E", "the inversion was not named");
    }

    #[test]
    fn silence_is_reported_as_no_chord() {
        let samples = vec![0.0_f32; RATE as usize * 3];
        let (spans, _) = detect_chords(&chromagram(&samples, RATE), &beats_over(3.0, 0.5), 4, 0);
        assert!(spans.iter().all(|span| span.chord.is_none()));
        assert_eq!(spans[0].label(), "N.C.");
    }

    #[test]
    fn the_key_of_a_diatonic_progression_is_found() {
        let samples = play(
            &[&[60, 64, 67], &[57, 60, 64], &[53, 57, 60], &[55, 59, 62]],
            2.0,
        );
        let (_, key) = detect_chords(&chromagram(&samples, RATE), &beats_over(8.0, 0.5), 4, 0);
        let key = key.expect("no key was found");
        assert!(
            key.name() == "C major" || key.name() == "A minor",
            "expected C major or its relative minor, got {}",
            key.name()
        );
    }

    #[test]
    fn a_key_scale_contains_the_right_notes() {
        let scale = Key {
            tonic: 0,
            is_minor: false,
        }
        .scale();
        // C major has no accidentals.
        assert!(scale[0] && scale[2] && scale[4] && scale[5] && scale[7] && scale[9] && scale[11]);
        assert!(!scale[1] && !scale[3] && !scale[6] && !scale[8] && !scale[10]);
    }

    #[test]
    fn chord_names_read_the_way_musicians_write_them() {
        assert_eq!(
            Chord {
                root: 0,
                quality: 0,
                bass: None
            }
            .name(),
            "C"
        );
        assert_eq!(
            Chord {
                root: 9,
                quality: 1,
                bass: None
            }
            .name(),
            "Am"
        );
        assert_eq!(
            Chord {
                root: 7,
                quality: 2,
                bass: None
            }
            .name(),
            "G7"
        );
        assert_eq!(
            Chord {
                root: 0,
                quality: 0,
                bass: Some(4)
            }
            .name(),
            "C/E"
        );
        // A bass equal to the root is not a slash chord.
        assert_eq!(
            Chord {
                root: 0,
                quality: 0,
                bass: Some(0)
            }
            .name(),
            "C"
        );
    }

    #[test]
    fn empty_input_is_handled() {
        let empty = Chromagram {
            frames: Vec::new(),
            frames_per_second: 23.0,
        };
        assert!(detect_chords(&empty, &[], 4, 0).0.is_empty());
        let samples = play(&[&[60, 64, 67]], 2.0);
        assert!(
            detect_chords(&chromagram(&samples, RATE), &[1.0], 4, 0)
                .0
                .is_empty()
        );
    }
}
