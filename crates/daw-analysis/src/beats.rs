#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

//! Tempo estimation and beat tracking.
//!
//! Two stages, following Ellis (2007). First a global tempo, chosen by
//! autocorrelating the onset envelope and weighting the result by a
//! perceptual prior — without that prior, half and double tempo score almost
//! identically and the answer flips between songs. Then the beats themselves,
//! by dynamic programming: the sequence that best balances landing on onsets
//! against keeping a steady period.
//!
//! Tracking beats globally rather than beat-to-beat is the whole point. A
//! greedy tracker follows every fill and rubato, which is what produces the
//! bar-to-bar tempo jitter seen in imported grids.

use crate::onset::OnsetEnvelope;

/// Tempo search range. Anything outside this is heard as a subdivision or a
/// half-time feel of something inside it.
pub const MIN_BPM: f64 = 50.0;
pub const MAX_BPM: f64 = 210.0;
/// Tempo the prior treats as most likely when nothing else is known, in BPM.
const PRIOR_CENTRE_BPM: f64 = 120.0;
/// Width of the prior in octaves.
const PRIOR_WIDTH_OCTAVES: f64 = 1.05;

/// Roughly where the tempo is expected to be.
///
/// Autocorrelation cannot tell a tempo from its double: both line up with the
/// beats, and a song at 174 correlates at 87 just as strongly. Something has to
/// break that tie, and by default it is a prior centred at 120 BPM — the tempo
/// most music is nearest to.
///
/// Which is wrong for the music that is not. Drum and bass at 174 is closer to
/// 120 when halved to 87, so the halved answer wins on the prior even though
/// the evidence is even, and the detector reports a grid at half speed. The
/// same happens to hardcore, and to anything else living at the fast end.
///
/// No amount of cleverness fixes this from the audio alone — the two readings
/// are genuinely equally consistent with it, which is why listeners sometimes
/// disagree about a track's tempo too. What resolves it is knowing what kind of
/// music it is, and the person importing the song already does. So they can say
/// so, and the prior moves to where they said.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TempoHint {
    centre_bpm: f64,
}

impl TempoHint {
    /// Named starting points, for a menu. The tempo is still detected from the
    /// audio — these only move where the tie is broken, so a house track
    /// imported as "drum and bass" still reports its own tempo.
    pub const PRESETS: [(&'static str, f64); 5] = [
        ("Auto", PRIOR_CENTRE_BPM),
        ("Slow — ballad, boom bap", 80.0),
        ("Moderate — pop, rock, house", 120.0),
        ("Fast — trap, techno, dubstep", 145.0),
        ("Very fast — drum & bass, hardcore", 174.0),
    ];

    /// A hint centred on `bpm`, clamped to the searchable range.
    #[must_use]
    pub fn around(bpm: f64) -> Self {
        Self {
            centre_bpm: if bpm.is_finite() {
                bpm.clamp(MIN_BPM, MAX_BPM)
            } else {
                PRIOR_CENTRE_BPM
            },
        }
    }

    /// Where the prior is centred.
    #[must_use]
    pub fn centre_bpm(self) -> f64 {
        self.centre_bpm
    }
}

impl Default for TempoHint {
    fn default() -> Self {
        Self::around(PRIOR_CENTRE_BPM)
    }
}
/// How strongly the tracker resists straying from the estimated period.
const TIGHTNESS: f64 = 380.0;
/// Resolution of the tempo search, in envelope frames.
const TEMPO_SEARCH_STEP: f64 = 0.1;

#[derive(Clone, Debug)]
pub struct BeatAnalysis {
    pub bpm: f64,
    /// Beat positions in seconds.
    pub beat_times: Vec<f64>,
    /// Index into `beat_times` of the first downbeat.
    pub downbeat_index: usize,
    /// Mean onset strength on the chosen beats. Above ~0.5 the pulse is
    /// clear; near zero the material is probably unmetred.
    pub confidence: f32,
}

impl BeatAnalysis {
    /// A flat result for material with no usable pulse.
    #[must_use]
    pub fn unusable(bpm: f64) -> Self {
        Self {
            bpm,
            beat_times: Vec::new(),
            downbeat_index: 0,
            confidence: 0.0,
        }
    }

    #[must_use]
    pub fn is_usable(&self) -> bool {
        self.beat_times.len() >= 4 && self.confidence > 0.05
    }

    /// Seconds of the first downbeat.
    #[must_use]
    pub fn first_downbeat(&self) -> f64 {
        self.beat_times
            .get(self.downbeat_index)
            .copied()
            .unwrap_or(0.0)
    }
}

/// How many multiples of a candidate period are scored, and how much each
/// counts. A real beat period has energy at every multiple; a period that is
/// three quarters of the real one lines up only occasionally, which is exactly
/// the mistake plain autocorrelation makes on shuffled material.
const HARMONIC_WEIGHTS: [f64; 4] = [1.0, 0.5, 0.25, 0.125];

/// Estimates a single global tempo from an onset envelope.
#[must_use]
pub fn estimate_tempo(envelope: &OnsetEnvelope) -> f64 {
    estimate_tempo_with(envelope, TempoHint::default())
}

/// Estimates a single global tempo, told roughly where to expect it.
///
/// See [`TempoHint`] for why the hint is the only thing that can settle a
/// tempo against its double.
#[must_use]
pub fn estimate_tempo_with(envelope: &OnsetEnvelope, hint: TempoHint) -> f64 {
    let values = &envelope.values;
    let fps = envelope.frames_per_second;
    if values.len() < 32 || fps <= 0.0 {
        return hint.centre_bpm();
    }

    let min_lag = ((fps * 60.0 / MAX_BPM).floor() as usize).max(1);
    let max_lag = ((fps * 60.0 / MIN_BPM).ceil() as usize).min(values.len() / 2);
    if max_lag <= min_lag {
        return hint.centre_bpm();
    }

    // Autocorrelate once, out to the highest multiple any candidate needs.
    let correlation = autocorrelation(values, max_lag * HARMONIC_WEIGHTS.len());

    // Search fractional periods. Whole-frame candidates alone would be biased:
    // a real period is rarely a whole number of frames, but its double
    // sometimes is, and that candidate would then line up exactly with every
    // harmonic and win on an artefact of the hop size rather than on evidence.
    let mut best_period = min_lag as f64;
    let mut best_score = f64::NEG_INFINITY;
    let mut period = min_lag as f64;
    while period <= max_lag as f64 {
        let score = harmonic_score(&correlation, period) * tempo_prior(fps * 60.0 / period, hint);
        if score > best_score {
            best_score = score;
            best_period = period;
        }
        period += TEMPO_SEARCH_STEP;
    }

    (fps * 60.0 / best_period).clamp(MIN_BPM, MAX_BPM)
}

/// Reads the autocorrelation at a fractional lag.
fn interpolate(correlation: &[f64], position: f64) -> f64 {
    if position <= 0.0 {
        return 0.0;
    }
    let index = position as usize;
    let fraction = position - index as f64;
    let lower = correlation.get(index).copied().unwrap_or(0.0);
    let upper = correlation.get(index + 1).copied().unwrap_or(0.0);
    lower + (upper - lower) * fraction
}

/// Normalised autocorrelation of the onset envelope for every lag up to
/// `max_lag`. Dividing by the overlap length keeps long lags comparable with
/// short ones instead of decaying purely from having fewer terms.
fn autocorrelation(values: &[f32], max_lag: usize) -> Vec<f64> {
    (0..=max_lag)
        .map(|lag| {
            if lag >= values.len() {
                return 0.0;
            }
            values
                .iter()
                .zip(values.iter().skip(lag))
                .map(|(now, later)| f64::from(now * later))
                .sum::<f64>()
                / (values.len() - lag) as f64
        })
        .collect()
}

/// Sums the autocorrelation at a candidate period and its multiples.
fn harmonic_score(correlation: &[f64], period: f64) -> f64 {
    HARMONIC_WEIGHTS
        .iter()
        .enumerate()
        .map(|(index, weight)| {
            interpolate(correlation, period * (index + 1) as f64).max(0.0) * weight
        })
        .sum()
}

/// Log-normal weighting: tempi near the hint's centre are more likely a
/// priori, which breaks the tie between a tempo and its double.
fn tempo_prior(bpm: f64, hint: TempoHint) -> f64 {
    let octaves = (bpm / hint.centre_bpm()).log2() / PRIOR_WIDTH_OCTAVES;
    (-0.5 * octaves * octaves).exp()
}

/// Finds beat positions for a known tempo.
///
/// The dynamic program maximises the total onset strength on beats, minus a
/// penalty for every interval that departs from the expected period.
#[must_use]
pub fn track_beats(envelope: &OnsetEnvelope, bpm: f64) -> BeatAnalysis {
    let values = &envelope.values;
    let fps = envelope.frames_per_second;
    if values.len() < 8 || fps <= 0.0 || bpm <= 0.0 {
        return BeatAnalysis::unusable(bpm);
    }

    let period = fps * 60.0 / bpm;
    if period < 2.0 || period as usize >= values.len() {
        return BeatAnalysis::unusable(bpm);
    }

    // Candidate predecessors span half to double the period, which lets the
    // tracker skip a missed beat without abandoning the pulse.
    let earliest = (period * 0.5).round().max(1.0) as usize;
    let latest = (period * 2.0).round() as usize;

    let mut score = vec![f64::NEG_INFINITY; values.len()];
    let mut backlink = vec![usize::MAX; values.len()];

    for frame in 0..values.len() {
        let strength = f64::from(values[frame]);
        let mut best = 0.0;
        let mut best_previous = usize::MAX;
        let lower = frame.saturating_sub(latest);
        let upper = frame.saturating_sub(earliest);
        for (previous, previous_score) in score.iter().enumerate().take(upper + 1).skip(lower) {
            if previous >= frame || *previous_score == f64::NEG_INFINITY {
                continue;
            }
            let interval = (frame - previous) as f64;
            let deviation = (interval / period).ln();
            let candidate = previous_score - TIGHTNESS * deviation * deviation;
            if candidate > best || best_previous == usize::MAX {
                best = candidate;
                best_previous = previous;
            }
        }
        score[frame] = strength
            + if best_previous == usize::MAX {
                0.0
            } else {
                best
            };
        backlink[frame] = best_previous;
    }

    // Start the backtrace from the best score in the final period, so the
    // chain is not cut short by a fade-out.
    let tail_start = values.len().saturating_sub(latest.max(1));
    let Some(mut cursor) =
        (tail_start..values.len()).max_by(|left, right| score[*left].total_cmp(&score[*right]))
    else {
        return BeatAnalysis::unusable(bpm);
    };

    let mut frames = Vec::new();
    while cursor != usize::MAX {
        frames.push(cursor);
        let next = backlink[cursor];
        if next == usize::MAX || next >= cursor {
            break;
        }
        cursor = next;
    }
    frames.reverse();

    if frames.len() < 4 {
        return BeatAnalysis::unusable(bpm);
    }

    let confidence = frames.iter().map(|frame| values[*frame]).sum::<f32>() / frames.len() as f32;
    let beat_times: Vec<f64> = frames
        .iter()
        .map(|frame| envelope.seconds_at(*frame))
        .collect();
    let downbeat_index = estimate_downbeat(values, &frames, 4);

    BeatAnalysis {
        bpm,
        beat_times,
        downbeat_index,
        confidence,
    }
}

/// Picks which beat of the bar carries the most weight.
///
/// Bar one starts on the strongest recurring accent; summing onset strength
/// over each candidate phase finds it without needing harmonic information.
fn estimate_downbeat(values: &[f32], frames: &[usize], beats_per_bar: usize) -> usize {
    if beats_per_bar == 0 || frames.is_empty() {
        return 0;
    }
    (0..beats_per_bar.min(frames.len()))
        .max_by(|left, right| {
            let strength = |phase: &usize| -> f32 {
                frames
                    .iter()
                    .skip(*phase)
                    .step_by(beats_per_bar)
                    .map(|frame| values.get(*frame).copied().unwrap_or(0.0))
                    .sum()
            };
            strength(left).total_cmp(&strength(right))
        })
        .unwrap_or(0)
}

/// Runs the whole chain: onsets, tempo, beats.
#[must_use]
pub fn analyse(envelope: &OnsetEnvelope) -> BeatAnalysis {
    analyse_with(envelope, TempoHint::default())
}

/// Estimates the tempo and tracks the beats, told roughly where to expect the
/// tempo. See [`TempoHint`].
#[must_use]
pub fn analyse_with(envelope: &OnsetEnvelope, hint: TempoHint) -> BeatAnalysis {
    let bpm = estimate_tempo_with(envelope, hint);
    track_beats(envelope, bpm)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An onset envelope with a spike every `period` frames.
    fn pulse_envelope(period: usize, count: usize, fps: f64) -> OnsetEnvelope {
        let mut values = vec![0.0_f32; period * count];
        for beat in 0..count {
            if let Some(slot) = values.get_mut(beat * period) {
                // Accent every fourth beat, as a bar would.
                *slot = if beat % 4 == 0 { 3.0 } else { 2.0 };
            }
        }
        OnsetEnvelope {
            values,
            frames_per_second: fps,
        }
    }

    #[test]
    fn a_steady_pulse_yields_its_own_tempo() {
        // 93.75 fps, a spike every 30 frames = 187.5 BPM... too fast for the
        // prior; use 47 frames, which is ~120 BPM.
        let fps = 93.75;
        let envelope = pulse_envelope(47, 60, fps);
        let bpm = estimate_tempo(&envelope);
        let expected = fps * 60.0 / 47.0;
        assert!(
            (bpm - expected).abs() < 1.5,
            "estimated {bpm:.2}, expected about {expected:.2}"
        );
    }

    #[test]
    fn tracked_beats_land_on_the_pulse() {
        let fps = 93.75;
        let period = 47;
        let envelope = pulse_envelope(period, 60, fps);
        let analysis = analyse(&envelope);
        assert!(analysis.is_usable());
        assert!(
            analysis.beat_times.len() > 40,
            "only found {} beats",
            analysis.beat_times.len()
        );
        for time in &analysis.beat_times {
            let frame = time * fps;
            let offset = (frame / period as f64 - (frame / period as f64).round()).abs();
            assert!(
                offset * (period as f64) < 1.5,
                "beat at {time:.3}s is off the pulse"
            );
        }
    }

    #[test]
    fn intervals_are_regular_rather_than_jittery() {
        let envelope = pulse_envelope(47, 60, 93.75);
        let analysis = analyse(&envelope);
        let intervals: Vec<f64> = analysis
            .beat_times
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .collect();
        let mean = intervals.iter().sum::<f64>() / intervals.len() as f64;
        let worst = intervals
            .iter()
            .map(|interval| (interval - mean).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            worst < mean * 0.05,
            "beat spacing wandered by {:.1}% of the period",
            worst / mean * 100.0
        );
    }

    #[test]
    fn the_prior_prevents_choosing_double_tempo() {
        // A pulse at 240 BPM has strong autocorrelation at both 240 and 120;
        // the prior must land on the musical answer.
        let fps = 93.75;
        let envelope = pulse_envelope(23, 120, fps);
        let bpm = estimate_tempo(&envelope);
        assert!(
            (60.0..=140.0).contains(&bpm),
            "chose {bpm:.1} BPM instead of the halved tempo"
        );
    }

    #[test]
    fn the_accented_beat_becomes_the_downbeat() {
        let envelope = pulse_envelope(47, 64, 93.75);
        let analysis = analyse(&envelope);
        // Beats were accented every fourth starting at zero, so whichever beat
        // the tracker starts on, the downbeat phase must be a multiple of four
        // away from the first accent.
        assert!(analysis.downbeat_index < 4);
    }

    #[test]
    fn silence_is_reported_as_unusable_rather_than_guessed() {
        let envelope = OnsetEnvelope {
            values: vec![0.0; 4_000],
            frames_per_second: 93.75,
        };
        assert!(!track_beats(&envelope, 120.0).is_usable());
    }

    #[test]
    fn empty_and_tiny_inputs_do_not_panic() {
        let empty = OnsetEnvelope {
            values: Vec::new(),
            frames_per_second: 93.75,
        };
        assert!(!analyse(&empty).is_usable());
        let tiny = OnsetEnvelope {
            values: vec![1.0, 0.0, 1.0],
            frames_per_second: 93.75,
        };
        assert!(!analyse(&tiny).is_usable());
    }

    #[test]
    fn a_hint_settles_a_tempo_against_its_double() {
        // 93.75 fps; a spike every 15 frames is 375 BPM, every 30 is 187.5.
        // Use 32 frames: 175.8 BPM, which the default prior halves.
        let envelope = pulse_envelope(32, 80, 93.75);
        let default = estimate_tempo(&envelope);
        let fast = estimate_tempo_with(&envelope, TempoHint::around(174.0));
        assert!(
            default < 110.0,
            "the default prior should read this as halved, got {default:.1}"
        );
        assert!(
            fast > 160.0,
            "a fast hint should read the true tempo, got {fast:.1}"
        );
    }

    #[test]
    fn a_hint_does_not_override_the_audio() {
        // A hint moves where ties are broken; it cannot invent a tempo the
        // evidence does not support. 120 BPM material stays 120 under a hint
        // one octave away only because 60 is a real alternative reading — so
        // check a hint in the same octave changes nothing.
        let envelope = pulse_envelope(47, 60, 93.75);
        let plain = estimate_tempo(&envelope);
        let nudged = estimate_tempo_with(&envelope, TempoHint::around(130.0));
        assert!(
            (plain - nudged).abs() < 1.0,
            "a hint near the truth must not move it: {plain:.1} then {nudged:.1}"
        );
    }

    #[test]
    fn the_default_hint_is_the_original_prior() {
        // The whole corpus was measured against this; it must not drift.
        assert!((TempoHint::default().centre_bpm() - PRIOR_CENTRE_BPM).abs() < f64::EPSILON);
        assert!((TempoHint::PRESETS[0].1 - PRIOR_CENTRE_BPM).abs() < f64::EPSILON);
    }

    #[test]
    fn a_hint_is_clamped_to_the_searchable_range() {
        assert!((TempoHint::around(5.0).centre_bpm() - MIN_BPM).abs() < f64::EPSILON);
        assert!((TempoHint::around(9_000.0).centre_bpm() - MAX_BPM).abs() < f64::EPSILON);
        assert!((TempoHint::around(f64::NAN).centre_bpm() - PRIOR_CENTRE_BPM).abs() < f64::EPSILON);
    }

    #[test]
    fn the_prior_is_strongest_at_its_centre() {
        let hint = TempoHint::default();
        assert!(tempo_prior(120.0, hint) > tempo_prior(60.0, hint));
        assert!(tempo_prior(120.0, hint) > tempo_prior(240.0, hint));
        assert!((tempo_prior(120.0, hint) - 1.0).abs() < 1e-12);
        // And the peak follows the hint.
        let fast = TempoHint::around(174.0);
        assert!(tempo_prior(174.0, fast) > tempo_prior(87.0, fast));
    }
}
