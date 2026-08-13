#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

//! The Hyrax brickwall limiter.
//!
//! A limiter that simply divided every peak down by its own excess would
//! distort: the gain would jump sample to sample, and a gain that jumps is
//! modulation. So the reduction is computed as an envelope and then smoothed
//! three ways, and the *largest* of the three is what gets applied.
//!
//! - The **hard clip** curve is the raw requirement: exactly enough reduction,
//!   applied exactly where it is needed.
//! - The **attack** curve reaches the reduction slightly before the peak
//!   arrives, so the gain is already down when it lands. It is filtered
//!   forwards and backwards, which is the only way to move an envelope
//!   *earlier* in time.
//! - The **release** curve holds the reduction after the peak and lets it back
//!   up slowly, so a run of peaks is ridden at a steady level instead of the
//!   gain pumping between each one.
//!
//! Taking the maximum of the three means the loudest demand always wins: the
//! ceiling is never exceeded, and the smoothing can only ever ask for *more*
//! reduction than the hard clip, never less.
//!
//! Ported from Matchering's `limiter/hyrax.py` (GPL-3.0, © 2016-2022 Sergree).

use crate::dsp;

/// How the limiter behaves. Times are milliseconds; the filter coefficients
/// are the reference implementation's, and are not in any natural unit.
#[derive(Clone, Copy, Debug)]
pub struct LimiterConfig {
    pub attack_ms: f32,
    pub hold_ms: f32,
    pub release_ms: f32,
    pub attack_filter_coefficient: f32,
    pub hold_filter_hz: f32,
    pub release_filter_coefficient: f32,
}

impl Default for LimiterConfig {
    fn default() -> Self {
        Self {
            attack_ms: 1.0,
            hold_ms: 1.0,
            release_ms: 3_000.0,
            attack_filter_coefficient: -2.0,
            hold_filter_hz: 7.0,
            release_filter_coefficient: 800.0,
        }
    }
}

/// Applies the limiter in place, holding every frame at or below `threshold`.
pub fn limit(frames: &mut [[f32; 2]], threshold: f32, sample_rate: f32, config: &LimiterConfig) {
    if frames.is_empty() || sample_rate <= 0.0 {
        return;
    }

    let rectified = dsp::rectify(frames, threshold);
    // Nothing reaches the ceiling, so there is nothing to take away.
    if rectified.iter().all(|value| (value - 1.0).abs() < 1e-9) {
        return;
    }

    // Work in "how much to remove" rather than "how much to keep": the running
    // maximum of several reductions is then the one that wins.
    let mut hard_clip: Vec<f32> = rectified.into_iter().map(|value| 1.0 / value).collect();
    dsp::flip(&mut hard_clip);

    let attack_samples = ms_to_samples(config.attack_ms, sample_rate).max(1);
    let hold_samples = ms_to_samples(config.hold_ms, sample_rate).max(1);

    let slided = sliding_max(&hard_clip, make_odd(attack_samples), Mode::Attack);
    let attack = {
        let coefficient = (config.attack_filter_coefficient / attack_samples as f32).exp();
        filtfilt_one_pole(&slided, coefficient)
    };
    let release = process_release(&slided, hold_samples, sample_rate, config);

    let mut gain = hard_clip;
    dsp::max_into(&mut gain, &attack);
    dsp::max_into(&mut gain, &release);
    dsp::flip(&mut gain);

    for (frame, factor) in frames.iter_mut().zip(gain.iter()) {
        frame[0] *= factor;
        frame[1] *= factor;
    }
}

/// The hold-and-release half of the envelope.
fn process_release(
    slided: &[f32],
    hold_samples: usize,
    sample_rate: f32,
    config: &LimiterConfig,
) -> Vec<f32> {
    let held = sliding_max(slided, hold_samples, Mode::Hold);

    let hold_filter = Butterworth::low_pass(config.hold_filter_hz, sample_rate);
    let hold_output = hold_filter.apply(&held);

    let release_filter = Butterworth::low_pass(
        config.release_filter_coefficient / config.release_ms,
        sample_rate,
    );
    let driven: Vec<f32> = held
        .iter()
        .zip(hold_output.iter())
        .map(|(a, b)| a.max(*b))
        .collect();
    let release_output = release_filter.apply(&driven);

    hold_output
        .iter()
        .zip(release_output.iter())
        .map(|(a, b)| a.max(*b))
        .collect()
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// Centred: the reduction spreads both before and after the peak.
    Attack,
    /// Backward-looking: the reduction is carried forward from the peak.
    Hold,
}

/// Running maximum over a sliding window, in one pass.
///
/// A monotonic deque holds the indices of the candidates that could still win:
/// anything smaller than a newer arrival can never be the maximum again and is
/// dropped on the spot, so each sample is pushed and popped at most once.
fn sliding_max(values: &[f32], window: usize, mode: Mode) -> Vec<f32> {
    let n = values.len();
    if n == 0 || window == 0 {
        return values.to_vec();
    }

    // Attack looks `window - 1` samples either side; hold looks only behind.
    let (behind, ahead) = match mode {
        Mode::Attack => (window - 1, window - 1),
        Mode::Hold => (window - 1, 0),
    };

    let mut result = vec![0.0_f32; n];
    let mut candidates: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    let mut next = 0_usize;

    for (index, slot) in result.iter_mut().enumerate() {
        let last = (index + ahead).min(n - 1);
        while next <= last {
            while candidates
                .back()
                .is_some_and(|back| values[*back] <= values[next])
            {
                candidates.pop_back();
            }
            candidates.push_back(next);
            next += 1;
        }
        // Anything that has fallen off the back of the window.
        let first = index.saturating_sub(behind);
        while candidates.front().is_some_and(|front| *front < first) {
            candidates.pop_front();
        }
        *slot = candidates.front().map_or(0.0, |front| values[*front]);
    }

    result
}

/// A one-pole low-pass run forwards and then backwards, so the envelope it
/// produces has no phase lag — which is what lets the attack curve begin
/// before the peak it is reducing.
fn filtfilt_one_pole(values: &[f32], coefficient: f32) -> Vec<f32> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }

    // Odd extension at both ends, reflected through the endpoint value. Without
    // it the filter starts from nothing and carves a dip into the first
    // milliseconds of the song.
    let pad = 6.min(n - 1);
    let mut extended = Vec::with_capacity(n + 2 * pad);
    for index in (1..=pad).rev() {
        extended.push(2.0 * values[0] - values[index]);
    }
    extended.extend_from_slice(values);
    for index in 1..=pad {
        extended.push(2.0 * values[n - 1] - values[n - 1 - index]);
    }

    one_pole(&mut extended, coefficient);
    extended.reverse();
    one_pole(&mut extended, coefficient);
    extended.reverse();

    extended[pad..pad + n].to_vec()
}

/// `y[n] = (1 - c) * x[n] + c * y[n - 1]`, started at the steady state for the
/// first sample rather than at zero.
fn one_pole(values: &mut [f32], coefficient: f32) {
    let gain = 1.0 - coefficient;
    let mut state = coefficient * values[0];
    for value in values.iter_mut() {
        let output = gain * *value + state;
        state = coefficient * output;
        *value = output;
    }
}

/// A first-order Butterworth low-pass, bilinear-transformed with the cutoff
/// pre-warped so the digital corner lands where the analogue one was asked for.
struct Butterworth {
    feedforward: f32,
    feedback: f32,
}

impl Butterworth {
    fn low_pass(cutoff_hz: f32, sample_rate: f32) -> Self {
        let warped = (std::f32::consts::PI * cutoff_hz / sample_rate).tan();
        Self {
            feedforward: warped / (1.0 + warped),
            feedback: (warped - 1.0) / (1.0 + warped),
        }
    }

    /// One forward pass, from rest.
    fn apply(&self, values: &[f32]) -> Vec<f32> {
        let mut previous_input = 0.0;
        let mut previous_output = 0.0;
        values
            .iter()
            .map(|input| {
                let output =
                    self.feedforward * (input + previous_input) - self.feedback * previous_output;
                previous_input = *input;
                previous_output = output;
                output
            })
            .collect()
    }
}

fn ms_to_samples(milliseconds: f32, sample_rate: f32) -> usize {
    (sample_rate * milliseconds * 1e-3) as usize
}

const fn make_odd(value: usize) -> usize {
    if value % 2 == 0 { value + 1 } else { value }
}

#[cfg(test)]
mod tests {
    // A sliding maximum returns one of its inputs unchanged, so comparing to
    // that exact value is the assertion rather than an approximation.
    #![allow(clippy::float_cmp)]

    use super::*;

    const RATE: f32 = 48_000.0;
    const THRESHOLD: f32 = 0.998_138;

    fn config() -> LimiterConfig {
        LimiterConfig::default()
    }

    #[test]
    fn a_signal_under_the_ceiling_is_untouched() {
        let mut frames: Vec<[f32; 2]> = (0..4_800)
            .map(|index| {
                let value = (index as f32 * 0.01).sin() * 0.5;
                [value, value]
            })
            .collect();
        let before = frames.clone();
        limit(&mut frames, THRESHOLD, RATE, &config());
        assert_eq!(frames, before, "nothing reaches the ceiling");
    }

    #[test]
    fn nothing_exceeds_the_ceiling_afterwards() {
        // A quiet bed with loud stabs, which is what a limiter is for.
        let mut frames: Vec<[f32; 2]> = (0..48_000)
            .map(|index| {
                let base = (index as f32 * 0.02).sin() * 0.3;
                let stab = if index % 8_000 < 200 { 2.5 } else { 0.0 };
                let value = base + stab * (index as f32 * 0.5).sin();
                [value, value]
            })
            .collect();
        limit(&mut frames, THRESHOLD, RATE, &config());
        let peak = dsp::peak_stereo(&frames);
        assert!(
            peak <= THRESHOLD * 1.001,
            "peak {peak} should sit at or below {THRESHOLD}"
        );
    }

    #[test]
    fn the_quiet_parts_keep_their_level() {
        // A limiter that turned the whole song down would pass the ceiling
        // test while being useless.
        let quiet = 0.2_f32;
        let mut frames: Vec<[f32; 2]> = (0..48_000)
            .map(|index| {
                let value = if index == 24_000 { 3.0 } else { quiet };
                [value, value]
            })
            .collect();
        limit(&mut frames, THRESHOLD, RATE, &config());
        // Far from the peak — beyond attack, hold and most of the release —
        // the level must be back where it started.
        assert!(
            (frames[100][0] - quiet).abs() < 0.01,
            "start should be untouched, got {}",
            frames[100][0]
        );
    }

    #[test]
    fn gain_reduction_begins_before_the_peak_lands() {
        // The point of the zero-phase attack filter: no overshoot on the way
        // in. A causal-only envelope would let the first sample through.
        let mut frames = vec![[0.1_f32, 0.1]; 10_000];
        frames[5_000] = [4.0, 4.0];
        limit(&mut frames, THRESHOLD, RATE, &config());
        assert!(
            frames[5_000][0] <= THRESHOLD * 1.001,
            "the peak itself must be caught: {}",
            frames[5_000][0]
        );
        assert!(
            frames[4_990][0] < 0.1,
            "reduction should already be under way ten samples earlier: {}",
            frames[4_990][0]
        );
    }

    #[test]
    fn the_envelope_recovers_after_a_peak() {
        let mut frames = vec![[0.5_f32, 0.5]; 48_000 * 4];
        frames[1_000] = [5.0, 5.0];
        limit(&mut frames, THRESHOLD, RATE, &config());
        let just_after = frames[2_000][0];
        let long_after = frames[48_000 * 3][0];
        assert!(
            long_after > just_after,
            "gain should come back up: {just_after} then {long_after}"
        );
        assert!(
            (long_after - 0.5).abs() < 0.02,
            "and return to unity, got {long_after}"
        );
    }

    #[test]
    fn a_sliding_maximum_sees_both_sides_on_attack() {
        let values = [0.0, 0.0, 5.0, 0.0, 0.0];
        let result = sliding_max(&values, 3, Mode::Attack);
        // Window 3 reaches two either side, so everything sees the spike.
        assert_eq!(result, vec![5.0; 5]);
    }

    #[test]
    fn a_sliding_maximum_only_looks_behind_on_hold() {
        let values = [0.0, 0.0, 5.0, 0.0, 0.0];
        let result = sliding_max(&values, 3, Mode::Hold);
        assert_eq!(result[0], 0.0, "before the spike nothing is held");
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 5.0, "the spike itself");
        assert_eq!(result[3], 5.0, "and it is carried forward");
        assert_eq!(result[4], 5.0);
    }

    #[test]
    fn a_sliding_maximum_matches_a_naive_scan() {
        let values: Vec<f32> = (0..500)
            .map(|index| ((index * 37) % 61) as f32 / 61.0)
            .collect();
        for window in [1_usize, 2, 7, 32] {
            let fast = sliding_max(&values, window, Mode::Hold);
            for (index, got) in fast.iter().enumerate() {
                let first = index.saturating_sub(window - 1);
                let expected = values[first..=index].iter().fold(0.0_f32, |a, b| a.max(*b));
                assert!(
                    (got - expected).abs() < 1e-9,
                    "window {window}, index {index}: {got} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn a_zero_phase_filter_does_not_shift_its_input() {
        // A symmetric bump must stay symmetric: that is what distinguishes
        // forward-and-backward filtering from a plain one-pole.
        let mut values = vec![0.0_f32; 401];
        for (index, value) in values.iter_mut().enumerate() {
            let distance = (index as f32 - 200.0).abs();
            *value = (-distance * distance / 800.0).exp();
        }
        let filtered = filtfilt_one_pole(&values, 0.9);
        for offset in 1..150 {
            let left = filtered[200 - offset];
            let right = filtered[200 + offset];
            assert!(
                (left - right).abs() < 1e-4,
                "offset {offset}: {left} vs {right} — the filter shifted the bump"
            );
        }
    }

    #[test]
    fn a_low_pass_passes_direct_current_and_stops_nyquist() {
        let filter = Butterworth::low_pass(7.0, RATE);
        let steady = filter.apply(&vec![1.0; 48_000]);
        assert!(
            (steady[47_999] - 1.0).abs() < 0.01,
            "DC should pass at unity, got {}",
            steady[47_999]
        );

        let alternating: Vec<f32> = (0..48_000)
            .map(|index| if index % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let stopped = filter.apply(&alternating);
        assert!(
            stopped[47_999].abs() < 0.01,
            "Nyquist should be rejected, got {}",
            stopped[47_999]
        );
    }

    #[test]
    fn empty_input_is_handled() {
        let mut frames: Vec<[f32; 2]> = Vec::new();
        limit(&mut frames, THRESHOLD, RATE, &config());
        assert!(frames.is_empty());
    }
}
