#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

//! The small signal operations the mastering stages are built from.
//!
//! Levels are accumulated in `f64` even though the audio itself is `f32`. A
//! five-minute mix is fifteen million samples, and summing that many squares
//! into a 24-bit mantissa loses the quiet ones entirely — the RMS the whole
//! match is steered by would drift with the length of the song.

/// Splits interleaved stereo into mid and side.
///
/// `mid = (l + r) / 2`, `side = (l - r) / 2`. Halving both is what makes
/// [`ms_to_lr`] an exact inverse.
#[must_use]
pub fn lr_to_ms(frames: &[[f32; 2]]) -> (Vec<f32>, Vec<f32>) {
    let mut mid = Vec::with_capacity(frames.len());
    let mut side = Vec::with_capacity(frames.len());
    for frame in frames {
        mid.push((frame[0] + frame[1]) * 0.5);
        side.push((frame[0] - frame[1]) * 0.5);
    }
    (mid, side)
}

/// Recombines mid and side into interleaved stereo.
#[must_use]
pub fn ms_to_lr(mid: &[f32], side: &[f32]) -> Vec<[f32; 2]> {
    mid.iter()
        .zip(side.iter())
        .map(|(m, s)| [m + s, m - s])
        .collect()
}

/// Root mean square of a signal.
#[must_use]
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|s| f64::from(*s) * f64::from(*s)).sum();
    (sum / samples.len() as f64).sqrt() as f32
}

/// The RMS of each equal-length piece, in order.
///
/// The tail that does not fill a whole piece is dropped, matching the way the
/// analysis divides a song into `divisions` pieces of `piece_size` samples.
#[must_use]
pub fn batch_rms(samples: &[f32], piece_size: usize, divisions: usize) -> Vec<f32> {
    if piece_size == 0 {
        return Vec::new();
    }
    (0..divisions)
        .filter_map(|piece| samples.get(piece * piece_size..(piece + 1) * piece_size))
        .map(rms)
        .collect()
}

/// The largest absolute sample.
#[must_use]
pub fn peak(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0_f32, |worst, s| worst.max(s.abs()))
}

/// The largest absolute sample across both channels.
#[must_use]
pub fn peak_stereo(frames: &[[f32; 2]]) -> f32 {
    frames
        .iter()
        .fold(0.0_f32, |worst, f| worst.max(f[0].abs()).max(f[1].abs()))
}

/// Scales every sample by `gain`.
pub fn amplify(samples: &mut [f32], gain: f32) {
    for sample in samples {
        *sample *= gain;
    }
}

/// Scales every frame by `gain`.
pub fn amplify_stereo(frames: &mut [[f32; 2]], gain: f32) {
    for frame in frames {
        frame[0] *= gain;
        frame[1] *= gain;
    }
}

/// Brings the peak down to `threshold`, returning the divisor that was used.
///
/// When `normalize_clipped` is false a signal already below the threshold is
/// left alone and the coefficient is `1`: the reference is only ever turned
/// down to make room, never turned up.
#[must_use]
pub fn normalize(frames: &mut [[f32; 2]], threshold: f32, epsilon: f32, clipped: bool) -> f32 {
    let max = peak_stereo(frames);
    let mut coefficient = 1.0;
    if max < threshold || clipped {
        coefficient = epsilon.max(max / threshold);
    }
    if (coefficient - 1.0).abs() > f32::EPSILON {
        let inverse = 1.0 / coefficient;
        amplify_stereo(frames, inverse);
    }
    coefficient
}

/// Clamps to +/- 1, the way a converter would.
#[must_use]
pub fn clip(samples: &[f32]) -> Vec<f32> {
    samples.iter().map(|s| s.clamp(-1.0, 1.0)).collect()
}

/// The per-frame envelope the limiter reduces: how far each frame exceeds the
/// threshold, as a ratio of at least `1`.
#[must_use]
pub fn rectify(frames: &[[f32; 2]], threshold: f32) -> Vec<f32> {
    frames
        .iter()
        .map(|frame| {
            let loudest = frame[0].abs().max(frame[1].abs());
            loudest.max(threshold) / threshold
        })
        .collect()
}

/// `1 - x`, applied in place. The limiter works on "how much to take away"
/// rather than "how much to keep", because the running maximum of several
/// reductions is the one that wins.
pub fn flip(values: &mut [f32]) {
    for value in values {
        *value = 1.0 - *value;
    }
}

/// Element-wise maximum of `source` into `into`.
pub fn max_into(into: &mut [f32], source: &[f32]) {
    for (target, value) in into.iter_mut().zip(source.iter()) {
        *target = target.max(*value);
    }
}

/// How many pieces a signal of `length` divides into, and how long each is.
///
/// Mirrors the reference implementation exactly, including the deliberate
/// `+ 1`: a song shorter than one maximum piece still yields one division.
#[must_use]
pub fn piece_sizes(length: usize, max_piece_size: usize) -> (usize, usize) {
    if max_piece_size == 0 || length == 0 {
        return (1, length);
    }
    let divisions = length / max_piece_size + 1;
    (divisions, length / divisions)
}

/// The indices of the pieces at or above the average, and the RMS across them.
///
/// This is what "the loudest parts of the song" means throughout: the chorus
/// rather than the intro, so a fade-in cannot drag the match down.
#[must_use]
pub fn loudest_pieces(rmses: &[f32], average: f32) -> (Vec<usize>, f32) {
    let indices: Vec<usize> = rmses
        .iter()
        .enumerate()
        .filter(|(_, value)| **value >= average)
        .map(|(index, _)| index)
        .collect();
    let loudest: Vec<f32> = indices.iter().map(|index| rmses[*index]).collect();
    let match_rms = rms(&loudest);
    (indices, match_rms)
}

/// Concatenates the chosen pieces of a signal into one buffer.
#[must_use]
pub fn gather_pieces(samples: &[f32], indices: &[usize], piece_size: usize) -> Vec<f32> {
    let mut gathered = Vec::with_capacity(indices.len() * piece_size);
    for index in indices {
        if let Some(piece) = samples.get(index * piece_size..(index + 1) * piece_size) {
            gathered.extend_from_slice(piece);
        }
    }
    gathered
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mid_side_round_trips() {
        let frames = [[0.5_f32, -0.25], [-0.75, 0.125], [0.0, 0.0], [1.0, 1.0]];
        let (mid, side) = lr_to_ms(&frames);
        let back = ms_to_lr(&mid, &side);
        for (original, restored) in frames.iter().zip(back.iter()) {
            assert!((original[0] - restored[0]).abs() < 1e-6);
            assert!((original[1] - restored[1]).abs() < 1e-6);
        }
    }

    #[test]
    fn a_centred_signal_has_no_side_content() {
        let frames = [[0.5_f32, 0.5], [-0.3, -0.3]];
        let (mid, side) = lr_to_ms(&frames);
        assert!((mid[0] - 0.5).abs() < 1e-6);
        assert!(side.iter().all(|s| s.abs() < 1e-6), "mono has no side");
    }

    #[test]
    fn rms_of_a_full_scale_square_is_one() {
        let square: Vec<f32> = (0..1000)
            .map(|index| if index % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        assert!((rms(&square) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rms_of_a_sine_is_the_reciprocal_root_of_two() {
        let sine: Vec<f32> = (0..44_100)
            .map(|index| (index as f32 * 0.01).sin())
            .collect();
        assert!((rms(&sine) - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-3);
    }

    #[test]
    fn only_pieces_at_or_above_the_average_are_loudest() {
        let rmses = [0.1_f32, 0.9, 0.5, 0.05];
        let average = rms(&rmses);
        let (indices, match_rms) = loudest_pieces(&rmses, average);
        assert!(indices.contains(&1), "the loud piece must be included");
        assert!(!indices.contains(&3), "the quiet piece must not be");
        assert!(match_rms >= average);
    }

    #[test]
    fn a_song_shorter_than_one_piece_still_divides_once() {
        let (divisions, piece_size) = piece_sizes(1000, 44_100 * 15);
        assert_eq!(divisions, 1);
        assert_eq!(piece_size, 1000);
    }

    #[test]
    fn rectify_reports_how_far_over_the_threshold_a_frame_is() {
        let frames = [[0.5_f32, 0.0], [1.0, 0.0]];
        let rectified = rectify(&frames, 0.5);
        assert!(
            (rectified[0] - 1.0).abs() < 1e-6,
            "at the threshold is unity"
        );
        assert!((rectified[1] - 2.0).abs() < 1e-6, "twice over reads two");
    }

    #[test]
    fn normalizing_a_quiet_reference_brings_its_peak_to_the_threshold() {
        let mut frames = [[0.25_f32, -0.25], [0.5, 0.5]];
        let coefficient = normalize(&mut frames, 0.999, 1e-6, false);
        assert!(coefficient < 1.0, "a quiet signal is turned up");
        assert!((peak_stereo(&frames) - 0.999).abs() < 1e-4);
    }
}
