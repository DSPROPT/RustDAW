#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

//! The matching EQ: a linear-phase FIR built from the difference between two
//! spectra.
//!
//! The idea is simple and the details are all that matter. Average the spectrum
//! of the loudest parts of each song, divide one by the other, and you have the
//! curve that turns the target's tone into the reference's. Turning that curve
//! into a usable filter is the work: the raw ratio is spiky, and following it
//! literally builds a comb filter that rings.
//!
//! So the curve is resampled onto a logarithmic frequency grid before it is
//! smoothed. Ears hear frequency logarithmically — the octave from 100 Hz to
//! 200 Hz matters as much as the one from 5 kHz to 10 kHz — but an FFT's bins
//! are linear, so smoothing on the linear grid would leave the bass barely
//! touched and scrub the treble flat. On the log grid one smoothing width means
//! the same fraction of an octave everywhere.

use daw_analysis::fft::{self, Planned};

use crate::interp::Spline;
use crate::lowess;

/// The average magnitude spectrum across every whole frame of `samples`.
///
/// Boxcar-windowed and non-overlapping, matching the reference implementation.
/// The frames come from the loudest pieces of the song, which are already
/// discontinuous at their joins, so there is nothing for a window to preserve.
#[must_use]
pub fn average_spectrum(samples: &[f32], fft_size: usize) -> Vec<f32> {
    let bins = fft_size / 2 + 1;
    let Some(planned) = Planned::new(fft_size) else {
        return vec![0.0; bins];
    };

    let mut total = vec![0.0_f64; bins];
    let mut frames = 0_usize;
    let mut real = vec![0.0_f32; fft_size];
    let mut imag = vec![0.0_f32; fft_size];

    for frame in samples.chunks_exact(fft_size) {
        real.copy_from_slice(frame);
        imag.fill(0.0);
        planned.forward(&mut real, &mut imag);
        for (bin, sum) in total.iter_mut().enumerate() {
            *sum += f64::from(real[bin].hypot(imag[bin]));
        }
        frames += 1;
    }

    if frames == 0 {
        return vec![0.0; bins];
    }
    total
        .into_iter()
        .map(|sum| (sum / frames as f64) as f32)
        .collect()
}

/// Smooths the matching curve across a logarithmic frequency grid.
///
/// The scaling by sample rate cancels on the way there and back, so the grids
/// are built on the normalised interval instead — but the *shape* of the log
/// grid is what does the work, and that is preserved exactly.
#[must_use]
pub fn smooth_exponentially(matching: &[f32], config: &Config) -> Vec<f32> {
    let half = config.fft_size / 2;
    if matching.len() != half + 1 {
        return matching.to_vec();
    }

    let linear: Vec<f64> = (0..=half).map(|bin| bin as f64 / half as f64).collect();

    // From 4/fft_size of Nyquist up to Nyquist, spaced evenly in decades.
    let log_points = half * config.lin_log_oversampling + 1;
    let lowest = (4.0 / config.fft_size as f64).log10();
    let logarithmic: Vec<f64> = (0..log_points)
        .map(|index| {
            let decade = lowest * (1.0 - index as f64 / (log_points - 1) as f64);
            10.0_f64.powf(decade)
        })
        .collect();

    let values: Vec<f64> = matching.iter().map(|value| f64::from(*value)).collect();
    let Some(to_log) = Spline::new(&linear, &values) else {
        return matching.to_vec();
    };
    let on_log_grid = to_log.evaluate_all(&logarithmic);

    let positions: Vec<f64> = (0..on_log_grid.len())
        .map(|index| index as f64 / (on_log_grid.len() - 1) as f64)
        .collect();
    let smoothed = lowess::smooth(
        &positions,
        &on_log_grid,
        config.lowess_frac,
        config.lowess_iterations,
        config.lowess_delta,
    );

    let Some(back) = Spline::new(&logarithmic, &smoothed) else {
        return matching.to_vec();
    };
    let mut result: Vec<f32> = back
        .evaluate_all(&linear)
        .into_iter()
        .map(|value| value as f32)
        .collect();

    // DC is removed outright — a mastering EQ has no business with an offset —
    // and the first bin is restored from the unsmoothed curve, because the log
    // grid starts above it and everything there is extrapolation.
    result[0] = 0.0;
    result[1] = matching[1];
    result
}

/// Builds the linear-phase FIR that turns the target's tone into the
/// reference's.
#[must_use]
pub fn matching_fir(target: &[f32], reference: &[f32], config: &Config) -> Vec<f32> {
    let target_spectrum = average_spectrum(target, config.fft_size);
    let reference_spectrum = average_spectrum(reference, config.fft_size);

    let matching: Vec<f32> = reference_spectrum
        .iter()
        .zip(target_spectrum.iter())
        .map(|(reference, target)| reference / target.max(config.min_value))
        .collect();

    let smoothed = smooth_exponentially(&matching, config);

    // Back to the time domain, then rotated so the peak sits in the middle.
    // A spectrum with no phase information produces an impulse response split
    // across both ends of the buffer; centring it is what makes the filter
    // linear-phase rather than an echo.
    let impulse = fft::inverse_real(&smoothed);
    if impulse.is_empty() {
        return Vec::new();
    }
    let half = impulse.len() / 2;
    let window = fft::hann_window(impulse.len());
    (0..impulse.len())
        .map(|index| impulse[(index + half) % impulse.len()] * window[index])
        .collect()
}

/// Convolves `signal` with `fir`, returning the centre section so the result
/// is the same length as the input.
///
/// Overlap-add: the signal is processed in blocks whose transforms are long
/// enough to hold the tail of the filter, and the tails are summed into the
/// following block. Doing it in one transform would mean a buffer the length of
/// the song rounded up to a power of two.
#[must_use]
pub fn convolve_same(signal: &[f32], fir: &[f32]) -> Vec<f32> {
    if fir.is_empty() || signal.is_empty() {
        return signal.to_vec();
    }

    let transform_size = (2 * fir.len()).next_power_of_two();
    let block = transform_size - fir.len() + 1;
    let Some(planned) = Planned::new(transform_size) else {
        return signal.to_vec();
    };

    let mut fir_real = vec![0.0_f32; transform_size];
    let mut fir_imag = vec![0.0_f32; transform_size];
    fir_real[..fir.len()].copy_from_slice(fir);
    planned.forward(&mut fir_real, &mut fir_imag);

    let full_length = signal.len() + fir.len() - 1;
    let mut full = vec![0.0_f32; full_length];
    let mut real = vec![0.0_f32; transform_size];
    let mut imag = vec![0.0_f32; transform_size];

    for (index, chunk) in signal.chunks(block).enumerate() {
        real.fill(0.0);
        imag.fill(0.0);
        real[..chunk.len()].copy_from_slice(chunk);
        planned.forward(&mut real, &mut imag);

        for bin in 0..transform_size {
            let re = real[bin] * fir_real[bin] - imag[bin] * fir_imag[bin];
            let im = real[bin] * fir_imag[bin] + imag[bin] * fir_real[bin];
            real[bin] = re;
            imag[bin] = im;
        }
        planned.inverse(&mut real, &mut imag);

        let offset = index * block;
        for (position, value) in real.iter().enumerate() {
            if let Some(slot) = full.get_mut(offset + position) {
                *slot += value;
            }
        }
    }

    // "Same" is the centre of the full convolution, which is where the
    // centred impulse response puts the undelayed signal.
    let start = (fir.len() - 1) / 2;
    full[start..start + signal.len()].to_vec()
}

/// The parameters the matching EQ is built with.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub fft_size: usize,
    pub lin_log_oversampling: usize,
    pub lowess_frac: f64,
    pub lowess_iterations: usize,
    pub lowess_delta: f64,
    pub min_value: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            fft_size: 4_096,
            lin_log_oversampling: 4,
            lowess_frac: 0.0375,
            lowess_iterations: 0,
            lowess_delta: 0.001,
            min_value: 1e-6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn sine(frequency: f32, rate: f32, length: usize) -> Vec<f32> {
        (0..length)
            .map(|index| (TAU * frequency * index as f32 / rate).sin() * 0.5)
            .collect()
    }

    #[test]
    fn a_tone_lands_in_its_own_bin_of_the_average_spectrum() {
        let rate = 48_000.0;
        let signal = sine(1_000.0, rate, 4_096 * 8);
        let spectrum = average_spectrum(&signal, 4_096);
        let peak = spectrum
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(index, _)| index)
            .unwrap();
        let expected = (1_000.0 / rate * 4_096.0).round() as usize;
        assert!(
            peak.abs_diff(expected) <= 1,
            "peak at bin {peak}, expected about {expected}"
        );
    }

    #[test]
    fn silence_produces_a_flat_spectrum() {
        let spectrum = average_spectrum(&vec![0.0; 8_192], 4_096);
        assert!(spectrum.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn matching_a_signal_to_itself_gives_a_flat_response() {
        let rate = 48_000.0;
        let signal: Vec<f32> = (0..4_096 * 8)
            .map(|index| {
                let time = index as f32 / rate;
                (TAU * 220.0 * time).sin() * 0.3 + (TAU * 3_000.0 * time).sin() * 0.2
            })
            .collect();
        let config = Config::default();
        let fir = matching_fir(&signal, &signal, &config);
        assert_eq!(fir.len(), config.fft_size);

        // An identity match is a centred spike: convolving with it returns the
        // signal, so the tail either side of centre must be small.
        let centre = fir.len() / 2;
        let peak = fir[centre].abs();
        let worst_tail = fir
            .iter()
            .enumerate()
            .filter(|(index, _)| index.abs_diff(centre) > 8)
            .map(|(_, value)| value.abs())
            .fold(0.0_f32, f32::max);
        assert!(
            worst_tail < peak * 0.1,
            "identity FIR should be a spike: peak {peak}, worst tail {worst_tail}"
        );
    }

    #[test]
    fn convolving_with_a_centred_impulse_returns_the_signal() {
        let signal = sine(440.0, 48_000.0, 20_000);
        let mut fir = vec![0.0_f32; 4_096];
        fir[(4_096 - 1) / 2] = 1.0;
        let result = convolve_same(&signal, &fir);
        assert_eq!(result.len(), signal.len());
        for (index, (got, expected)) in result.iter().zip(signal.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-3,
                "sample {index}: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn convolution_matches_a_direct_sum() {
        // Overlap-add is easy to get subtly wrong at the block joins.
        let signal: Vec<f32> = (0..3_000)
            .map(|index| (index as f32 * 0.05).sin() * 0.4)
            .collect();
        let fir: Vec<f32> = (0..64)
            .map(|index| if index == 20 { 0.8 } else { 0.01 })
            .collect();

        let fast = convolve_same(&signal, &fir);
        let start = (fir.len() - 1) / 2;
        for probe in [0_usize, 1, 500, 1_500, 2_999] {
            let mut expected = 0.0_f32;
            for (tap, coefficient) in fir.iter().enumerate() {
                let position = probe + start;
                if position >= tap {
                    if let Some(sample) = signal.get(position - tap) {
                        expected += sample * coefficient;
                    }
                }
            }
            assert!(
                (fast[probe] - expected).abs() < 1e-3,
                "sample {probe}: overlap-add {} vs direct {expected}",
                fast[probe]
            );
        }
    }

    #[test]
    fn a_brighter_reference_asks_for_treble() {
        let rate = 48_000.0;
        let length = 4_096 * 16;
        // Same two tones, but the reference has far more of the high one.
        let target: Vec<f32> = (0..length)
            .map(|index| {
                let time = index as f32 / rate;
                (TAU * 200.0 * time).sin() * 0.5 + (TAU * 8_000.0 * time).sin() * 0.02
            })
            .collect();
        let reference: Vec<f32> = (0..length)
            .map(|index| {
                let time = index as f32 / rate;
                (TAU * 200.0 * time).sin() * 0.5 + (TAU * 8_000.0 * time).sin() * 0.5
            })
            .collect();

        let config = Config::default();
        let fir = matching_fir(&target, &reference, &config);
        let response = average_spectrum(&fir, config.fft_size);

        let bin_of = |hz: f32| (hz / rate * config.fft_size as f32).round() as usize;
        assert!(
            response[bin_of(8_000.0)] > response[bin_of(200.0)],
            "the filter should lift 8 kHz above 200 Hz: {} vs {}",
            response[bin_of(8_000.0)],
            response[bin_of(200.0)]
        );
    }
}
