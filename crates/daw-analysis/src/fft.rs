#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

//! A radix-2 fast Fourier transform.
//!
//! Written here rather than pulled in as a dependency: the analysis needs one
//! power-of-two real-input transform and nothing else, and the workspace
//! forbids `unsafe`, which most FFT crates use internally for their kernels.

use std::f32::consts::PI;

/// In-place complex FFT. `real` and `imag` must be the same power-of-two
/// length; anything else is a caller bug and leaves the input untouched.
pub fn transform(real: &mut [f32], imag: &mut [f32]) {
    let n = real.len();
    if n != imag.len() || n < 2 || !n.is_power_of_two() {
        return;
    }

    // Decimation in time: reorder into bit-reversed index order.
    let mut target = 0_usize;
    for source in 1..n {
        let mut mask = n >> 1;
        while target & mask != 0 {
            target ^= mask;
            mask >>= 1;
        }
        target |= mask;
        if source < target {
            real.swap(source, target);
            imag.swap(source, target);
        }
    }

    let mut span = 2;
    while span <= n {
        let half = span / 2;
        let angle_step = -2.0 * PI / span as f32;
        for start in (0..n).step_by(span) {
            for offset in 0..half {
                let angle = angle_step * offset as f32;
                let (sin, cos) = angle.sin_cos();
                let upper = start + offset + half;
                let lower = start + offset;
                let real_product = real[upper] * cos - imag[upper] * sin;
                let imag_product = real[upper] * sin + imag[upper] * cos;
                real[upper] = real[lower] - real_product;
                imag[upper] = imag[lower] - imag_product;
                real[lower] += real_product;
                imag[lower] += imag_product;
            }
        }
        span <<= 1;
    }
}

/// A transform of one fixed size, with its twiddle factors worked out once.
///
/// [`transform`] calls `sin_cos` inside its inner loop, which costs nothing
/// when a tempo detector runs a few hundred frames but dominates when a
/// mastering convolution runs tens of thousands. Same arithmetic, same
/// results; the angles are just read from a table instead of recomputed.
pub struct Planned {
    size: usize,
    /// Per stage, the `(sin, cos)` pairs for that stage's half-span.
    stages: Vec<Vec<(f32, f32)>>,
}

impl Planned {
    /// Plans a transform of `size`, which must be a power of two of at least 2.
    #[must_use]
    pub fn new(size: usize) -> Option<Self> {
        if size < 2 || !size.is_power_of_two() {
            return None;
        }
        let mut stages = Vec::new();
        let mut span = 2;
        while span <= size {
            let half = span / 2;
            let angle_step = -2.0 * PI / span as f32;
            stages.push(
                (0..half)
                    .map(|offset| (angle_step * offset as f32).sin_cos())
                    .collect(),
            );
            span <<= 1;
        }
        Some(Self { size, stages })
    }

    /// The size this was planned for.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// In-place forward transform. Wrong-sized input is left untouched.
    pub fn forward(&self, real: &mut [f32], imag: &mut [f32]) {
        let n = self.size;
        if real.len() != n || imag.len() != n {
            return;
        }

        let mut target = 0_usize;
        for source in 1..n {
            let mut mask = n >> 1;
            while target & mask != 0 {
                target ^= mask;
                mask >>= 1;
            }
            target |= mask;
            if source < target {
                real.swap(source, target);
                imag.swap(source, target);
            }
        }

        let mut span = 2;
        for twiddles in &self.stages {
            let half = span / 2;
            for start in (0..n).step_by(span) {
                for (offset, (sin, cos)) in twiddles.iter().enumerate() {
                    let upper = start + offset + half;
                    let lower = start + offset;
                    let real_product = real[upper] * cos - imag[upper] * sin;
                    let imag_product = real[upper] * sin + imag[upper] * cos;
                    real[upper] = real[lower] - real_product;
                    imag[upper] = imag[lower] - imag_product;
                    real[lower] += real_product;
                    imag[lower] += imag_product;
                }
            }
            span <<= 1;
        }
    }

    /// In-place inverse transform, scaled so it undoes [`Planned::forward`].
    pub fn inverse(&self, real: &mut [f32], imag: &mut [f32]) {
        if real.len() != self.size || imag.len() != self.size {
            return;
        }
        for value in imag.iter_mut() {
            *value = -*value;
        }
        self.forward(real, imag);
        let scale = 1.0 / self.size as f32;
        for (re, im) in real.iter_mut().zip(imag.iter_mut()) {
            *re *= scale;
            *im *= -scale;
        }
    }
}

/// In-place inverse complex FFT, scaled so that `inverse(transform(x)) == x`.
///
/// Runs the forward transform over the conjugated input and conjugates the
/// result, which is the standard way to reuse one direction's butterflies for
/// both. Same size rules as [`transform`].
pub fn inverse(real: &mut [f32], imag: &mut [f32]) {
    let n = real.len();
    if n != imag.len() || n < 2 || !n.is_power_of_two() {
        return;
    }
    for value in imag.iter_mut() {
        *value = -*value;
    }
    transform(real, imag);
    let scale = 1.0 / n as f32;
    for (re, im) in real.iter_mut().zip(imag.iter_mut()) {
        *re *= scale;
        *im *= -scale;
    }
}

/// Inverse transform of a real signal's half-spectrum: the counterpart of
/// [`magnitude_spectrum`] for a spectrum that has been modified and needs to
/// come back as `(bins - 1) * 2` real samples.
///
/// `bins` are the `n / 2 + 1` non-negative frequencies. The negative half is
/// reconstructed by Hermitian symmetry, so the result is real by construction
/// and the imaginary part is discarded rather than checked.
#[must_use]
pub fn inverse_real(spectrum: &[f32]) -> Vec<f32> {
    if spectrum.len() < 2 {
        return Vec::new();
    }
    let n = (spectrum.len() - 1) * 2;
    if !n.is_power_of_two() {
        return Vec::new();
    }
    let mut real = vec![0.0; n];
    let mut imag = vec![0.0; n];
    real[..spectrum.len()].copy_from_slice(spectrum);
    // Mirror the positive frequencies onto the negative ones. The input is
    // purely real, so the conjugate of each bin is itself.
    for bin in 1..spectrum.len() - 1 {
        real[n - bin] = spectrum[bin];
    }
    inverse(&mut real, &mut imag);
    real
}

/// Magnitude spectrum of a real signal, returning the `n / 2 + 1` usable bins.
///
/// `scratch_real` and `scratch_imag` are supplied by the caller so a long
/// analysis reuses two buffers instead of allocating per frame.
pub fn magnitude_spectrum(
    windowed: &[f32],
    scratch_real: &mut Vec<f32>,
    scratch_imag: &mut Vec<f32>,
    output: &mut Vec<f32>,
) {
    scratch_real.clear();
    scratch_real.extend_from_slice(windowed);
    scratch_imag.clear();
    scratch_imag.resize(windowed.len(), 0.0);
    transform(scratch_real, scratch_imag);

    let bins = windowed.len() / 2 + 1;
    output.clear();
    output.extend(
        scratch_real
            .iter()
            .zip(scratch_imag.iter())
            .take(bins)
            .map(|(re, im)| re.hypot(*im)),
    );
}

/// A Hann window of `length` points.
#[must_use]
pub fn hann_window(length: usize) -> Vec<f32> {
    if length <= 1 {
        return vec![1.0; length];
    }
    let denominator = (length - 1) as f32;
    (0..length)
        .map(|index| {
            let phase = 2.0 * PI * index as f32 / denominator;
            0.5 - 0.5 * phase.cos()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive DFT, used only to check the fast version.
    fn reference_dft(input: &[f32]) -> Vec<(f32, f32)> {
        let n = input.len();
        (0..n)
            .map(|bin| {
                let mut re = 0.0;
                let mut im = 0.0;
                for (index, sample) in input.iter().enumerate() {
                    let angle = -2.0 * PI * (bin * index) as f32 / n as f32;
                    re += sample * angle.cos();
                    im += sample * angle.sin();
                }
                (re, im)
            })
            .collect()
    }

    #[test]
    fn matches_a_naive_dft() {
        let input: Vec<f32> = (0..64)
            .map(|index| {
                let time = index as f32;
                (time * 0.31).sin() * 0.7 + (time * 1.7).cos() * 0.2
            })
            .collect();
        let expected = reference_dft(&input);
        let mut real = input.clone();
        let mut imag = vec![0.0; input.len()];
        transform(&mut real, &mut imag);
        for (index, (re, im)) in expected.iter().enumerate() {
            assert!(
                (real[index] - re).abs() < 1e-2 && (imag[index] - im).abs() < 1e-2,
                "bin {index}: got ({}, {}), expected ({re}, {im})",
                real[index],
                imag[index]
            );
        }
    }

    #[test]
    fn a_pure_tone_peaks_in_its_own_bin() {
        const N: usize = 256;
        const BIN: usize = 17;
        let input: Vec<f32> = (0..N)
            .map(|index| {
                let phase = 2.0 * PI * (BIN * index) as f32 / N as f32;
                phase.sin()
            })
            .collect();
        let mut spectrum = Vec::new();
        let (mut re, mut im) = (Vec::new(), Vec::new());
        magnitude_spectrum(&input, &mut re, &mut im, &mut spectrum);
        let peak = spectrum
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.partial_cmp(right.1).unwrap())
            .map(|(index, _)| index)
            .unwrap();
        assert_eq!(peak, BIN);
    }

    #[test]
    fn non_power_of_two_input_is_left_alone() {
        let mut real = vec![1.0, 2.0, 3.0];
        let mut imag = vec![0.0; 3];
        transform(&mut real, &mut imag);
        assert_eq!(real, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn the_inverse_transform_undoes_the_forward_one() {
        let input: Vec<f32> = (0..128)
            .map(|index| {
                let time = index as f32;
                (time * 0.21).sin() * 0.6 + (time * 2.3).cos() * 0.3
            })
            .collect();
        let mut real = input.clone();
        let mut imag = vec![0.0; input.len()];
        transform(&mut real, &mut imag);
        inverse(&mut real, &mut imag);
        for (index, expected) in input.iter().enumerate() {
            assert!(
                (real[index] - expected).abs() < 1e-3,
                "sample {index}: got {}, expected {expected}",
                real[index]
            );
        }
    }

    #[test]
    fn a_planned_transform_matches_the_direct_one() {
        let input: Vec<f32> = (0..512)
            .map(|index| {
                let time = index as f32;
                (time * 0.13).sin() * 0.5 + (time * 1.1).cos() * 0.25
            })
            .collect();
        let (mut direct_re, mut direct_im) = (input.clone(), vec![0.0; input.len()]);
        transform(&mut direct_re, &mut direct_im);

        let planned = Planned::new(input.len()).expect("power of two");
        let (mut re, mut im) = (input.clone(), vec![0.0; input.len()]);
        planned.forward(&mut re, &mut im);

        for index in 0..input.len() {
            assert!(
                (re[index] - direct_re[index]).abs() < 1e-2
                    && (im[index] - direct_im[index]).abs() < 1e-2,
                "bin {index} diverged"
            );
        }

        planned.inverse(&mut re, &mut im);
        for (index, expected) in input.iter().enumerate() {
            assert!((re[index] - expected).abs() < 1e-3, "sample {index}");
        }
    }

    #[test]
    fn a_planned_transform_refuses_sizes_it_cannot_do() {
        assert!(Planned::new(100).is_none());
        assert!(Planned::new(1).is_none());
        assert!(Planned::new(256).is_some());
    }

    #[test]
    fn a_flat_half_spectrum_inverts_to_an_impulse() {
        // Every bin at one is a delta at sample zero: the definition of the
        // transform, and the cheapest check that the mirroring is right.
        let spectrum = vec![1.0_f32; 33];
        let signal = inverse_real(&spectrum);
        assert_eq!(signal.len(), 64);
        assert!((signal[0] - 1.0).abs() < 1e-3, "got {}", signal[0]);
        for (index, sample) in signal.iter().enumerate().skip(1) {
            assert!(sample.abs() < 1e-3, "sample {index} should be silent");
        }
    }

    #[test]
    fn a_hann_window_is_zero_at_both_ends() {
        let window = hann_window(64);
        assert!(window[0].abs() < 1e-6);
        assert!(window[63].abs() < 1e-6);
        assert!((window[32] - 1.0).abs() < 0.01);
    }
}
