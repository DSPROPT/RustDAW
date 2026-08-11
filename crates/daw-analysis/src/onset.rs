#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

//! Onset strength over time, the signal every tempo decision is made from.
//!
//! Spectral flux: the frame-to-frame rise in each frequency bin, summed. Only
//! rises count — a note ending is not an onset — which is what makes the
//! envelope peak on attacks rather than on loudness.

use crate::fft::{hann_window, magnitude_spectrum};

/// Analysis window. At 48 kHz this is 21 ms, short enough to separate two
/// sixteenth notes at 200 BPM and long enough to resolve bass drums.
pub const WINDOW: usize = 1_024;
/// Frames advance by a quarter window, giving ~187 envelope samples a second
/// at 48 kHz — about 5 ms of timing resolution on every beat.
pub const HOP: usize = 256;

#[derive(Clone, Debug)]
pub struct OnsetEnvelope {
    /// Onset strength per frame, normalised to zero mean and unit variance.
    pub values: Vec<f32>,
    pub frames_per_second: f64,
}

impl OnsetEnvelope {
    #[must_use]
    pub fn seconds_at(&self, frame: usize) -> f64 {
        if self.frames_per_second <= 0.0 {
            return 0.0;
        }
        frame as f64 / self.frames_per_second
    }

    #[must_use]
    pub fn frame_at(&self, seconds: f64) -> usize {
        (seconds * self.frames_per_second).max(0.0) as usize
    }

    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        self.seconds_at(self.values.len())
    }
}

/// Computes the onset envelope of a mono signal.
#[must_use]
pub fn onset_envelope(samples: &[f32], sample_rate: u32) -> OnsetEnvelope {
    let frames_per_second = f64::from(sample_rate) / HOP as f64;
    if samples.len() < WINDOW || sample_rate == 0 {
        return OnsetEnvelope {
            values: Vec::new(),
            frames_per_second,
        };
    }

    let window = hann_window(WINDOW);
    let mut windowed = vec![0.0_f32; WINDOW];
    let (mut scratch_real, mut scratch_imag) = (Vec::new(), Vec::new());
    let mut spectrum = Vec::new();
    let mut previous: Vec<f32> = Vec::new();
    let mut flux = Vec::with_capacity(samples.len() / HOP);

    let mut start = 0;
    while start + WINDOW <= samples.len() {
        for (index, slot) in windowed.iter_mut().enumerate() {
            *slot = samples[start + index] * window[index];
        }
        magnitude_spectrum(&windowed, &mut scratch_real, &mut scratch_imag, &mut spectrum);
        // Compress the magnitudes. A snare 30 dB above a hi-hat should not
        // count thirty times as much: on a logarithmic scale both read as
        // onsets, which is how a listener hears them.
        for bin in &mut spectrum {
            *bin = (1.0 + 1_000.0 * *bin).ln();
        }

        if previous.len() == spectrum.len() {
            let rise: f32 = spectrum
                .iter()
                .zip(previous.iter())
                .map(|(now, before)| (now - before).max(0.0))
                .sum();
            flux.push(rise);
        } else {
            flux.push(0.0);
        }
        previous.clear();
        previous.extend_from_slice(&spectrum);
        start += HOP;
    }

    normalise(&mut flux);
    OnsetEnvelope {
        values: flux,
        frames_per_second,
    }
}

/// Removes a slow-moving local average, then scales to unit standard
/// deviation. Subtracting the local mean is what lets one threshold work for
/// both a quiet intro and a loud chorus.
fn normalise(values: &mut [f32]) {
    const SMOOTHING: usize = 16;
    if values.is_empty() {
        return;
    }

    let smoothed: Vec<f32> = (0..values.len())
        .map(|index| {
            let start = index.saturating_sub(SMOOTHING);
            let end = (index + SMOOTHING + 1).min(values.len());
            let window = &values[start..end];
            window.iter().sum::<f32>() / window.len() as f32
        })
        .collect();
    for (value, mean) in values.iter_mut().zip(smoothed) {
        *value = (*value - mean).max(0.0);
    }

    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance =
        values.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / values.len() as f32;
    let deviation = variance.sqrt();
    if deviation > f32::EPSILON {
        for value in values.iter_mut() {
            *value /= deviation;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Clicks every `interval` seconds, with a short decaying burst of noise
    /// so the spectrum actually changes at each onset.
    fn click_track(sample_rate: u32, seconds: f64, interval: f64) -> Vec<f32> {
        let total = (f64::from(sample_rate) * seconds) as usize;
        let mut samples = vec![0.0_f32; total];
        let mut position = 0.0;
        let mut noise = 12_345_u32;
        while position < seconds {
            let start = (position * f64::from(sample_rate)) as usize;
            for offset in 0..(sample_rate as usize / 40) {
                let Some(slot) = samples.get_mut(start + offset) else {
                    break;
                };
                noise = noise.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let white = (noise >> 8) as f32 / f32::from(u16::MAX) / 128.0 - 1.0;
                let decay = 1.0 - offset as f32 / (sample_rate as f32 / 40.0);
                *slot = white * decay * decay;
            }
            position += interval;
        }
        samples
    }

    #[test]
    fn silence_produces_no_onsets() {
        let envelope = onset_envelope(&vec![0.0; 48_000], 48_000);
        assert!(envelope.values.iter().all(|value| value.abs() < 1e-6));
    }

    #[test]
    fn clicks_produce_peaks_at_the_right_times() {
        let samples = click_track(48_000, 4.0, 0.5);
        let envelope = onset_envelope(&samples, 48_000);
        assert!(!envelope.values.is_empty());

        // The strongest frames should sit close to multiples of 0.5 s.
        let mut peaks: Vec<usize> = (1..envelope.values.len() - 1)
            .filter(|index| {
                let value = envelope.values[*index];
                value > 1.0
                    && value >= envelope.values[index - 1]
                    && value >= envelope.values[index + 1]
            })
            .collect();
        peaks.sort_unstable();
        assert!(peaks.len() >= 6, "expected several onsets, found {}", peaks.len());
        for peak in peaks {
            let seconds = envelope.seconds_at(peak);
            let distance = (seconds / 0.5 - (seconds / 0.5).round()).abs() * 0.5;
            assert!(distance < 0.06, "peak at {seconds:.3} s is not on a click");
        }
    }

    #[test]
    fn short_input_is_handled_without_panicking() {
        assert!(onset_envelope(&[0.1, 0.2], 48_000).values.is_empty());
        assert!(onset_envelope(&[], 48_000).values.is_empty());
        assert!(onset_envelope(&vec![0.5; 4_096], 0).values.is_empty());
    }

    #[test]
    fn frame_and_second_conversions_agree() {
        let envelope = onset_envelope(&vec![0.0; 48_000], 48_000);
        assert_eq!(envelope.frame_at(envelope.seconds_at(30)), 30);
    }
}
