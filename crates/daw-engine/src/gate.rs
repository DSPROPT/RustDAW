#![allow(clippy::cast_precision_loss)]

//! A noise gate for the front of an amplifier.
//!
//! Distinct from the gate in [`crate::ChannelStrip`], which sits on the track
//! after the amp. This one sits *before* it, and the order is the whole point:
//! a high-gain amplifier lifts the hiss and hum between notes by as much as it
//! lifts the notes, so silence has to be established before the gain rather
//! than cleaned up after it. Gating afterwards means gating a signal whose
//! noise floor has already been amplified into the music.
//!
//! Real-time contract: [`NoiseGate::process`] allocates nothing.

use daw_core::SampleRate;

/// Below this the gate is considered off rather than merely very low, so a
/// control at the bottom of its travel passes everything.
pub const OPEN_THRESHOLD_DB: f32 = -95.0;
/// How quickly the gate opens once the signal is above the threshold. Fast,
/// or the front of every note is chewed off.
const ATTACK_MS: f32 = 1.0;
/// How quickly it closes again.
const RELEASE_MS: f32 = 80.0;

pub struct NoiseGate {
    sample_rate: f32,
    envelope: f32,
    gain: f32,
}

impl NoiseGate {
    #[must_use]
    pub fn new(sample_rate: SampleRate) -> Self {
        Self {
            sample_rate: sample_rate.get().max(1) as f32,
            envelope: 0.0,
            gain: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.envelope = 0.0;
        self.gain = 0.0;
    }

    /// Gates a mono block in place at `threshold_db`.
    ///
    /// A threshold at or below [`OPEN_THRESHOLD_DB`] passes the signal through
    /// untouched, which is what the bottom of the control means.
    pub fn process(&mut self, samples: &mut [f32], threshold_db: f32) {
        if threshold_db <= OPEN_THRESHOLD_DB {
            // Held open, and held ready: a gate that has been bypassed must not
            // slam shut on the first sample when it is switched back in.
            self.gain = 1.0;
            self.envelope = 1.0;
            return;
        }
        let attack = coefficient(ATTACK_MS, self.sample_rate);
        let release = coefficient(RELEASE_MS, self.sample_rate);
        let threshold = 10.0_f32.powf(threshold_db / 20.0);
        for sample in samples {
            let level = sample.abs();
            // Peak-follow up, decay down, so a note holds the gate open through
            // its own zero crossings rather than chattering at them.
            self.envelope = level.max(self.envelope * release);
            let target = f32::from(self.envelope >= threshold);
            let smoothing = if target > self.gain { attack } else { release };
            self.gain = target + smoothing * (self.gain - target);
            *sample *= self.gain;
        }
    }
}

fn coefficient(milliseconds: f32, sample_rate: f32) -> f32 {
    (-1.0 / (milliseconds.max(0.1) * 0.001 * sample_rate)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate() -> NoiseGate {
        NoiseGate::new(SampleRate::DEFAULT)
    }

    fn level(samples: &[f32]) -> f32 {
        samples.iter().map(|sample| sample.abs()).sum::<f32>() / samples.len() as f32
    }

    #[test]
    fn hiss_below_the_threshold_is_shut_out() {
        let mut gate = gate();
        let mut samples = vec![0.001_f32; 48_000];
        gate.process(&mut samples, -40.0);
        assert!(
            level(&samples[24_000..]) < 1e-5,
            "the gate let the noise floor through"
        );
    }

    #[test]
    fn playing_above_the_threshold_passes() {
        let mut gate = gate();
        let mut samples: Vec<f32> = (0..48_000)
            .map(|index| (index as f32 * 0.05).sin() * 0.5)
            .collect();
        let before = level(&samples);
        gate.process(&mut samples, -40.0);
        let after = level(&samples[24_000..]);
        assert!(
            after > before * 0.8,
            "the gate held a signal well above its threshold down"
        );
    }

    #[test]
    fn the_bottom_of_the_dial_passes_everything() {
        let mut gate = gate();
        let original: Vec<f32> = (0..1_024).map(|index| index as f32 * 1e-6).collect();
        let mut samples = original.clone();
        gate.process(&mut samples, OPEN_THRESHOLD_DB);
        assert_eq!(samples, original);
    }

    #[test]
    fn a_gate_switched_back_in_does_not_slam_shut_on_the_first_note() {
        // Held open, then thresholded: the first sample must not be silenced
        // while the envelope catches up.
        let mut gate = gate();
        gate.process(&mut [0.5_f32; 128], OPEN_THRESHOLD_DB);
        let mut samples = vec![0.5_f32; 256];
        gate.process(&mut samples, -40.0);
        assert!(samples[0].abs() > 0.4, "the first sample was cut: {}", samples[0]);
    }

    #[test]
    fn the_front_of_a_note_survives() {
        // A gate that takes too long to open eats the pick attack, which is
        // the part of a guitar note that carries its identity.
        let mut gate = gate();
        gate.process(&mut vec![0.0_f32; 48_000], -40.0);
        let mut note = vec![0.6_f32; 480];
        gate.process(&mut note, -40.0);
        // Open within five milliseconds.
        assert!(
            note[240].abs() > 0.3,
            "the gate was still opening after 5 ms: {}",
            note[240]
        );
    }

    #[test]
    fn resetting_closes_it_again() {
        let mut gate = gate();
        gate.process(&mut vec![0.9_f32; 4_800], -40.0);
        gate.reset();
        let mut quiet = vec![0.001_f32; 4_800];
        gate.process(&mut quiet, -40.0);
        assert!(level(&quiet) < 1e-5, "the gate stayed open across a reset");
    }

    #[test]
    fn an_empty_block_is_safe() {
        gate().process(&mut [], -40.0);
    }
}
