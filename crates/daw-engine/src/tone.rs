#![allow(clippy::cast_precision_loss)]

//! An amplifier's tone stack: bass, middle and treble.
//!
//! Separate from the channel EQ in [`crate::ChannelStrip`], and deliberately
//! so. A guitar amplifier's tone controls are part of the amplifier — they sit
//! between the preamp and the power stage and are half of what a given amp
//! sounds like. The channel EQ sits on the track and shapes it against the
//! rest of the mix. Sharing one set of controls between the two jobs means
//! never being able to do both.
//!
//! Controls run `0` to `10` and are flat at `5`, the way the markings on an
//! amplifier do, rather than in decibels.
//!
//! Real-time contract: [`ToneStack::process`] allocates nothing.

use daw_core::SampleRate;

/// Where the bass band gives way to the middle.
const LOW_CROSSOVER_HZ: f32 = 180.0;
/// Where the middle gives way to the treble.
const HIGH_CROSSOVER_HZ: f32 = 2_500.0;
/// Cut or boost at the ends of a control's travel.
const RANGE_DB: f32 = 12.0;
/// The setting at which a control does nothing.
pub const FLAT: f32 = 5.0;
/// The top of a control's travel.
pub const MAX: f32 = 10.0;

#[derive(Clone, Copy, Debug, Default)]
struct BandState {
    low: f32,
    low_mid: f32,
}

pub struct ToneStack {
    low_coefficient: f32,
    high_coefficient: f32,
    state: [BandState; 2],
}

impl ToneStack {
    #[must_use]
    pub fn new(sample_rate: SampleRate) -> Self {
        let rate = sample_rate.get().max(1) as f32;
        Self {
            low_coefficient: one_pole(LOW_CROSSOVER_HZ, rate),
            high_coefficient: one_pole(HIGH_CROSSOVER_HZ, rate),
            state: [BandState::default(); 2],
        }
    }

    /// Clears the filter memory, for a stop or a seek.
    pub fn reset(&mut self) {
        self.state = [BandState::default(); 2];
    }

    /// Shapes a block in place. `bass`, `middle` and `treble` run `0` to
    /// [`MAX`] and are flat at [`FLAT`].
    pub fn process(&mut self, frames: &mut [[f32; 2]], bass: f32, middle: f32, treble: f32) {
        let gains = [gain_for(bass), gain_for(middle), gain_for(treble)];
        // Flat controls must leave the signal exactly as it arrived, not
        // merely close to it: an amp with its tone at noon is not an effect.
        if gains.iter().all(|gain| (gain - 1.0).abs() < 1e-6) {
            return;
        }
        let (low_coefficient, high_coefficient) = (self.low_coefficient, self.high_coefficient);
        for frame in frames {
            for (channel, state) in self.state.iter_mut().enumerate() {
                let sample = frame[channel];
                // Split into three bands with two one-poles, then recombine at
                // the requested weights. The bands sum back to the input when
                // every weight is one, so flat really is flat.
                state.low += low_coefficient * (sample - state.low);
                state.low_mid += high_coefficient * (sample - state.low_mid);
                let low = state.low;
                let middle_band = state.low_mid - state.low;
                let high = sample - state.low_mid;
                frame[channel] = low * gains[0] + middle_band * gains[1] + high * gains[2];
            }
        }
    }
}

/// Linear gain for a control position.
fn gain_for(position: f32) -> f32 {
    let decibels = (position.clamp(0.0, MAX) - FLAT) / FLAT * RANGE_DB;
    10.0_f32.powf(decibels / 20.0)
}

fn one_pole(frequency: f32, sample_rate: f32) -> f32 {
    1.0 - (-std::f32::consts::TAU * frequency / sample_rate).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stack() -> ToneStack {
        ToneStack::new(SampleRate::DEFAULT)
    }

    /// A tone at `hertz`, one second of it.
    fn tone(hertz: f32) -> Vec<[f32; 2]> {
        let rate = SampleRate::DEFAULT.get() as f32;
        (0..48_000)
            .map(|index| {
                let value = (index as f32 / rate * hertz * std::f32::consts::TAU).sin() * 0.5;
                [value; 2]
            })
            .collect()
    }

    fn level(frames: &[[f32; 2]]) -> f32 {
        // The back half only, so the filters have settled.
        let tail = &frames[frames.len() / 2..];
        (tail.iter().map(|frame| frame[0] * frame[0]).sum::<f32>() / tail.len() as f32).sqrt()
    }

    #[test]
    fn controls_at_noon_leave_the_signal_untouched() {
        let mut stack = stack();
        let original = tone(440.0);
        let mut frames = original.clone();
        stack.process(&mut frames, FLAT, FLAT, FLAT);
        assert_eq!(frames, original);
    }

    #[test]
    fn bass_moves_the_low_end_and_leaves_the_top_alone() {
        let low = tone(60.0);
        let high = tone(8_000.0);
        let shaped = |input: &[[f32; 2]], bass: f32| -> f32 {
            let mut stack = stack();
            let mut frames = input.to_vec();
            stack.process(&mut frames, bass, FLAT, FLAT);
            level(&frames)
        };
        assert!(
            shaped(&low, MAX) > shaped(&low, FLAT) * 2.0,
            "bass did not lift"
        );
        assert!(
            shaped(&low, 0.0) < shaped(&low, FLAT) * 0.5,
            "bass did not cut"
        );
        let untouched = (shaped(&high, MAX) - shaped(&high, FLAT)).abs();
        assert!(
            untouched < shaped(&high, FLAT) * 0.1,
            "bass moved the treble band"
        );
    }

    #[test]
    fn treble_moves_the_top_and_leaves_the_bottom_alone() {
        let low = tone(60.0);
        let high = tone(8_000.0);
        let shaped = |input: &[[f32; 2]], treble: f32| -> f32 {
            let mut stack = stack();
            let mut frames = input.to_vec();
            stack.process(&mut frames, FLAT, FLAT, treble);
            level(&frames)
        };
        assert!(
            shaped(&high, MAX) > shaped(&high, FLAT) * 2.0,
            "treble did not lift"
        );
        assert!(
            shaped(&high, 0.0) < shaped(&high, FLAT) * 0.5,
            "treble did not cut"
        );
        let untouched = (shaped(&low, MAX) - shaped(&low, FLAT)).abs();
        assert!(
            untouched < shaped(&low, FLAT) * 0.1,
            "treble moved the bass band"
        );
    }

    #[test]
    fn the_middle_sits_between_the_other_two() {
        let mid = tone(900.0);
        let shaped = |middle: f32| -> f32 {
            let mut stack = stack();
            let mut frames = mid.clone();
            stack.process(&mut frames, FLAT, middle, FLAT);
            level(&frames)
        };
        assert!(shaped(MAX) > shaped(FLAT) * 1.5, "the middle did not lift");
        assert!(shaped(0.0) < shaped(FLAT) * 0.7, "the middle did not cut");
    }

    #[test]
    fn positions_outside_the_dial_are_clamped() {
        let mut stack = stack();
        let mut frames = tone(440.0);
        for position in [-10.0_f32, 0.0, MAX, 1_000.0, f32::INFINITY] {
            stack.process(&mut frames, position, position, position);
            assert!(
                frames.iter().all(|frame| frame[0].is_finite()),
                "{position} produced a non-finite sample"
            );
        }
    }

    #[test]
    fn resetting_clears_the_filter_memory() {
        let mut stack = stack();
        let mut loud = vec![[1.0_f32; 2]; 1_024];
        stack.process(&mut loud, MAX, FLAT, FLAT);
        stack.reset();
        let mut quiet = vec![[0.0_f32; 2]; 1_024];
        stack.process(&mut quiet, MAX, FLAT, FLAT);
        assert!(
            quiet.iter().all(|frame| frame[0].abs() < 1e-9),
            "the filter carried its history across a reset"
        );
    }

    #[test]
    fn an_empty_block_is_safe() {
        stack().process(&mut [], 0.0, MAX, FLAT);
    }
}
