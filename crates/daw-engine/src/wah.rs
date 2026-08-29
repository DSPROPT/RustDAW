#![allow(clippy::cast_precision_loss)]

//! A wah pedal: one resonant peak, swept by a foot.
//!
//! A wah is not a tone control and not a filter sweep in the synthesiser
//! sense. It is a single sharp peak dragged across the guitar's most
//! expressive octave and a half, and almost everything that makes one sound
//! like a wah rather than like an EQ being wobbled is [`RESONANCE`].
//!
//! The pedal belongs in front of the amplifier, which is where the runtime
//! puts it: a wah after the distortion is a different, thinner effect, and not
//! the one anybody means when they ask for one.
//!
//! Real-time contract: [`Wah::process_stereo`] and [`Wah::process_mono`]
//! allocate nothing.

use daw_core::SampleRate;

/// Where the peak sits with the pedal heel-down.
const HEEL_HZ: f32 = 400.0;
/// Where it sits toe-down.
const TOE_HZ: f32 = 2_000.0;
/// How sharp the peak is. Too low and the sweep is a tone control; too high
/// and it whistles instead of speaking.
const RESONANCE: f32 = 3.5;
/// How long the peak takes to catch up with the pedal, in milliseconds.
///
/// An expression pedal arrives as 128 steps, read at whatever rate the
/// interface polls it, and stepping the filter straight to each one is audible
/// as a zipper. Sliding between them hides the steps without lagging behind a
/// real sweep, which takes a few hundred milliseconds end to end.
const GLIDE_MS: f32 = 12.0;
/// The wet fraction a pedal starts at. Not all the way wet: the peak alone is
/// thin, and leaving some of the dry signal through is what keeps a wah
/// sounding like a guitar underneath the vowel.
pub const DEFAULT_MIX: f32 = 0.7;

/// A Chamberlin state variable filter's two integrators.
#[derive(Clone, Copy, Debug, Default)]
struct Resonator {
    low: f32,
    band: f32,
}

pub struct Wah {
    sample_rate: f32,
    /// The tuning coefficient the filter has actually reached, which trails
    /// the pedal by [`GLIDE_MS`].
    coefficient: f32,
    /// How much of the gap to the pedal's coefficient is closed each sample.
    glide: f32,
    state: [Resonator; 2],
}

impl Wah {
    #[must_use]
    pub fn new(sample_rate: SampleRate) -> Self {
        let rate = sample_rate.get().max(1) as f32;
        Self {
            sample_rate: rate,
            coefficient: coefficient_for(0.0, rate),
            glide: 1.0 - (-1_000.0 / (GLIDE_MS * rate)).exp(),
            state: [Resonator::default(); 2],
        }
    }

    /// Clears the filter memory, for a stop or a seek. The pedal itself has
    /// not moved, so where the peak is tuned to is left alone.
    pub fn reset(&mut self) {
        self.state = [Resonator::default(); 2];
    }

    /// Sweeps a stereo block in place.
    ///
    /// `position` runs `0` heel-down to `1` toe-down, and `mix` is the wet
    /// fraction.
    pub fn process_stereo(&mut self, frames: &mut [[f32; 2]], position: f32, mix: f32) {
        let (target, mix) = self.settings(position, mix);
        for frame in frames {
            self.coefficient += self.glide * (target - self.coefficient);
            let coefficient = self.coefficient;
            for (channel, state) in self.state.iter_mut().enumerate() {
                frame[channel] = resonate(state, frame[channel], coefficient, mix);
            }
        }
    }

    /// Sweeps a mono block in place, as the signal reaches a guitar amp.
    pub fn process_mono(&mut self, samples: &mut [f32], position: f32, mix: f32) {
        let (target, mix) = self.settings(position, mix);
        for sample in samples {
            self.coefficient += self.glide * (target - self.coefficient);
            *sample = resonate(&mut self.state[0], *sample, self.coefficient, mix);
        }
    }

    /// Where the pedal is asking the peak to sit, and how wet to run, both
    /// brought into range once per block rather than per sample.
    fn settings(&self, position: f32, mix: f32) -> (f32, f32) {
        // `f32::clamp` passes a NaN straight through, and a NaN wet fraction
        // would put one into the signal and keep it there.
        let mix = if mix.is_finite() { mix.clamp(0.0, 1.0) } else { 0.0 };
        (coefficient_for(position, self.sample_rate), mix)
    }

    /// Where the peak currently sits, in Hz, for a display that wants to show
    /// the pedal rather than guess at it.
    #[must_use]
    pub fn peak_hz(&self) -> f32 {
        // The inverse of `coefficient_for`, which is a sine of the frequency.
        let ratio = (self.coefficient * 0.5).clamp(-1.0, 1.0).asin();
        ratio * self.sample_rate / std::f32::consts::PI
    }
}

/// One sample through the resonator, mixed back against the dry signal.
///
/// The bandpass tap is what a wah's peak is: its gain at the tuned frequency
/// is [`RESONANCE`], and it falls away either side, so sweeping it across a
/// chord is the vowel the pedal makes.
fn resonate(state: &mut Resonator, input: f32, coefficient: f32, mix: f32) -> f32 {
    const DAMPING: f32 = 1.0 / RESONANCE;
    let high = input - state.low - DAMPING * state.band;
    state.band += coefficient * high;
    state.low += coefficient * state.band;
    // A filter that has been fed a non-finite sample never recovers on its
    // own, and a guitar that has gone permanently silent mid-take is worse
    // than a pedal that briefly does nothing.
    if !state.band.is_finite() || !state.low.is_finite() {
        *state = Resonator::default();
        return input;
    }
    input + mix * (state.band - input)
}

/// The state variable filter's tuning coefficient for a pedal position.
///
/// The sweep is exponential because pitch is: a linear sweep spends most of
/// its travel in the top octave and crosses the bottom one in an instant.
fn coefficient_for(position: f32, sample_rate: f32) -> f32 {
    let position = if position.is_finite() {
        position.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let hertz = HEEL_HZ * (TOE_HZ / HEEL_HZ).powf(position);
    // The Chamberlin topology is only stable while the peak stays well below
    // the Nyquist frequency; at any sample rate a guitar is recorded at this
    // clamp is far above the top of the sweep and never reached.
    let highest = sample_rate / 6.0;
    2.0 * (std::f32::consts::PI * hertz.min(highest) / sample_rate).sin()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wah() -> Wah {
        Wah::new(SampleRate::DEFAULT)
    }

    /// A second of a tone at `hertz`.
    fn tone(hertz: f32) -> Vec<[f32; 2]> {
        let rate = SampleRate::DEFAULT.get() as f32;
        (0..48_000)
            .map(|index| {
                let value = (index as f32 / rate * hertz * std::f32::consts::TAU).sin() * 0.5;
                [value; 2]
            })
            .collect()
    }

    /// The level of the back half, once the filter and the glide have settled.
    fn level(frames: &[[f32; 2]]) -> f32 {
        let tail = &frames[frames.len() / 2..];
        (tail.iter().map(|frame| frame[0] * frame[0]).sum::<f32>() / tail.len() as f32).sqrt()
    }

    /// The level of `hertz` with the pedal held at `position`.
    fn swept(hertz: f32, position: f32) -> f32 {
        let mut wah = wah();
        let mut frames = tone(hertz);
        wah.process_stereo(&mut frames, position, 1.0);
        level(&frames)
    }

    #[test]
    fn the_peak_follows_the_pedal_up() {
        // Heel-down favours the low note, toe-down the high one. That reversal
        // is the whole effect.
        assert!(swept(400.0, 0.0) > swept(400.0, 1.0) * 2.0, "heel lost its low");
        assert!(swept(2_000.0, 1.0) > swept(2_000.0, 0.0) * 2.0, "toe lost its high");
    }

    #[test]
    fn the_tuned_frequency_is_lifted_and_the_rest_is_not() {
        // A resonant peak, not a tilt: the note at the peak comes up, and a
        // note two octaves below it does not.
        let dry = tone(400.0);
        assert!(swept(400.0, 0.0) > level(&dry) * 2.0, "the peak did not lift");
        assert!(swept(100.0, 0.0) < level(&tone(100.0)) * 0.9, "two octaves down was lifted");
    }

    #[test]
    fn a_dry_pedal_is_not_in_the_signal_path() {
        // Bypass has to be exact, not merely close: a pedal that is off is off.
        let mut wah = wah();
        let original = tone(440.0);
        let mut frames = original.clone();
        wah.process_stereo(&mut frames, 0.5, 0.0);
        assert_eq!(frames, original);
    }

    #[test]
    fn both_channels_are_swept_the_same_way() {
        let mut wah = wah();
        let mut frames = tone(800.0);
        wah.process_stereo(&mut frames, 0.35, 1.0);
        assert!(frames.iter().all(|frame| (frame[0] - frame[1]).abs() < 1e-6));
    }

    #[test]
    fn a_mono_block_is_swept_like_the_left_channel() {
        let mut stereo = wah();
        let mut frames = tone(800.0);
        stereo.process_stereo(&mut frames, 0.8, 0.6);
        let mut mono_wah = wah();
        let mut samples: Vec<f32> = tone(800.0).iter().map(|frame| frame[0]).collect();
        mono_wah.process_mono(&mut samples, 0.8, 0.6);
        assert!(
            samples
                .iter()
                .zip(&frames)
                .all(|(sample, frame)| (sample - frame[0]).abs() < 1e-6)
        );
    }

    #[test]
    fn the_pedal_slides_rather_than_stepping() {
        // The steps an expression pedal arrives in must not reach the filter,
        // so a jump from one end to the other takes time to arrive.
        let mut wah = wah();
        let mut settle = tone(400.0);
        wah.process_stereo(&mut settle, 0.0, 1.0);
        let heel = wah.peak_hz();
        let mut block = vec![[0.0_f32; 2]; 32];
        wah.process_stereo(&mut block, 1.0, 1.0);
        let after = wah.peak_hz();
        assert!(after > heel, "the peak did not move");
        assert!(after < TOE_HZ * 0.9, "the peak jumped straight to the toe");
    }

    #[test]
    fn positions_outside_the_pedal_are_clamped() {
        let mut wah = wah();
        let mut frames = tone(440.0);
        for position in [-4.0_f32, 0.0, 1.0, 9.0, f32::INFINITY, f32::NAN] {
            for mix in [-1.0_f32, 0.5, 4.0, f32::NAN] {
                wah.process_stereo(&mut frames, position, mix);
                assert!(
                    frames.iter().all(|frame| frame[0].is_finite()),
                    "{position} at {mix} produced a non-finite sample"
                );
            }
        }
    }

    #[test]
    fn resetting_clears_the_filter_memory() {
        let mut wah = wah();
        let mut loud = vec![[1.0_f32; 2]; 4_096];
        wah.process_stereo(&mut loud, 0.5, 1.0);
        wah.reset();
        let mut quiet = vec![[0.0_f32; 2]; 4_096];
        wah.process_stereo(&mut quiet, 0.5, 1.0);
        assert!(quiet.iter().all(|frame| frame[0].abs() < 1e-9));
    }

    #[test]
    fn an_empty_block_is_safe() {
        wah().process_stereo(&mut [], 0.5, 1.0);
        wah().process_mono(&mut [], 0.5, 1.0);
    }
}
