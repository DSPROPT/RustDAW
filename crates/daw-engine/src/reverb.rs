#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

//! A stereo reverb for the instrument bus.
//!
//! Every acoustic instrument is heard in a room. A synthesised note played
//! perfectly dry is the one cue that no amount of work on the voices removes,
//! because the ear reads the absence of early reflections as "this was never in
//! a room" — so this buys more realism per line than any single voice
//! parameter.
//!
//! The topology is Schroeder's, in Jezar's Freeverb arrangement: eight damped
//! comb filters in parallel building the tail, then four allpasses in series
//! smearing it until individual echoes stop being countable. The two channels
//! use delay lengths a few dozen samples apart, which is what decorrelates them
//! into a stereo image rather than one mono tail in both ears.
//!
//! Real-time contract: [`Reverb::process`] allocates nothing and does bounded
//! work per block. Every delay line is sized and zeroed by [`Reverb::new`],
//! which is not for the audio thread.

use daw_core::SampleRate;

/// Comb delays in frames at 44.1 kHz, scaled to the stream's actual rate.
/// Mutually prime lengths, so their echoes take a long time to line up.
const COMB_FRAMES: [usize; 8] = [1_116, 1_188, 1_277, 1_356, 1_422, 1_491, 1_557, 1_617];
/// Allpass delays, likewise.
const ALLPASS_FRAMES: [usize; 4] = [556, 441, 341, 225];
/// How far the right channel's delays are offset from the left's.
const STEREO_OFFSET: usize = 23;
const REFERENCE_RATE: f32 = 44_100.0;

/// How long the tail runs, `0` to `1`. A medium hall: long enough to sit an
/// orchestra in, short enough not to smear a drum kit.
const ROOM_SIZE: f32 = 0.72;
/// How fast the high end is absorbed. Real rooms lose treble first.
const DAMPING: f32 = 0.35;
/// Input trim. Eight combs in parallel is a lot of gain to give away.
const INPUT_GAIN: f32 = 0.015;

/// A comb filter with a low-pass in its feedback path, so each pass round the
/// loop is duller than the last.
struct Comb {
    buffer: Vec<f32>,
    index: usize,
    damped: f32,
}

impl Comb {
    fn new(frames: usize) -> Self {
        Self {
            buffer: vec![0.0; frames.max(1)],
            index: 0,
            damped: 0.0,
        }
    }

    fn process(&mut self, input: f32, feedback: f32, damping: f32) -> f32 {
        let output = self.buffer[self.index];
        self.damped = output * (1.0 - damping) + self.damped * damping;
        self.buffer[self.index] = input + self.damped * feedback;
        self.index += 1;
        if self.index >= self.buffer.len() {
            self.index = 0;
        }
        output
    }

    fn clear(&mut self) {
        self.buffer.fill(0.0);
        self.index = 0;
        self.damped = 0.0;
    }
}

/// An allpass: passes every frequency at the same level but not at the same
/// time, which is how a countable set of echoes becomes a wash.
struct Allpass {
    buffer: Vec<f32>,
    index: usize,
}

impl Allpass {
    fn new(frames: usize) -> Self {
        Self {
            buffer: vec![0.0; frames.max(1)],
            index: 0,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        const FEEDBACK: f32 = 0.5;
        let delayed = self.buffer[self.index];
        self.buffer[self.index] = input + delayed * FEEDBACK;
        self.index += 1;
        if self.index >= self.buffer.len() {
            self.index = 0;
        }
        delayed - input
    }

    fn clear(&mut self) {
        self.buffer.fill(0.0);
        self.index = 0;
    }
}

struct Channel {
    combs: [Comb; 8],
    allpasses: [Allpass; 4],
}

impl Channel {
    fn new(sample_rate: f32, offset: usize) -> Self {
        let scale = sample_rate / REFERENCE_RATE;
        let scaled = |frames: usize| ((frames + offset) as f32 * scale).max(1.0) as usize;
        Self {
            combs: std::array::from_fn(|index| Comb::new(scaled(COMB_FRAMES[index]))),
            allpasses: std::array::from_fn(|index| Allpass::new(scaled(ALLPASS_FRAMES[index]))),
        }
    }

    fn process(&mut self, input: f32, feedback: f32, damping: f32) -> f32 {
        // The combs run in parallel and sum; the allpasses run in series.
        let mut output = 0.0;
        for comb in &mut self.combs {
            output += comb.process(input, feedback, damping);
        }
        for allpass in &mut self.allpasses {
            output = allpass.process(output);
        }
        output
    }

    fn clear(&mut self) {
        for comb in &mut self.combs {
            comb.clear();
        }
        for allpass in &mut self.allpasses {
            allpass.clear();
        }
    }
}

pub struct Reverb {
    left: Channel,
    right: Channel,
    feedback: f32,
    damping: f32,
}

impl Reverb {
    /// Sizes and zeroes every delay line. Allocates; call before the stream
    /// opens, never from the callback.
    #[must_use]
    pub fn new(sample_rate: SampleRate) -> Self {
        let rate = sample_rate.get().max(1) as f32;
        Self {
            left: Channel::new(rate, 0),
            right: Channel::new(rate, STEREO_OFFSET),
            feedback: 0.7 + 0.28 * ROOM_SIZE,
            damping: DAMPING * 0.4,
        }
    }

    /// Drops the tail. For a stop or a seek, where letting the previous
    /// position's reverb ring on over the new one would be a smear.
    pub fn clear(&mut self) {
        self.left.clear();
        self.right.clear();
    }

    /// Sets how long the tail runs and how fast its top end is absorbed, both
    /// `0` to `1`. Coefficients only — nothing is resized, so this is safe on
    /// the audio thread.
    pub fn set_room(&mut self, size: f32, damping: f32) {
        self.feedback = 0.7 + 0.28 * size.clamp(0.0, 1.0);
        self.damping = damping.clamp(0.0, 1.0) * 0.4;
    }

    /// Processes a block in place as an insert, blending `mix` of the
    /// reverberated signal back over the dry one.
    ///
    /// The bus form in [`Reverb::process`] adds a send into a mix that already
    /// contains the dry signal; an insert owns the whole signal and has to
    /// return both parts itself.
    pub fn process_insert(&mut self, frames: &mut [[f32; 2]], mix: f32) {
        let mix = mix.clamp(0.0, 1.0);
        let (feedback, damping) = (self.feedback, self.damping);
        for frame in frames {
            let input = (frame[0] + frame[1]) * INPUT_GAIN;
            let wet = [
                self.left.process(input, feedback, damping),
                self.right.process(input, feedback, damping),
            ];
            frame[0] = frame[0] * (1.0 - mix) + wet[0] * mix;
            frame[1] = frame[1] * (1.0 - mix) + wet[1] * mix;
        }
    }

    /// Reverberates `send` and adds the result into `left` and `right`.
    ///
    /// The send is what the caller has already scaled per track; the dry signal
    /// is expected to be in the output buffers already.
    pub fn process(&mut self, send: &[[f32; 2]], left: &mut [f32], right: &mut [f32]) {
        let frames = send.len().min(left.len()).min(right.len());
        for index in 0..frames {
            // Both channels are fed the same mono sum: a reverb's job is to
            // place a source in a room, and the room is what differs between
            // the ears, not the source.
            let input = (send[index][0] + send[index][1]) * INPUT_GAIN;
            left[index] += self.left.process(input, self.feedback, self.damping);
            right[index] += self.right.process(input, self.feedback, self.damping);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reverb() -> Reverb {
        Reverb::new(SampleRate::DEFAULT)
    }

    fn energy(buffer: &[f32]) -> f32 {
        buffer.iter().map(|value| value.abs()).sum()
    }

    /// Feeds one impulse and returns the following `frames` of tail.
    fn impulse_response(frames: usize) -> (Vec<f32>, Vec<f32>) {
        let mut reverb = reverb();
        let mut send = vec![[0.0_f32; 2]; frames];
        send[0] = [1.0, 1.0];
        let mut left = vec![0.0; frames];
        let mut right = vec![0.0; frames];
        reverb.process(&send, &mut left, &mut right);
        (left, right)
    }

    #[test]
    fn an_insert_returns_the_dry_signal_when_it_is_fully_dry() {
        let mut reverb = reverb();
        let original: Vec<[f32; 2]> = (0..512)
            .map(|index| [(index as f32 * 0.02).sin(); 2])
            .collect();
        let mut frames = original.clone();
        reverb.process_insert(&mut frames, 0.0);
        for (processed, dry) in frames.iter().zip(&original) {
            assert!((processed[0] - dry[0]).abs() < 1e-6);
        }
    }

    #[test]
    fn an_insert_wets_the_signal_and_keeps_ringing() {
        let mut reverb = reverb();
        let mut frames = vec![[0.0_f32; 2]; 48_000];
        frames[0] = [1.0, 1.0];
        reverb.process_insert(&mut frames, 0.5);
        assert!(
            energy(&frames.iter().map(|frame| frame[0]).collect::<Vec<_>>()[2_000..]) > 0.0,
            "the insert produced no tail"
        );
        assert!(frames.iter().all(|frame| frame[0].is_finite()));
    }

    #[test]
    fn a_bigger_room_rings_for_longer() {
        let tail = |size: f32| -> f32 {
            let mut reverb = reverb();
            reverb.set_room(size, 0.35);
            let mut frames = vec![[0.0_f32; 2]; 96_000];
            frames[0] = [1.0, 1.0];
            reverb.process_insert(&mut frames, 1.0);
            frames[60_000..].iter().map(|frame| frame[0].abs()).sum()
        };
        assert!(
            tail(1.0) > tail(0.0) * 2.0,
            "room size did not change the tail: {} against {}",
            tail(1.0),
            tail(0.0)
        );
    }

    #[test]
    fn silence_in_is_silence_out() {
        let mut reverb = reverb();
        let send = vec![[0.0_f32; 2]; 4_096];
        let mut left = vec![0.0; 4_096];
        let mut right = vec![0.0; 4_096];
        reverb.process(&send, &mut left, &mut right);
        assert!(left.iter().chain(right.iter()).all(|value| *value == 0.0));
    }

    #[test]
    fn an_impulse_rings_on_long_after_it_stopped() {
        let (left, _) = impulse_response(96_000);
        // Nothing arrives instantly: the first reflection is a comb delay away.
        assert!(energy(&left[..1_000]) < 1e-6, "the reverb has no pre-delay");
        assert!(energy(&left[1_200..8_000]) > 0.01, "no early reflections");
        assert!(energy(&left[40_000..60_000]) > 1e-4, "the tail died too soon");
    }

    #[test]
    fn the_tail_decays_rather_than_sustaining_or_growing() {
        let (left, _) = impulse_response(192_000);
        let early = energy(&left[2_000..22_000]);
        let middle = energy(&left[60_000..80_000]);
        let late = energy(&left[160_000..180_000]);
        assert!(middle < early, "the tail grew: {early} then {middle}");
        assert!(late < middle * 0.5, "the tail is not decaying: {middle} then {late}");
    }

    #[test]
    fn the_two_channels_differ() {
        // One tail in both ears is a mono effect wearing a stereo label.
        let (left, right) = impulse_response(48_000);
        let difference: f32 = left.iter().zip(&right).map(|(l, r)| (l - r).abs()).sum();
        assert!(difference > 0.01, "the channels are identical");
    }

    #[test]
    fn the_high_end_is_absorbed_before_the_low_end() {
        // Real rooms lose treble first, which is what damping models.
        let (left, _) = impulse_response(96_000);
        let tilt = |window: &[f32]| -> f32 {
            let change: f32 = window.windows(2).map(|pair| (pair[1] - pair[0]).powi(2)).sum();
            let total: f32 = window.iter().map(|value| value * value).sum();
            change / (total + 1e-12)
        };
        let early = tilt(&left[2_000..12_000]);
        let late = tilt(&left[60_000..70_000]);
        assert!(late < early, "the tail brightened: {early} then {late}");
    }

    #[test]
    fn a_sustained_input_does_not_run_away() {
        let mut reverb = reverb();
        let send = vec![[1.0_f32, 1.0]; 192_000];
        let mut left = vec![0.0; 192_000];
        let mut right = vec![0.0; 192_000];
        reverb.process(&send, &mut left, &mut right);
        let peak = left
            .iter()
            .chain(right.iter())
            .fold(0.0_f32, |peak, value| peak.max(value.abs()));
        assert!(peak.is_finite() && peak < 4.0, "the reverb peaked at {peak}");
    }

    #[test]
    fn clearing_drops_the_tail() {
        let mut reverb = reverb();
        let mut send = vec![[0.0_f32; 2]; 8_000];
        send[0] = [1.0, 1.0];
        let (mut left, mut right) = (vec![0.0; 8_000], vec![0.0; 8_000]);
        reverb.process(&send, &mut left, &mut right);
        assert!(energy(&left) > 0.0);

        reverb.clear();
        let quiet = vec![[0.0_f32; 2]; 8_000];
        let (mut left, mut right) = (vec![0.0; 8_000], vec![0.0; 8_000]);
        reverb.process(&quiet, &mut left, &mut right);
        assert!(energy(&left) < f32::EPSILON, "the tail survived a clear");
    }

    #[test]
    fn it_adds_to_the_output_rather_than_replacing_it() {
        let mut reverb = reverb();
        let send = vec![[0.0_f32; 2]; 256];
        let (mut left, mut right) = (vec![0.25_f32; 256], vec![0.25_f32; 256]);
        reverb.process(&send, &mut left, &mut right);
        assert!(left.iter().all(|value| (*value - 0.25).abs() < f32::EPSILON));
    }

    #[test]
    fn mismatched_and_empty_buffers_are_safe() {
        let mut reverb = reverb();
        reverb.process(&[], &mut [], &mut []);
        let send = vec![[0.5_f32; 2]; 64];
        let (mut left, mut right) = (vec![0.0; 16], vec![0.0; 32]);
        reverb.process(&send, &mut left, &mut right);
        assert!(left.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn every_sample_rate_builds_a_usable_room() {
        for hz in [44_100, 48_000, 96_000] {
            let rate = SampleRate::new(hz).expect("a standard rate");
            let mut reverb = Reverb::new(rate);
            let frames = hz as usize;
            let mut send = vec![[0.0_f32; 2]; frames];
            send[0] = [1.0, 1.0];
            let (mut left, mut right) = (vec![0.0; frames], vec![0.0; frames]);
            reverb.process(&send, &mut left, &mut right);
            assert!(energy(&left) > 0.0, "{hz} Hz produced no reverb");
            assert!(left.iter().all(|value| value.is_finite()));
        }
    }
}
