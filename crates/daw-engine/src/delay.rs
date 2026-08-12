#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss)]

//! A stereo delay for the channel strip.
//!
//! One delay line per channel, read a settable distance behind the write head,
//! with part of the output fed back in. That is the whole of it — the character
//! comes from what the feedback path does, not from the topology.
//!
//! The delay time is smoothed rather than jumped to. A delay line whose read
//! head teleports produces a click; one whose read head slides produces the
//! pitch bend a tape delay makes when its motor changes speed, which is the
//! sound people actually want from turning that knob.
//!
//! Real-time contract: [`Delay::process`] allocates nothing and does bounded
//! work per frame. The line is sized once by [`Delay::new`], which is not for
//! the audio thread.

use daw_core::SampleRate;

/// Longest delay the line can hold. A second is past the point where a guitar
/// delay is a delay rather than a looper, and the line costs memory in every
/// channel strip whether it is used or not.
pub const MAX_DELAY_SECONDS: f32 = 1.0;
/// Shortest delay worth having; below this it is a comb filter, not an echo.
pub const MIN_DELAY_SECONDS: f32 = 0.001;
/// Feedback ceiling. At 1.0 the line never decays and any DC or noise in it
/// grows without limit.
pub const MAX_FEEDBACK: f32 = 0.95;

/// How fast the read head slides to a new delay time, as a fraction of the
/// remaining distance per frame. Slow enough to bend audibly rather than jump.
const TIME_GLIDE: f32 = 0.0002;
/// Where the feedback path starts losing its top end. Every repeat comes back
/// duller than the last, as it does through any delay that is not purely
/// digital, and the tail settles into the mix instead of piling up on it.
const FEEDBACK_DAMPING_HZ: f32 = 4_000.0;
/// Pole of the DC blocker in the feedback path. A line that feeds itself will
/// otherwise accumulate any offset in its input without limit, however far
/// below one the feedback is set.
const DC_POLE: f32 = 0.9995;

pub struct Delay {
    buffer: Vec<[f32; 2]>,
    write: usize,
    /// The delay length actually in use, in frames, which chases the target.
    current_frames: f32,
    /// Whether a time has been asked for yet. The glide is for changing the
    /// time while playing; the first setting has nothing to glide from and
    /// must take effect at once, or switching a delay on slides its echo in
    /// from wherever the line happened to be left.
    primed: bool,
    sample_rate: f32,
    /// Feedback path filter state, per channel.
    damping: f32,
    damped: [f32; 2],
    dc_input: [f32; 2],
    dc_output: [f32; 2],
}

impl Delay {
    /// Sizes and zeroes the line. Allocates; call before the stream opens.
    #[must_use]
    pub fn new(sample_rate: SampleRate) -> Self {
        let rate = sample_rate.get().max(1) as f32;
        let frames = (rate * MAX_DELAY_SECONDS) as usize + 2;
        Self {
            buffer: vec![[0.0; 2]; frames],
            write: 0,
            current_frames: rate * 0.25,
            primed: false,
            sample_rate: rate,
            damping: 1.0 - (-std::f32::consts::TAU * FEEDBACK_DAMPING_HZ / rate).exp(),
            damped: [0.0; 2],
            dc_input: [0.0; 2],
            dc_output: [0.0; 2],
        }
    }

    /// Empties the line, for a stop or a seek.
    pub fn clear(&mut self) {
        self.buffer.fill([0.0; 2]);
        self.write = 0;
        self.primed = false;
        self.damped = [0.0; 2];
        self.dc_input = [0.0; 2];
        self.dc_output = [0.0; 2];
    }

    /// Processes a block in place.
    ///
    /// `mix` is the wet fraction: `0.0` leaves the signal alone, `1.0` returns
    /// only the echoes.
    pub fn process(&mut self, frames: &mut [[f32; 2]], seconds: f32, feedback: f32, mix: f32) {
        let length = self.buffer.len();
        if length < 2 || frames.is_empty() {
            return;
        }
        let target = (seconds.clamp(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS) * self.sample_rate)
            .clamp(1.0, (length - 1) as f32);
        let feedback = feedback.clamp(0.0, MAX_FEEDBACK);
        let mix = mix.clamp(0.0, 1.0);
        if !self.primed {
            self.current_frames = target;
            self.primed = true;
        }

        for frame in frames {
            // Slide towards the requested time rather than jumping to it.
            self.current_frames += (target - self.current_frames) * TIME_GLIDE;
            let delayed = self.read(self.current_frames);

            let dry = *frame;
            // What goes into the line is the input plus what came back out of
            // it, which is what makes an echo repeat rather than happen once.
            // The returning signal is filtered first: dulled, so each repeat is
            // darker than the last, and stripped of DC, so nothing accumulates
            // in a path that feeds itself.
            let mut returned = [0.0_f32; 2];
            for channel in 0..2 {
                self.damped[channel] +=
                    self.damping * (delayed[channel] - self.damped[channel]);
                let blocked = self.damped[channel] - self.dc_input[channel]
                    + DC_POLE * self.dc_output[channel];
                self.dc_input[channel] = self.damped[channel];
                self.dc_output[channel] = blocked;
                returned[channel] = blocked;
            }
            self.buffer[self.write] = [
                dry[0] + returned[0] * feedback,
                dry[1] + returned[1] * feedback,
            ];
            self.write = (self.write + 1) % length;

            frame[0] = dry[0] * (1.0 - mix) + delayed[0] * mix;
            frame[1] = dry[1] * (1.0 - mix) + delayed[1] * mix;
        }
    }

    /// Reads the line `frames_back` behind the write head, interpolating so a
    /// fractional distance — which is what sliding produces — is smooth.
    fn read(&self, frames_back: f32) -> [f32; 2] {
        let length = self.buffer.len();
        let back = frames_back.clamp(1.0, (length - 1) as f32);
        let whole = back.floor();
        let fraction = back - whole;
        let first = (self.write + length - whole as usize) % length;
        let second = (first + length - 1) % length;
        let (near, far) = (self.buffer[first], self.buffer[second]);
        [
            near[0] + (far[0] - near[0]) * fraction,
            near[1] + (far[1] - near[1]) * fraction,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn delay() -> Delay {
        Delay::new(SampleRate::DEFAULT)
    }

    fn energy(frames: &[[f32; 2]]) -> f32 {
        frames.iter().map(|frame| frame[0].abs() + frame[1].abs()).sum()
    }

    /// Feeds one impulse then silence, and returns the whole tail.
    fn impulse_response(seconds: f32, feedback: f32, mix: f32, frames: usize) -> Vec<[f32; 2]> {
        let mut delay = delay();
        let mut buffer = vec![[0.0_f32; 2]; frames];
        buffer[0] = [1.0, 1.0];
        delay.process(&mut buffer, seconds, feedback, mix);
        buffer
    }

    #[test]
    fn an_echo_arrives_after_the_time_it_was_given() {
        let seconds = 0.1;
        let expected = (seconds * SampleRate::DEFAULT.get() as f32) as usize;
        let response = impulse_response(seconds, 0.0, 1.0, 24_000);
        let loudest = response
            .iter()
            .enumerate()
            .max_by(|left, right| left.1[0].abs().total_cmp(&right.1[0].abs()))
            .map_or(0, |(index, _)| index);
        assert!(
            loudest.abs_diff(expected) < 64,
            "the echo landed at {loudest}, not {expected}"
        );
    }

    #[test]
    fn feedback_repeats_and_decays() {
        // Peaks, not summed magnitude: the DC blocker in the feedback path
        // gives each repeat a negative lobe, which adds to a sum of absolute
        // values while making no difference to how loud the echo is.
        let response = impulse_response(0.05, 0.6, 1.0, 48_000);
        let window = 2_400;
        let peak = |repeat: usize| -> f32 {
            response[window * repeat..window * (repeat + 1)]
                .iter()
                .fold(0.0_f32, |peak, frame| peak.max(frame[0].abs()))
        };
        assert!(peak(1) > 0.1 && peak(2) > 0.0, "the echo did not repeat");
        assert!(peak(2) < peak(1), "the repeats grew: {} then {}", peak(1), peak(2));
        assert!(peak(3) < peak(2), "the repeats grew: {} then {}", peak(2), peak(3));
        // And they are actually fading rather than creeping down.
        assert!(peak(3) < peak(1) * 0.5, "the tail is barely decaying");
    }

    #[test]
    fn feedback_cannot_run_away() {
        // Asking for more feedback than unity must not build a signal without
        // limit, and neither must a signal with an offset on it: a DC-heavy
        // input into a line that feeds itself is the classic way to blow one up.
        let mut delay = delay();
        let mut peak = 0.0_f32;
        for _ in 0..40 {
            // Fresh input each pass, so this measures the line's own feedback
            // rather than looping the output back round outside it.
            let mut buffer = vec![[0.5_f32; 2]; 24_000];
            delay.process(&mut buffer, 0.01, 10.0, 1.0);
            peak = buffer
                .iter()
                .fold(peak, |peak, frame| peak.max(frame[0].abs()));
        }
        assert!(peak.is_finite() && peak < 4.0, "the delay peaked at {peak}");
    }

    #[test]
    fn the_first_echo_lands_where_it_was_asked_to() {
        // The glide exists for turning the knob mid-take. Switching the delay
        // on must not slide the first echo in from whatever the line was last
        // set to, which would take seconds to settle.
        for seconds in [0.05_f32, 0.4] {
            let expected = (seconds * SampleRate::DEFAULT.get() as f32) as usize;
            let response = impulse_response(seconds, 0.0, 1.0, 48_000);
            let loudest = response
                .iter()
                .enumerate()
                .max_by(|left, right| left.1[0].abs().total_cmp(&right.1[0].abs()))
                .map_or(0, |(index, _)| index);
            assert!(
                loudest.abs_diff(expected) < 64,
                "a {seconds}s delay first echoed at {loudest}, not {expected}"
            );
        }
    }

    #[test]
    fn a_dry_mix_leaves_the_signal_alone() {
        let mut delay = delay();
        let original: Vec<[f32; 2]> = (0..512)
            .map(|index| [(index as f32 * 0.01).sin(); 2])
            .collect();
        let mut buffer = original.clone();
        delay.process(&mut buffer, 0.25, 0.5, 0.0);
        for (processed, dry) in buffer.iter().zip(&original) {
            assert!((processed[0] - dry[0]).abs() < 1e-6);
        }
    }

    #[test]
    fn a_wet_mix_replaces_it() {
        // Fully wet, the first block is the line's own silence, not the input.
        let mut delay = delay();
        let mut buffer = vec![[1.0_f32; 2]; 256];
        delay.process(&mut buffer, 0.5, 0.0, 1.0);
        assert!(
            energy(&buffer) < 1e-6,
            "a fully wet delay passed the dry signal through"
        );
    }

    #[test]
    fn the_time_is_clamped_to_what_the_line_can_hold() {
        // Both ends: a delay longer than the buffer would read out of bounds,
        // and one of zero would read the sample being written.
        let mut delay = delay();
        let mut buffer = vec![[0.3_f32; 2]; 1_024];
        for seconds in [-5.0, 0.0, 1e-9, MAX_DELAY_SECONDS * 4.0, f32::INFINITY] {
            delay.process(&mut buffer, seconds, 0.5, 0.5);
            assert!(
                buffer.iter().all(|frame| frame[0].is_finite()),
                "{seconds} seconds produced a non-finite sample"
            );
        }
    }

    #[test]
    fn clearing_drops_the_echoes() {
        let mut delay = delay();
        let mut buffer = vec![[0.0_f32; 2]; 8_000];
        buffer[0] = [1.0, 1.0];
        delay.process(&mut buffer, 0.05, 0.7, 1.0);
        assert!(energy(&buffer) > 0.0);

        delay.clear();
        let mut quiet = vec![[0.0_f32; 2]; 8_000];
        delay.process(&mut quiet, 0.05, 0.7, 1.0);
        assert!(energy(&quiet) < 1e-9, "an echo survived a clear");
    }

    #[test]
    fn an_empty_block_is_safe() {
        let mut delay = delay();
        delay.process(&mut [], 0.25, 0.5, 0.5);
    }

    #[test]
    fn every_sample_rate_builds_a_usable_line() {
        for hz in [44_100, 48_000, 96_000] {
            let rate = SampleRate::new(hz).expect("a standard rate");
            let mut delay = Delay::new(rate);
            let mut buffer = vec![[0.25_f32; 2]; 1_024];
            delay.process(&mut buffer, MAX_DELAY_SECONDS, 0.5, 0.5);
            assert!(buffer.iter().all(|frame| frame[0].is_finite()), "{hz} Hz");
        }
    }
}
