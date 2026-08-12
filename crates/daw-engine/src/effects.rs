use daw_core::SampleRate;

use crate::delay::{self, Delay};
use crate::reverb::Reverb;

/// A strip is a rack of independent modules and each one needs its own switch;
/// grouping them into an enum would stop them being usable together.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChannelStripParams {
    pub nam_enabled: bool,
    pub nam_input_db: f32,
    pub nam_output_db: f32,
    /// The amp's own noise gate, ahead of it rather than after it. At
    /// [`crate::OPEN_THRESHOLD_DB`] it is off.
    pub nam_gate_db: f32,
    /// The amp's tone stack, which is part of the amp and separate from the
    /// channel EQ below. Controls run 0 to 10 and are flat at 5.
    pub nam_tone_enabled: bool,
    pub nam_bass: f32,
    pub nam_middle: f32,
    pub nam_treble: f32,
    /// Levels the capture against its own measured loudness, so swapping amps
    /// does not mean re-dialling the gain staging.
    pub nam_normalize: bool,
    pub eq_enabled: bool,
    pub low_db: f32,
    pub mid_db: f32,
    pub high_db: f32,
    pub compressor_enabled: bool,
    pub compressor_threshold_db: f32,
    pub compressor_ratio: f32,
    pub compressor_attack_ms: f32,
    pub compressor_release_ms: f32,
    pub compressor_makeup_db: f32,
    pub gate_enabled: bool,
    pub gate_threshold_db: f32,
    pub gate_release_ms: f32,
    pub delay_enabled: bool,
    /// Time between repeats, in milliseconds.
    pub delay_time_ms: f32,
    /// How much of each repeat is fed back in, `0` to `1`.
    pub delay_feedback: f32,
    /// Wet fraction, `0` to `1`.
    pub delay_mix: f32,
    pub reverb_enabled: bool,
    /// How long the tail runs, `0` to `1`.
    pub reverb_size: f32,
    /// How fast the tail's top end is absorbed, `0` to `1`.
    pub reverb_damping: f32,
    pub reverb_mix: f32,
}

impl Default for ChannelStripParams {
    fn default() -> Self {
        Self {
            nam_enabled: false,
            nam_input_db: 0.0,
            nam_output_db: 0.0,
            nam_gate_db: crate::OPEN_THRESHOLD_DB,
            nam_tone_enabled: false,
            nam_bass: crate::tone::FLAT,
            nam_middle: crate::tone::FLAT,
            nam_treble: crate::tone::FLAT,
            nam_normalize: false,
            eq_enabled: false,
            low_db: 0.0,
            mid_db: 0.0,
            high_db: 0.0,
            compressor_enabled: false,
            compressor_threshold_db: -18.0,
            compressor_ratio: 4.0,
            compressor_attack_ms: 10.0,
            compressor_release_ms: 120.0,
            compressor_makeup_db: 0.0,
            gate_enabled: false,
            gate_threshold_db: -45.0,
            gate_release_ms: 120.0,
            delay_enabled: false,
            delay_time_ms: 350.0,
            delay_feedback: 0.35,
            delay_mix: 0.25,
            reverb_enabled: false,
            reverb_size: 0.6,
            reverb_damping: 0.4,
            reverb_mix: 0.2,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct ChannelState {
    low: f32,
    low_mid: f32,
    compressor_envelope: f32,
    gate_envelope: f32,
    gate_gain: f32,
}

pub struct ChannelStrip {
    params: ChannelStripParams,
    sample_rate: f32,
    state: [ChannelState; 2],
    /// Time-based modules, which unlike the rest of the strip carry a history
    /// and so own buffers. Built with the strip so the audio thread never has
    /// to allocate one, whether or not they are switched on.
    delay: Delay,
    reverb: Reverb,
}

impl ChannelStrip {
    #[must_use]
    pub fn new(sample_rate: SampleRate, params: ChannelStripParams) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let rate = sample_rate.get() as f32;
        let mut reverb = Reverb::new(sample_rate);
        reverb.set_room(params.reverb_size, params.reverb_damping);
        Self {
            params,
            sample_rate: rate,
            state: [ChannelState::default(); 2],
            delay: Delay::new(sample_rate),
            reverb,
        }
    }

    pub fn set_params(&mut self, params: ChannelStripParams) {
        self.params = params;
        // Coefficients only; nothing here resizes anything.
        self.reverb
            .set_room(params.reverb_size, params.reverb_damping);
    }

    /// Drops the tails, for a stop or a seek. Carrying an echo across an edit
    /// smears the old position over the new one.
    pub fn clear_tails(&mut self) {
        self.delay.clear();
        self.reverb.clear();
    }

    #[must_use]
    pub const fn params(&self) -> ChannelStripParams {
        self.params
    }

    pub fn process_stereo(&mut self, frames: &mut [[f32; 2]]) {
        for frame in frames.iter_mut() {
            frame[0] = process_sample(self.params, self.sample_rate, &mut self.state[0], frame[0]);
            frame[1] = process_sample(self.params, self.sample_rate, &mut self.state[1], frame[1]);
        }
        // Time-based modules go last, so the echoes and the room are of the
        // finished tone rather than of whatever it was before the EQ and the
        // compressor got to it. Delay before reverb: repeats happen in the
        // room, not the room in the repeats.
        if self.params.delay_enabled {
            self.delay.process(
                frames,
                self.params.delay_time_ms / 1_000.0,
                self.params.delay_feedback,
                self.params.delay_mix,
            );
        }
        if self.params.reverb_enabled {
            self.reverb.process_insert(frames, self.params.reverb_mix);
        }
    }

    /// The delay's longest setting, in milliseconds, for a control's range.
    #[must_use]
    pub fn max_delay_ms() -> f32 {
        delay::MAX_DELAY_SECONDS * 1_000.0
    }
}

fn process_sample(
    params: ChannelStripParams,
    sample_rate: f32,
    state: &mut ChannelState,
    mut sample: f32,
) -> f32 {
    if params.gate_enabled {
        let detector_release = coefficient(params.gate_release_ms, sample_rate);
        state.gate_envelope = sample.abs().max(state.gate_envelope * detector_release);
        let target = f32::from(linear_to_db(state.gate_envelope) >= params.gate_threshold_db);
        let smoothing = if target > state.gate_gain {
            coefficient(2.0, sample_rate)
        } else {
            detector_release
        };
        state.gate_gain = target + smoothing * (state.gate_gain - target);
        sample *= state.gate_gain;
    }

    if params.eq_enabled {
        let low_coefficient = low_pass_coefficient(180.0, sample_rate);
        let mid_coefficient = low_pass_coefficient(2_500.0, sample_rate);
        state.low += low_coefficient * (sample - state.low);
        state.low_mid += mid_coefficient * (sample - state.low_mid);
        let low = state.low;
        let mid = state.low_mid - state.low;
        let high = sample - state.low_mid;
        sample = low * db_to_linear(params.low_db)
            + mid * db_to_linear(params.mid_db)
            + high * db_to_linear(params.high_db);
    }

    if params.compressor_enabled {
        let absolute = sample.abs();
        let time = if absolute > state.compressor_envelope {
            params.compressor_attack_ms
        } else {
            params.compressor_release_ms
        };
        let smoothing = coefficient(time, sample_rate);
        state.compressor_envelope = absolute + smoothing * (state.compressor_envelope - absolute);
        let input_db = linear_to_db(state.compressor_envelope);
        let over_db = (input_db - params.compressor_threshold_db).max(0.0);
        let reduction_db = over_db * (1.0 - 1.0 / params.compressor_ratio.max(1.0));
        sample *= db_to_linear(params.compressor_makeup_db - reduction_db);
    }
    sample
}

fn coefficient(milliseconds: f32, sample_rate: f32) -> f32 {
    (-1.0 / (milliseconds.max(0.1) * 0.001 * sample_rate)).exp()
}

fn low_pass_coefficient(frequency: f32, sample_rate: f32) -> f32 {
    1.0 - (-std::f32::consts::TAU * frequency / sample_rate).exp()
}

fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

fn linear_to_db(linear: f32) -> f32 {
    20.0 * linear.max(0.000_000_001).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bypass_is_bit_transparent() {
        let mut strip = ChannelStrip::new(SampleRate::DEFAULT, ChannelStripParams::default());
        let mut frames = [[0.25, -0.5], [0.75, -0.125]];
        let original = frames;
        strip.process_stereo(&mut frames);
        assert_eq!(frames, original);
    }

    #[test]
    fn gate_suppresses_a_quiet_signal() {
        let params = ChannelStripParams {
            gate_enabled: true,
            gate_threshold_db: -30.0,
            ..ChannelStripParams::default()
        };
        let mut strip = ChannelStrip::new(SampleRate::DEFAULT, params);
        let mut frames = vec![[0.001, 0.001]; 4_800];
        strip.process_stereo(&mut frames);
        assert!(frames.last().unwrap()[0].abs() < 0.000_01);
    }

    #[test]
    fn compressor_reduces_sustained_signal() {
        let params = ChannelStripParams {
            compressor_enabled: true,
            compressor_threshold_db: -20.0,
            compressor_ratio: 10.0,
            compressor_attack_ms: 1.0,
            ..ChannelStripParams::default()
        };
        let mut strip = ChannelStrip::new(SampleRate::DEFAULT, params);
        let mut frames = vec![[1.0, 1.0]; 4_800];
        strip.process_stereo(&mut frames);
        assert!(frames.last().unwrap()[0] < 0.2);
    }

    #[test]
    fn the_delay_only_runs_when_it_is_switched_in() {
        let quiet = |params: ChannelStripParams| -> f32 {
            let mut strip = ChannelStrip::new(SampleRate::DEFAULT, params);
            let mut frames = vec![[0.0_f32; 2]; 48_000];
            frames[0] = [1.0, 1.0];
            strip.process_stereo(&mut frames);
            // Well past the delay time, where only an echo could be.
            frames[20_000..]
                .iter()
                .fold(0.0_f32, |peak, frame| peak.max(frame[0].abs()))
        };
        let off = ChannelStripParams::default();
        let on = ChannelStripParams {
            delay_enabled: true,
            delay_time_ms: 350.0,
            delay_feedback: 0.5,
            delay_mix: 0.5,
            ..ChannelStripParams::default()
        };
        assert!(quiet(off) < 1e-6, "a bypassed delay still echoed");
        assert!(quiet(on) > 1e-3, "the delay produced no echo");
    }

    #[test]
    fn the_reverb_only_runs_when_it_is_switched_in() {
        let tail = |params: ChannelStripParams| -> f32 {
            let mut strip = ChannelStrip::new(SampleRate::DEFAULT, params);
            let mut frames = vec![[0.0_f32; 2]; 48_000];
            frames[0] = [1.0, 1.0];
            strip.process_stereo(&mut frames);
            frames[4_000..].iter().map(|frame| frame[0].abs()).sum()
        };
        let off = ChannelStripParams::default();
        let on = ChannelStripParams {
            reverb_enabled: true,
            reverb_mix: 0.5,
            ..ChannelStripParams::default()
        };
        assert!(tail(off) < 1e-6, "a bypassed reverb still rang");
        assert!(tail(on) > 1e-4, "the reverb produced no tail");
    }

    #[test]
    fn clearing_tails_drops_the_echoes_but_not_the_settings() {
        // Stopping must not carry the old position's echoes over the new one.
        let params = ChannelStripParams {
            delay_enabled: true,
            delay_time_ms: 100.0,
            delay_feedback: 0.7,
            delay_mix: 1.0,
            reverb_enabled: true,
            reverb_mix: 0.5,
            ..ChannelStripParams::default()
        };
        let mut strip = ChannelStrip::new(SampleRate::DEFAULT, params);
        let mut frames = vec![[0.0_f32; 2]; 24_000];
        frames[0] = [1.0, 1.0];
        strip.process_stereo(&mut frames);
        assert!(frames.iter().any(|frame| frame[0].abs() > 1e-3));

        strip.clear_tails();
        let mut quiet = vec![[0.0_f32; 2]; 24_000];
        strip.process_stereo(&mut quiet);
        let left: f32 = quiet.iter().map(|frame| frame[0].abs()).sum();
        assert!(left < 1e-6, "a tail survived the clear");
        assert_eq!(strip.params(), params, "clearing must not change settings");
    }

    #[test]
    fn a_bypassed_strip_is_still_bit_transparent_with_the_new_modules() {
        // Every module off must leave the signal exactly as it arrived, or
        // "non-destructive" is not true.
        let mut strip = ChannelStrip::new(SampleRate::DEFAULT, ChannelStripParams::default());
        let mut frames = [[0.25, -0.5], [0.75, -0.125], [0.0, 1.0]];
        let original = frames;
        strip.process_stereo(&mut frames);
        assert_eq!(frames, original);
    }

    #[test]
    fn eq_boosts_low_frequency_energy() {
        let params = ChannelStripParams {
            eq_enabled: true,
            low_db: 6.0,
            ..ChannelStripParams::default()
        };
        let mut strip = ChannelStrip::new(SampleRate::DEFAULT, params);
        let mut frames = vec![[1.0, 1.0]; 4_800];
        strip.process_stereo(&mut frames);
        assert!(frames.last().unwrap()[0] > 1.9);
    }
}
