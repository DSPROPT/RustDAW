use daw_core::SampleRate;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChannelStripParams {
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
}

impl Default for ChannelStripParams {
    fn default() -> Self {
        Self {
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
}

impl ChannelStrip {
    #[must_use]
    pub fn new(sample_rate: SampleRate, params: ChannelStripParams) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let rate = sample_rate.get() as f32;
        Self {
            params,
            sample_rate: rate,
            state: [ChannelState::default(); 2],
        }
    }

    pub fn set_params(&mut self, params: ChannelStripParams) {
        self.params = params;
    }

    #[must_use]
    pub const fn params(&self) -> ChannelStripParams {
        self.params
    }

    pub fn process_stereo(&mut self, frames: &mut [[f32; 2]]) {
        for frame in frames {
            frame[0] = process_sample(self.params, self.sample_rate, &mut self.state[0], frame[0]);
            frame[1] = process_sample(self.params, self.sample_rate, &mut self.state[1], frame[1]);
        }
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
