use daw_core::{SamplePosition, SampleRate};
use std::error::Error;
use std::f32::consts::TAU;
use std::fmt;

const CLICK_LENGTH_MS: u64 = 20;
const ACCENT_FREQUENCY_HZ: f32 = 1_760.0;
const REGULAR_FREQUENCY_HZ: f32 = 1_320.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetronomeError {
    TempoOutOfRange,
    InvalidMeter,
}

impl fmt::Display for MetronomeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TempoOutOfRange => formatter.write_str("tempo must be between 20 and 300 BPM"),
            Self::InvalidMeter => formatter.write_str("meter must contain at least one beat"),
        }
    }
}

impl Error for MetronomeError {}

/// A synthesized, sample-position-driven metronome.
///
/// `render_mono` performs no allocation and is safe to call with any block
/// size. Beat positions are calculated from absolute time, avoiding drift.
#[derive(Clone, Copy, Debug)]
pub struct Metronome {
    sample_rate: SampleRate,
    tempo_bpm: u16,
    beats_per_bar: u16,
    beat_unit: u16,
    level: f32,
    enabled: bool,
}

impl Metronome {
    /// Creates a metronome with an integer tempo and numerator.
    ///
    /// # Errors
    ///
    /// Returns [`MetronomeError::TempoOutOfRange`] outside 20–300 BPM, or
    /// [`MetronomeError::InvalidMeter`] when `beats_per_bar` is zero.
    pub fn new(
        sample_rate: SampleRate,
        tempo_bpm: u16,
        beats_per_bar: u16,
    ) -> Result<Self, MetronomeError> {
        Self::with_meter(sample_rate, tempo_bpm, beats_per_bar, 4)
    }

    /// Creates a metronome for the supplied time-signature numerator and
    /// denominator. Tempo is measured in quarter notes per minute.
    ///
    /// # Errors
    ///
    /// Returns [`MetronomeError::TempoOutOfRange`] outside 20–300 BPM, or
    /// [`MetronomeError::InvalidMeter`] for a zero numerator/denominator.
    pub fn with_meter(
        sample_rate: SampleRate,
        tempo_bpm: u16,
        beats_per_bar: u16,
        beat_unit: u16,
    ) -> Result<Self, MetronomeError> {
        if !(20..=300).contains(&tempo_bpm) {
            return Err(MetronomeError::TempoOutOfRange);
        }
        if beats_per_bar == 0 || beat_unit == 0 {
            return Err(MetronomeError::InvalidMeter);
        }
        Ok(Self {
            sample_rate,
            tempo_bpm,
            beats_per_bar,
            beat_unit,
            level: 0.4,
            enabled: true,
        })
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn set_level(&mut self, level: f32) {
        self.level = level.clamp(0.0, 1.0);
    }

    #[must_use]
    pub fn frames_per_beat(&self) -> f64 {
        f64::from(self.sample_rate.get()) * 60.0 * 4.0
            / (f64::from(self.tempo_bpm) * f64::from(self.beat_unit))
    }

    /// Adds the click into an existing mono output buffer.
    pub fn render_mono(&self, block_start: SamplePosition, output: &mut [f32]) {
        if !self.enabled || self.level == 0.0 {
            return;
        }

        let click_frames = u64::from(self.sample_rate.get()) * CLICK_LENGTH_MS / 1_000;
        let beat_denominator = u128::from(self.sample_rate.get()) * 60 * 4;
        let tempo = u128::from(self.tempo_bpm) * u128::from(self.beat_unit);

        for (offset, sample) in output.iter_mut().enumerate() {
            let absolute_frame = block_start.get().saturating_add(offset as u64);
            let beat_index = u64::try_from(u128::from(absolute_frame) * tempo / beat_denominator)
                .unwrap_or(u64::MAX);
            let beat_start =
                u64::try_from((u128::from(beat_index) * beat_denominator + tempo / 2) / tempo)
                    .unwrap_or(u64::MAX);
            let age = absolute_frame.saturating_sub(beat_start);

            if age >= click_frames {
                continue;
            }

            let accent = beat_index % u64::from(self.beats_per_bar) == 0;
            let frequency = if accent {
                ACCENT_FREQUENCY_HZ
            } else {
                REGULAR_FREQUENCY_HZ
            };
            // Both values are bounded to a few milliseconds here. The f32
            // conversion is intentional DSP math and cannot lose useful range.
            #[allow(clippy::cast_precision_loss)]
            let phase = age as f32 * frequency * TAU / self.sample_rate.get() as f32;
            #[allow(clippy::cast_precision_loss)]
            let envelope = 1.0 - age as f32 / click_frames as f32;
            *sample += phase.sin() * envelope * self.level;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn energy(buffer: &[f32]) -> f32 {
        buffer.iter().map(|sample| sample.abs()).sum()
    }

    #[test]
    fn rejects_unsupported_tempo() {
        assert_eq!(
            Metronome::new(SampleRate::DEFAULT, 301, 4).unwrap_err(),
            MetronomeError::TempoOutOfRange
        );
    }

    #[test]
    fn click_occurs_at_sample_zero() {
        let metronome = Metronome::new(SampleRate::DEFAULT, 120, 4).unwrap();
        let mut buffer = [0.0; 1_024];
        metronome.render_mono(SamplePosition::default(), &mut buffer);
        assert!(energy(&buffer) > 1.0);
    }

    #[test]
    fn silence_between_clicks_is_untouched() {
        let metronome = Metronome::new(SampleRate::DEFAULT, 120, 4).unwrap();
        let mut buffer = [0.0; 256];
        metronome.render_mono(SamplePosition::new(4_000), &mut buffer);
        assert!(energy(&buffer) < f32::EPSILON);
    }

    #[test]
    fn beat_is_aligned_across_block_boundary() {
        let metronome = Metronome::new(SampleRate::DEFAULT, 120, 4).unwrap();
        let mut before = [0.0; 128];
        let mut after = [0.0; 128];
        metronome.render_mono(SamplePosition::new(23_872), &mut before);
        metronome.render_mono(SamplePosition::new(24_000), &mut after);
        assert!(energy(&before) < f32::EPSILON);
        assert!(energy(&after) > 1.0);
    }

    #[test]
    fn disabled_click_does_not_modify_output() {
        let mut metronome = Metronome::new(SampleRate::DEFAULT, 120, 4).unwrap();
        metronome.set_enabled(false);
        let mut buffer = [0.25; 128];
        metronome.render_mono(SamplePosition::default(), &mut buffer);
        assert!(
            buffer
                .iter()
                .all(|sample| (*sample - 0.25).abs() < f32::EPSILON)
        );
    }

    #[test]
    fn eighth_note_meter_ticks_twice_as_fast_as_quarter_note_meter() {
        let quarter = Metronome::with_meter(SampleRate::DEFAULT, 120, 4, 4).unwrap();
        let eighth = Metronome::with_meter(SampleRate::DEFAULT, 120, 6, 8).unwrap();
        assert!((quarter.frames_per_beat() - 24_000.0).abs() < f64::EPSILON);
        assert!((eighth.frames_per_beat() - 12_000.0).abs() < f64::EPSILON);
    }
}
