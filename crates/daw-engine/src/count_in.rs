//! A count-in rendered as audio: one bar of wood-block clicks to put in front
//! of a song.
//!
//! The metronome plays *over* the song and stops being useful the moment the
//! click is switched off or the mix is exported. A song that starts on the
//! guitar needs its count *in* the song — the clicks the guitarist hears before
//! the first bar whether the session is played here, exported, or sent to the
//! rest of the band — so this renders the count once into a buffer that becomes
//! an ordinary clip.
//!
//! The sound is a struck wood block rather than the metronome's beep: a
//! fundamental and one inharmonic overtone that ring for a few tens of
//! milliseconds under a short burst of noise for the strike. Beat one is the
//! high block and the rest of the bar the low one, so the downbeat is heard as
//! well as counted.

use crate::metronome::MetronomeError;
use daw_core::SampleRate;
use std::f32::consts::TAU;

/// Fundamental of the high block, on the downbeat.
const HIGH_BLOCK_HZ: f32 = 1_000.0;
/// Fundamental of the low block, on every other beat.
const LOW_BLOCK_HZ: f32 = 700.0;
/// A block's second mode sits well above an octave and is not harmonic, which
/// is most of what makes it sound like wood rather than a beep.
const OVERTONE_RATIO: f32 = 2.63;
/// How long each part of the hit takes to fall to 1/e of its level.
const BODY_DECAY_SECONDS: f32 = 0.018;
const OVERTONE_DECAY_SECONDS: f32 = 0.006;
const STRIKE_DECAY_SECONDS: f32 = 0.001_5;
/// The block is silent well before this; it bounds the work per hit.
const HIT_LENGTH_SECONDS: f32 = 0.1;
const ACCENT_LEVEL: f32 = 0.8;
const REGULAR_LEVEL: f32 = 0.6;

/// Renders `beats` clicks at `tempo_bpm` into a mono buffer exactly one bar
/// long.
///
/// A beat is a quarter note, the unit the ruler and the tempo map count in, so
/// the buffer ends on the bar line where the song begins and a clip made from
/// it sits flush against the first bar.
///
/// # Errors
///
/// Returns [`MetronomeError::TempoOutOfRange`] outside 20–300 BPM, or
/// [`MetronomeError::InvalidMeter`] when `beats` is zero.
pub fn render_count_in(
    sample_rate: SampleRate,
    tempo_bpm: u16,
    beats: u16,
) -> Result<Vec<f32>, MetronomeError> {
    if !(20..=300).contains(&tempo_bpm) {
        return Err(MetronomeError::TempoOutOfRange);
    }
    if beats == 0 {
        return Err(MetronomeError::InvalidMeter);
    }
    let frames_per_beat = f64::from(sample_rate.get()) * 60.0 / f64::from(tempo_bpm);
    let bar_frames = frames_per_beat * f64::from(beats);
    // Both are bounded: 300 beats at 20 BPM at any real sample rate is well
    // inside usize.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let mut output = vec![0.0; bar_frames.round() as usize];
    for beat in 0..beats {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let start = (frames_per_beat * f64::from(beat)).round() as usize;
        let (frequency, level) = if beat == 0 {
            (HIGH_BLOCK_HZ, ACCENT_LEVEL)
        } else {
            (LOW_BLOCK_HZ, REGULAR_LEVEL)
        };
        strike_wood_block(sample_rate, frequency, level, &mut output[start..]);
    }
    Ok(output)
}

/// Adds one hit of a wood block tuned to `frequency` at the head of `output`,
/// peaking at `level`.
///
/// The hit is shaped first and scaled to its peak afterwards, so the level
/// asked for is the level heard whatever the partials and the strike happen
/// to add up to. Every hit uses the same noise, so rendering the same count
/// twice gives the same samples; a count-in that differed from one render to
/// the next would be a strange thing to have to explain.
fn strike_wood_block(sample_rate: SampleRate, frequency: f32, level: f32, output: &mut [f32]) {
    #[allow(clippy::cast_precision_loss)]
    let rate = sample_rate.get() as f32;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let length = ((HIT_LENGTH_SECONDS * rate) as usize).min(output.len());
    let mut noise = Noise::default();
    let mut hit = Vec::with_capacity(length);
    for index in 0..length {
        #[allow(clippy::cast_precision_loss)]
        let seconds = index as f32 / rate;
        let body = (seconds * frequency * TAU).sin() * (-seconds / BODY_DECAY_SECONDS).exp();
        let overtone = (seconds * frequency * OVERTONE_RATIO * TAU).sin()
            * (-seconds / OVERTONE_DECAY_SECONDS).exp();
        let strike = noise.next() * (-seconds / STRIKE_DECAY_SECONDS).exp();
        hit.push(body + 0.5 * overtone + 0.4 * strike);
    }
    let peak = hit
        .iter()
        .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
    if peak <= f32::EPSILON {
        return;
    }
    let scale = level / peak;
    for (sample, value) in output.iter_mut().zip(hit) {
        *sample += value * scale;
    }
}

/// A small xorshift generator: white enough for a millisecond of strike, and
/// deterministic, which a random source would not be.
struct Noise(u32);

impl Default for Noise {
    fn default() -> Self {
        Self(0x2545_F491)
    }
}

impl Noise {
    /// The next value in −1..=1.
    fn next(&mut self) -> f32 {
        let mut state = self.0;
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        self.0 = state;
        #[allow(clippy::cast_precision_loss)]
        {
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peak(buffer: &[f32]) -> f32 {
        buffer
            .iter()
            .fold(0.0_f32, |peak, sample| peak.max(sample.abs()))
    }

    #[test]
    fn a_bar_of_four_at_120_is_two_seconds() {
        let count = render_count_in(SampleRate::DEFAULT, 120, 4).unwrap();
        assert_eq!(count.len(), 96_000);
    }

    #[test]
    fn there_is_a_click_on_every_beat_and_silence_between() {
        let count = render_count_in(SampleRate::DEFAULT, 120, 4).unwrap();
        for beat in 0..4 {
            let start = beat * 24_000;
            assert!(
                peak(&count[start..start + 480]) > 0.3,
                "beat {beat} is quiet"
            );
            // A block rings for a few tens of milliseconds, not for the beat.
            assert!(
                peak(&count[start + 12_000..start + 24_000]) < 0.001,
                "beat {beat} rings on"
            );
        }
    }

    #[test]
    fn the_downbeat_is_louder_than_the_rest() {
        let count = render_count_in(SampleRate::DEFAULT, 120, 4).unwrap();
        assert!(peak(&count[..4_800]) > peak(&count[24_000..28_800]));
    }

    #[test]
    fn the_count_stays_inside_full_scale() {
        for tempo in [20, 96, 300] {
            let count = render_count_in(SampleRate::DEFAULT, tempo, 4).unwrap();
            assert!(peak(&count) <= 1.0);
        }
    }

    #[test]
    fn a_short_last_beat_is_clipped_to_the_bar() {
        // At 300 BPM a beat is 9,600 frames — longer than a hit, so the hit is
        // whole; the point is the render never writes past the bar.
        let count = render_count_in(SampleRate::DEFAULT, 300, 3).unwrap();
        assert_eq!(count.len(), 28_800);
    }

    #[test]
    fn rendering_twice_gives_the_same_samples() {
        let first = render_count_in(SampleRate::DEFAULT, 124, 4).unwrap();
        let second = render_count_in(SampleRate::DEFAULT, 124, 4).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn rejects_the_same_inputs_the_metronome_does() {
        assert_eq!(
            render_count_in(SampleRate::DEFAULT, 19, 4).unwrap_err(),
            MetronomeError::TempoOutOfRange
        );
        assert_eq!(
            render_count_in(SampleRate::DEFAULT, 120, 0).unwrap_err(),
            MetronomeError::InvalidMeter
        );
    }
}
