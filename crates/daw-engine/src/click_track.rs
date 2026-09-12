//! The click rendered as audio: wood-block hits on the beat, as a clip.
//!
//! The metronome plays *over* the song and stops being useful the moment the
//! click is switched off or the mix is exported. Two things want the click
//! *in* the song instead. A song that starts on the guitar needs a count-in the
//! guitarist hears before the first bar, whether the session is played here or
//! sent to the rest of the band; and a band rehearsing to exported stems needs
//! the click as a stem of its own. Both are rendered here, once, into a buffer
//! that becomes an ordinary clip.
//!
//! The sound is a struck wood block rather than the metronome's beep: a
//! fundamental and one inharmonic overtone that ring for a few tens of
//! milliseconds under a short burst of noise for the strike. The first beat of
//! a bar is the high block and the rest the low one, so the downbeat is heard
//! as well as counted.

use crate::metronome::MetronomeError;
use daw_core::SampleRate;
use daw_midi::TempoMap;
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

/// The longest click track that will be rendered. A song is minutes; this is
/// what stops a clip dragged to the far end of the timeline asking for
/// gigabytes.
const MAX_CLICK_TRACK_SECONDS: u64 = 60 * 60;

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
    // Both are bounded: 300 beats at 20 BPM at any real sample rate is well
    // inside u64.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let hits = (0..beats).map(|beat| {
        (
            (frames_per_beat * f64::from(beat)).round() as u64,
            beat == 0,
        )
    });
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let length = (frames_per_beat * f64::from(beats)).round() as u64;
    Ok(render_hits(sample_rate, hits, length))
}

/// Renders the click for a whole song: one wood block on every beat from bar
/// 1 to the first bar line at or after `end_frame`, following the tempo map.
///
/// The bars come from the map the way the ruler's do, so a song whose tempo
/// was detected as changing gets a click that changes with it. The render
/// stops on a bar line rather than mid-bar, and always covers at least one
/// bar, so an empty session still gets something to play to.
///
/// # Errors
///
/// Returns [`MetronomeError::InvalidMeter`] when `beats_per_bar` is zero.
pub fn render_click_track(
    sample_rate: SampleRate,
    tempo: &TempoMap,
    beats_per_bar: u16,
    end_frame: u64,
) -> Result<Vec<f32>, MetronomeError> {
    if beats_per_bar == 0 {
        return Err(MetronomeError::InvalidMeter);
    }
    let rate = sample_rate.get();
    let end_frame = end_frame.min(u64::from(rate) * MAX_CLICK_TRACK_SECONDS);
    let ticks_per_beat = u64::from(tempo.ticks_per_quarter());
    let beats_per_bar = u64::from(beats_per_bar);

    let mut hits = Vec::new();
    let mut beat = 0_u64;
    let length = loop {
        let frame = tempo.tick_to_frame(beat.saturating_mul(ticks_per_beat), rate);
        let bar_line = beat % beats_per_bar == 0;
        if bar_line && beat > 0 && frame >= end_frame {
            break frame;
        }
        // A map converts tick offsets through a u32 and stops advancing past
        // it; that is hours away, but a beat that does not move on is the end.
        if hits.last().is_some_and(|&(previous, _)| frame <= previous) {
            break frame;
        }
        hits.push((frame, bar_line));
        beat += 1;
    };
    Ok(render_hits(sample_rate, hits, length))
}

/// A buffer `length` frames long with a block struck at each hit: the high
/// block where the hit is accented, the low one elsewhere.
#[allow(clippy::cast_possible_truncation)]
fn render_hits(
    sample_rate: SampleRate,
    hits: impl IntoIterator<Item = (u64, bool)>,
    length: u64,
) -> Vec<f32> {
    let mut output = vec![0.0; length as usize];
    for (frame, accent) in hits {
        let Some(tail) = output.get_mut(frame as usize..) else {
            continue;
        };
        let (frequency, level) = if accent {
            (HIGH_BLOCK_HZ, ACCENT_LEVEL)
        } else {
            (LOW_BLOCK_HZ, REGULAR_LEVEL)
        };
        strike_wood_block(sample_rate, frequency, level, tail);
    }
    output
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
    fn a_click_track_runs_to_the_bar_line_after_the_song() {
        // 120 BPM, 4/4: a bar is 96,000 frames. A song ending mid-bar 3 gets
        // three whole bars of click.
        let click = render_click_track(SampleRate::DEFAULT, &TempoMap::constant(120.0), 4, 200_000)
            .unwrap();
        assert_eq!(click.len(), 288_000);
        for beat in 0..12 {
            let start = beat * 24_000;
            assert!(
                peak(&click[start..start + 480]) > 0.3,
                "beat {beat} is quiet"
            );
        }
        // Bar lines are accented, the beats between them are not.
        assert!(peak(&click[96_000..100_800]) > peak(&click[120_000..124_800]));
    }

    #[test]
    fn a_song_ending_on_a_bar_line_gets_no_extra_bar() {
        let click = render_click_track(SampleRate::DEFAULT, &TempoMap::constant(120.0), 4, 192_000)
            .unwrap();
        assert_eq!(click.len(), 192_000);
    }

    #[test]
    fn an_empty_session_still_gets_one_bar() {
        let click =
            render_click_track(SampleRate::DEFAULT, &TempoMap::constant(120.0), 3, 0).unwrap();
        assert_eq!(click.len(), 72_000);
    }

    #[test]
    fn the_click_follows_a_tempo_change() {
        // Two bars at 120, then 60: the fifth beat lands where the map says.
        let map = TempoMap::new(
            vec![
                daw_midi::TempoPoint {
                    tick: 0,
                    bpm: 120.0,
                },
                daw_midi::TempoPoint {
                    tick: 4 * u64::from(daw_midi::TICKS_PER_QUARTER),
                    bpm: 60.0,
                },
            ],
            daw_midi::TICKS_PER_QUARTER,
        );
        let click = render_click_track(SampleRate::DEFAULT, &map, 4, 200_000).unwrap();
        // One bar at 120 (2 s) and one at 60 (4 s), ending on the bar line.
        assert_eq!(click.len(), 288_000);
        let fifth_beat = 96_000;
        let sixth_beat = 96_000 + 48_000;
        assert!(peak(&click[fifth_beat..fifth_beat + 480]) > 0.3);
        assert!(peak(&click[sixth_beat..sixth_beat + 480]) > 0.3);
        // Halfway between them, at 120 there would have been a beat.
        assert!(peak(&click[fifth_beat + 24_000..fifth_beat + 24_480]) < 0.001);
    }

    #[test]
    fn a_click_track_is_capped_at_an_hour() {
        let click =
            render_click_track(SampleRate::DEFAULT, &TempoMap::constant(120.0), 4, u64::MAX)
                .unwrap();
        assert_eq!(click.len(), 48_000 * 60 * 60);
    }

    #[test]
    fn a_click_track_needs_a_meter() {
        assert_eq!(
            render_click_track(SampleRate::DEFAULT, &TempoMap::constant(120.0), 0, 1).unwrap_err(),
            MetronomeError::InvalidMeter
        );
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
