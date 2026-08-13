#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

//! Reference mastering: make one mix sound like another.
//!
//! Give it the mix you made and a record you wish yours sounded like, and it
//! measures the difference between them and closes it — loudness first, then
//! tone, then the peaks. It is not a mastering engineer. What it does is the
//! part of the job that is measurement rather than judgement, which is most of
//! the distance between a mix and a record.
//!
//! The order matters and is not arbitrary:
//!
//! 1. **Levels.** Match the loudness of the loudest parts of each song. Doing
//!    this first means the tonal comparison that follows is between two songs
//!    at the same level, rather than being contaminated by one being quieter.
//! 2. **Frequency.** Build the EQ curve that turns the target's average tone
//!    into the reference's, and apply it. See [`fir`].
//! 3. **Levels again, four times.** Equalising changes the loudness, so the
//!    match is re-measured and re-applied until it settles.
//! 4. **Peaks.** Limit to the ceiling, then restore whatever headroom the
//!    reference itself was sitting at. See [`limiter`].
//!
//! Mid and side are carried separately all the way through, each with its own
//! EQ curve. That is what matches stereo width: a reference with more energy
//! in the side channel produces a side curve that lifts it, and the mix widens
//! to match without anything being told to widen it.
//!
//! Ported from Matchering by Sergree <https://github.com/sergree/matchering>,
//! GPL-3.0, © 2016-2022. `RustDAW` is GPL-3.0-or-later, so the algorithm is
//! reimplemented here rather than called out to — see the crate README section
//! in the repository root.
//!
//! One deliberate difference: the reference implementation works internally at
//! 44.1 kHz. This one works at whatever rate the session runs at, because
//! `RustDAW` refuses to resample media and would rather master at 48 kHz than
//! convert twice around a fixed-rate stage. Every rate-dependent constant is
//! specified in milliseconds or hertz, so they carry over unchanged.

pub mod dsp;
pub mod fir;
pub mod interp;
pub mod limiter;
pub mod lowess;

use anyhow::{Result, bail};
use std::path::Path;

/// How the mastering runs.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// The ceiling, just under full scale.
    pub threshold: f32,
    /// The floor used wherever a division could otherwise be by zero.
    pub min_value: f32,
    /// The longest piece the song is analysed in, in seconds.
    pub max_piece_seconds: f32,
    /// The longest song accepted, in seconds.
    pub max_length_seconds: f32,
    /// How many times the level match is re-applied after equalising.
    pub rms_correction_steps: usize,
    pub fir: fir::Config,
    pub limiter: limiter::LimiterConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            // Just under full scale, by the same margin the reference
            // implementation leaves: 61 counts short of a 16-bit maximum.
            threshold: (32_768.0 - 61.0) / 32_768.0,
            min_value: 1e-6,
            max_piece_seconds: 15.0,
            max_length_seconds: 15.0 * 60.0,
            rms_correction_steps: 4,
            fir: fir::Config::default(),
            limiter: limiter::LimiterConfig::default(),
        }
    }
}

/// What the analysis found out about one song.
struct Levels {
    mid: Vec<f32>,
    side: Vec<f32>,
    mid_loudest: Vec<f32>,
    side_loudest: Vec<f32>,
    match_rms: f32,
    divisions: usize,
    piece_size: usize,
}

/// Splits a song into mid and side, divides it into pieces, and picks out the
/// loud ones.
fn analyze(frames: &[[f32; 2]], max_piece_size: usize) -> Levels {
    let (mid, side) = dsp::lr_to_ms(frames);
    let (divisions, piece_size) = dsp::piece_sizes(mid.len(), max_piece_size);

    let rmses = dsp::batch_rms(&mid, piece_size, divisions);
    let average = dsp::rms(&rmses);
    let (indices, match_rms) = dsp::loudest_pieces(&rmses, average);

    let mid_loudest = dsp::gather_pieces(&mid, &indices, piece_size);
    let side_loudest = dsp::gather_pieces(&side, &indices, piece_size);

    Levels {
        mid,
        side,
        mid_loudest,
        side_loudest,
        match_rms,
        divisions,
        piece_size,
    }
}

/// Masters `target` against `reference`, in place.
///
/// Both must be at `sample_rate` and stereo. The reference is not modified.
///
/// # Errors
///
/// Returns an error when either song is empty, too short to analyse, or longer
/// than [`Config::max_length_seconds`].
pub fn master(
    target: &mut Vec<[f32; 2]>,
    reference: &[[f32; 2]],
    sample_rate: f32,
    config: &Config,
) -> Result<()> {
    if target.is_empty() {
        bail!("there is nothing to master: the mix is empty");
    }
    if reference.is_empty() {
        bail!("the reference track is empty");
    }
    if sample_rate <= 0.0 {
        bail!("the sample rate must be positive");
    }

    let longest = (config.max_length_seconds * sample_rate) as usize;
    if target.len() > longest || reference.len() > longest {
        bail!(
            "mastering is limited to {:.0} minutes per track",
            config.max_length_seconds / 60.0
        );
    }
    // Two frames of analysis, or the average spectrum has nothing to average.
    let shortest = config.fir.fft_size * 2;
    if target.len() < shortest || reference.len() < shortest {
        bail!("both tracks must be at least {shortest} frames long to be analysed");
    }

    let max_piece_size = (config.max_piece_seconds * sample_rate) as usize;

    // 1. Levels. The reference comes up to the ceiling first, and whatever it
    //    was turned up by is given back at the very end so the master lands at
    //    the loudness the reference actually sits at.
    let mut reference = reference.to_vec();
    let final_amplitude = dsp::normalize(&mut reference, config.threshold, config.min_value, false);

    let mut target_levels = analyze(target, max_piece_size);
    let reference_levels = analyze(&reference, max_piece_size);
    drop(reference);

    let rms_coefficient =
        reference_levels.match_rms / config.min_value.max(target_levels.match_rms);
    dsp::amplify(&mut target_levels.mid, rms_coefficient);
    dsp::amplify(&mut target_levels.side, rms_coefficient);
    dsp::amplify(&mut target_levels.mid_loudest, rms_coefficient);
    dsp::amplify(&mut target_levels.side_loudest, rms_coefficient);

    // 2. Frequency, mid and side each with their own curve.
    let mid_fir = fir::matching_fir(
        &target_levels.mid_loudest,
        &reference_levels.mid_loudest,
        &config.fir,
    );
    let side_fir = fir::matching_fir(
        &target_levels.side_loudest,
        &reference_levels.side_loudest,
        &config.fir,
    );

    let mut result_mid = fir::convolve_same(&target_levels.mid, &mid_fir);
    let result_side = fir::convolve_same(&target_levels.side, &side_fir);
    let mut result = dsp::ms_to_lr(&result_mid, &result_side);
    drop(result_side);

    // 3. Levels again. Equalising moved them, so measure and reapply — against
    //    a clipped copy, because what matters is the level that survives the
    //    ceiling rather than the level on paper.
    for _ in 0..config.rms_correction_steps {
        let clipped = dsp::clip(&result_mid);
        let rmses = dsp::batch_rms(&clipped, target_levels.piece_size, target_levels.divisions);
        let average = dsp::rms(&rmses);
        let (_, match_rms) = dsp::loudest_pieces(&rmses, average);

        let coefficient = reference_levels.match_rms / config.min_value.max(match_rms);
        dsp::amplify(&mut result_mid, coefficient);
        dsp::amplify_stereo(&mut result, coefficient);
    }

    // 4. Peaks.
    limiter::limit(&mut result, config.threshold, sample_rate, &config.limiter);
    dsp::amplify_stereo(&mut result, final_amplitude);

    *target = result;
    Ok(())
}

/// Reads a reference track from a WAV file.
///
/// # Errors
///
/// Returns an error for unreadable files, and for a sample rate that does not
/// match the session — the engine does not resample, and a reference converted
/// on the way in would be measured through whatever the converter did to it.
pub fn load_reference(path: &Path, sample_rate: u32) -> Result<Vec<[f32; 2]>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    if spec.sample_rate != sample_rate {
        bail!(
            "the reference is {} Hz but the session is {sample_rate} Hz; \
             convert it first, for example with `ffmpeg -i in.wav -ar {sample_rate} out.wav`",
            spec.sample_rate
        );
    }
    if spec.channels != 1 && spec.channels != 2 {
        bail!(
            "the reference must be mono or stereo, not {}",
            spec.channels
        );
    }

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / f64::from(1_i32 << (spec.bits_per_sample - 1));
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| (f64::from(value) * scale) as f32))
                .collect::<Result<_, _>>()?
        }
    };

    let frames = if spec.channels == 1 {
        samples.into_iter().map(|sample| [sample, sample]).collect()
    } else {
        samples
            .chunks_exact(2)
            .map(|pair| [pair[0], pair[1]])
            .collect()
    };
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    const RATE: f32 = 48_000.0;

    /// A tone-plus-harmonics bed, at a chosen level and brightness.
    fn song(length: usize, level: f32, brightness: f32) -> Vec<[f32; 2]> {
        (0..length)
            .map(|index| {
                let time = index as f32 / RATE;
                let low = (TAU * 110.0 * time).sin();
                let high = (TAU * 6_000.0 * time).sin() * brightness;
                let left = (low + high) * level;
                // A little decorrelation, so there is side content to match.
                let right = (low + high * 0.8) * level;
                [left, right]
            })
            .collect()
    }

    #[test]
    fn a_quiet_mix_is_brought_up_to_the_reference() {
        let mut target = song(48_000 * 4, 0.05, 0.3);
        let reference = song(48_000 * 4, 0.5, 0.3);
        let before = dsp::rms(&dsp::lr_to_ms(&target).0);
        let reference_rms = dsp::rms(&dsp::lr_to_ms(&reference).0);

        master(&mut target, &reference, RATE, &Config::default()).expect("masters");

        let after = dsp::rms(&dsp::lr_to_ms(&target).0);
        assert!(after > before * 4.0, "the quiet mix should come up");
        let ratio = after / reference_rms;
        assert!(
            (0.5..2.0).contains(&ratio),
            "level should land near the reference: ratio {ratio}"
        );
    }

    #[test]
    fn the_output_respects_the_ceiling() {
        let mut target = song(48_000 * 4, 0.4, 0.5);
        let reference = song(48_000 * 4, 0.9, 0.3);
        master(&mut target, &reference, RATE, &Config::default()).expect("masters");
        let peak = dsp::peak_stereo(&target);
        assert!(peak <= 1.0, "a master must not exceed full scale: {peak}");
    }

    /// Deterministic broadband noise, so the spectra being compared are dense
    /// rather than a few spikes.
    fn noise(length: usize, tilt: f32) -> Vec<[f32; 2]> {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / 8_388_608.0) - 1.0
        };
        // A one-pole low-pass, mixed back against the raw noise. `tilt` of 0
        // is dull, 1 is flat and bright.
        let mut low_left = 0.0_f32;
        let mut low_right = 0.0_f32;
        (0..length)
            .map(|_| {
                let (raw_left, raw_right) = (next(), next());
                low_left += 0.02 * (raw_left - low_left);
                low_right += 0.02 * (raw_right - low_right);
                [
                    (low_left * 8.0 + raw_left * tilt) * 0.2,
                    (low_right * 8.0 + raw_right * tilt) * 0.2,
                ]
            })
            .collect()
    }

    #[test]
    fn a_dull_mix_is_brightened_towards_a_bright_reference() {
        let length = 48_000 * 4;
        let mut target = noise(length, 0.02);
        let reference = noise(length, 1.0);

        let brightness = |frames: &[[f32; 2]]| {
            let (mid, _) = dsp::lr_to_ms(frames);
            let spectrum = fir::average_spectrum(&mid, 4_096);
            let bin = |hz: f32| (hz / RATE * 4_096.0).round() as usize;
            let high: f32 = spectrum[bin(4_000.0)..bin(10_000.0)].iter().sum();
            let low: f32 = spectrum[bin(60.0)..bin(300.0)].iter().sum();
            high / low.max(1e-9)
        };

        let before = brightness(&target);
        master(&mut target, &reference, RATE, &Config::default()).expect("masters");
        let after = brightness(&target);
        let wanted = brightness(&reference);

        assert!(
            after > before * 3.0,
            "the master should be much brighter: {before} then {after}"
        );
        // And it should land near the reference rather than overshooting.
        let ratio = after / wanted;
        assert!(
            (0.4..2.5).contains(&ratio),
            "brightness should approach the reference: {after} against {wanted}"
        );
    }

    #[test]
    fn mastering_against_itself_barely_changes_anything() {
        // The strongest correctness check available without a reference
        // rendering: matching a song to itself must be close to a no-op, once
        // the deliberate normalisation to the ceiling is accounted for.
        let original = song(48_000 * 4, 0.5, 0.4);
        let mut target = original.clone();
        master(&mut target, &original, RATE, &Config::default()).expect("masters");

        let (original_mid, _) = dsp::lr_to_ms(&original);
        let (result_mid, _) = dsp::lr_to_ms(&target);
        let gain = dsp::rms(&result_mid) / dsp::rms(&original_mid);
        assert!(
            (0.9..1.1).contains(&gain),
            "self-match should keep the level, got {gain}"
        );

        // And the tone should be untouched: the matching curve is a ratio of a
        // spectrum with itself, which is flat.
        let spectrum_of = |frames: &[[f32; 2]]| {
            let (mid, _) = dsp::lr_to_ms(frames);
            fir::average_spectrum(&mid, 4_096)
        };
        let before = spectrum_of(&original);
        let after = spectrum_of(&target);
        let bin = |hz: f32| (hz / RATE * 4_096.0).round() as usize;
        for hz in [110.0_f32, 6_000.0] {
            let ratio = (after[bin(hz)] / before[bin(hz)].max(1e-9)) / gain;
            assert!(
                (0.7..1.4).contains(&ratio),
                "{hz} Hz should be unchanged, got a factor of {ratio}"
            );
        }
    }

    #[test]
    fn an_empty_or_overlong_input_is_refused() {
        let reference = song(48_000 * 2, 0.5, 0.3);
        let mut empty: Vec<[f32; 2]> = Vec::new();
        assert!(master(&mut empty, &reference, RATE, &Config::default()).is_err());

        let mut short = song(100, 0.5, 0.3);
        assert!(
            master(&mut short, &reference, RATE, &Config::default()).is_err(),
            "a clip shorter than the analysis window cannot be matched"
        );

        let mut target = song(48_000 * 2, 0.5, 0.3);
        let tiny_limit = Config {
            max_length_seconds: 1.0,
            ..Config::default()
        };
        assert!(master(&mut target, &reference, RATE, &tiny_limit).is_err());
    }
}
