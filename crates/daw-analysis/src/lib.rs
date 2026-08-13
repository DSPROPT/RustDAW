#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

//! Native audio analysis: tempo, beats and a tempo map, in Rust.
//!
//! This replaces the imported beat grid. The point is not only independence
//! from the Python side — it is that a grid computed per beat carries the
//! tracker's own error as apparent tempo change, so a song at a steady 123 BPM
//! arrives as hundreds of tempi between 120 and 176. Here a global tempo is
//! decided first and the beats are fitted to it, so a steady song produces one
//! tempo and a song that really moves produces the changes that are really
//! there.

pub mod beats;
pub mod chords;
pub mod chroma;
pub mod fft;
pub mod onset;
pub mod pitch;

use anyhow::{Context, Result};
use std::path::Path;

pub use beats::BeatAnalysis;
pub use onset::OnsetEnvelope;
pub use pitch::{Pitch, detect};

use daw_midi::TempoMap;

/// Result of analysing one file.
#[derive(Clone, Debug)]
pub struct SongAnalysis {
    pub beats: BeatAnalysis,
    pub tempo_map: TempoMap,
    /// Length of the analysed audio in seconds.
    pub duration_seconds: f64,
}

impl SongAnalysis {
    #[must_use]
    pub fn bpm(&self) -> f64 {
        self.beats.bpm
    }
}

/// Analyses mono samples.
///
/// `tempo_tolerance_bpm` decides how much the tempo must move before the map
/// records a change; 3 BPM keeps genuine rubato while discarding tracking
/// noise.
#[must_use]
pub fn analyse_samples(
    samples: &[f32],
    sample_rate: u32,
    tempo_tolerance_bpm: f64,
) -> SongAnalysis {
    let envelope = onset::onset_envelope(samples, sample_rate);
    let beats = beats::analyse(&envelope);
    let tempo_map = if beats.is_usable() {
        TempoMap::from_beat_times(&beats.beat_times, tempo_tolerance_bpm)
    } else {
        TempoMap::constant(beats.bpm)
    };
    let duration_seconds = if sample_rate == 0 {
        0.0
    } else {
        samples.len() as f64 / f64::from(sample_rate)
    };
    SongAnalysis {
        beats,
        tempo_map,
        duration_seconds,
    }
}

/// Analyses a WAV file, mixing it down to mono first.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or decoded.
pub fn analyse_wav(path: &Path, tempo_tolerance_bpm: f64) -> Result<SongAnalysis> {
    let (samples, sample_rate) = read_wav_mono(path)?;
    Ok(analyse_samples(&samples, sample_rate, tempo_tolerance_bpm))
}

/// Reads a WAV file as mono f32 at its own sample rate.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or a sample cannot be read.
pub fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels).max(1);

    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(Result::ok).collect(),
        hound::SampleFormat::Int => {
            let scale = 2.0_f32.powi(i32::from(spec.bits_per_sample).saturating_sub(1));
            reader
                .samples::<i32>()
                .filter_map(Result::ok)
                .map(|sample| sample as f32 / scale)
                .collect()
        }
    };

    let mono = interleaved
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    Ok((mono, spec.sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_synthetic_click_track_is_measured_correctly() {
        // A 100 BPM click: one burst every 0.6 s.
        const RATE: u32 = 48_000;
        let mut samples = vec![0.0_f32; RATE as usize * 12];
        let mut noise = 99_991_u32;
        let mut position = 0.0_f64;
        while position < 12.0 {
            let start = (position * f64::from(RATE)) as usize;
            for offset in 0..1_200 {
                let Some(slot) = samples.get_mut(start + offset) else {
                    break;
                };
                noise = noise.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let white = (noise >> 16) as f32 / 32_768.0 - 1.0;
                let decay = 1.0 - offset as f32 / 1_200.0;
                *slot = white * decay * decay * 0.8;
            }
            position += 0.6;
        }

        let analysis = analyse_samples(&samples, RATE, 3.0);
        assert!(
            (analysis.bpm() - 100.0).abs() < 2.0,
            "measured {:.2} BPM, expected 100",
            analysis.bpm()
        );
        assert!(analysis.beats.is_usable());
        assert!(
            analysis.tempo_map.is_constant(),
            "a metronome must not produce a varying tempo map, got {} points",
            analysis.tempo_map.points().len()
        );
    }

    #[test]
    fn silence_falls_back_without_pretending_to_know() {
        let analysis = analyse_samples(&vec![0.0; 48_000 * 4], 48_000, 3.0);
        assert!(!analysis.beats.is_usable());
        assert!(analysis.tempo_map.is_constant());
    }

    #[test]
    fn duration_is_reported_from_the_samples() {
        let analysis = analyse_samples(&vec![0.0; 96_000], 48_000, 3.0);
        assert!((analysis.duration_seconds - 2.0).abs() < 1e-9);
        assert!((analyse_samples(&[], 0, 3.0).duration_seconds).abs() < f64::EPSILON);
    }
}
