//! Deterministic offline stereo rendering for `RustDAW` sessions.

use anyhow::{Context, Result, bail};
use daw_engine::{ChannelStrip, ChannelStripParams, NoiseGate, ToneStack};
use daw_project::ProjectDocument;
use daw_nam::NamProcessor;

/// The loudness a normalised capture is brought to, matching the runtime so an
/// export sounds like what was monitored.
const NORMALIZE_TARGET_DB: f64 = -18.0;
use std::path::Path;

/// Renders all unmuted clips to a stereo 24-bit WAV file.
///
/// # Errors
///
/// Returns an error for missing/unsupported media, sample-rate mismatches, or
/// output filesystem failures.
pub fn export_stereo(project: &ProjectDocument, destination: &Path) -> Result<u64> {
    let end_frame = project
        .tracks
        .iter()
        .flat_map(|track| &track.clips)
        .map(|clip| clip.end_frame)
        .max()
        .unwrap_or(0);
    let frame_count = usize::try_from(end_frame).context("session is too long to render")?;
    let mut mix = vec![[0.0_f32; 2]; frame_count];

    let any_solo = project.tracks.iter().any(|track| track.solo);
    for track in project
        .tracks
        .iter()
        .filter(|track| !track.muted && (!any_solo || track.solo))
    {
        let gain = db_to_gain(track.gain_db);
        let mut processor = ChannelStrip::new(
            daw_core::SampleRate::new(project.sample_rate)
                .context("project sample rate cannot be zero")?,
            ChannelStripParams {
                nam_enabled: track.effects.nam_enabled,
                nam_input_db: track.effects.nam_input_db,
                nam_output_db: track.effects.nam_output_db,
                nam_gate_db: track.effects.nam_gate_db,
                nam_tone_enabled: track.effects.nam_tone_enabled,
                nam_bass: track.effects.nam_bass,
                nam_middle: track.effects.nam_middle,
                nam_treble: track.effects.nam_treble,
                nam_normalize: track.effects.nam_normalize,
                delay_enabled: track.effects.delay_enabled,
                delay_time_ms: track.effects.delay_time_ms,
                delay_feedback: track.effects.delay_feedback,
                delay_mix: track.effects.delay_mix,
                reverb_enabled: track.effects.reverb_enabled,
                reverb_size: track.effects.reverb_size,
                reverb_damping: track.effects.reverb_damping,
                reverb_mix: track.effects.reverb_mix,
                eq_enabled: track.effects.eq_enabled,
                low_db: track.effects.low_db,
                mid_db: track.effects.mid_db,
                high_db: track.effects.high_db,
                compressor_enabled: track.effects.compressor_enabled,
                compressor_threshold_db: track.effects.compressor_threshold_db,
                compressor_ratio: track.effects.compressor_ratio,
                compressor_attack_ms: track.effects.compressor_attack_ms,
                compressor_release_ms: track.effects.compressor_release_ms,
                compressor_makeup_db: track.effects.compressor_makeup_db,
                gate_enabled: track.effects.gate_enabled,
                gate_threshold_db: track.effects.gate_threshold_db,
                gate_release_ms: track.effects.gate_release_ms,
            },
        );
        let sample_rate = daw_core::SampleRate::new(project.sample_rate)
            .context("project sample rate cannot be zero")?;
        let mut gate = NoiseGate::new(sample_rate);
        let mut tone = ToneStack::new(sample_rate);
        let mut nam = if track.effects.nam_enabled {
            track
                .nam_model
                .as_deref()
                .map(|path| NamProcessor::load(path, project.sample_rate, 2_048))
                .transpose()
                .map_err(anyhow::Error::msg)?
        } else {
            None
        };
        for clip in &track.clips {
            let mut samples = read_wav(&clip.path, project.sample_rate)?;
            if let Some(nam) = &mut nam {
                let input_gain = db_to_gain(track.effects.nam_input_db);
                let normalize = if track.effects.nam_normalize {
                    nam.loudness().map_or(1.0, |loudness| {
                        #[allow(clippy::cast_possible_truncation)]
                        let difference = (NORMALIZE_TARGET_DB - loudness) as f32;
                        db_to_gain(difference.clamp(-24.0, 24.0))
                    })
                } else {
                    1.0
                };
                let output_gain = db_to_gain(track.effects.nam_output_db) * normalize;
                let mut mono = vec![0.0_f32; 2_048];
                for block in samples.chunks_mut(2_048) {
                    let block_len = block.len();
                    for (sample, frame) in mono[..block_len].iter_mut().zip(block.iter()) {
                        *sample = (frame[0] + frame[1]) * 0.5 * input_gain;
                    }
                    gate.process(&mut mono[..block_len], track.effects.nam_gate_db);
                    nam.process(&mut mono[..block_len]).map_err(anyhow::Error::msg)?;
                    for (frame, sample) in block.iter_mut().zip(&mono[..block_len]) {
                        *frame = [*sample * output_gain; 2];
                    }
                    if track.effects.nam_tone_enabled {
                        tone.process(
                            block,
                            track.effects.nam_bass,
                            track.effects.nam_middle,
                            track.effects.nam_treble,
                        );
                    }
                }
            }
            processor.process_stereo(&mut samples);
            let available = clip.end_frame.saturating_sub(clip.start_frame);
            let wanted = usize::try_from(available).unwrap_or(usize::MAX);
            let start = usize::try_from(clip.start_frame).context("clip starts too late")?;
            for (offset, frame) in samples.iter().take(wanted).enumerate() {
                let Some(output) = mix.get_mut(start.saturating_add(offset)) else {
                    break;
                };
                let left_pan_gain = if track.pan > 0.0 {
                    1.0 - track.pan
                } else {
                    1.0
                };
                let right_pan_gain = if track.pan < 0.0 {
                    1.0 + track.pan
                } else {
                    1.0
                };
                output[0] += frame[0] * gain * left_pan_gain;
                output[1] += frame[1] * gain * right_pan_gain;
            }
        }
    }

    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: project.sample_rate,
        bits_per_sample: 24,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(destination, spec)
        .with_context(|| format!("failed to create {}", destination.display()))?;
    for frame in mix {
        writer.write_sample(float_to_i24(frame[0]))?;
        writer.write_sample(float_to_i24(frame[1]))?;
    }
    writer.finalize().context("failed to finalize mix WAV")?;
    Ok(end_frame)
}

fn read_wav(path: &Path, expected_rate: u32) -> Result<Vec<[f32; 2]>> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let spec = reader.spec();
    if spec.sample_rate != expected_rate {
        bail!(
            "{} uses {} Hz; session uses {} Hz",
            path.display(),
            spec.sample_rate,
            expected_rate
        );
    }
    if !(1..=2).contains(&spec.channels) {
        bail!("{} must be mono or stereo", path.display());
    }
    let scalar = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 2.0_f32.powi(i32::from(spec.bits_per_sample).saturating_sub(1));
            reader
                .samples::<i32>()
                .map(|sample| {
                    sample.map(|value| {
                        #[allow(clippy::cast_precision_loss)]
                        {
                            value as f32 / scale
                        }
                    })
                })
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
    };
    let channels = usize::from(spec.channels);
    Ok(scalar
        .chunks_exact(channels)
        .map(|frame| {
            if channels == 1 {
                [frame[0], frame[0]]
            } else {
                [frame[0], frame[1]]
            }
        })
        .collect())
}

fn db_to_gain(decibels: f32) -> f32 {
    10.0_f32.powf(decibels / 20.0)
}

fn float_to_i24(sample: f32) -> i32 {
    #[allow(clippy::cast_possible_truncation)]
    {
        (sample.clamp(-1.0, 1.0) * 8_388_607.0).round() as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daw_core::ChannelLayout;
    use daw_project::{ProjectClip, ProjectTrack};
    use std::path::PathBuf;

    #[test]
    fn zero_db_is_unity_gain() {
        assert!((db_to_gain(0.0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn minus_six_db_is_about_half_gain() {
        assert!((db_to_gain(-6.0) - 0.501_187).abs() < 0.000_01);
    }

    #[test]
    fn integer_conversion_clamps() {
        assert_eq!(float_to_i24(2.0), 8_388_607);
        assert_eq!(float_to_i24(-2.0), -8_388_607);
    }

    #[test]
    fn exports_mono_clip_as_stereo_mix() {
        let stem = format!("rustdaw-render-test-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stem}-input.wav"));
        let output = std::env::temp_dir().join(format!("{stem}-output.wav"));
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 24,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&input, spec).unwrap();
        for _ in 0..4 {
            writer.write_sample(2_097_152_i32).unwrap();
        }
        writer.finalize().unwrap();

        let mut track = ProjectTrack::new("Guitar", ChannelLayout::Mono);
        track.clips.push(ProjectClip {
            id: uuid::Uuid::new_v4(),
            name: "Take".to_owned(),
            path: PathBuf::from(&input),
            start_frame: 0,
            end_frame: 4,
        });
        let project = ProjectDocument {
            tracks: vec![track],
            ..ProjectDocument::default()
        };
        assert_eq!(export_stereo(&project, &output).unwrap(), 4);

        let mut rendered = hound::WavReader::open(&output).unwrap();
        assert_eq!(rendered.spec().channels, 2);
        assert_eq!(rendered.duration(), 4);
        let samples = rendered
            .samples::<i32>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(samples.len(), 8);
        assert!(samples.iter().all(|sample| *sample > 2_000_000));
        drop(rendered);
        std::fs::remove_file(input).unwrap();
        std::fs::remove_file(output).unwrap();
    }
}
