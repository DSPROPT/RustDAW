//! Deterministic offline stereo rendering for `RustDAW` sessions.

use anyhow::{Context, Result, bail};
use daw_engine::{ChannelStrip, ChannelStripParams, NoiseGate, ToneStack};
use daw_nam::NamProcessor;
use daw_project::ProjectDocument;

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
    let frame_count = render_length(project)?;
    let mut mix = vec![[0.0_f32; 2]; frame_count];

    for track in audible_tracks(project) {
        let rendered = render_track(track, project.sample_rate, frame_count)?;
        for (output, frame) in mix.iter_mut().zip(rendered) {
            output[0] += frame[0];
            output[1] += frame[1];
        }
    }

    // The master stage. Everything above is the mix; this is the one thing
    // that looks at it whole, so it runs last and only on the way out.
    if let Some(reference) = &project.master_reference {
        master_to_reference(&mut mix, reference, project.sample_rate)?;
    }

    write_stereo_wav(destination, &mix, project.sample_rate)?;
    u64::try_from(frame_count).context("session is too long to render")
}

/// One track written out on its own by [`export_stems`].
#[derive(Clone, Debug)]
pub struct StemExport {
    /// The track's name, as it appears in the mixer.
    pub track: String,
    /// The file that was written.
    pub path: std::path::PathBuf,
    /// The peak level rendered, before the 24-bit clamp. Over 1.0 means the
    /// track is hotter than full scale on its own, so the file it was written
    /// to is clipped — even where the finished mix is not, because other tracks
    /// pull the sum back under. Fix it by lowering that track's fader.
    pub peak: f32,
}

impl StemExport {
    /// Whether the written file lost signal to the 24-bit ceiling.
    #[must_use]
    pub fn clipped(&self) -> bool {
        self.peak > 1.0
    }
}

/// Renders every audible track to its own stereo 24-bit WAV in `directory`.
///
/// The stems are what the mix is made of: each one carries that track's insert
/// chain, gain and pan, and they are all the full length of the session so they
/// line up at zero wherever they are opened. Summing them reproduces the mix,
/// which is why muted tracks are left out and a solo is honoured — and why the
/// mastering reference is *not* applied here. Mastering measures a finished mix
/// as a whole; running it on each stem separately would master six songs and
/// they would no longer add up to one.
///
/// The sum is exact only while every stem fits in 24 bits. A track hotter than
/// full scale on its own clips on the way out even where the mix does not — the
/// other tracks pull the sum back under, but the stem has nothing to pull it.
/// [`StemExport::clipped`] reports that per stem rather than leaving it as a
/// difference nobody notices until the stems are used somewhere else.
///
/// Instrument tracks are skipped: their MIDI is played by the live synthesiser
/// and the offline renderer has no audio for them. Only tracks with audio clips
/// come out, and the returned list says which those were.
///
/// # Errors
///
/// Returns an error for missing/unsupported media, sample-rate mismatches, or
/// output filesystem failures.
pub fn export_stems(project: &ProjectDocument, directory: &Path) -> Result<Vec<StemExport>> {
    let frame_count = render_length(project)?;
    std::fs::create_dir_all(directory)
        .with_context(|| format!("failed to create {}", directory.display()))?;

    let mut written = Vec::new();
    for (index, track) in audible_tracks(project)
        .filter(|track| !track.clips.is_empty())
        .enumerate()
    {
        let rendered = render_track(track, project.sample_rate, frame_count)?;
        // Numbered so the files sort in mixer order, and so two tracks sharing
        // a name do not write over each other.
        let path = directory.join(format!(
            "{:02} {}.wav",
            index + 1,
            sanitize_file_name(&track.name)
        ));
        let peak = rendered
            .iter()
            .flat_map(|frame| [frame[0].abs(), frame[1].abs()])
            .fold(0.0_f32, f32::max);
        write_stereo_wav(&path, &rendered, project.sample_rate)?;
        written.push(StemExport {
            track: track.name.clone(),
            path,
            peak,
        });
    }
    Ok(written)
}

/// How long the rendered session is, in frames: the end of its last clip.
fn render_length(project: &ProjectDocument) -> Result<usize> {
    let end_frame = project
        .tracks
        .iter()
        .flat_map(|track| &track.clips)
        .map(|clip| clip.end_frame)
        .max()
        .unwrap_or(0);
    usize::try_from(end_frame).context("session is too long to render")
}

/// The tracks a listener would hear: unmuted, and soloed if anything is.
fn audible_tracks(
    project: &ProjectDocument,
) -> impl Iterator<Item = &daw_project::ProjectTrack> + '_ {
    let any_solo = project.tracks.iter().any(|track| track.solo);
    project
        .tracks
        .iter()
        .filter(move |track| !track.muted && (!any_solo || track.solo))
}

/// Replaces anything awkward in a track name so it can be a file name.
fn sanitize_file_name(name: &str) -> String {
    let sanitized = name
        .chars()
        .map(|character| {
            if character.is_alphanumeric() || matches!(character, '-' | '_' | ' ') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    let trimmed = sanitized.trim();
    if trimmed.is_empty() {
        "Track".to_owned()
    } else {
        trimmed.to_owned()
    }
}

/// Writes finished frames as a 24-bit stereo WAV, creating the folder if needed.
fn write_stereo_wav(destination: &Path, frames: &[[f32; 2]], sample_rate: u32) -> Result<()> {
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 24,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(destination, spec)
        .with_context(|| format!("failed to create {}", destination.display()))?;
    for frame in frames {
        writer.write_sample(float_to_i24(frame[0]))?;
        writer.write_sample(float_to_i24(frame[1]))?;
    }
    writer
        .finalize()
        .with_context(|| format!("failed to finalize {}", destination.display()))
}

/// Renders one track's clips into a buffer of `frame_count` frames, through its
/// insert chain, gain and pan — everything the mix does except the master stage.
// One pass over the track's clips through the full insert chain. Splitting it
// would mean threading the whole per-track state through helpers for no gain in
// clarity.
#[allow(clippy::too_many_lines)]
fn render_track(
    track: &daw_project::ProjectTrack,
    sample_rate_hz: u32,
    frame_count: usize,
) -> Result<Vec<[f32; 2]>> {
    let mut rendered = vec![[0.0_f32; 2]; frame_count];
    {
        let gain = db_to_gain(track.gain_db);
        let mut processor = ChannelStrip::new(
            daw_core::SampleRate::new(sample_rate_hz)
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
        let sample_rate = daw_core::SampleRate::new(sample_rate_hz)
            .context("project sample rate cannot be zero")?;
        let mut gate = NoiseGate::new(sample_rate);
        let mut tone = ToneStack::new(sample_rate);
        let mut nam = if track.effects.nam_enabled {
            track
                .nam_model
                .as_deref()
                .map(|path| NamProcessor::load(path, sample_rate_hz, 2_048))
                .transpose()
                .map_err(anyhow::Error::msg)?
        } else {
            None
        };
        for clip in &track.clips {
            let mut samples = read_wav(&clip.path, sample_rate_hz)?;
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
                    nam.process(&mut mono[..block_len])
                        .map_err(anyhow::Error::msg)?;
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
            let wanted = usize::try_from(clip.length()).unwrap_or(usize::MAX);
            let start = usize::try_from(clip.start_frame).context("clip starts too late")?;
            // The clip reads a window of its source rather than the whole file:
            // trimming and splitting move this offset, and the file is never
            // rewritten.
            let source_start = usize::try_from(clip.source_start_frame).unwrap_or(usize::MAX);
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
            for (offset, frame) in samples.iter().skip(source_start).take(wanted).enumerate() {
                let Some(output) = rendered.get_mut(start.saturating_add(offset)) else {
                    break;
                };
                output[0] += frame[0] * gain * left_pan_gain;
                output[1] += frame[1] * gain * right_pan_gain;
            }
        }
    }
    Ok(rendered)
}

/// Matches the finished mix to a reference record.
///
/// Failures here are reported against the reference rather than the export: a
/// missing or mismatched reference is something the user chose and can fix,
/// and silently exporting an unmastered mix under a name they expected to be
/// mastered would be worse than refusing.
fn master_to_reference(mix: &mut Vec<[f32; 2]>, reference: &Path, sample_rate: u32) -> Result<()> {
    let loaded = daw_master::load_reference(reference, sample_rate).with_context(|| {
        format!(
            "failed to read the mastering reference {}",
            reference.display()
        )
    })?;
    #[allow(clippy::cast_precision_loss)]
    let rate = sample_rate as f32;
    daw_master::master(mix, &loaded, rate, &daw_master::Config::default())
        .context("failed to master the mix against the reference")
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

    /// Writes a 48 kHz stereo WAV of `seconds` of decaying noise at `level`.
    fn write_noise(path: &Path, seconds: usize, level: f32) {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 24,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        let mut state = 0x1234_5678_9abc_def0_u64;
        for _ in 0..48_000 * seconds {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[allow(clippy::cast_precision_loss)]
            let sample = ((state >> 40) as f32 / 8_388_608.0 - 1.0) * level;
            writer.write_sample(float_to_i24(sample)).unwrap();
            writer.write_sample(float_to_i24(sample * 0.9)).unwrap();
        }
        writer.finalize().unwrap();
    }

    /// Reads a stereo WAV back as interleaved floats.
    fn read_frames(path: &Path) -> Vec<[f32; 2]> {
        let mut reader = hound::WavReader::open(path).unwrap();
        let samples: Vec<f32> = reader
            .samples::<i32>()
            .map(|sample| {
                #[allow(clippy::cast_precision_loss)]
                {
                    sample.unwrap() as f32 / 8_388_608.0
                }
            })
            .collect();
        samples
            .chunks_exact(2)
            .map(|pair| [pair[0], pair[1]])
            .collect()
    }

    fn noise_track(name: &str, source: &Path, seconds: u64) -> ProjectTrack {
        let mut track = ProjectTrack::new(name, ChannelLayout::Stereo);
        track.clips.push(ProjectClip {
            source_start_frame: 0,
            source_path: None,
            id: uuid::Uuid::new_v4(),
            name: "Take".to_owned(),
            path: source.to_path_buf(),
            start_frame: 0,
            end_frame: 48_000 * seconds,
        });
        track
    }

    #[test]
    fn stems_are_one_file_per_track_that_add_back_up_to_the_mix() {
        let stem = format!("rustdaw-stems-test-{}", std::process::id());
        let temp = std::env::temp_dir();
        let drums = temp.join(format!("{stem}-drums.wav"));
        let bass = temp.join(format!("{stem}-bass.wav"));
        let mixed = temp.join(format!("{stem}-mix.wav"));
        let directory = temp.join(format!("{stem}-stems"));
        write_noise(&drums, 1, 0.2);
        write_noise(&bass, 1, 0.1);

        // Gain and pan differ so the sum is only right if each stem carries its
        // own; the muted track must not appear in either the mix or the stems.
        let mut drum_track = noise_track("Drums", &drums, 1);
        drum_track.gain_db = -3.0;
        drum_track.pan = -0.4;
        let bass_track = noise_track("Bass / DI", &bass, 1);
        let mut muted = noise_track("Scratch", &drums, 1);
        muted.muted = true;
        let project = ProjectDocument {
            tracks: vec![drum_track, bass_track, muted],
            ..ProjectDocument::default()
        };

        export_stereo(&project, &mixed).unwrap();
        let written = export_stems(&project, &directory).unwrap();

        assert_eq!(written.len(), 2, "the muted track must not be written");
        assert_eq!(written[0].path.file_name().unwrap(), "01 Drums.wav");
        // A slash would have made a directory of it.
        assert_eq!(written[1].path.file_name().unwrap(), "02 Bass _ DI.wav");

        let mix = read_frames(&mixed);
        let stems: Vec<Vec<[f32; 2]>> = written.iter().map(|s| read_frames(&s.path)).collect();
        for stem in &stems {
            assert_eq!(stem.len(), mix.len(), "stems must be the session's length");
        }
        // Nothing here is hot enough to clip, which is the condition under
        // which the sum is exact.
        assert!(written.iter().all(|stem| !stem.clipped()));
        let mut worst = 0.0_f32;
        for (index, frame) in mix.iter().enumerate() {
            for channel in 0..2 {
                let summed: f32 = stems.iter().map(|stem| stem[index][channel]).sum();
                worst = worst.max((summed - frame[channel]).abs());
            }
        }
        assert!(worst < 1e-5, "stems differ from the mix by {worst}");

        for path in [drums, bass, mixed] {
            std::fs::remove_file(path).unwrap();
        }
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn a_track_hotter_than_full_scale_is_reported_as_a_clipped_stem() {
        let stem = format!("rustdaw-clip-test-{}", std::process::id());
        let temp = std::env::temp_dir();
        let loud = temp.join(format!("{stem}-loud.wav"));
        let quiet = temp.join(format!("{stem}-quiet.wav"));
        let directory = temp.join(format!("{stem}-stems"));
        write_noise(&loud, 1, 0.9);
        write_noise(&quiet, 1, 0.1);

        // +6 dB on already-loud noise puts this track over the ceiling on its
        // own, which is exactly the case the mix can hide.
        let mut hot = noise_track("Hot", &loud, 1);
        hot.gain_db = 6.0;
        let project = ProjectDocument {
            tracks: vec![hot, noise_track("Quiet", &quiet, 1)],
            ..ProjectDocument::default()
        };

        let written = export_stems(&project, &directory).unwrap();
        assert!(written[0].clipped(), "peak was {}", written[0].peak);
        assert!(!written[1].clipped(), "peak was {}", written[1].peak);

        for path in [loud, quiet] {
            std::fs::remove_file(path).unwrap();
        }
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn a_mastering_reference_changes_the_export_and_holds_the_ceiling() {
        let stem = format!("rustdaw-master-test-{}", std::process::id());
        let temp = std::env::temp_dir();
        let source = temp.join(format!("{stem}-source.wav"));
        let reference = temp.join(format!("{stem}-reference.wav"));
        let plain = temp.join(format!("{stem}-plain.wav"));
        let mastered = temp.join(format!("{stem}-mastered.wav"));

        // A quiet mix and a loud reference: mastering must close the gap.
        write_noise(&source, 2, 0.05);
        write_noise(&reference, 2, 0.7);

        let mut track = ProjectTrack::new("Mix", ChannelLayout::Stereo);
        track.clips.push(ProjectClip {
            source_start_frame: 0,
            source_path: None,
            id: uuid::Uuid::new_v4(),
            name: "Take".to_owned(),
            path: source.clone(),
            start_frame: 0,
            end_frame: 48_000 * 2,
        });
        let project = ProjectDocument {
            tracks: vec![track],
            ..ProjectDocument::default()
        };
        export_stereo(&project, &plain).unwrap();

        let with_reference = ProjectDocument {
            master_reference: Some(reference.clone()),
            ..project
        };
        export_stereo(&with_reference, &mastered).unwrap();

        let peak_of = |path: &Path| {
            let mut reader = hound::WavReader::open(path).unwrap();
            reader
                .samples::<i32>()
                .map(|sample| sample.unwrap().abs())
                .max()
                .unwrap_or(0)
        };
        let quiet = peak_of(&plain);
        let loud = peak_of(&mastered);

        assert!(
            loud > quiet * 2,
            "mastering should bring the quiet mix up: {quiet} then {loud}"
        );
        assert!(loud <= 8_388_607, "and must not exceed full scale: {loud}");

        for path in [source, reference, plain, mastered] {
            std::fs::remove_file(path).ok();
        }
    }

    #[test]
    fn a_missing_reference_fails_the_export_rather_than_exporting_unmastered() {
        let stem = format!("rustdaw-master-missing-{}", std::process::id());
        let temp = std::env::temp_dir();
        let source = temp.join(format!("{stem}-source.wav"));
        let output = temp.join(format!("{stem}-out.wav"));
        write_noise(&source, 1, 0.3);

        let mut track = ProjectTrack::new("Mix", ChannelLayout::Stereo);
        track.clips.push(ProjectClip {
            source_start_frame: 0,
            source_path: None,
            id: uuid::Uuid::new_v4(),
            name: "Take".to_owned(),
            path: source.clone(),
            start_frame: 0,
            end_frame: 48_000,
        });
        let project = ProjectDocument {
            tracks: vec![track],
            master_reference: Some(temp.join("no-such-reference.wav")),
            ..ProjectDocument::default()
        };

        let error = export_stereo(&project, &output).unwrap_err();
        assert!(
            format!("{error:#}").contains("reference"),
            "the error should name the reference: {error:#}"
        );

        std::fs::remove_file(source).ok();
        std::fs::remove_file(output).ok();
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
            source_start_frame: 0,
            source_path: None,
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
