use anyhow::{Context, Result};
use daw_audio_linux::{AudioRuntime, AudioRuntimeConfig};
use daw_core::ChannelLayout;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

fn main() -> Result<()> {
    let output = std::env::args_os()
        .nth(1)
        .map_or_else(|| PathBuf::from("Recordings/smoke-test.wav"), PathBuf::from);
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).context("failed to create recording directory")?;
    }

    println!("Opening Scarlett through PipeWire/PulseAudio…");
    let runtime = AudioRuntime::open(&AudioRuntimeConfig::default())?;
    println!("Capture source: {}", runtime.input_name());
    if runtime
        .input_name()
        .to_ascii_lowercase()
        .starts_with("monitor of ")
    {
        anyhow::bail!("playback monitor was selected instead of the hardware capture source");
    }
    runtime.set_click(false, 0.0);
    println!(
        "Recording Scarlett Input 2 to {} for 2 seconds…",
        output.display()
    );
    let _ = runtime.start_recording(output.clone(), ChannelLayout::Mono, 1, 1, 0)?;
    thread::sleep(Duration::from_secs(2));
    runtime.stop();
    thread::sleep(Duration::from_millis(250));
    drop(runtime);

    let reader = hound::WavReader::open(&output).context("recorded WAV could not be reopened")?;
    let spec = reader.spec();
    let frames = u64::from(reader.duration());
    #[allow(clippy::cast_precision_loss)]
    let seconds = frames as f64 / f64::from(spec.sample_rate);
    println!(
        "Validated: {} channel, {} Hz, {}-bit, {frames} frames ({:.2} s)",
        spec.channels, spec.sample_rate, spec.bits_per_sample, seconds
    );
    if spec.channels != 1 || spec.sample_rate != 48_000 || spec.bits_per_sample != 24 {
        anyhow::bail!("recorded WAV format does not match the MVP contract");
    }
    if frames < 48_000 {
        anyhow::bail!("recorded WAV is unexpectedly short");
    }
    drop(reader);

    let runtime = AudioRuntime::open(&AudioRuntimeConfig::default())?;
    runtime.set_click(false, 0.0);
    runtime.clear_playback()?;
    runtime.add_playback_file(&output, 0, 0.25)?;
    runtime.seek_to_start();
    runtime.play();
    thread::sleep(Duration::from_millis(300));
    let playback_position = runtime.snapshot().position_frames;
    runtime.stop();
    if playback_position < 4_800 {
        anyhow::bail!("playback transport did not advance");
    }
    println!("Playback scheduling validated at frame {playback_position}");

    let stereo_output = output.with_file_name("smoke-test-stereo.wav");
    let runtime = AudioRuntime::open(&AudioRuntimeConfig::default())?;
    runtime.set_click(false, 0.0);
    let _ = runtime.start_recording(stereo_output.clone(), ChannelLayout::Stereo, 0, 1, 0)?;
    thread::sleep(Duration::from_secs(1));
    runtime.stop();
    drop(runtime);
    let stereo = hound::WavReader::open(&stereo_output)?;
    if stereo.spec().channels != 2 || stereo.duration() < 24_000 {
        anyhow::bail!("stereo recording validation failed");
    }
    println!(
        "Stereo recording validated: {} frames at {} Hz",
        stereo.duration(),
        stereo.spec().sample_rate
    );
    Ok(())
}
