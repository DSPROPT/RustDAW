use anyhow::{Context, Result};
use daw_audio_linux::{AudioRuntime, AudioRuntimeConfig};
use daw_engine::ChannelStripParams;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn main() -> Result<()> {
    let paths = std::env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    anyhow::ensure!(!paths.is_empty(), "provide one or more WAV files");
    let runtime = AudioRuntime::open(&AudioRuntimeConfig::default())?;
    let preload_started = Instant::now();
    runtime.clear_playback()?;
    for (track_id, path) in paths.iter().enumerate() {
        runtime
            .add_track_playback_file(
                path,
                0,
                1.0,
                ChannelStripParams::default(),
                0.0,
                track_id,
                true,
            )
            .with_context(|| format!("failed to preload {}", path.display()))?;
    }
    let preload_elapsed = preload_started.elapsed();
    std::thread::sleep(Duration::from_millis(100));
    let cached_started = Instant::now();
    runtime.clear_playback()?;
    for (track_id, path) in paths.iter().enumerate() {
        runtime.add_track_playback_file(
            path,
            0,
            1.0,
            ChannelStripParams::default(),
            0.0,
            track_id,
            true,
        )?;
    }
    let cached_elapsed = cached_started.elapsed();
    std::thread::sleep(Duration::from_millis(100));
    let play_started = Instant::now();
    runtime.play();
    let play_elapsed = play_started.elapsed();
    let effects_started = Instant::now();
    runtime.set_track_effects(
        0,
        ChannelStripParams {
            eq_enabled: true,
            low_db: 3.0,
            compressor_enabled: true,
            gate_enabled: true,
            ..ChannelStripParams::default()
        },
    )?;
    let effects_elapsed = effects_started.elapsed();
    std::thread::sleep(Duration::from_millis(150));
    anyhow::ensure!(
        runtime.snapshot().position_frames > 0,
        "transport did not start"
    );
    runtime.stop();
    println!(
        "Preloaded {} files in {:.3} s; cached rebuild {:.3} ms; Play command {:.3} ms; effects update {:.3} ms",
        paths.len(),
        preload_elapsed.as_secs_f64(),
        cached_elapsed.as_secs_f64() * 1_000.0,
        play_elapsed.as_secs_f64() * 1_000.0,
        effects_elapsed.as_secs_f64() * 1_000.0
    );
    Ok(())
}
