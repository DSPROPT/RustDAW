use anyhow::{Context, Result, bail};
use daw_audio_linux::{AudioRuntime, AudioRuntimeConfig};
use daw_core::ChannelLayout;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

fn main() -> Result<()> {
    let seconds = std::env::args()
        .nth(1)
        .map_or(Ok(60_u64), |value| value.parse::<u64>())
        .context("duration must be an integer number of seconds")?;
    let output = PathBuf::from("Recordings/soak-test.wav");
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let runtime = AudioRuntime::open(&AudioRuntimeConfig::default())?;
    runtime.set_click(false, 0.0);
    let _ = runtime.start_recording(output.clone(), ChannelLayout::Mono, 1, 1, 0)?;
    for elapsed in 1..=seconds {
        thread::sleep(Duration::from_secs(1));
        if elapsed % 10 == 0 || elapsed == seconds {
            let snapshot = runtime.snapshot();
            println!(
                "{elapsed}/{seconds}s · XRUN {} · dropped {} · disk_error {}",
                snapshot.xruns, snapshot.dropped_record_frames, snapshot.disk_error
            );
        }
    }
    let snapshot = runtime.snapshot();
    runtime.stop();
    drop(runtime);

    if snapshot.xruns != 0 || snapshot.dropped_record_frames != 0 || snapshot.disk_error {
        bail!("soak test detected an audio or disk failure");
    }
    let reader = hound::WavReader::open(&output)?;
    let actual = u64::from(reader.duration());
    let expected = seconds.saturating_mul(u64::from(reader.spec().sample_rate));
    let tolerance = u64::from(reader.spec().sample_rate) / 2;
    if actual.abs_diff(expected) > tolerance {
        bail!("soak recording duration differs from the expected duration");
    }
    println!(
        "PASS · {} frames · {} bytes",
        actual,
        std::fs::metadata(output)?.len()
    );
    Ok(())
}
