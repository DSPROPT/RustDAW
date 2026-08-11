use anyhow::Result;
use daw_audio_linux::{AudioRuntime, AudioRuntimeConfig};
use daw_core::ChannelLayout;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

fn main() -> Result<()> {
    let output = std::env::args_os().nth(1).map_or_else(
        || PathBuf::from("Recordings/crash-recovery-smoke.wav"),
        PathBuf::from,
    );
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let runtime = AudioRuntime::open(&AudioRuntimeConfig::default())?;
    runtime.set_click(false, 0.0);
    let _ = runtime.start_recording(output, ChannelLayout::Mono, 1, 1, 0)?;
    thread::sleep(Duration::from_millis(2_500));

    // Deliberately bypasses Drop and WAV finalization to simulate an abrupt
    // application exit. The periodic writer flush is expected to leave a
    // valid, slightly shorter recoverable file.
    std::process::exit(0);
}
