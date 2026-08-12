//! Opens the audio runtime against whatever device the host offers and plays
//! the built-in 440 Hz test tone, to confirm playback works on this machine.

use daw_audio_linux::{AudioRuntime, AudioRuntimeConfig};

fn main() {
    let config = AudioRuntimeConfig::default();
    match AudioRuntime::open(&config) {
        Ok(runtime) => {
            println!("audio engine online");
            println!("  input : {}", runtime.input_name());
            println!("  output: {}", runtime.output_name());
            println!("  rate  : {} Hz", runtime.sample_rate().get());
            println!("  out ch: {}", runtime.output_channels());
            println!("playing a 1 second test tone…");
            runtime.trigger_output_test();
            std::thread::sleep(std::time::Duration::from_millis(1500));
            println!("done");
        }
        Err(error) => {
            eprintln!("audio engine failed to open: {error:#}");
            std::process::exit(1);
        }
    }
}
