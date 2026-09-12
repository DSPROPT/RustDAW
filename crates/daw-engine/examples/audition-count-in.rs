//! Renders the wood-block click to a WAV so it can be judged by ear, which is
//! the only way a click can be judged at all.
//!
//! ```text
//! cargo run -p daw-engine --example audition-count-in -- count-in.wav [bpm] [beats] [bars]
//! ```
//!
//! Defaults to a one-bar count-in of four beats at 120 BPM. Give a number of
//! bars to hear the whole-song click track instead, rendered the way the
//! **CLICK TRACK** button renders it. Run it after touching the block's
//! partials or decays: a test can say the hit is short and peaks where it
//! should, not whether it sounds like wood.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use daw_core::SampleRate;
use daw_engine::{render_click_track, render_count_in};
use daw_midi::TempoMap;

fn main() {
    let mut arguments = std::env::args().skip(1);
    let destination = arguments
        .next()
        .unwrap_or_else(|| "count-in.wav".to_owned());
    let tempo: u16 = arguments
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(120);
    let beats: u16 = arguments
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4);
    let bars: Option<u64> = arguments.next().and_then(|value| value.parse().ok());

    let rendered = match bars {
        None => render_count_in(SampleRate::DEFAULT, tempo, beats),
        Some(bars) => {
            let frames_per_bar =
                f64::from(SampleRate::DEFAULT.get()) * 60.0 / f64::from(tempo) * f64::from(beats);
            render_click_track(
                SampleRate::DEFAULT,
                &TempoMap::constant(f64::from(tempo)),
                beats,
                (frames_per_bar * bars as f64).round() as u64,
            )
        }
    };
    let samples = match rendered {
        Ok(samples) => samples,
        Err(error) => {
            eprintln!("could not render the click: {error}");
            std::process::exit(1);
        }
    };
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SampleRate::DEFAULT.get(),
        bits_per_sample: 24,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = match hound::WavWriter::create(&destination, spec) {
        Ok(writer) => writer,
        Err(error) => {
            eprintln!("could not create {destination}: {error}");
            std::process::exit(1);
        }
    };
    let full_scale = f32::from(i16::MAX) * 256.0;
    for sample in &samples {
        if let Err(error) = writer.write_sample((sample * full_scale).round() as i32) {
            eprintln!("could not write {destination}: {error}");
            std::process::exit(1);
        }
    }
    if let Err(error) = writer.finalize() {
        eprintln!("could not finish {destination}: {error}");
        std::process::exit(1);
    }
    println!(
        "Wrote {} at {tempo} BPM ({:.3} s) to {destination}",
        match bars {
            None => format!("{beats} clicks"),
            Some(bars) => format!("{bars} bar(s) of {beats}"),
        },
        samples.len() as f64 / f64::from(SampleRate::DEFAULT.get())
    );
}
