//! Masters one WAV against another from the command line.
//!
//! ```bash
//! cargo run --release -p daw-master --example master-to-reference -- \
//!     mix.wav reference.wav mastered.wav
//! ```
//!
//! The same code the export path runs, without the desktop application — which
//! is what makes it useful for checking the result against another mastering
//! tool on the same pair of files.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use anyhow::{Context, Result, bail};
use std::path::PathBuf;

fn main() -> Result<()> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let [target, reference, destination] = arguments.as_slice() else {
        bail!("usage: master-to-reference <mix.wav> <reference.wav> <out.wav>");
    };

    let target_path = PathBuf::from(target);
    let reader =
        hound::WavReader::open(&target_path).with_context(|| format!("failed to open {target}"))?;
    let rate = reader.spec().sample_rate;
    drop(reader);

    let mut mix = daw_master::load_reference(&target_path, rate)?;
    let record = daw_master::load_reference(&PathBuf::from(reference), rate)?;

    println!(
        "mastering {:.1} s against {:.1} s of reference at {rate} Hz",
        mix.len() as f32 / rate as f32,
        record.len() as f32 / rate as f32
    );
    let started = std::time::Instant::now();
    daw_master::master(
        &mut mix,
        &record,
        rate as f32,
        &daw_master::Config::default(),
    )?;
    println!("done in {:.2} s", started.elapsed().as_secs_f32());

    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: rate,
        bits_per_sample: 24,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(destination, spec)?;
    for frame in &mix {
        for sample in frame {
            writer.write_sample((sample.clamp(-1.0, 1.0) * 8_388_607.0).round() as i32)?;
        }
    }
    writer.finalize()?;
    println!("wrote {destination}");
    Ok(())
}
