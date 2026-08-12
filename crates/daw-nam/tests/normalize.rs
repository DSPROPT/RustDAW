//! Normalising is only worth having if it actually levels captures against
//! each other, which is a property of the models rather than of any one of
//! them. This measures it across every reference model that knows its loudness.

use std::path::{Path, PathBuf};

use daw_nam::NamProcessor;

const TARGET_DB: f64 = -18.0;

fn models() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../third_party/NeuralAmpModelerCore/example_models");
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|x| x == "nam"))
        .collect();
    paths.sort();
    paths
}

#[test]
fn normalising_brings_captures_within_reach_of_each_other() {
    let mut spread_before: Vec<f32> = Vec::new();
    let mut spread_after: Vec<f32> = Vec::new();

    for path in models() {
        let Ok(mut processor) = NamProcessor::load(&path, 48_000, 2_048) else {
            continue;
        };
        let Some(loudness) = processor.loudness() else {
            continue;
        };
        #[allow(clippy::cast_possible_truncation)]
        let normalize = 10.0_f32.powf(((TARGET_DB - loudness) as f32).clamp(-24.0, 24.0) / 20.0);

        let mut audio: Vec<f32> = (0..2_048)
            .map(|index| (index as f32 * 0.02).sin() * 0.25)
            .collect();
        if processor.process(&mut audio).is_err() {
            continue;
        }
        let rms = (audio.iter().map(|v| v * v).sum::<f32>() / audio.len() as f32).sqrt();
        if rms <= 0.0 {
            continue;
        }
        spread_before.push(rms);
        spread_after.push(rms * normalize);
    }

    if spread_before.len() < 2 {
        return; // Not enough models carry a loudness to say anything.
    }
    let ratio = |values: &[f32]| -> f32 {
        let max = values.iter().copied().fold(0.0_f32, f32::max);
        let min = values.iter().copied().fold(f32::MAX, f32::min);
        max / min.max(1e-9)
    };
    let before = ratio(&spread_before);
    let after = ratio(&spread_after);
    println!(
        "{} of {} reference models carry a loudness; spread {before:.1}x before, {after:.1}x after",
        spread_before.len(),
        models().len()
    );
    assert!(
        after < before,
        "normalising did not tighten the spread: {before:.1}x then {after:.1}x"
    );
}
