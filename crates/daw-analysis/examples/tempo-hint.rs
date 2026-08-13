//! Shows what the tempo hint changes: the same audio read with each preset.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use daw_analysis::{beats, beats::TempoHint, onset};

/// Alternating kick and snare on every beat — a drum and bass pattern, where
/// half the tempo is just as consistent with the audio as the truth.
fn pattern(bpm: f64, rate: u32, seconds: f64) -> Vec<f32> {
    let n = (f64::from(rate) * seconds) as usize;
    let mut out = vec![0.0_f32; n];
    let beat = 60.0 / bpm;
    let mut at = 0.0;
    let mut index = 0;
    while at < seconds {
        let start = (at * f64::from(rate)) as usize;
        let (tone, noise) = if index % 2 == 0 {
            (55.0, 0.05)
        } else {
            (210.0, 0.7)
        };
        let mut seed = 0x9E37_79B9_u32.wrapping_add(start as u32);
        for i in 0..rate as usize / 12 {
            if start + i >= n {
                break;
            }
            let t = i as f32 / rate as f32;
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise_sample = (seed >> 9) as f32 / 4_194_304.0 - 1.0;
            let body = (std::f32::consts::TAU * tone * t).sin();
            out[start + i] += (-t * 27.0).exp() * (body * (1.0 - noise) + noise_sample * noise);
        }
        at += beat;
        index += 1;
    }
    out
}

fn main() {
    let rate = 48_000_u32;
    for truth in [174.0_f64, 190.0, 124.0, 90.0] {
        let audio = pattern(truth, rate, 24.0);
        let envelope = onset::onset_envelope(&audio, rate);
        print!("{truth:>6.0} BPM source →");
        for (label, centre) in TempoHint::PRESETS {
            let found = beats::estimate_tempo_with(&envelope, TempoHint::around(centre));
            let short = label.split(' ').next().unwrap_or(label);
            let mark = if (found / truth - 1.0).abs() < 0.04 {
                "*"
            } else {
                " "
            };
            print!("  {short}: {found:>5.1}{mark}");
        }
        println!();
    }
    println!("\n* = matches the source tempo");
}
