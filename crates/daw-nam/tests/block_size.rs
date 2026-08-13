//! The block-size contract, which is easy to get wrong from the caller's side.
//!
//! A model is prepared for a maximum block at load time. Handed more frames
//! than that, it refuses the block rather than writing past its buffers — and a
//! caller that ignores the error passes the signal through untouched, so a
//! guitar amp that is quietly prepared for too small a block is heard as no amp
//! at all rather than as an error. Callers must prepare for the largest block
//! their audio path can actually produce.

use std::path::{Path, PathBuf};

use daw_nam::NamProcessor;

fn reference_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../third_party/NeuralAmpModelerCore/example_models/lstm.nam")
}

#[test]
fn a_block_larger_than_the_prepared_size_is_refused() {
    let mut processor = NamProcessor::load(&reference_model(), 48_000, 256)
        .expect("the reference model should load");
    let mut audio = vec![0.01_f32; 2_048];
    let refused = processor.process(&mut audio);
    assert!(refused.is_err(), "an oversized block was accepted");
    // And the caller's audio is left exactly as it was, not half-written.
    assert!(
        audio
            .iter()
            .all(|sample| (*sample - 0.01).abs() < f32::EPSILON)
    );
}

#[test]
fn a_block_up_to_the_prepared_size_is_processed() {
    let mut processor = NamProcessor::load(&reference_model(), 48_000, 2_048)
        .expect("the reference model should load");
    // Every size up to the prepared maximum, since the audio backend is free to
    // deliver short blocks as well as full ones.
    for frames in [1_usize, 64, 256, 1_024, 2_048] {
        let mut audio = vec![0.01_f32; frames];
        processor
            .process(&mut audio)
            .unwrap_or_else(|error| panic!("{frames} frames were refused: {error}"));
        assert!(
            audio.iter().all(|sample| sample.is_finite()),
            "{frames} frames came back with a non-finite sample"
        );
    }
}

#[test]
fn a_model_recorded_at_another_sample_rate_is_refused_by_name() {
    // Silently resampling would change the amp's character, so a mismatch is an
    // error the user can act on.
    // `NamProcessor` is deliberately not `Debug`, so unwrap the error by hand.
    let Err(error) = NamProcessor::load(&reference_model(), 44_100, 2_048) else {
        panic!("a 48 kHz model should not load into a 44.1 kHz session");
    };
    assert!(
        error.contains("44100") || error.contains("44,100") || error.contains("44100.0"),
        "the error should name the session rate: {error}"
    );
}

#[test]
fn a_model_reports_the_loudness_it_was_measured_at() {
    // Normalising captures against each other depends on this. A model that
    // was trained without one has to be usable anyway.
    let processor = NamProcessor::load(&reference_model(), 48_000, 2_048)
        .expect("the reference model should load");
    match processor.loudness() {
        Some(loudness) => assert!(
            (-60.0..=20.0).contains(&loudness),
            "{loudness} dB is not a plausible loudness"
        ),
        None => { /* Trained without one; the caller falls back to unity. */ }
    }
}
