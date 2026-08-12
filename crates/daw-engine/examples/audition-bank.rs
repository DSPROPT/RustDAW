//! Renders a short passage on several instruments so the bank can be judged by
//! ear, which is the only way a patch can be judged at all.
//!
//! ```text
//! cargo run -p daw-engine --release --example audition-bank -- bank.wav
//! ```
//!
//! Each instrument plays the same phrase in turn, over a drum groove, through
//! the same shared reverb the mixer uses. Run it before and after touching a
//! patch: the numbers a test can assert on say whether a piano decays, not
//! whether it sounds like a piano.
//!
//! When a SoundFont is installed, every instrument is played twice — once
//! synthesised, then the same bar from the font — so the two can be compared
//! back to back rather than from memory. Pass `--synth` to hear only the
//! synthesised bank.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    // A straight run of setup and rendering, clearer read top to bottom than
    // split across helpers that each have one caller.
    clippy::too_many_lines,
    // "SoundFont" is the name of the file format, not of a type.
    clippy::doc_markdown
)]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use daw_core::SampleRate;
use daw_engine::{GmBank, Reverb, SampledSynth, SoundFontBank, Synth, program_name};
use daw_midi::ScheduledNote;

/// Programs to audition, one bar each.
const PROGRAMS: [u8; 8] = [
    0,  // Acoustic Grand Piano
    24, // Acoustic Guitar (nylon)
    33, // Electric Bass (finger)
    40, // Violin
    48, // String Ensemble 1
    56, // Trumpet
    71, // Clarinet
    73, // Flute
];

/// A bar of the phrase, as (pitch, sixteenths from the bar's start, length).
const PHRASE: [(u8, u64, u64); 8] = [
    (60, 0, 2),
    (64, 2, 2),
    (67, 4, 2),
    (72, 6, 4),
    (71, 10, 2),
    (67, 12, 2),
    (64, 14, 2),
    (60, 16, 8),
];

/// A bar of the groove, as (drum note, sixteenths from the bar's start).
const GROOVE: [(u8, u64); 16] = [
    (36, 0),
    (42, 0),
    (42, 2),
    (38, 4),
    (42, 4),
    (42, 6),
    (36, 8),
    (42, 8),
    (36, 10),
    (42, 10),
    (38, 12),
    (42, 12),
    (42, 14),
    (46, 15),
    (49, 0),
    (51, 6),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let synth_only = arguments.iter().any(|argument| argument == "--synth");
    let destination: PathBuf = arguments
        .iter()
        .find(|argument| !argument.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "bank-audition.wav".to_string())
        .into();

    let sample_rate = SampleRate::DEFAULT;
    let rate = sample_rate.get() as f32;
    let started = Instant::now();
    let bank = Arc::new(GmBank::new(sample_rate));
    println!("built the synthesised bank in {:?}", started.elapsed());

    let font = if synth_only {
        None
    } else {
        let started = Instant::now();
        let found = SoundFontBank::discover();
        if let Some(bank) = &found {
            println!("loaded {} in {:?}", bank.name(), started.elapsed());
        } else {
            println!("no SoundFont found; auditioning the synthesised bank alone");
        }
        found
    };
    // With a font, each instrument gets two bars: synthesised, then sampled.
    let takes = if font.is_some() { 2 } else { 1 };

    // 96 BPM, so a sixteenth is a quarter of a beat.
    let sixteenth = (rate * 60.0 / 96.0 / 4.0) as u64;
    let bar = sixteenth * 24;
    let bars = PROGRAMS.len() as u64 * takes;
    let frames = (bar * bars + rate as u64 * 3) as usize;

    let mut left = vec![0.0_f32; frames];
    let mut right = vec![0.0_f32; frames];
    let mut send = vec![[0.0_f32; 2]; frames];
    let mut reverb = Reverb::new(sample_rate);

    for (index, program) in PROGRAMS.iter().enumerate() {
        for take in 0..takes {
            let start = bar * (index as u64 * takes + take);
            // A bass belongs an octave down, or it is a lead line.
            let transpose = if *program == 33 { -24 } else { 0 };
            let notes: Vec<ScheduledNote> = PHRASE
                .iter()
                .map(|(pitch, offset, length)| ScheduledNote {
                    start_frame: start + offset * sixteenth,
                    end_frame: start + (offset + length) * sixteenth,
                    pitch: (i16::from(*pitch) + transpose).clamp(0, 127) as u8,
                    // Play it, rather than typing it: an accent on the downbeat.
                    velocity: if offset % 4 == 0 { 104 } else { 82 },
                })
                .collect();

            let sampled_take = font.as_ref().filter(|_| take == 1);
            let source = if let Some(font) = sampled_take {
                let mut player = font.player(sample_rate)?;
                player.set_program(*program);
                render_into(&mut player, &notes, &mut left, &mut right, &mut send);
                "sampled"
            } else {
                let mut synth = Synth::new(sample_rate, Arc::clone(&bank));
                synth.set_program(*program);
                render_into(&mut synth, &notes, &mut left, &mut right, &mut send);
                "synthesised"
            };
            println!(
                "bar {:>2}: {:<24} {source}",
                index as u64 * takes + take + 1,
                program_name(*program)
            );
        }
    }

    let mut drums = Synth::new(sample_rate, Arc::clone(&bank));
    drums.set_drum_kit(true);
    drums.set_level(0.4);
    let mut sampled_drums = None;
    if let Some(font) = font.as_ref() {
        let mut player = font.player(sample_rate)?;
        player.set_drum_kit(true);
        player.set_level(0.7);
        sampled_drums = Some(player);
    }
    let mut hits: Vec<ScheduledNote> = (0..bars)
        .flat_map(|index| {
            GROOVE.iter().map(move |(note, offset)| ScheduledNote {
                start_frame: bar * index + offset * sixteenth,
                end_frame: bar * index + offset * sixteenth + sixteenth,
                pitch: *note,
                velocity: if *offset % 4 == 0 { 110 } else { 78 },
            })
        })
        .collect();
    hits.sort_by_key(|note| note.start_frame);
    // The kit alternates with the instruments, so the same groove is heard from
    // both sources rather than one being judged over the other's drums.
    let (synth_hits, sampled_hits): (Vec<_>, Vec<_>) = hits
        .into_iter()
        .partition(|hit| takes == 1 || (hit.start_frame / bar) % 2 == 0);
    render_into(&mut drums, &synth_hits, &mut left, &mut right, &mut send);
    if let Some(player) = sampled_drums.as_mut() {
        render_into(player, &sampled_hits, &mut left, &mut right, &mut send);
    }

    reverb.process(&send, &mut left, &mut right);

    let peak = left
        .iter()
        .chain(right.iter())
        .fold(0.0_f32, |peak, value| peak.max(value.abs()));
    println!("peak {peak:.3}");

    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: sample_rate.get(),
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&destination, spec)?;
    for (left, right) in left.iter().zip(right.iter()) {
        writer.write_sample((left.clamp(-1.0, 1.0) * 32_767.0) as i16)?;
        writer.write_sample((right.clamp(-1.0, 1.0) * 32_767.0) as i16)?;
    }
    writer.finalize()?;
    println!("wrote {}", destination.display());
    Ok(())
}

/// Either renderer, so the audition can drive both from one place.
trait Instrument {
    fn render(&mut self, notes: &[ScheduledNote], left: &mut [f32], right: &mut [f32]);
    fn reverb_send(&self) -> f32;
}

impl Instrument for Synth {
    fn render(&mut self, notes: &[ScheduledNote], left: &mut [f32], right: &mut [f32]) {
        Self::render(self, notes, 0, left, right);
    }
    fn reverb_send(&self) -> f32 {
        Self::reverb_send(self)
    }
}

impl Instrument for SampledSynth {
    fn render(&mut self, notes: &[ScheduledNote], left: &mut [f32], right: &mut [f32]) {
        Self::render(self, notes, 0, left, right);
    }
    fn reverb_send(&self) -> f32 {
        Self::reverb_send(self)
    }
}

/// Renders a whole part in one block, and takes its reverb send with it.
fn render_into(
    synth: &mut impl Instrument,
    notes: &[ScheduledNote],
    left: &mut [f32],
    right: &mut [f32],
    send: &mut [[f32; 2]],
) {
    let frames = left.len();
    let mut part_left = vec![0.0_f32; frames];
    let mut part_right = vec![0.0_f32; frames];
    synth.render(notes, &mut part_left, &mut part_right);
    let amount = synth.reverb_send();
    for index in 0..frames {
        left[index] += part_left[index];
        right[index] += part_right[index];
        send[index][0] += part_left[index] * amount;
        send[index][1] += part_right[index] * amount;
    }
}
