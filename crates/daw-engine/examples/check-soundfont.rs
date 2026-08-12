//! Reports which instruments RustDAW will play, and why.
//!
//! ```text
//! cargo run -p daw-engine --release --example check-soundfont
//! ```
//!
//! Instrument tracks play from a sound font when one can be found and from the
//! synthesised bank when one cannot, so "did it pick up my font?" is the first
//! question to ask when the instruments do not sound the way they should. This
//! runs exactly the search the engine runs, and then compares the two paths
//! note for note — a font whose levels sit far from the synthesised bank's will
//! be heard as a jump when a session moves between machines.

#![allow(
    clippy::cast_precision_loss,
    // "RustDAW" and "SoundFont" are names, not items.
    clippy::doc_markdown
)]

use std::sync::Arc;
use std::time::Instant;

use daw_core::SampleRate;
use daw_engine::soundfont::SOUNDFONT_ENV;
use daw_engine::{GmBank, SoundFontBank, Synth, program_name};
use daw_midi::ScheduledNote;

/// A spread of instruments across the families, struck and sustained.
const PROGRAMS: [u8; 12] = [0, 4, 12, 16, 24, 33, 40, 48, 56, 66, 71, 73];

fn main() {
    let sample_rate = SampleRate::DEFAULT;

    println!("{}: {}", SOUNDFONT_ENV, match std::env::var(SOUNDFONT_ENV) {
        Ok(value) => value,
        Err(_) => "unset (searching the usual places)".to_owned(),
    });

    let started = Instant::now();
    let Some(font) = SoundFontBank::discover() else {
        println!("\nNo sound font found. Instrument tracks play the synthesised bank.");
        println!("Install one — `sudo apt install fluid-soundfont-gm` on Ubuntu — or point");
        println!("{SOUNDFONT_ENV} at an .sf2 file.");
        return;
    };
    println!("\nPlaying from: {}", font.name());
    println!("  path:     {}", font.path().display());
    println!("  presets:  {}", font.preset_count());
    println!("  loaded in {:?}", started.elapsed());

    let Ok(mut player) = font.player(sample_rate) else {
        println!("\nThe font loaded but no player could be built for it at this sample rate.");
        return;
    };

    // Two seconds of a held note: long enough to judge a sustaining instrument
    // as well as one that decays away.
    let bank = Arc::new(GmBank::new(sample_rate));
    let frames = sample_rate.get() as usize * 2;
    let rms = |buffer: &[f32]| -> f32 {
        (buffer.iter().map(|value| value * value).sum::<f32>() / buffer.len() as f32).sqrt()
    };

    println!("\n{:<26}{:>10}{:>10}{:>8}", "program", "synth", "font", "ratio");
    let mut ratios = Vec::new();
    for program in PROGRAMS {
        let notes = [ScheduledNote {
            start_frame: 0,
            end_frame: frames as u64,
            pitch: 60,
            velocity: 100,
        }];

        let mut synth = Synth::new(sample_rate, Arc::clone(&bank));
        synth.set_program(program);
        let (mut left, mut right) = (vec![0.0; frames], vec![0.0; frames]);
        synth.render(&notes, 0, &mut left, &mut right);
        let synthesised = rms(&left);

        player.set_program(program);
        player.reset();
        let (mut left, mut right) = (vec![0.0; frames], vec![0.0; frames]);
        player.render(&notes, 0, &mut left, &mut right);
        let sampled = rms(&left);

        let ratio = sampled / synthesised;
        ratios.push(ratio);
        println!(
            "{:<26}{synthesised:>10.4}{sampled:>10.4}{ratio:>8.2}",
            program_name(program)
        );
    }

    ratios.sort_by(f32::total_cmp);
    let median = ratios[ratios.len() / 2];
    println!(
        "\nfont/synth level: median {median:.2}, range {:.2}–{:.2}",
        ratios[0],
        ratios[ratios.len() - 1]
    );
    // A median far from unity means switching between the two paths is heard as
    // a level change rather than a change of instrument.
    if (0.7..=1.4).contains(&median) {
        println!("The two paths are level-matched.");
    } else {
        println!("The two paths are not level-matched; the font is much louder or quieter.");
    }
}
