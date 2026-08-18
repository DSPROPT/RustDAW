//! Changes the key of a session that is already imported.
//!
//! ```text
//! cargo run -p daw-songimport --example rekey-session -- <session.rustdaw.json> -4
//! ```
//!
//! The stems are re-rendered from the originals the session holds, so this is
//! seconds rather than another import, and a key already on disk is instant.

use anyhow::{Context, Result};

fn main() -> Result<()> {
    let session = std::env::args().nth(1).context("pass a session file")?;
    let semitones: i32 = std::env::args()
        .nth(2)
        .context("pass a number of semitones")?
        .parse()?;
    let path = std::path::PathBuf::from(session);
    let directory = path.parent().context("the session has no folder")?;

    let mut document = daw_project::load(&path)?;
    println!(
        "before: {} at {:+} st, key {:?}",
        document.name, document.transpose_semitones, document.key
    );
    let started = std::time::Instant::now();
    let outcome = daw_songimport::rekey_session(&mut document, directory, semitones, &|fraction,
     stem| {
        println!("  {:.0}% {stem}", fraction * 100.0);
    })?;
    daw_project::save_atomic(&document, &path)?;
    println!(
        "after:  {:+} st, key {:?} — {} rendered, {} reused, in {:.1}s",
        document.transpose_semitones,
        document.key,
        outcome.rendered,
        outcome.reused,
        started.elapsed().as_secs_f32()
    );
    for note in &outcome.notes {
        println!("note: {note}");
    }
    Ok(())
}
