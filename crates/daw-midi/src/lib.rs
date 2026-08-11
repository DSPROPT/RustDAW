//! MIDI for `RustDAW`: notes in musical time, a tempo map, and Standard MIDI
//! File import and export.
//!
//! The rule this crate exists to enforce: audio is positioned in samples,
//! notes are positioned in ticks, and [`tempo::TempoMap`] is the only bridge
//! between them.

pub mod clip;
pub mod smf;
pub mod tempo;

pub use clip::{MidiClip, Note, ScheduledNote};
pub use smf::{SmfFile, SmfTrack};
pub use tempo::{MAX_BPM, MIN_BPM, TICKS_PER_QUARTER, TempoMap, TempoPoint};

/// Note name for a MIDI pitch, e.g. 60 becomes `C3`.
///
/// Octave numbering follows Yamaha/Steinberg convention (middle C = C3), which
/// is what the piano roll displays.
#[must_use]
pub fn pitch_name(pitch: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let octave = i32::from(pitch / 12) - 2;
    format!("{}{octave}", NAMES[usize::from(pitch % 12)])
}

/// True for the black keys of a piano keyboard.
#[must_use]
pub const fn is_accidental(pitch: u8) -> bool {
    matches!(pitch % 12, 1 | 3 | 6 | 8 | 10)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn middle_c_is_named_c3() {
        assert_eq!(pitch_name(60), "C3");
        assert_eq!(pitch_name(61), "C#3");
        assert_eq!(pitch_name(72), "C4");
        assert_eq!(pitch_name(0), "C-2");
    }

    #[test]
    fn accidentals_match_the_keyboard() {
        let black: Vec<u8> = (60..72).filter(|pitch| is_accidental(*pitch)).collect();
        assert_eq!(black, [61, 63, 66, 68, 70]);
    }
}
