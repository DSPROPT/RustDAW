#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    // "SoundFont" is the name of the file format, not of a type, everywhere it
    // appears in prose here.
    clippy::doc_markdown
)]

//! Instrument tracks played from a SoundFont, when one is available.
//!
//! [`crate::gm`] synthesises every instrument from a handful of parameters,
//! which always works and needs no asset file. A SoundFont is recordings of the
//! real thing, so when the user has one it wins on realism outright — and when
//! they do not, nothing breaks.
//!
//! [`SampledSynth`] presents the same surface as [`crate::Synth`]: give it the
//! track's notes and the block's start frame, and it adds a stereo block to the
//! output. The mixer picks whichever it has and treats the two identically.
//!
//! Real-time contract: [`SampledSynth::render`] allocates nothing and locks
//! nothing. Everything it needs — the voice pool, the scratch buffers, the
//! sample data behind an `Arc` — is in place before the stream opens.
//!
//! # Timing
//!
//! The underlying synthesiser applies note events at the start of its internal
//! block, so [`SampledSynth::render`] splits each callback at every note
//! boundary and renders the pieces. What remains is a rounding of at most one
//! internal block — under a millisecond, which is well inside how evenly a
//! human plays.

use std::fmt;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use daw_core::SampleRate;
use daw_midi::ScheduledNote;
use rustysynth::{SoundFont, Synthesizer, SynthesizerSettings};

use crate::gm;

/// Voices one track may sound at once. A single instrument part never needs
/// the 64 a whole GM device would; keeping it low is what lets every track own
/// a synthesiser outright rather than queueing for a shared one.
const TRACK_POLYPHONY: usize = 24;
/// Frames the underlying synthesiser renders at a time. Smaller means tighter
/// note timing and more per-block overhead; 32 frames is 0.7 ms at 48 kHz.
const BLOCK_FRAMES: usize = 32;
/// Longest callback the scratch buffers are sized for.
const MAX_BLOCK_FRAMES: usize = 4_096;
/// Notes one track may hold down at once, and so the number it can release.
const MAX_SOUNDING: usize = TRACK_POLYPHONY;
/// SoundFont samples are usually recorded in a room already, so they need less
/// help from the bus reverb than the synthesised bank does.
const SAMPLED_REVERB_SCALE: f32 = 0.5;
/// The MIDI channel a pitched track plays on, and the one a kit plays on. Nine
/// is percussion by convention, and the synthesiser reads its kit from bank 128
/// for that channel without being told.
const PITCHED_CHANNEL: i32 = 0;
const PERCUSSION_CHANNEL: i32 = 9;

/// Where to look for a SoundFont when the user has not named one.
///
/// These are where the distributions put the packages that provide one:
/// `fluid-soundfont-gm`, `soundfont-fluid`, `timgm6mb-soundfont`.
const SEARCH_PATHS: [&str; 7] = [
    "/usr/share/soundfonts/default.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
    "/usr/share/sounds/sf2/TimGM6mb.sf2",
    "/usr/share/soundfonts/GeneralUser.sf2",
    "/usr/share/soundfonts/TimGM6mb.sf2",
];

/// The environment variable that overrides discovery, for a font kept
/// somewhere of the user's own choosing.
pub const SOUNDFONT_ENV: &str = "RUSTDAW_SOUNDFONT";

#[derive(Debug)]
pub enum SoundFontError {
    Unreadable { path: PathBuf, source: std::io::Error },
    Malformed { path: PathBuf, reason: String },
    /// The font parsed but has no presets, so it can make no sound.
    Empty { path: PathBuf },
}

impl fmt::Display for SoundFontError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unreadable { path, source } => {
                write!(formatter, "could not read {}: {source}", path.display())
            }
            Self::Malformed { path, reason } => {
                write!(formatter, "{} is not a usable SoundFont: {reason}", path.display())
            }
            Self::Empty { path } => {
                write!(formatter, "{} contains no instruments", path.display())
            }
        }
    }
}

impl std::error::Error for SoundFontError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Unreadable { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// A loaded SoundFont, shared by every track that plays from it.
///
/// The sample data is the bulk of it — tens or hundreds of megabytes — and is
/// held behind an `Arc` so one copy serves the whole session.
pub struct SoundFontBank {
    font: Arc<SoundFont>,
    path: PathBuf,
    name: String,
}

impl SoundFontBank {
    /// Reads a SoundFont from disk.
    ///
    /// This is slow and allocates heavily; it belongs on the control thread,
    /// before the stream opens.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read, is not a SoundFont, or
    /// contains no instruments to play.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SoundFontError> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|source| SoundFontError::Unreadable {
            path: path.clone(),
            source,
        })?;
        let mut reader = BufReader::new(file);
        let font = SoundFont::new(&mut reader).map_err(|error| SoundFontError::Malformed {
            path: path.clone(),
            reason: error.to_string(),
        })?;
        if font.get_presets().is_empty() {
            return Err(SoundFontError::Empty { path });
        }
        let name = font.get_info().get_bank_name().trim().to_owned();
        let name = if name.is_empty() {
            path.file_name()
                .map_or_else(|| path.display().to_string(), |name| name.to_string_lossy().into_owned())
        } else {
            name
        };
        Ok(Self {
            font: Arc::new(font),
            path,
            name,
        })
    }

    /// Finds a SoundFont without being told where one is.
    ///
    /// Checks [`SOUNDFONT_ENV`] first, so a user who has one somewhere of their
    /// own can say so, then the paths the distributions install to. Returns
    /// `None` when there is nothing to find, which is not an error — the
    /// synthesised bank plays instead.
    #[must_use]
    pub fn discover() -> Option<Self> {
        if let Some(named) = std::env::var_os(SOUNDFONT_ENV) {
            // An explicit choice that does not load is worth reporting, unlike
            // a search that simply comes up empty.
            match Self::load(PathBuf::from(named)) {
                Ok(bank) => return Some(bank),
                Err(error) => eprintln!("{SOUNDFONT_ENV}: {error}"),
            }
        }
        SEARCH_PATHS
            .iter()
            .map(Path::new)
            .filter(|path| path.is_file())
            .find_map(|path| Self::load(path).ok())
    }

    /// The font's own name, for display.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// How many instruments the font offers, for display.
    #[must_use]
    pub fn preset_count(&self) -> usize {
        self.font.get_presets().len()
    }

    /// Builds a player for one track. Allocates; not for the audio thread.
    ///
    /// # Errors
    ///
    /// Returns an error when the sample rate is outside what the synthesiser
    /// supports.
    pub fn player(&self, sample_rate: SampleRate) -> Result<SampledSynth, SoundFontError> {
        let mut settings = SynthesizerSettings::new(sample_rate.get() as i32);
        settings.block_size = BLOCK_FRAMES;
        settings.maximum_polyphony = TRACK_POLYPHONY;
        // One reverb for the whole mix, on the bus, rather than sixty-four of
        // them running inside sixty-four synthesisers.
        settings.enable_reverb_and_chorus = false;
        let synthesizer =
            Synthesizer::new(&self.font, &settings).map_err(|error| SoundFontError::Malformed {
                path: self.path.clone(),
                reason: error.to_string(),
            })?;
        Ok(SampledSynth {
            synthesizer,
            program: 0,
            is_drum_kit: false,
            cursor: 0,
            expected_frame: u64::MAX,
            sounding: [(0, 0); MAX_SOUNDING],
            sounding_count: 0,
            scratch_left: vec![0.0; MAX_BLOCK_FRAMES],
            scratch_right: vec![0.0; MAX_BLOCK_FRAMES],
            level: 1.0,
        })
    }
}

// The sample data is tens of megabytes and would be useless in a log.
#[allow(clippy::missing_fields_in_debug)]
impl fmt::Debug for SoundFontBank {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SoundFontBank")
            .field("name", &self.name)
            .field("path", &self.path)
            .field("presets", &self.preset_count())
            .finish()
    }
}

/// One track's worth of SoundFont playback.
///
/// Mirrors [`crate::Synth`]: same scheduling model, same cursor, same
/// behaviour across a seek, and it adds into the output rather than replacing
/// it.
pub struct SampledSynth {
    synthesizer: Synthesizer,
    program: u8,
    is_drum_kit: bool,
    /// Index of the next note to consider, valid only while playing forward.
    cursor: usize,
    /// Frame the next block is expected to start at; a mismatch means a seek.
    expected_frame: u64,
    /// Notes currently held down, as `(end_frame, pitch)`, so each can be
    /// released at its own frame. Fixed size: the audio thread cannot grow it.
    sounding: [(u64, u8); MAX_SOUNDING],
    sounding_count: usize,
    scratch_left: Vec<f32>,
    scratch_right: Vec<f32>,
    level: f32,
}

impl SampledSynth {
    /// Selects the General MIDI program this track plays.
    pub fn set_program(&mut self, program: u8) {
        let program = program.min(127);
        if self.program != program {
            self.program = program;
            self.reset();
            self.send_program();
        }
    }

    /// Switches the track between pitched instrument and drum kit.
    ///
    /// A kit plays on the percussion channel, where the synthesiser draws from
    /// the font's drum bank rather than its melodic presets.
    pub fn set_drum_kit(&mut self, is_drum_kit: bool) {
        if self.is_drum_kit != is_drum_kit {
            self.is_drum_kit = is_drum_kit;
            self.reset();
            self.send_program();
        }
    }

    #[must_use]
    pub const fn program(&self) -> u8 {
        self.program
    }

    pub fn set_level(&mut self, level: f32) {
        self.level = level.clamp(0.0, 2.0);
    }

    /// How much of this track belongs in the shared reverb.
    ///
    /// Taken from the synthesised bank's judgement of the instrument — a hall
    /// for strings, next to nothing for a bass — but scaled back, because a
    /// recorded sample arrives with some of its own room already on it.
    #[must_use]
    pub fn reverb_send(&self) -> f32 {
        if self.is_drum_kit {
            0.06
        } else {
            gm::patch_for_program(self.program).reverb_send * SAMPLED_REVERB_SCALE
        }
    }

    /// Silences every voice, for a stop or a track becoming inaudible.
    pub fn reset(&mut self) {
        self.synthesizer.reset();
        self.cursor = 0;
        self.sounding_count = 0;
        self.expected_frame = u64::MAX;
        self.send_program();
    }

    #[must_use]
    pub fn active_voices(&self) -> usize {
        self.sounding_count
    }

    fn channel(&self) -> i32 {
        if self.is_drum_kit {
            PERCUSSION_CHANNEL
        } else {
            PITCHED_CHANNEL
        }
    }

    fn send_program(&mut self) {
        const PROGRAM_CHANGE: i32 = 0xC0;
        let channel = self.channel();
        self.synthesizer
            .process_midi_message(channel, PROGRAM_CHANGE, i32::from(self.program), 0);
    }

    /// Renders one block, adding into `left` and `right`.
    ///
    /// `notes` must be sorted by `start_frame`, exactly as [`crate::Synth`]
    /// requires.
    pub fn render(
        &mut self,
        notes: &[ScheduledNote],
        block_start: u64,
        left: &mut [f32],
        right: &mut [f32],
    ) {
        if block_start != self.expected_frame {
            self.seek(notes, block_start);
        }
        let frames = left.len().min(right.len());
        // The scratch bounds one pass through the loop, never the whole call: a
        // caller handing over a buffer longer than the scratch must get all of
        // it rendered, not the first few thousand frames and silence after.
        let scratch_frames = self.scratch_left.len().min(self.scratch_right.len()).max(1);
        let block_end = block_start.saturating_add(frames as u64);
        let channel = self.channel();
        let is_drum_kit = self.is_drum_kit;

        // Split the borrow: the synthesiser writes the scratch buffers while
        // the cursor and the held-note list are updated alongside it.
        let Self {
            synthesizer,
            cursor,
            sounding,
            sounding_count,
            scratch_left,
            scratch_right,
            level,
            ..
        } = self;

        let mut offset = 0;
        while offset < frames {
            let frame = block_start.saturating_add(offset as u64);

            // A held note whose time is up, released at its own frame. Drums
            // are one-shots and are never released: the sample's own decay is
            // the length of the hit, so a note written one frame long must not
            // cut a cymbal off.
            if !is_drum_kit {
                let mut index = 0;
                while index < *sounding_count {
                    if sounding[index].0 <= frame {
                        synthesizer.note_off(channel, i32::from(sounding[index].1));
                        sounding[index] = sounding[*sounding_count - 1];
                        *sounding_count -= 1;
                    } else {
                        index += 1;
                    }
                }
            }

            while let Some(note) = notes.get(*cursor) {
                if note.start_frame > frame {
                    break;
                }
                if note.end_frame > frame {
                    synthesizer.note_on(
                        channel,
                        i32::from(note.pitch),
                        i32::from(note.velocity.min(127)),
                    );
                    if !is_drum_kit && *sounding_count < MAX_SOUNDING {
                        sounding[*sounding_count] = (note.end_frame, note.pitch);
                        *sounding_count += 1;
                    }
                }
                *cursor += 1;
            }

            // Render up to the next event, so note timing is not rounded to
            // the callback but to the far smaller internal block.
            let next = next_event(notes, *cursor, sounding, *sounding_count, block_end);
            let chunk = (next.saturating_sub(frame) as usize)
                .clamp(1, (frames - offset).min(scratch_frames));
            synthesizer.render(&mut scratch_left[..chunk], &mut scratch_right[..chunk]);
            for index in 0..chunk {
                left[offset + index] += scratch_left[index] * *level;
                right[offset + index] += scratch_right[index] * *level;
            }
            offset += chunk;
        }

        self.expected_frame = block_end;
    }

    /// Rebuilds state after the transport jumped.
    ///
    /// Notes already sounding at the new position are restarted so scrubbing
    /// into the middle of a chord still plays it. Drums are not: they are
    /// instantaneous, and restarting them would fire every hit that had already
    /// passed, all at once.
    fn seek(&mut self, notes: &[ScheduledNote], frame: u64) {
        self.synthesizer.note_off_all(true);
        self.sounding_count = 0;
        // Strictly before: a note starting exactly on the seek position has not
        // happened yet and must stay pending.
        self.cursor = notes.partition_point(|note| note.start_frame < frame);
        if self.is_drum_kit {
            return;
        }
        let channel = self.channel();
        for note in notes[..self.cursor].iter().rev().take(MAX_SOUNDING) {
            if note.end_frame > frame && self.sounding_count < MAX_SOUNDING {
                self.synthesizer.note_on(
                    channel,
                    i32::from(note.pitch),
                    i32::from(note.velocity.min(127)),
                );
                self.sounding[self.sounding_count] = (note.end_frame, note.pitch);
                self.sounding_count += 1;
            }
        }
    }
}

/// The next frame at which something must happen: a note starting, a held note
/// ending, or the end of the block.
fn next_event(
    notes: &[ScheduledNote],
    cursor: usize,
    sounding: &[(u64, u8); MAX_SOUNDING],
    sounding_count: usize,
    block_end: u64,
) -> u64 {
    let mut next = block_end;
    if let Some(note) = notes.get(cursor) {
        next = next.min(note.start_frame);
    }
    for (end_frame, _) in &sounding[..sounding_count] {
        next = next.min(*end_frame);
    }
    next
}

/// A tiny valid SoundFont, built in memory.
///
/// One preset, one instrument, one region, one sample: enough to load and make
/// a sound. Tests that need a font can use this rather than depending on the
/// machine having one installed, which most do not.
#[cfg(any(test, feature = "fixtures"))]
pub mod fixture {
    /// Little-endian RIFF chunk.
    fn chunk(id: [u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(body.len() + 9);
        out.extend_from_slice(&id);
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(body);
        if body.len() % 2 == 1 {
            out.push(0);
        }
        out
    }

    fn list(form: [u8; 4], body: &[u8]) -> Vec<u8> {
        let mut inner = form.to_vec();
        inner.extend_from_slice(body);
        chunk(*b"LIST", &inner)
    }

    /// A fixed-width, NUL-padded name field.
    fn name(text: &str, width: usize) -> Vec<u8> {
        let mut out = text.as_bytes().to_vec();
        out.resize(width, 0);
        out
    }

    #[must_use]
pub fn soundfont() -> Vec<u8> {
        // INFO: version and a bank name.
        let mut info = Vec::new();
        info.extend_from_slice(&chunk(*b"ifil", &[2, 0, 1, 0]));
        info.extend_from_slice(&chunk(*b"isng", &name("EMU8000", 8)));
        info.extend_from_slice(&chunk(*b"INAM", &name("Test Bank", 10)));

        // sdta: one cycle of a square wave, then the 46 zero frames the
        // spec requires after every sample.
        let mut samples: Vec<i16> = Vec::new();
        for index in 0..256 {
            samples.push(if index < 128 { 12_000 } else { -12_000 });
        }
        samples.extend(std::iter::repeat_n(0, 46));
        let mut wave = Vec::with_capacity(samples.len() * 2);
        for sample in &samples {
            wave.extend_from_slice(&sample.to_le_bytes());
        }
        let sdta = list(*b"sdta", &chunk(*b"smpl", &wave));

        // pdta: the record tables, each terminated by a sentinel entry.
        let mut phdr = Vec::new();
        phdr.extend_from_slice(&name("Test Preset", 20));
        phdr.extend_from_slice(&0u16.to_le_bytes()); // preset
        phdr.extend_from_slice(&0u16.to_le_bytes()); // bank
        phdr.extend_from_slice(&0u16.to_le_bytes()); // bag index
        phdr.extend_from_slice(&[0; 12]); // library, genre, morphology
        phdr.extend_from_slice(&name("EOP", 20));
        phdr.extend_from_slice(&0u16.to_le_bytes());
        phdr.extend_from_slice(&0u16.to_le_bytes());
        phdr.extend_from_slice(&1u16.to_le_bytes());
        phdr.extend_from_slice(&[0; 12]);

        // One bag holding one generator, plus the terminal bag.
        let mut pbag = Vec::new();
        pbag.extend_from_slice(&0u16.to_le_bytes()); // generator index
        pbag.extend_from_slice(&0u16.to_le_bytes()); // modulator index
        pbag.extend_from_slice(&1u16.to_le_bytes());
        pbag.extend_from_slice(&0u16.to_le_bytes());

        let pmod = vec![0u8; 10]; // one terminal modulator

        // Generator 41 is "instrument", pointing at instrument 0.
        let mut pgen = Vec::new();
        pgen.extend_from_slice(&41u16.to_le_bytes());
        pgen.extend_from_slice(&0u16.to_le_bytes());
        pgen.extend_from_slice(&[0; 4]); // terminal generator

        let mut inst = Vec::new();
        inst.extend_from_slice(&name("Test Instrument", 20));
        inst.extend_from_slice(&0u16.to_le_bytes());
        inst.extend_from_slice(&name("EOI", 20));
        inst.extend_from_slice(&1u16.to_le_bytes());

        // The zone now holds two generators before the terminal bag.
        let mut ibag = Vec::new();
        ibag.extend_from_slice(&0u16.to_le_bytes());
        ibag.extend_from_slice(&0u16.to_le_bytes());
        ibag.extend_from_slice(&2u16.to_le_bytes());
        ibag.extend_from_slice(&0u16.to_le_bytes());

        let imod = vec![0u8; 10];

        // Generator 54 is "sampleModes"; 1 loops the sample for as long as the
        // key is held, which is what lets the fixture sustain rather than
        // stopping after a few hundred frames. Generator 53 is "sample", and
        // the spec requires it last in the zone.
        let mut igen = Vec::new();
        igen.extend_from_slice(&54u16.to_le_bytes());
        igen.extend_from_slice(&1u16.to_le_bytes());
        igen.extend_from_slice(&53u16.to_le_bytes());
        igen.extend_from_slice(&0u16.to_le_bytes());
        igen.extend_from_slice(&[0; 4]);

        let mut shdr = Vec::new();
        shdr.extend_from_slice(&name("Test Sample", 20));
        shdr.extend_from_slice(&0u32.to_le_bytes()); // start
        shdr.extend_from_slice(&256u32.to_le_bytes()); // end
        shdr.extend_from_slice(&0u32.to_le_bytes()); // loop start
        shdr.extend_from_slice(&256u32.to_le_bytes()); // loop end
        shdr.extend_from_slice(&44_100u32.to_le_bytes());
        shdr.push(60); // original key
        shdr.push(0); // correction
        shdr.extend_from_slice(&0u16.to_le_bytes()); // sample link
        shdr.extend_from_slice(&1u16.to_le_bytes()); // mono sample
        shdr.extend_from_slice(&name("EOS", 20));
        shdr.extend_from_slice(&[0; 26]);

        let mut pdta_body = Vec::new();
        for (id, body) in [
            (*b"phdr", &phdr),
            (*b"pbag", &pbag),
            (*b"pmod", &pmod),
            (*b"pgen", &pgen),
            (*b"inst", &inst),
            (*b"ibag", &ibag),
            (*b"imod", &imod),
            (*b"igen", &igen),
            (*b"shdr", &shdr),
        ] {
            pdta_body.extend_from_slice(&chunk(id, body));
        }

        let mut body = Vec::new();
        body.extend_from_slice(&list(*b"INFO", &info));
        body.extend_from_slice(&sdta);
        body.extend_from_slice(&list(*b"pdta", &pdta_body));
        let mut riff = b"sfbk".to_vec();
        riff.extend_from_slice(&body);
        chunk(*b"RIFF", &riff)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes the fixture font to a temporary file and loads it.
    fn bank() -> (SoundFontBank, PathBuf) {
        let path = std::env::temp_dir().join(format!(
            "rustdaw-test-{}-{:?}.sf2",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, super::fixture::soundfont()).expect("write the fixture font");
        let bank = SoundFontBank::load(&path).expect("the fixture font should load");
        (bank, path)
    }

    fn player() -> (SampledSynth, PathBuf) {
        let (bank, path) = bank();
        let player = bank
            .player(SampleRate::DEFAULT)
            .expect("a player at the default rate");
        (player, path)
    }

    fn note(start: u64, end: u64, pitch: u8) -> ScheduledNote {
        ScheduledNote {
            start_frame: start,
            end_frame: end,
            pitch,
            velocity: 100,
        }
    }

    fn energy(buffer: &[f32]) -> f32 {
        buffer.iter().map(|value| value.abs()).sum()
    }

    #[test]
    fn a_font_loads_and_describes_itself() {
        let (bank, path) = bank();
        assert_eq!(bank.name(), "Test Bank");
        assert_eq!(bank.preset_count(), 1);
        assert_eq!(bank.path(), path);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_missing_or_broken_font_is_an_error_rather_than_a_panic() {
        let missing = SoundFontBank::load("/nonexistent/nothing-here.sf2");
        assert!(matches!(missing, Err(SoundFontError::Unreadable { .. })));

        let path = std::env::temp_dir().join(format!("rustdaw-broken-{}.sf2", std::process::id()));
        std::fs::write(&path, b"this is not a soundfont").expect("write the junk file");
        let broken = SoundFontBank::load(&path);
        assert!(matches!(broken, Err(SoundFontError::Malformed { .. })));
        // Every error says which file it was about, or it cannot be acted on.
        assert!(broken.unwrap_err().to_string().contains("rustdaw-broken"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_note_sounds_only_while_it_lasts() {
        let (mut player, path) = player();
        let notes = [note(1_000, 5_000, 60)];

        let (mut left, mut right) = (vec![0.0; 512], vec![0.0; 512]);
        player.render(&notes, 0, &mut left, &mut right);
        assert!(energy(&left) < 1e-9, "sound before the note starts");

        let (mut left, mut right) = (vec![0.0; 512], vec![0.0; 512]);
        player.render(&notes, 1_024, &mut left, &mut right);
        assert!(energy(&left) > 0.0, "the note did not sound");

        let (mut left, mut right) = (vec![0.0; 512], vec![0.0; 512]);
        player.render(&notes, 240_000, &mut left, &mut right);
        assert!(energy(&left) < 1e-6, "the note outlived its release");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rendering_adds_to_the_buffer_rather_than_replacing_it() {
        // The mixer sums tracks into one pair of buffers; a renderer that
        // overwrote them would silence everything scheduled before it.
        let (mut player, path) = player();
        let (mut left, mut right) = (vec![0.25_f32; 256], vec![0.25_f32; 256]);
        player.render(&[], 0, &mut left, &mut right);
        assert!(left.iter().all(|value| (*value - 0.25).abs() < f32::EPSILON));
        assert!(right.iter().all(|value| (*value - 0.25).abs() < f32::EPSILON));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn block_size_does_not_change_when_notes_start() {
        // Splitting the callback at note boundaries is the whole point; without
        // it a note's start would be rounded to the callback size.
        let (mut whole, path) = player();
        let notes: Vec<ScheduledNote> = (0..4)
            .map(|index| note(index * 1_000 + 137, index * 1_000 + 800, 60 + index as u8))
            .collect();
        let frames = 4_096;
        let (mut left_whole, mut right_whole) = (vec![0.0; frames], vec![0.0; frames]);
        whole.render(&notes, 0, &mut left_whole, &mut right_whole);

        let (mut split, _) = player();
        let (mut left_split, mut right_split) = (vec![0.0; frames], vec![0.0; frames]);
        for start in (0..frames).step_by(128) {
            split.render(
                &notes,
                start as u64,
                &mut left_split[start..start + 128],
                &mut right_split[start..start + 128],
            );
        }
        // Both runs put every note on at the same frame, so the two renders
        // agree to within the synthesiser's own internal block.
        let difference: f32 = left_whole
            .iter()
            .zip(&left_split)
            .map(|(whole, split)| (whole - split).abs())
            .sum();
        let total = energy(&left_whole).max(1e-9);
        assert!(
            difference / total < 0.05,
            "the callback size changed the render by {:.1}%",
            100.0 * difference / total
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn seeking_into_a_held_note_still_plays_it() {
        let (mut player, path) = player();
        let notes = [note(0, 480_000, 60)];
        let (mut left, mut right) = (vec![0.0; 512], vec![0.0; 512]);
        player.render(&notes, 240_000, &mut left, &mut right);
        assert_eq!(player.active_voices(), 1);
        assert!(energy(&left) > 0.0, "the held note was not restarted");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn seeking_does_not_replay_drum_hits_that_already_passed() {
        let (mut player, path) = player();
        player.set_drum_kit(true);
        let notes: Vec<ScheduledNote> = (0..20)
            .map(|index| note(index * 1_000, index * 1_000 + 10, 36))
            .collect();
        let (mut left, mut right) = (vec![0.0; 256], vec![0.0; 256]);
        player.render(&notes, 19_000, &mut left, &mut right);
        assert_eq!(player.active_voices(), 0, "old hits were refired by the seek");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_drum_track_never_releases_its_hits() {
        // A one-frame note must not cut a cymbal short, so the kit sends no
        // note-off at all and lets the sample run its course.
        let (mut player, path) = player();
        player.set_drum_kit(true);
        let notes = [note(0, 1, 49)];
        let (mut left, mut right) = (vec![0.0; 4_096], vec![0.0; 4_096]);
        player.render(&notes, 0, &mut left, &mut right);
        assert_eq!(player.active_voices(), 0, "a kit should hold nothing down");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn more_notes_than_voices_does_not_overflow() {
        let (mut player, path) = player();
        let notes: Vec<ScheduledNote> = (0..MAX_SOUNDING * 4)
            .map(|index| note(0, 48_000, 30 + (index % 60) as u8))
            .collect();
        let (mut left, mut right) = (vec![0.0; 512], vec![0.0; 512]);
        player.render(&notes, 0, &mut left, &mut right);
        assert!(player.active_voices() <= MAX_SOUNDING);
        assert!(left.iter().all(|value| value.is_finite()));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn zero_length_and_empty_input_are_safe() {
        let (mut player, path) = player();
        player.render(&[], 0, &mut [], &mut []);
        let notes = [note(100, 100, 60)];
        let (mut left, mut right) = (vec![0.0; 64], vec![0.0; 64]);
        player.render(&notes, 0, &mut left, &mut right);
        assert!(left.iter().all(|value| value.is_finite()));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_callback_longer_than_the_scratch_is_rendered_in_full() {
        // The scratch bounds one pass through the render loop, not the call.
        // Truncating instead would leave a long buffer — an offline render, a
        // large callback — sounding for a fraction of a second and then silent.
        let (mut player, path) = player();
        let frames = MAX_BLOCK_FRAMES * 3;
        let notes = [note(0, frames as u64, 60)];
        let (mut left, mut right) = (vec![0.0; frames], vec![0.0; frames]);
        player.render(&notes, 0, &mut left, &mut right);
        assert!(left.iter().all(|value| value.is_finite()));

        let energy = |window: &[f32]| -> f32 { window.iter().map(|value| value.abs()).sum() };
        let first = energy(&left[..MAX_BLOCK_FRAMES]);
        let last = energy(&left[MAX_BLOCK_FRAMES * 2..]);
        assert!(first > 0.0, "the note never started");
        assert!(
            last > first * 0.25,
            "the render stopped at the scratch: {first} then {last}"
        );

        // And the next block still follows on, rather than a seek being
        // detected because the position was only advanced by the scratch.
        let (mut left, mut right) = (vec![0.0; 512], vec![0.0; 512]);
        player.render(&notes, frames as u64, &mut left, &mut right);
        assert!(energy(&left) > 0.0, "the note was dropped at the block edge");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn reset_silences_everything() {
        let (mut player, path) = player();
        let notes = [note(0, 480_000, 60)];
        let (mut left, mut right) = (vec![0.0; 256], vec![0.0; 256]);
        player.render(&notes, 0, &mut left, &mut right);
        assert_eq!(player.active_voices(), 1);
        player.reset();
        assert_eq!(player.active_voices(), 0);
        let (mut left, mut right) = (vec![0.0; 4_096], vec![0.0; 4_096]);
        player.render(&[], 480_000, &mut left, &mut right);
        assert!(energy(&left) < 1e-6, "a voice survived the reset");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_hall_instrument_is_sent_further_into_the_reverb_than_a_bass() {
        let (mut player, path) = player();
        player.set_program(48);
        let strings = player.reverb_send();
        player.set_program(33);
        let bass = player.reverb_send();
        assert!(strings > bass * 2.0, "strings {strings} against bass {bass}");
        // A sample arrives with some of its own room on it, so it needs less
        // than the synthesised bank does.
        assert!(strings < gm::patch_for_program(48).reverb_send);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_kit_plays_on_the_percussion_channel() {
        let (mut player, path) = player();
        assert_eq!(player.channel(), PITCHED_CHANNEL);
        player.set_drum_kit(true);
        assert_eq!(player.channel(), PERCUSSION_CHANNEL);
        let _ = std::fs::remove_file(path);
    }
}
