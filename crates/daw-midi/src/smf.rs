#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

//! Standard MIDI File reading and writing.
//!
//! Enough of SMF to be a good citizen: formats 0, 1 and 2, running status,
//! tempo and time-signature meta events, and track names. Everything a
//! `RustDAW` session cannot represent — controllers, pitch bend, sysex — is
//! skipped rather than rejected, because refusing to open a file over a
//! controller lane nobody asked about would be useless behaviour.

use anyhow::{Context, Result, bail, ensure};

use crate::clip::Note;
use crate::tempo::{TICKS_PER_QUARTER, TempoMap, TempoPoint};

const DEFAULT_MICROSECONDS_PER_QUARTER: u32 = 500_000; // 120 BPM

#[derive(Clone, Debug, Default)]
pub struct SmfTrack {
    pub name: String,
    /// The channel its notes were on; channel 9 is the GM drum kit.
    pub channel: Option<u8>,
    /// General MIDI program, when the track sets one.
    pub program: Option<u8>,
    /// Notes in [`TICKS_PER_QUARTER`] ticks from the start of the file.
    pub notes: Vec<Note>,
}

impl SmfTrack {
    #[must_use]
    pub fn is_drums(&self) -> bool {
        self.channel == Some(9)
    }

    #[must_use]
    pub fn end_tick(&self) -> u64 {
        self.notes.iter().map(|note| note.end_tick()).max().unwrap_or(0)
    }
}

#[derive(Clone, Debug)]
pub struct SmfFile {
    pub tracks: Vec<SmfTrack>,
    pub tempo_map: TempoMap,
    pub beats_per_bar: u16,
    pub beat_unit: u16,
}

impl SmfFile {
    /// Tracks that actually contain notes.
    pub fn sounding_tracks(&self) -> impl Iterator<Item = &SmfTrack> {
        self.tracks.iter().filter(|track| !track.notes.is_empty())
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.position)
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(count)
            .filter(|end| *end <= self.bytes.len())
            .context("MIDI file ended in the middle of a chunk")?;
        let slice = &self.bytes[self.position..end];
        self.position = end;
        Ok(slice)
    }

    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }

    fn be_u16(&mut self) -> Result<u16> {
        let bytes = self.take(2)?;
        Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
    }

    fn be_u32(&mut self) -> Result<u32> {
        let bytes = self.take(4)?;
        Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// MIDI variable-length quantity: seven bits per byte, high bit continues.
    fn varint(&mut self) -> Result<u32> {
        let mut value = 0_u32;
        for _ in 0..4 {
            let byte = self.byte()?;
            value = (value << 7) | u32::from(byte & 0x7F);
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        bail!("a variable-length value exceeded four bytes")
    }
}

/// One note-on waiting for its matching note-off.
#[derive(Clone, Copy)]
struct PendingNote {
    pitch: u8,
    velocity: u8,
    start_tick: u64,
}

/// Parses a Standard MIDI File.
///
/// # Errors
///
/// Returns an error if the header is not `MThd`, the file uses SMPTE timing,
/// or a chunk is truncated.
pub fn parse(bytes: &[u8]) -> Result<SmfFile> {
    let mut reader = Reader::new(bytes);
    ensure!(reader.take(4)? == b"MThd", "not a Standard MIDI File");
    let header_length = reader.be_u32()?;
    ensure!(header_length >= 6, "MIDI header chunk is too short");
    let _format = reader.be_u16()?;
    let track_count = reader.be_u16()?;
    let division = reader.be_u16()?;
    // Anything beyond the six header bytes we understand.
    if header_length > 6 {
        reader.take(usize::try_from(header_length - 6).unwrap_or(0))?;
    }
    ensure!(
        division & 0x8000 == 0,
        "SMPTE-timed MIDI files are not supported; this file is not tempo-based"
    );
    let source_ticks_per_quarter = u32::from(division & 0x7FFF).max(1);

    let mut tracks = Vec::new();
    let mut tempo_events: Vec<(u64, f64)> = Vec::new();
    let mut beats_per_bar = 4_u16;
    let mut beat_unit = 4_u16;

    for _ in 0..track_count {
        if reader.remaining() < 8 {
            break;
        }
        let tag = reader.take(4)?;
        let length = usize::try_from(reader.be_u32()?).unwrap_or(0);
        if tag != b"MTrk" {
            // Unknown chunk types must be skipped, per the specification.
            reader.take(length)?;
            continue;
        }
        let chunk = reader.take(length)?;
        let track = parse_track(
            chunk,
            source_ticks_per_quarter,
            &mut tempo_events,
            &mut beats_per_bar,
            &mut beat_unit,
        )?;
        tracks.push(track);
    }

    let mut points: Vec<TempoPoint> = tempo_events
        .into_iter()
        .map(|(tick, bpm)| TempoPoint { tick, bpm })
        .collect();
    if points.is_empty() {
        points.push(TempoPoint {
            tick: 0,
            bpm: microseconds_to_bpm(DEFAULT_MICROSECONDS_PER_QUARTER),
        });
    }

    Ok(SmfFile {
        tracks,
        tempo_map: TempoMap::new(points, TICKS_PER_QUARTER),
        beats_per_bar,
        beat_unit,
    })
}

fn parse_track(
    chunk: &[u8],
    source_ticks_per_quarter: u32,
    tempo_events: &mut Vec<(u64, f64)>,
    beats_per_bar: &mut u16,
    beat_unit: &mut u16,
) -> Result<SmfTrack> {
    let mut reader = Reader::new(chunk);
    let mut track = SmfTrack::default();
    let mut absolute_ticks = 0_u64;
    let mut running_status: Option<u8> = None;
    // One pending note-on per pitch per channel is the usual limit; a repeated
    // note-on for the same pitch ends the previous one, which is what most
    // sequencers do and keeps this bounded.
    let mut pending: Vec<PendingNote> = Vec::new();

    while reader.remaining() > 0 {
        absolute_ticks =
            absolute_ticks.saturating_add(u64::from(reader.varint().context("bad delta time")?));
        let tick = scale_tick(absolute_ticks, source_ticks_per_quarter);

        let status = match reader.peek() {
            Some(byte) if byte & 0x80 != 0 => {
                reader.byte()?;
                running_status = (byte < 0xF0).then_some(byte);
                byte
            }
            Some(_) => running_status.context("MIDI data byte with no running status")?,
            None => break,
        };

        match status {
            0xFF => {
                let meta_type = reader.byte()?;
                let length = usize::try_from(reader.varint()?).unwrap_or(0);
                let data = reader.take(length)?;
                match meta_type {
                    0x03 if track.name.is_empty() => {
                        String::from_utf8_lossy(data).trim().clone_into(&mut track.name);
                    }
                    0x51 if data.len() == 3 => {
                        let microseconds = u32::from(data[0]) << 16
                            | u32::from(data[1]) << 8
                            | u32::from(data[2]);
                        tempo_events.push((tick, microseconds_to_bpm(microseconds)));
                    }
                    0x58 if data.len() >= 2 && tick == 0 => {
                        *beats_per_bar = u16::from(data[0]).clamp(1, 32);
                        *beat_unit = 1_u16 << u32::from(data[1].min(6));
                    }
                    0x2F => break,
                    _ => {}
                }
            }
            0xF0 | 0xF7 => {
                let length = usize::try_from(reader.varint()?).unwrap_or(0);
                reader.take(length)?;
            }
            _ => {
                let channel = status & 0x0F;
                match status & 0xF0 {
                    0x90 | 0x80 => {
                        let pitch = reader.byte()? & 0x7F;
                        let velocity = reader.byte()? & 0x7F;
                        track.channel.get_or_insert(channel);
                        // A note-on with velocity zero is a note-off.
                        if status & 0xF0 == 0x90 && velocity > 0 {
                            close_note(&mut pending, &mut track.notes, pitch, tick);
                            pending.push(PendingNote {
                                pitch,
                                velocity,
                                start_tick: tick,
                            });
                        } else {
                            close_note(&mut pending, &mut track.notes, pitch, tick);
                        }
                    }
                    0xC0 => {
                        let program = reader.byte()? & 0x7F;
                        track.program.get_or_insert(program);
                    }
                    0xD0 => {
                        reader.byte()?;
                    }
                    0xA0 | 0xB0 | 0xE0 => {
                        reader.take(2)?;
                    }
                    _ => bail!("unsupported MIDI status byte {status:#04x}"),
                }
            }
        }
    }

    // A note still sounding at the end of the track gets a short tail rather
    // than being thrown away.
    let end_tick = track.notes.iter().map(|note| note.end_tick()).max().unwrap_or(0);
    for note in pending {
        let end = end_tick.max(note.start_tick + u64::from(TICKS_PER_QUARTER));
        track.notes.push(Note::new(
            note.pitch,
            note.velocity,
            note.start_tick,
            end - note.start_tick,
        ));
    }
    track.notes.sort_by_key(|note| (note.start_tick, note.pitch));
    Ok(track)
}

fn close_note(pending: &mut Vec<PendingNote>, notes: &mut Vec<Note>, pitch: u8, tick: u64) {
    if let Some(index) = pending.iter().rposition(|note| note.pitch == pitch) {
        let note = pending.remove(index);
        let length = tick.saturating_sub(note.start_tick).max(1);
        notes.push(Note::new(note.pitch, note.velocity, note.start_tick, length));
    }
}

/// Rescales a tick from the file's division to [`TICKS_PER_QUARTER`].
fn scale_tick(tick: u64, source_ticks_per_quarter: u32) -> u64 {
    if source_ticks_per_quarter == u32::from(u16::try_from(TICKS_PER_QUARTER).unwrap_or(u16::MAX)) {
        return tick;
    }
    u64::from(TICKS_PER_QUARTER)
        .saturating_mul(tick)
        .checked_div(u64::from(source_ticks_per_quarter))
        .unwrap_or(tick)
}

fn microseconds_to_bpm(microseconds_per_quarter: u32) -> f64 {
    if microseconds_per_quarter == 0 {
        return 120.0;
    }
    60_000_000.0 / f64::from(microseconds_per_quarter)
}

fn bpm_to_microseconds(bpm: f64) -> u32 {
    if bpm <= 0.0 {
        return DEFAULT_MICROSECONDS_PER_QUARTER;
    }
    (60_000_000.0 / bpm).round().clamp(1.0, f64::from(u32::MAX)) as u32
}

/// Writes a format-1 Standard MIDI File: one tempo track plus one track per
/// supplied name/notes pair.
#[must_use]
pub fn write(tracks: &[(&str, &[Note])], tempo: &TempoMap, beats_per_bar: u16) -> Vec<u8> {
    let mut output = Vec::new();
    output.extend_from_slice(b"MThd");
    output.extend_from_slice(&6_u32.to_be_bytes());
    output.extend_from_slice(&1_u16.to_be_bytes()); // format 1
    let track_count = u16::try_from(tracks.len() + 1).unwrap_or(u16::MAX);
    output.extend_from_slice(&track_count.to_be_bytes());
    output.extend_from_slice(
        &u16::try_from(TICKS_PER_QUARTER)
            .unwrap_or(960)
            .to_be_bytes(),
    );

    output.extend_from_slice(&tempo_track(tempo, beats_per_bar));
    for (index, (name, notes)) in tracks.iter().enumerate() {
        let channel = u8::try_from(index % 16).unwrap_or(0);
        output.extend_from_slice(&note_track(name, notes, channel));
    }
    output
}

fn tempo_track(tempo: &TempoMap, beats_per_bar: u16) -> Vec<u8> {
    let mut events = Vec::new();
    let denominator_power = 2_u8; // quarter-note beats
    write_varint(&mut events, 0);
    events.extend_from_slice(&[
        0xFF,
        0x58,
        0x04,
        u8::try_from(beats_per_bar.clamp(1, 32)).unwrap_or(4),
        denominator_power,
        24,
        8,
    ]);

    let mut previous_tick = 0_u64;
    for point in tempo.points() {
        let delta = u32::try_from(point.tick.saturating_sub(previous_tick)).unwrap_or(u32::MAX);
        write_varint(&mut events, delta);
        previous_tick = point.tick;
        let microseconds = bpm_to_microseconds(point.bpm);
        events.extend_from_slice(&[
            0xFF,
            0x51,
            0x03,
            u8::try_from(microseconds >> 16 & 0xFF).unwrap_or(0),
            u8::try_from(microseconds >> 8 & 0xFF).unwrap_or(0),
            u8::try_from(microseconds & 0xFF).unwrap_or(0),
        ]);
    }
    write_varint(&mut events, 0);
    events.extend_from_slice(&[0xFF, 0x2F, 0x00]);
    chunk(*b"MTrk", &events)
}

fn note_track(name: &str, notes: &[Note], channel: u8) -> Vec<u8> {
    let mut events = Vec::new();
    write_varint(&mut events, 0);
    events.extend_from_slice(&[0xFF, 0x03]);
    let name_bytes = name.as_bytes();
    let name_length = name_bytes.len().min(127);
    write_varint(&mut events, u32::try_from(name_length).unwrap_or(0));
    events.extend_from_slice(&name_bytes[..name_length]);

    // Note-on and note-off interleaved in absolute-time order.
    let mut points: Vec<(u64, bool, u8, u8)> = Vec::with_capacity(notes.len() * 2);
    for note in notes {
        points.push((note.start_tick, true, note.pitch, note.velocity));
        points.push((note.end_tick(), false, note.pitch, 0));
    }
    // Note-offs sort before note-ons at the same tick so a repeated pitch
    // retriggers instead of being cut short by its predecessor's release.
    points.sort_by_key(|(tick, is_on, pitch, _)| (*tick, *is_on, *pitch));

    let mut previous_tick = 0_u64;
    for (tick, is_on, pitch, velocity) in points {
        let delta = u32::try_from(tick.saturating_sub(previous_tick)).unwrap_or(u32::MAX);
        write_varint(&mut events, delta);
        previous_tick = tick;
        let status = if is_on { 0x90 } else { 0x80 } | (channel & 0x0F);
        events.extend_from_slice(&[status, pitch & 0x7F, velocity & 0x7F]);
    }
    write_varint(&mut events, 0);
    events.extend_from_slice(&[0xFF, 0x2F, 0x00]);
    chunk(*b"MTrk", &events)
}

fn chunk(tag: [u8; 4], body: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(body.len() + 8);
    output.extend_from_slice(&tag);
    output.extend_from_slice(&u32::try_from(body.len()).unwrap_or(u32::MAX).to_be_bytes());
    output.extend_from_slice(body);
    output
}

fn write_varint(output: &mut Vec<u8>, value: u32) {
    let mut buffer = [0_u8; 4];
    let mut count = 0;
    let mut remaining = value;
    loop {
        buffer[count] = u8::try_from(remaining & 0x7F).unwrap_or(0);
        count += 1;
        remaining >>= 7;
        if remaining == 0 || count == 4 {
            break;
        }
    }
    for index in (0..count).rev() {
        let last = index == 0;
        output.push(buffer[index] | if last { 0 } else { 0x80 });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varints_round_trip_at_the_boundaries() {
        for value in [0_u32, 1, 127, 128, 255, 16_383, 16_384, 2_097_151] {
            let mut encoded = Vec::new();
            write_varint(&mut encoded, value);
            let mut reader = Reader::new(&encoded);
            assert_eq!(reader.varint().unwrap(), value, "failed for {value}");
        }
    }

    #[test]
    fn a_written_file_reads_back_identically() {
        let notes = vec![
            Note::new(60, 100, 0, 480),
            Note::new(64, 90, 480, 480),
            Note::new(67, 80, 960, 1_920),
        ];
        let bytes = write(&[("Piano", &notes)], &TempoMap::constant(120.0), 4);
        let parsed = parse(&bytes).unwrap();
        assert_eq!(parsed.tracks.len(), 2, "tempo track plus one note track");
        let track = parsed.sounding_tracks().next().unwrap();
        assert_eq!(track.name, "Piano");
        assert_eq!(track.notes, notes);
        assert!((parsed.tempo_map.bpm_at_tick(0) - 120.0).abs() < 1e-9);
    }

    #[test]
    fn tempo_changes_survive_a_round_trip() {
        let tempo = TempoMap::new(
            vec![
                TempoPoint { tick: 0, bpm: 96.0 },
                TempoPoint { tick: 3_840, bpm: 144.0 },
            ],
            TICKS_PER_QUARTER,
        );
        let notes = [Note::new(60, 100, 0, 480)];
        let bytes = write(&[("Test", &notes)], &tempo, 3);
        let parsed = parse(&bytes).unwrap();
        assert_eq!(parsed.beats_per_bar, 3);
        assert_eq!(parsed.tempo_map.points().len(), 2);
        assert!((parsed.tempo_map.bpm_at_tick(4_000) - 144.0).abs() < 0.01);
    }

    #[test]
    fn running_status_is_understood() {
        // Two note-ons sharing one status byte, then two note-offs.
        let mut events = Vec::new();
        write_varint(&mut events, 0);
        events.extend_from_slice(&[0x90, 60, 100]);
        write_varint(&mut events, 0);
        events.extend_from_slice(&[62, 100]); // running status
        write_varint(&mut events, 480);
        events.extend_from_slice(&[60, 0]); // note-on velocity 0 = off
        write_varint(&mut events, 0);
        events.extend_from_slice(&[62, 0]);
        write_varint(&mut events, 0);
        events.extend_from_slice(&[0xFF, 0x2F, 0x00]);

        let mut file = Vec::new();
        file.extend_from_slice(b"MThd");
        file.extend_from_slice(&6_u32.to_be_bytes());
        file.extend_from_slice(&0_u16.to_be_bytes());
        file.extend_from_slice(&1_u16.to_be_bytes());
        file.extend_from_slice(&960_u16.to_be_bytes());
        file.extend_from_slice(&chunk(*b"MTrk", &events));

        let parsed = parse(&file).unwrap();
        assert_eq!(parsed.tracks[0].notes.len(), 2);
        assert!(parsed.tracks[0].notes.iter().all(|note| note.length_ticks == 480));
    }

    #[test]
    fn ticks_are_rescaled_from_the_files_division() {
        // 480 ticks per quarter in the file must become 960 internally.
        let mut events = Vec::new();
        write_varint(&mut events, 0);
        events.extend_from_slice(&[0x90, 60, 100]);
        write_varint(&mut events, 480);
        events.extend_from_slice(&[0x80, 60, 0]);
        write_varint(&mut events, 0);
        events.extend_from_slice(&[0xFF, 0x2F, 0x00]);

        let mut file = Vec::new();
        file.extend_from_slice(b"MThd");
        file.extend_from_slice(&6_u32.to_be_bytes());
        file.extend_from_slice(&0_u16.to_be_bytes());
        file.extend_from_slice(&1_u16.to_be_bytes());
        file.extend_from_slice(&480_u16.to_be_bytes());
        file.extend_from_slice(&chunk(*b"MTrk", &events));

        let parsed = parse(&file).unwrap();
        assert_eq!(parsed.tracks[0].notes[0].length_ticks, 960);
    }

    #[test]
    fn drum_tracks_are_recognised_by_channel() {
        let mut events = Vec::new();
        write_varint(&mut events, 0);
        events.extend_from_slice(&[0x99, 36, 100]); // channel 10 (index 9)
        write_varint(&mut events, 120);
        events.extend_from_slice(&[0x89, 36, 0]);
        write_varint(&mut events, 0);
        events.extend_from_slice(&[0xFF, 0x2F, 0x00]);

        let mut file = Vec::new();
        file.extend_from_slice(b"MThd");
        file.extend_from_slice(&6_u32.to_be_bytes());
        file.extend_from_slice(&0_u16.to_be_bytes());
        file.extend_from_slice(&1_u16.to_be_bytes());
        file.extend_from_slice(&960_u16.to_be_bytes());
        file.extend_from_slice(&chunk(*b"MTrk", &events));

        assert!(parse(&file).unwrap().tracks[0].is_drums());
    }

    #[test]
    fn junk_is_rejected_rather_than_misread() {
        assert!(parse(b"").is_err());
        assert!(parse(b"RIFF____WAVEfmt ").is_err());
        assert!(parse(b"MThd\x00\x00\x00\x06\x00\x00\x00\x01").is_err());
    }

    #[test]
    fn smpte_timing_is_reported_clearly() {
        let mut file = Vec::new();
        file.extend_from_slice(b"MThd");
        file.extend_from_slice(&6_u32.to_be_bytes());
        file.extend_from_slice(&0_u16.to_be_bytes());
        file.extend_from_slice(&1_u16.to_be_bytes());
        file.extend_from_slice(&0xE250_u16.to_be_bytes()); // SMPTE
        let error = parse(&file).unwrap_err().to_string();
        assert!(error.contains("SMPTE"), "unhelpful message: {error}");
    }

    #[test]
    fn a_note_left_hanging_still_sounds() {
        let mut events = Vec::new();
        write_varint(&mut events, 0);
        events.extend_from_slice(&[0x90, 60, 100]);
        write_varint(&mut events, 0);
        events.extend_from_slice(&[0xFF, 0x2F, 0x00]);

        let mut file = Vec::new();
        file.extend_from_slice(b"MThd");
        file.extend_from_slice(&6_u32.to_be_bytes());
        file.extend_from_slice(&0_u16.to_be_bytes());
        file.extend_from_slice(&1_u16.to_be_bytes());
        file.extend_from_slice(&960_u16.to_be_bytes());
        file.extend_from_slice(&chunk(*b"MTrk", &events));

        let parsed = parse(&file).unwrap();
        assert_eq!(parsed.tracks[0].notes.len(), 1);
        assert!(parsed.tracks[0].notes[0].length_ticks > 0);
    }
}
