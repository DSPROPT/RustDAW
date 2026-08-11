#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

//! Notes and MIDI clips, stored in musical time.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::tempo::TempoMap;

/// A note in a clip, positioned relative to the clip's own start.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Note {
    /// MIDI pitch, 0–127. Middle C is 60.
    pub pitch: u8,
    /// 1–127. Zero velocity is a note-off in MIDI and never a stored note.
    pub velocity: u8,
    pub start_tick: u64,
    pub length_ticks: u64,
}

impl Note {
    #[must_use]
    pub fn new(pitch: u8, velocity: u8, start_tick: u64, length_ticks: u64) -> Self {
        Self {
            pitch: pitch.min(127),
            velocity: velocity.clamp(1, 127),
            start_tick,
            length_ticks: length_ticks.max(1),
        }
    }

    #[must_use]
    pub const fn end_tick(self) -> u64 {
        self.start_tick.saturating_add(self.length_ticks)
    }
}

/// A note with absolute frame positions, ready for the audio thread.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScheduledNote {
    pub start_frame: u64,
    pub end_frame: u64,
    pub pitch: u8,
    pub velocity: u8,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MidiClip {
    #[serde(default = "Uuid::new_v4")]
    pub id: Uuid,
    pub name: String,
    /// Position of the clip on the timeline.
    pub start_tick: u64,
    pub length_ticks: u64,
    pub notes: Vec<Note>,
}

impl MidiClip {
    #[must_use]
    pub fn new(name: impl Into<String>, start_tick: u64, length_ticks: u64) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            start_tick,
            length_ticks,
            notes: Vec::new(),
        }
    }

    /// Adds a note and keeps the clip ordered by start tick.
    ///
    /// Order is an invariant, not a convenience: scheduling and drawing both
    /// walk notes in time order, and re-sorting per audio block would be an
    /// allocation on the wrong thread.
    pub fn insert_note(&mut self, note: Note) {
        let index = self
            .notes
            .partition_point(|existing| existing.start_tick <= note.start_tick);
        self.notes.insert(index, note);
        self.grow_to_fit(note.end_tick());
    }

    /// Extends the clip if a note runs past its end.
    fn grow_to_fit(&mut self, end_tick: u64) {
        self.length_ticks = self.length_ticks.max(end_tick);
    }

    pub fn remove_note(&mut self, index: usize) -> Option<Note> {
        (index < self.notes.len()).then(|| self.notes.remove(index))
    }

    /// Re-sorts after edits that may have moved notes past each other.
    pub fn resort(&mut self) {
        self.notes.sort_by_key(|note| (note.start_tick, note.pitch));
    }

    #[must_use]
    pub const fn end_tick(&self) -> u64 {
        self.start_tick.saturating_add(self.length_ticks)
    }

    #[must_use]
    pub fn lowest_pitch(&self) -> Option<u8> {
        self.notes.iter().map(|note| note.pitch).min()
    }

    #[must_use]
    pub fn highest_pitch(&self) -> Option<u8> {
        self.notes.iter().map(|note| note.pitch).max()
    }

    /// Converts every note to absolute frame positions.
    ///
    /// Notes are clipped to the clip's own bounds so dragging a clip shorter
    /// silences what now hangs past its end, matching what is drawn.
    #[must_use]
    pub fn schedule(&self, tempo: &TempoMap, sample_rate: u32) -> Vec<ScheduledNote> {
        let clip_end = self.end_tick();
        self.notes
            .iter()
            .filter_map(|note| {
                let start = self.start_tick.saturating_add(note.start_tick);
                if start >= clip_end {
                    return None;
                }
                let end = start.saturating_add(note.length_ticks).min(clip_end);
                let start_frame = tempo.tick_to_frame(start, sample_rate);
                let end_frame = tempo.tick_to_frame(end, sample_rate);
                (end_frame > start_frame).then_some(ScheduledNote {
                    start_frame,
                    end_frame,
                    pitch: note.pitch,
                    velocity: note.velocity,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tempo::TICKS_PER_QUARTER;

    #[test]
    fn notes_are_clamped_into_valid_midi_range() {
        let note = Note::new(200, 0, 0, 0);
        assert_eq!(note.pitch, 127);
        assert_eq!(note.velocity, 1, "velocity 0 would mean note-off");
        assert_eq!(note.length_ticks, 1, "a zero-length note is inaudible");
    }

    #[test]
    fn inserting_keeps_notes_ordered_by_start() {
        let mut clip = MidiClip::new("Test", 0, 0);
        for start in [960_u64, 0, 480, 1_920, 240] {
            clip.insert_note(Note::new(60, 100, start, 120));
        }
        let starts: Vec<u64> = clip.notes.iter().map(|note| note.start_tick).collect();
        assert_eq!(starts, [0, 240, 480, 960, 1_920]);
    }

    #[test]
    fn a_clip_grows_to_contain_its_notes() {
        let mut clip = MidiClip::new("Test", 0, 100);
        clip.insert_note(Note::new(60, 100, 1_000, 500));
        assert_eq!(clip.length_ticks, 1_500);
    }

    #[test]
    fn scheduling_places_notes_at_absolute_frames() {
        let tempo = TempoMap::constant(120.0);
        let mut clip = MidiClip::new("Test", u64::from(TICKS_PER_QUARTER), 0);
        clip.insert_note(Note::new(60, 100, 0, u64::from(TICKS_PER_QUARTER)));
        let scheduled = clip.schedule(&tempo, 48_000);
        // Clip starts one quarter in (0.5 s), the note lasts one quarter.
        assert_eq!(scheduled.len(), 1);
        assert_eq!(scheduled[0].start_frame, 24_000);
        assert_eq!(scheduled[0].end_frame, 48_000);
    }

    #[test]
    fn notes_past_the_clip_end_are_trimmed_not_dropped() {
        let tempo = TempoMap::constant(120.0);
        let mut clip = MidiClip::new("Test", 0, 0);
        clip.insert_note(Note::new(60, 100, 0, u64::from(TICKS_PER_QUARTER) * 4));
        clip.length_ticks = u64::from(TICKS_PER_QUARTER); // user dragged it shorter
        let scheduled = clip.schedule(&tempo, 48_000);
        assert_eq!(scheduled.len(), 1);
        assert_eq!(scheduled[0].end_frame, 24_000, "note must stop at the clip end");
    }

    #[test]
    fn notes_starting_past_the_clip_end_are_silent() {
        let tempo = TempoMap::constant(120.0);
        let mut clip = MidiClip::new("Test", 0, 0);
        clip.insert_note(Note::new(60, 100, u64::from(TICKS_PER_QUARTER) * 8, 480));
        clip.length_ticks = u64::from(TICKS_PER_QUARTER);
        assert!(clip.schedule(&tempo, 48_000).is_empty());
    }

    #[test]
    fn scheduling_follows_a_tempo_change() {
        use crate::tempo::TempoPoint;
        let tempo = TempoMap::new(
            vec![
                TempoPoint { tick: 0, bpm: 120.0 },
                TempoPoint {
                    tick: u64::from(TICKS_PER_QUARTER) * 2,
                    bpm: 60.0,
                },
            ],
            TICKS_PER_QUARTER,
        );
        let mut clip = MidiClip::new("Test", 0, 0);
        clip.insert_note(Note::new(60, 100, u64::from(TICKS_PER_QUARTER) * 2, u64::from(TICKS_PER_QUARTER)));
        let scheduled = clip.schedule(&tempo, 48_000);
        // Two quarters at 120 = 1 s; the note then lasts a full second at 60.
        assert_eq!(scheduled[0].start_frame, 48_000);
        assert_eq!(scheduled[0].end_frame, 96_000);
    }
}
