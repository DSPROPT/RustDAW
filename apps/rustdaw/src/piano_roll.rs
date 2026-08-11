#![allow(
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

//! The piano roll: a note editor for instrument tracks.
//!
//! Drawn on one canvas rather than as widgets. A transcribed guitar part is
//! several hundred notes and a widget each would cost a layout pass per frame,
//! so notes are painted directly and hit-tested by hand.
//!
//! Horizontal position is musical time in ticks, never seconds, so editing
//! stays correct when the tempo map changes underneath it.

use daw_midi::{MidiClip, Note, TICKS_PER_QUARTER, TempoMap, is_accidental, pitch_name};
use eframe::egui::{
    self, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2,
};

use crate::theme;

/// Width of the keyboard down the left edge.
const KEYBOARD_WIDTH: f32 = 56.0;
/// Vertical pixels per semitone at 100% zoom.
const DEFAULT_ROW_HEIGHT: f32 = 13.0;
const MIN_ROW_HEIGHT: f32 = 5.0;
const MAX_ROW_HEIGHT: f32 = 34.0;
/// Grab area at a note's right edge for changing its length.
const RESIZE_GRIP: f32 = 6.0;
const LOWEST_PITCH: u8 = 21;
const HIGHEST_PITCH: u8 = 108;

/// Note lengths offered by the grid selector, in ticks.
const GRID_DIVISIONS: [(&str, u64); 6] = [
    ("1/1", TICKS_PER_QUARTER as u64 * 4),
    ("1/2", TICKS_PER_QUARTER as u64 * 2),
    ("1/4", TICKS_PER_QUARTER as u64),
    ("1/8", TICKS_PER_QUARTER as u64 / 2),
    ("1/16", TICKS_PER_QUARTER as u64 / 4),
    ("1/32", TICKS_PER_QUARTER as u64 / 8),
];

/// What the pointer is currently doing to a note.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Drag {
    Move {
        index: usize,
        /// Offset from the note's start to where the pointer grabbed it.
        grab_ticks: i64,
        original_pitch: u8,
    },
    Resize {
        index: usize,
    },
}

pub struct PianoRollState {
    pub open: bool,
    /// Which track is being edited.
    pub track: usize,
    /// Which clip within that track.
    pub clip: usize,
    pub pixels_per_quarter: f32,
    pub row_height: f32,
    /// Leftmost visible tick.
    pub scroll_ticks: f64,
    /// Pitch shown at the bottom of the view.
    pub bottom_pitch: f32,
    pub grid_ticks: u64,
    pub snap: bool,
    pub default_velocity: u8,
    selected: Option<usize>,
    drag: Option<Drag>,
    /// Set when an edit changed the notes, so the caller can re-schedule.
    pub dirty: bool,
}

impl Default for PianoRollState {
    fn default() -> Self {
        Self {
            open: false,
            track: 0,
            clip: 0,
            pixels_per_quarter: 96.0,
            row_height: DEFAULT_ROW_HEIGHT,
            scroll_ticks: 0.0,
            bottom_pitch: 48.0,
            grid_ticks: u64::from(TICKS_PER_QUARTER) / 4,
            snap: true,
            default_velocity: 100,
            selected: None,
            drag: None,
            dirty: false,
        }
    }
}

impl PianoRollState {
    /// Opens the editor on a clip, framing its notes.
    pub fn open_clip(&mut self, track: usize, clip_index: usize, clip: &MidiClip) {
        self.open = true;
        self.track = track;
        self.clip = clip_index;
        self.selected = None;
        self.drag = None;
        if let (Some(low), Some(high)) = (clip.lowest_pitch(), clip.highest_pitch()) {
            let centre = f32::from(low) + f32::from(high - low) / 2.0;
            self.bottom_pitch = (centre - 12.0).clamp(f32::from(LOWEST_PITCH), 96.0);
        }
        self.scroll_ticks = 0.0;
    }

    fn ticks_per_pixel(&self) -> f64 {
        f64::from(TICKS_PER_QUARTER) / f64::from(self.pixels_per_quarter.max(1.0))
    }

    fn tick_at_x(&self, x: f32, origin_x: f32) -> i64 {
        let offset = f64::from(x - origin_x) * self.ticks_per_pixel();
        (self.scroll_ticks + offset) as i64
    }

    fn x_at_tick(&self, tick: f64, origin_x: f32) -> f32 {
        origin_x + ((tick - self.scroll_ticks) / self.ticks_per_pixel()) as f32
    }

    fn pitch_at_y(&self, y: f32, bottom_y: f32) -> i32 {
        // `floor`, not a truncating cast: below the bottom of the view the row
        // count goes negative, and truncation rounds those towards zero, which
        // would pick the wrong row for anything under the lowest visible pitch.
        let rows = ((bottom_y - y) / self.row_height).floor();
        self.bottom_pitch as i32 + rows as i32
    }

    fn y_at_pitch(&self, pitch: f32, bottom_y: f32) -> f32 {
        bottom_y - (pitch - self.bottom_pitch + 1.0) * self.row_height
    }

    fn snap_tick(&self, tick: i64) -> i64 {
        if !self.snap || self.grid_ticks == 0 {
            return tick.max(0);
        }
        let grid = self.grid_ticks as i64;
        ((tick + grid / 2) / grid * grid).max(0)
    }
}

/// Result of drawing the piano roll for one frame.
pub struct PianoRollResponse {
    /// Notes changed; the caller must re-schedule playback.
    pub edited: bool,
    /// The user asked to move the playhead to this tick.
    pub seek_to_tick: Option<u64>,
}

/// Draws the editor and applies edits to `clip`.
///
/// `playhead_tick` positions the cursor; `tempo` is used only for the ruler.
#[allow(clippy::too_many_lines)]
pub fn show(
    state: &mut PianoRollState,
    ui: &mut egui::Ui,
    clip: &mut MidiClip,
    tempo: &TempoMap,
    beats_per_bar: u16,
    playhead_tick: u64,
) -> PianoRollResponse {
    let mut response = PianoRollResponse {
        edited: false,
        seek_to_tick: None,
    };

    toolbar(state, ui, clip);
    ui.separator();

    let available = ui.available_size_before_wrap();
    let (rect, canvas) = ui.allocate_exact_size(
        Vec2::new(available.x, available.y.max(220.0)),
        Sense::click_and_drag(),
    );
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, CornerRadius::ZERO, theme::PANEL);

    let grid_rect = Rect::from_min_max(
        Pos2::new(rect.min.x + KEYBOARD_WIDTH, rect.min.y),
        rect.max,
    );
    let bottom_y = grid_rect.max.y;
    let origin_x = grid_rect.min.x;

    // Zoom and scroll before anything is drawn, so the frame reflects them.
    if canvas.hovered() {
        let scroll = ui.input(|input| input.smooth_scroll_delta);
        let (zoom_h, zoom_v) = ui.input(|input| {
            (
                input.modifiers.ctrl || input.modifiers.command,
                input.modifiers.shift,
            )
        });
        if zoom_h && scroll.y != 0.0 {
            state.pixels_per_quarter = (state.pixels_per_quarter * (1.0 + scroll.y * 0.006))
                .clamp(8.0, 900.0);
        } else if zoom_v && scroll.y != 0.0 {
            state.row_height =
                (state.row_height * (1.0 + scroll.y * 0.006)).clamp(MIN_ROW_HEIGHT, MAX_ROW_HEIGHT);
        } else {
            if scroll.y != 0.0 {
                state.bottom_pitch = (state.bottom_pitch - scroll.y / state.row_height)
                    .clamp(f32::from(LOWEST_PITCH), f32::from(HIGHEST_PITCH) - 4.0);
            }
            if scroll.x != 0.0 {
                state.scroll_ticks =
                    (state.scroll_ticks - f64::from(scroll.x) * state.ticks_per_pixel()).max(0.0);
            }
        }
    }

    draw_rows(state, &painter, grid_rect, bottom_y);
    draw_grid_lines(state, &painter, grid_rect, origin_x, tempo, beats_per_bar);
    draw_keyboard(state, &painter, rect, bottom_y);

    // Notes.
    let visible_end = state.tick_at_x(grid_rect.max.x, origin_x);
    for (index, note) in clip.notes.iter().enumerate() {
        let start = note.start_tick as i64;
        if start > visible_end || (note.end_tick() as i64) < state.scroll_ticks as i64 {
            continue;
        }
        let x0 = state.x_at_tick(note.start_tick as f64, origin_x);
        let x1 = state.x_at_tick(note.end_tick() as f64, origin_x).max(x0 + 2.0);
        let y = state.y_at_pitch(f32::from(note.pitch), bottom_y);
        if y > grid_rect.max.y || y + state.row_height < grid_rect.min.y {
            continue;
        }
        let note_rect = Rect::from_min_max(
            Pos2::new(x0, y + 1.0),
            Pos2::new(x1, y + state.row_height - 1.0),
        );
        let selected = state.selected == Some(index);
        // Velocity drives brightness, so dynamics are visible at a glance.
        let intensity = 0.35 + f32::from(note.velocity) / 127.0 * 0.65;
        let fill = if selected {
            theme::GREEN
        } else {
            Color32::from_rgb(
                (60.0 + 90.0 * intensity) as u8,
                (120.0 + 110.0 * intensity) as u8,
                (190.0 + 60.0 * intensity) as u8,
            )
        };
        painter.rect_filled(note_rect, CornerRadius::same(2), fill);
        painter.rect_stroke(
            note_rect,
            CornerRadius::same(2),
            Stroke::new(1.0_f32, if selected { Color32::WHITE } else { theme::BORDER }),
            StrokeKind::Inside,
        );
    }

    // Playhead.
    let playhead_x = state.x_at_tick(playhead_tick as f64, origin_x);
    if grid_rect.x_range().contains(playhead_x) {
        painter.line_segment(
            [
                Pos2::new(playhead_x, grid_rect.min.y),
                Pos2::new(playhead_x, grid_rect.max.y),
            ],
            Stroke::new(1.5_f32, theme::RED),
        );
    }

    handle_input(
        state,
        ui,
        &canvas,
        clip,
        origin_x,
        bottom_y,
        &mut response,
    );

    response.edited |= state.dirty;
    state.dirty = false;
    response
}

fn toolbar(state: &mut PianoRollState, ui: &mut egui::Ui, clip: &MidiClip) {
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(&clip.name)
                .strong()
                .color(theme::BLUE),
        );
        ui.label(
            egui::RichText::new(format!("{} notes", clip.notes.len()))
                .small()
                .color(theme::MUTED),
        );
        ui.separator();
        ui.checkbox(&mut state.snap, "Snap");
        egui::ComboBox::from_id_salt("piano_roll_grid")
            .selected_text(
                GRID_DIVISIONS
                    .iter()
                    .find(|(_, ticks)| *ticks == state.grid_ticks)
                    .map_or("1/16", |(label, _)| *label),
            )
            .width(64.0)
            .show_ui(ui, |ui| {
                for (label, ticks) in GRID_DIVISIONS {
                    ui.selectable_value(&mut state.grid_ticks, ticks, label);
                }
            });
        ui.separator();
        ui.label("Velocity");
        ui.add(
            egui::DragValue::new(&mut state.default_velocity)
                .range(1..=127)
                .speed(1.0),
        );
        ui.separator();
        ui.label(
            egui::RichText::new(
                "double-click adds · drag moves · right edge resizes · Del removes",
            )
            .small()
            .color(theme::MUTED),
        );
    });
}

fn draw_rows(state: &PianoRollState, painter: &egui::Painter, rect: Rect, bottom_y: f32) {
    let rows = (rect.height() / state.row_height).ceil() as i32 + 1;
    for row in 0..rows {
        let pitch = state.bottom_pitch as i32 + row;
        if !(0..=127).contains(&pitch) {
            continue;
        }
        let y = state.y_at_pitch(pitch as f32, bottom_y);
        let row_rect = Rect::from_min_max(
            Pos2::new(rect.min.x, y),
            Pos2::new(rect.max.x, y + state.row_height),
        );
        // Black-key rows are shaded so the octave is readable without labels.
        let fill = if is_accidental(pitch as u8) {
            Color32::from_rgb(24, 28, 34)
        } else {
            Color32::from_rgb(32, 37, 44)
        };
        painter.rect_filled(row_rect.intersect(rect), CornerRadius::ZERO, fill);
        if pitch % 12 == 0 {
            painter.line_segment(
                [Pos2::new(rect.min.x, y), Pos2::new(rect.max.x, y)],
                Stroke::new(1.0_f32, Color32::from_rgb(60, 66, 76)),
            );
        }
    }
}

fn draw_grid_lines(
    state: &PianoRollState,
    painter: &egui::Painter,
    rect: Rect,
    origin_x: f32,
    _tempo: &TempoMap,
    beats_per_bar: u16,
) {
    let quarter = f64::from(TICKS_PER_QUARTER);
    let bar_ticks = quarter * f64::from(beats_per_bar.max(1));
    let step = state.grid_ticks.max(1) as f64;
    // Only draw the subdivision when it is wide enough to read.
    let draw_subdivision = (step / state.ticks_per_pixel()) > 5.0;

    let first = (state.scroll_ticks / step).floor() * step;
    let mut tick = first;
    let end = state.tick_at_x(rect.max.x, origin_x) as f64;
    while tick <= end {
        let x = state.x_at_tick(tick, origin_x);
        if x >= origin_x {
            let on_bar = (tick % bar_ticks).abs() < 0.5;
            let on_beat = (tick % quarter).abs() < 0.5;
            if on_bar || on_beat || draw_subdivision {
                let colour = if on_bar {
                    Color32::from_rgb(90, 100, 116)
                } else if on_beat {
                    Color32::from_rgb(60, 68, 80)
                } else {
                    Color32::from_rgb(44, 50, 60)
                };
                painter.line_segment(
                    [Pos2::new(x, rect.min.y), Pos2::new(x, rect.max.y)],
                    Stroke::new(if on_bar { 1.4_f32 } else { 1.0_f32 }, colour),
                );
            }
            if on_bar {
                                let bar = (tick / bar_ticks) as u64 + 1;
                painter.text(
                    Pos2::new(x + 3.0, rect.min.y + 2.0),
                    egui::Align2::LEFT_TOP,
                    bar.to_string(),
                    FontId::proportional(10.0),
                    theme::MUTED,
                );
            }
        }
        tick += step;
    }
}

fn draw_keyboard(state: &PianoRollState, painter: &egui::Painter, rect: Rect, bottom_y: f32) {
    let keys = Rect::from_min_max(
        rect.min,
        Pos2::new(rect.min.x + KEYBOARD_WIDTH, rect.max.y),
    );
    painter.rect_filled(keys, CornerRadius::ZERO, Color32::from_rgb(18, 21, 26));
    let rows = (rect.height() / state.row_height).ceil() as i32 + 1;
    for row in 0..rows {
        let pitch = state.bottom_pitch as i32 + row;
        if !(0..=127).contains(&pitch) {
            continue;
        }
        let y = state.y_at_pitch(pitch as f32, bottom_y);
        let key_rect = Rect::from_min_max(
            Pos2::new(keys.min.x, y + 0.5),
            Pos2::new(keys.max.x - 2.0, y + state.row_height - 0.5),
        );
        if !key_rect.intersects(keys) {
            continue;
        }
        let black = is_accidental(pitch as u8);
        painter.rect_filled(
            key_rect.intersect(keys),
            CornerRadius::same(1),
            if black {
                Color32::from_rgb(30, 34, 40)
            } else {
                Color32::from_rgb(206, 212, 222)
            },
        );
        if pitch % 12 == 0 && state.row_height > 9.0 {
            painter.text(
                Pos2::new(key_rect.max.x - 3.0, key_rect.center().y),
                egui::Align2::RIGHT_CENTER,
                pitch_name(pitch as u8),
                FontId::proportional(9.0),
                Color32::from_rgb(70, 76, 86),
            );
        }
    }
    painter.line_segment(
        [
            Pos2::new(keys.max.x, rect.min.y),
            Pos2::new(keys.max.x, rect.max.y),
        ],
        Stroke::new(1.0_f32, theme::BORDER),
    );
}

#[allow(clippy::too_many_arguments)]
fn handle_input(
    state: &mut PianoRollState,
    ui: &egui::Ui,
    canvas: &egui::Response,
    clip: &mut MidiClip,
    origin_x: f32,
    bottom_y: f32,
    response: &mut PianoRollResponse,
) {
    let pointer = canvas.hover_pos().or_else(|| canvas.interact_pointer_pos());
    let Some(pointer) = pointer else {
        if !canvas.dragged() {
            state.drag = None;
        }
        return;
    };
    if pointer.x < origin_x {
        return;
    }

    let tick = state.tick_at_x(pointer.x, origin_x);
    let pitch = state.pitch_at_y(pointer.y, bottom_y);
    let hit = note_at(state, clip, pointer, origin_x, bottom_y);

    // Double-click on empty space writes a note; on a note, removes it.
    if canvas.double_clicked() {
        if let Some((index, _)) = hit {
            clip.notes.remove(index);
            state.selected = None;
        } else if (0..=127).contains(&pitch) {
            let start = state.snap_tick(tick).max(0);
            let length = if state.snap {
                state.grid_ticks.max(1)
            } else {
                u64::from(TICKS_PER_QUARTER) / 4
            };
            #[allow(clippy::cast_sign_loss)]
            clip.insert_note(Note::new(
                pitch as u8,
                state.default_velocity,
                start as u64,
                length,
            ));
        }
        state.dirty = true;
        return;
    }

    if canvas.drag_started() {
        state.drag = hit.map(|(index, on_edge)| {
            state.selected = Some(index);
            if on_edge {
                Drag::Resize { index }
            } else {
                Drag::Move {
                    index,
                    grab_ticks: tick - clip.notes[index].start_tick as i64,
                    original_pitch: clip.notes[index].pitch,
                }
            }
        });
        if state.drag.is_none() {
            // Clicking empty space moves the playhead instead.
            state.selected = None;
            response.seek_to_tick = Some(state.snap_tick(tick).max(0) as u64);
        }
    }

    if canvas.dragged() {
        match state.drag {
            Some(Drag::Move {
                index,
                grab_ticks,
                original_pitch,
            }) => {
                if let Some(note) = clip.notes.get_mut(index) {
                    let start = state.snap_tick(tick - grab_ticks).max(0);
                    note.start_tick = start as u64;
                    let delta = pitch - i32::from(original_pitch);
                    let moved = i32::from(original_pitch) + delta;
                    if (0..=127).contains(&moved) {
                        note.pitch = moved as u8;
                    }
                    state.dirty = true;
                }
            }
            Some(Drag::Resize { index }) => {
                if let Some(note) = clip.notes.get_mut(index) {
                    let end = state.snap_tick(tick).max(note.start_tick as i64 + 1);
                    note.length_ticks = (end as u64).saturating_sub(note.start_tick).max(1);
                    state.dirty = true;
                }
            }
            None => {}
        }
    }

    if canvas.drag_stopped() {
        if state.drag.is_some() {
            // Moving a note can put it out of order; the scheduler and the
            // painter both assume start-tick order.
            let selected_note: Option<Note> =
                state.selected.and_then(|index| clip.notes.get(index).copied());
            clip.resort();
            state.selected = selected_note.and_then(|note| {
                clip.notes.iter().position(|candidate| *candidate == note)
            });
            state.dirty = true;
        }
        state.drag = None;
    }

    if canvas.hovered()
        && ui.input(|input| input.key_pressed(egui::Key::Delete) || input.key_pressed(egui::Key::Backspace))
    {
        if let Some(index) = state.selected.take() {
            if index < clip.notes.len() {
                clip.notes.remove(index);
                state.dirty = true;
            }
        }
    }
}

/// Finds the note under the pointer, and whether the pointer is on its right
/// edge (the resize grip).
fn note_at(
    state: &PianoRollState,
    clip: &MidiClip,
    pointer: Pos2,
    origin_x: f32,
    bottom_y: f32,
) -> Option<(usize, bool)> {
    // Later notes are drawn on top, so search backwards to match what is seen.
    for (index, note) in clip.notes.iter().enumerate().rev() {
        let x0 = state.x_at_tick(note.start_tick as f64, origin_x);
        let x1 = state.x_at_tick(note.end_tick() as f64, origin_x).max(x0 + 2.0);
        let y = state.y_at_pitch(f32::from(note.pitch), bottom_y);
        let rect = Rect::from_min_max(Pos2::new(x0, y), Pos2::new(x1, y + state.row_height));
        if rect.contains(pointer) {
            return Some((index, pointer.x >= x1 - RESIZE_GRIP));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> PianoRollState {
        PianoRollState {
            pixels_per_quarter: 96.0,
            row_height: 13.0,
            ..PianoRollState::default()
        }
    }

    #[test]
    fn tick_and_pixel_conversions_are_inverse() {
        let state = state();
        for tick in [0_i64, 240, 960, 12_345] {
            let x = state.x_at_tick(tick as f64, 100.0);
            assert!(
                (state.tick_at_x(x, 100.0) - tick).abs() <= 1,
                "tick {tick} did not survive the round trip"
            );
        }
    }

    #[test]
    fn pitch_and_pixel_conversions_are_inverse() {
        let state = state();
        for pitch in [21_i32, 60, 72, 108] {
            let y = state.y_at_pitch(pitch as f32, 500.0);
            // A pitch's row spans row_height pixels; probe inside it.
            assert_eq!(state.pitch_at_y(y + 1.0, 500.0), pitch);
        }
    }

    #[test]
    fn snapping_rounds_to_the_nearest_grid_line() {
        let mut state = state();
        state.grid_ticks = 240; // sixteenth notes
        assert_eq!(state.snap_tick(0), 0);
        assert_eq!(state.snap_tick(100), 0);
        assert_eq!(state.snap_tick(130), 240);
        assert_eq!(state.snap_tick(239), 240);
        assert_eq!(state.snap_tick(-50), 0, "notes cannot start before the clip");
    }

    #[test]
    fn snapping_off_keeps_the_exact_tick() {
        let mut state = state();
        state.snap = false;
        assert_eq!(state.snap_tick(1_234), 1_234);
    }

    #[test]
    fn hit_testing_finds_the_note_under_the_pointer() {
        let state = state();
        let mut clip = MidiClip::new("Test", 0, 0);
        clip.insert_note(Note::new(60, 100, 0, 960));
        let x = state.x_at_tick(100.0, 0.0);
        let y = state.y_at_pitch(60.0, 400.0) + 2.0;
        let (index, on_edge) = note_at(&state, &clip, Pos2::new(x, y), 0.0, 400.0).unwrap();
        assert_eq!(index, 0);
        assert!(!on_edge);
    }

    #[test]
    fn the_right_edge_of_a_note_is_the_resize_grip() {
        let state = state();
        let mut clip = MidiClip::new("Test", 0, 0);
        clip.insert_note(Note::new(60, 100, 0, 960));
        let x = state.x_at_tick(960.0, 0.0) - 2.0;
        let y = state.y_at_pitch(60.0, 400.0) + 2.0;
        let (_, on_edge) = note_at(&state, &clip, Pos2::new(x, y), 0.0, 400.0).unwrap();
        assert!(on_edge);
    }

    #[test]
    fn empty_space_hits_nothing() {
        let state = state();
        let mut clip = MidiClip::new("Test", 0, 0);
        clip.insert_note(Note::new(60, 100, 0, 960));
        let y = state.y_at_pitch(72.0, 400.0) + 2.0;
        assert!(note_at(&state, &clip, Pos2::new(10.0, y), 0.0, 400.0).is_none());
    }

    #[test]
    fn opening_a_clip_frames_its_notes() {
        let mut state = state();
        let mut clip = MidiClip::new("Bass", 0, 0);
        clip.insert_note(Note::new(40, 100, 0, 480));
        clip.insert_note(Note::new(52, 100, 480, 480));
        state.open_clip(2, 1, &clip);
        assert!(state.open);
        assert_eq!(state.track, 2);
        assert_eq!(state.clip, 1);
        // The view should sit below the lowest note so both are visible.
        assert!(state.bottom_pitch <= 40.0, "got {}", state.bottom_pitch);
    }
}
