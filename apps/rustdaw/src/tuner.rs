#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

//! The tuner window: a needle dial, the note, and how far off it is.
//!
//! The reading is smoothed before it is drawn. Pitch detection on a plucked
//! string is noisy for the first moments while the harmonics settle, and a
//! needle that shows every estimate jitters too much to tune against. The
//! reactivity control is the trade between a needle that responds and a needle
//! that sits still.

use daw_analysis::pitch::{self, Reading};
use eframe::egui::{self, Align2, Color32, FontId, RichText, Sense, Stroke, Vec2};

use crate::theme;

/// Frames of input the detector reads. Two periods of the lowest note plus
/// room to spare; at 48 kHz this is 85 ms, which is short enough to feel live.
pub const WINDOW_FRAMES: usize = 4_096;
/// The bottom of the search range, in Hz. Below a five-string bass's low B.
const LOWEST_HZ: f32 = 28.0;
/// The top. Above a guitar's twelfth-fret high E, and well below the harmonics
/// that would otherwise be mistaken for notes.
const HIGHEST_HZ: f32 = 1_400.0;
/// A reading this uncertain is a string being damped or a room being noisy,
/// not a note being tuned.
const MIN_CONFIDENCE: f32 = 0.5;
/// How far either side of the note the dial reads, in cents.
const DIAL_RANGE_CENTS: f32 = 50.0;
/// Readings older than this leave the display, rather than freezing the last
/// note on screen indefinitely.
const HOLD_SECONDS: f32 = 1.5;

pub struct TunerState {
    pub open: bool,
    /// What A4 is tuned to. Moving it retunes every note with it.
    pub reference_hz: f32,
    /// How fast the needle follows, `0` to `1`.
    pub reactivity: f32,
    /// Bass mode: the search stops below the guitar range, so a bass's
    /// fundamental is not passed over in favour of its second harmonic.
    pub bass: bool,
    /// Input frames, slid forward as the interface drains them.
    window: Vec<f32>,
    /// The smoothed reading, and when it was last updated.
    smoothed_cents: f32,
    last: Option<Reading>,
    age: f32,
}

impl Default for TunerState {
    fn default() -> Self {
        Self {
            open: false,
            reference_hz: pitch::DEFAULT_REFERENCE_HZ,
            reactivity: 0.35,
            bass: false,
            window: Vec::with_capacity(WINDOW_FRAMES),
            smoothed_cents: 0.0,
            last: None,
            age: f32::MAX,
        }
    }
}

impl TunerState {
    /// The window the runtime should fill.
    pub fn window_mut(&mut self) -> &mut Vec<f32> {
        &mut self.window
    }

    /// Runs detection over the current window and folds the result into the
    /// smoothed display. `elapsed` is the time since the last call.
    pub fn analyse(&mut self, sample_rate: f32, elapsed: f32) {
        self.age += elapsed;
        let lowest = if self.bass { LOWEST_HZ } else { 60.0 };
        let found = pitch::detect(&self.window, sample_rate, lowest, HIGHEST_HZ)
            .filter(|pitch| pitch.confidence >= MIN_CONFIDENCE)
            .and_then(|pitch| pitch::reading(pitch, self.reference_hz));

        if let Some(reading) = found {
            // A new note snaps; the same note is followed smoothly. Without
            // that, changing string would slide the needle across the dial.
            let same_note = self.last.is_some_and(|last| last.midi == reading.midi);
            let follow = self.reactivity.clamp(0.02, 1.0);
            self.smoothed_cents = if same_note && self.age < HOLD_SECONDS {
                self.smoothed_cents + (reading.cents - self.smoothed_cents) * follow
            } else {
                reading.cents
            };
            self.last = Some(reading);
            self.age = 0.0;
        } else if self.age > HOLD_SECONDS {
            self.last = None;
        }
    }

    /// The reading currently on display, if one is fresh enough to show.
    fn current(&self) -> Option<Reading> {
        self.last.filter(|_| self.age <= HOLD_SECONDS)
    }
}

/// Draws the tuner window. Returns nothing; the state carries the controls.
pub fn window(context: &egui::Context, state: &mut TunerState) {
    let mut open = state.open;
    egui::Window::new("TUNER")
        .open(&mut open)
        .default_width(420.0)
        .default_height(430.0)
        .resizable(false)
        .show(context, |ui| {
            egui::Frame::new()
                .fill(theme::PANEL)
                .inner_margin(14.0)
                .show(ui, |ui| {
                    readouts(ui, state);
                    dial(ui, state);
                    ui.add_space(8.0);
                    strings(ui, state);
                    ui.separator();
                    controls(ui, state);
                });
        });
    state.open = open;
}

/// Frequency on the left, cents on the right, as the hardware does it.
fn readouts(ui: &mut egui::Ui, state: &TunerState) {
    let reading = state.current();
    ui.horizontal(|ui| {
        let hertz = reading.map_or_else(
            || "— Hz".to_owned(),
            |reading| format!("{:.1} Hz", reading.hertz),
        );
        ui.label(RichText::new(hertz).monospace().color(theme::MUTED));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let cents = reading.map_or_else(
                || "—".to_owned(),
                |_| format!("{:+.0} cent", state.smoothed_cents),
            );
            ui.label(RichText::new(cents).monospace().color(theme::MUTED));
        });
    });
}

/// The dial: a ring of ticks, a needle, and the note in the middle.
fn dial(ui: &mut egui::Ui, state: &TunerState) {
    let size = Vec2::splat(300.0);
    let (response, painter) = ui.allocate_painter(size, Sense::hover());
    let rect = response.rect;
    let centre = rect.center() + Vec2::new(0.0, 24.0);
    let radius = rect.width() * 0.45;
    let reading = state.current();
    // The needle only claims to be in tune when there is something to be in
    // tune with, so an idle dial is never green.
    let lit = reading.is_some_and(|_| state.smoothed_cents.abs() <= 3.0);
    let accent = if lit { theme::GREEN } else { theme::RED };

    // Ticks over the top half, every two cents, taller every ten.
    let sweep = std::f32::consts::PI * 0.72;
    let steps = DIAL_RANGE_CENTS as usize;
    for step in 0..=steps {
        let cents = step as f32 * 2.0 - DIAL_RANGE_CENTS;
        let angle = -std::f32::consts::FRAC_PI_2 + cents / DIAL_RANGE_CENTS * sweep * 0.5;
        let major = (cents.abs() % 10.0) < 0.01;
        let length = if major { 18.0 } else { 9.0 };
        let direction = Vec2::new(angle.cos(), angle.sin());
        let outer = centre + direction * radius;
        let inner = centre + direction * (radius - length);
        let colour = if major { theme::TEXT } else { theme::MUTED };
        let width = if major { 2.0_f32 } else { 1.0 };
        painter.line_segment([inner, outer], Stroke::new(width, colour));
    }

    // The in-tune window, drawn as a wedge so the target is a place rather
    // than a line you have to balance on.
    if reading.is_some() {
        let half = 3.0 / DIAL_RANGE_CENTS * sweep * 0.5;
        for step in 0..=12_usize {
            let angle =
                (half * 2.0).mul_add(step as f32 / 12.0, -std::f32::consts::FRAC_PI_2 - half);
            let direction = Vec2::new(angle.cos(), angle.sin());
            painter.line_segment(
                [
                    centre + direction * (radius - 20.0),
                    centre + direction * radius,
                ],
                Stroke::new(2.0_f32, Color32::from_rgba_unmultiplied(120, 200, 120, 60)),
            );
        }
    }

    // The needle.
    if reading.is_some() {
        let cents = state
            .smoothed_cents
            .clamp(-DIAL_RANGE_CENTS, DIAL_RANGE_CENTS);
        let angle = -std::f32::consts::FRAC_PI_2 + cents / DIAL_RANGE_CENTS * sweep * 0.5;
        let direction = Vec2::new(angle.cos(), angle.sin());
        painter.line_segment(
            [
                centre + direction * 60.0,
                centre + direction * (radius - 6.0),
            ],
            Stroke::new(3.0_f32, accent),
        );
    }

    // The note itself, on a disc in the middle.
    painter.circle_filled(centre, 62.0, theme::BG);
    painter.circle_stroke(centre, 62.0, Stroke::new(2.0_f32, accent));
    let label = reading.map_or_else(|| "—".to_owned(), |reading| reading.label());
    painter.text(
        centre,
        Align2::CENTER_CENTER,
        label,
        FontId::proportional(46.0),
        if reading.is_some() {
            accent
        } else {
            theme::MUTED
        },
    );

    // Which way to turn, which is the thing a player actually needs.
    if let Some(reading) = reading {
        let hint = if state.smoothed_cents.abs() <= 3.0 {
            "IN TUNE"
        } else if state.smoothed_cents < 0.0 {
            "FLAT — TIGHTEN"
        } else {
            "SHARP — LOOSEN"
        };
        let _ = reading;
        painter.text(
            centre + Vec2::new(0.0, 92.0),
            Align2::CENTER_CENTER,
            hint,
            FontId::monospace(15.0),
            accent,
        );
    }
}

/// The open strings, with the one being tuned lit.
fn strings(ui: &mut egui::Ui, state: &TunerState) {
    let (notes, names): (&[i32], &[&str]) = if state.bass {
        (&pitch::BASS_STRINGS, &["E", "A", "D", "G"])
    } else {
        (&pitch::GUITAR_STRINGS, &["E", "A", "D", "G", "B", "E"])
    };
    let active = state
        .current()
        .and_then(|reading| pitch::nearest_string(&reading, notes));
    ui.horizontal(|ui| {
        ui.add_space(4.0);
        for (index, name) in names.iter().enumerate() {
            let lit = active == Some(index);
            let (rect, _) = ui.allocate_exact_size(Vec2::new(44.0, 30.0), Sense::hover());
            let painter = ui.painter();
            painter.rect_filled(rect, 4.0, if lit { theme::PANEL_2 } else { theme::BG });
            painter.rect_stroke(
                rect,
                4.0,
                Stroke::new(1.0_f32, if lit { theme::GREEN } else { theme::BORDER }),
                egui::StrokeKind::Inside,
            );
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                name,
                FontId::monospace(15.0),
                if lit { theme::GREEN } else { theme::MUTED },
            );
        }
    });
}

fn controls(ui: &mut egui::Ui, state: &mut TunerState) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("REFERENCE").small().color(theme::MUTED));
        ui.add(
            egui::DragValue::new(&mut state.reference_hz)
                .speed(0.5)
                .range(415.0..=466.0)
                .suffix(" Hz"),
        );
        ui.separator();
        ui.label(RichText::new("REACTIVITY").small().color(theme::MUTED));
        ui.add(egui::Slider::new(&mut state.reactivity, 0.05..=1.0).show_value(false));
        ui.separator();
        if ui
            .selectable_label(state.bass, "BASS")
            .on_hover_text("Search down to a five-string bass's low B")
            .clicked()
        {
            state.bass = !state.bass;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use daw_analysis::pitch::Pitch;

    const RATE: f32 = 48_000.0;

    fn fill(state: &mut TunerState, hertz: f32) {
        state.window.clear();
        for index in 0..WINDOW_FRAMES {
            let phase = index as f32 / RATE * hertz * std::f32::consts::TAU;
            state
                .window
                .push((phase.sin() * 0.3 + (phase * 2.0).sin() * 0.2) * 0.5);
        }
    }

    #[test]
    fn a_held_note_settles_on_its_reading() {
        let mut state = TunerState::default();
        fill(&mut state, 110.0);
        for _ in 0..40 {
            state.analyse(RATE, 0.05);
        }
        let reading = state.current().expect("a note should be showing");
        assert_eq!(reading.label(), "A2");
        assert!(
            state.smoothed_cents.abs() < 2.0,
            "an in-tune note read {:+.1} cents",
            state.smoothed_cents
        );
    }

    #[test]
    fn changing_string_snaps_rather_than_sweeping_across_the_dial() {
        // Smoothing between two different notes would drag the needle through
        // every cent in between, which reads as the string being wildly out.
        let mut state = TunerState {
            reactivity: 0.05,
            ..TunerState::default()
        };
        fill(&mut state, 82.407);
        for _ in 0..20 {
            state.analyse(RATE, 0.05);
        }
        assert_eq!(state.current().expect("low E").label(), "E2");

        fill(&mut state, 329.628);
        state.analyse(RATE, 0.05);
        let reading = state.current().expect("high E");
        assert_eq!(reading.label(), "E4");
        assert!(
            state.smoothed_cents.abs() < 5.0,
            "the needle was still travelling: {:+.1} cents",
            state.smoothed_cents
        );
    }

    #[test]
    fn silence_clears_the_display_rather_than_freezing_it() {
        let mut state = TunerState::default();
        fill(&mut state, 196.0);
        state.analyse(RATE, 0.05);
        assert!(state.current().is_some());

        state.window.clear();
        state.window.resize(WINDOW_FRAMES, 0.0);
        // Held briefly, so a note decaying between plucks does not flicker.
        state.analyse(RATE, 0.2);
        assert!(
            state.current().is_some(),
            "the reading vanished too eagerly"
        );
        state.analyse(RATE, HOLD_SECONDS + 0.1);
        assert!(state.current().is_none(), "the reading never cleared");
    }

    #[test]
    fn the_reference_control_retunes_every_note() {
        let mut state = TunerState {
            reference_hz: 432.0,
            ..TunerState::default()
        };
        fill(&mut state, 216.0); // A3 against a 432 Hz reference.
        for _ in 0..20 {
            state.analyse(RATE, 0.05);
        }
        let reading = state.current().expect("a note");
        assert_eq!(reading.label(), "A3");
        assert!(
            state.smoothed_cents.abs() < 2.0,
            "A=432 read {:+.1} cents out",
            state.smoothed_cents
        );
    }

    #[test]
    fn an_uncertain_reading_is_not_shown() {
        // Noise must not put a confident note on the dial.
        let mut state = TunerState::default();
        state.window.clear();
        // Genuine noise: an indexed expression repeats, and a repeating
        // signal has a pitch, which would test the opposite of the point.
        let mut seed = 0x2545_F491_u32;
        state.window = (0..WINDOW_FRAMES)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                ((seed >> 8) as f32 / 8_388_608.0 - 1.0) * 0.2
            })
            .collect();
        state.analyse(RATE, 0.05);
        state.analyse(RATE, HOLD_SECONDS + 0.1);
        assert!(state.current().is_none(), "noise produced a reading");
    }

    #[test]
    fn a_reading_knows_when_it_is_in_tune() {
        let reading = pitch::reading(
            Pitch {
                hertz: 440.0,
                confidence: 1.0,
            },
            pitch::DEFAULT_REFERENCE_HZ,
        )
        .expect("named");
        assert!(reading.in_tune());
    }
}
