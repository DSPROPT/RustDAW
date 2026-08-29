//! Teaching the application how far an expression pedal actually travels.
//!
//! Almost no expression pedal sends the full `0` to `127`. The pot never quite
//! reaches either stop, the cable and the socket eat a little more, and every
//! pedal is different — so one that reads `9` at the heel and `112` at the toe
//! is entirely normal. Taken at face value that pedal never closes a wah and
//! never fully opens one, which reads as a broken pedal rather than as an
//! uncalibrated one.
//!
//! Worse, the wiring is not agreed on either. A pedal built for one
//! manufacturer's socket sends `127` at the heel and counts down. Which is why
//! this stores a heel reading and a toe reading rather than a minimum and a
//! maximum: the two ends are named for where the foot is, and either may be
//! the larger number.
//!
//! A run also works out *which* controller the pedal is on, since it is the
//! one that moved. That saves asking somebody to know a number their pedal
//! never told them.

use serde::{Deserialize, Serialize};

use crate::Message;

/// The full travel of a MIDI controller.
const FULL_SCALE: f32 = 127.0;

/// How far apart the two ends must be for the reading to be worth keeping.
///
/// A pedal that moved less than this was not swept: a switch was pressed, or
/// the socket is empty and a stray control change wandered in. Dividing by
/// that span would turn a millimetre of travel into the whole sweep.
const MIN_TRAVEL: u8 = 16;

/// How far an expression pedal travels, in the values it actually sends.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Calibration {
    /// What the pedal sends with the foot rocked all the way back.
    pub heel: u8,
    /// What it sends all the way forward. Smaller than `heel` on a pedal
    /// wired the other way round, which is a difference in wiring and not a
    /// mistake to correct.
    pub toe: u8,
}

impl Default for Calibration {
    /// The whole range, which is what an uncalibrated pedal is assumed to
    /// send until it has been asked.
    fn default() -> Self {
        Self { heel: 0, toe: 127 }
    }
}

impl Calibration {
    /// Where the pedal is, `0` heel-down to `1` toe-down.
    ///
    /// Readings past either end are clamped rather than extrapolated. Pedals
    /// drift, and a foot leaning on the toe stop should be all the way on, not
    /// past it.
    #[must_use]
    pub fn position(&self, value: u8) -> f32 {
        if !self.is_usable() {
            // Nothing sensible to scale against, so pass the pedal through at
            // full range: an uncalibrated sweep is better than none.
            return f32::from(value) / FULL_SCALE;
        }
        let heel = f32::from(self.heel);
        let span = f32::from(self.toe) - heel;
        ((f32::from(value) - heel) / span).clamp(0.0, 1.0)
    }

    /// How far the pedal moved between its two ends.
    #[must_use]
    pub const fn travel(&self) -> u8 {
        self.heel.abs_diff(self.toe)
    }

    /// Whether the two ends are far enough apart to divide by.
    #[must_use]
    pub const fn is_usable(&self) -> bool {
        self.travel() >= MIN_TRAVEL
    }

    /// Whether the pedal counts down as it is pushed forward.
    #[must_use]
    pub const fn is_inverted(&self) -> bool {
        self.heel > self.toe
    }
}

/// Which end of the travel is being read.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stage {
    Heel,
    Toe,
}

impl Stage {
    /// What to do, in the order the two steps are shown.
    #[must_use]
    pub const fn instruction(self) -> &'static str {
        match self {
            Self::Heel => "Rock the pedal all the way back, heel down, and hold it there.",
            Self::Toe => "Now push it all the way forward, toe down, and hold it there.",
        }
    }

    /// The label on this step, as the two ends are marked on the hardware.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Heel => "MIN",
            Self::Toe => "MAX",
        }
    }
}

/// One pass through the two ends of the pedal's travel.
#[derive(Clone, Debug, Default)]
pub struct CalibrationRun {
    stage: Option<Stage>,
    /// The last value every controller sent at each end.
    ///
    /// The last rather than the lowest or the highest, because which of those
    /// is the heel depends on how the pedal is wired. Where the foot came to
    /// rest is the reading, whichever direction it approached from.
    heel: Vec<(u8, u8)>,
    toe: Vec<(u8, u8)>,
}

impl CalibrationRun {
    /// Starts at the heel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            stage: Some(Stage::Heel),
            ..Self::default()
        }
    }

    /// The step being read, or `None` once both ends are done.
    #[must_use]
    pub const fn stage(&self) -> Option<Stage> {
        self.stage
    }

    /// Records a message against the current end.
    ///
    /// Returns what the pedal is reading now, when the message was one a pedal
    /// could have sent. Anything else — a switch, a note — is ignored: a
    /// program change arriving mid-calibration is a foot finding its balance
    /// on the wrong part of the board.
    pub fn observe(&mut self, message: Message) -> Option<(u8, u8)> {
        let Message::Control {
            controller, value, ..
        } = message
        else {
            return None;
        };
        let readings = match self.stage? {
            Stage::Heel => &mut self.heel,
            Stage::Toe => &mut self.toe,
        };
        if let Some(reading) = readings.iter_mut().find(|(seen, _)| *seen == controller) {
            reading.1 = value;
        } else {
            readings.push((controller, value));
        }
        Some((controller, value))
    }

    /// Moves on to the next end, or past the last one.
    pub fn advance(&mut self) {
        self.stage = match self.stage {
            Some(Stage::Heel) => Some(Stage::Toe),
            Some(Stage::Toe) | None => None,
        };
    }

    /// What has been read at `end` so far, as controller and value pairs.
    ///
    /// Shown while a run is going, so the two ends can be compared as they
    /// are taken. A pedal that reads at one stop and goes silent at the other
    /// is a specific, common fault, and it is invisible unless both steps are
    /// on screen at once.
    #[must_use]
    pub fn readings(&self, end: Stage) -> &[(u8, u8)] {
        match end {
            Stage::Heel => &self.heel,
            Stage::Toe => &self.toe,
        }
    }

    /// Whether anything at all has arrived for the current end.
    #[must_use]
    pub fn has_reading(&self) -> bool {
        match self.stage {
            Some(Stage::Heel) => !self.heel.is_empty(),
            Some(Stage::Toe) => !self.toe.is_empty(),
            None => false,
        }
    }

    /// The controller the pedal turned out to be on, and how far it travelled.
    ///
    /// The widest-travelling controller wins. That is not a heuristic so much
    /// as the definition: the expression pedal is the one that swept between
    /// the two ends, and the bank select a footswitch emits alongside its
    /// program change never moves at all.
    ///
    /// # Errors
    ///
    /// If nothing arrived, or nothing moved far enough to scale against.
    pub fn finish(&self) -> Result<(u8, Calibration), String> {
        let mut best: Option<(u8, Calibration)> = None;
        for (controller, heel) in &self.heel {
            let Some((_, toe)) = self.toe.iter().find(|(seen, _)| seen == controller) else {
                continue;
            };
            let candidate = Calibration {
                heel: *heel,
                toe: *toe,
            };
            if best.is_none_or(|(_, widest)| candidate.travel() > widest.travel()) {
                best = Some((*controller, candidate));
            }
        }
        match best {
            Some((controller, calibration)) if calibration.is_usable() => {
                Ok((controller, calibration))
            }
            Some(_) => Err(
                "The pedal barely moved between the two steps. Push it all the way to each \
                 stop and try again."
                    .to_owned(),
            ),
            None => Err(
                "Nothing arrived from the pedal at both ends. Check it is plugged into the \
                 controller's expression socket, and that the controller is connected."
                    .to_owned(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn control(controller: u8, value: u8) -> Message {
        Message::Control {
            channel: 0,
            controller,
            value,
        }
    }

    /// A run that swept `controller` from `heel` to `toe`.
    fn swept(controller: u8, heel: u8, toe: u8) -> CalibrationRun {
        let mut run = CalibrationRun::new();
        run.observe(control(controller, heel));
        run.advance();
        run.observe(control(controller, toe));
        run
    }

    #[test]
    fn an_uncalibrated_pedal_still_sweeps_end_to_end() {
        let calibration = Calibration::default();
        assert!((calibration.position(0) - 0.0).abs() < 1e-6);
        assert!((calibration.position(127) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_pedal_that_never_reaches_the_stops_still_closes_and_opens() {
        // The whole point: 9 to 112 has to mean 0% to 100%, or the wah never
        // shuts and never fully opens.
        let calibration = Calibration {
            heel: 9,
            toe: 112,
        };
        assert!((calibration.position(9) - 0.0).abs() < 1e-6);
        assert!((calibration.position(112) - 1.0).abs() < 1e-6);
        assert!((calibration.position(60) - 0.495).abs() < 0.01);
    }

    #[test]
    fn a_pedal_wired_backwards_is_read_the_right_way_round() {
        let calibration = Calibration { heel: 127, toe: 0 };
        assert!(calibration.is_inverted());
        assert!((calibration.position(127) - 0.0).abs() < 1e-6);
        assert!((calibration.position(0) - 1.0).abs() < 1e-6);
        assert!((calibration.position(64) - 0.496).abs() < 0.01);
    }

    #[test]
    fn leaning_past_either_stop_does_not_run_past_the_ends() {
        // Pedals drift, and a reading outside the calibrated travel is a foot
        // on the stop rather than a request for 110%.
        let calibration = Calibration { heel: 20, toe: 100 };
        assert!((calibration.position(0) - 0.0).abs() < 1e-6);
        assert!((calibration.position(127) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_calibration_that_barely_moved_is_ignored_rather_than_amplified() {
        // Dividing an eight-value span into a full sweep would make the pedal
        // unusably twitchy, so an unusable reading falls back to full range.
        let calibration = Calibration { heel: 60, toe: 68 };
        assert!(!calibration.is_usable());
        assert!((calibration.position(64) - 64.0 / 127.0).abs() < 1e-6);
    }

    #[test]
    fn a_run_finds_the_pedal_and_its_travel() {
        let run = swept(11, 9, 112);
        assert_eq!(
            run.finish(),
            Ok((11, Calibration { heel: 9, toe: 112 }))
        );
    }

    #[test]
    fn the_controller_that_swept_wins_over_one_that_sat_still() {
        // A footswitch emits a bank select alongside its program change, and
        // it is the same value every time. The pedal is the thing that moved.
        let mut run = CalibrationRun::new();
        run.observe(control(0, 0));
        run.observe(control(11, 5));
        run.advance();
        run.observe(control(0, 0));
        run.observe(control(11, 120));
        let (controller, calibration) = run.finish().expect("a calibrated pedal");
        assert_eq!(controller, 11);
        assert_eq!(calibration, Calibration { heel: 5, toe: 120 });
    }

    #[test]
    fn the_reading_is_where_the_foot_came_to_rest() {
        // Sweeping to the heel passes through everything above it on the way,
        // so the extreme seen is not the answer — the last one is.
        let mut run = CalibrationRun::new();
        for value in [90, 60, 30, 4] {
            run.observe(control(11, value));
        }
        run.advance();
        for value in [40, 80, 118] {
            run.observe(control(11, value));
        }
        assert_eq!(run.finish(), Ok((11, Calibration { heel: 4, toe: 118 })));
    }

    #[test]
    fn a_pedal_that_did_not_move_is_reported_rather_than_saved() {
        let run = swept(11, 64, 68);
        assert!(run.finish().unwrap_err().contains("barely moved"));
    }

    #[test]
    fn a_socket_with_nothing_in_it_is_reported() {
        let run = CalibrationRun::new();
        assert!(run.finish().unwrap_err().contains("Nothing arrived"));
        // Reaching only one end is the same problem: half a sweep is no sweep.
        let mut half = CalibrationRun::new();
        half.observe(control(11, 3));
        half.advance();
        assert!(half.finish().is_err());
    }

    #[test]
    fn switches_pressed_mid_calibration_are_not_mistaken_for_the_pedal() {
        let mut run = CalibrationRun::new();
        assert_eq!(
            run.observe(Message::Program {
                channel: 0,
                program: 2
            }),
            None
        );
        assert!(!run.has_reading());
    }

    #[test]
    fn each_end_reports_what_it_read() {
        // A pedal that answers at one stop and not the other has to be
        // legible as exactly that while the run is still on screen.
        let mut run = CalibrationRun::new();
        run.observe(control(11, 4));
        assert_eq!(run.readings(Stage::Heel), [(11, 4)]);
        assert!(run.readings(Stage::Toe).is_empty());
        run.advance();
        assert!(run.readings(Stage::Toe).is_empty());
        run.observe(control(11, 118));
        assert_eq!(run.readings(Stage::Toe), [(11, 118)]);
    }

    #[test]
    fn the_two_steps_run_heel_then_toe_and_then_stop() {
        let mut run = CalibrationRun::new();
        assert_eq!(run.stage(), Some(Stage::Heel));
        assert_eq!(run.stage().map(Stage::label), Some("MIN"));
        run.advance();
        assert_eq!(run.stage(), Some(Stage::Toe));
        assert_eq!(run.stage().map(Stage::label), Some("MAX"));
        run.advance();
        assert_eq!(run.stage(), None);
        // Past the end nothing more is recorded.
        assert_eq!(run.observe(control(11, 40)), None);
    }

    #[test]
    fn a_calibration_survives_a_round_trip_through_the_preferences_file() {
        let calibration = Calibration { heel: 9, toe: 112 };
        let json = serde_json::to_string(&calibration).expect("serialise");
        assert_eq!(
            serde_json::from_str::<Calibration>(&json).expect("deserialise"),
            calibration
        );
    }
}
