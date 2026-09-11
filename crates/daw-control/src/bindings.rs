//! What a switch does when it is stepped on.

use serde::{Deserialize, Serialize};

use crate::Message;

/// How many amps a pedal can choose between.
///
/// Eight because that is more switches than any of these boxes has, so the
/// limit is never the thing in the way; four-switch pedals use the first four
/// and the rest sit unbound.
pub const AMP_SLOTS: usize = 8;

/// How many whole rigs a pedal can choose between.
///
/// The same reasoning as [`AMP_SLOTS`], and the same number: a preset switch
/// and an amp switch compete for the same feet, and there is no case for
/// offering fewer of one than the other.
pub const PRESET_SLOTS: usize = 8;

/// The value at which a switch counts as pressed.
///
/// Switches that send control changes send `127` down and `0` up. Acting on
/// the halfway mark means the release does not fire the action a second time.
const PRESS_THRESHOLD: u8 = 64;

/// The full travel of a MIDI controller.
const FULL_SCALE: f32 = 127.0;

/// What arrived, stripped of everything a binding does not match on.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub enum Trigger {
    Program(u8),
    Control(u8),
    Note(u8),
    /// Something no switch can be bound to, keyed by its status byte so the
    /// monitor can still group repeats of it.
    Other(u8),
}

impl Trigger {
    /// What this message would be bound as. The value is dropped: a switch is
    /// the same switch whether it is going down or coming back up.
    #[must_use]
    pub const fn of(message: Message) -> Self {
        match message {
            Message::Program { program, .. } => Self::Program(program),
            Message::Control { controller, .. } => Self::Control(controller),
            Message::Note { note, .. } => Self::Note(note),
            Message::Unhandled { status, .. } => Self::Other(status),
        }
    }

    /// How this reads next to an action, in the same terms the pedal's own
    /// editor uses.
    #[must_use]
    pub fn describe(self) -> String {
        match self {
            Self::Program(program) => format!("PC {program}"),
            Self::Control(controller) => format!("CC {controller}"),
            Self::Note(note) => format!("Note {note}"),
            Self::Other(status) => format!("Status {status:#04X}"),
        }
    }
}

/// Something a foot can do to the session while both hands are busy.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum Action {
    /// Load the capture in one of the amp slots, and switch the amp on.
    SelectAmp(u8),
    NextAmp,
    PreviousAmp,
    ToggleAmp,
    /// Recall the whole rig in one of the preset slots: the capture, the wah,
    /// the tone stack, the dynamics and the time effects, all at once.
    SelectPreset(u8),
    NextPreset,
    PreviousPreset,
    ToggleWah,
    /// The expression pedal itself, which sweeps the wah rather than
    /// switching anything.
    WahPedal,
    ToggleDelay,
    ToggleReverb,
    PlayPause,
    Record,
    Stop,
}

impl Action {
    /// Everything a switch can be bound to, in the order it is worth offering.
    #[must_use]
    pub fn all() -> Vec<Self> {
        let mut actions: Vec<Self> = (0..AMP_SLOTS)
            .map(|slot| Self::SelectAmp(u8::try_from(slot).unwrap_or(0)))
            .collect();
        actions.extend(
            (0..PRESET_SLOTS).map(|slot| Self::SelectPreset(u8::try_from(slot).unwrap_or(0))),
        );
        actions.extend([
            Self::NextAmp,
            Self::PreviousAmp,
            Self::ToggleAmp,
            Self::NextPreset,
            Self::PreviousPreset,
            Self::ToggleWah,
            Self::WahPedal,
            Self::ToggleDelay,
            Self::ToggleReverb,
            Self::PlayPause,
            Self::Record,
            Self::Stop,
        ]);
        actions
    }

    /// A pedal that sweeps rather than switches. It acts on every value it
    /// sends, where a switch acts only on the way down.
    #[must_use]
    pub const fn is_continuous(self) -> bool {
        matches!(self, Self::WahPedal)
    }

    #[must_use]
    pub fn label(self) -> String {
        match self {
            Self::SelectAmp(slot) => format!("Amp {}", slot.saturating_add(1)),
            Self::NextAmp => "Next amp".to_owned(),
            Self::PreviousAmp => "Previous amp".to_owned(),
            Self::ToggleAmp => "Amp on/off".to_owned(),
            Self::SelectPreset(slot) => format!("Preset {}", slot.saturating_add(1)),
            Self::NextPreset => "Next preset".to_owned(),
            Self::PreviousPreset => "Previous preset".to_owned(),
            Self::ToggleWah => "Wah on/off".to_owned(),
            Self::WahPedal => "Wah pedal".to_owned(),
            Self::ToggleDelay => "Delay on/off".to_owned(),
            Self::ToggleReverb => "Reverb on/off".to_owned(),
            Self::PlayPause => "Play / pause".to_owned(),
            Self::Record => "Record".to_owned(),
            Self::Stop => "Stop".to_owned(),
        }
    }
}

/// One switch, and what it does.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Binding {
    pub trigger: Trigger,
    pub action: Action,
}

/// An action to carry out, with the pedal's position when it has one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Command {
    /// A switch went down.
    Press(Action),
    /// A pedal moved, `0` heel-down to `1` toe-down.
    Move(Action, f32),
}

/// Every switch on the surface, and what each one does.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct Bindings(Vec<Binding>);

impl Bindings {
    /// A guess at a four-switch pedal with an expression input.
    ///
    /// Every one of these boxes that prompted this crate leaves the factory
    /// sending program changes `0` to `3`, and CC 11 is what the MIDI
    /// specification has called expression since 1991, so an expression pedal
    /// is more likely to send it than anything else. Both are guesses, and
    /// both are meant to be re-learned when they are wrong — which is why the
    /// monitor showing what actually arrives sits next to the list.
    #[must_use]
    pub fn footswitch() -> Self {
        let mut bindings = Self::default();
        for slot in 0..4 {
            bindings.bind(Trigger::Program(slot), Action::SelectAmp(slot));
        }
        bindings.bind(Trigger::Control(11), Action::WahPedal);
        bindings
    }

    #[must_use]
    pub fn entries(&self) -> &[Binding] {
        &self.0
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Which switch does `action`, if any.
    #[must_use]
    pub fn trigger_for(&self, action: Action) -> Option<Trigger> {
        self.0
            .iter()
            .find(|binding| binding.action == action)
            .map(|binding| binding.trigger)
    }

    /// Points `trigger` at `action`, displacing whatever either of them was
    /// doing before.
    ///
    /// Both directions have to be unique or the list stops being readable: a
    /// switch that does two things at once is a bug somebody will spend an
    /// evening on, and an action reachable from two switches is a switch
    /// somebody thinks is broken.
    pub fn bind(&mut self, trigger: Trigger, action: Action) {
        self.0
            .retain(|binding| binding.trigger != trigger && binding.action != action);
        self.0.push(Binding { trigger, action });
    }

    /// Unbinds whatever switch does `action`.
    pub fn clear(&mut self, action: Action) {
        self.0.retain(|binding| binding.action != action);
    }

    /// What to do about a message, if anything.
    #[must_use]
    pub fn resolve(&self, message: Message) -> Option<Command> {
        let trigger = Trigger::of(message);
        let action = self
            .0
            .iter()
            .find(|binding| binding.trigger == trigger)
            .map(|binding| binding.action)?;
        let value = match message {
            // A program change has no value; it is the press.
            Message::Program { .. } => return Some(Command::Press(action)),
            Message::Control { value, .. }
            | Message::Note {
                velocity: value, ..
            } => value,
            // Nothing to act on, however it came to be bound.
            Message::Unhandled { .. } => return None,
        };
        if action.is_continuous() {
            return Some(Command::Move(action, f32::from(value) / FULL_SCALE));
        }
        (value >= PRESS_THRESHOLD).then_some(Command::Press(action))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn program(program: u8) -> Message {
        Message::Program {
            channel: 0,
            program,
        }
    }

    fn control(controller: u8, value: u8) -> Message {
        Message::Control {
            channel: 0,
            controller,
            value,
        }
    }

    #[test]
    fn the_four_switches_of_a_stock_pedal_pick_the_four_amps() {
        let bindings = Bindings::footswitch();
        for slot in 0..4 {
            assert_eq!(
                bindings.resolve(program(slot)),
                Some(Command::Press(Action::SelectAmp(slot)))
            );
        }
        // A fifth switch nobody has bound does nothing at all.
        assert_eq!(bindings.resolve(program(4)), None);
    }

    #[test]
    fn an_expression_pedal_sweeps_rather_than_switching() {
        let bindings = Bindings::footswitch();
        assert_eq!(
            bindings.resolve(control(11, 0)),
            Some(Command::Move(Action::WahPedal, 0.0))
        );
        assert_eq!(
            bindings.resolve(control(11, 127)),
            Some(Command::Move(Action::WahPedal, 1.0))
        );
        // Every value on the way through, not only the ends: a wah that only
        // knew heel and toe would not be a wah.
        let Some(Command::Move(_, middle)) = bindings.resolve(control(11, 64)) else {
            panic!("the pedal stopped halfway");
        };
        assert!((middle - 0.504).abs() < 0.01, "{middle}");
    }

    #[test]
    fn a_switch_acts_on_the_way_down_and_not_on_the_way_up() {
        let mut bindings = Bindings::default();
        bindings.bind(Trigger::Control(64), Action::ToggleWah);
        assert_eq!(
            bindings.resolve(control(64, 127)),
            Some(Command::Press(Action::ToggleWah))
        );
        // The release must not toggle it straight back off.
        assert_eq!(bindings.resolve(control(64, 0)), None);
    }

    #[test]
    fn a_switch_wired_as_a_note_works_like_any_other() {
        let mut bindings = Bindings::default();
        bindings.bind(Trigger::Note(60), Action::Record);
        let pressed = Message::Note {
            channel: 0,
            note: 60,
            velocity: 100,
        };
        let released = Message::Note {
            channel: 0,
            note: 60,
            velocity: 0,
        };
        assert_eq!(
            bindings.resolve(pressed),
            Some(Command::Press(Action::Record))
        );
        assert_eq!(bindings.resolve(released), None);
    }

    #[test]
    fn the_channel_a_pedal_is_set_to_does_not_matter() {
        // Somebody moving their pedal to channel 5 has not asked for every
        // binding to stop working.
        let bindings = Bindings::footswitch();
        let elsewhere = Message::Program {
            channel: 4,
            program: 1,
        };
        assert_eq!(
            bindings.resolve(elsewhere),
            Some(Command::Press(Action::SelectAmp(1)))
        );
    }

    #[test]
    fn learning_a_switch_displaces_what_it_did_before() {
        let mut bindings = Bindings::footswitch();
        bindings.bind(Trigger::Program(0), Action::PlayPause);
        assert_eq!(
            bindings.resolve(program(0)),
            Some(Command::Press(Action::PlayPause))
        );
        assert_eq!(bindings.trigger_for(Action::SelectAmp(0)), None);
    }

    #[test]
    fn an_action_is_only_ever_on_one_switch() {
        // Otherwise the second switch looks broken and the first looks haunted.
        let mut bindings = Bindings::default();
        bindings.bind(Trigger::Program(0), Action::ToggleAmp);
        bindings.bind(Trigger::Program(1), Action::ToggleAmp);
        assert_eq!(bindings.entries().len(), 1);
        assert_eq!(bindings.resolve(program(0)), None);
        assert_eq!(
            bindings.resolve(program(1)),
            Some(Command::Press(Action::ToggleAmp))
        );
    }

    #[test]
    fn unbinding_leaves_the_switch_doing_nothing() {
        let mut bindings = Bindings::footswitch();
        bindings.clear(Action::SelectAmp(2));
        assert_eq!(bindings.resolve(program(2)), None);
        assert_eq!(bindings.trigger_for(Action::SelectAmp(2)), None);
    }

    #[test]
    fn every_action_can_be_named_and_offered() {
        let actions = Action::all();
        assert_eq!(actions.len(), AMP_SLOTS + PRESET_SLOTS + 12);
        assert!(actions.iter().all(|action| !action.label().is_empty()));
        assert_eq!(
            actions
                .iter()
                .filter(|action| action.is_continuous())
                .count(),
            1,
            "only the expression pedal sweeps"
        );
    }

    #[test]
    fn bindings_survive_a_round_trip_through_the_preferences_file() {
        let bindings = Bindings::footswitch();
        let json = serde_json::to_string(&bindings).expect("serialise");
        let restored: Bindings = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(restored, bindings);
    }
}
