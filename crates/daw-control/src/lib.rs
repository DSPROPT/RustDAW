//! Live MIDI from a control surface: a foot controller, a knob box, anything
//! sending switches and pedals rather than notes.
//!
//! This is a different job from [`daw_midi`](../daw_midi/index.html), which
//! reads and writes files. Nothing here is recorded or placed on a timeline.
//! A switch arrives, something on the screen changes, and that is the whole
//! of it — which is the point, because the hands it frees are holding a
//! guitar.
//!
//! What a pedal sends is not standardised in any useful way. The same four
//! switches are program changes on one box, control changes on another, and
//! notes on a third, and the manual is usually wrong. So nothing here is
//! hard-coded to a device: a [`Trigger`] is learned from whatever actually
//! arrives, and [`Bindings`] maps it to an [`Action`].
//!
//! [`Bindings::footswitch`] is the exception, and it is only a starting point
//! — the four-switch pedals that led to this crate all send program changes
//! `0` to `3`, so that guess is right often enough to be worth making.

mod bindings;
mod calibration;

pub use bindings::{AMP_SLOTS, Action, Binding, Bindings, Command, PRESET_SLOTS, Trigger};
pub use calibration::{Calibration, CalibrationRun, Stage};

use crossbeam_queue::ArrayQueue;
use midir::{Ignore, MidiInput, MidiInputConnection};
use std::sync::Arc;

/// The name the surface appears under in other applications' port lists.
const CLIENT_NAME: &str = "RustDAW";

/// Messages held between one repaint and the next.
///
/// An expression pedal sweeping end to end sends a few hundred; the interface
/// drains this thirty times a second, so the only way to reach the bottom of
/// it is a device that has gone haywire, which is exactly when dropping the
/// backlog is the right answer.
const QUEUE_CAPACITY: usize = 512;

/// The parts of a MIDI message a control surface uses.
///
/// Channel is carried but never matched on. A pedal that has been reconfigured
/// onto another channel is still the same pedal under the same foot, and
/// having every binding quietly stop working is not a useful way to report
/// that a setting moved.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Message {
    Program {
        channel: u8,
        program: u8,
    },
    Control {
        channel: u8,
        controller: u8,
        value: u8,
    },
    /// A switch wired as a note, which some controllers do. Note-off arrives
    /// as a velocity of zero, the way running status has always sent it.
    Note {
        channel: u8,
        note: u8,
        velocity: u8,
    },
    /// Something arrived that a control surface cannot act on.
    ///
    /// Kept rather than dropped, and shown in the monitor, because a pedal
    /// sending something unexpected is precisely the situation somebody opens
    /// the monitor to diagnose. Silently discarding it turns "your controller
    /// speaks a dialect I do not" into "your controller is not plugged in".
    Unhandled {
        status: u8,
        /// How many bytes arrived, capped so one long dump cannot be mistaken
        /// for a number.
        length: u8,
    },
}

impl Message {
    /// Reads one message off the wire.
    ///
    /// Anything this does not recognise comes back as [`Message::Unhandled`]
    /// rather than as nothing, so it can still be shown. Only an empty buffer
    /// yields `None`.
    #[must_use]
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        let (status, data) = bytes.split_first()?;
        let channel = status & 0x0F;
        match status & 0xF0 {
            0xC0 => Some(Self::Program {
                channel,
                program: *data.first()?,
            }),
            0xB0 => Some(Self::Control {
                channel,
                controller: *data.first()?,
                value: *data.get(1)?,
            }),
            0x90 => Some(Self::Note {
                channel,
                note: *data.first()?,
                velocity: *data.get(1)?,
            }),
            0x80 => Some(Self::Note {
                channel,
                note: *data.first()?,
                velocity: 0,
            }),
            _ => Some(Self::Unhandled {
                status: *status,
                length: u8::try_from(bytes.len()).unwrap_or(u8::MAX),
            }),
        }
    }

    /// The MIDI channel this arrived on, counting from one as every device's
    /// display does. Zero for a message that has no channel.
    #[must_use]
    pub const fn channel(self) -> u8 {
        match self {
            Self::Program { channel, .. }
            | Self::Control { channel, .. }
            | Self::Note { channel, .. } => channel + 1,
            Self::Unhandled { .. } => 0,
        }
    }

    /// How this reads in a monitor, so somebody can see what their pedal
    /// sends without leaving the application to find out.
    #[must_use]
    pub fn describe(self) -> String {
        match self {
            Self::Program { program, .. } => {
                format!("Program {program} · ch {}", self.channel())
            }
            Self::Control {
                controller, value, ..
            } => format!("CC {controller} = {value} · ch {}", self.channel()),
            Self::Note { note, velocity, .. } => {
                format!("Note {note} = {velocity} · ch {}", self.channel())
            }
            Self::Unhandled { status, length } => match status {
                0xF0 => format!("System exclusive · {length} bytes"),
                0xE0..=0xEF => format!("Pitch bend · ch {}", (status & 0x0F) + 1),
                0xA0..=0xAF => format!("Aftertouch · ch {}", (status & 0x0F) + 1),
                0xD0..=0xDF => format!("Channel pressure · ch {}", (status & 0x0F) + 1),
                other => format!("Status {other:#04X} · {length} bytes"),
            },
        }
    }
}

/// An open connection to one control surface.
///
/// Dropping it closes the port.
pub struct ControlSurface {
    /// Held to keep the port open; the callback it owns does the work.
    _connection: MidiInputConnection<Arc<ArrayQueue<Message>>>,
    port: String,
    messages: Arc<ArrayQueue<Message>>,
}

impl ControlSurface {
    /// Opens the first port whose name contains `port`.
    ///
    /// # Errors
    ///
    /// If MIDI is unavailable, no port matches, or the port is already held
    /// by something else.
    pub fn open(port: &str) -> Result<Self, String> {
        let mut input = MidiInput::new(CLIENT_NAME).map_err(|error| error.to_string())?;
        // Clock and active sensing arrive by the hundred and mean nothing to
        // a control surface, so they go. System exclusive stays: it is the
        // one place a controller can put something this parser does not
        // recognise, and a monitor that silently discards it cannot be used
        // to find out what a pedal is really sending — which is most of what
        // the monitor is for.
        input.ignore(Ignore::TimeAndActiveSense);
        let target = input
            .ports()
            .into_iter()
            .find(|candidate| {
                input
                    .port_name(candidate)
                    .is_ok_and(|name| name.contains(port))
            })
            .ok_or_else(|| format!("no MIDI input named {port}"))?;
        let name = input
            .port_name(&target)
            .map_err(|error| error.to_string())?;
        let messages = Arc::new(ArrayQueue::new(QUEUE_CAPACITY));
        let connection = input
            .connect(
                &target,
                CLIENT_NAME,
                |_when, bytes, queue| {
                    if let Some(message) = Message::parse(bytes) {
                        // Full means the interface has stopped draining, and
                        // the newest position of a pedal is worth more than
                        // the oldest one still in the queue.
                        if queue.push(message).is_err() {
                            let _ = queue.pop();
                            let _ = queue.push(message);
                        }
                    }
                },
                Arc::clone(&messages),
            )
            .map_err(|error| error.to_string())?;
        Ok(Self {
            _connection: connection,
            port: name,
            messages,
        })
    }

    /// The port this is listening to, by its full name.
    #[must_use]
    pub fn port(&self) -> &str {
        &self.port
    }

    /// Takes everything that has arrived since the last call, oldest first.
    pub fn drain(&self) -> impl Iterator<Item = Message> + '_ {
        std::iter::from_fn(|| self.messages.pop())
    }
}

/// Every MIDI input on the system, by name.
///
/// Empty when MIDI itself is unavailable, which on a machine without ALSA
/// sequencer support is not an error worth reporting: there is simply nothing
/// to connect to.
#[must_use]
pub fn input_ports() -> Vec<String> {
    let Ok(input) = MidiInput::new(CLIENT_NAME) else {
        return Vec::new();
    };
    input
        .ports()
        .iter()
        .filter_map(|port| input.port_name(port).ok())
        .filter(|name| !name.starts_with("Midi Through"))
        .collect()
}

/// The port most likely to be a foot controller.
///
/// Guessing is worth it because the alternative is a menu of ALSA port names,
/// which say nothing to somebody who just wants their pedal to work. A guess
/// that is wrong costs one trip to the menu; no guess costs everybody one.
#[must_use]
pub fn likely_foot_controller(ports: &[String]) -> Option<&String> {
    /// Fragments of the names these pedals appear under. `SINCO` and
    /// `FootCtrl` are what the M-Vave boxes report over USB and Bluetooth
    /// respectively, neither of which is the name on the pedal.
    const HINTS: [&str; 8] = [
        "footctrl", "sinco", "m-vave", "mvave", "chocolate", "foot", "pedal", "switch",
    ];
    ports.iter().find(|name| {
        let name = name.to_ascii_lowercase();
        HINTS.iter().any(|hint| name.contains(hint))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_program_change_is_read_off_the_wire() {
        // What the four switches of an M-Vave Chocolate actually send.
        assert_eq!(
            Message::parse(&[0xC0, 2]),
            Some(Message::Program {
                channel: 0,
                program: 2
            })
        );
    }

    #[test]
    fn a_control_change_carries_its_value() {
        assert_eq!(
            Message::parse(&[0xB3, 11, 64]),
            Some(Message::Control {
                channel: 3,
                controller: 11,
                value: 64
            })
        );
    }

    #[test]
    fn a_note_off_is_a_note_at_no_velocity() {
        let released = Message::parse(&[0x83, 60, 40]).expect("a note off");
        assert_eq!(
            released,
            Message::Note {
                channel: 3,
                note: 60,
                velocity: 0
            }
        );
        assert_eq!(Message::parse(&[0x93, 60, 0]), Some(released));
    }

    #[test]
    fn channels_are_numbered_the_way_the_hardware_shows_them() {
        // Every pedal's display counts from one; only the wire counts from zero.
        assert_eq!(Message::parse(&[0xC0, 0]).unwrap().channel(), 1);
        assert_eq!(Message::parse(&[0xCF, 0]).unwrap().channel(), 16);
    }

    #[test]
    fn what_a_surface_cannot_use_is_dropped() {
        // Nothing but an empty buffer or a truncated message vanishes. The
        // rest is reported, because a pedal speaking an unexpected dialect is
        // the thing somebody is trying to find out about.
        for bytes in [&[][..], &[0xC0][..], &[0xB0, 11][..]] {
            assert_eq!(Message::parse(bytes), None, "{bytes:?} was not dropped");
        }
    }

    #[test]
    fn what_the_surface_cannot_act_on_is_still_reported() {
        // Discarding these turns "your controller speaks a dialect I do not"
        // into "your controller is not plugged in", which is the harder
        // problem to diagnose by a wide margin.
        let sysex = Message::parse(&[0xF0, 0x7E, 0x01, 0xF7]).expect("sysex");
        assert_eq!(
            sysex,
            Message::Unhandled {
                status: 0xF0,
                length: 4
            }
        );
        assert_eq!(sysex.describe(), "System exclusive · 4 bytes");
        assert_eq!(
            Message::parse(&[0xE0, 0, 64]).unwrap().describe(),
            "Pitch bend · ch 1"
        );
        assert_eq!(
            Message::parse(&[0xA3, 60, 40]).unwrap().describe(),
            "Aftertouch · ch 4"
        );
    }

    #[test]
    fn a_message_reads_as_something_a_person_can_check_against_their_pedal() {
        assert_eq!(
            Message::parse(&[0xC0, 3]).unwrap().describe(),
            "Program 3 · ch 1"
        );
        assert_eq!(
            Message::parse(&[0xB0, 11, 127]).unwrap().describe(),
            "CC 11 = 127 · ch 1"
        );
    }

    #[test]
    fn a_foot_controller_is_picked_out_of_the_ports() {
        let ports = [
            "Midi Through:Midi Through Port-0 14:0".to_owned(),
            "SINCO:SINCO MIDI 1 20:0".to_owned(),
        ]
        .to_vec();
        assert_eq!(likely_foot_controller(&ports), Some(&ports[1]));
        // Over Bluetooth the same pedal arrives under a different name again.
        let bluetooth = ["FootCtrl:FootCtrl Bluetooth 128:0".to_owned()].to_vec();
        assert_eq!(likely_foot_controller(&bluetooth), Some(&bluetooth[0]));
        assert_eq!(likely_foot_controller(&ports[..1]), None);
    }
}
