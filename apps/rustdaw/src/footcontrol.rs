//! The foot controller window: what the pedal is sending, and what it does.
//!
//! A guitarist's hands are on the guitar. Everything in here exists so that
//! the things you need mid-take — a different amp, the wah, the record button
//! — can be reached without putting it down.
//!
//! The window is deliberately half monitor. No two of these pedals send the
//! same thing, their manuals disagree with them, and the fastest way to bind a
//! switch correctly is to watch what arrives when you step on it. So the
//! stream of incoming messages sits directly above the list of bindings, and
//! LEARN takes whatever comes next.

use std::collections::VecDeque;
use std::path::PathBuf;

use daw_control::{
    AMP_SLOTS, Action, Bindings, Calibration, CalibrationRun, Command, ControlSurface, Message,
    PRESET_SLOTS, Stage, Trigger,
};
use daw_project::PresetLibrary;
use eframe::egui::{self, Align2, Color32, FontId, RichText, Stroke};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::theme;

/// How many messages the monitor shows. Enough to see a switch's press and
/// release, and to tell a pedal sweeping from a pedal that has stopped.
const MONITOR_DEPTH: usize = 8;

/// What is remembered about the pedal between sessions.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct FootPreferences {
    #[serde(default)]
    pub port: Option<String>,
    /// Falling back to the stock guess rather than to nothing, so a file
    /// written before a binding existed, or edited by hand, still drives the
    /// pedal. A list somebody deliberately emptied is honoured as it stands.
    #[serde(default = "Bindings::footswitch")]
    pub bindings: Bindings,
    /// The capture each amp switch loads. Shorter than the number of slots on
    /// an older file, which is why it is a list rather than a fixed array.
    #[serde(default)]
    pub slots: Vec<Option<PathBuf>>,
    /// The rig each preset switch recalls, by id rather than by name so
    /// renaming a preset does not silently unbind the switch it is on.
    #[serde(default)]
    pub preset_slots: Vec<Option<Uuid>>,
    /// How far the expression pedal actually travels.
    #[serde(default)]
    pub calibration: Calibration,
}

pub struct FootControlState {
    pub open: bool,
    /// The port to reopen at startup.
    port: Option<String>,
    bindings: Bindings,
    slots: Vec<Option<PathBuf>>,
    preset_slots: Vec<Option<Uuid>>,
    /// The open connection. Dropping it releases the port.
    surface: Option<ControlSurface>,
    /// Why the last attempt to connect failed, shown until the next one.
    error: Option<String>,
    /// MIDI inputs as of the last scan. Enumerating them opens the MIDI
    /// client, so it happens on request rather than every repaint.
    ports: Vec<String>,
    /// The newest messages, oldest first: the switch each came from, how it
    /// reads, and how many arrived in a row.
    ///
    /// A sweeping pedal replaces its own line rather than flooding the
    /// monitor, but the count has to be kept and shown. Collapsing a hundred
    /// identical messages into one line that looks like one message hides the
    /// most diagnostic thing there is — a control that is transmitting
    /// constantly and never changing.
    seen: VecDeque<(Trigger, String, u32)>,
    /// The action waiting to be taught a switch.
    learning: Option<Action>,
    /// Where the expression pedal last was, calibrated, which is what the wah
    /// is given.
    pedal: f32,
    /// The last control change to arrive, and what it was on.
    ///
    /// Tracked whatever it is bound to, and whether it is bound at all.
    /// Somebody working out whether the thing under their foot is even
    /// plugged in has to be able to see it move before there is anything to
    /// bind it to — which is the whole reason the pedal is drawn.
    last_control: Option<(u8, u8)>,
    calibration: Calibration,
    /// A calibration in progress. While one is running nothing else acts on
    /// the pedal: a sweep is being measured, not played.
    run: Option<CalibrationRun>,
    /// Why the last calibration did not take.
    calibration_error: Option<String>,
    /// The amp slot last selected, so the panel can light it up.
    active: Option<u8>,
    /// The preset slot last recalled, lit for the same reason.
    active_preset: Option<u8>,
    /// Set when something worth writing to disk changed.
    pub dirty: bool,
}

impl Default for FootControlState {
    fn default() -> Self {
        Self {
            open: false,
            port: None,
            bindings: Bindings::footswitch(),
            slots: vec![None; AMP_SLOTS],
            preset_slots: vec![None; PRESET_SLOTS],
            surface: None,
            error: None,
            ports: Vec::new(),
            seen: VecDeque::new(),
            learning: None,
            pedal: 0.0,
            last_control: None,
            calibration: Calibration::default(),
            run: None,
            calibration_error: None,
            active: None,
            active_preset: None,
            dirty: false,
        }
    }
}

impl FootControlState {
    #[must_use]
    pub fn from_preferences(preferences: FootPreferences) -> Self {
        let mut state = Self {
            port: preferences.port,
            bindings: preferences.bindings,
            calibration: preferences.calibration,
            ..Self::default()
        };
        for (slot, saved) in state.slots.iter_mut().zip(preferences.slots) {
            *slot = saved;
        }
        for (slot, saved) in state.preset_slots.iter_mut().zip(preferences.preset_slots) {
            *slot = saved;
        }
        state
    }

    #[must_use]
    pub fn preferences(&self) -> FootPreferences {
        FootPreferences {
            port: self.port.clone(),
            bindings: self.bindings.clone(),
            slots: self.slots.clone(),
            preset_slots: self.preset_slots.clone(),
            calibration: self.calibration,
        }
    }

    /// Fills empty amp switches from the captures on disk.
    ///
    /// A pedal that does nothing until eight menus have been filled in is a
    /// pedal nobody gets working. Four captures on four switches is wrong as
    /// often as not, but it is wrong in a way that can be heard and fixed,
    /// rather than silent.
    pub fn seed(&mut self, library: &[daw_nam::AmpModel]) {
        for (slot, model) in self.slots.iter_mut().zip(library) {
            if slot.is_none() {
                *slot = Some(model.path.clone());
            }
        }
    }

    /// The capture an amp switch loads.
    #[must_use]
    pub fn slot(&self, slot: u8) -> Option<&PathBuf> {
        self.slots.get(usize::from(slot)).and_then(Option::as_ref)
    }

    /// Remembers which switch is lit, once the amp has actually loaded.
    pub fn set_active(&mut self, slot: Option<u8>) {
        self.active = slot;
    }

    /// The rig a preset switch recalls.
    #[must_use]
    pub fn preset_slot(&self, slot: u8) -> Option<Uuid> {
        self.preset_slots.get(usize::from(slot)).copied().flatten()
    }

    /// Puts a rig on a preset switch, or takes it off with `None`.
    pub fn set_preset_slot(&mut self, slot: u8, preset: Option<Uuid>) {
        if let Some(entry) = self.preset_slots.get_mut(usize::from(slot)) {
            *entry = preset;
            self.dirty = true;
        }
    }

    /// Remembers which preset switch is lit.
    pub fn set_active_preset(&mut self, slot: Option<u8>) {
        self.active_preset = slot;
    }

    /// Fills empty preset switches from the library, in its own order.
    ///
    /// Same reasoning as [`Self::seed`]: eight menus to fill in before a
    /// pedal does anything is eight reasons not to bother. A rig on the wrong
    /// switch is heard and moved in seconds.
    pub fn seed_presets(&mut self, presets: &PresetLibrary) {
        let unassigned: Vec<Uuid> = presets
            .presets()
            .iter()
            .map(|preset| preset.id)
            .filter(|id| !self.preset_slots.contains(&Some(*id)))
            .collect();
        let empty = self.preset_slots.iter_mut().filter(|slot| slot.is_none());
        for (slot, id) in empty.zip(unassigned) {
            *slot = Some(id);
            self.dirty = true;
        }
    }

    /// Takes a deleted rig off every switch it was on.
    pub fn forget_preset(&mut self, preset: Uuid) {
        for slot in &mut self.preset_slots {
            if *slot == Some(preset) {
                *slot = None;
                self.dirty = true;
            }
        }
    }

    #[must_use]
    pub fn connected(&self) -> bool {
        self.surface.is_some()
    }

    /// Opens `port`, replacing any connection already open.
    pub fn connect(&mut self, port: &str) {
        // Dropped first: the same port cannot be opened twice, so reconnecting
        // without this fails against nothing but ourselves.
        self.surface = None;
        match ControlSurface::open(port) {
            Ok(surface) => {
                self.port = Some(port.to_owned());
                self.surface = Some(surface);
                self.error = None;
                self.dirty = true;
            }
            Err(error) => self.error = Some(error),
        }
    }

    pub fn disconnect(&mut self) {
        self.surface = None;
        self.learning = None;
    }

    /// Rescans the MIDI inputs.
    pub fn refresh_ports(&mut self) {
        self.ports = daw_control::input_ports();
    }

    /// Opens the remembered port, or whatever looks most like a foot
    /// controller. Silent when there is nothing to connect to: a session
    /// started without the pedal plugged in is the ordinary case.
    pub fn autoconnect(&mut self) {
        self.refresh_ports();
        let target = self
            .port
            .clone()
            .filter(|remembered| {
                self.ports
                    .iter()
                    .any(|port| port.contains(remembered.as_str()))
            })
            .or_else(|| daw_control::likely_foot_controller(&self.ports).cloned());
        if let Some(port) = target {
            self.connect(&port);
        }
    }

    /// Everything the pedal has asked for since the last repaint.
    ///
    /// A message arriving while a switch is being learned teaches it instead
    /// of acting, so the switch being bound cannot fire the action it is about
    /// to be bound to.
    pub fn poll(&mut self) -> Vec<Command> {
        let Some(surface) = &self.surface else {
            return Vec::new();
        };
        let arrived: Vec<Message> = surface.drain().collect();
        self.handle(arrived)
    }

    /// Works through a batch of messages, whatever brought them in.
    ///
    /// Split from [`Self::poll`] so the rules — a calibration run swallowing
    /// everything, a switch being learned, a movement being calibrated — can
    /// be exercised without a pedal plugged in.
    fn handle(&mut self, arrived: Vec<Message>) -> Vec<Command> {
        let mut commands = Vec::with_capacity(arrived.len());
        for message in arrived {
            // Captured before the message is folded in, so a repeat can be
            // told from a movement.
            let previous = self.last_control;
            self.remember(message);
            // A calibration run takes every message it can use. Nothing acts
            // on the pedal while it is being measured: the sweep being read
            // is not a sweep anybody is playing.
            if let Some(run) = &mut self.run {
                run.observe(message);
                continue;
            }
            if let Some(action) = self.learning {
                if operated(message, previous) {
                    self.learning = None;
                    self.bindings.bind(Trigger::of(message), action);
                    self.dirty = true;
                }
                continue;
            }
            if let Some(command) = self.bindings.resolve(message) {
                commands.push(self.calibrated(command, message));
            }
        }
        commands
    }

    /// Re-reads a pedal movement against the calibration.
    ///
    /// The bindings only know the full `0` to `127`, because that is all a
    /// message carries. What that range means for this particular pedal is
    /// known here.
    fn calibrated(&mut self, command: Command, message: Message) -> Command {
        let (Command::Move(action, _), Message::Control { value, .. }) = (command, message) else {
            return command;
        };
        let position = self.calibration.position(value);
        if action == Action::WahPedal {
            self.pedal = position;
        }
        Command::Move(action, position)
    }

    /// Where to draw the treadle.
    ///
    /// Whatever control moved last, calibrated when it is the one the wah is
    /// on and raw when it is not — an unbound pedal still has to be seen to
    /// move, or there is no way to tell a dead socket from an unbound one.
    /// Mid-run it is always raw: the calibration being replaced is not
    /// something to measure the replacement against.
    fn displayed_position(&self) -> f32 {
        let Some((controller, value)) = self.last_control else {
            return 0.0;
        };
        if self.run.is_none() && self.pedal_controller() == Some(controller) {
            return self.calibration.position(value);
        }
        f32::from(value) / 127.0
    }

    /// Which controller the expression pedal is on, once it is known.
    fn pedal_controller(&self) -> Option<u8> {
        match self.bindings.trigger_for(Action::WahPedal) {
            Some(Trigger::Control(controller)) => Some(controller),
            _ => None,
        }
    }

    /// Puts every switch back where it started.
    pub fn reset_bindings(&mut self) {
        self.bindings = Bindings::footswitch();
        self.learning = None;
        self.dirty = true;
    }

    /// Starts reading the two ends of the pedal's travel.
    pub fn calibrate(&mut self) {
        self.calibration_error = None;
        self.last_control = None;
        self.learning = None;
        self.run = Some(CalibrationRun::new());
    }

    fn remember(&mut self, message: Message) {
        if let Message::Control {
            controller, value, ..
        } = message
        {
            // Every control change, on any controller, bound or not. Nothing
            // is filtered: a pedal whose socket is wired to a controller
            // nobody expects is exactly the case this drawing exists to make
            // visible, and a rule clever enough to hide bank select is clever
            // enough to hide that.
            self.last_control = Some((controller, value));
        }
        let trigger = Trigger::of(message);
        let line = message.describe();
        // A pedal mid-sweep sends a hundred of these a second. Replacing its
        // own line keeps the switch presses either side of it readable.
        if let Some(last) = self.seen.back_mut() {
            if last.0 == trigger {
                last.1 = line;
                last.2 = last.2.saturating_add(1);
                return;
            }
        }
        self.seen.push_back((trigger, line, 1));
        while self.seen.len() > MONITOR_DEPTH {
            self.seen.pop_front();
        }
    }
}

/// Whether a message is somebody operating a control, rather than a control
/// that transmits whether or not anybody touches it.
///
/// Some controllers stream a control change continuously — the pedal this was
/// written against emits a bank select hundreds of times without being asked.
/// With LEARN taking whatever arrives next, that noise wins every race, and
/// every switch anybody tries to teach lands on it instead. So a control has
/// to have actually moved since the last one to count as operated. A program
/// change or a note needs no such test: those are events, and they only ever
/// arrive because a foot made them.
fn operated(message: Message, previous: Option<(u8, u8)>) -> bool {
    if matches!(message, Message::Unhandled { .. }) {
        // Visible in the monitor, but there is nothing here a switch could be
        // bound to and acted on later.
        return false;
    }
    let Message::Control {
        controller, value, ..
    } = message
    else {
        return true;
    };
    // With nothing to compare against there is no way to tell a control being
    // moved from one that was already sitting there talking, so the first
    // reading only establishes what unchanged looks like. The movement is the
    // one after it, and anybody sweeping a pedal sends dozens.
    previous.is_some_and(|last| last != (controller, value))
}

/// Draws the window. Everything it changes is its own state; what the pedal
/// asked for is applied by the caller, which is the only thing that knows
/// which track the foot is aimed at.
pub fn window(
    context: &egui::Context,
    state: &mut FootControlState,
    library: &[daw_nam::AmpModel],
    presets: &PresetLibrary,
    target: Option<&str>,
) {
    let mut open = state.open;
    egui::Window::new("FOOT CONTROLLER")
        .open(&mut open)
        .default_width(560.0)
        .default_height(620.0)
        .resizable(true)
        .show(context, |ui| {
            connection(ui, state);
            // Which track the switches and the pedal land on. Everything here
            // acts on one track, and with a session open there is no way to
            // tell which from the window itself — so a pedal that appears to
            // do nothing is often a pedal doing it somewhere else.
            ui.label(
                RichText::new(target.map_or_else(
                    || "No track to act on".to_owned(),
                    |name| format!("Acting on: {name}"),
                ))
                .small()
                .color(if target.is_some() {
                    theme::TEXT
                } else {
                    theme::MUTED
                }),
            );
            ui.separator();
            monitor(ui, state);
            ui.separator();
            expression_pedal(ui, state);
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| {
                // Rigs before amps: a preset switch changes the capture too,
                // and more besides, so it is the one most feet want.
                preset_switches(ui, state, presets);
                ui.add_space(10.0);
                amp_switches(ui, state, library);
                ui.add_space(10.0);
                other_switches(ui, state);
            });
        });
    state.open = open;
}

fn connection(ui: &mut egui::Ui, state: &mut FootControlState) {
    ui.horizontal(|ui| {
        let (colour, label) = if state.connected() {
            (theme::GREEN, "CONNECTED")
        } else {
            (theme::MUTED, "NOT CONNECTED")
        };
        ui.label(RichText::new("●").color(colour).size(16.0));
        ui.label(RichText::new(label).monospace().small().color(colour));
        if let Some(port) = &state.port {
            ui.label(RichText::new(port).small().color(theme::MUTED));
        }
    });
    ui.horizontal(|ui| {
        let mut chosen = state.port.clone();
        let selected = chosen.clone().unwrap_or_else(|| "Select a port...".into());
        egui::ComboBox::from_id_salt("foot_port")
            .selected_text(RichText::new(selected).small())
            .width(320.0)
            .show_ui(ui, |ui| {
                for port in &state.ports {
                    if ui
                        .selectable_label(chosen.as_ref() == Some(port), port)
                        .clicked()
                    {
                        chosen = Some(port.clone());
                    }
                }
                if state.ports.is_empty() {
                    ui.label(
                        RichText::new("No MIDI inputs found")
                            .small()
                            .color(theme::MUTED),
                    );
                }
            });
        if chosen != state.port {
            if let Some(port) = chosen {
                state.connect(&port);
            }
        }
        if ui
            .button("RESCAN")
            .on_hover_text(
                "Look for the pedal again. A pedal paired over Bluetooth appears under a \
                 different name from the same pedal over USB, and only once it has connected.",
            )
            .clicked()
        {
            state.refresh_ports();
        }
        if state.connected() {
            if ui.button("DISCONNECT").clicked() {
                state.disconnect();
            }
        } else if ui.button("CONNECT").clicked() {
            state.autoconnect();
        }
        if ui
            .button("DEFAULTS")
            .on_hover_text(
                "Put every switch back to the stock layout: the four switches on Amps 1 to 4, \
                 and the expression pedal on the wah. Learning is easy to do by accident with \
                 a pedal that talks this much.",
            )
            .clicked()
        {
            state.reset_bindings();
        }
    });
    if let Some(error) = &state.error {
        ui.label(RichText::new(error).small().color(theme::RED));
    }
}

fn monitor(ui: &mut egui::Ui, state: &FootControlState) {
    ui.label(
        RichText::new("WHAT THE PEDAL IS SENDING")
            .small()
            .monospace()
            .color(theme::MUTED),
    );
    egui::Frame::new()
        .fill(theme::BG)
        .stroke(Stroke::new(1.0_f32, theme::BORDER))
        .inner_margin(8.0)
        .show(ui, |ui| {
            ui.set_min_size(egui::vec2(ui.available_width(), 96.0));
            if state.seen.is_empty() {
                ui.label(
                    RichText::new(if state.connected() {
                        "Step on a switch."
                    } else {
                        "Connect the pedal to see what it sends."
                    })
                    .small()
                    .color(theme::MUTED),
                );
            }
            for (index, (_, line, repeats)) in state.seen.iter().enumerate() {
                // The newest line is the one being read; the rest are context.
                let newest = index + 1 == state.seen.len();
                let counted = if *repeats > 1 {
                    format!("{line}  ×{repeats}")
                } else {
                    line.clone()
                };
                ui.label(RichText::new(counted).monospace().small().color(if newest {
                    theme::GREEN
                } else {
                    theme::MUTED
                }));
            }
        });
}

/// The expression pedal: what it is doing now, and how to teach it its own
/// travel.
fn expression_pedal(ui: &mut egui::Ui, state: &mut FootControlState) {
    ui.label(
        RichText::new("EXPRESSION PEDAL")
            .small()
            .monospace()
            .color(theme::MUTED),
    );
    let mut advance = false;
    let mut cancel = false;
    let mut start = false;
    ui.horizontal(|ui| {
        pedal_view(ui, state.displayed_position(), state.last_control.is_some());
        ui.vertical(|ui| {
            readings(ui, state);
            ui.add_space(4.0);
            match state.run.as_ref().and_then(CalibrationRun::stage) {
                None => {
                    if ui
                        .button("CALIBRATE")
                        .on_hover_text(
                            "Read the two ends of the pedal's travel. Almost no pedal sends \
                             the full range, so without this the wah never quite shuts and \
                             never quite opens.",
                        )
                        .clicked()
                    {
                        start = true;
                    }
                }
                Some(end) => {
                    let step = if end == Stage::Heel { 1 } else { 2 };
                    ui.label(
                        RichText::new(format!("STEP {step} OF 2 · {}", end.label()))
                            .monospace()
                            .small()
                            .color(theme::YELLOW),
                    );
                    ui.label(RichText::new(end.instruction()).small());
                    if let Some(run) = state.run.as_ref() {
                        captured(ui, run);
                    }
                    // The one thing a person watching the treadle move while
                    // the wah sits still needs told. It is deliberate — a
                    // sweep being measured is not a sweep being played — but
                    // deliberate and unexplained is indistinguishable from
                    // broken.
                    ui.label(
                        RichText::new("The wah is not driven until this finishes.")
                            .small()
                            .color(theme::MUTED),
                    );
                    ui.horizontal(|ui| {
                        let ready = state.run.as_ref().is_some_and(CalibrationRun::has_reading);
                        let label = if end == Stage::Heel { "NEXT" } else { "FINISH" };
                        if ui
                            .add_enabled(ready, egui::Button::new(label))
                            .on_hover_text(if ready {
                                "Read this end and carry on"
                            } else {
                                "Move the pedal first, so there is something to read"
                            })
                            .clicked()
                        {
                            advance = true;
                        }
                        if ui.button("CANCEL").clicked() {
                            cancel = true;
                        }
                    });
                }
            }
        });
    });
    if let Some(error) = &state.calibration_error {
        ui.label(RichText::new(error).small().color(theme::RED));
    }
    if start {
        state.calibrate();
    }
    if cancel {
        state.run = None;
    }
    if advance {
        finish_step(state);
    }
}

/// Takes one end of the travel, and saves the pedal once both are in.
fn finish_step(state: &mut FootControlState) {
    let Some(run) = &mut state.run else {
        return;
    };
    run.advance();
    if run.stage().is_some() {
        // Still one end to go; the next one is read the same way.
        return;
    }
    match run.finish() {
        Ok((controller, calibration)) => {
            state.calibration = calibration;
            // A run says which controller swept, so calibrating also finds a
            // pedal nobody has bound yet. Nobody knows their pedal's CC
            // number, and now nobody has to.
            state
                .bindings
                .bind(Trigger::Control(controller), Action::WahPedal);
            state.calibration_error = None;
            state.dirty = true;
        }
        Err(error) => state.calibration_error = Some(error),
    }
    state.run = None;
}

/// What each end of the travel has read so far.
///
/// Both ends stay on screen through the whole run. A pedal that answers at one
/// stop and says nothing at the other is a common and specific fault — a cable
/// with no ring conductor, most often — and it is unreadable unless the two
/// steps can be compared while they are being taken.
fn captured(ui: &mut egui::Ui, run: &CalibrationRun) {
    for end in [Stage::Heel, Stage::Toe] {
        let readings = run.readings(end);
        let text = if readings.is_empty() {
            "nothing yet".to_owned()
        } else {
            readings
                .iter()
                .map(|(controller, value)| format!("CC {controller} = {value}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        ui.label(
            RichText::new(format!("{:<4}{text}", end.label()))
                .monospace()
                .small()
                .color(if readings.is_empty() {
                    theme::MUTED
                } else {
                    theme::GREEN
                }),
        );
    }
}

/// What the pedal is on, what it last sent, and how far it is known to travel.
fn readings(ui: &mut egui::Ui, state: &FootControlState) {
    let bound = state.pedal_controller();
    let arriving = state.last_control;
    // Which control the drawing is following, and whether that is the one the
    // wah is on. A pedal sending something other than what is bound is the
    // common case before calibration, and saying so is more use than showing
    // the binding and leaving the movement unexplained.
    let (label, colour) = match (arriving, bound) {
        (Some((controller, _)), Some(pedal)) if controller == pedal => {
            (format!("CC {controller}"), theme::TEXT)
        }
        (Some((controller, _)), _) => (format!("CC {controller} · not bound yet"), theme::YELLOW),
        (None, Some(pedal)) => (format!("CC {pedal} · nothing arriving"), theme::MUTED),
        (None, None) => (
            "Nothing from the expression socket yet".to_owned(),
            theme::MUTED,
        ),
    };
    ui.label(RichText::new(label).monospace().small().color(colour));
    ui.label(
        RichText::new(arriving.map_or_else(
            || "sending nothing".to_owned(),
            |(_, value)| format!("sending {value}"),
        ))
        .monospace()
        .small()
        .color(if arriving.is_some() {
            theme::GREEN
        } else {
            theme::MUTED
        }),
    );
    let travel = if state.calibration.is_usable() {
        let inverted = if state.calibration.is_inverted() {
            " · reversed"
        } else {
            ""
        };
        format!(
            "travel {}–{}{inverted}",
            state.calibration.heel, state.calibration.toe
        )
    } else {
        "not calibrated".to_owned()
    };
    ui.label(
        RichText::new(travel)
            .monospace()
            .small()
            .color(theme::MUTED),
    );
}

/// A side-on drawing of the pedal, tilted where the foot has it.
///
/// The number alone does not answer the question somebody actually has, which
/// is whether the thing under their foot is working. A treadle that moves when
/// they move theirs answers it in one glance, and shows up a pedal that only
/// covers half its travel or runs backwards without anybody having to read
/// values off a list.
fn pedal_view(ui: &mut egui::Ui, position: f32, live: bool) {
    /// How far the toe rises off the base plate, heel-down.
    const LIFT: f32 = 46.0;
    /// How thick the treadle is drawn.
    const THICKNESS: f32 = 13.0;

    let (response, painter) = ui.allocate_painter(egui::vec2(200.0, 106.0), egui::Sense::hover());
    let rect = response.rect;
    let position = position.clamp(0.0, 1.0);
    let left = rect.left() + 16.0;
    let right = rect.right() - 16.0;
    let floor = rect.bottom() - 22.0;

    // The base plate the treadle is hinged to.
    painter.rect_filled(
        egui::Rect::from_min_max(
            egui::pos2(left - 8.0, floor),
            egui::pos2(right + 8.0, floor + 11.0),
        ),
        egui::CornerRadius::same(2),
        theme::BG,
    );

    // Where the toe would be at either stop, so the travel is visible even
    // when the pedal is sitting still in the middle of it.
    let hinge_y = floor - 2.0;
    let toe_y = hinge_y - LIFT * (1.0 - position);
    for extreme in [hinge_y - LIFT, hinge_y] {
        painter.line_segment(
            [
                egui::pos2(right - 20.0, extreme),
                egui::pos2(right, extreme),
            ],
            Stroke::new(1.0_f32, theme::BORDER),
        );
    }

    let accent = if live { theme::YELLOW } else { theme::MUTED };
    painter.add(egui::Shape::convex_polygon(
        vec![
            egui::pos2(left, hinge_y - THICKNESS),
            egui::pos2(right, toe_y - THICKNESS),
            egui::pos2(right, toe_y),
            egui::pos2(left, hinge_y),
        ],
        accent,
        Stroke::new(1.0_f32, theme::BG),
    ));
    // The hinge, which is the one part that does not move.
    painter.circle_filled(egui::pos2(left, hinge_y - THICKNESS * 0.5), 3.0, theme::BG);

    painter.text(
        egui::pos2(left - 8.0, floor + 14.0),
        Align2::LEFT_TOP,
        "MIN",
        FontId::monospace(9.0),
        theme::MUTED,
    );
    painter.text(
        egui::pos2(right + 8.0, floor + 14.0),
        Align2::RIGHT_TOP,
        "MAX",
        FontId::monospace(9.0),
        theme::MUTED,
    );
    painter.text(
        egui::pos2(rect.center().x, rect.top() + 2.0),
        Align2::CENTER_TOP,
        format!("{:.0}%", position * 100.0),
        FontId::monospace(15.0),
        if live { theme::TEXT } else { theme::MUTED },
    );
}

/// The switches that recall whole rigs, and which rig each one holds.
fn preset_switches(ui: &mut egui::Ui, state: &mut FootControlState, presets: &PresetLibrary) {
    ui.label(
        RichText::new("PRESETS ON THE SWITCHES")
            .small()
            .monospace()
            .color(theme::MUTED),
    );
    ui.label(
        RichText::new(
            "One switch, one whole rig: the capture, the wah, the tone stack, the dynamics and \
             the time effects together. Build them in the channel strip.",
        )
        .small()
        .color(theme::MUTED),
    );
    if presets.is_empty() {
        ui.label(
            RichText::new(
                "No presets yet. Dial a sound in the channel strip and press SAVE AS to keep it.",
            )
            .small()
            .color(theme::YELLOW),
        );
        return;
    }
    egui::Grid::new("foot_preset_slots")
        .num_columns(4)
        .spacing([8.0, 6.0])
        .show(ui, |ui| {
            for slot in 0..PRESET_SLOTS {
                let index = u8::try_from(slot).unwrap_or(0);
                let action = Action::SelectPreset(index);
                let lit = state.active_preset == Some(index);
                ui.label(
                    RichText::new(action.label())
                        .monospace()
                        .small()
                        .color(if lit { theme::GREEN } else { theme::TEXT }),
                );
                learn_button(ui, state, action);
                let chosen = state.preset_slot(index);
                // A switch holding a rig that has since been deleted says so
                // rather than reading as empty: the two are fixed differently.
                let name = chosen.map_or_else(
                    || "Empty".to_owned(),
                    |id| {
                        presets
                            .get(id)
                            .map_or_else(|| "Deleted".to_owned(), |preset| preset.name.clone())
                    },
                );
                let mut wanted = None;
                egui::ComboBox::from_id_salt(("foot_preset_slot", slot))
                    .selected_text(RichText::new(name).small())
                    .width(240.0)
                    .show_ui(ui, |ui| {
                        for preset in presets.presets() {
                            if ui
                                .selectable_label(chosen == Some(preset.id), &preset.name)
                                .clicked()
                            {
                                wanted = Some(Some(preset.id));
                            }
                        }
                    });
                if ui.button("✕").on_hover_text("Empty this switch").clicked() {
                    wanted = Some(None);
                }
                if let Some(wanted) = wanted {
                    state.set_preset_slot(index, wanted);
                }
                ui.end_row();
            }
        });
}

fn amp_switches(ui: &mut egui::Ui, state: &mut FootControlState, library: &[daw_nam::AmpModel]) {
    ui.label(
        RichText::new("AMPS ON THE SWITCHES")
            .small()
            .monospace()
            .color(theme::MUTED),
    );
    ui.label(
        RichText::new(
            "Each switch loads one capture onto the track you are monitoring, or the selected \
             track if none is.",
        )
        .small()
        .color(theme::MUTED),
    );
    egui::Grid::new("foot_amp_slots")
        .num_columns(4)
        .spacing([8.0, 6.0])
        .show(ui, |ui| {
            for slot in 0..AMP_SLOTS {
                let index = u8::try_from(slot).unwrap_or(0);
                let action = Action::SelectAmp(index);
                let lit = state.active == Some(index);
                ui.label(
                    RichText::new(action.label())
                        .monospace()
                        .small()
                        .color(if lit { theme::GREEN } else { theme::TEXT }),
                );
                learn_button(ui, state, action);
                let chosen = state.slot(index).cloned();
                let name = chosen
                    .as_deref()
                    .and_then(std::path::Path::file_stem)
                    .and_then(|stem| stem.to_str())
                    .unwrap_or("Empty")
                    .to_owned();
                egui::ComboBox::from_id_salt(("foot_slot", slot))
                    .selected_text(RichText::new(name).small())
                    .width(240.0)
                    .show_ui(ui, |ui| {
                        for model in library {
                            if ui
                                .selectable_label(
                                    chosen.as_deref() == Some(model.path.as_path()),
                                    &model.name,
                                )
                                .clicked()
                            {
                                state.slots[slot] = Some(model.path.clone());
                                state.dirty = true;
                            }
                        }
                        if library.is_empty() {
                            ui.label(
                                RichText::new("No captures found")
                                    .small()
                                    .color(theme::MUTED),
                            );
                        }
                    });
                if ui.button("✕").on_hover_text("Empty this switch").clicked() {
                    state.slots[slot] = None;
                    state.dirty = true;
                }
                ui.end_row();
            }
        });
}

fn other_switches(ui: &mut egui::Ui, state: &mut FootControlState) {
    ui.label(
        RichText::new("EVERYTHING ELSE")
            .small()
            .monospace()
            .color(theme::MUTED),
    );
    egui::Grid::new("foot_other")
        .num_columns(3)
        .spacing([8.0, 6.0])
        .show(ui, |ui| {
            for action in Action::all()
                .into_iter()
                .filter(|action| !matches!(action, Action::SelectAmp(_) | Action::SelectPreset(_)))
            {
                ui.label(RichText::new(action.label()).monospace().small());
                learn_button(ui, state, action);
                ui.label(
                    RichText::new(if action.is_continuous() {
                        "Sweeps the wah. Rock the expression pedal to learn it."
                    } else {
                        ""
                    })
                    .small()
                    .color(theme::MUTED),
                );
                ui.end_row();
            }
        });
}

/// The switch bound to `action`, and the button that rebinds it.
fn learn_button(ui: &mut egui::Ui, state: &mut FootControlState, action: Action) {
    let bound = state.bindings.trigger_for(action);
    let learning = state.learning == Some(action);
    let label = if learning {
        "PRESS IT".to_owned()
    } else {
        bound.map_or_else(|| "LEARN".to_owned(), Trigger::describe)
    };
    let colour = if learning {
        theme::YELLOW
    } else if bound.is_some() {
        theme::BLUE
    } else {
        theme::PANEL_2
    };
    let response = ui.add(
        egui::Button::new(RichText::new(label).monospace().small().color(if learning {
            Color32::BLACK
        } else {
            theme::TEXT
        }))
        .fill(colour)
        .min_size(egui::vec2(76.0, 0.0)),
    );
    if response.clicked() {
        state.learning = if learning { None } else { Some(action) };
    }
    if response.secondary_clicked() {
        state.bindings.clear(action);
        state.learning = None;
        state.dirty = true;
    }
    response.on_hover_text("Click, then step on the switch you want. Right-click to unbind it.");
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

    /// A pedal swept to `heel`, then to `toe`, through the panel's own steps.
    fn calibrated(heel: u8, toe: u8) -> FootControlState {
        let mut state = FootControlState::default();
        state.calibrate();
        state.handle(vec![control(11, heel)]);
        finish_step(&mut state);
        state.handle(vec![control(11, toe)]);
        finish_step(&mut state);
        state
    }

    #[test]
    fn calibrating_reads_both_ends_and_finds_the_pedal() {
        let state = calibrated(9, 112);
        assert_eq!(state.calibration.heel, 9);
        assert_eq!(state.calibration.toe, 112);
        // Nobody knows their expression pedal's controller number, so the run
        // that measures it also binds it.
        assert_eq!(state.pedal_controller(), Some(11));
        assert!(state.dirty, "a calibration has to reach the disk");
    }

    #[test]
    fn a_calibrated_pedal_reaches_both_ends_of_the_wah() {
        let mut state = calibrated(9, 112);
        assert_eq!(
            state.handle(vec![control(11, 9)]),
            [Command::Move(Action::WahPedal, 0.0)]
        );
        assert_eq!(
            state.handle(vec![control(11, 112)]),
            [Command::Move(Action::WahPedal, 1.0)]
        );
    }

    #[test]
    fn an_uncalibrated_pedal_still_works_across_the_full_range() {
        let mut state = FootControlState::default();
        assert_eq!(
            state.handle(vec![control(11, 127)]),
            [Command::Move(Action::WahPedal, 1.0)]
        );
    }

    #[test]
    fn nothing_acts_while_the_pedal_is_being_measured() {
        // The sweep being read is not a sweep anybody is playing, and a
        // switch caught by a stray foot must not change the amp mid-run.
        let mut state = FootControlState::default();
        state.calibrate();
        let commands = state.handle(vec![
            control(11, 40),
            Message::Program {
                channel: 0,
                program: 1,
            },
        ]);
        assert!(commands.is_empty(), "{commands:?}");
        assert_eq!(state.last_control, Some((11, 40)));
    }

    #[test]
    fn an_unbound_pedal_still_moves_the_drawing() {
        // The question somebody has before anything is bound is whether the
        // socket works at all, and the only way to answer it is to move.
        let mut state = FootControlState::default();
        state.handle(vec![control(7, 127)]);
        assert_eq!(state.last_control, Some((7, 127)));
        assert!((state.displayed_position() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_control_that_never_changes_cannot_hijack_learning() {
        // This pedal streams a bank select without being touched. Taking
        // whatever arrives next would bind every switch to that noise before
        // a foot could reach the board.
        let mut state = FootControlState {
            learning: Some(Action::Record),
            ..FootControlState::default()
        };
        state.handle(vec![control(0, 0), control(0, 0), control(0, 0)]);
        assert_eq!(state.bindings.trigger_for(Action::Record), None);
        assert_eq!(state.learning, Some(Action::Record), "learning gave up");
        // A control somebody actually moved is taken at once.
        state.handle(vec![control(0, 64)]);
        assert_eq!(
            state.bindings.trigger_for(Action::Record),
            Some(Trigger::Control(0))
        );
    }

    #[test]
    fn a_switch_is_learned_through_the_noise() {
        // A program change is an event: it only arrives because a foot made
        // it, so it is taken however much else is streaming past.
        let mut state = FootControlState {
            learning: Some(Action::Stop),
            ..FootControlState::default()
        };
        state.handle(vec![
            control(0, 0),
            Message::Program {
                channel: 0,
                program: 6,
            },
        ]);
        assert_eq!(
            state.bindings.trigger_for(Action::Stop),
            Some(Trigger::Program(6))
        );
    }

    #[test]
    fn defaults_put_the_four_switches_back_on_the_amps() {
        let mut state = FootControlState::default();
        state.bindings.bind(Trigger::Program(0), Action::PlayPause);
        state
            .bindings
            .bind(Trigger::Program(1), Action::ToggleDelay);
        state.reset_bindings();
        for slot in 0..4 {
            assert_eq!(
                state.bindings.trigger_for(Action::SelectAmp(slot)),
                Some(Trigger::Program(slot))
            );
        }
        assert_eq!(state.bindings.trigger_for(Action::PlayPause), None);
    }

    #[test]
    fn whatever_arrives_last_is_what_is_drawn() {
        // Including a controller nobody expected. Hiding one would hide the
        // pedal on a box that wires its socket somewhere unusual.
        let mut state = FootControlState::default();
        state.handle(vec![control(0, 90)]);
        assert_eq!(state.last_control, Some((0, 90)));
        assert!((state.displayed_position() - 0.708).abs() < 0.01);
    }

    #[test]
    fn the_drawing_follows_the_hardware_while_calibrating() {
        // Mid-run the old calibration is being replaced, so the treadle has
        // to follow the raw reading or it would be measured against the very
        // thing under repair.
        let mut state = calibrated(0, 63);
        state.calibrate();
        state.handle(vec![control(11, 63)]);
        assert!((state.displayed_position() - 0.496).abs() < 0.01);
        state.run = None;
        state.handle(vec![control(11, 63)]);
        assert!((state.displayed_position() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_pedal_that_did_not_move_leaves_the_old_calibration_alone() {
        let mut state = calibrated(9, 112);
        state.calibrate();
        state.handle(vec![control(11, 64)]);
        finish_step(&mut state);
        state.handle(vec![control(11, 66)]);
        finish_step(&mut state);
        assert!(state.calibration_error.is_some());
        assert_eq!(
            state.calibration.heel, 9,
            "a failed run overwrote a good one"
        );
        assert_eq!(state.calibration.toe, 112);
    }

    #[test]
    fn a_backwards_pedal_is_calibrated_the_right_way_round() {
        let state = calibrated(127, 4);
        assert!(state.calibration.is_inverted());
        let mut state = state;
        assert_eq!(
            state.handle(vec![control(11, 127)]),
            [Command::Move(Action::WahPedal, 0.0)]
        );
    }

    #[test]
    fn cancelling_a_run_changes_nothing() {
        let mut state = calibrated(9, 112);
        state.calibrate();
        state.handle(vec![control(11, 70)]);
        state.run = None;
        assert_eq!(state.calibration.heel, 9);
        assert_eq!(state.calibration.toe, 112);
    }

    #[test]
    fn a_calibration_is_remembered_between_sessions() {
        let state = calibrated(9, 112);
        let restored = FootControlState::from_preferences(state.preferences());
        assert_eq!(restored.calibration, state.calibration);
        assert_eq!(restored.pedal_controller(), Some(11));
    }

    /// Two rigs, in the order a foot would step through them.
    fn library() -> PresetLibrary {
        let mut presets = PresetLibrary::default();
        presets.add("Clean", daw_project::TrackEffects::default(), None);
        presets.add("Solo", daw_project::TrackEffects::default(), None);
        presets
    }

    #[test]
    fn saved_rigs_land_on_the_switches_without_anybody_filling_in_a_menu() {
        let presets = library();
        let mut state = FootControlState::default();
        state.seed_presets(&presets);
        assert_eq!(state.preset_slot(0), Some(presets.presets()[0].id));
        assert_eq!(state.preset_slot(1), Some(presets.presets()[1].id));
        assert_eq!(state.preset_slot(2), None);
        assert!(state.dirty, "the seeding has to reach the disk");
    }

    #[test]
    fn seeding_twice_does_not_put_one_rig_on_two_switches() {
        // Every save runs the seed again, and a rig reachable from two
        // switches is a switch somebody thinks is broken.
        let mut presets = library();
        let mut state = FootControlState::default();
        state.seed_presets(&presets);
        let third = presets.add("Funk", daw_project::TrackEffects::default(), None);
        state.seed_presets(&presets);
        assert_eq!(state.preset_slot(2), Some(third));
        assert_eq!(state.preset_slot(3), None);
        assert_eq!(state.preset_slot(0), Some(presets.presets()[0].id));
    }

    #[test]
    fn a_rig_moved_by_hand_is_left_where_it_was_put() {
        let presets = library();
        let mut state = FootControlState::default();
        let solo = presets.presets()[1].id;
        state.set_preset_slot(4, Some(solo));
        state.seed_presets(&presets);
        assert_eq!(state.preset_slot(4), Some(solo), "the switch was moved");
        // Only the rig that was not on a switch yet gets seeded.
        assert_eq!(state.preset_slot(0), Some(presets.presets()[0].id));
        assert_eq!(state.preset_slot(1), None);
    }

    #[test]
    fn deleting_a_rig_takes_it_off_every_switch_it_was_on() {
        let presets = library();
        let mut state = FootControlState::default();
        state.seed_presets(&presets);
        let clean = presets.presets()[0].id;
        state.set_preset_slot(5, Some(clean));
        state.forget_preset(clean);
        assert_eq!(state.preset_slot(0), None);
        assert_eq!(state.preset_slot(5), None);
        assert_eq!(state.preset_slot(1), Some(presets.presets()[1].id));
    }

    #[test]
    fn the_rigs_on_the_switches_are_remembered_between_sessions() {
        let presets = library();
        let mut state = FootControlState::default();
        state.seed_presets(&presets);
        let restored = FootControlState::from_preferences(state.preferences());
        assert_eq!(restored.preset_slot(0), Some(presets.presets()[0].id));
        assert_eq!(restored.preset_slot(1), Some(presets.presets()[1].id));
    }

    #[test]
    fn a_preferences_file_written_before_presets_existed_still_loads() {
        // The pedal predates the rigs, so every file already on disk has no
        // preset switches in it at all.
        let json = r#"{"port":"SINCO","slots":[null,null]}"#;
        let preferences: FootPreferences = serde_json::from_str(json).expect("an older file");
        let state = FootControlState::from_preferences(preferences);
        assert_eq!(state.preset_slot(0), None);
        // And the pedal it does describe is still driven by it.
        assert_eq!(
            state.bindings.trigger_for(Action::SelectAmp(0)),
            Some(Trigger::Program(0))
        );
    }
}
