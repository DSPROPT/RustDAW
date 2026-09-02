//! The preset bar across the top of the channel strip.
//!
//! Everything below it dials one sound. This row is how that sound gets a
//! name, gets kept, and gets back onto the strip later — from the mouse here,
//! or from a switch on the floor once it has been put on one in FOOT.
//!
//! It reports what was asked for rather than doing it: the strip is borrowed
//! while the window draws, and the library and the track cannot both be held
//! at once. The caller acts on the request once the window has closed.

use daw_project::PresetLibrary;
use eframe::egui::{self, Color32, RichText};
use uuid::Uuid;

use crate::theme;

/// What the bar was asked to do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PresetRequest {
    /// Put a stored rig on the strip. Also what REVERT asks for: discarding
    /// an edit is recalling what was there before it.
    Recall(Uuid),
    /// Write the strip as it stands over the rig it came from.
    Save(Uuid),
    /// Keep the strip as a new rig under this name.
    SaveAs(String),
    Rename(Uuid, String),
    Delete(Uuid),
}

/// A name being typed, and what it is for.
#[derive(Clone, Copy, Eq, PartialEq)]
enum Naming {
    New,
    Rename,
}

/// What the bar is in the middle of, between repaints.
#[derive(Default)]
pub struct PresetPanel {
    naming: Option<Naming>,
    /// The name being typed. Kept across repaints, and only across them.
    name: String,
    /// A delete waiting to be confirmed.
    ///
    /// Rigs are dialled in over an evening and there is no undo for losing
    /// one, so the button asks twice. The confirmation is on the id rather
    /// than a flag: changing preset between the two clicks must not delete
    /// the one that arrived in the meantime.
    confirm_delete: Option<Uuid>,
}

impl PresetPanel {
    /// Abandons anything half-typed, for a preset arriving from elsewhere.
    pub fn reset(&mut self) {
        self.naming = None;
        self.name.clear();
        self.confirm_delete = None;
    }
}

/// Draws the bar and returns what it was asked to do.
pub fn bar(
    ui: &mut egui::Ui,
    panel: &mut PresetPanel,
    presets: &PresetLibrary,
    current: Option<Uuid>,
    modified: bool,
) -> Option<PresetRequest> {
    let mut request = controls(ui, panel, presets, current, modified);
    if let Some(named) = name_entry(ui, panel, current) {
        request = Some(named);
    }
    if request.is_some() {
        panel.confirm_delete = None;
    }
    request
}

/// The row itself: which rig is on the strip, and what can be done to it.
fn controls(
    ui: &mut egui::Ui,
    panel: &mut PresetPanel,
    presets: &PresetLibrary,
    current: Option<Uuid>,
    modified: bool,
) -> Option<PresetRequest> {
    let mut request = None;
    ui.horizontal(|ui| {
        ui.label(
            RichText::new("PRESET")
                .small()
                .monospace()
                .color(theme::MUTED),
        );
        let name = current
            .and_then(|id| presets.get(id))
            .map_or("— none —", |preset| preset.name.as_str());
        egui::ComboBox::from_id_salt("channel_preset")
            .selected_text(RichText::new(name).color(if modified {
                theme::YELLOW
            } else {
                theme::TEXT
            }))
            .width(220.0)
            .show_ui(ui, |ui| {
                if presets.is_empty() {
                    ui.label(RichText::new("No presets yet").small().color(theme::MUTED));
                }
                for preset in presets.presets() {
                    if ui
                        .selectable_label(current == Some(preset.id), &preset.name)
                        .clicked()
                    {
                        request = Some(PresetRequest::Recall(preset.id));
                    }
                }
            });
        // The dot is the whole reason SAVE is ever pressed, so it sits against
        // the name rather than at the end of the row.
        if modified {
            ui.label(RichText::new("●").color(theme::YELLOW))
                .on_hover_text(
                    "Dialled away from the stored preset. SAVE keeps it, REVERT drops it.",
                );
        }
        if let Some(id) = current.filter(|id| presets.get(*id).is_some()) {
            if ui
                .add_enabled(modified, egui::Button::new("SAVE"))
                .on_hover_text("Write the strip as it stands over this preset")
                .clicked()
            {
                request = Some(PresetRequest::Save(id));
            }
            if ui
                .add_enabled(modified, egui::Button::new("REVERT"))
                .on_hover_text("Put the stored preset back, dropping the changes")
                .clicked()
            {
                request = Some(PresetRequest::Recall(id));
            }
        }
        if ui
            .button("SAVE AS…")
            .on_hover_text(
                "Keep the whole strip — wah, amp and capture, tone, EQ, dynamics, delay and \
                 reverb — as a new preset",
            )
            .clicked()
        {
            panel.confirm_delete = None;
            panel.name = suggested_name(presets, current);
            panel.naming = Some(Naming::New);
        }
        if let Some(id) = current.and_then(|id| presets.get(id)) {
            if ui.button("RENAME").clicked() {
                panel.confirm_delete = None;
                panel.name.clone_from(&id.name);
                panel.naming = Some(Naming::Rename);
            }
            let pending = panel.confirm_delete == Some(id.id);
            let delete = ui.add(
                egui::Button::new(if pending { "SURE?" } else { "DELETE" }).fill(if pending {
                    theme::RED
                } else {
                    theme::PANEL_2
                }),
            );
            if delete.clicked() {
                if pending {
                    panel.confirm_delete = None;
                    request = Some(PresetRequest::Delete(id.id));
                } else {
                    panel.naming = None;
                    panel.confirm_delete = Some(id.id);
                }
            }
            delete.on_hover_text("Click twice. A deleted preset cannot be brought back.");
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                RichText::new("Put a preset on a switch in FOOT")
                    .small()
                    .color(theme::MUTED),
            );
        });
    });
    request
}

/// The name being typed for a new rig or a rename, when one is.
fn name_entry(
    ui: &mut egui::Ui,
    panel: &mut PresetPanel,
    current: Option<Uuid>,
) -> Option<PresetRequest> {
    let naming = panel.naming?;
    let mut request = None;
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(match naming {
                Naming::New => "NAME",
                Naming::Rename => "RENAME TO",
            })
            .small()
            .monospace()
            .color(theme::MUTED),
        );
        let entry = ui.add(
            egui::TextEdit::singleline(&mut panel.name)
                .desired_width(240.0)
                .hint_text("Solo · Funk rhythm · Clean verse"),
        );
        // Typing a name and pressing return is how this is used at speed;
        // the button is for the mouse that is already on the screen.
        let entered = entry.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
        entry.request_focus();
        let keep = ui
            .add(egui::Button::new("KEEP").fill(theme::BLUE_DARK))
            .clicked();
        let cancel =
            ui.button("CANCEL").clicked() || ui.input(|input| input.key_pressed(egui::Key::Escape));
        if entered || keep {
            request = Some(match naming {
                Naming::New => PresetRequest::SaveAs(panel.name.clone()),
                Naming::Rename => current.map_or_else(
                    || PresetRequest::SaveAs(panel.name.clone()),
                    |id| PresetRequest::Rename(id, panel.name.clone()),
                ),
            });
            panel.naming = None;
            panel.name.clear();
        } else if cancel {
            panel.naming = None;
            panel.name.clear();
        }
    });
    request
}

/// What to put in the name box for a new rig.
///
/// A rig saved from another is usually a variation on it, so the name it came
/// from is a better starting point than an empty box — the library numbers it
/// if it comes back unchanged.
fn suggested_name(presets: &PresetLibrary, current: Option<Uuid>) -> String {
    current
        .and_then(|id| presets.get(id))
        .map_or_else(String::new, |preset| preset.name.clone())
}

/// The bar's own colour, for a strip drawn on a dark panel.
pub const BAR_FILL: Color32 = Color32::from_rgb(36, 39, 42);

#[cfg(test)]
mod tests {
    use super::*;
    use daw_project::TrackEffects;

    #[test]
    fn saving_from_a_rig_starts_from_its_name() {
        let mut presets = PresetLibrary::default();
        let id = presets.add("Solo", TrackEffects::default(), None);
        assert_eq!(suggested_name(&presets, Some(id)), "Solo");
        assert_eq!(suggested_name(&presets, None), "");
    }

    #[test]
    fn a_half_typed_name_is_dropped_when_a_preset_arrives_from_the_pedal() {
        let mut panel = PresetPanel {
            naming: Some(Naming::Rename),
            name: "half typed".to_owned(),
            confirm_delete: Some(Uuid::new_v4()),
        };
        panel.reset();
        assert!(panel.naming.is_none());
        assert!(panel.name.is_empty());
        assert!(panel.confirm_delete.is_none());
    }
}
