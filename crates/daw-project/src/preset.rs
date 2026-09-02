//! Whole rigs, saved under a name.
//!
//! A channel strip is a rack of a dozen modules, and the difference between a
//! high-gain solo and a clean funk rhythm is a change to nearly every one of
//! them at once: another capture in the amp, the gate opened up, the
//! compressor in, the delay and the reverb out, the wah switched on. Nobody
//! makes that change with a mouse between two bars.
//!
//! So a preset holds the entire strip — [`TrackEffects`] and the capture the
//! amp is playing — and recalling one puts all of it back in a single step,
//! which is what a footswitch can do.
//!
//! The one thing a preset does not carry is where the expression pedal is.
//! That is not a setting; it is where a foot happens to be resting, and
//! restoring it would swing the wah somewhere nobody asked for the instant a
//! preset landed. It is stored at zero and left alone on recall.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::TrackEffects;

/// The schema the preset file is written at.
pub const CURRENT_PRESET_VERSION: u32 = 1;

/// The name a preset saved without one gets.
const UNTITLED: &str = "Preset";

/// One rig: everything the channel strip was doing, under a name.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ChannelPreset {
    /// Stable across renames, because a switch is bound to a preset rather
    /// than to a name — renaming one mid-rehearsal must not unbind a pedal.
    #[serde(default = "Uuid::new_v4")]
    pub id: Uuid,
    pub name: String,
    #[serde(default)]
    pub effects: TrackEffects,
    /// The capture the amp is playing, which is most of what separates one
    /// rig from another. `None` is a strip with no amp in it.
    #[serde(default)]
    pub nam_model: Option<PathBuf>,
}

impl ChannelPreset {
    /// Takes a strip as it stands.
    #[must_use]
    pub fn capture(
        name: impl Into<String>,
        effects: TrackEffects,
        nam_model: Option<PathBuf>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            // Where the pedal was standing is not part of the sound; see the
            // module comment.
            effects: TrackEffects {
                wah_position: 0.0,
                ..effects
            },
            nam_model,
        }
    }

    /// Puts this rig back on a strip.
    ///
    /// The pedal keeps its position: the wah is switched to wherever the
    /// preset says, but swept to wherever the foot is.
    pub fn recall(&self, effects: &mut TrackEffects, nam_model: &mut Option<PathBuf>) {
        let pedal = effects.wah_position;
        *effects = self.effects;
        effects.wah_position = pedal;
        nam_model.clone_from(&self.nam_model);
    }

    /// Whether a strip is still the rig this preset holds.
    ///
    /// Rocking the expression pedal is not an edit, so it does not count as
    /// one — otherwise every preset would read as modified the moment the
    /// wah was played.
    #[must_use]
    pub fn holds(&self, effects: &TrackEffects, nam_model: Option<&Path>) -> bool {
        let settled = TrackEffects {
            wah_position: self.effects.wah_position,
            ..*effects
        };
        self.effects == settled && self.nam_model.as_deref() == nam_model
    }
}

/// Every rig on this machine, in the order a foot steps through them.
///
/// Kept beside the preferences rather than inside a session: a rig is a
/// property of the player and their captures, not of one song, and having to
/// build the clean sound again in every new session is how a preset system
/// stops being used.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PresetLibrary {
    #[serde(default = "default_version")]
    pub version: u32,
    #[serde(default)]
    presets: Vec<ChannelPreset>,
}

impl Default for PresetLibrary {
    fn default() -> Self {
        Self {
            version: CURRENT_PRESET_VERSION,
            presets: Vec::new(),
        }
    }
}

impl PresetLibrary {
    #[must_use]
    pub fn presets(&self) -> &[ChannelPreset] {
        &self.presets
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.presets.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.presets.len()
    }

    #[must_use]
    pub fn get(&self, id: Uuid) -> Option<&ChannelPreset> {
        self.presets.iter().find(|preset| preset.id == id)
    }

    /// Where a preset sits in the order, for stepping through it.
    #[must_use]
    pub fn position(&self, id: Uuid) -> Option<usize> {
        self.presets.iter().position(|preset| preset.id == id)
    }

    /// Saves a strip as a new rig, and hands back the id a switch binds to.
    ///
    /// The name is made unique first. Two presets called "Solo" are a menu
    /// nobody can use and a switch nobody can label.
    pub fn add(&mut self, name: &str, effects: TrackEffects, nam_model: Option<PathBuf>) -> Uuid {
        let name = self.unique_name(name, None);
        let preset = ChannelPreset::capture(name, effects, nam_model);
        let id = preset.id;
        self.presets.push(preset);
        id
    }

    /// Writes a strip over the rig `id` already held, keeping its name and the
    /// switch it is bound to.
    pub fn update(&mut self, id: Uuid, effects: TrackEffects, nam_model: Option<PathBuf>) -> bool {
        let Some(preset) = self.presets.iter_mut().find(|preset| preset.id == id) else {
            return false;
        };
        preset.effects = TrackEffects {
            wah_position: 0.0,
            ..effects
        };
        preset.nam_model = nam_model;
        true
    }

    /// Renames a rig, making the new name unique against the rest.
    pub fn rename(&mut self, id: Uuid, name: &str) -> bool {
        if self.get(id).is_none() {
            return false;
        }
        let name = self.unique_name(name, Some(id));
        if let Some(preset) = self.presets.iter_mut().find(|preset| preset.id == id) {
            preset.name = name;
        }
        true
    }

    pub fn remove(&mut self, id: Uuid) -> bool {
        let before = self.presets.len();
        self.presets.retain(|preset| preset.id != id);
        self.presets.len() != before
    }

    /// The rig either side of `from`, wrapping at both ends.
    ///
    /// Stepping through the rigs with a foot is a loop; running off the end of
    /// the list and having to look down to fix it is not a thing to do on
    /// stage. With nothing recalled yet, stepping forward starts at the top of
    /// the list and stepping back starts at the bottom.
    #[must_use]
    pub fn stepped(&self, from: Option<Uuid>, forward: bool) -> Option<&ChannelPreset> {
        let count = self.presets.len();
        if count == 0 {
            return None;
        }
        let index = match from.and_then(|id| self.position(id)) {
            None if forward => 0,
            None => count - 1,
            Some(current) if forward => (current + 1) % count,
            Some(current) => (current + count - 1) % count,
        };
        self.presets.get(index)
    }

    /// `wanted`, with a number on the end if the library already has it.
    ///
    /// Case-insensitive: "solo" and "Solo" are the same name to everybody
    /// reading a menu, whatever the file says.
    fn unique_name(&self, wanted: &str, except: Option<Uuid>) -> String {
        let wanted = match wanted.trim() {
            "" => UNTITLED,
            trimmed => trimmed,
        };
        let taken = |candidate: &str| {
            self.presets.iter().any(|preset| {
                Some(preset.id) != except && preset.name.eq_ignore_ascii_case(candidate)
            })
        };
        if !taken(wanted) {
            return wanted.to_owned();
        }
        for suffix in 2..u32::MAX {
            let candidate = format!("{wanted} {suffix}");
            if !taken(&candidate) {
                return candidate;
            }
        }
        wanted.to_owned()
    }
}

/// Reads the rigs from disk.
///
/// # Errors
///
/// If the file cannot be read or is not a preset library.
pub fn load(path: &Path) -> Result<PresetLibrary> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read presets {}", path.display()))?;
    serde_json::from_slice(&bytes).context("preset file is invalid")
}

/// Writes the rigs out, through a temporary file and an atomic rename.
///
/// # Errors
///
/// If serialization or any part of the write fails.
pub fn save(library: &PresetLibrary, path: &Path) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(library).context("failed to serialize presets")?;
    crate::write_atomic(&bytes, path)
}

const fn default_version() -> u32 {
    CURRENT_PRESET_VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A strip dialled up for a solo: amp in, delay and reverb behind it.
    fn solo() -> TrackEffects {
        TrackEffects {
            nam_enabled: true,
            nam_input_db: 6.0,
            delay_enabled: true,
            reverb_enabled: true,
            ..TrackEffects::default()
        }
    }

    /// A strip dialled up for funk: compressor and wah, no amp gain.
    fn funk() -> TrackEffects {
        TrackEffects {
            wah_enabled: true,
            compressor_enabled: true,
            ..TrackEffects::default()
        }
    }

    #[test]
    fn a_preset_carries_the_whole_rig_including_the_capture() {
        let mut library = PresetLibrary::default();
        let id = library.add("Solo", solo(), Some(PathBuf::from("5150.nam")));
        let mut effects = funk();
        let mut model = Some(PathBuf::from("twin.nam"));
        library
            .get(id)
            .expect("the rig")
            .recall(&mut effects, &mut model);
        assert!(effects.nam_enabled && effects.delay_enabled && effects.reverb_enabled);
        assert!(!effects.compressor_enabled, "the funk compressor stayed in");
        assert_eq!(model, Some(PathBuf::from("5150.nam")));
    }

    #[test]
    fn recalling_a_preset_leaves_the_pedal_where_the_foot_is() {
        // Snapping the wah to a stored position would swing it somewhere
        // nobody asked for in the middle of a bar.
        let mut library = PresetLibrary::default();
        let id = library.add(
            "Wah",
            TrackEffects {
                wah_enabled: true,
                wah_position: 1.0,
                ..TrackEffects::default()
            },
            None,
        );
        assert!(
            library.get(id).expect("the rig").effects.wah_position.abs() < f32::EPSILON,
            "the pedal position was stored"
        );
        let mut effects = TrackEffects {
            wah_position: 0.4,
            ..TrackEffects::default()
        };
        let mut model = None;
        library
            .get(id)
            .expect("the rig")
            .recall(&mut effects, &mut model);
        assert!(effects.wah_enabled);
        assert!((effects.wah_position - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn playing_the_wah_does_not_make_a_preset_read_as_edited() {
        let mut library = PresetLibrary::default();
        let id = library.add("Funk", funk(), None);
        let preset = library.get(id).expect("the rig");
        let swept = TrackEffects {
            wah_position: 0.9,
            ..funk()
        };
        assert!(preset.holds(&swept, None));
        // Anything else is an edit.
        assert!(!preset.holds(&solo(), None));
        assert!(!preset.holds(&funk(), Some(Path::new("twin.nam"))));
    }

    #[test]
    fn two_rigs_cannot_share_a_name() {
        // A menu with two identical rows, and a switch nobody can label.
        let mut library = PresetLibrary::default();
        library.add("Solo", solo(), None);
        library.add("solo", solo(), None);
        library.add("Solo", solo(), None);
        let names: Vec<&str> = library
            .presets()
            .iter()
            .map(|preset| preset.name.as_str())
            .collect();
        assert_eq!(names, ["Solo", "solo 2", "Solo 3"]);
    }

    #[test]
    fn a_rig_saved_without_a_name_still_gets_one() {
        let mut library = PresetLibrary::default();
        let id = library.add("   ", solo(), None);
        assert_eq!(library.get(id).expect("the rig").name, UNTITLED);
    }

    #[test]
    fn renaming_keeps_the_switch_it_is_bound_to() {
        let mut library = PresetLibrary::default();
        let id = library.add("Solo", solo(), None);
        assert!(library.rename(id, "Lead"));
        assert_eq!(library.get(id).expect("the rig").name, "Lead");
        // A rename onto a name already in use is numbered rather than refused.
        let other = library.add("Clean", funk(), None);
        library.rename(other, "Lead");
        assert_eq!(library.get(other).expect("the rig").name, "Lead 2");
    }

    #[test]
    fn saving_over_a_rig_keeps_its_name_and_its_id() {
        let mut library = PresetLibrary::default();
        let id = library.add("Solo", solo(), None);
        assert!(library.update(id, funk(), Some(PathBuf::from("twin.nam"))));
        let preset = library.get(id).expect("the rig");
        assert_eq!(preset.name, "Solo");
        assert!(preset.effects.compressor_enabled);
        assert_eq!(preset.nam_model, Some(PathBuf::from("twin.nam")));
        assert!(!library.update(Uuid::new_v4(), solo(), None));
    }

    #[test]
    fn stepping_through_the_rigs_wraps_at_both_ends() {
        let mut library = PresetLibrary::default();
        let first = library.add("Clean", funk(), None);
        let second = library.add("Crunch", solo(), None);
        let third = library.add("Solo", solo(), None);
        let step = |from, forward| library.stepped(from, forward).map(|preset| preset.id);
        assert_eq!(step(Some(first), true), Some(second));
        assert_eq!(step(Some(third), true), Some(first));
        assert_eq!(step(Some(first), false), Some(third));
        // With nothing recalled, the step lands at the end it came from.
        assert_eq!(step(None, true), Some(first));
        assert_eq!(step(None, false), Some(third));
    }

    #[test]
    fn an_empty_library_steps_nowhere() {
        let library = PresetLibrary::default();
        assert!(library.stepped(None, true).is_none());
        assert!(library.stepped(Some(Uuid::new_v4()), false).is_none());
    }

    #[test]
    fn a_removed_rig_is_gone_from_the_order_as_well() {
        let mut library = PresetLibrary::default();
        let first = library.add("Clean", funk(), None);
        let second = library.add("Solo", solo(), None);
        assert!(library.remove(first));
        assert!(!library.remove(first));
        assert_eq!(library.len(), 1);
        assert_eq!(library.position(second), Some(0));
    }

    #[test]
    fn a_library_survives_the_round_trip_to_disk() {
        let mut library = PresetLibrary::default();
        library.add("Solo", solo(), Some(PathBuf::from("5150.nam")));
        library.add("Funk", funk(), None);
        let json = serde_json::to_string(&library).expect("serialise");
        let restored: PresetLibrary = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(restored, library);
    }

    #[test]
    fn the_library_goes_to_disk_and_comes_back() {
        let directory = std::env::temp_dir().join(format!("rustdaw-presets-{}", Uuid::new_v4()));
        let path = directory.join("presets.json");
        let mut library = PresetLibrary::default();
        library.add("Solo", solo(), Some(PathBuf::from("5150.nam")));
        // The directory does not exist yet: a first save has to make it.
        save(&library, &path).expect("save");
        assert_eq!(load(&path).expect("load"), library);
        // And a second save replaces the first rather than appending to it.
        library.add("Funk", funk(), None);
        save(&library, &path).expect("save again");
        assert_eq!(load(&path).expect("load again").len(), 2);
        std::fs::remove_dir_all(&directory).expect("clean up");
    }

    #[test]
    fn a_file_written_before_a_module_existed_still_opens() {
        // Every field on the strip carries a serde default, so a library from
        // an older build loads with the new modules switched off rather than
        // failing and taking every rig with it.
        let json = r#"{"presets":[{"name":"Solo","effects":{
            "eq_enabled":false,"low_db":0.0,"mid_db":0.0,"high_db":0.0,
            "compressor_enabled":false,"compressor_threshold_db":-18.0,
            "compressor_ratio":4.0,"compressor_attack_ms":10.0,
            "compressor_release_ms":120.0,"compressor_makeup_db":0.0,
            "gate_enabled":false,"gate_threshold_db":-45.0,"gate_release_ms":120.0}}]}"#;
        let library: PresetLibrary = serde_json::from_str(json).expect("an old library");
        assert_eq!(library.version, CURRENT_PRESET_VERSION);
        let preset = &library.presets()[0];
        assert_eq!(preset.name, "Solo");
        assert!(!preset.effects.reverb_enabled);
        assert!(preset.nam_model.is_none());
    }
}
