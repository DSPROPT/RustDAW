//! The DSPRO Studio `project.json` pipeline manifest.
//!
//! Only the fields `RustDAW` needs are modelled. Everything is optional because
//! the manifest is written stage by stage: a project whose pipeline failed
//! halfway is still a readable document, and reporting "no stems yet" is more
//! useful than refusing to parse it.

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

/// Stem order used for the imported track list. This is deliberately not the
/// worker's `STEM_NAMES` order: tracks read top to bottom in a DAW, and
/// rhythm section first with vocals last is the conventional arrangement.
pub const STEM_ORDER: [&str; 6] = ["drums", "bass", "guitar", "piano", "other", "vocals"];

/// Drum component order, low to high, matching how a kit is usually laid out.
pub const DRUMKIT_ORDER: [&str; 4] = ["kick", "snare", "toms", "cymbals"];

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SongManifest {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub artist: Option<String>,
    #[serde(default)]
    pub style: Option<String>,
    #[serde(default)]
    pub source_url: Option<String>,
    #[serde(default)]
    pub duration: Option<f64>,
    #[serde(default)]
    pub files: SongFiles,
    #[serde(default)]
    pub beat_grid: Option<BeatGrid>,
    #[serde(default)]
    pub stages: BTreeMap<String, StageRecord>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct SongFiles {
    #[serde(default)]
    pub stems: BTreeMap<String, String>,
    #[serde(default)]
    pub drumkit: BTreeMap<String, String>,
    #[serde(default)]
    pub midi: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct StageRecord {
    #[serde(default)]
    pub status: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BeatGrid {
    #[serde(default)]
    pub beat_times: Vec<f64>,
    #[serde(default)]
    pub beats_per_bar: Option<u16>,
    #[serde(default)]
    pub downbeat_index: Option<usize>,
    #[serde(default)]
    pub source: Option<String>,
}

/// Tempo recovered from a beat grid, with the evidence needed to judge it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectedTempo {
    /// Median beats per minute across the grid.
    pub bpm: f64,
    /// Largest deviation from the median interval, as a fraction of it. A
    /// steady programmed track sits near zero; a live take that speeds up
    /// reads high, which is a warning that one fixed tempo will drift.
    pub drift: f64,
    /// Timeline seconds of the first detected downbeat.
    pub first_downbeat: f64,
    pub beats_per_bar: u16,
}

impl SongManifest {
    /// Reads and parses `project.json` from a DSPRO project directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is missing or is not valid JSON.
    pub fn load(project_dir: &Path) -> Result<Self> {
        let path = project_dir.join("project.json");
        let bytes = std::fs::read(&path)
            .with_context(|| format!("failed to read manifest {}", path.display()))?;
        serde_json::from_slice(&bytes)
            .with_context(|| format!("{} is not a valid project manifest", path.display()))
    }

    /// A session name of the form `Artist - Title`, falling back sensibly.
    #[must_use]
    pub fn display_name(&self) -> String {
        let title = self
            .title
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let artist = self
            .artist
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        match (artist, title) {
            (Some(artist), Some(title)) => format!("{artist} - {title}"),
            (None, Some(title)) => title.to_owned(),
            (Some(artist), None) => artist.to_owned(),
            (None, None) => "Imported Song".to_owned(),
        }
    }

    /// True when the separation stage completed and stems are on disk.
    #[must_use]
    pub fn has_stems(&self) -> bool {
        !self.files.stems.is_empty()
    }

    /// Stems in [`STEM_ORDER`], then any stem the worker produced that this
    /// build does not know about, so a new Demucs model does not go missing.
    #[must_use]
    pub fn ordered_stems(&self) -> Vec<(String, String)> {
        ordered_by(&self.files.stems, &STEM_ORDER)
    }

    /// Drum components in [`DRUMKIT_ORDER`].
    #[must_use]
    pub fn ordered_drumkit(&self) -> Vec<(String, String)> {
        ordered_by(&self.files.drumkit, &DRUMKIT_ORDER)
    }

    /// Tempo, meter and downbeat recovered from the stored beat grid.
    #[must_use]
    pub fn detected_tempo(&self) -> Option<DetectedTempo> {
        self.beat_grid.as_ref().and_then(BeatGrid::detect)
    }
}

fn ordered_by(files: &BTreeMap<String, String>, order: &[&str]) -> Vec<(String, String)> {
    let mut result: Vec<(String, String)> = order
        .iter()
        .filter_map(|name| {
            files
                .get(*name)
                .map(|path| ((*name).to_owned(), path.clone()))
        })
        .collect();
    for (name, path) in files {
        if !order.contains(&name.as_str()) {
            result.push((name.clone(), path.clone()));
        }
    }
    result
}

impl BeatGrid {
    /// Derives tempo from beat spacing.
    ///
    /// The median interval is used rather than the mean because a single
    /// dropped or doubled beat anywhere in the grid would drag an average
    /// well off the real tempo.
    #[must_use]
    pub fn detect(&self) -> Option<DetectedTempo> {
        let mut intervals: Vec<f64> = self
            .beat_times
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .filter(|interval| *interval > 0.0)
            .collect();
        if intervals.is_empty() {
            return None;
        }
        intervals.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
        let median = intervals[intervals.len() / 2];
        if median <= 0.0 {
            return None;
        }
        let drift = intervals
            .iter()
            .map(|interval| ((interval - median) / median).abs())
            .fold(0.0_f64, f64::max);
        let beats_per_bar = self.beats_per_bar.filter(|value| (2..=12).contains(value));
        let downbeat_index = self.downbeat_index.unwrap_or(0);
        let first_downbeat = self
            .beat_times
            .get(downbeat_index)
            .copied()
            .or_else(|| self.beat_times.first().copied())
            .unwrap_or(0.0);
        Some(DetectedTempo {
            bpm: 60.0 / median,
            drift,
            first_downbeat: first_downbeat.max(0.0),
            beats_per_bar: beats_per_bar.unwrap_or(4),
        })
    }
}

impl DetectedTempo {
    /// The tempo rounded into the range `RustDAW` sessions accept.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to 20..=300, which every u16 represents exactly"
    )]
    pub fn rounded_bpm(self) -> u16 {
        self.bpm.round().clamp(20.0, 300.0) as u16
    }

    /// Seconds of silence to insert so the first downbeat lands on a bar line
    /// of `RustDAW`'s click, which always starts counting at frame zero.
    ///
    /// Measured against [`Self::rounded_bpm`], not the raw detected tempo: the
    /// click will run at the rounded value, and aligning to anything else
    /// leaves the downbeat tens of milliseconds off the very first bar.
    #[must_use]
    pub fn bar_alignment_offset(self) -> f64 {
        let seconds_per_bar =
            60.0 / f64::from(self.rounded_bpm()) * f64::from(self.beats_per_bar);
        if seconds_per_bar <= 0.0 {
            return 0.0;
        }
        let position_in_bar = self.first_downbeat % seconds_per_bar;
        if position_in_bar <= f64::EPSILON {
            0.0
        } else {
            seconds_per_bar - position_in_bar
        }
    }

    /// True when one fixed tempo cannot represent this grid honestly.
    #[must_use]
    pub fn is_unsteady(self) -> bool {
        self.drift > 0.08
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(beat_times: Vec<f64>, beats_per_bar: u16) -> BeatGrid {
        BeatGrid {
            beat_times,
            beats_per_bar: Some(beats_per_bar),
            downbeat_index: Some(0),
            source: None,
        }
    }

    #[test]
    fn tempo_comes_from_the_median_interval() {
        let beats = (0..16).map(|index| f64::from(index) * 0.5).collect();
        let detected = grid(beats, 4).detect().unwrap();
        assert!((detected.bpm - 120.0).abs() < 1e-9);
        assert_eq!(detected.rounded_bpm(), 120);
        assert!(!detected.is_unsteady());
    }

    #[test]
    fn one_dropped_beat_does_not_move_the_tempo() {
        // A single doubled gap in the middle would pull a mean tempo down by
        // several BPM; the median must ignore it.
        let mut beats: Vec<f64> = (0..16).map(|index| f64::from(index) * 0.5).collect();
        for beat in &mut beats[8..] {
            *beat += 0.5;
        }
        let detected = grid(beats, 4).detect().unwrap();
        assert_eq!(detected.rounded_bpm(), 120);
        assert!(detected.is_unsteady(), "the dropped beat must be reported");
    }

    #[test]
    fn alignment_offset_pushes_the_downbeat_onto_a_bar_line() {
        // 120 BPM in 4/4 is a two-second bar. A downbeat at 13.5 s sits 1.5 s
        // into the seventh bar, so the song is delayed the remaining 0.5 s to
        // land it on the next bar line.
        let detected = DetectedTempo {
            bpm: 120.0,
            drift: 0.0,
            first_downbeat: 13.5,
            beats_per_bar: 4,
        };
        assert!((detected.bar_alignment_offset() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn a_downbeat_already_on_a_bar_line_is_not_moved() {
        let detected = DetectedTempo {
            bpm: 120.0,
            drift: 0.0,
            first_downbeat: 8.0,
            beats_per_bar: 4,
        };
        assert!(detected.bar_alignment_offset().abs() < 1e-9);
    }

    #[test]
    fn alignment_uses_the_tempo_the_click_will_actually_run_at() {
        // 95.2 BPM is stored as 95, so the offset must land the downbeat on a
        // 95 BPM bar line. Aligning to the raw 95.2 leaves it ~30 ms early.
        let detected = DetectedTempo {
            bpm: 95.2,
            drift: 0.0,
            first_downbeat: 13.5,
            beats_per_bar: 4,
        };
        let bar = 60.0 / f64::from(detected.rounded_bpm()) * 4.0;
        let downbeat = detected.first_downbeat + detected.bar_alignment_offset();
        assert!(
            (downbeat % bar).min(bar - (downbeat % bar)) < 1e-9,
            "downbeat at {downbeat} s is not on a {bar} s bar line"
        );
    }

    #[test]
    fn tempo_is_clamped_into_the_session_range() {
        let fast = DetectedTempo {
            bpm: 999.0,
            drift: 0.0,
            first_downbeat: 0.0,
            beats_per_bar: 4,
        };
        assert_eq!(fast.rounded_bpm(), 300);
    }

    #[test]
    fn empty_and_single_beat_grids_report_no_tempo() {
        assert!(grid(Vec::new(), 4).detect().is_none());
        assert!(grid(vec![1.0], 4).detect().is_none());
    }

    #[test]
    fn unknown_stems_are_kept_after_the_known_order() {
        let mut stems = BTreeMap::new();
        stems.insert("vocals".to_owned(), "stems/vocals.wav".to_owned());
        stems.insert("drums".to_owned(), "stems/drums.wav".to_owned());
        stems.insert("strings".to_owned(), "stems/strings.wav".to_owned());
        let manifest = SongManifest {
            files: SongFiles {
                stems,
                ..SongFiles::default()
            },
            ..SongManifest::default()
        };
        let names: Vec<String> = manifest
            .ordered_stems()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        assert_eq!(names, ["drums", "vocals", "strings"]);
    }

    #[test]
    fn display_name_survives_missing_metadata() {
        assert_eq!(SongManifest::default().display_name(), "Imported Song");
        let titled = SongManifest {
            title: Some("  ".to_owned()),
            artist: Some("Maluma".to_owned()),
            ..SongManifest::default()
        };
        assert_eq!(titled.display_name(), "Maluma");
    }

    #[test]
    fn real_manifest_shape_parses() {
        let json = br#"{
            "stages": {"download": {"status": "done", "updatedAt": 1.0}},
            "title": "11 P.M.", "artist": "Maluma", "style": "Pop",
            "sourceUrl": "https://www.youtube.com/watch?v=jiAs2JjHfYY",
            "duration": 175.68,
            "files": {"stems": {"drums": "stems/drums.wav"}, "midi": {"song": "midi/song.mid"}},
            "beatGrid": {"beatTimes": [0.0, 0.5, 1.0], "beatsPerBar": 4, "downbeatIndex": 0, "source": "app"}
        }"#;
        let manifest: SongManifest = serde_json::from_slice(json).unwrap();
        assert_eq!(manifest.display_name(), "Maluma - 11 P.M.");
        assert_eq!(manifest.stages["download"].status, "done");
        assert!(manifest.has_stems());
        assert_eq!(manifest.detected_tempo().unwrap().rounded_bpm(), 120);
    }
}
