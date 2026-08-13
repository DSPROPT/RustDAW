#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

//! Musical time: a tempo map that converts between ticks and samples.
//!
//! `RustDAW` places audio with integer sample positions, which is exactly right
//! for recorded takes and wrong for notes: transposing a session to a new
//! tempo must move every note and leave every recording where it is. So MIDI is
//! stored in ticks and converted here, and the conversion is the only place
//! that knows about tempo.
//!
//! A map is a list of points, each starting a constant-tempo span. A song with
//! one steady tempo has exactly one point, so the common case costs nothing.

use serde::{Deserialize, Serialize};

/// Ticks per quarter note. 960 divides cleanly by 2, 3, 4, 5, 6 and 8, so
/// triplets and quintuplets land on integers instead of drifting.
pub const TICKS_PER_QUARTER: u32 = 960;

/// The tempo range the transport and click accept.
pub const MIN_BPM: f64 = 20.0;
pub const MAX_BPM: f64 = 300.0;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct TempoPoint {
    /// Tick at which this tempo takes effect.
    pub tick: u64,
    pub bpm: f64,
}

/// Serialised form. The second offsets are derived, never stored, so a
/// hand-edited session can never carry a cache that disagrees with its tempi.
#[derive(Clone, Debug, Deserialize, Serialize)]
struct TempoMapData {
    points: Vec<TempoPoint>,
    #[serde(default = "default_ticks_per_quarter")]
    ticks_per_quarter: u32,
}

const fn default_ticks_per_quarter() -> u32 {
    TICKS_PER_QUARTER
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(from = "TempoMapData", into = "TempoMapData")]
pub struct TempoMap {
    points: Vec<TempoPoint>,
    ticks_per_quarter: u32,
    /// Seconds from the start of the timeline to `points[i].tick`.
    starts: Vec<f64>,
}

impl Default for TempoMap {
    fn default() -> Self {
        Self::constant(120.0)
    }
}

impl From<TempoMapData> for TempoMap {
    fn from(data: TempoMapData) -> Self {
        Self::new(data.points, data.ticks_per_quarter)
    }
}

impl From<TempoMap> for TempoMapData {
    fn from(map: TempoMap) -> Self {
        Self {
            points: map.points,
            ticks_per_quarter: map.ticks_per_quarter,
        }
    }
}

impl PartialEq for TempoMap {
    fn eq(&self, other: &Self) -> bool {
        self.points == other.points && self.ticks_per_quarter == other.ticks_per_quarter
    }
}

impl TempoMap {
    /// Builds a map from tempo points. Points are sorted, de-duplicated by
    /// tick, and clamped into the supported range; an empty list becomes 120.
    #[must_use]
    pub fn new(mut points: Vec<TempoPoint>, ticks_per_quarter: u32) -> Self {
        let ticks_per_quarter = ticks_per_quarter.max(1);
        points.retain(|point| point.bpm.is_finite() && point.bpm > 0.0);
        for point in &mut points {
            point.bpm = point.bpm.clamp(MIN_BPM, MAX_BPM);
        }
        points.sort_by_key(|point| point.tick);
        points.dedup_by_key(|point| point.tick);
        // Files that write a tempo event per beat carry hundreds of identical
        // tempi; keeping only the changes makes `is_constant` mean what it says.
        points.dedup_by(|later, earlier| (later.bpm - earlier.bpm).abs() < 1e-6);
        if points.first().is_none_or(|point| point.tick != 0) {
            let first_bpm = points.first().map_or(120.0, |point| point.bpm);
            points.insert(
                0,
                TempoPoint {
                    tick: 0,
                    bpm: first_bpm,
                },
            );
        }

        let mut starts = Vec::with_capacity(points.len());
        let mut seconds = 0.0;
        for (index, point) in points.iter().enumerate() {
            starts.push(seconds);
            if let Some(next) = points.get(index + 1) {
                let span = f64::from(u32::try_from(next.tick - point.tick).unwrap_or(u32::MAX));
                seconds += span * 60.0 / (point.bpm * f64::from(ticks_per_quarter));
            }
        }

        Self {
            points,
            ticks_per_quarter,
            starts,
        }
    }

    #[must_use]
    pub fn constant(bpm: f64) -> Self {
        Self::new(vec![TempoPoint { tick: 0, bpm }], TICKS_PER_QUARTER)
    }

    /// Builds a map from detected beat times in seconds.
    ///
    /// A new point is emitted only where the tempo actually changes by more
    /// than `tolerance_bpm`, so a steady song collapses to a single point
    /// instead of one per beat. `beats_per_bar` positions do not matter here;
    /// every entry in `beat_times` is treated as one quarter note.
    ///
    /// Intervals are median-filtered first. One beat landing a frame early is
    /// tracking noise, not a tempo change, and writing it into the map would
    /// make the bar lines wobble for the rest of the song.
    #[must_use]
    pub fn from_beat_times(beat_times: &[f64], tolerance_bpm: f64) -> Self {
        let raw: Vec<f64> = beat_times
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .filter(|interval| *interval > 0.0)
            .collect();
        if raw.is_empty() {
            return Self::constant(120.0);
        }
        let intervals = median_filter(&raw, 5);

        let mut points = Vec::new();
        let mut current_bpm = f64::NAN;
        for (index, interval) in intervals.iter().enumerate() {
            let bpm = (60.0 / interval).clamp(MIN_BPM, MAX_BPM);
            if !current_bpm.is_finite() || (bpm - current_bpm).abs() > tolerance_bpm {
                points.push(TempoPoint {
                    tick: u64::try_from(index).unwrap_or(0) * u64::from(TICKS_PER_QUARTER),
                    bpm,
                });
                current_bpm = bpm;
            }
        }
        Self::new(points, TICKS_PER_QUARTER)
    }

    #[must_use]
    pub fn points(&self) -> &[TempoPoint] {
        &self.points
    }

    #[must_use]
    pub const fn ticks_per_quarter(&self) -> u32 {
        self.ticks_per_quarter
    }

    /// True when the whole song runs at one tempo.
    #[must_use]
    pub fn is_constant(&self) -> bool {
        self.points.len() <= 1
    }

    /// The tempo in force at a tick.
    #[must_use]
    pub fn bpm_at_tick(&self, tick: u64) -> f64 {
        let index = self.segment_for_tick(tick);
        self.points[index].bpm
    }

    /// The tempo in force at a moment in seconds.
    #[must_use]
    pub fn bpm_at_seconds(&self, seconds: f64) -> f64 {
        let index = self.segment_for_seconds(seconds);
        self.points[index].bpm
    }

    /// Converts a tick to a position in seconds.
    #[must_use]
    pub fn tick_to_seconds(&self, tick: u64) -> f64 {
        let index = self.segment_for_tick(tick);
        let point = self.points[index];
        let offset = f64::from(u32::try_from(tick - point.tick).unwrap_or(u32::MAX));
        self.starts[index] + offset * 60.0 / (point.bpm * f64::from(self.ticks_per_quarter))
    }

    /// Converts a position in seconds to a tick.
    #[must_use]
    pub fn seconds_to_tick(&self, seconds: f64) -> u64 {
        if seconds <= 0.0 {
            return 0;
        }
        let index = self.segment_for_seconds(seconds);
        let point = self.points[index];
        let elapsed = seconds - self.starts[index];
        let ticks = elapsed * point.bpm * f64::from(self.ticks_per_quarter) / 60.0;
        point.tick.saturating_add(ticks.max(0.0).round() as u64)
    }

    /// Converts a tick to an absolute frame position at a sample rate.
    #[must_use]
    pub fn tick_to_frame(&self, tick: u64, sample_rate: u32) -> u64 {
        (self.tick_to_seconds(tick) * f64::from(sample_rate))
            .max(0.0)
            .round() as u64
    }

    /// Converts an absolute frame position to a tick.
    #[must_use]
    pub fn frame_to_tick(&self, frame: u64, sample_rate: u32) -> u64 {
        if sample_rate == 0 {
            return 0;
        }
        self.seconds_to_tick(frame as f64 / f64::from(sample_rate))
    }

    /// Seconds occupied by one bar starting at `tick`.
    #[must_use]
    pub fn seconds_per_bar(&self, tick: u64, beats_per_bar: u16) -> f64 {
        60.0 / self.bpm_at_tick(tick) * f64::from(beats_per_bar.max(1))
    }

    fn segment_for_tick(&self, tick: u64) -> usize {
        match self.points.binary_search_by_key(&tick, |point| point.tick) {
            Ok(index) => index,
            Err(index) => index.saturating_sub(1),
        }
    }

    fn segment_for_seconds(&self, seconds: f64) -> usize {
        self.starts
            .partition_point(|start| *start <= seconds)
            .saturating_sub(1)
    }
}

/// Replaces each value with the median of the `width` values around it.
/// Preserves sustained changes while discarding isolated outliers.
fn median_filter(values: &[f64], width: usize) -> Vec<f64> {
    let half = width / 2;
    if width < 3 || values.len() <= width {
        return values.to_vec();
    }
    let mut window = Vec::with_capacity(width);
    (0..values.len())
        .map(|index| {
            let start = index.saturating_sub(half);
            let end = (index + half + 1).min(values.len());
            window.clear();
            window.extend_from_slice(&values[start..end]);
            window.sort_by(f64::total_cmp);
            window[window.len() / 2]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_stray_beat_does_not_become_a_tempo_change() {
        // A steady 120 BPM grid with a single beat landing 25 ms early.
        let mut beats: Vec<f64> = (0..40).map(|index| f64::from(index) * 0.5).collect();
        beats[20] -= 0.025;
        let map = TempoMap::from_beat_times(&beats, 3.0);
        assert!(
            map.is_constant(),
            "tracking noise produced {} tempo points",
            map.points().len()
        );
    }

    #[test]
    fn a_sustained_change_survives_the_median_filter() {
        let mut beats = vec![0.0];
        for _ in 0..20 {
            let last = *beats.last().unwrap();
            beats.push(last + 0.5); // 120 BPM
        }
        for _ in 0..20 {
            let last = *beats.last().unwrap();
            beats.push(last + 0.4); // 150 BPM
        }
        let map = TempoMap::from_beat_times(&beats, 3.0);
        assert_eq!(map.points().len(), 2);
        assert!((map.points()[1].bpm - 150.0).abs() < 1.0);
    }

    #[test]
    fn a_constant_map_converts_both_ways() {
        let map = TempoMap::constant(120.0);
        // One quarter note at 120 BPM is half a second.
        assert!((map.tick_to_seconds(u64::from(TICKS_PER_QUARTER)) - 0.5).abs() < 1e-12);
        assert_eq!(map.seconds_to_tick(0.5), u64::from(TICKS_PER_QUARTER));
        assert_eq!(
            map.tick_to_frame(u64::from(TICKS_PER_QUARTER), 48_000),
            24_000
        );
        assert_eq!(
            map.frame_to_tick(24_000, 48_000),
            u64::from(TICKS_PER_QUARTER)
        );
        assert!(map.is_constant());
    }

    #[test]
    fn tempo_changes_accumulate_across_segments() {
        // Four quarters at 120 (2 s), then quarters at 60 (1 s each).
        let map = TempoMap::new(
            vec![
                TempoPoint {
                    tick: 0,
                    bpm: 120.0,
                },
                TempoPoint {
                    tick: u64::from(TICKS_PER_QUARTER) * 4,
                    bpm: 60.0,
                },
            ],
            TICKS_PER_QUARTER,
        );
        assert!((map.tick_to_seconds(u64::from(TICKS_PER_QUARTER) * 4) - 2.0).abs() < 1e-12);
        assert!((map.tick_to_seconds(u64::from(TICKS_PER_QUARTER) * 5) - 3.0).abs() < 1e-12);
        assert!(!map.is_constant());
        assert!((map.bpm_at_tick(u64::from(TICKS_PER_QUARTER) * 6) - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn conversion_round_trips_through_a_tempo_change() {
        let map = TempoMap::new(
            vec![
                TempoPoint {
                    tick: 0,
                    bpm: 137.0,
                },
                TempoPoint {
                    tick: 5_000,
                    bpm: 96.5,
                },
                TempoPoint {
                    tick: 40_000,
                    bpm: 180.0,
                },
            ],
            TICKS_PER_QUARTER,
        );
        for tick in [0_u64, 1, 4_999, 5_000, 5_001, 39_999, 40_000, 123_456] {
            let seconds = map.tick_to_seconds(tick);
            assert_eq!(
                map.seconds_to_tick(seconds),
                tick,
                "tick {tick} did not round trip"
            );
        }
    }

    #[test]
    fn a_steady_beat_grid_collapses_to_one_point() {
        let beats: Vec<f64> = (0..200).map(|index| f64::from(index) * 0.5).collect();
        let map = TempoMap::from_beat_times(&beats, 0.5);
        assert!(map.is_constant(), "a steady song must not store 200 tempi");
        assert!((map.bpm_at_tick(0) - 120.0).abs() < 1e-9);
    }

    #[test]
    fn a_song_that_speeds_up_keeps_its_changes() {
        // 120 BPM for eight beats, then 150 BPM.
        let mut beats = vec![0.0];
        for _ in 0..8 {
            let last = *beats.last().unwrap();
            beats.push(last + 0.5);
        }
        for _ in 0..8 {
            let last = *beats.last().unwrap();
            beats.push(last + 0.4);
        }
        let map = TempoMap::from_beat_times(&beats, 0.5);
        assert_eq!(map.points().len(), 2);
        assert!((map.points()[0].bpm - 120.0).abs() < 1e-9);
        assert!((map.points()[1].bpm - 150.0).abs() < 1e-9);
    }

    #[test]
    fn repeated_identical_tempi_collapse_to_one_point() {
        // A tempo event per beat, all at the same speed, is one tempo.
        let points = (0..64)
            .map(|beat| TempoPoint {
                tick: u64::from(TICKS_PER_QUARTER) * beat,
                bpm: 120.0,
            })
            .collect();
        let map = TempoMap::new(points, TICKS_PER_QUARTER);
        assert!(map.is_constant());
        assert!((map.tick_to_seconds(u64::from(TICKS_PER_QUARTER) * 64) - 32.0).abs() < 1e-9);
    }

    #[test]
    fn a_map_always_starts_at_tick_zero() {
        let map = TempoMap::new(
            vec![TempoPoint {
                tick: 4_000,
                bpm: 90.0,
            }],
            TICKS_PER_QUARTER,
        );
        assert_eq!(map.points()[0].tick, 0);
        assert!((map.bpm_at_tick(0) - 90.0).abs() < f64::EPSILON);
    }

    #[test]
    fn absurd_tempi_are_clamped_and_junk_is_dropped() {
        let map = TempoMap::new(
            vec![
                TempoPoint { tick: 0, bpm: 1e9 },
                TempoPoint {
                    tick: 500,
                    bpm: f64::NAN,
                },
                TempoPoint {
                    tick: 900,
                    bpm: -4.0,
                },
                TempoPoint {
                    tick: 1_000,
                    bpm: 0.001,
                },
            ],
            TICKS_PER_QUARTER,
        );
        assert!(
            map.points()
                .iter()
                .all(|point| point.bpm >= MIN_BPM && point.bpm <= MAX_BPM)
        );
        assert!(map.points().iter().all(|point| point.bpm.is_finite()));
    }

    #[test]
    fn empty_beat_grids_fall_back_to_a_usable_tempo() {
        assert!((TempoMap::from_beat_times(&[], 0.5).bpm_at_tick(0) - 120.0).abs() < f64::EPSILON);
        assert!(
            (TempoMap::from_beat_times(&[1.0], 0.5).bpm_at_tick(0) - 120.0).abs() < f64::EPSILON
        );
    }

    #[test]
    fn serde_round_trip_rebuilds_the_second_offsets() {
        let map = TempoMap::new(
            vec![
                TempoPoint {
                    tick: 0,
                    bpm: 100.0,
                },
                TempoPoint {
                    tick: 9_600,
                    bpm: 140.0,
                },
            ],
            TICKS_PER_QUARTER,
        );
        let json = serde_json::to_string(&map).unwrap();
        let restored: TempoMap = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, map);
        assert!(
            (restored.tick_to_seconds(20_000) - map.tick_to_seconds(20_000)).abs() < 1e-12,
            "derived offsets must survive a round trip"
        );
    }
}
