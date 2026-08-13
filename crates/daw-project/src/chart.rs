//! The chord chart: what a musician would write on a lead sheet.
//!
//! Detection produces spans in seconds, with edges wherever the analysis
//! decided the harmony moved. That is the right thing to store — the chart
//! describes the recording, and must not move if the tempo is later corrected
//! by hand — but it is not what anyone wants to read. Chords change on beats,
//! and a span that starts 40 ms before the beat is describing the analysis
//! rather than the music.
//!
//! So the chart is resolved onto the beat grid: one slot per quarter note,
//! each carrying the chord that is sounding then. And a chord is printed only
//! where it *changes*, because that is how charts are written —
//!
//! ```text
//! | Am  .  .  .  | G   .  .  .  | Dm  .  .  .  | Am  .  G   .  |
//! ```
//!
//! — a dot meaning "still the last one". Repeating `Am Am Am Am` would be
//! noise on the page, and the eye is looking for the changes anyway.

use crate::ChordEvent;
use daw_midi::TempoMap;

/// One quarter-note of the chart.
#[derive(Clone, Debug, PartialEq)]
pub struct ChartBeat {
    /// Beats from the start of the session.
    pub beat: u32,
    /// Where this beat falls, in seconds.
    pub seconds: f64,
    /// Bar number, counting from 1.
    pub bar: u32,
    /// Position within the bar, counting from 1.
    pub beat_in_bar: u16,
    /// The chord printed here, or `None` where it is held from the beat
    /// before — the dot.
    pub label: Option<String>,
    /// The chord sounding on this beat whether or not it is printed, so a
    /// reader landing mid-bar still knows what is playing.
    pub sounding: Option<String>,
    /// Confidence of the sounding chord, `0` to `1`.
    pub confidence: f32,
}

impl ChartBeat {
    /// Whether this beat starts the bar.
    #[must_use]
    pub fn is_downbeat(&self) -> bool {
        self.beat_in_bar == 1
    }

    /// Whether the chord changed here.
    #[must_use]
    pub fn is_change(&self) -> bool {
        self.label.is_some()
    }
}

/// Resolves detected chord spans onto the beat grid.
///
/// `end_seconds` bounds the chart; beats past the last chord are still
/// produced so the grid keeps running to the end of the song.
#[must_use]
pub fn chord_chart(
    chords: &[ChordEvent],
    tempo_map: &TempoMap,
    beats_per_bar: u16,
    end_seconds: f64,
) -> Vec<ChartBeat> {
    if end_seconds <= 0.0 || beats_per_bar == 0 {
        return Vec::new();
    }

    let beats_per_bar = beats_per_bar.max(1);
    let mut chart: Vec<ChartBeat> = Vec::new();
    let mut previous: Option<String> = None;

    for beat in 0..u32::MAX {
        let tick = u64::from(beat) * u64::from(daw_midi::TICKS_PER_QUARTER);
        let seconds = tempo_map.tick_to_seconds(tick);
        if seconds > end_seconds {
            break;
        }

        // The chord sounding at this beat. Sampling at the beat rather than
        // asking which chord overlaps it most is deliberate: what a player
        // needs to know is what to play when the beat arrives.
        let sounding = chords
            .iter()
            .find(|event| seconds >= event.start_seconds && seconds < event.end_seconds)
            .filter(|event| !event.is_silent());

        let name = sounding.map(|event| event.label.clone());
        let label = if name == previous {
            None
        } else {
            previous.clone_from(&name);
            name.clone()
        };

        chart.push(ChartBeat {
            beat,
            seconds,
            bar: beat / u32::from(beats_per_bar) + 1,
            // The remainder is below `beats_per_bar`, which is a `u16`.
            beat_in_bar: u16::try_from(beat % u32::from(beats_per_bar)).unwrap_or(0) + 1,
            label,
            sounding: name,
            confidence: sounding.map_or(0.0, |event| event.confidence),
        });
    }

    chart
}

/// Renders a chart the way it would be written down, one line per bar.
///
/// Used by the command-line example and by the tests, and the clearest
/// statement of what the chart means.
#[must_use]
pub fn format_chart(chart: &[ChartBeat], beats_per_bar: u16) -> String {
    let width = chart
        .iter()
        .filter_map(|beat| beat.label.as_ref())
        .map(String::len)
        .max()
        .unwrap_or(3)
        .max(3);

    let mut lines = Vec::new();
    for bar in chart.chunks(usize::from(beats_per_bar.max(1))) {
        let cells: Vec<String> = bar
            .iter()
            .map(|beat| match &beat.label {
                Some(label) => format!("{label:<width$}"),
                None => format!("{:<width$}", "."),
            })
            .collect();
        let number = bar.first().map_or(0, |beat| beat.bar);
        lines.push(format!("{number:>4} | {} |", cells.join(" ")));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(start: f64, end: f64, label: &str) -> ChordEvent {
        ChordEvent {
            start_seconds: start,
            end_seconds: end,
            label: label.to_owned(),
            confidence: 0.9,
        }
    }

    /// 120 BPM: one beat every half second, four to the bar.
    fn map() -> TempoMap {
        TempoMap::constant(120.0)
    }

    #[test]
    fn a_held_chord_is_printed_once_and_then_dotted() {
        // Am for a whole bar.
        let chords = [event(0.0, 2.0, "Am")];
        let chart = chord_chart(&chords, &map(), 4, 1.9);
        assert_eq!(chart.len(), 4);
        assert_eq!(chart[0].label.as_deref(), Some("Am"));
        assert_eq!(chart[1].label, None, "beat 2 is a dot");
        assert_eq!(chart[2].label, None);
        assert_eq!(chart[3].label, None);
        // But every beat still knows what is playing.
        assert!(chart.iter().all(|b| b.sounding.as_deref() == Some("Am")));
    }

    #[test]
    fn the_example_from_the_lead_sheet() {
        // Am . . . | G . . . | Dm . . . | Am . G .
        let chords = [
            event(0.0, 2.0, "Am"),
            event(2.0, 4.0, "G"),
            event(4.0, 6.0, "Dm"),
            event(6.0, 7.0, "Am"),
            event(7.0, 8.0, "G"),
        ];
        let chart = chord_chart(&chords, &map(), 4, 7.9);
        let printed: Vec<Option<&str>> = chart.iter().map(|b| b.label.as_deref()).collect();
        assert_eq!(
            printed,
            vec![
                Some("Am"),
                None,
                None,
                None,
                Some("G"),
                None,
                None,
                None,
                Some("Dm"),
                None,
                None,
                None,
                Some("Am"),
                None,
                Some("G"),
                None,
            ]
        );
    }

    #[test]
    fn bars_and_beats_are_numbered_from_one() {
        let chart = chord_chart(&[event(0.0, 8.0, "C")], &map(), 4, 3.9);
        assert_eq!((chart[0].bar, chart[0].beat_in_bar), (1, 1));
        assert_eq!((chart[3].bar, chart[3].beat_in_bar), (1, 4));
        assert_eq!((chart[4].bar, chart[4].beat_in_bar), (2, 1));
        assert!(chart[4].is_downbeat());
        assert!(!chart[3].is_downbeat());
    }

    #[test]
    fn a_chord_returning_after_another_is_printed_again() {
        // Am G Am must print Am twice: the second is a change, not a hold.
        let chords = [
            event(0.0, 0.5, "Am"),
            event(0.5, 1.0, "G"),
            event(1.0, 1.5, "Am"),
        ];
        let chart = chord_chart(&chords, &map(), 4, 1.4);
        assert_eq!(chart[0].label.as_deref(), Some("Am"));
        assert_eq!(chart[1].label.as_deref(), Some("G"));
        assert_eq!(chart[2].label.as_deref(), Some("Am"));
    }

    #[test]
    fn silence_reads_as_no_chord_rather_than_holding_the_last_one() {
        let chords = [
            event(0.0, 1.0, "Am"),
            event(1.0, 2.0, "N.C."),
            event(2.0, 3.0, "Am"),
        ];
        let chart = chord_chart(&chords, &map(), 4, 2.9);
        assert_eq!(chart[0].label.as_deref(), Some("Am"));
        assert_eq!(chart[2].sounding, None, "N.C. sounds as nothing");
        assert_eq!(
            chart[4].label.as_deref(),
            Some("Am"),
            "and the chord returning is printed again"
        );
    }

    #[test]
    fn a_change_landing_between_beats_is_heard_on_the_beat_it_reaches() {
        // The detector put the change 40 ms early; the chart puts it on the
        // beat, which is where the band played it.
        let chords = [event(0.0, 1.96, "Am"), event(1.96, 4.0, "G")];
        let chart = chord_chart(&chords, &map(), 4, 3.9);
        assert_eq!(chart[3].label, None, "beat 4 still belongs to Am");
        assert_eq!(chart[4].label.as_deref(), Some("G"), "G arrives on bar 2");
    }

    #[test]
    fn the_chart_runs_to_the_end_even_where_nothing_was_detected() {
        let chart = chord_chart(&[event(0.0, 1.0, "C")], &map(), 4, 3.9);
        assert_eq!(chart.len(), 8, "the grid keeps going");
        assert!(chart[6].sounding.is_none());
    }

    #[test]
    fn an_empty_or_impossible_request_yields_nothing() {
        assert!(chord_chart(&[], &map(), 4, 0.0).is_empty());
        assert!(chord_chart(&[event(0.0, 1.0, "C")], &map(), 0, 4.0).is_empty());
    }

    #[test]
    fn formatting_reads_like_a_lead_sheet() {
        let chords = [
            event(0.0, 2.0, "Am"),
            event(2.0, 3.0, "G"),
            event(3.0, 4.0, "Dm"),
        ];
        let chart = chord_chart(&chords, &map(), 4, 3.9);
        let text = format_chart(&chart, 4);
        assert_eq!(text, "   1 | Am  .   .   .   |\n   2 | G   .   Dm  .   |");
    }
}
