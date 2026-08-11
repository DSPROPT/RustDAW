//! Domain types shared by the session, engine, and application layers.

use serde::{Deserialize, Serialize};
use std::fmt;

/// The channel layout of a track or audio buffer.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ChannelLayout {
    Mono,
    Stereo,
}

impl ChannelLayout {
    #[must_use]
    pub const fn channel_count(self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
        }
    }
}

/// A positive audio sample rate in frames per second.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct SampleRate(u32);

impl SampleRate {
    pub const DEFAULT: Self = Self(48_000);

    #[must_use]
    pub const fn new(hz: u32) -> Option<Self> {
        if hz == 0 { None } else { Some(Self(hz)) }
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl fmt::Display for SampleRate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} Hz", self.0)
    }
}

/// An absolute frame position on the engine timeline.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub struct SamplePosition(u64);

impl SamplePosition {
    #[must_use]
    pub const fn new(frame: u64) -> Self {
        Self(frame)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    #[must_use]
    pub const fn advanced_by(self, frames: usize) -> Self {
        Self(self.0.saturating_add(frames as u64))
    }
}

/// The input selected for a track. Backend identifiers must remain stable
/// across UI renames and device reordering.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InputRoute {
    pub device_id: String,
    pub channels: Vec<u16>,
}

impl InputRoute {
    pub fn mono(device_id: impl Into<String>, channel: u16) -> Self {
        Self {
            device_id: device_id.into(),
            channels: vec![channel],
        }
    }

    pub fn stereo(device_id: impl Into<String>, left: u16, right: u16) -> Self {
        Self {
            device_id: device_id.into(),
            channels: vec![left, right],
        }
    }

    #[must_use]
    pub fn layout(&self) -> Option<ChannelLayout> {
        match self.channels.len() {
            1 => Some(ChannelLayout::Mono),
            2 => Some(ChannelLayout::Stereo),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_route_layout_matches_selected_channels() {
        assert_eq!(
            InputRoute::mono("scarlett", 0).layout(),
            Some(ChannelLayout::Mono)
        );
        assert_eq!(
            InputRoute::stereo("scarlett", 0, 1).layout(),
            Some(ChannelLayout::Stereo)
        );
    }

    #[test]
    fn sample_position_advance_saturates() {
        assert_eq!(SamplePosition::new(u64::MAX).advanced_by(1).get(), u64::MAX);
    }
}
