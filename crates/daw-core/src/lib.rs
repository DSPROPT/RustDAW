//! Domain types shared by the session, engine, and application layers.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The writable base directory for `RustDAW`'s media — Songs, Recordings,
/// Sessions and Exports all hang off it.
///
/// When the process has a writable working directory — a project checkout or any
/// terminal launch — media lives beside it, keeping the established layout. A GUI
/// app started from Finder or Launchpad instead begins at `/`, the read-only
/// system volume, so this falls back to a per-user `~/RustDAW`. Computed once per
/// process so callers can use it on a render path without repeating filesystem
/// probes.
#[must_use]
pub fn media_root() -> PathBuf {
    static MEDIA_ROOT: OnceLock<PathBuf> = OnceLock::new();
    MEDIA_ROOT.get_or_init(compute_media_root).clone()
}

/// The media subdirectory `name`, under [`media_root`].
#[must_use]
pub fn media_dir(name: &str) -> PathBuf {
    media_root().join(name)
}

fn compute_media_root() -> PathBuf {
    if let Ok(cwd) = std::env::current_dir() {
        if can_write_dir(&cwd) {
            return cwd;
        }
    }
    if let Some(home) = std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
    {
        let user = home.join("RustDAW");
        if can_write_dir(&user) {
            return user;
        }
    }
    std::env::temp_dir().join("RustDAW")
}

/// Whether a directory exists (or can be created) and accepts a written file.
///
/// An actual create-and-write is the only reliable test on macOS's sealed system
/// volume, where the permission bits on `/` do not reveal that it is read-only.
fn can_write_dir(base: &Path) -> bool {
    if std::fs::create_dir_all(base).is_err() {
        return false;
    }
    let probe = base.join(".rustdaw-write-test");
    match std::fs::File::create(&probe) {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

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

    #[test]
    fn media_root_is_absolute_and_actually_writable() {
        let root = media_root();
        assert!(root.is_absolute(), "media root {root:?} should be absolute");
        assert!(can_write_dir(&root), "media root {root:?} must be writable");
        assert!(media_dir("Songs").ends_with("Songs"));
    }

    #[test]
    fn a_path_under_a_read_only_root_is_rejected() {
        // Nothing can create a directory directly under "/", which is exactly
        // the case that broke a Finder-launched app writing to /Songs.
        assert!(!can_write_dir(Path::new(
            "/rustdaw-should-not-be-creatable"
        )));
    }
}
