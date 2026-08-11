use daw_core::SamplePosition;

/// Sample-clock transport state. It is intended to be owned by the audio
/// thread and changed only at audio block boundaries.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Transport {
    position: SamplePosition,
    playing: bool,
    recording: bool,
}

impl Transport {
    #[must_use]
    pub const fn position(self) -> SamplePosition {
        self.position
    }

    #[must_use]
    pub const fn is_playing(self) -> bool {
        self.playing
    }

    #[must_use]
    pub const fn is_recording(self) -> bool {
        self.recording
    }

    pub const fn play(&mut self) {
        self.playing = true;
    }

    pub const fn record(&mut self) {
        self.playing = true;
        self.recording = true;
    }

    pub const fn stop(&mut self) {
        self.playing = false;
        self.recording = false;
    }

    pub const fn seek(&mut self, position: SamplePosition) {
        self.position = position;
    }

    pub const fn advance(&mut self, frames: usize) {
        if self.playing {
            self.position = self.position.advanced_by(frames);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stopped_transport_does_not_advance() {
        let mut transport = Transport::default();
        transport.advance(256);
        assert_eq!(transport.position().get(), 0);
    }

    #[test]
    fn recording_also_starts_playback() {
        let mut transport = Transport::default();
        transport.record();
        transport.advance(256);
        assert!(transport.is_playing());
        assert!(transport.is_recording());
        assert_eq!(transport.position().get(), 256);
    }
}
