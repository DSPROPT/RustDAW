//! Backend-neutral, allocation-free audio processing primitives.

mod effects;
pub mod gm;
mod metronome;
mod synth;
mod transport;

pub use effects::{ChannelStrip, ChannelStripParams};
pub use gm::{DrumVoice, Family, GmBank, Patch, drum_voice, patch_for_program, program_name};
pub use metronome::{Metronome, MetronomeError};
pub use synth::{MAX_VOICES, Synth, midi_to_frequency};
pub use transport::Transport;
