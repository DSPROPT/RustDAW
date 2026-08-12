//! Backend-neutral, allocation-free audio processing primitives.

mod delay;
mod effects;
mod gate;
pub mod gm;
mod metronome;
mod reverb;
pub mod soundfont;
mod synth;
pub(crate) mod tone;
mod transport;

pub use delay::Delay;
pub use effects::{ChannelStrip, ChannelStripParams};
pub use gate::{NoiseGate, OPEN_THRESHOLD_DB};
pub use gm::{
    DrumVoice, Family, GmBank, Patch, Wavetable, drum_voice, patch_for_program, program_name,
};
pub use metronome::{Metronome, MetronomeError};
pub use reverb::Reverb;
pub use soundfont::{SampledSynth, SoundFontBank, SoundFontError};
pub use synth::{MAX_VOICES, Synth, midi_to_frequency};
pub use tone::ToneStack;
pub use transport::Transport;
