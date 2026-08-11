//! Linux audio backend discovery. Stream ownership will live here as the
//! recording engine grows, keeping backend types out of the core crates.

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait};

mod runtime;

pub use runtime::{AudioRuntime, AudioRuntimeConfig, RuntimeSnapshot, RuntimeTransportState};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamRange {
    pub channels: u16,
    pub minimum_sample_rate: u32,
    pub maximum_sample_rate: u32,
    pub sample_format: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AudioDeviceInfo {
    pub name: String,
    pub is_default_input: bool,
    pub is_default_output: bool,
    pub default_input: Option<String>,
    pub default_output: Option<String>,
    pub input_ranges: Vec<StreamRange>,
    pub output_ranges: Vec<StreamRange>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AudioHostInfo {
    pub id: String,
    pub devices: Vec<AudioDeviceInfo>,
}

/// Enumerates every audio host and its currently visible devices.
///
/// # Errors
///
/// Returns an error when a host cannot initialize or its devices cannot be
/// enumerated or described.
pub fn enumerate_audio() -> Result<Vec<AudioHostInfo>> {
    cpal::available_hosts()
        .into_iter()
        .filter(|host_id| matches!(host_id.name(), "ALSA" | "PulseAudio" | "PipeWire"))
        .map(|host_id| {
            let host = cpal::host_from_id(host_id)
                .with_context(|| format!("failed to initialize audio host {host_id}"))?;
            let default_input = host
                .default_input_device()
                .and_then(|device| device.description().ok())
                .map(|description| description.name().to_owned());
            let default_output = host
                .default_output_device()
                .and_then(|device| device.description().ok())
                .map(|description| description.name().to_owned());
            let devices = host
                .devices()
                .context("failed to enumerate audio devices")?;

            let devices = devices
                .map(|device| {
                    inspect_device(&device, default_input.as_deref(), default_output.as_deref())
                })
                .collect::<Result<Vec<_>>>()?;

            Ok(AudioHostInfo {
                id: format!("{host_id:?}"),
                devices,
            })
        })
        .collect()
}

/// Enumerates devices exposed by the `PulseAudio` compatibility layer used by
/// `PipeWire` without probing unrelated ALSA or JACK devices.
///
/// # Errors
///
/// Returns an error when the PulseAudio/PipeWire host is unavailable or its
/// devices cannot be enumerated.
pub fn enumerate_pipewire_devices() -> Result<Vec<AudioDeviceInfo>> {
    let host_id = cpal::available_hosts()
        .into_iter()
        .find(|host_id| host_id.name() == "PulseAudio")
        .context("PulseAudio/PipeWire audio host is unavailable")?;
    let host = cpal::host_from_id(host_id).context("failed to initialize PipeWire audio")?;
    let default_input = host
        .default_input_device()
        .and_then(|device| device.description().ok())
        .map(|description| description.name().to_owned());
    let default_output = host
        .default_output_device()
        .and_then(|device| device.description().ok())
        .map(|description| description.name().to_owned());
    host.devices()
        .context("failed to enumerate PipeWire audio devices")?
        .map(|device| inspect_device(&device, default_input.as_deref(), default_output.as_deref()))
        .collect()
}

fn inspect_device(
    device: &cpal::Device,
    default_input_name: Option<&str>,
    default_output_name: Option<&str>,
) -> Result<AudioDeviceInfo> {
    let name = device
        .description()
        .context("audio device has no readable description")?
        .name()
        .to_owned();
    let default_input = device
        .default_input_config()
        .ok()
        .map(|config| format_config(&config));
    let default_output = device
        .default_output_config()
        .ok()
        .map(|config| format_config(&config));
    let input_ranges = device
        .supported_input_configs()
        .map(|ranges| ranges.map(map_range).collect())
        .unwrap_or_default();
    let output_ranges = device
        .supported_output_configs()
        .map(|ranges| ranges.map(map_range).collect())
        .unwrap_or_default();

    Ok(AudioDeviceInfo {
        is_default_input: default_input_name == Some(name.as_str()),
        is_default_output: default_output_name == Some(name.as_str()),
        name,
        default_input,
        default_output,
        input_ranges,
        output_ranges,
    })
}

fn map_range(range: cpal::SupportedStreamConfigRange) -> StreamRange {
    StreamRange {
        channels: range.channels(),
        minimum_sample_rate: range.min_sample_rate(),
        maximum_sample_rate: range.max_sample_rate(),
        sample_format: range.sample_format().to_string(),
    }
}

fn format_config(config: &cpal::SupportedStreamConfig) -> String {
    format!(
        "{} ch, {} Hz, {}",
        config.channels(),
        config.sample_rate(),
        config.sample_format()
    )
}
