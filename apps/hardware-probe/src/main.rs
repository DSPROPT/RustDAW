use anyhow::Result;
use daw_audio_linux::{AudioDeviceInfo, StreamRange, enumerate_audio};

fn main() -> Result<()> {
    println!("RustDAW hardware probe (read-only)\n");

    for host in enumerate_audio()? {
        println!("Host: {}", host.id);
        if host.devices.is_empty() {
            println!("  No devices found");
        }
        for device in &host.devices {
            print_device(device);
        }
    }

    Ok(())
}

fn print_device(device: &AudioDeviceInfo) {
    let mut defaults = Vec::with_capacity(2);
    if device.is_default_input {
        defaults.push("default input");
    }
    if device.is_default_output {
        defaults.push("default output");
    }
    let suffix = if defaults.is_empty() {
        String::new()
    } else {
        format!(" [{}]", defaults.join(", "))
    };

    println!("  {}{}", device.name, suffix);
    if let Some(config) = &device.default_input {
        println!("    Input default:  {config}");
    }
    let input_ranges = practical_ranges(&device.input_ranges);
    for range in input_ranges.iter().take(12) {
        print_range("input ", range);
    }
    print_omitted(input_ranges.len());
    if let Some(config) = &device.default_output {
        println!("    Output default: {config}");
    }
    let output_ranges = practical_ranges(&device.output_ranges);
    for range in output_ranges.iter().take(12) {
        print_range("output", range);
    }
    print_omitted(output_ranges.len());

    if device.default_input.is_none()
        && device.default_output.is_none()
        && input_ranges.is_empty()
        && output_ranges.is_empty()
    {
        println!("    No stream formats available (the hardware may be busy)");
    }
}

fn practical_ranges(ranges: &[StreamRange]) -> Vec<&StreamRange> {
    ranges
        .iter()
        .filter(|range| {
            range.channels <= 8
                && range.minimum_sample_rate >= 8_000
                && range.maximum_sample_rate <= 384_000
        })
        .collect()
}

fn print_range(direction: &str, range: &StreamRange) {
    println!(
        "      {direction}: {} ch, {}–{} Hz, {}",
        range.channels, range.minimum_sample_rate, range.maximum_sample_rate, range.sample_format
    );
}

fn print_omitted(range_count: usize) {
    if range_count > 12 {
        println!("      … {} additional practical formats", range_count - 12);
    }
}
