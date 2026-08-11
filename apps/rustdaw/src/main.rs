mod piano_roll;
mod theme;
mod ui;

use anyhow::Result;
use ui::RustDawApp;

fn main() -> Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("RustDAW")
            .with_inner_size([1440.0, 880.0])
            .with_min_inner_size([1024.0, 640.0])
            .with_icon(app_icon()),
        ..Default::default()
    };

    eframe::run_native(
        "RustDAW",
        native_options,
        Box::new(move |context| {
            theme::install(&context.egui_ctx);
            Ok(Box::new(RustDawApp::new()))
        }),
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn app_icon() -> eframe::egui::IconData {
    const SIZE: u32 = 128;
    let mut rgba = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    let waveform = [
        (20.0, 66.0),
        (33.0, 66.0),
        (40.0, 40.0),
        (51.0, 91.0),
        (62.0, 52.0),
        (73.0, 80.0),
        (84.0, 61.0),
        (94.0, 66.0),
        (108.0, 66.0),
    ];
    for y in 0..SIZE {
        for x in 0..SIZE {
            let xf = x as f32;
            let yf = y as f32;
            let in_panel = (13..=114).contains(&x) && (18..=109).contains(&y);
            let mut color = if in_panel {
                [31_u8, 35, 39, 255]
            } else {
                [22_u8, 25, 28, 255]
            };
            if waveform
                .windows(2)
                .any(|line| point_segment_distance((xf, yf), line[0], line[1]) <= 3.0)
            {
                let blend = xf / SIZE as f32;
                color = [
                    (54.0 + 20.0 * blend) as u8,
                    (142.0 + 54.0 * blend) as u8,
                    (203.0 - 91.0 * blend) as u8,
                    255,
                ];
            }
            for (center, light) in [
                ((25.0, 29.0), [218, 72, 72, 255]),
                ((37.0, 29.0), [226, 183, 63, 255]),
                ((49.0, 29.0), [74, 196, 112, 255]),
            ] {
                if (xf - center.0).powi(2) + (yf - center.1).powi(2) <= 3.5_f32.powi(2) {
                    color = light;
                }
            }
            rgba.extend_from_slice(&color);
        }
    }
    eframe::egui::IconData {
        rgba,
        width: SIZE,
        height: SIZE,
    }
}

fn point_segment_distance(point: (f32, f32), start: (f32, f32), end: (f32, f32)) -> f32 {
    let segment = (end.0 - start.0, end.1 - start.1);
    let length_squared = segment.0.mul_add(segment.0, segment.1 * segment.1);
    let projection = (((point.0 - start.0) * segment.0 + (point.1 - start.1) * segment.1)
        / length_squared)
        .clamp(0.0, 1.0);
    let nearest = (
        start.0 + projection * segment.0,
        start.1 + projection * segment.1,
    );
    (point.0 - nearest.0).hypot(point.1 - nearest.1)
}
