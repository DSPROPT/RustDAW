use eframe::egui::{
    Color32, Context, CornerRadius, FontFamily, FontId, Stroke, TextStyle, Visuals,
};

pub const BG: Color32 = Color32::from_rgb(22, 25, 28);
pub const PANEL: Color32 = Color32::from_rgb(31, 35, 39);
pub const PANEL_2: Color32 = Color32::from_rgb(40, 45, 50);
pub const BORDER: Color32 = Color32::from_rgb(67, 73, 79);
pub const TEXT: Color32 = Color32::from_rgb(220, 224, 228);
pub const MUTED: Color32 = Color32::from_rgb(139, 147, 154);
pub const BLUE: Color32 = Color32::from_rgb(54, 142, 203);
pub const BLUE_DARK: Color32 = Color32::from_rgb(27, 78, 112);
pub const GREEN: Color32 = Color32::from_rgb(74, 196, 112);
pub const YELLOW: Color32 = Color32::from_rgb(226, 183, 63);
pub const RED: Color32 = Color32::from_rgb(218, 72, 72);

pub fn install(context: &Context) {
    let mut style = (*context.style()).clone();
    style.text_styles.insert(
        TextStyle::Heading,
        FontId::new(18.0, FontFamily::Proportional),
    );
    style
        .text_styles
        .insert(TextStyle::Body, FontId::new(13.0, FontFamily::Proportional));
    style.text_styles.insert(
        TextStyle::Button,
        FontId::new(12.0, FontFamily::Proportional),
    );
    style.spacing.item_spacing = eframe::egui::vec2(7.0, 7.0);
    style.spacing.button_padding = eframe::egui::vec2(9.0, 5.0);

    let mut visuals = Visuals::dark();
    visuals.panel_fill = PANEL;
    visuals.window_fill = PANEL;
    visuals.extreme_bg_color = BG;
    visuals.faint_bg_color = PANEL_2;
    visuals.override_text_color = Some(TEXT);
    visuals.selection.bg_fill = BLUE_DARK;
    visuals.widgets.noninteractive.bg_fill = PANEL;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, BORDER);
    visuals.widgets.noninteractive.corner_radius = CornerRadius::same(2);
    visuals.widgets.inactive.bg_fill = PANEL_2;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, BORDER);
    visuals.widgets.inactive.corner_radius = CornerRadius::same(2);
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(53, 60, 66);
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, BLUE);
    visuals.widgets.active.bg_fill = BLUE_DARK;
    visuals.widgets.active.bg_stroke = Stroke::new(1.0_f32, BLUE);
    visuals.window_corner_radius = CornerRadius::same(3);
    style.visuals = visuals;
    context.set_style(style);
}
