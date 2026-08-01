import inspect

import gradio as gr

ACCENT = "#22C55E"
ACCENT_HOVER = "#16A34A"
ACCENT_INK = "#04140B"

INK_0 = "#0B1220"
INK_1 = "#0F172A"
INK_2 = "#111A2E"
INK_3 = "#1A2438"
INK_CONSOLE = "#060B14"

FG = "#E2E8F0"
FG_MUTED = "#94A3B8"
FG_FAINT = "#64748B"

BORDER = "rgba(148, 163, 184, 0.16)"
BORDER_STRONG = "rgba(148, 163, 184, 0.28)"


def build_theme() -> gr.themes.Base:
    tokens: dict[str, str] = {
        "color_accent": ACCENT,
        "color_accent_soft": "rgba(34, 197, 94, 0.16)",
        "body_background_fill": INK_0,
        "background_fill_primary": INK_1,
        "background_fill_secondary": INK_0,
        "body_text_color": FG,
        "block_background_fill": INK_1,
        "block_border_color": BORDER,
        "block_label_text_color": FG_MUTED,
        "block_title_text_color": FG,
        "panel_background_fill": INK_1,
        "panel_border_color": BORDER,
        "input_background_fill": INK_2,
        "input_background_fill_hover": INK_2,
        "input_background_fill_focus": INK_2,
        "input_border_color": BORDER_STRONG,
        "input_border_color_hover": BORDER_STRONG,
        "input_border_color_focus": ACCENT,
        "input_placeholder_color": FG_FAINT,
        "border_color_primary": BORDER_STRONG,
        "border_color_accent": ACCENT,
        "border_color_accent_subdued": "rgba(34, 197, 94, 0.32)",
        "shadow_drop": "0 1px 2px rgba(0, 0, 0, 0.4)",
        "shadow_drop_lg": "0 8px 24px rgba(0, 0, 0, 0.45)",
        "button_primary_background_fill": ACCENT,
        "button_primary_background_fill_hover": ACCENT_HOVER,
        "button_primary_text_color": ACCENT_INK,
        "button_primary_text_color_hover": ACCENT_INK,
        "button_primary_border_color": ACCENT,
        "button_primary_border_color_hover": ACCENT_HOVER,
        "button_secondary_background_fill": INK_2,
        "button_secondary_background_fill_hover": INK_3,
        "button_secondary_text_color": FG_MUTED,
        "button_secondary_text_color_hover": FG,
        "button_secondary_border_color": BORDER_STRONG,
        "button_secondary_border_color_hover": BORDER_STRONG,
        "button_cancel_background_fill": "#7F1D1D",
        "button_cancel_background_fill_hover": "#991B1B",
        "button_cancel_text_color": "#FECACA",
        "button_cancel_text_color_hover": "#FECACA",
        "button_cancel_border_color": "#7F1D1D",
        "button_cancel_border_color_hover": "#991B1B",
        "checkbox_label_background_fill": INK_2,
        "checkbox_label_background_fill_hover": INK_3,
        "checkbox_label_background_fill_selected": ACCENT,
        "checkbox_label_text_color": FG,
        "checkbox_label_text_color_selected": ACCENT_INK,
        "checkbox_label_border_color": BORDER_STRONG,
        "checkbox_label_border_color_hover": BORDER_STRONG,
        "checkbox_label_border_color_selected": ACCENT,
        "checkbox_background_color": INK_2,
        "checkbox_background_color_selected": ACCENT,
        "checkbox_border_color": BORDER_STRONG,
        "checkbox_border_color_selected": ACCENT,
        "checkbox_check": f"radial-gradient(circle, {ACCENT_INK} 35%, transparent 36%)",
        "error_background_fill": "#1C1417",
        "error_border_color": "#7F1D1D",
        "error_text_color": "#FECACA",
        "loader_color": ACCENT,
        "slider_color": ACCENT,
        "table_border_color": BORDER,
        "table_odd_background_fill": INK_1,
        "table_even_background_fill": INK_2,
        "table_text_color": FG,
    }

    merged: dict[str, str | None] = dict(tokens)
    _set_params = set(inspect.signature(gr.themes.Base.set).parameters)
    for name, value in tokens.items():
        if f"{name}_dark" in _set_params:
            merged[f"{name}_dark"] = value

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.green,
        neutral_hue=gr.themes.colors.slate,
        font=[
            gr.themes.GoogleFont("Inter"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ],
        font_mono=[
            gr.themes.GoogleFont("JetBrains Mono"),
            "ui-monospace",
            "SFMono-Regular",
            "Menlo",
            "monospace",
        ],
    )
    return theme.set(**merged)


theme = build_theme()
