MODALITIES = [
    "General X-ray", 
    "Mammography", 
    "Fluoroscopy",
    "CT"
]

PLOT_STYLES = [
    'ggplot2', 
    'seaborn', 
    'simple_white', 
    'plotly',
    'plotly_white', 
    'plotly_dark', 
    'presentation'
]

PLOT_COLOURS = [
    "royalblue", 
    "orange", 
    "lime", 
    "magenta", 
    "cyan", 
    "gold", 
    "crimson", 
    "deeppink", 
    "grey"
]

# --- helper: named colour -> rgba (only need the names you use) ---
_CSS_COLOR_RGB = {
    "royalblue": (65, 105, 225),
    "orange": (255, 165, 0),
    "lime": (0, 255, 0),
    "magenta": (255, 0, 255),
    "cyan": (0, 255, 255),
    "gold": (255, 215, 0),
    "crimson": (220, 20, 60),
    "deeppink": (255, 20, 147),
    "grey": (128, 128, 128),
}
def rgba(name: str, alpha: float) -> str:
    r, g, b = _CSS_COLOR_RGB.get(str(name).lower(), (0, 0, 0))
    return f"rgba({r},{g},{b},{alpha})"