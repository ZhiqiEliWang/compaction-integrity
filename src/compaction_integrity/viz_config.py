import matplotlib.pyplot as plt
import seaborn as sns

PAPER_STYLE_CONFIGS = {
    "usenix": {
        "width_inch": 3.3,
        "font_size": 10,
        "axes_label_size": 10,
        "axes_title_size": 10,
        "tick_label_size": 8,
        "legend_font_size": 8,
    },
    "acl": {
        "width_inch": 7.7 / 2.54,
        "font_size": 11,
        "axes_label_size": 11,
        "axes_title_size": 11,
        "tick_label_size": 10,
        "legend_font_size": 10,
    },
}


def set_paper_style(use_latex=True, venue="acl"):
    """
    Sets the plot style to match conference paper layouts.
    Call this function at the top of your scripts.
    """
    sns.set_theme(style="whitegrid", context="paper")

    style_config = PAPER_STYLE_CONFIGS[venue]
    width_inch = style_config["width_inch"]
    height_inch = width_inch / 1.618

    params = {
        # --- LaTeX & Font Integration ---
        'text.usetex': use_latex,
        'font.family': 'serif',         # Match paper font
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'], 
        'mathtext.fontset': 'stix',
        
        # --- Font Sizes ---
        'font.size': style_config["font_size"],
        'axes.labelsize': style_config["axes_label_size"],
        'axes.titlesize': style_config["axes_title_size"],
        'xtick.labelsize': style_config["tick_label_size"],
        'ytick.labelsize': style_config["tick_label_size"],
        'legend.fontsize': style_config["legend_font_size"],

        # --- Figure Layout ---
        'figure.figsize': [width_inch, height_inch],
        'figure.constrained_layout.use': True,
        # --- Line Styles ---
        'lines.linewidth': 1.5,         # Thicker lines for visibility
        'lines.markersize': 4,
        'grid.alpha': 0.3,              # Light grid lines
    }

    sns.set_palette("colorblind")
    
    
    plt.rcParams.update(params)

def set_talk_style(use_latex=True):
    """
    Sets the plot style to match talk style.
    """
    sns.set_theme(style="whitegrid", context="talk")
    sns.set_palette("colorblind")
    params = {
        'text.usetex': use_latex,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'stix',
    }
    plt.rcParams.update(params)

def save_fig(filename):
    """
    Standardized saving helper. 
    Ensures vector format (PDF) and removes whitespace.
    """
        
    plt.savefig(
        filename, 
        format='pdf', 
        bbox_inches='tight', 
        pad_inches=0.02,     
    )
    print(f"Saved figure: {filename}")
