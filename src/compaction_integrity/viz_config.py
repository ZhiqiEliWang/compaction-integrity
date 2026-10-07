import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

# Categorical: fixed order, each color paired with a linestyle so series survive grayscale print.
# 7th (indigo) is >=20 dE from all six under deutan/protan/tritan simulation and far from its neighbors.
PALETTE = ["#97374f", "#3577ab", "#c2892c", "#328053", "#8463a8", "#c66843", "#4840a0"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1)), (0, (8, 2)), (0, (4, 1, 4, 1, 1, 1))]
TAUPE = "#7a716b"
# Fixed compactor -> color (raw names) so a compactor keeps one color in every figure.
# Assigned so neighbors in the canonical bar order and the three LLM compactors stay
# CVD-separable; recent_5 is the heuristic baseline, so it gets the neutral.
COMPACTOR_COLORS = {
    "recent_5": TAUPE,
    "llmlingua2_t500": PALETTE[1],
    "gpt_oss_120b_anthropic_prompt": PALETTE[2],
    "gpt_oss_120b_anthropic_sc_targeted_prompt": PALETTE[6],
    "gpt_oss_120b_pi_mono_prompt": PALETTE[0],
    "qwen30b_anthropic_prompt": PALETTE[4],
    "qwen30b_anthropic_sc_targeted_prompt": PALETTE[5],
    "gemma_4_anthropic_prompt": PALETTE[3],
}
# Scatter: only the first three are pairwise colorblind-safe; >3 groups -> split panels.
SCATTER_COLORS = PALETTE[:3]
SCATTER_MARKERS = ["o", "s", "^"]

# Sequential (magnitude): blush -> oxblood, monotone lightness.
WINE_STOPS = [
    "#fbf0f2", "#f4d5da", "#ebb8c0", "#dd98a5", "#cc7689",
    "#b8556e", "#9a3a56", "#76283f", "#521a2b",
]
# Diverging (signed): slate <- warm gray -> wine; slate stops are lightness-matched to the wine side.
SLATE_WINE_STOPS = [
    "#07304c", "#1c608f", "#6893be", "#b3c7e0",
    "#f2f0ec",
    "#ebb8c0", "#cc7689", "#9a3a56", "#521a2b",
]
for _name, _stops in [("wine", WINE_STOPS), ("slate_wine", SLATE_WINE_STOPS)]:
    if _name not in mpl.colormaps:
        mpl.colormaps.register(LinearSegmentedColormap.from_list(_name, _stops))

# Shared heatmap colormaps. sns.heatmap ignores rcParams["image.cmap"], so pass these explicitly.
# Diverging maps lose the sign in grayscale: center them at 0 and label the colorbar ends.
HEATMAP_CMAP = "wine"
HEATMAP_DIVERGING_CMAP = "slate_wine"

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
    """Set the plot style for conference-paper figures; call before plotting."""
    sns.set_theme(style="whitegrid", context="paper")

    style_config = PAPER_STYLE_CONFIGS[venue]
    width_inch = style_config["width_inch"]
    height_inch = width_inch / 1.618

    params = {
        'text.usetex': use_latex,
        'font.family': 'serif',         # Match paper font
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'], 
        'mathtext.fontset': 'stix',

        'font.size': style_config["font_size"],
        'axes.labelsize': style_config["axes_label_size"],
        'axes.titlesize': style_config["axes_title_size"],
        'xtick.labelsize': style_config["tick_label_size"],
        'ytick.labelsize': style_config["tick_label_size"],
        'legend.fontsize': style_config["legend_font_size"],

        'figure.figsize': [width_inch, height_inch],
        'figure.constrained_layout.use': True,
        'lines.linewidth': 1.6,         # Ochre is faint at thin widths
        'lines.markersize': 4,
        'grid.alpha': 0.3,
        'image.cmap': HEATMAP_CMAP,
    }

    sns.set_palette(PALETTE)
    params['axes.prop_cycle'] = cycler(color=PALETTE, linestyle=LINESTYLES)

    plt.rcParams.update(params)

def set_talk_style(use_latex=True):
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
    """Save the current figure as a tightly cropped vector PDF."""
        
    plt.savefig(
        filename, 
        format='pdf', 
        bbox_inches='tight', 
        pad_inches=0.02,     
    )
    print(f"Saved figure: {filename}")
