"""Shared canvas and neutrals for the schematic figures in figure/.

A Canvas is a matplotlib axes in pixel units (y pointing down) at dpi=100, so coordinates read
like a drawing tool's. Data colors come from viz_config; the neutrals here are for text, panels
and cards.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath
from PIL import Image

INK = "#2b2326"
TAUPE = "#857a74"
STONE = "#f5f2ee"
PANEL_EDGE = "#dcd4cc"
CARD_EDGE = "#e3dbd3"
CHIP = "#efeae4"

SANS = "Liberation Sans"
MONO = "DejaVu Sans Mono"
DPI = 100
PILL_PAD, PILL_SCALE = 7, 0.86


def tint(color, amount):
    """Mix `color` with white; `amount` is the share of the color kept."""
    return to_hex([1 - amount * (1 - c) for c in to_rgb(color)])


def shade(color, amount):
    """Mix `color` with black; `amount` is the share of the color kept."""
    return to_hex([amount * c for c in to_rgb(color)])


def pt(px):
    return px * 72 / DPI


class Canvas:
    def __init__(self, width, height):
        plt.rcParams.update({
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": SANS,
            "mathtext.fontset": "custom",
            "mathtext.rm": SANS,
            "mathtext.it": f"{SANS}:italic",
            "mathtext.bf": f"{SANS}:bold",
            "mathtext.fallback": "stixsans",
        })
        self.fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)
        self.ax.axis("off")
        self.renderer = self.fig.canvas.get_renderer()

    def box(self, x0, y0, x1, y1, fc="white", ec=CARD_EDGE, lw=1.6, ls="-", r=14, z=1):
        self.ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                         boxstyle=f"round,pad=0,rounding_size={r}",
                                         fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))

    def text(self, x, y, s, size=26, color=INK, family=SANS, weight="normal", style="normal",
             ha="left", va="baseline", ls=1.3, z=5):
        # one artist per line so line 1 sits at y and later lines go down
        for i, line in enumerate(s.split("\n")):
            t = self.ax.text(x, y + i * size * ls, line, fontsize=pt(size), color=color,
                             family=family, weight=weight, style=style, ha=ha, va=va, zorder=z)
        return t

    def measure(self, s, size, weight="normal", style="normal", family=SANS):
        prop = FontProperties(family=family, size=pt(size), weight=weight, style=style)
        return self.renderer.get_text_width_height_descent(s, prop, ismath=False)[0]

    def wrap(self, runs, width, size, max_lines=None):
        """Greedy word wrap of styled runs [(text, style)]; returns lines of (word, style, w)."""
        words = [(w, st) for s, st in runs for w in s.split(" ") if w]
        space = self.measure("a a", size) - self.measure("aa", size)
        lines, cur, cur_w = [], [], 0.0
        for word, st in words:
            if st.get("pill"):
                w = self.measure(word, size * PILL_SCALE, "bold") + 2 * PILL_PAD
            else:
                w = self.measure(word, size, st.get("weight", "normal"), st.get("style", "normal"),
                                 st.get("family", SANS))
            if cur and cur_w + space + w > width:
                lines.append(cur)
                cur, cur_w = [], 0.0
            cur_w += w + (space if cur else 0)
            cur.append((word, st, w))
        lines.append(cur)
        if max_lines and len(lines) > max_lines:
            lines = lines[:max_lines]
            word, st, _ = lines[-1][-1]
            word = word.rstrip(",.;:") + "…"
            lines[-1][-1] = (word, st, self.measure(word, size, st.get("weight", "normal"),
                                                    st.get("style", "normal")))
        return lines, space

    def rich(self, x, y, width, runs, size=16, ls=1.3, color=INK, align="left", max_lines=None,
             z=5):
        """Draw wrapped styled runs. Style keys: weight, style, color, family, ul (underline color),
        pill (fill color: the word becomes a white bold tag). Returns the number of lines drawn."""
        lines, space = self.wrap(runs, width, size, max_lines)
        for i, line in enumerate(lines):
            yy = y + i * size * ls
            line_w = sum(w for _, _, w in line) + space * (len(line) - 1)
            xx = x if align == "left" else x - line_w / 2
            for j, (word, st, w) in enumerate(line):
                if st.get("pill"):
                    h = size * 1.08
                    self.box(xx, yy - size * 0.34 - h / 2, xx + w, yy - size * 0.34 + h / 2,
                             fc=st["pill"], ec=st["pill"], r=h / 2, z=z - 0.5)
                    self.ax.text(xx + w / 2, yy - size * 0.34, word, fontsize=pt(size * PILL_SCALE),
                                 color="white", weight="bold", ha="center", va="center_baseline",
                                 zorder=z)
                    xx += w + space
                    continue
                self.ax.text(xx, yy, word, fontsize=pt(size), color=st.get("color", color),
                             family=st.get("family", SANS), weight=st.get("weight", "normal"),
                             style=st.get("style", "normal"), va="baseline", zorder=z)
                if st.get("ul"):
                    joined = j > 0 and line[j - 1][1].get("ul")
                    x0 = xx - (space if joined else 0)
                    self.ax.plot([x0, xx + w], [yy + size * 0.2] * 2, color=st["ul"], lw=1.5,
                                 solid_capstyle="butt", zorder=z)
                xx += w + space
        return len(lines)

    def n_lines(self, runs, width, size, max_lines=None):
        return len(self.wrap(runs, width, size, max_lines)[0])

    def arrow(self, points, color=INK, lw=1.8, ls="-", head=10, z=4):
        """Polyline through `points` with an arrowhead at the last point."""
        (x0, y0), (x1, y1) = points[-2], points[-1]
        if len(points) > 2 or ls != "-":
            xs, ys = zip(*points)
            self.ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=z)
            # solid stub carrying the head, so dashes never cut it
            d = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            stub = min(d, head + 2)
            x0, y0 = x1 - (x1 - x0) * stub / d, y1 - (y1 - y0) * stub / d
        self.ax.add_patch(FancyArrowPatch(
            path=MplPath([(x0, y0), (x1, y1)]),
            arrowstyle=f"-|>,head_length={head},head_width={head * 0.6}",
            color=color, lw=lw, zorder=z, shrinkA=0, shrinkB=0))

    def badge(self, cx, cy, w, h, s, fc, size=30, color="white", r=12):
        self.box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, fc=fc, ec=fc, r=r, z=3)
        self.text(cx, cy, s, size=size, color=color, weight="bold", ha="center",
                  va="center_baseline")

    def save(self, stem):
        """Write stem.{pdf,svg,png} plus stem-gray.png for the grayscale check."""
        stem = Path(stem)
        for ext in ("pdf", "svg", "png"):
            self.fig.savefig(f"{stem}.{ext}", dpi=DPI * (2 if ext == "png" else 1))
        Image.open(f"{stem}.png").convert("L").save(f"{stem}-gray.png")
