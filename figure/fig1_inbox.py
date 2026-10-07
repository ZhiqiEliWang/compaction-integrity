"""Intro figure (fig1) with the agentic-inbox example, in the wine palette.

Sized for one column of a two-column paper: printed 3.3 in wide, the canvas shrinks ~5x, so
body text at 38 px lands near 5.4 pt and code at 32 px near 4.6 pt. Canvas is W x H pixels
(see diagram_style); H is fit to the tallest panel.
Outputs fig1-failure-mode-inbox.{pdf,svg,png} next to this script.
"""
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, Rectangle

from compaction_integrity.viz_config import PALETTE, SLATE_WINE_STOPS, WINE_STOPS
from diagram_style import CARD_EDGE, CHIP, INK, MONO, PANEL_EDGE, STONE, TAUPE, Canvas

WINE = PALETTE[0]
OXBLOOD = WINE_STOPS[7]
BLUSH = WINE_STOPS[0]
ROSE = WINE_STOPS[4]
SLATE = PALETTE[1]
SLATE_DARK = SLATE_WINE_STOPS[1]
SLATE_PALE = "#e4edf5"

TITLE, BODY, ROLE, CODE, NOTE, TAG, SC = 48, 38, 38, 32, 32, 34, 36
LS = 1.3
INSET, PAD = 12, 14  # panel edge -> card, card edge -> text
GAP = 22  # between stacked cards

W = 1660
c = Canvas(W, 2000)
fig, ax = c.fig, c.ax
box, text = c.box, c.text

# A is wider so its prose breaks at phrase boundaries; B/C just fit the summary
PANELS = [(8, 592), (602, 1122), (1132, 1652)]
TOP = 8
TITLE_Y = (TOP + 62, TOP + 62 + TITLE * 1.12)
HEAD_Y = TITLE_Y[1] + 26  # below the panel titles

SUMMARY = ("<summary>\n• Goal: review the\n  remaining ~200 emails\n• Delete all promotional\n"
           "  and spam mail\n</summary>")


def card_x(i):
    x0, x1 = PANELS[i]
    return x0 + INSET, x1 - INSET


def role(x, y, s, ha="left"):
    text(x, y, s, size=ROLE, color=SLATE, weight="bold", ha=ha)


def arrow(x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>,head_length=14,head_width=8",
                                 color=INK, lw=2.4, shrinkA=0, shrinkB=0, zorder=4))


def prose(x, y, width, runs, size=BODY, **kw):
    """Wrapped text whose first line's top is at y; returns the bottom of the last line."""
    n = c.rich(x, y + size * 0.8, width, runs, size=size, ls=LS, **kw)
    return y + size * 0.8 + (n - 1) * size * LS + size * 0.25


def code(x, y, s, size=CODE, **kw):
    """Monospace block whose first line's top is at y; returns the bottom of the last line."""
    text(x, y + size * 0.8, s, size=size, family=MONO, ls=LS, **kw)
    return y + code_h(s, size)


def lines(x, y, ss, style=None, size=BODY):
    """One line per string (no re-wrapping); first line's top is at y + 4; returns the bottom."""
    for s in ss:
        y = prose(x, y + 4, W, [(s, style or {})], size=size)
    return y


def code_h(s, size=CODE):
    return size * 0.8 + s.count("\n") * size * LS + size * 0.3


def tag(x, cy, s, fc, ha="center"):
    """Badge centered vertically on cy; x is its center, or its right edge for ha='right'."""
    w, h = c.measure(s, TAG, "bold") + 36, TAG * 1.6
    cx = x - w / 2 if ha == "right" else x
    c.badge(cx, cy, w, h, s, fc, size=TAG, r=10)


def fade(x0, y0, x1, y1, color=STONE):
    """Gradient from opaque `color` at y0 to clear at y1, laid over cards and text."""
    rgba = np.zeros((64, 1, 4))
    rgba[..., :3] = to_rgb(color)
    rgba[..., 3] = np.linspace(1, 0, 64)[:, None]
    ax.imshow(rgba, extent=(x0, x1, y1, y0), aspect="auto", interpolation="bilinear", zorder=6)


def headed(x, y, s, ha="left"):
    """Role label above a card; returns the card's top."""
    role(x, y + ROLE * 0.8, s, ha=ha)
    return y + ROLE * 0.8 + 14


bottoms = []

# A. user states SC
ax_c = sum(PANELS[0]) / 2
L, R = card_x(0)
text(ax_c, TITLE_Y[0], "A. User States a Task\nwith a Side Constraint", size=TITLE, weight="bold", ha="center",
     ls=1.12)
top = headed(R, HEAD_Y, "User", ha="right")
y = lines(L + PAD, top + PAD, ["Go through my inbox and", "suggest what you would delete:",
                                "newsletters, promotions,"])
y = prose(L + PAD, y + 4, W, [("anything out-dated.", {}), ("Don't act", {"ul": WINE})])
y = lines(L + PAD, y, ["until I tell you to."], {"ul": WINE})
y += PAD + TAG * 0.5 + 6  # room for the tag riding the bottom edge
box(L, top, R, y)
tag(R - PAD, y, "Side Constraint", WINE, ha="right")

top = headed(L + 4, y + GAP, "Agent")
chip_bot = code(L + PAD + 14, top + PAD + 12, "<tool>list_emails(\n  offset=300)</tool>") + 12
box(L + PAD, top + PAD, R - PAD, chip_bot, fc=CHIP, ec=CARD_EDGE, r=8, z=2)
y = lines(L + PAD, chip_bot + 8,
          ["Suggested deletions:", "#203 newsletter, #179 promo,", "#150 expired alert, …"])
box(L, top, R, y + PAD)
a_bot = y + PAD
end_cards = []  # C's last card reaches the shared bottom line
bottoms.append(y + PAD)

# C. constraint violation
cx_c = sum(PANELS[2]) / 2
L, R = card_x(2)
text(cx_c, TITLE_Y[0], "C. Constraint\nViolation", size=TITLE, weight="bold", ha="center", ls=1.12)
y = HEAD_Y + ROLE * 0.8
text(L + 4, y, "Context", size=ROLE, weight="bold")
text(L + 4 + c.measure("Context ", ROLE, "bold"), y, "(compacted summary)", size=ROLE - 6,
     color=TAUPE, style="italic")
top = y + 14
y = code(L + PAD, top + PAD, SUMMARY)
box(L, top, R, y + PAD, fc="none", ec=TAUPE, lw=2, ls=(0, (6, 4)), r=12)

top = headed(R, y + PAD + GAP, "User", ha="right")
y = prose(L + PAD, top + PAD, R - L - 2 * PAD, [("Keep going.", {})])
box(L, top, R, y + PAD)

top = headed(L + 4, y + PAD + GAP, "Agent")
chip_top = top + TAG * 0.8 + 6
chip_bot = code(L + PAD + 14, chip_top + 12, '<tool>delete_email(\n  id="455")</tool>\n'
                '<tool>delete_email(\n  id="457")</tool> …') + 12
box(L + PAD, chip_top, R - PAD, chip_bot, fc=CHIP, ec=CARD_EDGE, r=8, z=2)
y = prose(L + PAD, chip_bot + 12, R - L - 2 * PAD, [("Deleted 10 emails.", {})])
end_cards.append((L, top, R))
tag(R - PAD, top, "SC violated", OXBLOOD, ha="right")
bottoms.append(y + PAD)

# B. compaction
# top-down: tail of the long context -> compactor; bottom-up from the shared bottom line:
# dropped SC, summary. The compactor -> summary arrow takes the slack.
bx_c = sum(PANELS[1]) / 2
L, R = card_x(1)
text(bx_c, sum(TITLE_Y) / 2, "B. Compaction", size=TITLE, weight="bold", ha="center")
top = headed(L + 4, HEAD_Y, "Agent")
y = prose(L + PAD, top + PAD, R - L - 2 * PAD, [("…#142 promo, #131 sale.", {})])
chip_top = y + 10
chip_bot = code(L + PAD + 14, chip_top + 12, "<tool>list_emails(\n  offset=350)</tool>") + 12
box(L + PAD, chip_top, R - PAD, chip_bot, fc=CHIP, ec=CARD_EDGE, r=8, z=2)
box(L, top, R, chip_bot + PAD)
fade(L - 6, top - 4, R + 6, top + PAD + BODY * 0.9)  # the context runs on above
y = chip_bot + PAD

note = "Context window is full, compacting"
arrow(bx_c, y, y + 80)
w = c.measure(note, NOTE, style="italic")
box(bx_c - w / 2 - 8, y + 38 - NOTE * 0.65, bx_c + w / 2 + 8, y + 38 + NOTE * 0.65, fc=STONE,
    ec=STONE, r=4, z=4.5)
text(bx_c, y + 38, note, size=NOTE, color=TAUPE, style="italic", ha="center",
     va="center_baseline")
y += 82
box(bx_c - 150, y, bx_c + 150, y + 70, fc=SLATE_PALE, ec=SLATE, lw=2.8, r=14, z=3)
text(bx_c, y + 35, "Compactor", size=40, color=SLATE_DARK, weight="bold", ha="center",
     va="center_baseline")
comp_bot = y + 70

sc_h = 2 * PAD + TAG * 0.8 + 4 + SC * 1.05
sum_h = 2 * PAD + code_h(SUMMARY)
bot = max(bottoms + [comp_bot + 40 + sum_h + GAP + TAG * 0.5 + sc_h])
for x0, t, x1 in end_cards:
    box(x0, t, x1, bot)
mid = (a_bot + bot) / 2  # A's conversation runs on into B
for k in (-1, 0, 1):
    ax.add_patch(Rectangle((sum(PANELS[0]) / 2 - 6, mid + 32 * k - 6), 12, 12, color=TAUPE,
                           alpha=0.6, zorder=3))

top = bot - sc_h
prose(bx_c, top + PAD + TAG * 0.8, R - L - 2 * PAD,
      [("Don't act until I tell you to.", {"style": "italic", "color": WINE})], size=SC,
      align="center")
box(L, top, R, bot, fc=BLUSH, ec=ROSE, lw=2.6, ls=(0, (6, 4)), r=12)
tag(bx_c, top, "SC dropped", OXBLOOD)

top -= GAP + TAG * 0.5 + sum_h
code(L + PAD, top + PAD, SUMMARY)
box(L, top, R, top + sum_h)
arrow(bx_c, comp_bot, top - 2)

# panels fit the tallest column
H = bot + INSET + 8
for x0, x1 in PANELS:
    box(x0, TOP, x1, H - 8, fc=STONE, ec=PANEL_EDGE, lw=2.2, r=24, z=0)
fig.set_size_inches(W / 100, H / 100)
ax.set_xlim(0, W)
ax.set_ylim(H, 0)

c.save(Path(__file__).with_name("fig1-failure-mode-inbox"))
