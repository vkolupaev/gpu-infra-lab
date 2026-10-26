"""plotstyle.py - shared colors and axis styling for the lab's matplotlib plots."""

INK, INK2, BLUE, ORANGE, GRID, SURF, PHASE = (
    "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#e6e5e0", "#fcfcfb", "#f0efec")
# One color per line (group or metric), in this order
SERIES = (BLUE, ORANGE, "#1f9e74", "#8a5cd6", "#c2417a", "#8c6d1f")


def style_axis(ax, ylabel):
    """Light surface, horizontal grid only, no top/right spines, muted ticks."""
    ax.set_facecolor(SURF)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
