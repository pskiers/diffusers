"""Plot helpers shared by the interpretability ablations."""


def cycle_boundaries(k: str, n_updates: int, L_cycles: int, H_cycles: int, kept_L: list | None = None) -> tuple:
    """Cycle_boundaries to visualize on plots: position (in the recorded updates) of the first update after each boundary.

    z_L: minor = end of an L-loop (a z_H update happens in between), major = end of a reasoning step (n_sup).
    z_H: major = end of a reasoning step (H_cycles z_H updates per step), no minor ones.
    kept_L: the original z_L update numbers that were recorded (keep_L_indices); None = all of them, in order.

    Returns:
        tuple: (minor_boundaries, major_boundaries), lists of positions.
    """
    idx = list(kept_L) if (k == "z_L" and kept_L is not None) else list(range(n_updates))

    def crossings(period: int) -> list:
        return [j for j in range(1, len(idx)) if idx[j] // period != idx[j - 1] // period]

    if k == "z_L":
        major_boundaries = crossings(H_cycles * L_cycles)
        minor_boundaries = [b for b in crossings(L_cycles) if b not in major_boundaries]
    else:
        major_boundaries = crossings(H_cycles)
        minor_boundaries = []
    return minor_boundaries, major_boundaries


def draw_boundaries(ax, minor_boundaries: list, major_boundaries: list, horizontal: bool = False, offset: float = 0.0,
                    major_color: str = "black", label: bool = True):
    """Draw the cycle boundaries on `ax`: vertical lines (updates on the X axis) or horizontal ones (heatmap rows).

    offset: where the line sits relative to the boundary position b, e.g. -0.5 to draw it between update b - 1 and b
            (a heatmap row b spans b - 0.5 .. b + 0.5).
    """
    line = ax.axhline if horizontal else ax.axvline
    for i, b in enumerate(minor_boundaries):
        line(b + offset, color="red", linestyle="--", alpha=0.5, linewidth=0.8,
             label="Koniec pętli L (update z_H)" if label and i == 0 else None)
    for i, b in enumerate(major_boundaries):
        line(b + offset, color=major_color, linestyle="-", alpha=0.7, linewidth=1.0,
             label="Koniec kroku rozumowania (n_sup)" if label and i == 0 else None)
