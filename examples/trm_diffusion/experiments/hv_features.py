"""Side-specific wall features per token, from the maze image at 144 px (shared by wall_decompose.py and interp_hv_split.py).

Per token (12x12-px patch, 12x12 grid):
  wall_sides  (6): dark fraction in 3-px bands at the patch's left / right / top / bottom edge, and in the middle
                   horizontal / vertical band
  cell_walls  (4): walls of the maze cell that contains the token's centre: left / right / top / bottom (0/1)
  corridor    (3): that cell is a horizontal corridor (walls top+bottom, open left+right), a vertical corridor, or other
"""
import numpy as np

G, P, IMG = 12, 12, 144
WALL_DARK = 80


def cell_wall_sides(dark, r, c, cs):
    """(4,) bool: walls on the left / right / top / bottom border of cell (r, c); dark: (144, 144) bool."""
    y0, y1, x0, x1 = int(round(r * cs)), int(round((r + 1) * cs)), int(round(c * cs)), int(round((c + 1) * cs))
    my = slice(int(round(y0 + 0.3 * cs)), int(round(y0 + 0.7 * cs)))
    mx = slice(int(round(x0 + 0.3 * cs)), int(round(x0 + 0.7 * cs)))
    # walls are ~1 px lines and the grid pitch is not exactly cs, so: some line inside a ±3 px band is mostly dark
    vline = lambda a: a.size and a.mean(0).max() > 0.5          # a vertical wall: a mostly dark column
    hline = lambda a: a.size and a.mean(1).max() > 0.5          # a horizontal wall: a mostly dark row
    return np.array([vline(dark[my, max(0, x0 - 3):x0 + 4]), vline(dark[my, max(0, x1 - 4):x1 + 3]),
                     hline(dark[max(0, y0 - 3):y0 + 4, mx]), hline(dark[max(0, y1 - 4):y1 + 3, mx])], dtype=float)


def wall_features(maze_u8, n_cells):
    """maze_u8: (3, 144, 144) uint8 tensor/array; n_cells: maze side length. Returns dict name -> (144, k) float."""
    m = np.asarray(maze_u8).astype(float)
    dark = m.mean(0) < WALL_DARK
    cs = IMG / n_cells
    sides, cw, cor = [], [], []
    cache = {}
    for tr in range(G):
        for tc in range(G):
            p = dark[tr * P:(tr + 1) * P, tc * P:(tc + 1) * P]
            sides.append([p[:, :3].mean(), p[:, -3:].mean(), p[:3, :].mean(), p[-3:, :].mean(),
                          p[4:8, :].mean(), p[:, 4:8].mean()])
            r, c = int((tr * P + P / 2) // cs), int((tc * P + P / 2) // cs)
            if (r, c) not in cache:
                cache[(r, c)] = cell_wall_sides(dark, r, c, cs)
            w = cache[(r, c)]
            cw.append(w)
            L, R, T, B = w
            cor.append([T and B and not L and not R, L and R and not T and not B, 0.0])
            cor[-1][2] = float(not (cor[-1][0] or cor[-1][1]))
    return {"wall_sides": np.array(sides), "cell_walls": np.array(cw), "corridor": np.array(cor, dtype=float)}
