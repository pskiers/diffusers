"""TEMPORARY (Claude, amaze interpretability): figures from interp_prune_mech.py's prune_mech.pt (CPU only).

    python interp_prune_plots.py runs/amaze_interp_v33/prune_mech.pt [out_dir]

fig1_flips.png           where and when tokens are dropped on one board (probe logits over calls)
overview_pruning.png     every drop: when (crossing call) and why (reason at onset); A one board, B all boards, C by branch position
fig2_<reason>_*.png      one drop, one update, piece by piece (waterfall + sender maps), first drop of each reason
fig3_<reason>_*.png      the same drop over the updates around it
fig4_inner.png           all "inner" drops (a tip-side neighbour ★ crossed before the onset): statistics
fig5_removal.png         removal check: ★'s messages to i replaced by their mean vs controls
summary.txt              the numbers behind the figures
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle

G, T = 12, 144
BLUE, ORANGE, MID = "#2a78d6", "#eb6834", "#f0efec"
INK, INK2, MUTED, SURF, LIGHT = "#0b0b0b", "#52514e", "#9a9893", "#fcfcfb", "#c9c7c1"
DIV = LinearSegmentedColormap.from_list("off_path", [ORANGE, MID, BLUE]); DIV.set_bad(SURF)
SEQ = LinearSegmentedColormap.from_list("seq", ["#e3e1dc", "#3b3a37"])
CAT_COL = {"neighbour": "#1baf7a", "walls": "#eda100", "z_H": "#6250d6", "fade": "#e87ba4", "unclear": "#b9b7b0"}
CAT_LET = {"neighbour": "N", "walls": "W", "z_H": "H", "fade": "F", "unclear": "?"}
CAT_DESC = {"neighbour": "a corridor neighbour dropped ≤ 2 updates before and sends the biggest new 'off' push",
            "z_H": "first z_L update after a z_H update; injection is the biggest push down",
            "walls": "no corridor neighbour dropped yet; dead-end cell or wall-covered token pushes most",
            "fade": "dilution is the biggest push down; no sender carries ≥ 25% of the fall",
            "unclear": "none of the above"}
CAT_ORDER = ["neighbour", "walls", "z_H", "fade", "unclear"]
PIECE_LABELS = ["dilution", "injection\n(z_H + maze)", "att0", "mlp0", "att1", "mlp1"]
MAP_NAMES = ["att0", "mlp0", "att1", "mlp1"]
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF})


# ---------------------------------------------------------------- helpers

def walls(ax, bd, r0=0.0, c0=0.0, lw=1.3):
    for x0, y0, x1, y1 in bd["walls"]:
        ax.plot([x0 - c0, x1 - c0], [y0 - r0, y1 - r0], color=INK, lw=lw, solid_capstyle="butt")
    ax.add_patch(Rectangle((-0.5 - c0, -0.5 - r0), G, G, fill=False, ec=INK, lw=lw))


def clean_ax(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def board(ax, bd, vals, alpha=1.0, vmax=4.0):
    ax.imshow(np.asarray(vals).reshape(G, G), cmap=DIV, vmin=-vmax, vmax=vmax, alpha=alpha, interpolation="nearest")
    walls(ax, bd); clean_ax(ax)


def box(ax, t, r0=0, c0=0, lw=2.0, color=INK):
    r, c = divmod(int(t), G)
    ax.add_patch(Rectangle((c - c0 - 0.5, r - r0 - 0.5), 1, 1, fill=False, ec=color, lw=lw))


def star(ax, t, r0=0, c0=0, size=11):
    r, c = divmod(int(t), G)
    ax.text(c - c0, r - r0 - 0.08, "★", ha="center", va="center", fontsize=size, color=INK)


def window(m, t, W=3):
    """(T,) map -> (2W+1, 2W+1) masked window around token t."""
    r, c = divmod(int(t), G)
    out = np.full((2 * W + 1, 2 * W + 1), np.nan)
    for dy in range(-W, W + 1):
        for dx in range(-W, W + 1):
            if 0 <= r + dy < G and 0 <= c + dx < G:
                out[dy + W, dx + W] = m[(r + dy) * G + c + dx]
    return np.ma.masked_invalid(out)


def in_window(t, j, W=3):
    return abs(t // G - j // G) <= W and abs(t % G - j % G) <= W


def sender_map(ax, bd, m, t, st_tok, vmax, W=3, annotate=True):
    win = window(m, t, W)
    im = ax.imshow(win, cmap=DIV, vmin=-vmax, vmax=vmax, interpolation="nearest")
    r, c = divmod(int(t), G)
    walls(ax, bd, r - W, c - W, lw=1.1)
    ax.set_xlim(-0.5, 2 * W + 0.5); ax.set_ylim(2 * W + 0.5, -0.5)
    box(ax, t, r - W, c - W)
    if st_tok is not None and in_window(t, st_tok, W):
        star(ax, st_tok, r - W, c - W, size=9)
    if annotate:
        for (y, x), v in np.ndenumerate(win.filled(np.nan)):
            if np.isfinite(v) and abs(v) >= max(0.1, 0.15 * vmax):
                ax.text(x, y + 0.32, f"{v:+.2f}", ha="center", va="center", fontsize=6.3, color=INK)
    clean_ax(ax)
    return im


def nice_vmax(x, lo=0.2):
    x = max(lo, float(x))
    for s in (0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
        if x <= s:
            return s
    return x


def zh_between(zl_calls, k):
    """True if a z_H update happens between z_L updates k-1 and k."""
    return k > 0 and zl_calls[k] - zl_calls[k - 1] > 1


def fig_header(fig, title, sub=None):
    fig.text(0.03, 0.975, title, fontsize=12, color=INK, weight="bold", va="top")
    if sub:
        fig.text(0.03, 0.948, sub, fontsize=9.5, color=INK2, va="top")


# ---------------------------------------------------------------- figure 1

def fig1(d, bid, out):
    bd = d["boards"][bid]; L = bd["logits"]; zl = d["zl_calls"]
    fl = [f for f in d["flips"] if f["bid"] == bid]
    want = [6, 8, 9, 10, 13, 15, 20]
    ks = [zl.index(c) for c in want if c in zl] + [len(zl) - 1]
    fig = plt.figure(figsize=(16, 9.5))
    fig_header(fig, f"Figure 1 · where and when the thinker drops tokens (board {bd['board']}, t = {d['cfg']['t_start']}, z_L)",
               "Top: probe logit after selected calls (blue path, orange off); black dots = tokens it first calls path (p > 0.8) "
               "and later drops (p < 0.2, for good). Bottom left: their logits over the calls. Bottom right: when each was dropped.")
    for n, k in enumerate(ks):
        ax = fig.add_axes([0.03 + n * 0.118, 0.56, 0.11, 0.33])
        board(ax, bd, L[k])
        for f in fl:
            r_, c_ = divmod(f["token"], G)
            ax.plot(c_, r_, "o", ms=2.6, color=INK)
        ax.set_title(f"call {zl[k]}" + (" (last)" if k == len(zl) - 1 else ""), fontsize=9, color=INK, loc="left")
    ax = fig.add_axes([0.06, 0.08, 0.5, 0.38])
    kmax = max([f["kx"] for f in fl] + [10]) + 4
    kmax = min(kmax, len(zl) - 1)
    xs = zl[:kmax + 1]
    for f in fl:
        y = L[:kmax + 1, f["token"]]
        ax.plot(xs, y, color=INK2, lw=1, alpha=0.6)
        ax.plot(zl[f["ko"]], L[f["ko"], f["token"]], "o", ms=4, color=INK)
        ax.plot(zl[f["kx"]], L[f["kx"], f["token"]], "x", ms=5, color=INK)
    for y_, txt in ((1.39, "p = 0.8"), (-1.39, "p = 0.2")):
        ax.axhline(y_, color=MUTED, lw=0.8, ls=(0, (3, 3))); ax.text(xs[-1], y_, " " + txt, fontsize=8, color=INK2, va="center")
    ax.axhline(0, color=INK2, lw=0.8)
    for c in range(7, xs[-1] + 1, 7):
        ax.axvline(c, color=MUTED, lw=0.6, ls=":")
    ax.set_xlabel("call (dotted: z_H updates)  ·  ● onset, ✕ crossing"); ax.set_ylabel("probe logit")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{len(fl)} dropped tokens on this board", loc="left", fontsize=10, color=INK)
    ax = fig.add_axes([0.66, 0.08, 0.3, 0.38])
    v = np.full(T, np.nan)
    for f in fl:
        v[f["token"]] = f["cross_call"]
    ax.imshow(np.full((G, G), 0.0), cmap=DIV, vmin=-1, vmax=1)
    im = ax.imshow(np.ma.masked_invalid(v.reshape(G, G)), cmap=SEQ, interpolation="nearest")
    walls(ax, bd); clean_ax(ax)
    for f in fl:
        r, c = divmod(f["token"], G)
        ax.text(c, r, str(f["cross_call"]), ha="center", va="center", fontsize=6.5, color=SURF if f["cross_call"] > np.nanmean(v) else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.outline.set_visible(False); cb.set_label("crossing call")
    ax.set_title("crossing call of each dropped token", loc="left", fontsize=10, color=INK)
    fig.savefig(out / "fig1_flips.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------- pruning overview

def choose_panels(per_call, zl, max_panels=6, quiet=1):
    """Groups of consecutive z_L updates with > quiet drops (split at updates with <= quiet drops) if there are
    2..max_panels of them, else the max_panels updates with the most drops, in time order."""
    busy = [c for c in zl if per_call.get(c, 0) > quiet]
    groups = []
    for c in busy:
        if groups and zl.index(c) == zl.index(groups[-1][-1]) + 1:
            groups[-1].append(c)
        else:
            groups.append([c])
    if 2 <= len(groups) <= max_panels:
        return [list(range(g[0], g[-1] + 1)) for g in groups], "groups"
    calls = [c for c in per_call if per_call[c] > 0]
    return [[c] for c in sorted(sorted(calls, key=lambda c: -per_call[c])[:max_panels])], "top"


def overview(d, bid, out, summary):
    zl = d["zl_calls"]
    fl_all = d["flips"]
    fl = [f for f in fl_all if not f["late"]]
    bd = d["boards"][bid]
    fb = [f for f in fl if f["bid"] == bid]
    per_call = Counter(f["cross_call"] for f in fb)
    panels, how = choose_panels(per_call, zl)
    fig = plt.figure(figsize=(16, 15))
    n_never = sum(int((b["cross_k"] == -1).sum()) for b in d["boards"])
    fig_header(fig, "Pruning overview · every token the thinker first calls path (p > 0.8) and later drops for good (p < 0.2): "
                    "WHEN (crossing call) and WHY (reason at the onset)",
               f"{len(fl_all)} drops on {len(d['boards'])} boards ({len(fl_all) - len(fl)} after call {d['cfg']['c_dec']} not analysed); "
               f"{n_never} tokens were never path and are not drops.\nBoard colour = probe logit (faded); dropping tokens are "
               "filled with their reason colour; arrow = from the strongest new sender of a 'neighbour' drop.")
    rule = ("groups of calls with drops, split at calls with ≤ 1 drop; number in a token = its crossing call" if how == "groups"
            else "the 6 calls with the most drops, in time order; letter = reason")
    fig.text(0.03, 0.912, f"A (board {bd['board']}): " + rule + ".", fontsize=9.5, color=INK2)
    show = [["before"]] + panels + [["last"]]
    k_before = max(0, min((f["ko"] for f in fb), default=1) - 1)
    for n, grp in enumerate(show[:8]):
        ax = fig.add_axes([0.03 + (n % 4) * 0.24, 0.705 - (n // 4) * 0.215, 0.2, 0.19])
        if grp == ["before"]:
            k = k_before; title = f"A   call {zl[k]} (before the first drop)"
        elif grp == ["last"]:
            k = len(zl) - 1; title = "last z_L update"
        else:
            k = zl.index(grp[-1])
            n_drop = sum(per_call[c] for c in grp)
            title = (f"call {grp[0]} · {n_drop} dropped" if len(grp) == 1 else f"calls {grp[0]}–{grp[-1]} · {n_drop} dropped")
        board(ax, bd, bd["logits"][k], alpha=0.45)
        if grp not in (["before"], ["last"]):
            for f in fb:
                if f["cross_call"] in grp:
                    r, c = divmod(f["token"], G)
                    col = CAT_COL.get(f["category"], LIGHT)
                    ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, color=col, ec=INK, lw=0.7))
                    txt = CAT_LET.get(f["category"], "") if len(grp) == 1 else str(f["cross_call"])
                    ax.text(c, r, txt, ha="center", va="center", fontsize=6.5 if len(grp) == 1 else 5.5, color=INK, weight="bold")
                    if f["category"] == "neighbour":
                        rj, cj = divmod(f["jstar"], G)
                        ax.add_patch(FancyArrowPatch((cj, rj), (c, r), arrowstyle="-|>", mutation_scale=7, color=INK, lw=1.0,
                                                     shrinkA=1, shrinkB=3))
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    fig.legend(handles=[Patch(color=CAT_COL[c], label=f"{CAT_LET[c]}  {c}: {CAT_DESC[c]}") for c in CAT_ORDER],
               loc="upper left", bbox_to_anchor=(0.03, 0.49), ncol=2, fontsize=8.5, frameon=False, handlelength=1.2)
    # B
    calls_b = sorted({f["cross_call"] for f in fl})
    axB = fig.add_axes([0.06, 0.06, 0.52, 0.31])
    x = np.arange(len(calls_b)); bottom = np.zeros(len(calls_b))
    for cat in CAT_ORDER:
        v = np.array([sum(1 for f in fl if f["cross_call"] == c and f["category"] == cat) for c in calls_b], float)
        axB.bar(x, v, bottom=bottom, color=CAT_COL[cat], width=0.75, edgecolor=SURF, linewidth=1.0, label=cat)
        bottom += v
    for xx, tot in zip(x, bottom):
        axB.text(xx, tot + max(bottom) * 0.01, f"{int(tot)}", ha="center", fontsize=7, color=INK2)
    for n in range(1, len(calls_b)):
        if any(c % 7 == 0 for c in range(calls_b[n - 1] + 1, calls_b[n])):
            axB.axvline(n - 0.5, color=MUTED, lw=0.8, ls=":")
    axB.set_xticks(x, [str(c) for c in calls_b], fontsize=7.5, rotation=90 if len(x) > 25 else 0)
    axB.set_xlabel("crossing call (z_L updates; dotted = a z_H update in between)"); axB.set_ylabel("drops (all boards)")
    axB.spines[["top", "right"]].set_visible(False)
    axB.set_title(f"B   all {len(fl)} analysed drops: when, and for which reason", loc="left", fontsize=10.5, color=INK, weight="bold")
    # C
    axC = fig.add_axes([0.66, 0.06, 0.31, 0.28])
    dist = []
    for f in fl:
        bdf = d["boards"][f["bid"]]
        rnd = bdf["rounds"][tuple(bdf["tok_cell"][f["token"]])]
        dist.append(None if rnd == 0 else min(rnd - 1, 5))
    rows = [0, 1, 2, 3, 4, 5]
    y = np.arange(len(rows))[::-1]; left = np.zeros(len(rows))
    n_row = np.array([sum(1 for dd in dist if dd == r) for r in rows], float)
    for cat in CAT_ORDER:
        v = np.array([sum(1 for f, dd in zip(fl, dist) if dd == r and f["category"] == cat) for r in rows], float)
        v = 100 * v / np.maximum(n_row, 1)
        axC.barh(y, v, left=left, color=CAT_COL[cat], height=0.7, edgecolor=SURF, linewidth=1.0)
        for yy, l, wv in zip(y, left, v):
            if wv >= 12:
                axC.text(l + wv / 2, yy, f"{wv:.0f}%", ha="center", va="center", fontsize=8, color=INK)
        left += v
    axC.set_yticks(y, [f"{r}\n(tip)" if r == 0 else (f"≥{r}" if r == 5 else str(r)) for r in rows])
    for yy, n in zip(y, n_row):
        axC.text(101, yy, f"n={int(n)}", va="center", fontsize=7.5, color=INK2)
    axC.set_ylabel("cells from the dead-end tip"); axC.set_xlabel("% of drops at that distance"); axC.set_xlim(0, 100)
    axC.spines[["top", "right"]].set_visible(False)
    n_sol = sum(dd is None for dd in dist)
    axC.set_title(f"C   reason by position in the branch\n     ({n_sol} drops in solution cells left out)", loc="left",
                  fontsize=10, color=INK, weight="bold")
    shares = Counter(f["category"] for f in fl)
    summary.append("reasons of the analysed drops: " + ", ".join(f"{c} {shares[c]} ({shares[c] / max(1, len(fl)):.0%})" for c in CAT_ORDER))
    fig.text(0.66, 0.41, f"'neighbour' drops = {shares['neighbour'] / max(1, len(fl)):.0%} of the analysed drops",
             fontsize=10, color=INK, bbox=dict(boxstyle="round,pad=0.4", fc=MID, ec="none"))
    fig.savefig(out / "overview_pruning.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------- figure 2 / 3

def fig2(d, f, out, tag):
    bd = d["boards"][f["bid"]]; L = bd["logits"]; zl = d["zl_calls"]; i = f["token"]
    win = list(range(f["ko"], f["kx"] + 1))
    pcs = bd["pieces"][:, :, i]
    k = win[int(np.argmin([pcs[kk].sum() for kk in win]))]
    maps = f["maps"].float().numpy()[k]                                      # (4, S)
    fig = plt.figure(figsize=(15, 9.8))
    fig_header(fig, f"Figure 2 · one drop, one update: board {bd['board']}, token {divmod(i, G)}, update at call {zl[k]} "
                    f"(reason: {f['category']})",
               "Boxed = token i; ★ = its neighbour towards the dead-end tip (if any). Blue = towards path, orange = towards off. "
               "B adds up exactly (float32 replay of the update).")
    for n, (kk, ttl) in enumerate(((k - 1, f"A   probe logit before call {zl[k]}"), (k, f"     after call {zl[k]}"))):
        ax = fig.add_axes([0.03 + n * 0.21, 0.5, 0.19, 0.36])
        vals = L[kk] if kk >= 0 else np.full(T, d["logit_init"])
        board(ax, bd, vals); box(ax, i)
        if f["star"] is not None:
            star(ax, f["star"])
        ax.set_title(f"{ttl}\n     token i: {vals[i]:+.2f}", fontsize=9.5, color=INK, loc="left")
    axB = fig.add_axes([0.5, 0.52, 0.42, 0.34])
    before = L[k - 1, i] if k >= 1 else d["logit_init"]
    level = before
    labels = ["before"] + PIECE_LABELS + ["after\n(replay)"]
    axB.bar(0, before, width=0.62, color=MUTED); axB.text(0, before, f"{before:+.2f}", ha="center",
                                                         va="bottom" if before >= 0 else "top", fontsize=8.5)
    for n, v in enumerate(pcs[k], start=1):
        lo, hi = sorted((level, level + v))
        axB.bar(n, hi - lo, bottom=lo, width=0.62, color=BLUE if v > 0 else ORANGE)
        axB.plot([n - 1 + 0.31, n - 0.31], [level, level], color=MUTED, lw=0.8)
        axB.text(n, hi if v > 0 else lo, f"{v:+.2f}", ha="center", va="bottom" if v > 0 else "top", fontsize=8.5)
        level += v
    axB.plot([len(pcs[k]) + 0.31 - 1 + 1 - 0.62, len(pcs[k]) + 1 - 0.31], [level, level], color=MUTED, lw=0.8)
    axB.bar(len(pcs[k]) + 1, level, width=0.62, color=MUTED)
    axB.text(len(pcs[k]) + 1, level, f"{level:+.2f}", ha="center", va="bottom" if level >= 0 else "top", fontsize=8.5)
    axB.axhline(0, color=INK2, lw=0.9)
    for y_, txt in ((1.39, "p = 0.8"), (-1.39, "p = 0.2")):
        axB.axhline(y_, color=MUTED, lw=0.8, ls=(0, (3, 3)))
        axB.text(1.01, y_, txt, transform=axB.get_yaxis_transform(), fontsize=8, color=INK2, va="center")
    lv = np.cumsum(np.r_[before, pcs[k]])
    axB.set_ylim(min(lv.min(), 0, -1.6) - 0.6, max(lv.max(), 0, 1.6) + 0.6)
    axB.set_xticks(range(len(labels)), labels, fontsize=8); axB.set_ylabel("token i's probe logit")
    axB.spines[["top", "right"]].set_visible(False)
    axB.set_title(f"B   which part of the update moved token i's logit (recorded after: {L[k, i]:+.2f})", loc="left",
                  fontsize=10, color=INK)
    tot = maps.sum(0)
    vmax = nice_vmax(np.nanmax(np.abs(np.concatenate([window(m, i).compressed() for m in list(maps) + [tot]]))))
    names = MAP_NAMES + ["total"]
    for n, m in enumerate(list(maps) + [tot]):
        ax = fig.add_axes([0.03 + n * 0.19, 0.1, 0.17, 0.3])
        im = sender_map(ax, bd, m, i, f["star"], vmax)
        exact = n in (0, 2)
        piece = pcs[k][2 + n] if n < 4 else pcs[k][2:].sum()
        rest = m.sum() - window(m, i).sum()
        ax.set_title(("C   " if n == 0 else "") + {"att0": "att0: direct messages, layer 0", "mlp0": "mlp0: MLP's reaction, layer 0",
                                                    "att1": "att1: direct messages, layer 1", "mlp1": "mlp1: MLP's reaction, layer 1",
                                                    "total": "total per sender"}[names[n]], fontsize=9, color=INK, loc="left")
        ax.set_xlabel((f"sum = {m.sum():+.2f} (bar {piece:+.2f})" if exact else
                       f"sum {m.sum():+.2f} (bar {piece:+.2f}; MLP not additive)") + f"\nrest of board {rest:+.2f}", fontsize=8)
    cb = fig.colorbar(im, cax=fig.add_axes([0.3, 0.035, 0.4, 0.012]), orientation="horizontal", ticks=[-vmax, 0, vmax])
    cb.ax.set_xticklabels([f"−{vmax}: towards off", "0", f"+{vmax}: towards path"], fontsize=8); cb.outline.set_visible(False)
    fig.savefig(out / f"fig2_{tag}.png", dpi=140); plt.close(fig)


def fig3(d, f, out, tag):
    bd = d["boards"][f["bid"]]; L = bd["logits"]; zl = d["zl_calls"]; i = f["token"]
    ks = list(range(max(1, f["ko"] - 2), min(f["kx"] + 1, d["n_dec"] - 1) + 1))[-10:]
    pcs = bd["pieces"][:, :, i]
    fig = plt.figure(figsize=(max(10, 1.5 * len(ks) + 4), 12.5))
    fig_header(fig, f"Figure 3 · one drop over several updates: board {bd['board']}, token {divmod(i, G)} (reason: {f['category']})",
               "Each column = one z_L update = one figure-2 waterfall. ★ = neighbour towards the dead-end tip.")
    Lft, R = 0.17, 0.95
    axA = fig.add_axes([Lft, 0.62, R - Lft, 0.26]); axB = fig.add_axes([Lft, 0.3, R - Lft, 0.26])
    xs = np.arange(len(ks)); xlim = (-1.5, len(ks) - 0.5)
    k0 = ks[0] - 1
    axA.plot(np.r_[-1, xs], L[[k0] + ks, i], color=INK, lw=2, marker="o", ms=4, label="token i")
    if f["star"] is not None:
        axA.plot(np.r_[-1, xs], L[[k0] + ks, f["star"]], color=MUTED, lw=2, ls=(0, (4, 3)), marker="o", ms=3, label="★")
    axA.legend(frameon=False, fontsize=9, loc="lower left")
    for y_ in (1.39, -1.39):
        axA.axhline(y_, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    axA.axhline(0, color=INK2, lw=0.9)
    for n, k in enumerate(ks):
        if zh_between(zl, k):
            for ax in (axA, axB):
                ax.axvline(n - 0.5, color=INK2, lw=1, ls=(0, (1, 2)))
            axA.text(n - 0.5, axA.get_ylim()[1], "z_H", ha="center", va="bottom", fontsize=8, color=INK2)
    axA.set_xlim(*xlim); axA.set_xticks(np.r_[-1, xs], [str(zl[k0])] + [str(zl[k]) for k in ks])
    if f["ko"] in ks:
        axA.annotate("onset", (ks.index(f["ko"]), L[f["ko"], i]), (ks.index(f["ko"]) + 0.2, L[f["ko"], i] + 1.0), fontsize=8.5,
                     arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    if f["kx"] in ks:
        axA.annotate("crossing", (ks.index(f["kx"]), L[f["kx"], i]), (ks.index(f["kx"]) + 0.2, L[f["kx"], i] + 1.0), fontsize=8.5,
                     arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    axA.set_ylabel("probe logit"); axA.spines[["top", "right"]].set_visible(False)
    axA.set_title("A   logit after each call", loc="left", fontsize=10, color=INK)
    Tb = np.vstack([pcs[ks].T, np.full(len(ks), np.nan), pcs[ks].sum(1)])
    vm = nice_vmax(np.nanmax(np.abs(Tb)))
    axB.imshow(np.ma.masked_invalid(Tb), cmap=DIV, vmin=-vm, vmax=vm, aspect="auto",
               extent=(-0.5, len(ks) - 0.5, len(Tb) - 0.5, -0.5), interpolation="nearest")
    for (r, k), v in np.ndenumerate(Tb):
        if np.isfinite(v):
            axB.text(k, r, f"{v:+.2f}", ha="center", va="center", fontsize=8.5, weight="bold" if r == len(Tb) - 1 else "normal")
    axB.set_xlim(*xlim); axB.set_ylim(len(Tb) - 0.5, -0.5)
    axB.set_yticks(range(len(Tb)), PIECE_LABELS + ["", "net change"], fontsize=8.5)
    axB.set_xticks(xs, [f"call {zl[k]}" for k in ks], fontsize=8.5); axB.tick_params(length=0)
    for s in axB.spines.values():
        s.set_visible(False)
    axB.set_title("B   push of each part on token i's logit per update (column sum = the step in A)", loc="left",
                  fontsize=10, color=INK)
    maps = f["maps"].float().numpy()
    tots = [maps[k].sum(0) for k in ks]
    vmax = nice_vmax(max(np.abs(window(m, i).compressed()).max() for m in tots))
    to_fig = lambda x: fig.transFigure.inverted().transform(axA.transData.transform((x, 0)))[0]
    colw = to_fig(1) - to_fig(0)
    for n, (k, m) in enumerate(zip(ks, tots)):
        ax = fig.add_axes([to_fig(n) - 0.46 * colw, 0.07, 0.92 * colw, 0.92 * colw * fig.get_figwidth() / fig.get_figheight()])
        im = sender_map(ax, bd, m, i, f["star"], vmax, annotate=False)
        stv = f" · ★ {m[f['star']]:+.2f}" if f["star"] is not None else ""
        ax.set_xlabel(f"call {zl[k]}{stv}", fontsize=8)
    fig.text(Lft, 0.255, "C   'total' sender map per update (7×7 around i)", fontsize=10, color=INK)
    cb = fig.colorbar(im, cax=fig.add_axes([0.03, 0.08, 0.012, 0.13]), ticks=[-vmax, 0, vmax]); cb.outline.set_visible(False)
    cb.ax.set_yticklabels([f"−{vmax} off", "0", f"+{vmax} path"], fontsize=8)
    fig.savefig(out / f"fig3_{tag}.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------- figure 4

def rot_offset(dr, dc, sdir):
    """rotate offset (dr, dc) so that the ★ direction sdir becomes 'up' (-1, 0)."""
    for _ in range(4):
        if sdir == (-1, 0):
            return dr, dc
        dr, dc = -dc, dr                    # rotate 90° counter-clockwise (right -> up)
        sdir = (-sdir[1], sdir[0])
    return dr, dc


def fig4(d, out, summary):
    fl = [f for f in d["flips"] if not f["late"] and f["star"] is not None and f["star_before"]]
    n_an = sum(1 for f in d["flips"] if not f["late"])
    summary.append(f"inner drops (★ crossed before the onset): {len(fl)} of {n_an} analysed ({len(fl) / max(1, n_an):.0%})")
    if not fl:
        return
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5), gridspec_kw=dict(hspace=0.6, wspace=0.42, left=0.1, right=0.97,
                                                                         top=0.84, bottom=0.1))
    cats = Counter(f["category"] for f in fl)
    fig_header(fig, f"Figure 4 · inner drops: {len(fl)} drops ({len(fl) / max(1, n_an):.0%} of the analysed) whose neighbour "
                    "towards the dead-end tip (★) was dropped before them",
               "Reasons among them: " + ", ".join(f"{c} {cats[c] / len(fl):.0%}" for c in CAT_ORDER)
               + ". Window = onset … crossing; error bars = s.e. over drops.")
    # (a)
    ax = axes[0, 0]
    sh = np.array([f["pieces_win"] / f["fall"] for f in fl if f["fall"] < 0]) * 100
    m, se = sh.mean(0), sh.std(0) / np.sqrt(len(sh))
    y = np.arange(6)[::-1]
    ax.barh(y, m, xerr=se, color=[ORANGE if v > 0 else BLUE for v in m], height=0.6, error_kw=dict(ecolor=INK2, lw=1, capsize=2))
    for yy, v, e in zip(y, m, se):
        ax.text(v + (e + 2 if v >= 0 else -e - 2), yy, f"{v:.0f}%", va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.axvline(0, color=INK2, lw=0.9); ax.set_yticks(y, PIECE_LABELS, fontsize=8.5)
    ax.set_xlabel("share of the fall (onset → crossing), %; bars add up to 100%")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("a   which part of the update makes them drop?", loc="left", fontsize=10, color=INK)
    # (b) aligned mean maps
    W = 2
    acc = {"raw": np.zeros((2 * W + 1, 2 * W + 1)), "chg": np.zeros((2 * W + 1, 2 * W + 1))}
    cnt = np.zeros((2 * W + 1, 2 * W + 1))
    for f in fl:
        i, s = f["token"], f["star"]
        sdir = (s // G - i // G, s % G - i % G)
        for j in range(T):
            dr, dc = j // G - i // G, j % G - i % G
            if abs(dr) > W or abs(dc) > W:
                continue
            rr, cc = rot_offset(dr, dc, sdir)
            acc["raw"][rr + W, cc + W] += f["S_raw"][j]; acc["chg"][rr + W, cc + W] += f["S_chg"][j]; cnt[rr + W, cc + W] += 1
    ax = axes[0, 1]
    both = np.hstack([acc["raw"] / np.maximum(cnt, 1), np.full((2 * W + 1, 1), np.nan), acc["chg"] / np.maximum(cnt, 1)])
    vm = nice_vmax(np.nanmax(np.abs(both)))
    im = ax.imshow(np.ma.masked_invalid(both), cmap=DIV, vmin=-vm, vmax=vm)
    for (r, c), v in np.ndenumerate(both):
        if np.isfinite(v) and abs(v) >= 0.1 * vm:
            ax.text(c, r + 0.3, f"{v:+.2f}", ha="center", va="center", fontsize=7)
    for c0 in (0, 2 * W + 2):
        ax.add_patch(Rectangle((c0 + W - 0.5, W - 0.5), 1, 1, fill=False, ec=INK, lw=1.8))
        ax.text(c0 + W, W - 1.08, "★", ha="center", va="center", fontsize=10)
    ax.set_xticks([W, 3 * W + 2], ["raw push\n(summed over the window)", "change from\nbefore the onset"]); ax.set_yticks([])
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03, ticks=[-vm, 0, vm]); cb.outline.set_visible(False)
    ax.set_title("b   where does the 'off' come from? (mean 5×5 map,\n     rotated so ★ is always above i)", loc="left",
                 fontsize=10, color=INK)
    # (c)
    ax = axes[0, 2]
    kinds = []
    for f in fl:
        bd = d["boards"][f["bid"]]; i, j = f["token"], f["jstar"]
        cell = lambda t: tuple(bd["tok_cell"][t])
        if j == f["star"]:
            kinds.append("★")
        elif j in f["star_cell"]:
            kinds.append("another token of ★'s cell")
        elif j == i:
            kinds.append("token i itself")
        elif abs(j // G - i // G) + abs(j % G - i % G) == 1:
            if cell(j) == cell(i):
                kinds.append("same cell as i")
            elif (cell(i), cell(j)) in set(map(lambda p: (tuple(p[0]), tuple(p[1])), bd["open_pairs"])):
                kinds.append("other corridor neighbour")
            else:
                kinds.append("behind a wall")
        else:
            kinds.append("farther tokens")
    order = ["★", "another token of ★'s cell", "other corridor neighbour", "same cell as i", "behind a wall",
             "token i itself", "farther tokens"]
    frac = np.array([kinds.count(k) for k in order]) / len(kinds) * 100
    y = np.arange(len(order))[::-1]
    ax.barh(y, frac, color=[INK] + [LIGHT] * (len(order) - 1), height=0.6)
    for yy, v in zip(y, frac):
        ax.text(v + 1, yy, f"{v:.0f}%", va="center", fontsize=8.5)
    ax.axvline(25, color=INK2, lw=1, ls=(0, (3, 3))); ax.text(26, y[-1] - 0.45, "1 of 4\nneighbours", fontsize=7.5, color=INK2)
    ax.set_yticks(y, order, fontsize=8.5); ax.set_xlim(0, 100); ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("% of drops where this token gives the biggest new 'off' push")
    ax.set_title("c   is the strongest new 'off' sender ★?", loc="left", fontsize=10, color=INK)
    # (d) heads
    ax = axes[1, 0]
    dirs = {"left": (0, -1), "right": (0, 1), "up": (-1, 0), "down": (1, 0)}
    rows, labels = [], []
    nH = fl[0]["head_nb"].shape[2]
    for l in range(fl[0]["head_nb"].shape[1]):
        for name, dv in dirs.items():
            sel = [f for f in fl if (f["star"] // G - f["token"] // G, f["star"] % G - f["token"] % G) == dv]
            n_idx = 1 + list(dirs).index(name)
            v = (np.mean([f["head_nb"][f["ko"]:f["kx"] + 1, l, :, n_idx].float().numpy().sum(0) for f in sel], 0)
                 if sel else np.full(nH, np.nan))
            rows.append(v); labels.append(f"layer {l} · ★ {name} of i (n={len(sel)})")
    Hm = np.array(rows)
    vm = nice_vmax(np.nanmax(np.abs(Hm)))
    im = ax.imshow(np.ma.masked_invalid(Hm), cmap=DIV, vmin=-vm, vmax=vm, aspect="auto")
    for (r, c), v in np.ndenumerate(Hm):
        if np.isfinite(v) and abs(v) >= 0.15 * vm:
            ax.text(c, r, f"{v:+.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(nH), [f"H{h}" for h in range(nH)], fontsize=8); ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
    ax.axhline(3.5, color=SURF, lw=3); ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, ticks=[-vm, 0, vm]); cb.outline.set_visible(False)
    ax.set_title("d   which head carries ★'s message? (mean direct\n     push of ★ → i per head, summed over the window)",
                 loc="left", fontsize=10, color=INK)
    # (e) lag, from ★'s crossing and from ★'s own onset (same peak rule as for i)
    ax = axes[1, 1]
    def onset_of(Ls, kx, w=6):
        lo = max(0, kx - w)
        return lo + int(np.argmax(Ls[lo:kx])) + 1
    lag_x = np.array([f["ko"] - f["star_cross_k"] for f in fl])
    lag_o = np.array([f["ko"] - onset_of(d["boards"][f["bid"]]["logits"][:, f["star"]], f["star_cross_k"])
                      if f["star_cross_k"] > 0 else np.nan for f in fl])
    bins = ["≤ −1", "0", "1", "2", "3", "≥ 4"]
    def hist(v):
        v = v[np.isfinite(v)]
        return np.array([(v <= -1).sum(), (v == 0).sum(), (v == 1).sum(), (v == 2).sum(), (v == 3).sum(), (v >= 4).sum()]) / max(1, len(v)) * 100
    for off, v, col, lab in ((-0.18, lag_x, INK, "from ★'s crossing"), (0.18, lag_o, LIGHT, "from ★'s onset")):
        hv = hist(v)
        ax.bar(np.arange(6) + off, hv, width=0.34, color=col, label=lab)
        for x_, h in enumerate(hv):
            ax.text(x_ + off, h + 1, f"{h:.0f}", ha="center", fontsize=7.5)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_xticks(range(6), bins); ax.set_xlabel("z_L updates until i's onset"); ax.set_ylabel("% of drops")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("e   does i start falling right after ★?", loc="left", fontsize=10, color=INK)
    # (f) H-cycle position, all analysed drops
    ax = axes[1, 2]
    per = d["cfg"]["per"]
    allf = [f for f in d["flips"] if not f["late"]]
    pos = np.array([f["onset_call"] % per for f in allf])
    pc = np.array([(pos == p).sum() for p in range(1, per)]) / max(1, len(pos)) * 100
    ax.bar(range(1, per), pc, color=LIGHT, width=0.6)
    for x_, v in zip(range(1, per), pc):
        ax.text(x_, v + 0.5, f"{v:.0f}%", ha="center", fontsize=8.5)
    ax.axhline(100 / (per - 1), color=INK2, lw=1, ls=(0, (3, 3)))
    ax.set_xticks(range(1, per), ["1st\n(right after\nz_H update)"] + [f"{p}th" if p > 3 else ["", "", "2nd", "3rd"][p]
                                                                         for p in range(2, per)], fontsize=8)
    ax.set_xlabel("which z_L update of the H-cycle the onset falls on"); ax.set_ylabel("% of drops")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"f   are drops tied to z_H updates? (all {len(allf)} analysed drops)", loc="left", fontsize=10, color=INK)
    fig.savefig(out / "fig4_inner.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------- figure 5

def fig5(d, out, summary):
    rem = d["removal"]
    if not rem:
        summary.append("removal check: no runs")
        return
    zl = d["zl_calls"]
    by = {}
    for r in rem:
        by.setdefault((r["bid"], r["token"]), {})[r["cond"]] = r
    conds = ["star", "star_cell", "star_cell_all_calls", "other_nb", "wall_nb", "far"]
    titles = {"star": "★ only (z_L calls)", "star_cell": "all tokens of ★'s cell (z_L calls)",
              "star_cell_all_calls": "★'s cell, z_L and z_H calls", "other_nb": "control: other corridor neighbour",
              "wall_nb": "control: wall-covered neighbour", "far": "control: random far token"}
    agree = [r["kx_new"] == r["kx"] for r in rem if r["cond"] == "none"]
    summary.append(f"removal check: {len(by)} drops; unedited rerun reproduces the crossing in {np.mean(agree):.0%}")
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5), gridspec_kw=dict(hspace=0.45, wspace=0.35, left=0.06, right=0.98,
                                                                        top=0.85, bottom=0.08))
    fig_header(fig, "Figure 5 · removal check: the messages of J to token i replaced by their mean (all heads, both layers), "
                    "from ★'s crossing on — does i drop later?",
               "One dot per drop: crossing call without (x) vs with the edit (y). Above the diagonal = delayed; "
               "dots at the top = still path at the last checked call.")
    stats = []
    for n, c in enumerate(conds):
        ax = axes.flat[n]
        xs, ys, nd = [], [], 0
        for key, rr in by.items():
            if c not in rr or "none" not in rr:
                continue
            base, r = rr["none"], rr[c]
            if base["kx_new"] < 0:
                continue
            xs.append(zl[base["kx_new"]])
            if r["kx_new"] >= 0:
                ys.append(zl[r["kx_new"]])
            else:
                ys.append(r["c_stop"] + 3); nd += 1
        xs, ys = np.array(xs), np.array(ys)
        if len(xs):
            lim = (min(xs.min(), ys.min()) - 2, max(xs.max(), ys.max()) + 2)
            ax.plot(lim, lim, color=MUTED, lw=0.8)
            ax.scatter(xs + np.random.default_rng(0).uniform(-0.25, 0.25, len(xs)), ys, s=12, color=INK, alpha=0.6)
            ax.set_xlim(*lim); ax.set_ylim(*lim)
            delayed = (ys > xs).mean() * 100
            dl = np.array([zl.index(y) - zl.index(x) if y in zl else np.nan for x, y in zip(xs, ys)], float)
            stats.append((c, len(xs), delayed, nd / len(xs) * 100, np.nanmean(dl)))
            ax.set_title(f"{titles[c]}\nn={len(xs)} · delayed {delayed:.0f}% · not dropped {nd / len(xs):.0%}",
                         loc="left", fontsize=9, color=INK)
        ax.set_xlabel("crossing call, unedited"); ax.set_ylabel("crossing call, edited")
        ax.spines[["top", "right"]].set_visible(False)
    ax = axes.flat[6]
    if stats:
        y = np.arange(len(stats))[::-1]
        ax.barh(y, [s[2] for s in stats], color=[INK if s[0].startswith("star") else LIGHT for s in stats], height=0.6)
        for yy, s in zip(y, stats):
            ax.text(s[2] + 1, yy, f"{s[2]:.0f}% (mean {s[4]:+.1f} upd.)", va="center", fontsize=8)
        ax.set_yticks(y, [titles[s[0]] for s in stats], fontsize=8); ax.set_xlim(0, 110)
        ax.set_xlabel("% of drops delayed (dropped later or not at all)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("delayed drops per edit", loc="left", fontsize=10, color=INK)
    axes.flat[7].axis("off")
    for s in stats:
        summary.append(f"  {s[0]:22s} n={s[1]:4d} delayed {s[2]:5.1f}%  not dropped {s[3]:5.1f}%  mean delay of dropped {s[4]:+.2f} updates")
    fig.savefig(out / "fig5_removal.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------- all

def make_all(pt, out=None):
    pt = Path(pt); out = Path(out) if out else pt.parent
    out.mkdir(parents=True, exist_ok=True)
    d = torch.load(pt, weights_only=False)
    summary = [f"boards {len(d['boards'])}, drops {len(d['flips'])} (late {sum(f['late'] for f in d['flips'])}), "
               f"replay |Δlogit| max {d['replay_err'].max():.3g} mean {d['replay_err'].mean():.3g}, "
               f"split sum error max {d['sum_err'].max():.3g}"]
    tips = [(bid, int((bd["rounds"] == 1).sum())) for bid, bd in enumerate(d["boards"])]
    bid = next((b for b, n in tips if n >= 3 and any(f["bid"] == b for f in d["flips"])), 0)
    fig1(d, bid, out)
    overview(d, bid, out, summary)
    done = set()
    for f in d["flips"]:
        if f["late"] or f["category"] in done:
            continue
        done.add(f["category"])
        tag = f"{f['category']}_b{d['boards'][f['bid']]['board']}_t{f['token']}"
        fig2(d, f, out, tag)
        fig3(d, f, out, tag)
    fig4(d, out, summary)
    fig5(d, out, summary)
    (out / "summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    make_all(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
