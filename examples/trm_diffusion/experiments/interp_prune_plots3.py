"""TEMPORARY (Claude, amaze interpretability): the pruning story, from interp_prune_mech2/3.py's prune_mech2.pt (CPU only).

    python interp_prune_plots3.py <prune_mech2.pt> [out_dir]

One population for every figure: drops of side-branch corridor tokens (inside a cell, or across an open passage, of a
cell that dead-end filling removes), dropped by call c_dec, and confirmed by a probe trained at the call of the
token's peak (so that early drops that are only the final-state probe misreading an unformed state are left out).
A drop = probe logit > +1.39 (p 0.8), later < −1.39 and < 0 from then on. ★ = the corridor token of the neighbouring
cell on the dead-end side (fixed by the maze). "Decisive update" = the z_L update in which token i's logit falls most.
fig1_tip_inwards.png   pruning on one board over the calls; order of i and ★ on all boards
fig2_causal.png        cutting what ★'s cell sends to i delays the drop; controls
fig3_how.png           averaged around the decisive update: ★ falls, then i's MLP switches to 'off'
fig4_zH.png            the z_H refresh keeps pruning going
figS_example_<dir>.png one drop step by step, one example per dead-end direction
summary.txt            the numbers
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from interp_prune_plots import BLUE, DIV, INK, INK2, LIGHT, MID, MUTED, ORANGE, SURF, clean_ax, nice_vmax, sender_map, window
from interp_prune_plots2 import crossing_index

G, T = 12, 144
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "axes.titlesize": 10.5, "axes.titlelocation": "left"})
GREEN = "#1baf7a"
DIRS = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}


def title(fig, text, sub=None):
    fig.text(0.02, 0.975, text, fontsize=13, color=INK, weight="bold", va="top")
    if sub:
        fig.text(0.02, 0.937, sub, fontsize=10, color=INK2, va="top")


def spines(ax):
    ax.spines[["top", "right"]].set_visible(False)


def population(d):
    B = d["boards"]
    P = [f for f in d["flips"] if not f["late"] and f["corridor"]
         and B[f["bid"]]["logits_pc"][f["ko"] - 1, f["token"]] > 1.39]
    for f in P:
        pc = B[f["bid"]]["pieces"][:, :, f["token"]]
        f["kdec"] = max(range(f["ko"], f["kx"] + 1), key=lambda kk: -pc[kk].sum())
        f["dir"] = next((n for n, dv in DIRS.items() if f["star"] is not None and
                         (f["star"] // G - f["token"] // G, f["star"] % G - f["token"] % G) == dv), None)
    return P


def order_group(f):
    sk = f["star_cross_k"]
    if f["star"] is None:
        return "no ★ found"
    if f["star_before"]:
        return "★ dropped first"
    if sk == -1:
        return "★ off from the start"
    return "★ and i fall together" if 0 <= sk <= f["kx"] else "★ dropped after i"


def sol_board(ax, bd):
    ax.imshow(bd["sol"], extent=(-0.5, G - 0.5, G - 0.5, -0.5), interpolation="bilinear")
    clean_ax(ax)


def removal_table(d, P):
    ids = {(f["bid"], f["token"]) for f in P}
    by = {}
    for r in d["removal"]:
        if (r["bid"], r["token"]) in ids:
            by.setdefault((r["bid"], r["token"]), {})[r["cond"]] = r
    return {k: rr for k, rr in by.items() if "none" in rr and rr["none"]["kx_new"] >= 0}


# ---------------------------------------------------------------- figure 1

def fig1(d, P, out, S):
    zl = d["zl_calls"]
    x = np.array([f["round"] - 1 for f in P]); y = np.array([f["kx"] for f in P])
    slope = np.polyfit(x, y, 1)[0]
    rk = lambda a: np.argsort(np.argsort(a))
    rho = np.corrcoef(rk(x), rk(y))[0, 1]
    og = Counter(order_group(f) for f in P)
    fit = (og["★ dropped first"] + og["★ and i fall together"] + og["★ off from the start"]) / len(P)
    S.append(f"population: {len(P)} confirmed side-branch corridor drops on {len(d['boards'])} boards")
    S.append(f"tip inwards: {slope:.2f} z_L updates per cell, rank correlation {rho:.2f}; order: "
             + ", ".join(f"{k} {v}" for k, v in og.most_common()) + f"; fits tip-first {fit:.0%}")
    bid = Counter(f["bid"] for f in P).most_common(1)[0][0]
    bd = d["boards"][bid]
    fb = [f for f in P if f["bid"] == bid]
    calls = sorted({f["cross_call"] for f in fb})
    groups = []                                           # one panel per H-cycle that has drops
    for c in calls:
        h = (c - 1) // 7
        if groups and (groups[-1][0] - 1) // 7 == h:
            groups[-1].append(c)
        else:
            groups.append([c])
    show = [("start", 1)] + [(g, zl.index(g[-1])) for g in groups[:4]] + [("end", len(zl) - 1)]
    side = [t for t in range(T) if bd["rounds"][tuple(bd["tok_cell"][t])] > 0 and bd["ttype"][t] in ("interior", "open_link")]
    fig = plt.figure(figsize=(17, 8.6))
    title(fig, f"Dead ends are pruned from the tip inwards: about one cell per z_L update ({slope:.2f} updates per cell, "
               f"rank correlation {rho:.2f})")
    W = 0.94 / len(show)
    for n, (g, k) in enumerate(show):
        ax = fig.add_axes([0.02 + n * W, 0.44, W * 0.92, W * 0.92 * 17 / 8.6])
        sol_board(ax, bd)
        for t in side:
            r, c = divmod(t, G)
            f_now = next((f for f in fb if f["token"] == t and g not in ("start", "end") and f["cross_call"] in g), None)
            if f_now is not None:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fc=ORANGE, ec=INK, lw=1.0))
                ax.text(c, r, str(f_now["cross_call"]), ha="center", va="center", fontsize=7, color=INK)
            elif bd["logits"][k, t] >= 0:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fc=BLUE, ec="none", alpha=0.3))
            else:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fc=ORANGE, ec="none", alpha=0.4))
        if g == "start":
            ttl = f"call {zl[k]}"
        elif g == "end":
            ttl = "end of reasoning"
        else:
            ttl = (f"calls {g[0]}–{g[-1]}" if len(g) > 1 else f"call {g[0]}") + f"  (H-cycle {(g[0] - 1) // 7 + 1})"
        ax.set_title(ttl)
    for x0, (fc, a, ec, lab) in zip((0.02, 0.17, 0.27), ((BLUE, 0.3, "none", "side branch, still path"),
                                                          (ORANGE, 0.4, "none", "dropped"),
                                                          (ORANGE, 1.0, INK, "dropped in this H-cycle (number = call)"))):
        axl = fig.add_axes([x0, 0.385, 0.011, 0.022]); axl.add_patch(Rectangle((0, 0), 1, 1, fc=fc, ec=ec, alpha=a))
        clean_ax(axl); fig.text(x0 + 0.015, 0.39, lab, fontsize=9.5, color=INK2)
    # B: order of i and ★
    ax = fig.add_axes([0.2, 0.06, 0.5, 0.25])
    order = ["★ dropped first", "★ and i fall together", "★ off from the start", "★ dropped after i", "no ★ found"]
    show_lab = {"★ dropped first": "★ dropped before i started falling", "★ and i fall together": "★ dropped while i was falling",
                "★ off from the start": "★ never read as path (off from the start)", "★ dropped after i": "★ dropped after i (wrong order)",
                "no ★ found": "no ★ found"}
    v = np.array([og[o] for o in order]) / len(P) * 100
    y = np.arange(len(order))[::-1]
    ax.barh(y, v, color=[INK, INK, INK, LIGHT, LIGHT], height=0.62)
    for yy, vv in zip(y, v):
        ax.text(vv + 1, yy, f"{vv:.0f}%", va="center", fontsize=9.5)
    ax.set_yticks(y, [show_lab[o] for o in order], fontsize=9.5); ax.set_xlim(0, 100); spines(ax)
    ax.set_xlabel(f"% of {len(P)} drops on {len(d['boards'])} boards")
    ax.set_title(f"token i and its neighbour on the dead-end side (★): ★ is off first or at the same time in {fit:.0%}")
    fig.savefig(out / "fig1_tip_inwards.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------- figure 2

ROWS = [("cell_zL", "all tokens of ★'s cell"), ("star_zL", "★ alone"), ("cell_zH", "★'s cell, only inside z_H updates"),
        ("other_both", "control: the other corridor neighbour"), ("wall_both", "control: a neighbour behind a wall"),
        ("far_both", "control: a random far token")]


def fig2(d, P, out, S):
    zl = d["zl_calls"]
    by = removal_table(d, P)

    def stats(c):
        v = [(rr[c]["kx_new"], rr["none"]["kx_new"], rr[c]["c_stop"]) for rr in by.values() if c in rr]
        dl = [n == -2 or n > b for n, b, _ in v]
        dd = [n - b for n, b, _ in v if n > b]
        return v, (100 * np.mean(dl) if v else np.nan), (np.median(dd) if dd else 0)
    st = {c: stats(c) for c, _ in ROWS}
    ctrl = max(st[c][1] for c in ("other_both", "wall_both", "far_both"))
    fig = plt.figure(figsize=(17, 6.6))
    title(fig, f"The neighbour on the dead-end side tells token i to drop: cutting what ★'s cell sends delays "
               f"{st['cell_zL'][1]:.0f}% of drops (median +{st['cell_zL'][2]:.0f} updates); other tokens ≤ {ctrl:.0f}%",
          "Rerun of the thinker: from 2 updates before the drop, the messages of the chosen tokens to i (all attention heads, "
          "both layers) are replaced by an average message.")
    for n, (c, lab) in enumerate((("cell_zL", "cut: all tokens of ★'s cell"), ("far_both", "control: cut a random far token"))):
        ax = fig.add_axes([0.05 + n * 0.25, 0.13, 0.19, 0.19 * 17 / 6.6])
        v = st[c][0]
        xs = np.array([zl[b] for _, b, _ in v], float)
        ys = np.array([zl[nw] if nw >= 0 else cs + 4 for nw, _, cs in v], float)
        rng = np.random.default_rng(n)
        lim = (2, max(30, np.nanmax(ys) + 2))
        ax.plot(lim, lim, color=MUTED, lw=0.9)
        ax.scatter(xs + rng.uniform(-0.3, 0.3, len(xs)), ys + rng.uniform(-0.3, 0.3, len(ys)), s=9, color=INK if n == 0 else MUTED,
                   alpha=0.5)
        ax.set_xlim(*lim); ax.set_ylim(*lim); spines(ax)
        ax.set_xlabel("call of the drop, unedited"); ax.set_ylabel("call of the drop, edited")
        ax.set_title(f"{lab}\nabove the line = dropped later ({st[c][1]:.0f}%)")
    ax = fig.add_axes([0.66, 0.13, 0.15, 0.68])
    y = np.arange(len(ROWS))[::-1]
    ax.barh(y, [st[c][1] for c, _ in ROWS], color=[INK, INK, MUTED, LIGHT, LIGHT, LIGHT], height=0.62)
    for yy, (c, _) in zip(y, ROWS):
        ax.text(st[c][1] + 1.5, yy, f"{st[c][1]:.0f}%  (n={len(st[c][0])})", va="center", fontsize=9)
        S.append(f"  cut {c:10s} n={len(st[c][0]):4d} delayed {st[c][1]:5.1f}%  median delay {st[c][2]:.0f} updates")
    ax.set_yticks(y, [lab for _, lab in ROWS], fontsize=9.5); ax.set_xlim(0, 80); spines(ax)
    ax.set_xlabel("% of drops delayed"); ax.set_title("whose messages are cut")
    agree = np.mean([rr["none"]["kx_new"] == rr["none"]["kx"] for rr in by.values()])
    S.append(f"  unedited rerun reproduces the crossing in {agree:.0%} of {len(by)} drops")
    fig.savefig(out / "fig2_causal.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------- figure 3

DIST_GROUPS = ["★'s cell", "other neighbours", "2 tokens away", "3 tokens away", "4–5 away", "6+ away"]


def dist_group(f, j):
    i = f["token"]
    if j in f["star_cell"]:
        return 0
    dd = abs(j // G - i // G) + abs(j % G - i % G)
    if dd == 0:
        return None
    return 1 if dd == 1 else (2 if dd == 2 else (3 if dd == 3 else (4 if dd <= 5 else 5)))


def change_by_distance(f, ch):
    """per-token change of the pushes from each distance group: the flip update vs the 3 updates before the fall."""
    mp = f["maps"].float().numpy()
    pre = list(range(max(0, f["ko"] - 3), f["ko"]))
    if not pre:
        return None
    chg = mp[f["kx"]][ch].sum(0) - mp[pre][:, ch].sum(1).mean(0)
    out = [[] for _ in DIST_GROUPS]
    for j in range(T):
        g = dist_group(f, j)
        if g is not None:
            out[g].append(chg[j])
    return [np.mean(v) if v else np.nan for v in out]


def fig3(d, P, out, S):
    B = d["boards"]
    Pw = [f for f in P if f["star"] is not None]
    R = list(range(-4, 4))
    Li, Ls, Pc = ({r: [] for r in R} for _ in range(3))
    for f in Pw:
        bd = B[f["bid"]]; pc = bd["pieces"][:, :, f["token"]]
        for r in R:
            kk = f["kx"] + r
            if 1 <= kk < d["n_dec"]:
                Li[r].append(bd["logits"][kk, f["token"]]); Ls[r].append(bd["logits"][kk, f["star"]]); Pc[r].append(pc[kk])
    m = lambda dct: np.array([np.mean(dct[r], 0) for r in R])
    se = lambda dct: np.array([np.std(dct[r], 0) / np.sqrt(len(dct[r])) for r in R])
    Mi, Ms, MP = m(Li), m(Ls), m(Pc)
    pn = d["pieces_names"]
    mlp1_b, mlp1_0 = MP[R.index(-3), pn.index("mlp1")], MP[R.index(0), pn.index("mlp1")]
    head = ("One update after ★ turns off, token i's second MLP switches from writing 'path' to writing 'off'"
            if mlp1_b > 0 > mlp1_0 else "What changes around the flip")
    fig = plt.figure(figsize=(17, 6.6))
    title(fig, f"{head} (mlp1: {mlp1_b:+.1f} → {mlp1_0:+.1f})",
          f"Average over all {len(Pw)} analysed drops that have a ★, aligned on the flip (0) = the z_L update in which "
          "token i's logit goes below 0.")
    S.append("aligned on the flip (rel −3..+1): logit i " + " ".join(f"{v:+.1f}" for v in Mi[1:6])
             + "; logit ★ " + " ".join(f"{v:+.1f}" for v in Ms[1:6])
             + "; mlp1 " + " ".join(f"{v:+.1f}" for v in MP[1:6, pn.index('mlp1')]))
    ax = fig.add_axes([0.05, 0.14, 0.26, 0.66])
    for M, E, col, lab in ((Mi, se(Li), INK, "token i"), (Ms, se(Ls), ORANGE, "★ (its neighbour on the dead-end side)")):
        ax.errorbar(R, M, yerr=E, color=col, lw=2.2, marker="o", ms=4, capsize=2, label=lab)
    ax.axhline(0, color=INK2, lw=0.8); ax.axvline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("z_L update relative to i's flip"); ax.set_ylabel("probe logit (> 0: path)")
    ax.legend(frameon=False, fontsize=9); spines(ax); ax.set_title("A   ★ turns off first, i one update later")
    ax = fig.add_axes([0.38, 0.14, 0.28, 0.66])
    styles = {"mlp1": (ORANGE, 2.6, "-"), "mlp0": (ORANGE, 1.6, "--"), "att1": (BLUE, 2.0, "-"), "att0": (BLUE, 1.4, "--"),
              "injection": (GREEN, 1.6, "-")}
    names = {"mlp1": "MLP, layer 1", "mlp0": "MLP, layer 0", "att1": "attention, layer 1", "att0": "attention, layer 0",
             "injection": "input injection (z_H + maze)"}
    Ep = se(Pc)
    for p, (col, lw, ls) in styles.items():
        j = pn.index(p)
        ax.errorbar(R, MP[:, j], yerr=Ep[:, j], color=col, lw=lw, ls=ls, marker="o", ms=3, capsize=2, label=names[p])
    ax.axhline(0, color=INK2, lw=0.8); ax.axvline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("z_L update relative to i's flip"); ax.set_ylabel("push on i's logit in that update")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left"); spines(ax)
    ax.set_title("B   what each part of the update writes into i")
    vals = {}
    for lab, ch in (("MLP's reaction to the token's message", [1, 3]), ("the token's direct attention message", [0, 2])):
        rows = [r for r in (change_by_distance(f, ch) for f in Pw) if r is not None]
        vals[lab] = np.nanmean(np.array(rows, float), 0)
        S.append(f"  change at the flip vs before, per token, {lab}: "
                 + ", ".join(f"{g} {v:+.3f}" for g, v in zip(DIST_GROUPS, vals[lab])))
    ax = fig.add_axes([0.74, 0.14, 0.24, 0.66])
    y = np.arange(len(DIST_GROUPS))[::-1]
    for off, (lab, col) in zip((-0.18, 0.18), zip(vals, (ORANGE, BLUE))):
        ax.barh(y + off, vals[lab], height=0.34, color=col, label=lab)
    ax.axvline(0, color=INK2, lw=0.8); ax.set_yticks(y, DIST_GROUPS); spines(ax)
    ax.set_xlabel("change per token (flip vs before the fall)"); ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.set_title("C   which tokens the change comes from\n     (per token, by distance from i)")
    fig.savefig(out / "fig3_how.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------- figure 4

def fig4(d, P, out, S):
    zl = d["zl_calls"]
    B = d["boards"]
    later = [f for f in P if f["cross_call"] > 14]
    new_calls, dl = [], []
    for f in P:
        La = d["skips"][f["bid"]]["skip_zH14"][:, f["token"]]
        La = La[np.isfinite(La)]
        b0, b1 = crossing_index(B[f["bid"]]["logits"][:len(La), f["token"]]), crossing_index(La)
        new_calls.append(zl[b1] if b1 >= 0 else None)
        if f["cross_call"] > 14 and b0 >= 0:
            dl.append(b1 == -2 or b1 > b0)
    p_skip = 100 * np.mean(dl)
    dl7 = []
    for f in P:
        La = d["skips"][f["bid"]]["skip_zH7"][:, f["token"]]
        La = La[np.isfinite(La)]
        b0, b1 = crossing_index(B[f["bid"]]["logits"][:len(La), f["token"]]), crossing_index(La)
        if f["cross_call"] > 7 and b0 >= 0:
            dl7.append(b1 == -2 or b1 > b0)
    p_skip7 = 100 * np.mean(dl7)
    by = removal_table(d, P)
    frz = [rr["freeze_zH_i"]["kx_new"] == -2 or rr["freeze_zH_i"]["kx_new"] > rr["none"]["kx_new"]
           for rr in by.values() if "freeze_zH_i" in rr]
    czh = [rr["cell_zH"]["kx_new"] == -2 or rr["cell_zH"]["kx_new"] > rr["none"]["kx_new"] for rr in by.values() if "cell_zH" in rr]
    S.append(f"  z_H update at call 14 skipped: {p_skip:.0f}% of {len(dl)} later drops delayed; only i's z_H not updated: "
             f"{100 * np.mean(frz):.0f}%; ★'s cell cut inside z_H updates: {100 * np.mean(czh):.0f}%")
    fig = plt.figure(figsize=(17, 6.2))
    title(fig, f"Pruning needs the regular z_H refresh: without the z_H update at call 14, {p_skip:.0f}% of the later drops "
               f"come later")
    ax = fig.add_axes([0.05, 0.24, 0.5, 0.62])
    calls = [c for c in zl if 3 <= c <= 35]
    n0 = np.array([sum(1 for f in P if f["cross_call"] == c) for c in calls])
    n1 = np.array([sum(1 for c2 in new_calls if c2 == c) for c in calls])
    x = np.arange(len(calls))
    ax.bar(x - 0.2, n0, width=0.4, color=INK, label="normal")
    ax.bar(x + 0.2, n1, width=0.4, color=ORANGE, label="no z_H update at call 14 (z_H keeps its call-7 value until call 21)")
    for n in range(1, len(calls)):
        if any(c % 7 == 0 for c in range(calls[n - 1] + 1, calls[n])):
            ax.axvline(n - 0.5, color=MUTED, lw=0.8, ls=":")
            c_zh = next(c for c in range(calls[n - 1] + 1, calls[n]) if c % 7 == 0)
            ax.text(n - 0.5, ax.get_ylim()[1] * 0.98, f"z_H {c_zh}", ha="center", va="top", fontsize=8, color=INK2,
                    backgroundcolor=SURF)
    ax.set_xticks(x, [str(c) for c in calls], fontsize=8); ax.set_xlabel("call of the drop (z_L updates)")
    ax.set_ylabel(f"drops (of {len(P)})"); spines(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2)
    ax.set_title("A   when the drops happen")
    ax = fig.add_axes([0.8, 0.24, 0.17, 0.62])
    rows = [(f"no z_H update at call 7\n(z_H stays at its start value until call 14; drops after call 7, n={len(dl7)})", p_skip7),
            (f"no z_H update at call 14\n(z_H keeps its call-7 value until call 21; drops after call 14, n={len(dl)})", p_skip),
            ("every token's z_H is updated except token i's", 100 * np.mean(frz)),
            ("z_H updated, but ★'s cell's messages cut\ninside the z_H updates", 100 * np.mean(czh))]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[INK, INK, LIGHT, LIGHT], height=0.55)
    for yy, (_, v) in zip(y, rows):
        ax.text(v + 1.5, yy, f"{v:.0f}%", va="center", fontsize=9.5)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8.8); ax.set_xlim(0, 100); spines(ax)
    ax.set_xlabel("% of drops that come later")
    ax.set_title("B   each bar = one rerun with one change")
    fig.savefig(out / "fig4_zH.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------- examples

def figS(d, P, out, S):
    zl = d["zl_calls"]; B = d["boards"]
    labels = ["before", "dilution", "injection\n(z_H + maze)", "att0", "mlp0", "att1", "mlp1", "after"]
    R = list(range(-3, 2))
    # population average of (logit i, logit ★) around the flip, to pick representative examples
    ref = []
    for f in P:
        if f["star"] is None:
            continue
        L = B[f["bid"]]["logits"]
        if f["kx"] + R[0] >= 0 and f["kx"] + R[-1] < len(L):
            ref.append(np.r_[L[f["kx"] + np.array(R), f["token"]], L[f["kx"] + np.array(R), f["star"]]])
    ref = np.mean(ref, 0)
    for dname in DIRS:
        def ok(f):
            if f["dir"] != dname or f["cross_call"] < 8 or f["star_cross_k"] is None or f["star_cross_k"] < 0:
                return False
            L = B[f["bid"]]["logits"]
            stable = f["ko"] >= 3 and (L[f["ko"] - 3:f["ko"], f["token"]] > 1.39).all()
            star_was_path = (L[:f["star_cross_k"], f["star"]] > 1.39).any()
            return stable and star_was_path and f["star_cross_k"] < f["kx"] and f["kx"] + R[-1] < len(L)
        cands = [f for f in P if ok(f)]
        if not cands:
            continue
        def dist(f):
            L = B[f["bid"]]["logits"]
            v = np.r_[L[f["kx"] + np.array(R), f["token"]], L[f["kx"] + np.array(R), f["star"]]]
            return float(((v - ref) ** 2).sum())
        f = min(cands, key=dist)
        bd = B[f["bid"]]; L = bd["logits"]; i = f["token"]; k = f["kx"]
        pcs = bd["pieces"][k, :, i]
        fig = plt.figure(figsize=(17, 10))
        where = {"right": "to the right of", "left": "to the left of", "up": "above", "down": "below"}[dname]
        title(fig, f"Example: dead end {where} token i — the update at call {zl[k]}, in which i flips to off "
                   f"({L[k - 1, i]:+.1f} → {L[k, i]:+.1f})",
              f"board {bd['board']}; chosen as the drop closest to the average time course (figure 3A) among "
              f"{len(cands)} clean ones; boxed = token i, ★ = its neighbour on the dead-end side")
        for n, kk in enumerate((k - 1, k)):
            ax = fig.add_axes([0.02 + n * 0.17, 0.5, 0.16, 0.16 * 17 / 10])
            sol_board(ax, bd)
            r, c = divmod(i, G); ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec=INK, lw=2.2))
            rs, cs = divmod(f["star"], G); ax.text(cs, rs, "★", ha="center", va="center", fontsize=12)
            ax.set_title(f"{'before' if n == 0 else 'after'} call {zl[k]}: i {L[kk, i]:+.1f}, ★ {L[kk, f['star']]:+.1f}")
        ax = fig.add_axes([0.42, 0.52, 0.32, 0.36])
        lv = L[k - 1, i]
        ax.bar(0, lv, width=0.6, color=MUTED); ax.text(0, lv, f"{lv:+.1f}", ha="center", va="bottom" if lv >= 0 else "top", fontsize=9)
        for n, v in enumerate(pcs, start=1):
            lo, hi = sorted((lv, lv + v))
            ax.bar(n, hi - lo, bottom=lo, width=0.6, color=BLUE if v > 0 else ORANGE)
            ax.text(n, hi if v > 0 else lo, f"{v:+.1f}", ha="center", va="bottom" if v > 0 else "top", fontsize=9)
            lv += v
        ax.bar(len(labels) - 1, lv, width=0.6, color=MUTED)
        ax.text(len(labels) - 1, lv, f"{lv:+.1f}", ha="center", va="bottom" if lv >= 0 else "top", fontsize=9)
        cum = np.cumsum(np.r_[L[k - 1, i], pcs])
        ax.set_ylim(min(cum.min(), 0) - 1.2, max(cum.max(), 0) + 1.2)
        ax.axhline(0, color=INK2, lw=0.8); ax.set_xticks(range(len(labels)), labels, fontsize=8.5); spines(ax)
        ax.set_ylabel("token i's probe logit"); ax.set_title("the flip update, piece by piece (adds up exactly)")
        ax = fig.add_axes([0.8, 0.52, 0.18, 0.36])
        ks = list(range(max(1, k - 5), min(d["n_dec"], k + 4)))
        ax.plot([zl[q] for q in ks], L[ks, i], color=INK, lw=2, marker="o", ms=3, label="token i")
        ax.plot([zl[q] for q in ks], L[ks, f["star"]], color=ORANGE, lw=2, marker="o", ms=3, label="★")
        ax.axhline(0, color=INK2, lw=0.8); ax.axvline(zl[k], color=MUTED, lw=0.8, ls=":")
        ax.legend(frameon=False, fontsize=8.5); spines(ax); ax.set_xlabel("call"); ax.set_title("logits around the flip")
        mp = f["maps"].float().numpy()
        pre = list(range(max(0, f["ko"] - 3), f["ko"]))
        for n, (ch, nm) in enumerate((([0, 2], "attention: change of each token's message"),
                                      ([1, 3], "MLP: change of its reaction to each token's message"))):
            chg = mp[k][ch].sum(0) - mp[pre][:, ch].sum(1).mean(0)
            ax = fig.add_axes([0.06 + n * 0.3, 0.05, 0.22, 0.34])
            vmax = nice_vmax(np.abs(window(chg, i).compressed()).max(), lo=0.1)
            im = sender_map(ax, bd, chg, i, f["star"], vmax)
            ax.set_title(nm, fontsize=9.5)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[-vmax, 0, vmax]); cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=7.5)
        fig.text(0.66, 0.2, "Maps: the flip update compared with the 3 updates\nbefore the fall. Orange = the token now pushes i\n"
                            "more towards off. The MLP map shows how much the\nMLP's output depends on each single token's message;\n"
                            "these do not add up to the MLP bars above, because\nthe MLP also reacts to i's own state and to\n"
                            "combinations of messages.", fontsize=9, color=INK2, va="center")
        fig.savefig(out / f"figS_example_{dname}.png", dpi=140); plt.close(fig)
        S.append(f"  example ({dname}): board {bd['board']}, token {divmod(i, G)}, flip at call {zl[k]} ({len(cands)} candidates)")


def make_all(pt, out=None):
    pt = Path(pt); out = Path(out) if out else pt.parent
    out.mkdir(parents=True, exist_ok=True)
    d = torch.load(pt, weights_only=False)
    P = population(d)
    S = [f"model: {d.get('model_kind', 'path')}"]
    for name, fn in (("fig1", fig1), ("fig2", fig2), ("fig3", fig3), ("fig4", fig4), ("figS", figS)):
        try:
            fn(d, P, out, S)
        except Exception as ex:
            S.append(f"{name} not drawn: {ex!r}")
    (out / "summary3.txt").write_text("\n".join(S) + "\n")
    print("\n".join(S))


if __name__ == "__main__":
    make_all(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
