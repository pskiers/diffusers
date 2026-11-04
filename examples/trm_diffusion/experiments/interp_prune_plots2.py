"""TEMPORARY (Claude, amaze interpretability): story figures from interp_prune_mech2.py's prune_mech2.pt (CPU only).

    python interp_prune_plots2.py runs/amaze_interp_v34/prune_mech2.pt [out_dir]

Analysed drops = side-branch corridor tokens (inside a cell, or across an open boundary, of a cell that dead-end
filling removes; not tokens across a wall between cells or on cell corners), dropped by call c_dec; a drop = probe logit > +1.39 (p 0.8) at some update, < −1.39 later and < 0 from then on.
fig1_pruning.png   how pruning unfolds: one board over the calls + drops per call on all boards
fig2_causal.png    blocking the tip-side neighbour ★ (z_L / z_H / both) vs controls; one example
fig3_mechanism.png which heads carry ★'s message; what changes in the update when a token falls
fig4_zH.png        the z_H route: timing after z_H updates, z_H vs z_L, skipping a z_H update
figS_one_drop.png  one drop, one update, piece by piece (method explainer)
summary.txt        the numbers
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

from interp_prune_plots import (BLUE, DIV, INK, INK2, LIGHT, MID, MUTED, ORANGE, PIECE_LABELS, SURF, choose_panels,
                                clean_ax, nice_vmax, rot_offset, sender_map, walls, window)

G, T = 12, 144
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "axes.titlesize": 10, "axes.titlelocation": "left"})
ZL_ROW = ["dilution", "injection\n(z_H + maze)", "att0", "mlp0", "att1", "mlp1"]


def crossing_index(L):
    nonneg = np.flatnonzero(L >= 0)
    if len(nonneg) == 0:
        return -1
    if nonneg[-1] == len(L) - 1:
        return -2
    return int(nonneg[-1] + 1)


def title(fig, text):
    fig.text(0.02, 0.975, text, fontsize=12.5, color=INK, weight="bold", va="top")


def spines(ax):
    ax.spines[["top", "right"]].set_visible(False)


def analysed(d):
    return [f for f in d["flips"] if not f["late"] and f["corridor"]]


def sol_board(ax, bd):
    ax.imshow(bd["sol"], extent=(-0.5, G - 0.5, G - 0.5, -0.5), interpolation="bilinear")
    clean_ax(ax)


def delayed_stats(d):
    by = {}
    for r in d["removal"]:
        by.setdefault((r["bid"], r["token"]), {})[r["cond"]] = r
    out = {}
    for key, rr in by.items():
        if "none" not in rr or rr["none"]["kx_new"] < 0:
            continue
        base = rr["none"]["kx_new"]
        for c, r in rr.items():
            if c == "none":
                continue
            x = r["kx_new"]
            out.setdefault(c, []).append(dict(key=key, base=base, new=x, delayed=(x == -2) or x > base,
                                              not_dropped=x == -2, earlier=(0 <= x < base) or x == -1))
    return by, out


# ---------------------------------------------------------------- figure 1

def fig1(d, out, S):
    zl = d["zl_calls"]
    fl = analysed(d)
    xs = np.array([f["round"] - 1 for f in fl]); ys = np.array([f["kx"] for f in fl])
    slope = np.polyfit(xs, ys, 1)[0] if len(fl) > 2 else np.nan
    rho = np.corrcoef(np.argsort(np.argsort(xs)), np.argsort(np.argsort(ys)))[0, 1] if len(fl) > 2 else np.nan
    S.append(f"analysed drops (side-branch corridor tokens): {len(fl)}; slope {slope:.2f} z_L updates per cell from the tip, "
             f"rank correlation {rho:.2f}")
    def grp(f):
        sk = f["star_cross_k"]
        if f["star"] is None:
            return "no ★ found"
        if f["star_before"]:
            return "★ crossed before i started falling"
        if sk == -1:
            return "★ never read as path"
        return "★ crossed while i was falling" if 0 <= sk <= f["kx"] else "★ crossed after i (wrong order)"
    og = Counter(grp(f) for f in fl)
    S.append("order vs ★: " + ", ".join(f"{k} {v} ({v / max(1, len(fl)):.0%})" for k, v in og.most_common()))
    cnt = Counter(f["bid"] for f in fl)
    bid = cnt.most_common(1)[0][0]
    bd = d["boards"][bid]
    fb = [f for f in fl if f["bid"] == bid]
    per_call = Counter(f["cross_call"] for f in fb)
    panels, how = choose_panels(per_call, zl, max_panels=6, quiet=0)
    if how == "groups":                              # also split groups at z_H updates (one panel per H-cycle part)
        split = []
        for g in panels:
            cur = [g[0]]
            for c in g[1:]:
                if c % 7 == 0:
                    continue
                if any(x % 7 == 0 for x in range(cur[-1] + 1, c)):
                    split.append(cur); cur = [c]
                else:
                    cur.append(c)
            split.append(cur)
        panels = [g for g in split if any(per_call.get(c, 0) for c in g)][:6]
    show = [[zl[1]]] + panels + [["last"]]
    side = [t for t in range(T) if bd["rounds"][tuple(bd["tok_cell"][t])] > 0 and bd["ttype"][t] in ("interior", "open_link")]
    fig = plt.figure(figsize=(16, 13))
    head = "Side branches are dropped from the dead-end tip inwards" if rho > 0.3 else "Order in which side-branch tokens are dropped"
    title(fig, f"{head}: {slope:.2f} z_L updates per cell from the tip (rank correlation {rho:.2f}, "
               f"{len(fl)} drops on {len(d['boards'])} boards)")
    for n, grp in enumerate(show[:8]):
        ax = fig.add_axes([0.03 + (n % 4) * 0.245, 0.68 - (n // 4) * 0.285, 0.2, 0.246])
        k = len(zl) - 1 if grp == ["last"] else zl.index(grp[-1])
        sol_board(ax, bd)
        for t in side:
            r, c = divmod(t, G)
            dropping = grp != ["last"] and any(f["token"] == t and f["cross_call"] in grp for f in fb)
            if dropping:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fc=ORANGE, ec=INK, lw=1.0, alpha=0.95))
                if len(grp) > 1:
                    ax.text(c, r, str(next(f["cross_call"] for f in fb if f["token"] == t)), ha="center", va="center",
                            fontsize=6.5, color=INK)
            elif bd["logits"][k, t] >= 0:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fc=BLUE, ec="none", alpha=0.28))
            else:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fc=ORANGE, ec="none", alpha=0.45))
        if grp == ["last"]:
            ttl = "last call"
        elif len(grp) == 1:
            ttl = f"call {grp[0]}"
        else:
            ttl = f"calls {grp[0]}–{grp[-1]}"
        if grp != ["last"] and (grp[0] - 1) % 7 == 0 and grp[0] > 1:
            ttl += "  (after z_H update)"
        ax.set_title(ttl, fontsize=9.5)
    fig.text(0.02, 0.375, "side-branch tokens:", fontsize=9, color=INK2)
    for x0, (fc, a, ec, lab) in zip((0.11, 0.22, 0.31), ((BLUE, 0.28, "none", "still read as path"), (ORANGE, 0.45, "none", "dropped"),
                                                       (ORANGE, 0.95, INK, "dropped in this panel (number = call)"))):
        axl = fig.add_axes([x0, 0.371, 0.01, 0.013]); axl.add_patch(Rectangle((0, 0), 1, 1, fc=fc, ec=ec, alpha=a))
        clean_ax(axl); fig.text(x0 + 0.014, 0.375, lab, fontsize=9, color=INK2)
    # B: drops per call, all boards
    ax = fig.add_axes([0.06, 0.06, 0.9, 0.25])
    cmax = max([f["cross_call"] for f in fl] + [21])
    calls = [c for c in zl if c <= cmax]
    conf = lambda f: d["boards"][f["bid"]]["logits_pc"][f["ko"] - 1, f["token"]] > 1.39 if f["ko"] - 1 < d["n_dec"] else True
    yc = np.array([sum(1 for f in fl if f["cross_call"] == c and conf(f)) for c in calls])
    yu = np.array([sum(1 for f in fl if f["cross_call"] == c and not conf(f)) for c in calls])
    walls_n = np.array([sum(1 for f in d["flips"] if f["ttype"] in ("wall_link", "corner") and f["cross_call"] == c) for c in calls])
    ax.bar(calls, yc, color=ORANGE, width=0.8, label="side-branch corridor drops")
    ax.bar(calls, yu, bottom=yc, color=ORANGE, alpha=0.35, width=0.8, label="… not 'path' for a probe fitted at the peak's call")
    ax.step(calls, walls_n, where="mid", color=MUTED, lw=1.3, label="tokens on walls between cells / cell corners (not analysed)")
    for c in range(7, cmax + 1, 7):
        ax.axvline(c, color=INK2, lw=0.8, ls=(0, (2, 2)))
        ax.text(c, ax.get_ylim()[1], "z_H", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(0, cmax + 1, 7 if cmax > 30 else 1))
    ax.set_xlabel("call at which the token is dropped"); ax.set_ylabel("drops (all boards)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right"); spines(ax)
    S.append(f"drops confirmed by the per-call probe at their peak: {np.mean([conf(f) for f in fl]):.0%}")
    fig.savefig(out / "fig1_pruning.png", dpi=140); plt.close(fig)
    return bid


# ---------------------------------------------------------------- figure 2

COND = [("star_zL", "★  ·  z_L updates"), ("star_zH", "★  ·  z_H updates"), ("star_both", "★  ·  both"),
        ("cell_zL", "★'s cell  ·  z_L updates"), ("cell_zH", "★'s cell  ·  z_H updates"), ("cell_both", "★'s cell  ·  both"),
        ("other_both", "control: other corridor neighbour"), ("wall_both", "control: wall-covered neighbour"),
        ("far_both", "control: random far token"), ("freeze_zH_i", "token i's own z_H not updated")]


def fig2(d, out, S):
    zl = d["zl_calls"]
    by, stats = delayed_stats(d)
    rows = [(c, lab, stats.get(c, [])) for c, lab in COND]
    pct = {c: (100 * np.mean([s["delayed"] for s in v]) if v else np.nan) for c, _, v in rows}
    ctrl = max(pct[c] for c in ("other_both", "wall_both", "far_both") if np.isfinite(pct[c]))
    fig = plt.figure(figsize=(16, 7.2))
    head = ("Blocking what the tip-side neighbour ★ sends delays the drop" if pct["cell_both"] > ctrl + 20
            else "Blocking what the tip-side neighbour ★ sends")
    title(fig, f"{head}: {pct['cell_both']:.0f}% of drops delayed (★'s cell, both routes); controls ≤ {ctrl:.0f}%")
    ax = fig.add_axes([0.2, 0.1, 0.36, 0.78])
    y = np.arange(len(rows))[::-1]
    cols = [INK if c.startswith(("star", "cell")) else (MUTED if c == "freeze_zH_i" else LIGHT) for c, _, _ in rows]
    ax.barh(y, [pct[c] for c, _, _ in rows], color=cols, height=0.62)
    for yy, (c, _, v) in zip(y, rows):
        nd = 100 * np.mean([s["not_dropped"] for s in v]) if v else np.nan
        ax.text((pct[c] if np.isfinite(pct[c]) else 0) + 1.5, yy, f"{pct[c]:.0f}%  (n={len(v)}, not dropped {nd:.0f}%)",
                va="center", fontsize=8.5, color=INK)
        S.append(f"  {c:12s} n={len(v):4d} delayed {pct[c]:5.1f}%  not dropped {nd:5.1f}%")
    ax.set_yticks(y, [lab for _, lab, _ in rows], fontsize=9); ax.set_xlim(0, 150); ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("% of drops delayed (dropped later, or still path at the end of the check)"); spines(ax)
    ax.set_title("messages to token i replaced by their mean")
    agree = np.mean([rr["none"]["kx_new"] == rr["none"]["kx"] for rr in by.values() if "none" in rr])
    S.append(f"removal: {len(by)} drops; unedited rerun reproduces the crossing in {agree:.0%}")
    # B: representative example: median delay among delayed cell_both drops
    v = [s for s in stats.get("cell_both", []) if s["delayed"] and s["new"] >= 0]
    ax = fig.add_axes([0.63, 0.14, 0.34, 0.68])
    if v:
        dl = np.array([s["new"] - s["base"] for s in v])
        s = v[int(np.argsort(dl)[len(dl) // 2])]
        rr = by[s["key"]]
        n_k = len(rr["none"]["traj"])
        cx = zl[:n_k]
        ax.plot(cx, rr["none"]["traj"], color=INK2, lw=2, label="unedited")
        ax.plot(cx, rr["cell_both"]["traj"][:n_k], color=ORANGE, lw=2, label="★'s cell blocked")
        for c in range(7, cx[-1] + 1, 7):
            ax.axvline(c, color=INK2, lw=0.8, ls=(0, (2, 2)))
        for y_ in (1.39, -1.39):
            ax.axhline(y_, color=MUTED, lw=0.7, ls=(0, (3, 3)))
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axvline(rr["cell_both"]["start_call"], color=ORANGE, lw=0.8, ls=":")
        ax.legend(frameon=False, fontsize=9); spines(ax)
        bd = d["boards"][s["key"][0]]
        ax.set_title(f"a typical delayed drop (median delay): board {bd['board']}, token {divmod(s['key'][1], G)}\n"
                     "dashed = z_H updates, dotted = blocking starts")
        ax.set_xlabel("call"); ax.set_ylabel("token i's probe logit")
    fig.savefig(out / "fig2_causal.png", dpi=140); plt.close(fig)
    return (v and s["key"]) or None


# ---------------------------------------------------------------- figure 3

def fig3(d, out, S):
    fl = [f for f in analysed(d) if f["star"] is not None and f["ko"] >= 1]
    adj = lambda f: abs(f["star"] // G - f["token"] // G) + abs(f["star"] % G - f["token"] % G) == 1
    W_PRE = d["cfg"]["W_PRE"]
    before, during, mlp_chg = [], [], []
    for f in fl:
        P = d["boards"][f["bid"]]["pieces"][:, :, f["token"]]
        pre = list(range(max(0, f["ko"] - W_PRE), f["ko"]))
        if not pre:
            continue
        before.append(P[pre].mean(0)); during.append(P[f["ko"]:f["kx"] + 1].mean(0))
    before, during = np.array(before), np.array(during)
    mb, md = before.mean(0), during.mean(0)
    mlp_b, mlp_d = mb[3] + mb[5], md[3] + md[5]; att_b, att_d = mb[2] + mb[4], md[2] + md[4]
    fig = plt.figure(figsize=(16, 6.4))
    head = ("When a token falls, its MLPs stop rewriting 'path'" if (mlp_b - mlp_d) > 2 * abs(att_b - att_d)
            else "What changes in the update when a token falls")
    title(fig, f"{head}: MLP push {mlp_b:+.1f} → {mlp_d:+.1f} per update, attention {att_b:+.1f} → {att_d:+.1f}")
    S.append(f"pieces per update, before → during the fall (n={len(before)}): "
             + ", ".join(f"{n} {a:+.2f}→{b:+.2f}" for n, a, b in zip(d["pieces_names"], mb, md)))
    # A: pieces before vs during
    ax = fig.add_axes([0.05, 0.14, 0.33, 0.7])
    x = np.arange(6)
    for off, v, col, lab in ((-0.19, before, LIGHT, f"3 updates before the fall"), (0.19, during, INK, "during the fall")):
        m, se = v.mean(0), v.std(0) / np.sqrt(len(v))
        ax.bar(x + off, m, yerr=se, width=0.36, color=col, label=lab, error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    ax.axhline(0, color=INK2, lw=0.8); ax.set_xticks(x, ZL_ROW, fontsize=8.5)
    ax.set_ylabel("push on the token's probe logit per update"); ax.legend(frameon=False, fontsize=9); spines(ax)
    ax.set_title(f"same {len(before)} tokens: what each part of the update adds")
    # B: heads
    ax = fig.add_axes([0.47, 0.14, 0.22, 0.7])
    dirs = {"left": (0, -1), "right": (0, 1), "up": (-1, 0), "down": (1, 0)}
    rows, labels = [], []
    for l in range(2):
        for name, dv in dirs.items():
            sel = [f for f in fl if adj(f) and (f["star"] // G - f["token"] // G, f["star"] % G - f["token"] % G) == dv]
            n_idx = 1 + list(dirs).index(name)
            rows.append(np.mean([f["head_nb"][f["ko"]:f["kx"] + 1, l, :, n_idx].float().numpy().sum(0) for f in sel], 0)
                        if sel else np.full(8, np.nan))
            labels.append(f"L{l} · ★ {name} (n={len(sel)})")
    Hm = np.array(rows); vm = nice_vmax(np.nanmax(np.abs(Hm)))
    im = ax.imshow(np.ma.masked_invalid(Hm), cmap=DIV, vmin=-vm, vmax=vm, aspect="auto")
    for (r, c), v in np.ndenumerate(Hm):
        if np.isfinite(v) and abs(v) >= 0.25 * vm:
            ax.text(c, r, f"{v:+.2f}", ha="center", va="center", fontsize=7.5)
    ax.set_xticks(range(8), [f"H{h}" for h in range(8)], fontsize=8); ax.set_yticks(range(8), labels, fontsize=8)
    ax.axhline(3.5, color=SURF, lw=3); ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title("which head carries ★'s message\n(direct push ★ → i, onset … crossing)")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, ticks=[-vm, 0, vm]); cb.outline.set_visible(False)
    cb.ax.set_yticklabels([f"−{vm} off", "0", f"+{vm} path"], fontsize=8)
    # C: MLP reaction change map aligned
    W = 2
    acc = np.zeros((2 * W + 1, 2 * W + 1)); cnt = np.zeros_like(acc)
    for f in [f for f in fl if adj(f)]:
        mp = f["maps"].float().numpy()
        mlp = mp[:, 1] + mp[:, 3]
        pre = list(range(max(0, f["ko"] - W_PRE), f["ko"]))
        if not pre:
            continue
        chg = mlp[f["ko"]:f["kx"] + 1].mean(0) - mlp[pre].mean(0)
        i, s = f["token"], f["star"]
        sdir = (s // G - i // G, s % G - i % G)
        for j in range(T):
            dr, dc = j // G - i // G, j % G - i % G
            if abs(dr) <= W and abs(dc) <= W:
                rr, cc = rot_offset(dr, dc, sdir)
                acc[rr + W, cc + W] += chg[j]; cnt[rr + W, cc + W] += 1
    M = acc / np.maximum(cnt, 1); vm = nice_vmax(np.abs(M).max())
    ax = fig.add_axes([0.77, 0.14, 0.2, 0.7])
    ax.imshow(M, cmap=DIV, vmin=-vm, vmax=vm)
    for (r, c), v in np.ndenumerate(M):
        if abs(v) >= 0.15 * vm:
            ax.text(c, r + 0.3, f"{v:+.2f}", ha="center", va="center", fontsize=7.5)
    ax.add_patch(Rectangle((W - 0.5, W - 0.5), 1, 1, fill=False, ec=INK, lw=1.8))
    ax.text(W, W - 1.08, "★", ha="center", va="center", fontsize=11); ax.text(W, W - 0.1, "i", ha="center", fontsize=10)
    clean_ax(ax)
    ax.set_title("which neighbour the MLPs react to\n(change during the fall, ★ above i)")
    fig.savefig(out / "fig3_mechanism.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------- figure 4

def fig4(d, out, S):
    zl, zh = d["zl_calls"], d["zh_calls"]; per = d["cfg"]["per"]
    fl = analysed(d)
    late = [f for f in fl if f["cross_call"] > per]
    pos = np.array([f["cross_call"] % per for f in late])
    pc = np.array([(pos == p).sum() for p in range(1, per)]) / max(1, len(pos)) * 100
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.6), gridspec_kw=dict(wspace=0.38, left=0.04, right=0.98, top=0.8, bottom=0.2))
    head = ("Drops cluster right after z_H updates" if pc[0] > 1.5 * 100 / (per - 1) else "When drops cross within the H-cycle")
    title(fig, f"{head}: {pc[0]:.0f}% cross at the 1st z_L update after a z_H update (even spread: {100 / (per - 1):.0f}%)")
    ax = axes[0]
    ax.bar(range(1, per), pc, color=[INK] + [LIGHT] * (per - 2), width=0.6)
    for x_, v in zip(range(1, per), pc):
        ax.text(x_, v + 0.8, f"{v:.0f}%", ha="center", fontsize=8.5)
    ax.axhline(100 / (per - 1), color=INK2, lw=1, ls=(0, (3, 3)))
    ax.set_xticks(range(1, per), ["1st"] + [str(p) for p in range(2, per)])
    ax.set_xlabel("z_L update of the H-cycle at which the token crosses"); ax.set_ylabel("% of drops"); spines(ax)
    ax.set_title(f"when drops cross (n={len(late)}, after call {per})")
    # B: z_H vs z_L
    diffs, never = [], 0
    for f in fl:
        zk = crossing_index(d["boards"][f["bid"]]["zh_logits"][:, f["token"]])
        if zk < 0:
            never += 1
            continue
        diffs.append(zh[zk] - f["cross_call"])
    diffs = np.array(diffs)
    ax = axes[1]
    bins = np.arange(-21.5, 22.5, 1)
    ax.hist(np.clip(diffs, -21, 21), bins=bins, color=INK)
    ax.axvline(0, color=ORANGE, lw=1.2)
    ax.set_xlabel("z_H crossing call − z_L crossing call  (< 0: z_H first)"); ax.set_ylabel("drops"); spines(ax)
    ax.set_title(f"does z_H drop the token before z_L?\nz_H first {np.mean(diffs < 0):.0%}, never drops in z_H: {never}")
    S.append(f"z_H vs z_L crossing: z_H first {np.mean(diffs < 0):.0%}, median diff {np.median(diffs):+.0f} calls, never in z_H {never}")
    # C: skip a z_H update
    ax = axes[2]
    res = []
    for name, c0, lab in (("skip_zH7", 7, "no z_H update at call 7\n(z_H stays at its initial value)"), ("skip_zH14", 14, "no z_H update at call 14\n(z_H stays at its call-7 value)")):
        dl = []
        for f in fl:
            if f["cross_call"] <= c0:
                continue
            La = d["skips"][f["bid"]][name][:, f["token"]]
            La = La[np.isfinite(La)]
            kx_base = crossing_index(d["boards"][f["bid"]]["logits"][:len(La), f["token"]])
            kx_new = crossing_index(La)
            if kx_base < 0:
                continue
            dl.append(kx_new == -2 or kx_new > kx_base)
        res.append((lab, 100 * np.mean(dl) if dl else np.nan, len(dl)))
    _, stats = delayed_stats(d)
    v = stats.get("freeze_zH_i", [])
    res.append(("only token i's z_H not updated", 100 * np.mean([s["delayed"] for s in v]) if v else np.nan, len(v)))
    y = np.arange(len(res))[::-1]
    ax.barh(y, [r[1] for r in res], color=INK, height=0.55)
    for yy, (lab, p, n) in zip(y, res):
        ax.text(p + 1.5, yy, f"{p:.0f}% (n={n})", va="center", fontsize=8.5)
        S.append(f"  {lab}: delayed {p:.1f}% (n={n})")
    ax.set_yticks(y, [r[0] for r in res], fontsize=8.5); ax.set_xlim(0, 110); spines(ax)
    ax.set_xlabel("% of later drops delayed"); ax.set_title("skipping a z_H update")
    # D: where blocked drops land
    ax = axes[3]
    for off, c, col, lab in ((-0.19, "cell_zL", INK, "★'s cell blocked in z_L"), (0.19, "cell_both", ORANGE, "… in z_L and z_H")):
        v = [s for s in stats.get(c, []) if s["delayed"] and s["new"] >= 0]
        p = np.array([zl[s["new"]] % per for s in v])
        h = np.array([(p == q).sum() for q in range(1, per)]) / max(1, len(p)) * 100
        ax.bar(np.arange(1, per) + off, h, width=0.36, color=col, label=f"{lab} (n={len(v)})")
    ax.axhline(100 / (per - 1), color=INK2, lw=1, ls=(0, (3, 3)))
    ax.set_xticks(range(1, per), ["1st"] + [str(p) for p in range(2, per)]); ax.legend(frameon=False, fontsize=8.5)
    ax.set_xlabel("z_L update of the H-cycle"); ax.set_ylabel("% of delayed drops"); spines(ax)
    ax.set_title("where delayed drops finally cross")
    fig.savefig(out / "fig4_zH.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------- supplement

def figS(d, out, key):
    zl = d["zl_calls"]
    f = next((f for f in analysed(d) if key and (f["bid"], f["token"]) == tuple(key)), None) or \
        next((f for f in analysed(d) if f["star"] is not None and f["star_before"]), None)
    if f is None:
        return
    bd = d["boards"][f["bid"]]; L = bd["logits"]; i = f["token"]
    pcs = bd["pieces"][:, :, i]
    win = list(range(f["ko"], f["kx"] + 1))
    k = win[int(np.argmin([pcs[kk].sum() for kk in win]))]
    maps = f["maps"].float().numpy()[k]
    fig = plt.figure(figsize=(16, 9.4))
    title(fig, f"One drop, one update (call {zl[k]}): how each part of the update moved token i's logit, and from which tokens")
    for n, kk in enumerate((k - 1, k)):
        ax = fig.add_axes([0.02 + n * 0.2, 0.5, 0.18, 0.38])
        sol_board(ax, bd)
        r, c = divmod(i, G)
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec=INK, lw=2.2))
        if f["star"] is not None:
            rs, cs = divmod(f["star"], G); ax.text(cs, rs, "★", ha="center", va="center", fontsize=12)
        ax.set_title(f"{'before' if n == 0 else 'after'} call {zl[k]}: logit {L[kk, i] if kk >= 0 else d['logit_init']:+.2f}")
    axB = fig.add_axes([0.47, 0.52, 0.48, 0.36])
    before = L[k - 1, i]; level = before
    labels = ["before"] + ZL_ROW + ["after"]
    axB.bar(0, before, width=0.6, color=MUTED)
    for n, v in enumerate(pcs[k], start=1):
        lo, hi = sorted((level, level + v))
        axB.bar(n, hi - lo, bottom=lo, width=0.6, color=BLUE if v > 0 else ORANGE)
        axB.text(n, hi if v > 0 else lo, f"{v:+.2f}", ha="center", va="bottom" if v > 0 else "top", fontsize=8.5)
        level += v
    axB.bar(len(labels) - 1, level, width=0.6, color=MUTED)
    axB.text(len(labels) - 1, level, f"{level:+.2f}", ha="center", va="bottom" if level >= 0 else "top", fontsize=8.5)
    axB.text(0, before, f"{before:+.2f}", ha="center", va="bottom" if before >= 0 else "top", fontsize=8.5)
    lv = np.cumsum(np.r_[before, pcs[k]])
    axB.set_ylim(min(lv.min(), 0) - 0.8, max(lv.max(), 0) + 0.8)
    axB.axhline(0, color=INK2, lw=0.8); axB.set_xticks(range(len(labels)), labels, fontsize=8.5); spines(axB)
    axB.set_ylabel("token i's probe logit"); axB.set_title("the update, piece by piece (adds up exactly)")
    tot = maps.sum(0)
    vmax = nice_vmax(max(np.abs(window(m, i).compressed()).max() for m in list(maps) + [tot]))
    names = ["att0: direct messages", "mlp0: MLP's reaction", "att1: direct messages", "mlp1: MLP's reaction", "total"]
    for n, m in enumerate(list(maps) + [tot]):
        ax = fig.add_axes([0.02 + n * 0.195, 0.08, 0.17, 0.3])
        im = sender_map(ax, bd, m, i, f["star"], vmax)
        ax.set_title(names[n], fontsize=9)
    cb = fig.colorbar(im, cax=fig.add_axes([0.35, 0.03, 0.3, 0.012]), orientation="horizontal", ticks=[-vmax, 0, vmax])
    cb.ax.set_xticklabels([f"−{vmax}: towards off", "0", f"+{vmax}: towards path"], fontsize=8); cb.outline.set_visible(False)
    fig.savefig(out / "figS_one_drop.png", dpi=140); plt.close(fig)


def make_all(pt, out=None):
    pt = Path(pt); out = Path(out) if out else pt.parent
    out.mkdir(parents=True, exist_ok=True)
    d = torch.load(pt, weights_only=False)
    tt = Counter(f["ttype"] for f in d["flips"])
    S = [f"boards {len(d['boards'])}, all drops {len(d['flips'])} by token type {dict(tt)}, "
         f"side-branch corridor {sum(f['corridor'] for f in d['flips'])}, "
         f"replay |Δlogit| max {d['replay_err'].max():.3g}, split sum error max {d['sum_err'].max():.3g}"]
    S.append(f"model: {d.get('model_kind', 'path')}")
    key = None
    for name, fn in (("fig1", lambda: fig1(d, out, S)), ("fig2", lambda: fig2(d, out, S)), ("fig3", lambda: fig3(d, out, S)),
                     ("fig4", lambda: fig4(d, out, S))):
        try:
            r = fn()
            if name == "fig2":
                key = r
        except Exception as ex:                   # e.g. no analysed drops / no removal runs: keep the other figures
            S.append(f"{name} not drawn: {ex!r}")
    try:
        figS(d, out, key)
    except Exception as ex:
        S.append(f"figS not drawn: {ex!r}")
    (out / "summary.txt").write_text("\n".join(S) + "\n")
    print("\n".join(S))


if __name__ == "__main__":
    make_all(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
