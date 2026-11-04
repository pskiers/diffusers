"""TEMPORARY (Claude): the plain TRM's version of probe_corrections_t70.png (and a trace for decision_order_v2.py).

Same probe and plot as ZPathEncodingAblation (its MLP probe TokenPathClassifier via _train_probe, trained on the LAST z_L
update; its _plot_corrections), applied to the plain TRM (maze only, no diffusion). 2-fold cross-fit over the usable
n8_test boards (probe trained on one half, applied to the other), so every board gets probabilities.
Usage (from examples/trm_diffusion): python plain_corrections.py <plain checkpoint> <n8_test.parquet> <out_dir> [probe_steps=3000]
Output: probe_corrections_tplain_w18.png, probe_corrections_tplain_w36.png, zstate_trace_plain.pt
"""
import shutil
import sys
from pathlib import Path

import torch

import maze_corruption_lib as mcl
from datasets.amaze_dataset import AmazeDataset
from experiments.interpretability.z_path_encoding_ablation import ZPathEncodingAblation as Z
from train_plain_trm import PlainTRM


def main(ckpt, parquet, out_dir, probe_steps=3000):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    plain = PlainTRM(n_sup=int(ck["args"].get("n_sup", 16))); plain.load_state_dict(ck["ema"]); plain = plain.to(device).eval()
    inner = plain.thinker.inner
    per = inner.config.L_cycles + 1

    ds = AmazeDataset(dataset_path=parquet, split="test", image_size=144, condition_field="m_original_img",
                      target_field="sol_img", num_channels=3, include_metadata=True)
    idx = [i for i in range(len(ds)) if mcl.build_maze_sample(ds[i].metadata) is not None]
    batch = AmazeDataset.collate_fn([ds[i] for i in idx])
    maze, sol = batch.spatial_conditions, batch.images
    labels = Z._token_labels(sol, 12, 0.1)                                         # (N, 144): 1 path, 0 off, -1 ignored

    st = {"mode": None, "call": 0, "out": []}

    def hook(module, args, kwargs, output):
        st["call"] += 1
        if st["call"] % per == 0:                                                  # z_H update
            if st["mode"] == "apply":
                st["out"].append(st["out"][-1])                                    # keep the last z_L reading
            return None
        if st["mode"] == "last":
            st["out"] = [output.detach().float()]
        else:
            B, T, D = output.shape
            st["out"].append(torch.sigmoid(st["probe"](output.float().reshape(-1, D))).reshape(B, T).half().cpu())
        return None
    inner.L_level.register_forward_hook(hook, with_kwargs=True)

    def run(x, mode, probe=None):
        outs = []
        for k in range(0, len(x), 50):
            st.update(mode=mode, call=0, out=[], probe=probe)
            with torch.no_grad():
                plain.think(x[k:k + 50].to(device))
            outs.append(st["out"][-1].cpu() if mode == "last" else torch.stack(st["out"], 1))
        return torch.cat(outs)

    z_last = run(maze, "last")                                                     # (N, 144, 512)
    N = len(idx)
    folds = [list(range(0, N, 2)), list(range(1, N, 2))]
    P = torch.zeros(N, inner.config.H_cycles * per * plain.n_sup, 144, dtype=torch.float16)
    for f in range(2):
        tr, te = folds[f], folds[1 - f]
        y = labels[tr].flatten(); keep = y >= 0
        probe = Z._train_probe(z_last[tr].flatten(0, 1)[keep], y[keep], use_nonlinearity=True, steps=probe_steps, device=device)
        P[te] = run(maze[te], "apply", probe)
        print(f"fold {f}: probe on {len(tr)} boards applied to {len(te)}", flush=True)

    zl = [c for c in range(P.shape[1]) if (c + 1) % per != 0]                      # z_L calls (0-based call index)
    p_zl = P[:, zl].float()                                                        # (N, 288, 144)
    for w in (18, 36):
        Z._plot_corrections(f"plain{w}", p_zl, labels, sol, idx, w, 12, out_dir)
        shutil.move(out_dir / f"probe_corrections_tplain{w}.png", out_dir / f"probe_corrections_tplain_w{w}.png")
    u8 = lambda x: (x.clamp(0, 1) * 255).round().to(torch.uint8)
    torch.save({70: {"p": P}, "sol": u8(sol), "maze": u8(maze), "clean": u8(maze), "gt": labels, "per": per,
                "n_sup": plain.n_sup, "boards": idx}, out_dir / "zstate_trace_plain.pt")
    final_ok = ((P[:, -1].float() > 0.5) == (labels == 1))[labels >= 0].float().mean()
    print(f"{N} boards; probe agreement with the labels at the last call {final_ok:.3f}; wrote {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], *(int(a) for a in sys.argv[4:]))
