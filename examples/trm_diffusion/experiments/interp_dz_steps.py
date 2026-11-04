"""TEMPORARY (Claude): does the recursion act as (near) identity? ‖Δz_H‖ per z_H update, PaTh thinker.

Real CFG sampling; on the conditional pass at the denoising steps in `record_steps`, z_H is recorded after every z_H
update (H_cycles per reasoning step x n_sup steps; with 3 x 16 = 48 updates) and z_L after the last z_L update of
every H cycle. Per update u (z_0 = H_init broadcast):
  rel   ‖z_u - z_{u-1}‖ / ‖z_u‖ per token, averaged over tokens (all / GT path tokens / non-path tokens)
  cos   cos(z_u, z_final) per token, averaged
  flip  share of tokens whose argmax(lm_head(z_u)) differs from that at the previous update
  rel_in  ‖z_u - z_{u-1}‖ / ‖input_emb‖, the size of the update relative to the re-injected input
Output: <out_dir>/dz_steps.pt and printed tables.
"""
import json
import logging
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig

from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model
from models.sampling import CFGPredictor
from interp_subtoken_probe import CONFIG_DIR, blue, ints, sub_labels


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig):
    ic = cfg.get("interprebability_config", {})
    out_dir = Path(str(ic.out_dir)); out_dir.mkdir(parents=True, exist_ok=True)
    sizes = ints(ic.get("maze_sizes", "8,16"))
    record_steps = ints(ic.get("record_steps", "0,10,19"))
    data_tmpl = str(ic.get("data_tmpl", "data/amaze/maze/square/n{N}_test.parquet"))
    n_boards, bs, seed = int(ic.get("num_boards", 64)), int(ic.get("batch_size", 32)), int(ic.get("seed", 0))
    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device
    model = build_model(cfg, instantiate(cfg.diffusion))
    load_checkpoint(model, str(ic.checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = accelerator.unwrap_model(accelerator.prepare(model)); model.eval()
    pipe = instantiate(cfg.sampling)
    inner = model.thinker.inner
    per = inner.config.L_cycles + 1

    st = {"on": False}

    def pre_hook(module, args):
        if st["on"]:
            st["call"] += 1

    def level_hook(module, args, output):
        if not st["on"]:
            return
        if st["call"] % per == 0:
            st["zh"].append(output.detach().float())
        else:
            st["zl_last"] = output.detach().float()
            if st["call"] % per == per - 1:
                st["zl"].append(st["zl_last"])

    def enc_hook(module, args, kwargs, out):
        if st["on"]:
            st["enc"] = out.enc_emb.detach().float()

    inner.L_level.register_forward_pre_hook(pre_hook)
    inner.L_level.register_forward_hook(level_hook)
    model.condition_encoder.register_forward_hook(enc_hook, with_kwargs=True)

    def metrics(zs, z0, enc, path):
        """zs: list of (B, T, D) after each update. Returns dict of (U,) tensors."""
        zs = [z0] + zs
        fin = zs[-1]
        out = {k: [] for k in ("rel", "rel_path", "rel_nonpath", "cos", "flip", "rel_in")}
        en = enc.norm(dim=-1)
        prev_arg = inner.lm_head(z0.to(inner.lm_head.weight.dtype)).argmax(-1)
        for a, b in zip(zs[:-1], zs[1:]):
            d = (b - a).norm(dim=-1)
            r = d / b.norm(dim=-1).clamp_min(1e-6)                           # (B, T)
            out["rel"].append(r.mean()); out["rel_path"].append(r[path].mean()); out["rel_nonpath"].append(r[~path].mean())
            out["rel_in"].append((d / en.clamp_min(1e-6)).mean())
            out["cos"].append(torch.nn.functional.cosine_similarity(b, fin, dim=-1).mean())
            arg = inner.lm_head(b.to(inner.lm_head.weight.dtype)).argmax(-1)
            out["flip"].append((arg != prev_arg).float().mean()); prev_arg = arg
        return {k: torch.stack(v).cpu() for k, v in out.items()}

    class RecPredictor(CFGPredictor):
        def predict(self, model_, sample, t, T):
            st.update(on=True, call=0, zh=[], zl=[])
            try:
                pred_cond = model_(sample).pred
            finally:
                st["on"] = False
            if st["step"] in record_steps:
                z0 = inner.H_init.float().view(1, 1, -1).expand_as(st["zh"][0])
                st["rec"][st["step"]] = dict(zH=metrics(st["zh"], z0, st["enc"], st["path"]),
                                             zL=metrics(st["zl"], inner.L_init.float().view(1, 1, -1).expand_as(st["zl"][0]),
                                                        st["enc"], st["path"]),
                                             t=int(t))
            st["step"] += 1
            pred_uncond = model_(model_.null_condition_sample(sample)).pred
            return pred_uncond + self.scale * (pred_cond - pred_uncond)

    pipe.predictor = RecPredictor(pipe.cfg_scale)
    res = {}
    for N in sizes:
        ds = AmazeDataset(dataset_path=data_tmpl.format(N=N), split="test", image_size=144, condition_field="m_original_img",
                          target_field="sol_img", num_channels=3)
        nb_ = min(n_boards, len(ds))
        acc = []
        for k0 in range(0, nb_, bs):
            smps = [ds[i] for i in range(k0, min(nb_, k0 + bs))]
            batch = AmazeDataset.collate_fn(smps)
            st.update(step=0, rec={}, path=(sub_labels(blue(batch.images.float()), 1)[..., 0] == 1).to(device))
            pipe.sample_one_batch(model, model._batch_to_sample(batch, device), device,
                                  generator=torch.Generator(device=device).manual_seed(seed + k0))
            acc.append((len(smps), st["rec"]))
            logging.info(f"n{N}: {k0 + len(smps)}/{nb_}")
        tot = sum(n for n, _ in acc)
        res[N] = {k: {zk: {m: sum(n * r[k][zk][m] for n, r in acc) / tot for m in acc[0][1][k][zk]} for zk in ("zH", "zL")}
                  | {"t": acc[0][1][k]["t"]} for k in record_steps}
        H = inner.config.H_cycles
        for k in record_steps:
            r = res[N][k]["zH"]
            U = len(r["rel"])
            print(f"\n=== n{N}, denoising step {k} (t={res[N][k]['t']}), z_H: {U} updates ({H} per reasoning step)")
            print(f"  {'upd':>4s} {'sup':>4s} {'rel':>7s} {'rel_pth':>7s} {'rel_non':>7s} {'rel_in':>7s} {'cos_fin':>7s} {'flip':>6s}")
            show = list(range(0, min(U, 2 * H))) + list(range(2 * H, U, H))
            for u in show:
                print(f"  {u + 1:4d} {u // H + 1:4d} {r['rel'][u]:7.4f} {r['rel_path'][u]:7.4f} {r['rel_nonpath'][u]:7.4f} "
                      f"{r['rel_in'][u]:7.4f} {r['cos'][u]:7.4f} {r['flip'][u]:6.3f}")
            rl = res[N][k]["zL"]
            print(f"  z_L (last per H cycle) rel at updates 1,2,3,6,12,24,48: "
                  + " ".join(f"{rl['rel'][u - 1]:.4f}" for u in (1, 2, 3, 6, 12, 24, 48) if u <= len(rl["rel"])))
    torch.save(res, out_dir / "dz_steps.pt")
    (out_dir / "dz_steps.json").write_text(json.dumps(
        {str(N): {str(k): {zk: {m: v.tolist() for m, v in d[zk].items()} for zk in ("zH", "zL")} for k, d in r.items()}
         for N, r in res.items()}))
    print(f"saved {out_dir}/dz_steps.pt")


if __name__ == "__main__":
    main()
