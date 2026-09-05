"""Self-consistency test for the AMAZE maze pass metric.

Feed the GROUND-TRUTH solution image (sol_img) into the scorer as if it were the
model's generation. A correct pass metric MUST return pass=1.0 for the GT itself.
If it returns 0, the metric is too strict / misaligned (independent of the model).
"""
import base64, io, json, sys
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, "/home/gosia/STUDIA/DYPLOM/reasoning-diffusion/trm/diffusers/examples/trm_diffusion")
from eval.amaze_eval import AmazeMetrics


def decode(raw):
    if raw is None or (isinstance(raw, float)):
        return None
    if isinstance(raw, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(raw))).convert("RGB")
    if isinstance(raw, str):
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
    return None


def build_meta(row):
    return {
        "id": row.get("id"),
        "metadata": row.get("metadata"),
        "sample_json": row.get("sample_json"),
        "original_img": decode(row.get("original_img")),
        "m_original_img": decode(row.get("m_original_img")),
        "sol_img": decode(row.get("sol_img")),
        "mask_img": decode(row.get("mask_img")),
        "cell_map": decode(row.get("cell_map")),
    }


def pil_to_chw01(img):
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


def main(path, n=12):
    df = pd.read_parquet(path)
    print(f"=== {path}  rows={len(df)} ===")
    # scorer = AmazeMetrics(task="maze")
    scorer = AmazeMetrics(task="queens")
    idxs = np.linspace(0, len(df) - 1, min(n, len(df))).astype(int)
    passes, covs, viols = [], [], []
    for i in idxs:
        row = df.iloc[int(i)]
        meta = build_meta(row)
        gen = pil_to_chw01(meta["sol_img"])          # GT sol as the "generation"
        res = scorer._compute_queen_metrics(gen, meta)
        # also recompute predicted vs gt cell-id sets for detail
        # m = json.loads(meta["metadata"]) if isinstance(meta["metadata"], str) else meta["metadata"]
        # gt = {int(c) for c in m.get("path_cell_ids", [])}
        # cell_ids = scorer.decode_cell_map_ids(meta["cell_map"])
        # size = (cell_ids.shape[1], cell_ids.shape[0])
        # gen_arr = scorer._to_pixel_array(gen, size)
        # from third_party.amaze.infer.maze_metrics import extract_blue_path
        # blue = scorer._morph_open(extract_blue_path(gen_arr))
        # pred = {int(c) for c in cell_ids[blue].tolist()}; pred.discard(0)
        # extra = pred - gt
        # missing = gt - pred
        print(f" row {int(i):4d}  pass={res['pass']:.0f}  cov={res['gt_cell_coverage']:.3f}"
              f"  viol={res['background_violation']:.3f}"  )
            #   |gt|={len(gt)} |pred|={len(pred)}"
            #   f"  extra={len(extra)} missing={len(missing)}")
        passes.append(res["pass"]); covs.append(res["gt_cell_coverage"]); viols.append(res["background_violation"])
    print(f" --- GT-vs-GT  mean pass={np.mean(passes):.3f}  cov={np.mean(covs):.3f}  viol={np.mean(viols):.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/n7_square_test.parquet")
