"""
steering_weight_experiment.py — Did the ControlNet/IP-Adapter steering
pathway learn anything at all?

Cheapest possible check for the "stronger conditioning signal drowns out the
weaker/steering one" hypothesis: ControlNet's zero-convs and IP-Adapter's
to_out_ip are zero-initialized by design (so training starts identical to
the unsteered baseline). If they're still ~zero after thousands of training
steps, the frozen painter's own conditioning already explains the loss well
enough that the optimizer never found a use for the steering pathway — no
eval run needed to see this, just inspect the checkpoint.

Usage:
    python experiments/steering_weight_experiment.py \\
        runs/pusht_hybrid_unet1d_thinker_controlnet/checkpoint_step-40000.pt
    python experiments/steering_weight_experiment.py \\
        runs/clevr_thinker_v0_controlnet_bit_unet/checkpoint_final.pt --state ema_state
"""
import argparse
import re

import torch

# Zero-initialized weight patterns, across every steering mechanism in this
# repo — a given checkpoint will only ever match one of these depending on
# which translator/painter it used.
ZERO_INIT_PATTERNS = [
    re.compile(r".*\bzero_convs\.\d+\.(weight|bias)$"),  # ConditioningPyramid / ConditioningPyramid1D
    re.compile(r".*\bmid_zero_conv\.(weight|bias)$"),     # ConditioningPyramid / ConditioningPyramid1D
    re.compile(r".*\bto_out_ip\.(weight|bias)$"),         # _IPAdapterDiTBlock / _DirectIPAdapterDiTBlock / _IPAdapterEncoderLayer
]


def find_zero_init_params(state_dict):
    return {k: v for k, v in state_dict.items() if any(p.match(k) for p in ZERO_INIT_PATTERNS)}


def check_state(label, state_dict):
    found = find_zero_init_params(state_dict)
    if not found:
        print(f"\n[{label}] No zero-init steering params found — this checkpoint may not be a "
              "thinker-steered model, or uses a steering mechanism not covered by ZERO_INIT_PATTERNS.")
        return

    print(f"\n[{label}] {len(found)} zero-init steering param(s):")
    max_norm = 0.0
    for key, tensor in sorted(found.items()):
        norm = tensor.float().norm().item()
        max_norm = max(max_norm, norm)
        flag = "  <-- still ~zero-init" if norm < 1e-3 else ""
        print(f"    {key:<70s} norm={norm:.6f}{flag}")

    if max_norm < 1e-3:
        print(f"\n  ALL steering weights in [{label}] are still ~zero-init — the steering pathway "
              "has not learned anything distinguishable from the unsteered baseline.")
    else:
        print(f"\n  Steering weights in [{label}] have moved (max norm={max_norm:.6f}) — something "
              "was learned; use eval.py's +steering_scale=0.0/2.0/5.0/10.0 ablation/amplification "
              "to check whether it actually helps at rollout time.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="Path to a checkpoint_*.pt file saved by train_trm.py")
    parser.add_argument(
        "--state", choices=["both", "model_state", "ema_state"], default="both",
        help="Which weights to check — ema_state is what eval.py actually uses by default (use_ema=True)."
    )
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise SystemExit(f"{args.checkpoint} doesn't look like a train_trm.py checkpoint (no 'model_state' key).")

    print(f"Checkpoint: {args.checkpoint}  (step={ckpt.get('step')})")

    if args.state in ("both", "model_state"):
        check_state("model_state", ckpt["model_state"])
    if args.state in ("both", "ema_state"):
        if ckpt.get("ema_state"):
            check_state("ema_state", ckpt["ema_state"])
        elif args.state == "ema_state":
            print("No ema_state in this checkpoint.")


if __name__ == "__main__":
    main()
