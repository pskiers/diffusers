"""
Vendored from real-stanford/diffusion_policy's OWN fork of google-research/ibc
(Apache License 2.0), not the raw upstream google-research/ibc:
  diffusion_policy/env/block_pushing/{block_pushing,block_pushing_multimodal}.py
  diffusion_policy/env/block_pushing/utils/{utils_pybullet,xarm_sim_robot,pose3d}.py
  diffusion_policy/env/block_pushing/assets/*

This matters: the raw google-research/ibc BlockPushMultimodal env (originally
vendored here) is NOT behaviorally equivalent to the fork diffusion_policy
actually used to collect the published multimodal_push_seed.zarr training
data. Two concrete differences that mattered:
  - diffusion_policy's fork's _reset_target_poses() adds an explicit
    `add = 0.12 * rng.choice([-1, 1])` left/right bias when placing the two
    targets — this is the actual "multimodal" part of the task (which side
    each target ends up on). The raw google-research/ibc version has no such
    bias. Evaluating a diffusion_policy-trained painter against the
    unbiased env means every rollout's initial condition is out of the
    training distribution — this alone was enough to produce ~0% success
    rate on an otherwise correctly-trained policy.
  - _get_reward() differs: google-research/ibc's is binary (0 or 1, only
    when both blocks reach different targets); diffusion_policy's fork
    additionally emits 0.49 per individual block reaching a target, so
    partial credit is observable per-step. Our BlockPushEvalCallback's p1/p2
    scoring (sum of unique reward values, thresholds >0.4/>0.9) is written
    against *this* reward scheme, matching upstream's blockpush_lowdim_runner.py.

Only the import-path rewrites below were needed on top of a straight copy —
this fork already resolves its own "third_party/py/envs/assets/..."-style
URDF path constants to a local assets/ directory at runtime (see
utils/utils_pybullet.py's load_urdf()), and has no tf_agents/TensorFlow
dependency at all (unlike the old google-research/ibc copy, which needed a
lazy-import workaround for its metrics.py).

Changes made here:
  - imports rewritten from `diffusion_policy.env.block_pushing...` to
    `third_party.block_pushing...`
  - the gym<0.26 `registration.registry.env_specs` deregistration blocks
    (only relevant for interactive/notebook module reloads) are wrapped in
    try/except AttributeError — gym>=0.26's registry is a plain dict with
    no .env_specs attribute

Requires `pybullet-svl` (not plain `pybullet`) — the xarm6_robot.urdf asset
referenced by xarm_sim_robot.py isn't published in google-research/ibc or
real-stanford/diffusion_policy's public repos, but is bundled in pybullet-svl
(Stanford Vision & Learning Lab's pybullet fork, which upstream
diffusion_policy's own conda_environment.yaml pins) at
pybullet_data/xarm/xarm6_robot.urdf.
"""
