"""
Vendored from google-research/ibc (Apache License 2.0):
  environments/block_pushing/{block_pushing,block_pushing_multimodal}.py
  environments/utils/{utils_pybullet,xarm_sim_robot,pose3d}.py
  environments/assets/*

Needed because block_pushing_multimodal.py's own internal imports are rooted
at `ibc.environments...` (not importable as a bare top-level `block_pushing`
package the way our eval callback originally assumed), and because
block_pushing.py unconditionally imports a `metrics.py` that requires
tf_agents (-> TensorFlow) for a get_metrics() method we never call.

Changes made here (beyond import-path rewrites):
  - the `metrics` import is lazy (inside get_metrics(), which dp_eval_callbacks.py
    never calls) so this package has no TensorFlow dependency
  - *_URDF_PATH constants point at real filesystem paths in ./assets/ instead
    of Google-internal path conventions that don't resolve outside that
    environment

Requires `pybullet-svl` (not plain `pybullet`) — the xarm6_robot.urdf asset
referenced by xarm_sim_robot.py isn't published in google-research/ibc or
real-stanford/diffusion_policy's public repos, but is bundled in pybullet-svl
(Stanford Vision & Learning Lab's pybullet fork, which upstream
diffusion_policy's own conda_environment.yaml pins) at
pybullet_data/xarm/xarm6_robot.urdf.
"""
