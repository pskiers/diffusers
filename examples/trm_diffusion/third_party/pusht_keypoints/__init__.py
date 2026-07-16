"""
Vendored from real-stanford/diffusion_policy (MIT license, Copyright (c) 2023
Columbia Artificial Intelligence and Robotics Lab):
  diffusion_policy/env/pusht/{pusht_env,pusht_keypoints_env,pymunk_keypoint_manager,pymunk_override}.py

Needed for exact fidelity to the 9-keypoint PushT-lowdim observation our
PushTLowdimDataset was trained on — the standalone gym-pusht package's
built-in "environment_state_agent_pos" option uses a different, incompatible
8-keypoint scheme, so it can't be used for closed-loop eval of that model.

Only import path changes were made (cross-file imports rewritten to be
relative to this package); the environment logic itself is untouched.
"""
