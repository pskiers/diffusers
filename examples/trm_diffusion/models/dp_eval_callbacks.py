"""
models/dp_eval_callbacks.py — Closed-loop evaluation callbacks for diffusion policy tasks.

Unlike eval_callbacks.py (which does batch inference on a dataloader), these
callbacks run actual gym simulations to measure closed-loop task performance.

Callback interface:
    callback(model, dataloader, accelerator, **kwargs) -> dict[str, float]

Model contract:
    model.predict_action(obs_dict) -> {'action': tensor (B, T_a, Da)}

Supported environments:
    PushT      — gym-pusht (pip install gym-pusht pymunk shapely)
    BlockPush  — google-research block_pushing
    ToolHang   — robosuite (pip install robosuite robomimic)
"""

import logging
import collections

import numpy as np
import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


# ── ObsBuffer ─────────────────────────────────────────────────────────────────


class ObsBuffer:
    """Rolling window buffer for the last n_obs_steps observations."""

    def __init__(self, n_obs_steps):
        self.n_obs_steps = n_obs_steps
        self._buf = None

    def reset(self, first_obs):
        # first_obs: dict of str -> np.ndarray (single obs)
        # fill buffer with first_obs repeated n_obs_steps times
        self._buf = {
            k: collections.deque([v] * self.n_obs_steps, maxlen=self.n_obs_steps)
            for k, v in first_obs.items()
        }

    def push(self, obs):
        for k, v in obs.items():
            self._buf[k].append(v)

    def get(self):
        # returns dict of str -> np.ndarray (n_obs_steps, *obs_shape)
        return {k: np.stack(list(v)) for k, v in self._buf.items()}


# ── PushT ─────────────────────────────────────────────────────────────────────


class PushTImageEvalCallback:
    """Closed-loop evaluation on the PushT environment.

    Paper metric: mean of per-episode max coverage (area fraction of T-block
    overlapping target region), reported as a fraction in [0, 1].

    Paper result: 0.73 mean coverage (DDPM-CNN baseline).

    Supports two obs modalities:
      obs_type='pixels_agent_pos' (default) — image + agent_pos via the
        gym-pusht package, for PushTHybridDataset-trained painters
        (spatial_conditions + embedding_conditions).
      obs_type='keypoints' — the vendored third_party/pusht_keypoints env
        (real-stanford/diffusion_policy's own PushTKeypointsEnv), for
        PushTLowdimDataset-trained painters (embedding_conditions only).
        gym-pusht's own built-in keypoints option uses a different,
        incompatible 8-keypoint scheme and can't be used here — our dataset
        was built with upstream's 9-keypoint PushTKeypointsEnv.

    Requires: pip install gym-pusht "pymunk<7" shapely gym pygame scikit-image
    (gym/pygame/scikit-image are only needed for obs_type='keypoints').
    """

    def __init__(
        self,
        normalizer=None,
        n_eval_episodes=50,
        max_steps=300,
        n_obs_steps=2,
        n_action_steps=8,
        obs_type='pixels_agent_pos',
        test_start_seed=100000,
        render=False,
        device='cpu',
    ):
        self.normalizer = normalizer
        self.n_eval_episodes = n_eval_episodes
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.obs_type = obs_type
        self.test_start_seed = test_start_seed
        self.render = render
        self.device = device

    def _raw_obs_to_dict(self, obs):
        # obs_type='pixels_agent_pos': dict with 'pixels' (96,96,3) uint8 and
        # 'agent_pos' (2,) float32.
        return {'image': obs['pixels'], 'agent_pos': obs['agent_pos']}

    def _stacked_to_obs_tensor(self, stacked_obs, device):
        normalizer = self.normalizer
        image_norm = stacked_obs['image'].astype(np.float32) / 255.0 * 2.0 - 1.0
        image_norm = image_norm.transpose(0, 3, 1, 2)  # (T, 3, H, W)

        agent_pos = stacked_obs['agent_pos'].astype(np.float32)
        try:
            agent_pos_t = torch.from_numpy(agent_pos).unsqueeze(0).to(device)
            agent_pos = normalizer['agent_pos'].normalize(agent_pos_t).squeeze(0).cpu().numpy()
        except (KeyError, Exception):
            pass

        # Keys match the DataSample fields the painter's condition
        # encoder reads (see models/condition_encoders.py).
        return {
            'spatial_conditions': torch.from_numpy(image_norm).unsqueeze(0).to(device),
            'embedding_conditions': torch.from_numpy(agent_pos).unsqueeze(0).to(device),
        }

    def _unnormalize_action(self, actions, device):
        try:
            actions_t = torch.from_numpy(actions).unsqueeze(0).to(device)
            action_exec = self.normalizer['action'].unnormalize(actions_t)
            return action_exec.squeeze(0).cpu().numpy()
        except (KeyError, Exception):
            return actions

    def _run_pixels_agent_pos(self, model):
        try:
            # gym-pusht registers its env under the gymnasium registry, not
            # the legacy `gym` package (confirmed against its PyPI metadata:
            # it depends on gymnasium>=0.29.1, not gym).
            import gymnasium as gym
            import gym_pusht  # noqa: F401
        except ImportError:
            logger.warning(
                "gym-pusht not installed. Skipping PushT eval. "
                "pip install gym-pusht \"pymunk<7\" shapely"
            )
            return None

        device = self.device
        obs_buffer = ObsBuffer(self.n_obs_steps)

        env = gym.make('gym_pusht/PushT-v0', obs_type='pixels_agent_pos')
        all_max_coverage = []

        seeds = range(self.test_start_seed, self.test_start_seed + self.n_eval_episodes)
        pbar = tqdm(seeds, desc="PushT eval (pixels_agent_pos)", total=self.n_eval_episodes, leave=False)
        for seed in pbar:
            obs, info = env.reset(seed=seed)
            obs_buffer.reset(self._raw_obs_to_dict(obs))

            max_coverage = 0.0
            done = False
            step = 0

            while not done and step < self.max_steps:
                stacked_obs = obs_buffer.get()
                obs_tensor = self._stacked_to_obs_tensor(stacked_obs, device)

                with torch.no_grad():
                    action_dict = model.predict_action(obs_tensor)

                actions = action_dict['action'][0].cpu().numpy()  # (T_a, 2) normalized
                action_exec = self._unnormalize_action(actions, device)

                T_a = action_exec.shape[0]
                for t in range(min(self.n_action_steps, T_a)):
                    obs, reward, terminated, truncated, info = env.step(action_exec[t])
                    coverage = info.get('coverage', reward if reward is not None else 0.0)
                    max_coverage = max(max_coverage, coverage)
                    obs_buffer.push(self._raw_obs_to_dict(obs))
                    step += 1
                    done = terminated or truncated
                    if done:
                        break
                pbar.set_postfix(step=f"{step}/{self.max_steps}", coverage=f"{max_coverage:.3f}")

            all_max_coverage.append(max_coverage)
            pbar.set_postfix(mean_coverage=f"{np.mean(all_max_coverage):.3f}")

        env.close()
        return all_max_coverage

    def _run_keypoints(self, model):
        try:
            from third_party.pusht_keypoints.pusht_keypoints_env import PushTKeypointsEnv
            from third_party.pusht_keypoints.pusht_env import pymunk_to_shapely
        except ImportError as exc:
            logger.warning(
                "PushT keypoints env dependencies not installed. Skipping PushT eval. "
                "pip install \"pymunk<7\" shapely gym pygame scikit-image matplotlib opencv-python "
                f"(missing: {exc})"
            )
            return None

        device = self.device
        obs_buffer = ObsBuffer(self.n_obs_steps)

        kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
        env = PushTKeypointsEnv(legacy=False, keypoint_visible_rate=1.0, agent_keypoints=False, **kp_kwargs)
        all_max_coverage = []

        def coverage_of(env):
            goal_body = env._get_goal_pose_body(env.goal_pose)
            goal_geom = pymunk_to_shapely(goal_body, env.block.shapes)
            block_geom = pymunk_to_shapely(env.block, env.block.shapes)
            return goal_geom.intersection(block_geom).area / goal_geom.area

        seeds = range(self.test_start_seed, self.test_start_seed + self.n_eval_episodes)
        pbar = tqdm(seeds, desc="PushT eval (keypoints)", total=self.n_eval_episodes, leave=False)
        for seed in pbar:
            env.seed(seed)
            raw_obs = env.reset()
            Do = raw_obs.shape[-1] // 2
            obs_buffer.reset({'state': raw_obs[:Do].astype(np.float32)})

            max_coverage = 0.0
            done = False
            step = 0

            while not done and step < self.max_steps:
                state = obs_buffer.get()['state'].astype(np.float32)
                try:
                    state_t = torch.from_numpy(state).unsqueeze(0).to(device)
                    state = self.normalizer['obs'].normalize(state_t).squeeze(0).cpu().numpy()
                except (KeyError, Exception):
                    pass
                obs_tensor = {'embedding_conditions': torch.from_numpy(state).unsqueeze(0).to(device)}

                with torch.no_grad():
                    action_dict = model.predict_action(obs_tensor)

                actions = action_dict['action'][0].cpu().numpy()  # (T_a, 2) normalized
                action_exec = self._unnormalize_action(actions, device)

                T_a = action_exec.shape[0]
                for t in range(min(self.n_action_steps, T_a)):
                    raw_obs, reward, done, info = env.step(action_exec[t])
                    max_coverage = max(max_coverage, coverage_of(env))
                    obs_buffer.push({'state': raw_obs[:Do].astype(np.float32)})
                    step += 1
                    if done:
                        break
                pbar.set_postfix(step=f"{step}/{self.max_steps}", coverage=f"{max_coverage:.3f}")

            all_max_coverage.append(max_coverage)
            pbar.set_postfix(mean_coverage=f"{np.mean(all_max_coverage):.3f}")

        env.close()
        return all_max_coverage

    def __call__(self, model, dataloader, accelerator, **kwargs):
        if self.obs_type == 'keypoints':
            all_max_coverage = self._run_keypoints(model)
        else:
            all_max_coverage = self._run_pixels_agent_pos(model)

        if all_max_coverage is None:
            return {}

        return {
            'pusht/mean_coverage': float(np.mean(all_max_coverage)),
            'pusht/median_coverage': float(np.median(all_max_coverage)),
        }


# ── BlockPush ─────────────────────────────────────────────────────────────────


class BlockPushEvalCallback:
    """Closed-loop evaluation on BlockPush environment (16D state, 2D action).

    Metrics:
      blockpush/mean_score: mean normalized score per episode
      blockpush/p1: fraction of episodes with at least 1 block pushed to target
      blockpush/p2: fraction of episodes with both blocks pushed to target

    Requires the block_pushing environment. If not available, returns {}.
    Install: follow instructions at https://github.com/real-stanford/diffusion_policy
    or the google-research block-pushing repository.
    """

    def __init__(
        self,
        normalizer=None,
        n_eval_episodes=50,
        max_steps=350,
        n_obs_steps=2,
        n_action_steps=8,
        test_start_seed=100000,
        device='cpu',
    ):
        self.normalizer = normalizer
        self.n_eval_episodes = n_eval_episodes
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.test_start_seed = test_start_seed
        self.device = device

    def _make_env(self):
        try:
            # Vendored (third_party/block_pushing/) — google-research/ibc's
            # own `block_pushing` isn't pip-installable and its internal
            # imports are rooted at `ibc.environments...`, not a bare
            # top-level `block_pushing` package. Requires pybullet-svl (not
            # plain pybullet) for the xarm6_robot.urdf asset — see
            # third_party/block_pushing/__init__.py.
            from third_party.block_pushing import block_pushing_multimodal
            return block_pushing_multimodal.BlockPushMultimodal(
                control_frequency=10,
                shared_memory=False,
            )
        except ImportError as exc:
            logger.warning("Failed to create BlockPush env (ImportError): %s", exc)

        return None

    def __call__(self, model, dataloader, accelerator, **kwargs):
        env = self._make_env()
        if env is None:
            logger.warning(
                "BlockPush env not available. Skipping. "
                "See diffusion_policy repo for setup."
            )
            return {}

        device = self.device
        normalizer = self.normalizer
        obs_buffer = ObsBuffer(self.n_obs_steps)

        all_scores = []
        n_p1 = 0
        n_p2 = 0

        pbar = tqdm(range(self.n_eval_episodes), desc="BlockPush eval", leave=False)
        for episode_idx in pbar:
            seed = self.test_start_seed + episode_idx
            try:
                obs = env.reset(seed=seed)
            except TypeError:
                try:
                    obs, _info = env.reset(seed=seed)
                except TypeError:
                    # Env doesn't support a seed kwarg at all — fall back to
                    # unseeded reset rather than failing eval entirely.
                    try:
                        obs = env.reset()
                    except TypeError:
                        obs, _info = env.reset()

            # obs is a 16D state vector or a dict; normalise to ndarray
            if isinstance(obs, dict):
                obs_arr = np.concatenate([np.asarray(v).flatten() for v in obs.values()]).astype(np.float32)
            else:
                obs_arr = np.array(obs, dtype=np.float32)

            obs_buffer.reset({'state': obs_arr})

            # Upstream (blockpush_lowdim_runner.py) scores an episode as the
            # sum of *unique* reward values observed over the rollout — the
            # env's reward is a per-block completion indicator that jumps to
            # a new fixed value the first time each block reaches its target,
            # not a running max of a single 'score' info key.
            seen_rewards = set()
            episode_score = 0.0
            done = False
            step = 0

            while not done and step < self.max_steps:
                stacked_obs = obs_buffer.get()  # {'state': (n_obs_steps, D)}
                state = stacked_obs['state'].astype(np.float32)

                # Normalize state if possible
                try:
                    state_t = torch.from_numpy(state).unsqueeze(0).to(device)
                    state_norm = normalizer['obs'].normalize(state_t)
                    state = state_norm.squeeze(0).cpu().numpy()
                except (KeyError, Exception):
                    pass

                # Key matches the DataSample field the painter's condition
                # encoder reads (see models/condition_encoders.py).
                obs_tensor = {
                    'embedding_conditions': torch.from_numpy(state).unsqueeze(0).to(device),
                }

                with torch.no_grad():
                    action_dict = model.predict_action(obs_tensor)

                actions = action_dict['action'][0].cpu().numpy()  # (T_a, Da) normalized

                # Unnormalize actions
                try:
                    actions_t = torch.from_numpy(actions).unsqueeze(0).to(device)
                    action_exec = normalizer['action'].unnormalize(actions_t)
                    action_exec = action_exec.squeeze(0).cpu().numpy()
                except (KeyError, Exception):
                    action_exec = actions

                T_a = action_exec.shape[0]
                for t in range(min(self.n_action_steps, T_a)):
                    step_result = env.step(action_exec[t])
                    # Support both (obs, reward, done, info) and (obs, reward, terminated, truncated, info)
                    if len(step_result) == 5:
                        obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        obs, reward, done_flag, info = step_result
                        done = done_flag

                    reward_key = round(float(reward), 6)
                    if reward_key not in seen_rewards:
                        seen_rewards.add(reward_key)
                        episode_score += float(reward)

                    if isinstance(obs, dict):
                        obs_arr = np.concatenate([np.asarray(v).flatten() for v in obs.values()]).astype(np.float32)
                    else:
                        obs_arr = np.array(obs, dtype=np.float32)
                    obs_buffer.push({'state': obs_arr})

                    step += 1
                    if done:
                        break
                pbar.set_postfix(step=f"{step}/{self.max_steps}", score=f"{episode_score:.3f}")

            all_scores.append(episode_score)
            # Thresholds match upstream blockpush_lowdim_runner.py.
            if episode_score > 0.4:
                n_p1 += 1
            if episode_score > 0.9:
                n_p2 += 1
            pbar.set_postfix(mean_score=f"{np.mean(all_scores):.3f}", p1=n_p1, p2=n_p2)

        env.close()

        n = self.n_eval_episodes
        return {
            'blockpush/mean_score': float(np.mean(all_scores)),
            'blockpush/p1': float(n_p1 / n),
            'blockpush/p2': float(n_p2 / n),
        }


# ── ToolHang ──────────────────────────────────────────────────────────────────


class ToolHangEvalCallback:
    """Closed-loop evaluation on ToolHang environment (robosuite).

    Metric:
      toolhang/success_rate: fraction of episodes with successful completion

    Requires: pip install robosuite robomimic
    """

    def __init__(
        self,
        normalizer=None,
        n_eval_episodes=50,
        max_steps=700,
        n_obs_steps=2,
        n_action_steps=8,
        use_camera_obs=True,
        obs_keys=None,
        test_start_seed=100000,
        device='cpu',
    ):
        self.normalizer = normalizer
        self.n_eval_episodes = n_eval_episodes
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.use_camera_obs = use_camera_obs
        self.test_start_seed = test_start_seed
        if obs_keys is not None:
            self.obs_keys = obs_keys
        elif use_camera_obs:
            # Camera + proprioceptive keys for ToolHangImageDataset-trained
            # painters (sideview, not agentview — matches upstream tool_hang_image.yaml).
            self.obs_keys = [
                'robot0_eye_in_hand_image',
                'sideview_image',
                'robot0_eef_pos',
                'robot0_eef_quat',
                'robot0_gripper_qpos',
            ]
        else:
            # Lowdim keys for ToolHangLowdimDataset-trained painters
            # (object-first, matching the dataset's obs concat order).
            self.obs_keys = [
                'object',
                'robot0_eef_pos',
                'robot0_eef_quat',
                'robot0_gripper_qpos',
            ]
        self.device = device

    def _make_env(self):
        try:
            # The pinned robosuite 1.2.0 fork (cheng-chi/robosuite) still does
            # `from collections import Iterable` / `collections.Iterable` in
            # a couple of places (multi_table_arena.py, placement_samplers.py)
            # — that alias moved to collections.abc in Python 3.3 and was
            # hard-removed from collections itself in Python 3.10. Restore it
            # before robosuite (or anything it imports) runs.
            import collections
            import collections.abc
            if not hasattr(collections, "Iterable"):
                collections.Iterable = collections.abc.Iterable
            import robosuite as suite
            kwargs = dict(
                robots='Panda',
                has_renderer=False,
                has_offscreen_renderer=self.use_camera_obs,
                use_camera_obs=self.use_camera_obs,
                reward_shaping=False,
                control_freq=20,
                horizon=self.max_steps,
            )
            if self.use_camera_obs:
                kwargs.update(
                    camera_names=['sideview', 'robot0_eye_in_hand'],
                    camera_heights=240,
                    camera_widths=240,
                )
            env = suite.make('ToolHang', **kwargs)
            return env
        except ImportError as exc:
            logger.warning("Failed to create ToolHang env (ImportError): %s", exc)
            return None
        except Exception as exc:
            logger.warning("Failed to create ToolHang env: %s", exc)
            return None

    def _obs_to_dict(self, raw_obs):
        """Extract and stack relevant observation keys from a robosuite obs dict."""
        result = {}
        for key in self.obs_keys:
            if key in raw_obs:
                val = np.array(raw_obs[key], dtype=np.float32)
                result[key] = val
        return result

    def _build_obs_tensor(self, stacked_obs):
        """Assemble the raw per-key robosuite obs into the DataSample fields
        the painter's condition encoder reads.

        Image variant (use_camera_obs=True): spatial_conditions (two camera
        views stacked, matching ToolHangImageDataset's view order [sideview,
        hand]) + embedding_conditions (robot proprioception, normalized with
        the 'obs_robot' global-scalar normalizer).

        Lowdim variant (use_camera_obs=False): embedding_conditions only,
        object + robot proprioception concatenated object-first (matching
        ToolHangLowdimDataset's obs order) and normalized with the 'obs'
        global-scalar normalizer.
        """
        normalizer = self.normalizer

        if not self.use_camera_obs:
            obs_lowdim = np.concatenate([
                stacked_obs['object'],
                stacked_obs['robot0_eef_pos'],
                stacked_obs['robot0_eef_quat'],
                stacked_obs['robot0_gripper_qpos'],
            ], axis=-1).astype(np.float32)  # (T, 53)
            obs_t = torch.from_numpy(obs_lowdim).unsqueeze(0).to(self.device)
            try:
                obs_t = normalizer['obs'].normalize(obs_t)
            except (KeyError, Exception):
                pass
            return {'embedding_conditions': obs_t}

        def _img(key):
            arr = stacked_obs[key].astype(np.float32) / 255.0 * 2.0 - 1.0
            return torch.from_numpy(arr.transpose(0, 3, 1, 2))  # (T, 3, H, W)

        views = [
            _img(key) for key in ('sideview_image', 'robot0_eye_in_hand_image') if key in stacked_obs
        ]
        spatial_conditions = torch.stack(views, dim=1).unsqueeze(0).to(self.device) if views else None

        embedding_conditions = None
        if all(k in stacked_obs for k in ('robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos')):
            obs_robot = np.concatenate([
                stacked_obs['robot0_eef_pos'],
                stacked_obs['robot0_eef_quat'],
                stacked_obs['robot0_gripper_qpos'],
            ], axis=-1).astype(np.float32)  # (T, 9)
            try:
                obs_t = torch.from_numpy(obs_robot).unsqueeze(0).to(self.device)
                obs_t = normalizer['obs_robot'].normalize(obs_t)
            except (KeyError, Exception):
                obs_t = torch.from_numpy(obs_robot).unsqueeze(0).to(self.device)
            embedding_conditions = obs_t

        obs_tensor = {}
        if spatial_conditions is not None:
            obs_tensor['spatial_conditions'] = spatial_conditions
        if embedding_conditions is not None:
            obs_tensor['embedding_conditions'] = embedding_conditions
        return obs_tensor

    def __call__(self, model, dataloader, accelerator, **kwargs):
        env = self._make_env()
        if env is None:
            logger.warning(
                "robosuite not installed or ToolHang env unavailable. "
                "Skipping ToolHang eval. pip install robosuite robomimic"
            )
            return {}

        normalizer = self.normalizer
        obs_buffer = ObsBuffer(self.n_obs_steps)
        n_success = 0

        pbar = tqdm(range(self.n_eval_episodes), desc="ToolHang eval", leave=False)
        for episode_idx in pbar:
            seed = self.test_start_seed + episode_idx
            try:
                env.seed(seed)  # robosuite/legacy-gym seeding API
            except (AttributeError, Exception):
                pass
            raw_obs = env.reset()
            first_obs = self._obs_to_dict(raw_obs)
            obs_buffer.reset(first_obs)

            done = False
            step = 0
            success = False

            while not done and step < self.max_steps:
                stacked_obs = obs_buffer.get()
                obs_tensor = self._build_obs_tensor(stacked_obs)

                with torch.no_grad():
                    action_dict = model.predict_action(obs_tensor)

                actions = action_dict['action'][0].cpu().numpy()  # (T_a, Da) normalized

                # Unnormalize actions
                try:
                    actions_t = torch.from_numpy(actions).unsqueeze(0).to(self.device)
                    action_exec = normalizer['action'].unnormalize(actions_t)
                    action_exec = action_exec.squeeze(0).cpu().numpy()
                except (KeyError, Exception):
                    action_exec = actions

                T_a = action_exec.shape[0]
                for t in range(min(self.n_action_steps, T_a)):
                    raw_obs, reward, done_flag, info = env.step(action_exec[t])
                    # robosuite returns done when horizon is reached
                    done = bool(done_flag)
                    if isinstance(info, dict) and info.get('success', False):
                        success = True
                    obs_buffer.push(self._obs_to_dict(raw_obs))
                    step += 1
                    if done:
                        break
                pbar.set_postfix(step=f"{step}/{self.max_steps}", success=success)

            if success:
                n_success += 1
            pbar.set_postfix(success_rate=f"{n_success / (episode_idx + 1):.3f}")

        env.close()

        return {
            'toolhang/success_rate': float(n_success / self.n_eval_episodes),
        }


# ── Download helpers ───────────────────────────────────────────────────────────


def get_download_instructions():
    """Print download instructions for all supported datasets."""
    instructions = """
Diffusion Policy Dataset Download Instructions
==============================================

PushT:
  python -c "from datasets.pusht_dataset import download_pusht; download_pusht('data')"

BlockPush:
  python -c "from datasets.blockpush_dataset import download_block_push; download_block_push('data')"

ToolHang (lowdim, ~300MB):
  python -c "from datasets.toolhang_dataset import download_tool_hang; download_tool_hang('data', 'lowdim')"

ToolHang (image, ~78GB):
  python -c "from datasets.toolhang_dataset import download_tool_hang; download_tool_hang('data', 'image')"

Environment Dependencies:
  PushT:     pip install gym-pusht pymunk shapely
  BlockPush: See https://github.com/real-stanford/diffusion_policy for setup
  ToolHang:  pip install robosuite robomimic
"""
    print(instructions)
