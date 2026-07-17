"""
models/dp_eval_callbacks.py — Closed-loop evaluation callbacks for diffusion policy tasks.

Unlike eval_callbacks.py (which does batch inference on a dataloader), these
callbacks run actual gym simulations to measure closed-loop task performance.

Callback interface:
    callback(model, dataloader, accelerator, **kwargs) -> dict[str, float]

Model contract:
    model.predict_action(obs_dict, n_action_steps=N) -> {'action': tensor (B, N, Da)}
    (already sliced to the executable [n_obs_steps-1 : n_obs_steps-1+N] window
    of the predicted horizon — see ActionPainterBase.predict_action)

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


class VectorizedObsBuffer:
    """A list of n_envs independent ObsBuffer instances, stacked into batched
    arrays for one model.predict_action(...) call per chunk-replan instead of
    n_envs separate batch-1 calls.

    Reuses ObsBuffer's existing per-env reset/push logic unchanged — this
    only adds the stacking-across-envs step, which is where the actual
    speedup comes from (model.predict_action is already batch-agnostic, see
    models/sampling.py::_batch_size_of).
    """

    def __init__(self, n_envs, n_obs_steps):
        self.n_envs = n_envs
        self._buffers = [ObsBuffer(n_obs_steps) for _ in range(n_envs)]

    def reset(self, i, first_obs):
        self._buffers[i].reset(first_obs)

    def push(self, i, obs):
        self._buffers[i].push(obs)

    def get_batched(self):
        """dict of str -> np.ndarray (n_envs, n_obs_steps, *obs_shape)."""
        per_env = [b.get() for b in self._buffers]
        keys = per_env[0].keys()
        return {k: np.stack([e[k] for e in per_env], axis=0) for k in keys}


def _chunk_sizes(n_eval_episodes, n_envs):
    """[n_envs, n_envs, ..., remainder] summing to n_eval_episodes, no padding."""
    sizes = []
    remaining = n_eval_episodes
    while remaining > 0:
        sizes.append(min(n_envs, remaining))
        remaining -= sizes[-1]
    return sizes


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
        n_eval_episodes=128,
        max_steps=300,
        n_obs_steps=2,
        n_action_steps=8,
        obs_type='pixels_agent_pos',
        test_start_seed=100000,
        n_envs=32,
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
        self.n_envs = n_envs
        self.render = render
        self.device = device

    def _raw_obs_to_dict(self, obs):
        # obs_type='pixels_agent_pos': dict with 'pixels' (96,96,3) uint8 and
        # 'agent_pos' (2,) float32.
        return {'image': obs['pixels'], 'agent_pos': obs['agent_pos']}

    def _stacked_to_obs_tensor(self, stacked_obs, device):
        """stacked_obs: dict of (N, n_obs_steps, *shape) arrays from
        VectorizedObsBuffer.get_batched() — N is the chunk's active env count."""
        normalizer = self.normalizer
        image_norm = stacked_obs['image'].astype(np.float32) / 255.0 * 2.0 - 1.0
        image_norm = image_norm.transpose(0, 1, 4, 2, 3)  # (N, T, 3, H, W)

        agent_pos = stacked_obs['agent_pos'].astype(np.float32)  # (N, T, 2)
        try:
            agent_pos_t = torch.from_numpy(agent_pos).to(device)
            agent_pos = normalizer['agent_pos'].normalize(agent_pos_t).cpu().numpy()
        except (KeyError, Exception):
            pass

        # Keys match the DataSample fields the painter's condition
        # encoder reads (see models/condition_encoders.py).
        return {
            'spatial_conditions': torch.from_numpy(image_norm).to(device),
            'embedding_conditions': torch.from_numpy(agent_pos).to(device),
        }

    def _unnormalize_action(self, actions, device):
        """actions: (N, T_a, action_dim)."""
        try:
            actions_t = torch.from_numpy(actions).to(device)
            action_exec = self.normalizer['action'].unnormalize(actions_t)
            return action_exec.cpu().numpy()
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
        all_max_coverage = []
        episode_offset = 0

        pbar = tqdm(total=self.n_eval_episodes, desc="PushT eval (pixels_agent_pos)", leave=False)
        for chunk_size in _chunk_sizes(self.n_eval_episodes, self.n_envs):
            envs = [gym.make('gym_pusht/PushT-v0', obs_type='pixels_agent_pos') for _ in range(chunk_size)]
            obs_buffer = VectorizedObsBuffer(chunk_size, self.n_obs_steps)
            max_coverage = np.zeros(chunk_size, dtype=np.float64)
            done_flags = np.zeros(chunk_size, dtype=bool)
            steps = np.zeros(chunk_size, dtype=np.int64)

            for i, env in enumerate(envs):
                obs, info = env.reset(seed=self.test_start_seed + episode_offset + i)
                obs_buffer.reset(i, self._raw_obs_to_dict(obs))

            while not np.all(done_flags | (steps >= self.max_steps)):
                stacked_obs = obs_buffer.get_batched()
                obs_tensor = self._stacked_to_obs_tensor(stacked_obs, device)

                with torch.no_grad():
                    action_dict = model.predict_action(obs_tensor, n_action_steps=self.n_action_steps)

                actions = action_dict['action'].cpu().numpy()  # (N, T_a, 2) normalized
                action_exec = self._unnormalize_action(actions, device)

                T_a = action_exec.shape[1]
                for t in range(min(self.n_action_steps, T_a)):
                    for i, env in enumerate(envs):
                        if done_flags[i] or steps[i] >= self.max_steps:
                            continue
                        obs, reward, terminated, truncated, info = env.step(action_exec[i, t])
                        coverage = info.get('coverage', reward if reward is not None else 0.0)
                        max_coverage[i] = max(max_coverage[i], coverage)
                        obs_buffer.push(i, self._raw_obs_to_dict(obs))
                        steps[i] += 1
                        if terminated or truncated:
                            done_flags[i] = True
                    if np.all(done_flags | (steps >= self.max_steps)):
                        break
                pbar.set_postfix(
                    active=int((~done_flags).sum()),
                    max_step=int(steps.max()),
                    coverage=f"{max_coverage.mean():.3f}",
                )

            for env in envs:
                env.close()
            all_max_coverage.extend(max_coverage.tolist())
            episode_offset += chunk_size
            pbar.update(chunk_size)
            pbar.set_postfix(mean_coverage=f"{np.mean(all_max_coverage):.3f}")
        pbar.close()

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
        all_max_coverage = []
        episode_offset = 0

        def make_env():
            kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
            return PushTKeypointsEnv(legacy=False, keypoint_visible_rate=1.0, agent_keypoints=False, **kp_kwargs)

        def coverage_of(env):
            goal_body = env._get_goal_pose_body(env.goal_pose)
            goal_geom = pymunk_to_shapely(goal_body, env.block.shapes)
            block_geom = pymunk_to_shapely(env.block, env.block.shapes)
            return goal_geom.intersection(block_geom).area / goal_geom.area

        pbar = tqdm(total=self.n_eval_episodes, desc="PushT eval (keypoints)", leave=False)
        for chunk_size in _chunk_sizes(self.n_eval_episodes, self.n_envs):
            envs = [make_env() for _ in range(chunk_size)]
            obs_buffer = VectorizedObsBuffer(chunk_size, self.n_obs_steps)
            max_coverage = np.zeros(chunk_size, dtype=np.float64)
            done_flags = np.zeros(chunk_size, dtype=bool)
            steps = np.zeros(chunk_size, dtype=np.int64)
            Do = None

            for i, env in enumerate(envs):
                env.seed(self.test_start_seed + episode_offset + i)
                raw_obs = env.reset()
                if Do is None:
                    Do = raw_obs.shape[-1] // 2
                obs_buffer.reset(i, {'state': raw_obs[:Do].astype(np.float32)})

            while not np.all(done_flags | (steps >= self.max_steps)):
                state = obs_buffer.get_batched()['state'].astype(np.float32)  # (N, T, Do)
                try:
                    state_t = torch.from_numpy(state).to(device)
                    state = self.normalizer['obs'].normalize(state_t).cpu().numpy()
                except (KeyError, Exception):
                    pass
                obs_tensor = {'embedding_conditions': torch.from_numpy(state).to(device)}

                with torch.no_grad():
                    action_dict = model.predict_action(obs_tensor, n_action_steps=self.n_action_steps)

                actions = action_dict['action'].cpu().numpy()  # (N, T_a, 2) normalized
                action_exec = self._unnormalize_action(actions, device)

                T_a = action_exec.shape[1]
                for t in range(min(self.n_action_steps, T_a)):
                    for i, env in enumerate(envs):
                        if done_flags[i] or steps[i] >= self.max_steps:
                            continue
                        raw_obs, reward, done, info = env.step(action_exec[i, t])
                        max_coverage[i] = max(max_coverage[i], coverage_of(env))
                        obs_buffer.push(i, {'state': raw_obs[:Do].astype(np.float32)})
                        steps[i] += 1
                        if done:
                            done_flags[i] = True
                    if np.all(done_flags | (steps >= self.max_steps)):
                        break
                pbar.set_postfix(
                    active=int((~done_flags).sum()),
                    max_step=int(steps.max()),
                    coverage=f"{max_coverage.mean():.3f}",
                )

            all_max_coverage.extend(max_coverage.tolist())
            episode_offset += chunk_size
            pbar.update(chunk_size)
            pbar.set_postfix(mean_coverage=f"{np.mean(all_max_coverage):.3f}")
        pbar.close()

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
        n_eval_episodes=128,
        max_steps=350,
        n_obs_steps=2,
        n_action_steps=8,
        test_start_seed=100000,
        n_envs=32,
        device='cpu',
    ):
        self.normalizer = normalizer
        self.n_eval_episodes = n_eval_episodes
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.test_start_seed = test_start_seed
        self.n_envs = n_envs
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

    @staticmethod
    def _obs_to_arr(obs):
        if isinstance(obs, dict):
            return np.concatenate([np.asarray(v).flatten() for v in obs.values()]).astype(np.float32)
        return np.array(obs, dtype=np.float32)

    def _reset_env(self, env, seed):
        # BlockPushMultimodal (gym<0.26-era API) has no `seed` kwarg on
        # reset() — it's a separate .seed() method that reseeds the internal
        # RandomState, called once before reset(). Without this, every
        # "episode" was silently continuing one shared unseeded RNG stream
        # instead of using deterministic per-episode seeds.
        try:
            env.seed(seed)
        except AttributeError:
            pass
        try:
            obs = env.reset(seed=seed)
        except TypeError:
            try:
                obs, _info = env.reset(seed=seed)
            except TypeError:
                try:
                    obs = env.reset()
                except TypeError:
                    obs, _info = env.reset()
        return obs

    def __call__(self, model, dataloader, accelerator, **kwargs):
        probe_env = self._make_env()
        if probe_env is None:
            logger.warning(
                "BlockPush env not available. Skipping. "
                "See diffusion_policy repo for setup."
            )
            return {}
        probe_env.close()

        device = self.device
        normalizer = self.normalizer

        all_scores = []
        n_p1 = 0
        n_p2 = 0
        episode_offset = 0

        pbar = tqdm(total=self.n_eval_episodes, desc="BlockPush eval", leave=False)
        for chunk_size in _chunk_sizes(self.n_eval_episodes, self.n_envs):
            envs = [self._make_env() for _ in range(chunk_size)]
            obs_buffer = VectorizedObsBuffer(chunk_size, self.n_obs_steps)
            # Upstream (blockpush_lowdim_runner.py) scores an episode as the
            # sum of *unique* reward values observed over the rollout — the
            # env's reward is a per-block completion indicator that jumps to
            # a new fixed value the first time each block reaches its target,
            # not a running max of a single 'score' info key.
            seen_rewards = [set() for _ in range(chunk_size)]
            episode_score = np.zeros(chunk_size, dtype=np.float64)
            done_flags = np.zeros(chunk_size, dtype=bool)
            steps = np.zeros(chunk_size, dtype=np.int64)

            for i, env in enumerate(envs):
                obs = self._reset_env(env, self.test_start_seed + episode_offset + i)
                obs_buffer.reset(i, {'state': self._obs_to_arr(obs)})

            while not np.all(done_flags | (steps >= self.max_steps)):
                stacked_obs = obs_buffer.get_batched()  # {'state': (N, n_obs_steps, D)}
                state = stacked_obs['state'].astype(np.float32)

                try:
                    state_t = torch.from_numpy(state).to(device)
                    state = normalizer['obs'].normalize(state_t).cpu().numpy()
                except (KeyError, Exception):
                    pass

                # Key matches the DataSample field the painter's condition
                # encoder reads (see models/condition_encoders.py).
                obs_tensor = {'embedding_conditions': torch.from_numpy(state).to(device)}

                with torch.no_grad():
                    action_dict = model.predict_action(obs_tensor, n_action_steps=self.n_action_steps)

                actions = action_dict['action'].cpu().numpy()  # (N, T_a, Da) normalized

                try:
                    actions_t = torch.from_numpy(actions).to(device)
                    action_exec = normalizer['action'].unnormalize(actions_t)
                    action_exec = action_exec.cpu().numpy()
                except (KeyError, Exception):
                    action_exec = actions

                T_a = action_exec.shape[1]
                for t in range(min(self.n_action_steps, T_a)):
                    for i, env in enumerate(envs):
                        if done_flags[i] or steps[i] >= self.max_steps:
                            continue
                        step_result = env.step(action_exec[i, t])
                        # Support both (obs, reward, done, info) and (obs, reward, terminated, truncated, info)
                        if len(step_result) == 5:
                            obs, reward, terminated, truncated, info = step_result
                            done = terminated or truncated
                        else:
                            obs, reward, done_flag, info = step_result
                            done = done_flag

                        reward_key = round(float(reward), 6)
                        if reward_key not in seen_rewards[i]:
                            seen_rewards[i].add(reward_key)
                            episode_score[i] += float(reward)

                        obs_buffer.push(i, {'state': self._obs_to_arr(obs)})
                        steps[i] += 1
                        if done:
                            done_flags[i] = True
                    if np.all(done_flags | (steps >= self.max_steps)):
                        break
                pbar.set_postfix(
                    active=int((~done_flags).sum()),
                    max_step=int(steps.max()),
                    score=f"{episode_score.mean():.3f}",
                )

            for env in envs:
                env.close()

            all_scores.extend(episode_score.tolist())
            # Thresholds match upstream blockpush_lowdim_runner.py.
            n_p1 += int((episode_score > 0.4).sum())
            n_p2 += int((episode_score > 0.9).sum())
            episode_offset += chunk_size
            pbar.update(chunk_size)
            pbar.set_postfix(mean_score=f"{np.mean(all_scores):.3f}", p1=n_p1, p2=n_p2)
        pbar.close()

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
        n_eval_episodes=128,
        max_steps=700,
        n_obs_steps=2,
        n_action_steps=8,
        use_camera_obs=True,
        obs_keys=None,
        test_start_seed=100000,
        n_envs=32,
        device='cpu',
    ):
        self.normalizer = normalizer
        self.n_eval_episodes = n_eval_episodes
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.use_camera_obs = use_camera_obs
        self.test_start_seed = test_start_seed
        self.n_envs = n_envs
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
            from robosuite.controllers import load_controller_config
            kwargs = dict(
                robots='Panda',
                # robomimic's public ToolHang dataset (which the painter is
                # trained on) was collected with OSC_POSE (6D pose delta + 1
                # gripper = 7-dim actions) — Panda's default when
                # unspecified is JOINT_VELOCITY (7 joint vels + 1 gripper =
                # 8-dim), which doesn't match the trained action space.
                controller_configs=load_controller_config(default_controller="OSC_POSE"),
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
        """Extract and stack relevant observation keys from a robosuite obs dict.

        Raw robosuite observables group every modality="object" sensor into
        a single concatenated 'object-state' key — 'object' is robomimic's
        own renamed convention for that same vector when it builds its
        offline datasets (which is what obs_keys/the trained painter expect),
        so alias it here rather than assuming robosuite exposes 'object'
        directly.
        """
        result = {}
        for key in self.obs_keys:
            raw_key = 'object-state' if key == 'object' and key not in raw_obs else key
            if raw_key in raw_obs:
                val = np.array(raw_obs[raw_key], dtype=np.float32)
                result[key] = val
        return result

    def _build_obs_tensor(self, stacked_obs):
        """Assemble the raw per-key robosuite obs into the DataSample fields
        the painter's condition encoder reads.

        stacked_obs values are (N, n_obs_steps, *shape) arrays from
        VectorizedObsBuffer.get_batched() — N is the chunk's active env count.

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
            ], axis=-1).astype(np.float32)  # (N, T, 53)
            obs_t = torch.from_numpy(obs_lowdim).to(self.device)
            try:
                obs_t = normalizer['obs'].normalize(obs_t)
            except (KeyError, Exception):
                pass
            return {'embedding_conditions': obs_t}

        def _img(key):
            arr = stacked_obs[key].astype(np.float32) / 255.0 * 2.0 - 1.0
            return torch.from_numpy(arr.transpose(0, 1, 4, 2, 3))  # (N, T, 3, H, W)

        views = [
            _img(key) for key in ('sideview_image', 'robot0_eye_in_hand_image') if key in stacked_obs
        ]
        spatial_conditions = torch.stack(views, dim=2).to(self.device) if views else None  # (N, T, 2, 3, H, W)

        embedding_conditions = None
        if all(k in stacked_obs for k in ('robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos')):
            obs_robot = np.concatenate([
                stacked_obs['robot0_eef_pos'],
                stacked_obs['robot0_eef_quat'],
                stacked_obs['robot0_gripper_qpos'],
            ], axis=-1).astype(np.float32)  # (N, T, 9)
            try:
                obs_t = torch.from_numpy(obs_robot).to(self.device)
                obs_t = normalizer['obs_robot'].normalize(obs_t)
            except (KeyError, Exception):
                obs_t = torch.from_numpy(obs_robot).to(self.device)
            embedding_conditions = obs_t

        obs_tensor = {}
        if spatial_conditions is not None:
            obs_tensor['spatial_conditions'] = spatial_conditions
        if embedding_conditions is not None:
            obs_tensor['embedding_conditions'] = embedding_conditions
        return obs_tensor

    def __call__(self, model, dataloader, accelerator, **kwargs):
        probe_env = self._make_env()
        if probe_env is None:
            logger.warning(
                "robosuite not installed or ToolHang env unavailable. "
                "Skipping ToolHang eval. pip install robosuite robomimic"
            )
            return {}
        probe_env.close()

        normalizer = self.normalizer
        n_success = 0
        episode_offset = 0

        pbar = tqdm(total=self.n_eval_episodes, desc="ToolHang eval", leave=False)
        for chunk_size in _chunk_sizes(self.n_eval_episodes, self.n_envs):
            envs = [self._make_env() for _ in range(chunk_size)]
            obs_buffer = VectorizedObsBuffer(chunk_size, self.n_obs_steps)
            success_flags = np.zeros(chunk_size, dtype=bool)
            done_flags = np.zeros(chunk_size, dtype=bool)
            steps = np.zeros(chunk_size, dtype=np.int64)

            for i, env in enumerate(envs):
                try:
                    env.seed(self.test_start_seed + episode_offset + i)  # robosuite/legacy-gym seeding API
                except (AttributeError, Exception):
                    pass
                raw_obs = env.reset()
                obs_buffer.reset(i, self._obs_to_dict(raw_obs))

            while not np.all(done_flags | (steps >= self.max_steps)):
                stacked_obs = obs_buffer.get_batched()
                obs_tensor = self._build_obs_tensor(stacked_obs)

                with torch.no_grad():
                    action_dict = model.predict_action(obs_tensor, n_action_steps=self.n_action_steps)

                actions = action_dict['action'].cpu().numpy()  # (N, T_a, Da) normalized

                try:
                    actions_t = torch.from_numpy(actions).to(self.device)
                    action_exec = normalizer['action'].unnormalize(actions_t)
                    action_exec = action_exec.cpu().numpy()
                except (KeyError, Exception):
                    action_exec = actions

                T_a = action_exec.shape[1]
                for t in range(min(self.n_action_steps, T_a)):
                    for i, env in enumerate(envs):
                        if done_flags[i] or steps[i] >= self.max_steps:
                            continue
                        raw_obs, reward, done_flag, info = env.step(action_exec[i, t])
                        # robosuite returns done when horizon is reached
                        done = bool(done_flag)
                        # robosuite's base env._post_action() always returns an
                        # EMPTY info dict ({}) — info.get('success', False) can
                        # never be True. Success must be queried directly via the
                        # task env's own _check_success(), exactly like
                        # robomimic's EnvRobosuite.is_success() does.
                        if env._check_success():
                            success_flags[i] = True
                        obs_buffer.push(i, self._obs_to_dict(raw_obs))
                        steps[i] += 1
                        if done:
                            done_flags[i] = True
                    if np.all(done_flags | (steps >= self.max_steps)):
                        break
                pbar.set_postfix(
                    active=int((~done_flags).sum()),
                    max_step=int(steps.max()),
                    success=int(success_flags.sum()),
                )

            for env in envs:
                env.close()

            n_success += int(success_flags.sum())
            episode_offset += chunk_size
            pbar.update(chunk_size)
            pbar.set_postfix(success_rate=f"{n_success / episode_offset:.3f}")
        pbar.close()

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
