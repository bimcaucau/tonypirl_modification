"""PPO with a privileged Discrete(7) expert (mix + DAgger CE + BC warmup).

collect_rollouts is a thin fork of SB3 2.9.0 OnPolicyAlgorithm.collect_rollouts:
the executed action is what goes in the buffer. Do not wrap env.step to swap
actions — stock PPO would store the policy sample instead.
"""
from __future__ import annotations

from collections.abc import Generator
from typing import NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv


class GuidedDictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    expert_actions: th.Tensor
    expert_valid: th.Tensor


class GuidedDictRolloutBuffer(DictRolloutBuffer):
    """DictRolloutBuffer plus expert action labels aligned with get() permutation."""

    def reset(self) -> None:
        super().reset()
        self.expert_actions = np.full((self.buffer_size, self.n_envs), -1, dtype=np.int64)
        self.expert_valid = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, *args, expert_actions=None, expert_valid=None, **kwargs) -> None:
        pos = self.pos
        super().add(*args, **kwargs)
        n = self.n_envs
        if expert_actions is None:
            self.expert_actions[pos] = -1
            self.expert_valid[pos] = 0.0
        else:
            self.expert_actions[pos] = np.asarray(expert_actions, dtype=np.int64).reshape(n)
            if expert_valid is None:
                self.expert_valid[pos] = 1.0
            else:
                self.expert_valid[pos] = np.asarray(expert_valid, dtype=np.float32).reshape(n)

    def get(self, batch_size: int | None = None) -> Generator[GuidedDictRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)
            for tensor in ("actions", "values", "log_probs", "advantages", "returns",
                           "expert_actions", "expert_valid"):
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs
        start_idx = 0
        n = self.buffer_size * self.n_envs
        while start_idx < n:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> GuidedDictRolloutBufferSamples:
        return GuidedDictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for key, obs in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds].astype(np.float32, copy=False)),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            expert_actions=self.to_torch(self.expert_actions[batch_inds].flatten()),
            expert_valid=self.to_torch(self.expert_valid[batch_inds].flatten()),
        )


class GuidedPPO(PPO):
    """PPO + decaying expert mix at collect time + DAgger CE on labelled states."""

    def __init__(
        self,
        *args,
        bc_coef: float = 0.15,
        bc_anneal_steps: int = 2_000_000,
        mix_prob: float = 0.3,
        mix_anneal_steps: int = 1_000_000,
        **kwargs,
    ):
        kwargs.setdefault("rollout_buffer_class", GuidedDictRolloutBuffer)
        super().__init__(*args, **kwargs)
        self.bc_coef0 = float(bc_coef)
        self.bc_anneal_steps = int(bc_anneal_steps)
        self.mix_prob0 = float(mix_prob)
        self.mix_anneal_steps = int(mix_anneal_steps)
        self._mix_rng = np.random.RandomState(0)
        self._last_expert_agree = 0.0
        self._nonfinite_skipped = 0

    def _lin(self, start: float, t_end: int) -> float:
        # Deploy requires 0 at t_end; do not clamp to a floor.
        if t_end <= 0:
            return float(start)
        return float(start) * max(0.0, 1.0 - float(self.num_timesteps) / float(t_end))

    def _mix_prob(self) -> float:
        return self._lin(self.mix_prob0, self.mix_anneal_steps)

    def _bc_coef(self) -> float:
        return self._lin(self.bc_coef0, self.bc_anneal_steps)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer,
        n_rollout_steps: int,
    ) -> bool:
        # Forked from SB3 2.9.0 OnPolicyAlgorithm.collect_rollouts.
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        agree_acc = []

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = np.asarray(actions.cpu().numpy())
            act_shape = actions.shape
            flat_actions = np.asarray(actions, dtype=np.int64).reshape(env.num_envs)

            mix_p = self._mix_prob()
            need_expert = mix_p > 0.0 or self._bc_coef() > 0.0
            if need_expert:
                expert = np.asarray(env.env_method("expert_action"), dtype=np.int64).reshape(env.num_envs)
                valid = np.ones(env.num_envs, dtype=np.float32)
                agree_acc.append(float((flat_actions == expert).mean()))
                if mix_p > 0.0:
                    mix_mask = self._mix_rng.random(env.num_envs) < mix_p
                    if np.any(mix_mask):
                        flat_actions = flat_actions.copy()
                        flat_actions[mix_mask] = expert[mix_mask]
                        actions = flat_actions.reshape(act_shape)
                        with th.no_grad():
                            dist = self.policy.get_distribution(obs_tensor)
                            executed = th.as_tensor(flat_actions, device=self.device, dtype=th.long)
                            log_probs = dist.log_prob(executed)
            else:
                expert = np.full(env.num_envs, -1, dtype=np.int64)
                valid = np.zeros(env.num_envs, dtype=np.float32)

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                expert_actions=expert,
                expert_valid=valid,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        self._last_expert_agree = float(np.mean(agree_acc)) if agree_acc else 0.0
        self.logger.record("rollout/mix_prob", self._mix_prob())
        self.logger.record("rollout/expert_agree", self._last_expert_agree)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses, bc_losses = [], [], []
        clip_fractions = []
        bc_coef = self._bc_coef()

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Ratio clamp: clamp log_ratio to [-3.0, 3.0] and ratio to [0.05, 20.0]
                # to prevent explosive ratio when old_log_prob is very negative from expert mix
                log_ratio = log_prob - rollout_data.old_log_prob
                ratio = th.exp(th.clamp(log_ratio, -3.0, 3.0)).clamp(0.05, 20.0)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                if bc_coef > 0.0 and hasattr(rollout_data, "expert_actions"):
                    expert_a = rollout_data.expert_actions.long().flatten()
                    valid = rollout_data.expert_valid.flatten()
                    if valid.sum() > 0:
                        dist = self.policy.get_distribution(rollout_data.observations)
                        nll = (-dist.log_prob(expert_a)).clamp(max=20.0)
                        bc_loss = (nll * valid).sum() / (valid.sum() + 1e-8)
                    else:
                        bc_loss = th.zeros((), device=self.device)
                else:
                    bc_loss = th.zeros((), device=self.device)
                bc_losses.append(float(bc_loss.detach().cpu()))

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + bc_coef * bc_loss

                with th.no_grad():
                    log_ratio_raw = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(th.clamp(log_ratio_raw, -10.0, 10.0)) - 1) - log_ratio_raw).cpu().numpy()
                    approx_kl_divs.append(float(approx_kl_div))

                if self.target_kl is not None and (approx_kl_div > 1.5 * self.target_kl or not np.isfinite(approx_kl_div)):
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at epoch {epoch} due to reaching max kl: {approx_kl_div:.4f}")
                    break

                # Skip optimizer step if loss or gradients are not finite
                if not th.isfinite(loss):
                    self._nonfinite_skipped += 1
                    continue

                self.policy.optimizer.zero_grad()
                loss.backward()
                grad_norm = th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                if not th.isfinite(grad_norm):
                    self._nonfinite_skipped += 1
                    self.policy.optimizer.zero_grad()
                    continue
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/bc_loss", np.mean(bc_losses) if bc_losses else 0.0)
        self.logger.record("train/bc_coef", bc_coef)
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs) if approx_kl_divs else 0.0)
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item() if th.is_tensor(loss) else float(loss))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/nonfinite_skipped", self._nonfinite_skipped)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


def bc_warmup(
    model: GuidedPPO,
    vec_env: VecEnv,
    n_transitions: int,
    n_epochs: int = 8,
    batch_size: int = 1024,
    lr: float | None = None,
) -> None:
    """Drive the vec env with expert_action and clone Dict obs → a*."""
    from stable_baselines3.common.utils import update_learning_rate

    n_envs = vec_env.num_envs
    n_loops = max(1, int(np.ceil(n_transitions / max(1, n_envs))))
    print(f"--> BC warmup: {n_loops * n_envs} transitions, {n_epochs} epochs, batch {batch_size}")
    obs = vec_env.reset()
    obs_chunks = []
    act_chunks = []
    for _ in range(n_loops):
        a_star = np.asarray(vec_env.env_method("expert_action"), dtype=np.int64)
        obs_chunks.append({k: np.copy(v) for k, v in obs.items()})
        act_chunks.append(a_star.copy())
        obs, _rewards, _dones, _infos = vec_env.step(a_star)

    obs_arr = {k: np.concatenate([c[k] for c in obs_chunks], axis=0) for k in obs_chunks[0]}
    acts = np.concatenate(act_chunks, axis=0)
    n = int(acts.shape[0])
    policy = model.policy
    policy.set_training_mode(True)
    if lr is not None:
        update_learning_rate(policy.optimizer, float(lr))

    last_loss = None
    first_loss = None
    for epoch in range(n_epochs):
        perm = np.random.permutation(n)
        epoch_losses = []
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch_obs = {k: v[idx] for k, v in obs_arr.items()}
            obs_t = obs_as_tensor(batch_obs, model.device)
            act_t = th.as_tensor(acts[idx], device=model.device, dtype=th.long)
            dist = policy.get_distribution(obs_t)
            loss = -dist.log_prob(act_t).mean()
            policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), model.max_grad_norm)
            policy.optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        last_loss = float(np.mean(epoch_losses))
        if first_loss is None:
            first_loss = last_loss
        print(f"    BC epoch {epoch + 1}/{n_epochs}  nll={last_loss:.4f}")
    print(f"--> BC warmup done  nll {first_loss:.4f} -> {last_loss:.4f}")
    policy.set_training_mode(False)
