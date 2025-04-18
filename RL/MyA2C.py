import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

from RL.MyPPO import OnPolicyAlgorithm
import numpy as np
import torch as th
from torch.nn import functional as F
from gymnasium import spaces

from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.buffers import RolloutBuffer, DictRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, explained_variance, get_schedule_fn
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

SelfA2C = TypeVar("SelfA2C", bound="MyA2C")

class MyA2C(OnPolicyAlgorithm):
    """
    Advantage Actor-Critic (A2C) algorithm.
    """
    policy_aliases: ClassVar[Dict[str, Type[ActorCriticPolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        sim_env,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        normalize_advantage: bool = False,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            sim_env,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.normalize_advantage = normalize_advantage
        
        # A2C 只用一次更新，不分 minibatch
        self.batch_size = self.n_steps * (self.env.num_envs if self.env else 1)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        # 可选：学习率 schedule
        self.lr_schedule = get_schedule_fn(self.learning_rate)

    def train(self, rollout_data) -> None:
        """
        Update policy using a single gradient step on collected rollouts.
        """
        observations = rollout_data["observations"]
        actions = rollout_data["actions"]
        returns = rollout_data["returns"]
        old_values = rollout_data["old_values"]
        advantages = rollout_data["advantages"]

        # 切换到训练模式
        self.policy.set_training_mode(True)
        # 更新 lr
        self._update_learning_rate(self.policy.optimizer)

        # 评估当前策略
        values, log_prob, entropy = self.policy.evaluate_actions(observations, actions)
        values = values.flatten()

        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # 策略损失
        policy_loss = -(advantages * log_prob).mean()
        # 价值损失
        value_loss = F.mse_loss(returns, values)
        # 熵损失
        if entropy is None:
            entropy_loss = -th.mean(-log_prob)
        else:
            entropy_loss = -th.mean(entropy)

        loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

        # 优化一步
        self.policy.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

    def get_rollout_data(self):
        data = next(self.rollout_buffer.get(self.batch_size))
        return {
            "observations": data.observations,
            "actions": data.actions,
            "advantages": data.advantages,
            "returns": data.returns,
            "old_values": data.old_values,
        }

    def learn(
        self: SelfA2C,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "A2C",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfA2C:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
