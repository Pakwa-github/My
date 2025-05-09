import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
from termcolor import cprint
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from stable_baselines3.common.policies import ActorCriticCnnPolicy,BasePolicy, MultiInputActorCriticPolicy
from RL.BaseModel import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

SelfPPO = TypeVar("SelfPPO", bound="PPO")

import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")



class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    rollout_buffer: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        sim_env,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )
        self.sim_env = sim_env
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}
        self.step_num = 0

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = DictRolloutBuffer
            else:
                self.rollout_buffer_class = RolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)
        # Warn when not using CPU with MlpPolicy
        self._maybe_recommend_cpu()

    def _maybe_recommend_cpu(self, mlp_class_name: str = "ActorCriticPolicy") -> None:
        """
        Recommend to use CPU only when using A2C/PPO with MlpPolicy.

        :param: The name of the class for the default MlpPolicy.
        """
        policy_class_name = self.policy_class.__name__
        if self.device != th.device("cpu") and policy_class_name == mlp_class_name:
            warnings.warn(
                f"You are trying to run {self.__class__.__name__} on the GPU, "
                "but it is primarily intended to run on the CPU when not using a CNN policy "
                f"(you are using {policy_class_name} which should be a MlpPolicy). "
                "See https://github.com/DLR-RM/stable-baselines3/issues/1245 "
                "for more info. "
                "You can pass `device='cpu'` or `export CUDA_VISIBLE_DEVICES=` to force using the CPU."
                "Note: The model will train, but the GPU utilization will be poor and "
                "the training might take longer than on CPU.",
                UserWarning,
            )

    def eval_policy(
        self,
        num_envs,
        # callback: BaseCallback,
        # rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        这边没有 rollout buffer，也没有 callback。
        仅仅是每一步拿 obs；扔进 policy 拿 action；步进环境；累加 reward（用于计算评估指标，比如 success rate）
        sb3没有？
        """
        self.policy.set_training_mode(False)
        n_steps = 0
        success_count = 0
        grasp_fail_count = 0
        dropped = 0
        if self.use_sde:
            self.policy.reset_noise(num_envs)

        succ_list = []

        while n_steps < n_rollout_steps:
            last_obs = self.sim_env.get_obs()
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(last_obs, self.device)
                obs_tensor = obs_tensor.unsqueeze(0)
                actions, values, log_probs = self.policy(obs_tensor, True)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            new_obs, reward, done, infos = self.sim_env.step(clipped_actions, True)

            self.num_timesteps += num_envs

            if infos.get("success", False):
                success_count += 1
            if not infos.get("grasp_success", True):
                grasp_fail_count += 1
            dropped += infos.get("dropped", 0)
            n_steps += 1
            succ_list.append(reward)

            if done == True:
                self.sim_env.reset(self.sim_env.config.garment_num)

        succ_rate = sum(succ_list)/len(succ_list)
        cprint(f"n_steps:{n_steps}\nevaluation success rate:{succ_rate}", "green")
        success_rate = success_count / n_steps if n_steps else 0.0
        cprint(f"[Success Rate] {success_rate:.2%}", "green")
        fail_rate = grasp_fail_count / n_steps if n_steps else 0.0
        cprint(f"[Grasp Fail Rate] {fail_rate:.2%}", "red")
        dropped_rate = dropped / n_steps if n_steps else 0.0
        cprint(f"[Dropped Rate] {dropped_rate:.2%}", "red")


    def collect_rollouts(
        self,
        num_envs,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        MyFlag: bool = True,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
    
        """env(VecEnv)换成了 num_envs，只取这个里面的变量"""
        
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:

            last_obs = self.sim_env.get_obs() # ！me

            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(num_envs)
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(last_obs, self.device)
                # unsqueeze()函数起升维的作用,参数表示在哪个地方加一个维度。
                # 0 在第一个维度(中括号)的每个元素加中括号
                obs_tensor = obs_tensor.unsqueeze(0)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action (direct mapping for your task)
            clipped_actions = actions  # # For specific task, no need to clip actions

            new_obs, reward, done, info = self.sim_env.step(clipped_actions, flag=MyFlag)
            
            self.num_timesteps += num_envs

            # me
            self.step_num += 1

            # False通常因为回调要求提前停止learn训练
            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            # self._update_info_buffer(info, done) # 报错的
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            rollout_buffer.add(
                last_obs, 
                actions,
                reward,
                done,
                values,
                log_probs,
            )

            if done == True:
                self.sim_env.reset(self.sim_env.config.garment_num)

        # ！
        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = obs_as_tensor(new_obs, self.device)
            obs_tensor = obs_tensor.unsqueeze(0)
            values = self.policy.predict_values(obs_tensor)

        # 计算 GAE、归一化 return、执行 callback（比如 tensorboard logging)
        done = np.array(done)
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=done)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def get_rollout_data(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def _dump_logs(self, iteration: int) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        import time
        from datetime import timedelta
        start_time = time.time()

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            
            # me
            if self.num_timesteps <= total_timesteps: 
                MyFlag = True
            else:
                MyFlag = False

            continue_training = self.collect_rollouts(self.env.num_envs, callback, self.rollout_buffer, n_rollout_steps=self.n_steps, MyFlag=MyFlag)
            
            if not continue_training:
                break
            
            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            
            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            # me
            rollout_data = self.get_rollout_data()
            self.train(rollout_data)
            
            # me ？
            print("training iteration finish:", iteration)
            if self.num_timesteps % (log_interval * self.n_steps) < self.n_steps:
                elapsed = time.time() - start_time
                progress = self.num_timesteps / total_timesteps
                est_total_time = elapsed / progress
                est_remaining_time = est_total_time - elapsed
                print(f"[Time Estimator] {self.num_timesteps}/{total_timesteps} timesteps | "
                  f"Elapsed: {timedelta(seconds=int(elapsed))} | "
                  f"Remaining: {timedelta(seconds=int(est_remaining_time))}")

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        sim_env,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
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
            stats_window_size=stats_window_size,
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

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self, rollout_data) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """

        observations = rollout_data["observations"]
        actions = rollout_data["actions"]
        advantages = rollout_data["advantages"]
        old_log_prob = rollout_data["old_log_prob"]
        old_values = rollout_data["old_values"]
        returns = rollout_data["returns"]

        var_values = rollout_data["var_values"]
        var_returns = rollout_data["var_returns"]

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            # Re-sample the noise matrix because the log_std has changed
            # if self.use_sde:
            #     self.policy.reset_noise(self.batch_size)

            values, log_prob, entropy = self.policy.evaluate_actions(observations, actions)
            values = values.flatten()
            # Normalize advantage
            # Normalization does not make sense if mini batchsize == 1, see GH issue #325
            if self.normalize_advantage and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ratio between old and new policy, should be one at the first iteration
            ratio = th.exp(log_prob - old_log_prob)

            # clipped surrogate loss
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

            # Logging
            pg_losses.append(policy_loss.item())
            clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
            clip_fractions.append(clip_fraction)

            if self.clip_range_vf is None:
                # No clipping
                values_pred = values
            else:
                # Clip the difference between old and new value
                # NOTE: this depends on the reward scaling
                values_pred = old_values + th.clamp(
                    values - old_values, -clip_range_vf, clip_range_vf
                )
            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(returns, values_pred)
            value_losses.append(value_loss.item())

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            entropy_losses.append(entropy_loss.item())

            """ !me 注释掉了部分loss"""
            """ !!! 这一处修改意味着当前训练过程中，你只在优化策略网络，
            而没有同时通过梯度下降更新价值函数参数，也未鼓励足够的探索（熵项未加）。
            这样处理或许是针对你的任务在反馈设计上已经有较明确的“奖励—成功率”指标，
            但可能会使得优势估计出现偏差，并且降低策略多样性。"""
            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            # 这是极其危险的！因为你完全放弃了策略梯度优化，变成了用 reward 做监督回归
            # （像个 fake baseline classifier），是完全背离 RL 的逻辑的。
            # loss = th.mean(returns,dim = 0)

            # Calculate approximate form of reverse KL Divergence for early stopping
            # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
            # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
            # and Schulman blog: http://joschu.net/blog/kl-approx.html
            with th.no_grad():
                log_ratio = log_prob - old_log_prob
                approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                approx_kl_divs.append(approx_kl_div)

            if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                continue_training = False
                if self.verbose >= 1:
                    print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                break

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(var_values, var_returns)

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
    
    def get_rollout_data(self):
        rollout_data = [*self.rollout_buffer.get(self.batch_size)][0]
        return {
            "actions":rollout_data.actions,
            "observations":rollout_data.observations,
            "advantages":rollout_data.advantages,
            "old_log_prob":rollout_data.old_log_prob,
            "old_values":rollout_data.old_values,
            "returns":rollout_data.returns,
            "var_values":self.rollout_buffer.values.flatten(),
            "var_returns":self.rollout_buffer.returns.flatten(),
        }