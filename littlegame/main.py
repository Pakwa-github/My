import gymnasium
import warnings
import os
import time
import imageio
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from gymnasium import RewardWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)  # 取消警告


# 修改环境奖励：①修改源码 ②继承重写 ③回调函数 ④Wrapper
# 使用Wrapper
class CustomRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.helipad_center_x = 0.0  # 着陆场地中心的 x 坐标
        self.helipad_width = 1.0  # 假设着陆场地的宽度
        self.previous_fuel = 0.0

    def step(self, action):
        # 获取reward
        obs, reward, done, truncated, info = self.env.step(action)

        # 修改reward
        custom_reward = self.compute_custom_reward(obs, reward)

        return obs, custom_reward, done, truncated, info

    def compute_custom_reward(self, obs, reward):
        position = obs[0:2]  # 飞船的位置
        velocity = obs[2:4]  # 飞船的速度
        angle = obs[4]  # 飞船的角度
        m_power = getattr(self.env.unwrapped, 'm_power', 0)
        s_power = getattr(self.env.unwrapped, 's_power', 0)

        custom_reward = reward
        # 增加着陆中心位置控制
        custom_reward -= np.clip(abs(position[0] - self.helipad_center_x), 0, self.helipad_width) * 5

        # 增加燃料消耗惩罚
        fuel_used = m_power * 0.30 + s_power * 0.03
        custom_reward -= 0.02 * fuel_used

        return custom_reward

# 自定义回调函数，用于在每个训练步骤中显示进度
class TQDMProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super(TQDMProgressBarCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps)

    def _on_step(self) -> bool:
        self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()


# 创建环境
env_name = "LunarLander-v3"
env = gymnasium.make(env_name)
env = CustomRewardWrapper(env)

# 模型参数参考
# n_envs: 16
#   n_timesteps: !!float 1e6
#   policy: 'MlpPolicy'
#   n_steps: 1024
#   batch_size: 64
#   gae_lambda: 0.98
#   gamma: 0.999
#   n_epochs: 4
#   ent_coef: 0.01

# 1e6次的Pak1
# policy="MlpPolicy",       # 使用多层感知器（MLP）作为策略网络
# env=env,
# verbose=1,                # verbose 控制输出的详细程度。0，1，2。
# n_steps=1024,             # 在更新策略之前，每个并行环境中收集的时间步数。
# batch_size=64,            # 每次梯度更新的批次大小
# gae_lambda=0.98,          # 广义优势估计（GAE）中的折扣因子，平衡偏差和方差
# gamma=0.999,              # 奖励的折扣因子，表示重视未来奖励的程度
# n_epochs=4,               # 每个更新阶段的迭代次数，每次策略更新时策略网络会在样本上进行多次更新
# ent_coef=0.01             # 熵系数，较高的熵系数可以鼓励策略探索更多的动作
# total_timesteps = 1e6     # 这个值决定了模型在环境中进行多少次交互和学习

conditionPak = 0
total_timesteps = int(1e6)

if conditionPak:
    # env = make_vec_env(env_name, n_envs=1, monitor_dir="./logs")
    # global env
    # PPO模型定义，训练，保存
    model = PPO(policy="MlpPolicy",
                env=env,
                verbose=1,
                n_steps=1024,
                batch_size=64,
                gae_lambda=0.98,
                gamma=0.999,
                n_epochs=4,
                ent_coef=0.01
                )
    progress_bar_callback = TQDMProgressBarCallback(total_timesteps)
    model.learn(total_timesteps=total_timesteps, callback=progress_bar_callback)
    model.save("./model/Pak_lunarlander3.pkl")
    print("训练结束!")

model_path = "./model/Pak_lunarlander3.pkl"
model = PPO.load(model_path)
model_name = os.path.basename(model_path).replace(".pkl", "")

# 评估模型
input(f"当前评估的是{model_name}模型，按任意键开始评估模型的表现...")
env = gymnasium.make(env_name, render_mode="human")
env = CustomRewardWrapper(env)
obs, _ = env.reset()
total_reward = 0
done = False
images = []
print("开始着陆！")
scheduler = BackgroundScheduler()
scheduler.add_job(lambda: print("着陆中..."), 'interval', seconds=3)
scheduler.start()
while not done:
    action = model.predict(obs, deterministic=True)[0]
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward

scheduler.shutdown()
print()
print("着陆成功！\n总奖励 : {}".format(total_reward))
input("按任意键关闭窗口...")
env.close()

# 创建gif
# imageio.mimsave('.\gif\Pak1.gif', [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=29)

# 使用了 stable_baselines3 中的 evaluate_policy 函数来评估强化学习模型的性能
# env = make_vec_env(env_name, n_envs=1, monitor_dir="./logs")
env = gymnasium.make(env_name)
env = Monitor(env)  # 添加 Monitor 包装器来记录奖励和步数
env = CustomRewardWrapper(env)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
print(f"平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")
