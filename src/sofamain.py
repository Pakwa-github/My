import os
import sys
import time

from RL.MyA2C import MyA2C
import cv2
from termcolor import cprint
sys.path.append("/home/pakwa/My")
from RL.SofaSimEnv import SofaSimEnv
from RL.MyTaskDefine import MyTask
import torch
import yaml
from RL.MyPPO import PPO
from RL.RLEnv import RlEnvBase
# from RL.SaveModel import custom_load, save
# from RL.SimEnv import SimEnv
# from RL.TaskDefine import FoldTask
import carb
import numpy as np


import threading
import time
def monitor_training(model, check_interval=300):
    cprint("监控线程启动！！","green")
    last_step = model.step_num
    while True:
        time.sleep(check_interval)
        if model.step_num <= last_step:
            print("⚠️ 训练似乎卡住，正在保存模型checkpoint...")
            cprint(model.step_num, "green")
            save(model, "./model/ppo_checkpoint_on_stall.zip")
            print("保存成功")
        last_step = model.step_num

def rgb_periodic_saver(camera, interval_sec=60):
    count = 0
    while True:
        filename = camera.get_rgb_graph(count=count)
        cprint(f"📸 Saved RGB frame: {filename}", "magenta")
        count += 1
        time.sleep(interval_sec)

def make_sim_env():
    from RL.SofaSimEnv import SofaSimEnv
    env = SofaSimEnv()
    return env


def returnA2C():
    env = SofaSimEnv()
    rl_env = RlEnvBase(headless=False)
    rl_env.set_task(MyTask(), backend="torch")
    model = MyA2C(
        policy="MlpPolicy",
        sim_env=env,
        env=rl_env,
        learning_rate=7e-4,
        n_steps=8,
        gamma=0.99,
        normalize_advantage=True,
        gae_lambda=1.0,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./tsb/SofaGrasp_A2C",
        verbose=1,
        device="cuda:0",
    )
    return model, env

def init():
    # 初始化环境
    env = SofaSimEnv()
    # env.get_obs()
    rl_env = RlEnvBase(headless=False)
    rl_env.set_task(MyTask(), backend="torch")

    # 在初始化环境后启动新线程显示RGB图像
    import threading
    display_thread = threading.Thread(target=rgb_periodic_saver, args=(env.point_cloud_camera,))
    display_thread.daemon = True
    # display_thread.start()
    
    from RL.VisionEncoder import PointNetFeaturesExtractor
    policy_kwargs = dict(
        features_extractor_class=PointNetFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=1024)
    )
    model = PPO(
        policy="MlpPolicy",
        sim_env=env,
        env=rl_env,
        learning_rate=1e-4,
        n_steps=8,     # 一次rollout 走16步 感觉不够啊
        batch_size=8,  # 每次更新中使用的样本数量
        n_epochs=3,     # 一个batch批次内进行优化轮次的数量 1
        gamma=0.99,
        normalize_advantage=True,   # 是否对优势进行归一化
        ent_coef=0.005, # 熵系数 0.01 ～ 0.001 原本是0.01
        vf_coef=0.5,  # 价值函数损失的权重
        max_grad_norm=0.5,
        tensorboard_log="./tsb/PPO_504",
        verbose=1,
        # policy_kwargs=policy_kwargs,
        device="cuda:0", 
    )
    
    return model, env

def save(
    model:PPO,
    path,
    exclude = None,
    include = None,
) -> None:
    # ?
    data = model.__dict__.copy()

    if exclude is None:
        exclude = []
    exclude = set(exclude).union(model._excluded_save_params())

    if include is not None:
        exclude = exclude.difference(include)

    state_dicts_names, torch_variable_names =model._get_torch_save_params()
    all_pytorch_variables = state_dicts_names + torch_variable_names
    for torch_var in all_pytorch_variables:
        var_name = torch_var.split(".")[0]
        exclude.add(var_name)

    for param_name in exclude:
        data.pop(param_name, None)

    # 非核心 torch 变量（例如优化器状态、某些内部计数器或者临时变量
    # 如果你需要恢复优化器状态或其他内部变量 那么注释部分的代码就很重要 
    # 会出现所谓的“冷启动”现象，即优化器状态丢失，虽然这在强化学习中通常不至于影响最终表现太多
    # # Build dict of torch variables
    # pytorch_variables = None
    # if torch_variable_names is not None:
    #     pytorch_variables = {}
    #     for name in torch_variable_names:
    #         attr = recursive_getattr(self, name)
    #         pytorch_variables[name] = attr

    params_to_save = model.get_parameters()
    torch.save(params_to_save, path)

if __name__ == "__main__":
    
    mode = "train"
    assert mode in ["train", "eval", "retrain", "sb3", "trainA2C", "evalA2C"]
    cprint(f"当前mode {mode}", "green")

    if mode == "train":
        model, env = init()
        monitor_thread = threading.Thread(target=monitor_training, args=(model, 200))
        monitor_thread.daemon = True
        monitor_thread.start()

        cprint("\nTraining model...\n\nTraining model...\n\nTraining model...\n", "red")
        try:
            env.reset(5)
            model.learn(total_timesteps=200)
        except Exception as e:
            print("⚠️ Training interrupted by error:", e)
            print("🔁 Saving model before exit...")
            cprint(model.step_num, "green")
            save(model, "./model_2/newppo_mid.zip")
            raise
        print("success Saving model...")
        save(model, "./model_2/newppo_fin.zip")

    elif mode == "retrain":
    
        print("🪄 从断点恢复训练")
        model, env = init()
        loaded_data = torch.load("./model_2/stage2_80.zip")
        model.policy.load_state_dict(loaded_data["policy"])

        monitor_thread = threading.Thread(target=monitor_training, args=(model, 200))
        monitor_thread.daemon = True
        monitor_thread.start()
        cprint("\nReTraining model...\n\nReTraining model...\n\nReTraining model...\n", "red")
        try:
            env.reset(2)
            model.learn(total_timesteps=120)
        except Exception as e:
            print("⚠️ Training interrupted by error:", e)
            print("🔁 Saving model before exit...")
            cprint(model.step_num, "green")
            save(model, "./model_2/GL_mid.zip")
            raise
        print("Saving model...")
        save(model, "./model_2/GL_fin.zip")

    elif mode == "eval":
        # 加载模型进行评估
        model, env = init()
        env.reset(8)
        loaded_data = torch.load("./model/GL_ppo1000.zip")
        model.policy.load_state_dict(loaded_data["policy"])
        model.eval_policy(num_envs=1, n_rollout_steps=30)

    elif mode == "sb3":
        sim_env = SofaSimEnv()
        rl_env = RlEnvBase(headless=False)
        rl_env.set_task(MyTask(), backend="torch")
        model = PPO.load("./model/ppo_checkpoint_on_crash_56.zip", env=rl_env, device="cuda:0")
        model.sim_env = sim_env
        model.eval_policy(num_envs=1, n_rollout_steps=10)

    elif mode == "trainA2C":
        
        model, _ = returnA2C()
        loaded_data = torch.load("./model/A2C_572.zip")
        model.policy.load_state_dict(loaded_data["policy"])

        monitor_thread = threading.Thread(target=monitor_training, args=(model, 200))
        monitor_thread.daemon = True
        monitor_thread.start()

        # 训练模型
        cprint("\nTraining model...\n\nTraining model...\n\nTraining model...\n", "red")
        try:
            model.learn(total_timesteps=120)
        except Exception as e:
            print("⚠️ Training interrupted by error:", e)
            print("🔁 Saving model before exit...")
            cprint(model.step_num, "green")
            save(model, "./model/A2C_mid.zip")
            raise

        print("success Saving model...")
        save(model, "./model/A2C.zip")

    elif mode == "evalA2C":
        # 加载模型进行评估
        model, _ = returnA2C()
        # loaded_data = torch.load("./model/A2C_500?.zip")
        # model.policy.load_state_dict(loaded_data["policy"])
        model.eval_policy(num_envs=1, n_rollout_steps=30)