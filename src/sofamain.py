import sys
import time

import cv2
from termcolor import cprint
sys.path.append("/home/pakwa/My")
from RL.SofaSimEnv import SofaSimEnv
from RL.MyTaskDefine import MyTask
import torch
import yaml
from RL.MyPPO import PPO
from RL.RLEnv import RlEnvBase
from RL.SaveModel import save
# from RL.SimEnv import SimEnv
# from RL.TaskDefine import FoldTask
import carb
import numpy as np

import matplotlib
matplotlib.use("TkAgg")  # 或 Qt5Agg 等

import matplotlib.pyplot as plt
import numpy as np

def rgb_periodic_saver(camera, interval_sec=60):
    count = 0
    while True:
        filename = camera.get_rgb_graph(count=count)
        cprint(f"📸 Saved RGB frame: {filename}", "magenta")
        count += 1
        time.sleep(interval_sec)

def init():
    # 初始化环境
    env = SofaSimEnv()
    env.get_obs()
    rl_env = RlEnvBase(headless=False)
    rl_env.set_task(MyTask(), backend="torch")

    # 在初始化环境后启动新线程显示RGB图像
    import threading
    display_thread = threading.Thread(target=rgb_periodic_saver, args=(env.point_cloud_camera,))
    display_thread.daemon = True
    display_thread.start()

    env._do_stir()
    
    # 初始化PPO模型
    model = PPO(
        policy="MlpPolicy",
        sim_env=env,
        env=rl_env,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        learning_rate=1e-4,
        gamma=0.99,
        device="cuda:0",
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./tsb/SofaGrasp",
        normalize_advantage=True,
    )
    
    return model, env

if __name__ == "__main__":
    model, env = init()
    
    mode = "train"
    assert mode in ["train", "eval"]
    
    if mode == "train":
        # 训练模型
        cprint("\nTraining model...\n\nTraining model...\n\nTraining model...\n", "red")
        model.learn(total_timesteps=160)
        # 保存模型
        save(model, "./model/SofaGrasp.ckpt")
    else:
        # 加载模型进行评估
        loaded_data = torch.load("./model/SofaGrasp.ckpt")
        model.policy.load_state_dict(loaded_data["policy"])
        
        # 评估模型
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
        print(f"Total reward: {total_reward}")
