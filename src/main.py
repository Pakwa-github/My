import torch
import yaml
from Env.Config.GarmentConfig import GarmentConfig
from RL.MyPPO import PPO
from RL.RLEnv import RlEnvBase
from RL.SaveModel import save
from RL.SimEnv import SimEnv
from RL.TaskDefine import FoldTask
import carb
from isaacsim import SimulationApp
import numpy as np

simulation_app = SimulationApp({"headless": False})

from Env.Config.FrankaConfig import FrankaConfig
from Env.env.BaseEnv import BaseEnv

def init():
    
    filename = "/home/pakwa/My/RL/config/config.yaml"
    with open(filename, 'r') as file:
        task_config = yaml.safe_load(file)

    
    task = FoldTask()
    garment_config = GarmentConfig(usd_path=task_config["garment_config"]["garment_path"])
    garment_config.particle_contact_offset = 0.01
    rl_env = RlEnvBase(headless=False)
    rl_env.set_task(task, [garment_config], backend="torch")

    franka_config = FrankaConfig(franka_num=2, 
                                 pos=[np.array([-2,0,0.]),np.array([-4,0,0.])], 
                                 ori=[np.array([0,0,0]),np.array([0,0,0])])
    sim_env=SimEnv(garment_config=[garment_config], 
                   franka_config=franka_config, 
                   task_config=task_config)
    
    sim_env.get_demo(task_config["demo_point"], wo_gripper=True, debug = False)
    
    model = PPO(
        "MlpPolicy",
        sim_env,
        rl_env,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        learning_rate=1e-3,
        gamma=0.99,
        device="cuda:0",
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=1.0,
        verbose=1,
        tensorboard_log="./tsb/fold",
        normalize_advantage=False,
    )
    return model


if __name__ == "__main__":

    model = init()

    mode = "train"
    assert mode in ["train", "eval"]
    if mode == "train":
        model.learn(total_timesteps=160)
        save(model, "./model/fold.ckpt")

    else:
        loaded_data = torch.load("./model/fold.ckpt")
        model.policy_1.load_state_dict(loaded_data["policy_1"])
        model.policy_2.load_state_dict(loaded_data["policy_2"])
        model.eval_policy(num_envs=1, n_rollout_steps=30)





    # garment_config=GarmentConfig(
    #     usd_path="/home/pakwa/GarmentLab/Assets/Garment/Tops/NoCollar_Lsleeve_FrontClose/TNLC_Dress057/TNLC_Dress057_obj.usd",
    #     pos=np.array([0,-0.95,0.3]),ori=np.array([0,0,0]),particle_contact_offset=0.01)
    
    # franka_config=FrankaConfig(ori=[np.array([0,0,-np.pi/2])])

    # env = BaseEnv(garment_config=[garment_config],
    #             franka_config=franka_config,)

    # env.reset()
    # carb.log_info("环境初始化完成，开始仿真。")


    # env.control.robot_reset()
    # env.control.grasp([np.array([0.35282,-0.26231,0.02])],[None],[True])

    # # env.control.move([np.array([0.28138,-0.26231,0.22])],[None],[True])
    # env.control.move([np.array([0.1,-0.26231,0.22])],[None],[True])
    # env.control.move([np.array([-0.12,-0.26231,0.1])],[None],[True])
    # env.control.ungrasp([False],)
    # env.control.grasp([np.array([-0.377,-0.26231,0.02])],[None],[True])

    # # env.control.move([np.array([-0.3,-0.26231,0.22])],[None],[True])
    # env.control.move([np.array([0.0,-0.26231,0.22])],[None],[True])
    # env.control.move([np.array([0.07,-0.26231,0.03])],[None],[True])
    # env.control.ungrasp([False],)

    # env.control.grasp([np.array([0.0,-0.65,0.015])],[None],[True])

    # env.control.move([np.array([-0.0,-0.515,0.22])],[None],[True])
    # env.control.move([np.array([0.0,-0.25,0.05])],[None],[True])

    # env.control.ungrasp([True],)

    # env.control.robot_goto_position([np.array([-0.0,-0.566,0.22])],[None],[True])


    # while simulation_app.is_running():
    #         env.step()
    
    # env.stop()
    # simulation_app.close()
