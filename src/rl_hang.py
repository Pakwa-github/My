import sys
sys.path.append("/home/pakwa/GarmentLab")

from Env.Config.GarmentConfig import GarmentConfig
from Env.Config.FrankaConfig import FrankaConfig  
import numpy as np
import yaml
import torch
from rl_utils.rl_env import RlEnvBase

from rl_utils.simulation_env import AffordanceEnv
from rl_utils.task_defne import HangTask



def save(
    model,
    path,
    exclude = None,
    include = None,
) -> None:
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

    params_to_save = model.get_parameters()
    torch.save(params_to_save, path)


if __name__=="__main__":

    mode = "train"

    assert mode in ["train", "eval"]

    filename = "/home/pakwa/GPs/GarmentLab/LearningBaseline/rl_hang/config/config_0.yaml"
    with open(filename, 'r') as file:
        task_config = yaml.safe_load(file)

    garment_config = GarmentConfig(usd_path=task_config["garment_config"]["garment_path"])
    garment_config.pos = np.array(task_config["garment_config"]["garment_pos"])
    garment_config.ori = np.array(task_config["garment_config"]["garment_ori"])
    garment_config.scale = np.array(task_config["garment_config"]["garment_scale"])
    garment_config.particle_contact_offset = 0.01
    franka_config = FrankaConfig(franka_num=1, pos=[np.array([0,0.2,0.])], ori=[np.array([0,0,0])])
    
    rl_env = RlEnvBase(headless=False)
    task = HangTask()
    rl_env.set_task(task, [garment_config], backend="torch")

    env=AffordanceEnv(garment_config=[garment_config], franka_config=franka_config, task_config=task_config)


    print("\nget demo...\n")
    
    succ_data = env.get_demo(task_config["demo_point"], wo_gripper=True, debug=False)


    from rl_utils.ppo import PPO

    model = PPO(
        "MlpPolicy",
        env,
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
        tensorboard_log="./hang",
        normalize_advantage=False,
    )
    
    print()
    print('start training')
    print()

    if mode == "train":
        model.learn(total_timesteps=160)
        save(model, "./model/hang_new.ckpt")

    else:
        loaded_data = torch.load("./model/hang.ckpt")
        model.policy.load_state_dict(loaded_data["policy"])
        model.eval_policy(num_envs=1, n_rollout_steps=30)

    