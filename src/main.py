from Env.Config.GarmentConfig import GarmentConfig
import carb
from isaacsim import SimulationApp
import numpy as np

# 启动 Isaac Sim 仿真环境（非无头模式，可根据需要修改参数）
simulation_app = SimulationApp({"headless": False})

# 自定义模块（请确保路径正确）
from Env.Config.FrankaConfig import FrankaConfig
from Env.env.BaseEnv import BaseEnv

###############################################################################
# 主程序入口
###############################################################################
if __name__ == "__main__":

    garment_config=GarmentConfig(
        usd_path="/home/pakwa/GarmentLab/Assets/Garment/Tops/NoCollar_Lsleeve_FrontClose/TNLC_Dress057/TNLC_Dress057_obj.usd",
        pos=np.array([0,-0.95,0.3]),ori=np.array([0,0,0]),particle_contact_offset=0.01)
    
    franka_config=FrankaConfig(ori=[np.array([0,0,-np.pi/2])])

    env = BaseEnv(garment_config=[garment_config],
                franka_config=franka_config,)

    env.reset()
    carb.log_info("环境初始化完成，开始仿真。")


    env.control.robot_reset()
    env.control.grasp([np.array([0.35282,-0.26231,0.02])],[None],[True])

    # env.control.move([np.array([0.28138,-0.26231,0.22])],[None],[True])
    env.control.move([np.array([0.1,-0.26231,0.22])],[None],[True])
    env.control.move([np.array([-0.12,-0.26231,0.1])],[None],[True])
    env.control.ungrasp([False],)
    env.control.grasp([np.array([-0.377,-0.26231,0.02])],[None],[True])

    # env.control.move([np.array([-0.3,-0.26231,0.22])],[None],[True])
    env.control.move([np.array([0.0,-0.26231,0.22])],[None],[True])
    env.control.move([np.array([0.07,-0.26231,0.03])],[None],[True])
    env.control.ungrasp([False],)

    env.control.grasp([np.array([0.0,-0.65,0.015])],[None],[True])

    env.control.move([np.array([-0.0,-0.515,0.22])],[None],[True])
    env.control.move([np.array([0.0,-0.25,0.05])],[None],[True])

    env.control.ungrasp([True],)

    env.control.robot_goto_position([np.array([-0.0,-0.566,0.22])],[None],[True])


    while simulation_app.is_running():
            env.step()
    
    env.stop()
    simulation_app.close()
