import carb
from isaacsim import SimulationApp

# 启动 Isaac Sim 仿真环境（非无头模式，可根据需要修改参数）
simulation_app = SimulationApp({"headless": False})

# 自定义模块（请确保路径正确）
from Env.Config.FrankaConfig import FrankaConfig
from Env.env.BaseEnv import BaseEnv

###############################################################################
# 主程序入口
###############################################################################
if __name__ == "__main__":

    # 创建环境实例（加载房间、默认地面、灯光等）
    env = BaseEnv()
    
    try:    
        # 示例：加载 Franka 机器人（请确保 FrankaConfig 配置正确）
        # 若 FrankaConfig 配置为：franka_num=1, pos=[np.array([...])], ori=[np.array([...])]
        franka_config = FrankaConfig()  # 请根据需要初始化配置
        franka_list = env.import_franka(franka_config)

        # 示例：加载衣物资源（请替换为您实际的衣物 USD 路径）
        garment_usd_path = "/home/pakwa/GarmentLab/Assets/Garment/Dress/Long_ShortSleeve/DLSS_Dress182/DLSS_Dress182_obj.usd"
        env.import_garment(garment_usd_path)

        # 重置环境
        env.reset()
        carb.log_info("环境初始化完成，开始仿真。")

        # 主仿真循环（建议设置退出条件或按键监听）
        while simulation_app.is_running():
            env.step()
    except KeyboardInterrupt:
        carb.log_info("检测到中断信号，停止仿真。")
    except Exception as e:
        carb.log_error(f"仿真过程中出现错误：{e}")
    finally:
        env.stop()
        simulation_app.close()
