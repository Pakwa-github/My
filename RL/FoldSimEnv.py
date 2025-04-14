import numpy as np
from isaacsim import SimulationApp
import torch
import os

simulation_app = SimulationApp({"headless": False})
import numpy as np
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.franka import Franka
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.franka import KinematicsSolver
from omni.isaac.core.utils.types import ArticulationAction
from Env.Utils.transforms import euler_angles_to_quat
import torch
from Env.Utils.transforms import quat_diff_rad
from Env.env.BaseEnv import BaseEnv
from Env.Garment.Garment import Garment
from Env.Rigid.Rigid import RigidAfford
from Env.Robot.Franka.MyFranka import MyFranka
from Control.Control import Control
from Env.Config.GarmentConfig import GarmentConfig
from Env.Config.FrankaConfig import FrankaConfig
from Env.Config.DeformableConfig import DeformableConfig


class SimEnv(BaseEnv):
    def __init__(self,garment_config:GarmentConfig=None,franka_config:FrankaConfig=None,Deformable_Config:DeformableConfig=None, task_config = None):
        
        BaseEnv.__init__(self)
        self.deformable_config = Deformable_Config
        self.task_name = task_config["task_name"]
        self.block_num = task_config["block_num"]
        self.target_point = np.array(
            task_config["target_point"]).reshape(-1,self.block_num,3)
        
        # ! 重复
        if garment_config is None:
            self.garmentconfig=GarmentConfig()
        else:
            self.garmentconfig=garment_config
        self.garment = list()
        particle_system = None
        for config in garment_config:
            garment = Garment(self.world, config, particle_system=particle_system)
            self.garment.append(garment)
            particle_system = garment.get_particle_system()
        self.garment_name = task_config["garment_name"]

        if franka_config is None:
            self.robotConfig=FrankaConfig(franka_num=1,pos=[np.array([-0.5,0,0]),np.array([0,-1,0])],ori=[np.array([0,0,-np.pi/2]),np.array([0,0,np.pi/2])])
        else:
            self.robotConfig=franka_config
        self.robots=self.import_franka(self.robotConfig)

        
        self.rigid = RigidAfford()
        # self.rigid = RigidHangCloth()  HangEnv.py中

        self.control=Control(self.world,self.robots,self.garment)        

        self.root_path = f"/home/pakwa/isaacgarment/affordance/{self.task_name}_{self.garment_name}"
        if not os.path.exists(self.root_path):
            os.mkdir(self.root_path)
        self.trial_path = os.path.join(self.root_path, "trial_pts.npy")
        self.model_path = os.path.join(self.root_path, "model.npy")


    def wait(self):
        while True:
            simulation_app.update()


    def step(self,action, eval_succ = False):
        # self.reset(random = True)
        self.reset()
        self.control.robot_reset()
        for _ in range(20):
            self.world.step()
        # point=self.allocate_point(0, save_path=self.trial_path)

        """
        处理动作：
        1、将动作展平为一维数组，并分割为两个 3D 动作向量
        2、将这两个动作向量加上当前衣物点云的质心
        3、调用 get_point 方法得到衣物上最接近这两个点的实际抓取点
        """
        action = action.reshape(-1)
        action_1 = action[:3]
        action_2 = action[3:]
        action_1 += self.centroid
        action_2 += self.centroid
        action_1 = self.get_point(action_1)
        action_2 = self.get_point(action_2)
        # 计算抓取动作与衣物当前状态之间的误差，转换为奖励。
        reward_action = self.compute_reward_action(action_1, action_2)

        self.control.grasp(pos=[action_1, action_2],ori=[None, None],flag=[True, True], wo_gripper=True)
        for idx in range(self.target_point.shape[0]):
            self.control.move(pos=[self.target_point[idx,0],self.target_point[idx,1]],ori=[None, None],flag=[True, True])

        for _ in range(150):
            self.world.step()

        reward_result = self.compute_reward_result()
        reward = reward_action * 0.8 + reward_result * 0.2
        self.control.ungrasp([False, False])
        for _ in range(10):
            self.world.step()
        self.world.stop()
        if eval_succ:
            succ_flag = True if (reward_action[0] > 0) and (reward_action[1] > 0) else False
            return None, succ_flag, np.array([True]), None
        # return None, reward_action, np.array([True]), None
        return None, reward, np.array([True]), None


    # 抓取动作的奖励，抓取点与目标点越近奖励越高
    def compute_reward_action(self, action_1, action_2):
        particles = self.get_cloth_in_world_pose()
        # sel_particle_index_1 = demopoint_1
        error_1 = np.linalg.norm(particles[self.sel_particle_index_1] - action_1) 
        error_2 = np.linalg.norm(particles[self.sel_particle_index_2] - action_2) 
        reward_1 = 0.1 - error_1
        reward_2 = 0.1 - error_2
        reward_1 = reward_1 if reward_1 < 0 else 1/(0.2 - reward_1)
        reward_2 = reward_2 if reward_2 < 0 else 1/(0.2 - reward_2)
        return np.array([reward_1, reward_2])


    # 与demo的succ_data比较？
    def compute_reward_result(self):
        final_data = self.get_cloth_in_world_pose()
        error = np.linalg.norm((self.succ_data - final_data), ord = 2)/final_data.shape[0]
        reward = 0.2 - error
        if reward < 0:
            return reward
        else:
            return 1/(0.3 - reward)


    def compute_succ(self, error):
        reward = 0.1 - error
        return reward
    

    # 将点云所有点相对于质心归一化后返回
    def get_obs(self):
        obs = self.get_all_points()
        self.centroid = np.mean(obs, axis = 0)
        return obs - self.centroid

    # 根据标准模型数据（从标准文件中读取）和当前衣物状态计算任务是否成功。
    def get_reward(self, standard_model, garment_id = 0):
        succ_example = np.loadtxt(standard_model)
        
        # 归一化处理pts
        pts = self.garment[garment_id].get_vertices_positions()
        centroid = np.mean(pts, axis=0)
        pts = pts - centroid
        max_scale = np.max(np.abs(pts))
        pts = pts / max_scale

        dist = np.linalg.norm(pts - succ_example, ord=2)
        if dist < 0.1:
            print("################ succ ################")
            return True
        else:
            print("################ fail ################")
            return False

    def get_demo(self, assign_point, wo_gripper, debug = False, log = False):
        """
        assign_point：两个三维点（共 6 个数），代表两个抓取点
        wo_gripper: 从名字来看可能代表“without gripper”的意思
        
        重置环境和机器人状态；

        1、通过 get_cloth_in_world_pose 获取衣物（cloth）的顶点数据；

        2、根据输入的 assign_point 参数（应包含两个三维点的信息）计算与衣物上各点的距离，从而选出离这两个点最近的衣物粒子索引；

        3、调用 control.grasp，让机器人抓取衣物（分别在选出的两个点处）；

        4、依次通过 control.move 将衣物移动到预定目标（存储在 self.target_point 中，每一行对应一个目标）；

        5、经过一段仿真步后，再调用 control.ungrasp 释放抓取；

        6、最后记录下最终的衣物状态数据（succ_data）
        """
        self.reset()
        self.control.robot_reset()
        for _ in range(20):
            self.world.step()
        if debug:
            self.wait()
        
        point_1 = np.array(assign_point[:3])
        point_2 = np.array(assign_point[3:])
        start_data=self.get_cloth_in_world_pose()
        if log:
            np.savetxt("start_data.txt", start_data)
        # 计算衣物上所有点到 point_1 和 point_2 的欧氏距离。
        dist_1 = np.linalg.norm(start_data - point_1[None,:], axis = -1)
        dist_2 = np.linalg.norm(start_data - point_2[None,:], axis = -1)
        # 分别选出与 point_1 和 point_2 最近的点在衣物顶点数据中的索引。
        self.sel_particle_index_1 = np.argmin(dist_1, axis=0)
        self.sel_particle_index_2 = np.argmin(dist_2, axis=0)
        # 这里指定两个抓取点（point_1 和 point_2）并用 flag 标记两个机器人都参与抓取。
        self.control.grasp(pos=[point_1, point_2],ori=[None, None],flag=[True, True])
        for idx in range(self.target_point.shape[0]):
            self.control.move(pos=[self.target_point[idx,0],self.target_point[idx,1]],
                              ori=[None, None],flag=[True, True])

        for _ in range(150):
            self.world.step()

        finish_data=self.get_cloth_in_world_pose()
        if log:
            np.savetxt("finish_data.txt", finish_data)

        self.control.ungrasp([False, False])
        for _ in range(10):
            self.world.step()
        self.succ_data = self.get_cloth_in_world_pose()


    def get_cloth_in_world_pose(self):
        """
        获取衣物在世界坐标下的顶点位置。
        首先获取衣物的顶点数据（get_vertices_positions），
        然后结合衣物的缩放、旋转与平移（通过 garment_mesh 的 get_world_pose），
        对点云进行变换，返回最终位置数据。
        """
        particle_positions = self.garment[0].get_vertices_positions()
        position, orientation = self.garment[0].garment_mesh.get_world_pose()
        if True:
            # particle_positions = particle_positions + self.pose
            particle_positions = particle_positions * self.garment[0].garment_config.scale
            particle_positions = self.rotate_point_cloud(particle_positions, orientation)
            particle_positions = particle_positions + position
            # 
        return particle_positions

    # 将点云数据根据给定的四元数进行旋转。
    def rotate_point_cloud(self, points, quaternion):
        w, x, y, z = quaternion
        rotation_matrix = np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]
        ])
        rotated_points = points @ rotation_matrix.T
        return rotated_points


    def exploration(self):
        """
        功能：进行多次试验（例如 800 次），在每次试验中：

        1、重置环境、分配抓取点（allocate_point），抓取、移动衣物；

        2、经过一定时间后采集最终衣物状态数据，并保存为 npy 文件；

        3、计算与成功状态（succ_data）的误差，利用该误差计算奖励；

        4、释放抓取；

        5、最后返回一个最终奖励。
        """
        for i in range(800):
            self.reset()
            self.control.robot_reset()
            for _ in range(20):
                self.world.step()
            point=self.allocate_point(i, save_path=self.trial_path)
            self.control.grasp(pos=[point],ori=[None],flag=[True], wo_gripper=True)
            self.control.move(pos=[self.target_point],ori=[None],flag=[True])
            # self.control.move(pos=[self.target_point+np.array([0.1,0,0])],ori=[None],flag=[True])
            for _ in range(100):
                self.world.step()
            
            final_data=self.garment[0].get_vertices_positions()
            final_path=os.path.join(self.root_path,f"final_pts_{i}.npy")
            error = np.linalg.norm(self.succ_data- final_data, ord = 2)/final_data.shape[0]
            reward = -error
            np.save(final_path,final_data)

            self.control.ungrasp([False])
            for _ in range(10):
                self.world.step()
            self.world.stop()
        return reward


    # 为试验分配一个抓取点。
    def allocate_point(self, index, save_path):
        """
        如果 index 为 0，则：
        获取衣物顶点数据，并根据衣物的缩放、旋转和平移处理，得到一个选点池；
        使用随机采样（torch.randperm）从选点池中选取 800 个点，并保存到文件；

        根据传入的 index 返回选点池中对应的抓取点。
        """
        if index==0:
            self.selected_pool=self.garment[0].get_vertices_positions()*self.garment[0].garment_config.scale
            q = euler_angles_to_quat(self.garment[0].garment_config.ori)
            self.selected_pool=self.Rotation(q,self.selected_pool)
            centroid, _ = self.garment[0].garment_mesh.get_world_pose()
            self.selected_pool=self.selected_pool + centroid
            np.savetxt("/home/pakwa/GarmentLab/select.txt",self.selected_pool)

            indices=torch.randperm(self.selected_pool.shape[0])[:800]
            self.selected_pool=self.selected_pool[indices]
            np.save(save_path, self.selected_pool)
        point=self.selected_pool[index]
        return point


    # 返回离给定位置position最近的点。
    def get_point(self, position):
        points = self.get_all_points()
        dist = np.linalg.norm(points - position.reshape(1,-1), ord = 2, axis=-1)
        idx = np.argmin(dist)
        return points[idx] 


    # 获取衣物所有顶点数据，
    # 经过缩放、旋转和平移后，
    # 再对点云进行下采样（使用 FPS 采样至 256 个点），
    # 返回下采样后的点云数据。
    def get_all_points(self):
        self.selected_pool=self.garment[0].get_vertices_positions()*self.garment[0].garment_config.scale
        q = euler_angles_to_quat(self.ori)
        self.selected_pool=self.Rotation(q,self.selected_pool)
        centroid, _ = self.garment[0].garment_mesh.get_world_pose()
        self.selected_pool=self.selected_pool + centroid
        pcd, _, _ =self.fps_np(self.selected_pool, 256)

        return pcd


    def fps_np(self, pcd, particle_num, init_idx=-1, seed = 0):
        """
        功能：实现基于“最远点采样”（Furthest Point Sampling, FPS）算法对输入点云进行下采样：

        1、首先从随机索引（或指定的 init_idx）选取初始点；

        2、迭代选择离当前已选点集合最远的点，直到采样数量达到 particle_num；

        3、返回下采样后的点云、采样点的索引列表和最大距离。
        """
        np.random.seed(seed)
        fps_idx = []
        assert pcd.shape[0] > 0
        if init_idx == -1:
            rand_idx = np.random.randint(pcd.shape[0])
        else:
            rand_idx = init_idx
        fps_idx.append(rand_idx)
        pcd_fps_lst = [pcd[rand_idx]]
        dist = np.linalg.norm(pcd - pcd_fps_lst[0], axis=1)
        while len(pcd_fps_lst) < particle_num:
            fps_idx.append(dist.argmax())
            pcd_fps_lst.append(pcd[dist.argmax()])
            dist = np.minimum(dist, np.linalg.norm(pcd - pcd_fps_lst[-1], axis=1))
        pcd_fps = np.stack(pcd_fps_lst, axis=0)
        return pcd_fps, fps_idx, dist.max()
        

    def Rotation(self,q,vector):
        vector = torch.from_numpy(vector).to(torch.float64)
        q0=q[0].item()
        q1=q[1].item()
        q2=q[2].item()
        q3=q[3].item()
        R=torch.tensor(
            [
                [1-2*q2**2-2*q3**2,2*q1*q2-2*q0*q3,2*q1*q3+2*q0*q2],
                [2*q1*q2+2*q0*q3,1-2*q1**2-2*q3**2,2*q2*q3-2*q0*q1],
                [2*q1*q3-2*q0*q2,2*q2*q3+2*q0*q1,1-2*q1**2-2*q2**2],
            ]
        ).to(torch.float64)
        vector=torch.mm(vector,R.transpose(1,0))
        return vector.numpy()