import copy
import threading
import cv2
import yaml
from Env.Config.SofaSceneConfig import SofaSceneConfig
from isaacsim import SimulationApp
from termcolor import cprint
import torch
hl = False
simulation_app = SimulationApp({"headless": hl})

from time import gmtime, strftime
from Env.Garment.Garment1 import Garment
from Env.Utils.transforms import euler_angles_to_quat
import carb
from Env.env.SofaEnv import SofaEnv
from Env.Camera.Sofa_Point_Cloud_Camera import Point_Cloud_Camera
import numpy as np
import os
from omni.isaac.core.utils.prims import delete_prim
from Control.Control import Control
from pxr import UsdGeom, UsdLux, Sdf, Gf, Vt, Usd, UsdPhysics, PhysxSchema
import random

import os
import sys
import isaacsim
# from omni.isaac.kit import SimulationApp
sys.path.append(os.getcwd())
# simulation_app = SimulationApp({"headless": False})

# Open the Simulation App


from Env.Camera.Recording_Camera1 import Recording_Camera
from Env.Camera.Sofa_Point_Cloud_Camera import Point_Cloud_Camera
from Env.Config.SofaSceneConfig import SofaSceneConfig
from Env.Garment.Garment1 import WrapGarment
from Env.Robot.Franka.WrapFranka import WrapFranka
from Env.Room.Room import Wrap_base, Wrap_basket, Wrap_room, Wrap_sofa
from Env.Utils.AttachmentBlock import AttachmentBlock
from Env.Utils.Sofa_Collision_Group import Collision_Group
from Env.Utils.utils import get_unique_filename, load_sofa_transport_helper, sofa_judge_final_poses, write_rgb_image
import carb


# ---------------------coding begin---------------------#
import numpy as np
import omni.replicator.core as rep
import threading
import time
import random
from termcolor import cprint
from omni.isaac.core import World
from omni.isaac.core import SimulationContext

from omni.kit.async_engine import run_coroutine
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.physx import acquire_physx_interface
from pxr import UsdGeom, UsdLux, Sdf, Gf, Vt, Usd, UsdPhysics, PhysxSchema
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from omni.isaac.sensor import Camera
from omni.kit.viewport.utility import get_active_viewport
from omni.isaac.core.utils.prims import delete_prim, set_prim_visibility
from omni.isaac.core.utils.viewports import set_camera_view



class SofaSimEnv:
    def __init__(self):
        self.world = World(backend="torch", device="cpu")  # ! important


        set_camera_view(
            eye=[-2.0, 1.1, 1.8],
            target=[0.0, 1.7, 0.2],
            camera_prim_path="/OmniverseKit_Persp",
        )
        physx_interface = acquire_physx_interface()
        physx_interface.overwrite_gpu_setting(1)  # garment render request

        # self.stage = self.world.scene.stage
        self.stage = self.world.stage

        # self.scene = world.scene
        self.scene = self.world.get_physics_context()._physics_scene
        self.scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0.0, -1))
        self.scene.CreateGravityMagnitudeAttr().Set(9.8)
        
        # 创建地板
        self.ground = FixedCuboid(
            prim_path="/World/Ground",
            name="ground",
            position=np.array([0.0, 0.0, 0.0]),
            scale=np.array([20.0, 20.0, 0.1]),
        )
        self.ground.set_collision_enabled(True)
        # self.ground.set_rigid_body_enabled(True)
        
        self.context = SimulationContext()
    
        self.set_physics_scene()

        self.demo_light = rep.create.light(position=[0, 0, 10], light_type="dome")

        # no need default ground plane
        # self.scene.add_default_ground_plane()
        
        self.config = SofaSceneConfig()
        


        self.recording_camera = Recording_Camera(
            self.config.recording_camera_position,
            self.config.recording_camera_orientation,
        )

        self.point_cloud_camera = Point_Cloud_Camera(
            self.config.point_cloud_camera_position,
            self.config.point_cloud_camera_orientation,
            garment_num=self.config.garment_num,
        )
        
        self.franka = WrapFranka(
            self.world,
            self.config.robot_position,
            self.config.robot_orientation,
            prim_path="/World/Franka",
            robot_name="franka_robot",
            usd_path="/home/pakwa/GPs/My/Assets/Robot/franka.usd",
        )

        self.base_layer = Wrap_base(
            self.config.base_layer_position,
            self.config.base_layer_orientation,
            self.config.base_layer_scale,
            self.config.base_layer_usd_path,
            self.config.base_layer_prim_path,
        )

        self.basket = Wrap_basket(
            self.config.basket_position,
            self.config.basket_orientation,
            self.config.basket_scale,
            self.config.basket_usd_path,
            self.config.basket_prim_path,
        )

        self.basket_2 = Wrap_basket(
            self.config.basket_2_position,
            self.config.basket_2_orientation,
            self.config.basket_2_scale,
            self.config.basket_2_usd_path,
            self.config.basket_2_prim_path,
        )

        self.sofa = Wrap_sofa(
            self.config.sofa_position,
            self.config.sofa_orientation,
            self.config.sofa_scale,
            self.config.sofa_usd_path,
            self.config.sofa_prim_path
        )


        delete_prim(f"/World/Room")
        # self.room = Wrap_room(
        #     self.config.room_position,
        #     self.config.room_orientation,
        #     self.config.room_scale,
        #     self.config.room_usd_path,
        #     self.config.room_prim_path,
        # )

        for i in range(self.config.garment_num):
            delete_prim(f"/World/Garment/garment_{i}")
        self.config.garment_num = random.choices([5])[0]
        print(f"garment_num: {self.config.garment_num}")
        self.wrapgarment = WrapGarment(
            self.stage,
            self.scene,
            self.config.garment_num,
            self.config.clothpath,
            self.config.garment_position,
            self.config.garment_orientation,
            self.config.garment_scale,
        )
        self.garment_index = [True] * self.config.garment_num

        load_sofa_transport_helper(self.world)

        self.collision = Collision_Group(self.stage)

        self.world.reset()
        cprint("world load successfully", "green", on_color="on_green")


        # initialize camera
        self.point_cloud_camera.initialize(self.config.garment_num)
        self.recording_camera.initialize()
        cprint("camera initialize successfully", "green")

        # begin to record gif
        gif_generation_thread = threading.Thread(
            target=self.recording_camera.get_rgb_graph
        )
        # gif_generation_thread.start()

        # transport garment
        self.garment_transportation()
        cprint("garment transportation finish!", "green")
        
        # ?
        for garment in self.wrapgarment.garment_group:
            garment.particle_material.set_friction(1.0)
            garment.particle_material.set_damping(10.0)
            garment.particle_material.set_lift(0.0)

        # delete helper
        delete_prim(f"/World/transport_helper/transport_helper_1")
        delete_prim(f"/World/transport_helper/transport_helper_2")
        delete_prim(f"/World/transport_helper/transport_helper_3")
        delete_prim(f"/World/transport_helper/transport_helper_4")

        self.franka.return_to_initial_position(self.config.initial_position)

        self.create_attach_block()

        cprint("world ready!", "green", on_color="on_green")
        self.robots = [self.franka]
        self.garments = self.wrapgarment.garment_group
        self.control = Control(self.world, self.robots, self.garments)





        filename = "/home/pakwa/GPs/My/RL/config/config0084.yaml"
        with open(filename, 'r') as file:
            task_config = yaml.safe_load(file)
        self.task_config = task_config

        self.target_point = np.array(self.task_config["target_point"])

        # 初始化衣物位置
        self.garment_positions = []
        self.garment_orientations = []
        self.num_garments = self.config.garment_num
        
        # 初始化搅拌计数器
        self.stir_count = 0
        self.max_stir_attempts = 5
        
        # 创建训练数据目录
        self.root_path = "/home/pakwa/GPs/SofaTrain"
        if not os.path.exists(self.root_path):
            os.mkdir(self.root_path)
        self.trial_path = os.path.join(self.root_path, "trial_pts.npy")
        self.model_path = os.path.join(self.root_path, "model.npy")

        # 初始化其他变量
        self.centroid = np.zeros(3)
        # 用于 PPO 的环境需要定义 observation_space 和 action_space
        from gymnasium import spaces
        self.observation_space = spaces.Box(low=-10, high=10, shape=(256,3), dtype=np.float32)
        self.action_space = spaces.Box(low=-5, high=5, shape=(3,), dtype=np.float32)

        self.successful_grasps = 0
        self.target_grasp_num = self.config.garment_num

    # 需要完成的接口有两个：
    # last_obs = self.sim_env.get_obs()
    # new_obs, rewards, dones, infos = self.sim_env.step(clipped_actions, True)


    # 原本的GarmentLab实现
    """
    def get_obs(self):
        obs = self.get_all_points()
        self.centroid = np.mean(obs, axis = 0)
        return obs - self.centroid

    def step(self,action, eval_succ = False):
        self.reset()
        self.control.robot_reset()
        for _ in range(20):
            self.world.step()
        # point=self.allocate_point(0, save_path=self.trial_path)
        action = action.reshape(-1)
        action += self.centroid
        action = self.get_point(action)


        # self.centroid 可能是衣物的质心，将 action 偏移到质心位置。
        # get_point(action) 计算最近的匹配点。

        # 计算 粒子点云 到目标点的距离 point_dist。
        # compute_reward()：用于计算奖励（目标点越接近，奖励越高）。


        particles = self.get_cloth_in_world_pose()
        point_dist = np.linalg.norm(particles[self.sel_particle_index] - action)
        reward = self.compute_reward(point_dist) if not eval_succ else self.compute_succ(point_dist)
        
        # self.control.grasp(pos=[action],ori=[None],flag=[True], wo_gripper=True)
        self.control.grasp(pos=[action],ori=[None],flag=[True])

        self.control.move(pos=[self.target_point],ori=[None],flag=[True])
        # self.control.move(pos=[self.target_point+np.array([0.1,0,0])],ori=[None],flag=[True])
        for _ in range(100):
            self.world.step()
        final_data=self.garment[0].get_vertices_positions()
        # reward = self.compute_reward(final_data)
        self.control.ungrasp([False])
        for _ in range(10):
            self.world.step()
        self.world.stop()
        if eval_succ:
            succ_flag = True if reward > 0 else False
            return None, succ_flag, np.array([True]), None
        return None, reward, np.array([True]), None
    """
    


    def reset(self, seed=None, options=None):  # !!!
        """重置环境状态"""
        
        # 使用config中的衣物位置
        for i, garment in enumerate(self.garments):
            position = self.config.garment_position[i]
            orientation = self.config.garment_orientation[i]
            garment.set_position(position)
            garment.set_orientation(orientation)

        self.franka.return_to_initial_position(self.config.initial_position)

        # 等待物理模拟稳定
        for _ in range(100):
            self.world.step(render=self._render)

        if not self.franka.is_initialized():
            print("机械臂初始化未完成，等待...")
            for _ in range(50):
                self.world.step(render=self._render)
        obs = self.get_obs()
        
        return obs, {}


    def get_obs(self):
        point_cloud, _ = self.point_cloud_camera.get_point_cloud_data(sample_flag=True, sample_num=1024)
        if point_cloud is None or len(point_cloud) == 0:
            # 返回全 0 状态
            return np.zeros((1024, 3), dtype=np.float32)
        point_cloud = np.array(point_cloud)
        self.centroid = np.mean(point_cloud, axis=0)

        cprint(f"相机位置: {self.point_cloud_camera.camera_position}", "magenta")
        cprint(f"点云中心: {np.mean(point_cloud, axis=0)}", "magenta")
        self.print_cloth_region()

        normalized_point_cloud = point_cloud - self.centroid
        if normalized_point_cloud.shape[0] > 256:
            normalized_point_cloud, _, _ = self.fps_np(normalized_point_cloud, 256)
        elif normalized_point_cloud.shape[0] < 256:
            pad = np.zeros((256 - normalized_point_cloud.shape[0], 3), dtype=np.float32)
            normalized_point_cloud = np.concatenate([normalized_point_cloud, pad], axis=0)
        return normalized_point_cloud.astype(np.float32)

    def step(self, action, eval_succ=False):            # !!!
        """执行一步动作"""
        """eval_succ:切换train和eval的逻辑"""
        action = action.reshape(-1)
        # 增加归一化处理
        scaled_action = np.clip(action, -0.5, 0.5)
        candidate = scaled_action + self.centroid
        print(f"PPO action: {action}, scaled: {scaled_action}")
        print(f"当前质心: {self.centroid}, candidate: {candidate}")

        grasp_point = self.get_point(candidate)
        print(f"最终抓取点: {grasp_point}")
        
        # 选择最近的衣物
        target_idx, target_garment = self.select_target_garment(grasp_point)
        if target_garment is None:
            print("未找到合适的衣物！")
            reward = -0.1
            done = False
            obs = self.get_obs()
            return obs, reward, done, {"reason": "no_garment_found"}
        print(f"选中的衣物索引：{target_idx}")

        reward_action = self.compute_reward_action(grasp_point)
        print(f"reward_action为{reward_action}\n")
        
        if reward_action < 0.1:
            self.stir_count += 1
            self._do_stir()
            reward = -0.1
            done = False
        else:
            print("grasp..\n")
            try:
                # self.control.grasp(pos=[grasp_point], ori=[None], flag=[True], assign_garment=target_idx)            
                self.set_attach_to_garment(attach_position=grasp_point)
                target_positions = copy.deepcopy(self.config.target_positions)
                fetch_result = self.franka.sofa_pick_place_procedure(
                    target_positions, self.attach
                )
                if not fetch_result:
                    cprint("failed", "red")
                self.attach.detach()
                self.franka.open()
                for _ in range(150):
                    self.world.step()
                self.franka.return_to_initial_position(self.config.initial_position)
                for _ in range(50):
                    self.world.step()
                self.stir = True

                self.successful_grasps += 1
                if self.successful_grasps < self.target_grasp_num:
                    done = False
                    reward = reward_action  # 可以根据需要对reward做一些调整
                else:
                    # 达到抓取目标，结束当前场景
                    done = True
                    reward = reward_action * 0.8 + self.compute_reward_result() * 0.2

            except Exception as e:
                print(f"❌ Move failed: {e}")
                self.franka.return_to_initial_position(self.config.initial_position)
                reward = -0.1
                done = False
                obs = self.get_obs()
                return obs, reward, done, {"reason": "move_exception"}
            # self.control.move(pos=[self.target_point], ori=[None], flag=[True])
            for _ in range(150):
                self.world.step()
            print("move complete?\n")
            # reward_result = self.compute_reward_result()
            # reward = reward_action * 0.8 + reward_result * 0.2
            # self.control.ungrasp([False])
            # done = True
        
        obs = self.get_obs()
        info = {"is_success": reward > 0.5, "successful_grasps": self.successful_grasps}
        
        return obs, reward, done, info

    def compute_reward_action(self, grasp_point):
        """计算抓取动作的奖励"""
        all_points = []
        for garment in self.garments[:self.num_garments]:
            points = garment.get_world_position()
            all_points.extend(points)
        all_points = np.array(all_points)
        # 计算抓取点到最近点的距离
        dist = np.linalg.norm(all_points - grasp_point, axis=1)
        min_dist = np.min(dist)
        # 计算奖励
        # 为什么总是10
        reward = 0.1 - min_dist
        if reward < 0:
            reward = reward  
        else:
            reward = 1/(0.2 - reward)
        return reward


    def compute_reward_result(self):
        """计算最终结果的奖励"""
        # 检查是否掉落
        # fallen_garments = 0
        # avg_distance_to_target = 0.0
        # for garment in self.garments[:self.num_garments]:
        #     points = garment.get_world_position()
        #     if np.any(points[:, 2] < -0.1):
        #         fallen_garments += 1
        #     distances = np.linalg.norm(points - self.target_point, axis=1)
        #     avg_distance_to_target += np.mean(distances)
        # avg_distance_to_target /= self.num_garments
        # # 距离目标点越近，奖励越高（小于 0.2 给满分）
        # target_reward = max(1.0 - avg_distance_to_target, 0.0)
        # # 如果掉了布料，减分
        # fall_penalty = fallen_garments / self.num_garments
        # reward = target_reward * (1.0 - fall_penalty)
        # return reward
        return 0

    def _do_stir(self):
        """执行搅拌动作，采用随机扰动当前质心附近多个候选点"""
        cprint("开始搅拌...", "red")
        # 例如，在当前质心附近取一定范围内的 3 个随机扰动点
        stir_offsets = []
        for _ in range(3):
            offset = np.random.uniform(-0.1, 0.1, size=3)
            stir_offsets.append(offset)
        # 对每个扰动点执行机器人移动并等待仿真更新
        for offset in stir_offsets:
            target_point = self.centroid + offset
            for i in target_point:
                self.franka.move(end_loc=i, env_ori=None)
            # 快速仿真若干步
            for _ in range(50):
                self.world.step()


    def get_point(self, position):
        """获取离给定位置最近的点"""
        all_points = []
        for garment in self.garments[:self.num_garments]:
            points = garment.get_world_position()
            all_points.extend(points)
        all_points = np.array(all_points)
        if all_points.shape[0] == 0:
            print("⚠️ Warning: No garment points found!")
            return np.zeros(3)
        dist = np.linalg.norm(all_points - position.reshape(1, -1), axis=1)
        idx = np.argmin(dist)
        return all_points[idx]
        
    def garment_transportation(self):
        """
        Let the clothes fly onto the sofa
        by changing the direction of gravity.
        """
        # change gravity direction
        self.scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1))
        self.scene.CreateGravityMagnitudeAttr().Set(15)
        for i in range(100):
            if not simulation_app.is_running():
                simulation_app.close()
            simulation_app.update()

    def create_attach_block(self, init_position=np.array([0.0, 0.0, 1.0]), scale=None):
        """
        Create attachment block and update the collision group at the same time.
        """
        # create attach block and finish attach
        self.attach = AttachmentBlock(
            self.world,
            self.stage,
            "/World/AttachmentBlock",
            self.wrapgarment.garment_mesh_path,
        )
        self.attach.create_block(
            block_name="attach",
            block_position=init_position,
            block_visible=False,
            scale=scale,
        )
        # update attach collision group
        self.collision.update_after_attach()
        for i in range(100):
            # simulation_app.update()
            self.world.step(render=True)

        cprint("attach block create successfully", "green")

    def set_attach_to_garment(self, attach_position):
        """
        push attach_block to new grasp point and attach to the garment
        """
        # set the position of block
        self.attach.set_block_position(attach_position)
        # create attach
        self.attach.attach()
        # render the world
        self.world.step(render=True)
        cprint("attach block set successfully", "green")

    def set_physics_scene(self):
        """配置物理仿真参数"""
        self.physics = self.world.get_physics_context()
        self.physics.enable_ccd(True)
        self.physics.enable_gpu_dynamics(True)
        self.physics.set_broadphase_type("gpu")
        self.physics.enable_stablization(True)
        # if self.rigid:
        #     self.physics.set_solver_type("TGS")
        #     self.physics.set_gpu_max_rigid_contact_count(10240000)
        #     self.physics.set_gpu_max_rigid_patch_count(10240000)
        carb.log_info("物理仿真参数已设置。")

    # 计算衣物点云包围盒
    def get_cloth_region(self):
        all_points = []
        for garment in self.garments[:self.num_garments]:
            pts = garment.get_world_position()
            all_points.extend(pts)
        all_points = np.array(all_points)
        min_xyz = np.min(all_points, axis=0)
        max_xyz = np.max(all_points, axis=0)
        center_xyz = (min_xyz + max_xyz) / 2.0
        return center_xyz, min_xyz, max_xyz
    
    def print_cloth_region(self):
        center, min_xyz, max_xyz = self.get_cloth_region()
        cprint("👕 get_world_position:", "magenta")
        cprint(f" - 中心: {center}", "magenta")
        cprint(f" - 范围 X: [{min_xyz[0]:.3f}, {max_xyz[0]:.3f}]", "magenta")
        cprint(f" - 范围 Y: [{min_xyz[1]:.3f}, {max_xyz[1]:.3f}]", "magenta")
        cprint(f" - 范围 Z: [{min_xyz[2]:.3f}, {max_xyz[2]:.3f}]", "magenta")


    def remove_garment(self, garment):
        """
        从仿真场景中删除指定衣物，并从内部列表中移除。
        garment 应该包含其 prim 路径或其它标识符。
        """
        prim_path = garment.get_prim_path()
        delete_prim(prim_path)
        # 从列表中删除该衣物
        if garment in self.garments:
            self.garments.remove(garment)
        cprint(f"Garment at {prim_path} removed from scene.", "green")
        self.num_garments -= 1

    def select_target_garment(self, grasp_point):
        """
        根据抓取点在世界坐标系下的坐标，选择离抓取点最近的衣物对象。
        这里我们假设每个 garment 都有一个 get_world_positions() 方法，
        或者你可以直接计算其所有粒子在世界坐标系下的中心点（质心）。
        """
        min_dist = 0.01
        target_garment = None
        target_idx = -1
        for idx, garment in enumerate(self.garments[:self.num_garments]):
            # 如果 garment 提供 get_world_positions()，使用它计算衣物的质心：
            world_points = garment.get_world_position()
            if world_points.size == 0:
                continue
            distances = np.linalg.norm(world_points - grasp_point, axis=1)
            local_min = np.min(distances)
            if local_min < min_dist:
                min_dist = local_min
                target_garment = garment
                target_idx = idx
        return target_idx, target_garment
    
    def fps_np(self, pcd, particle_num, init_idx=-1, seed = 0):
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