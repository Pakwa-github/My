from isaacsim import SimulationApp
hl = False
simulation_app = SimulationApp({"headless": hl})



import copy

import threading
from Env.env.SofaEnvBase import SofaSimEnvBase
import cv2
import yaml
from Env.Config.SofaSceneConfig import SofaSceneConfig

from termcolor import cprint
import torch


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
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name


class SofaSimEnv(SofaSimEnvBase):
    def __init__(self):
        super().__init__()

    def reset(self):
        print("🔁 Resetting simulation environment...")
        self.franka.return_to_initial_position(self.config.initial_position)
        load_sofa_transport_helper(self.world)
        for garment in self.garments:
            delete_prim(garment.get_garment_prim_path())
        self.garments.clear()
        self.config.garment_num = random.choices([5])[0]
        # self.config.garment_num = random.choices([3, 4, 5], [0.1, 0.45, 0.45])[0]
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
        self.garments = self.wrapgarment.garment_group
        self.num_garments = self.config.garment_num
        self.target_grasp_num = self.config.garment_num
        # 重置一些抓取变量
        self.centroid = np.zeros(3)
        self.successful_grasps = 0
        self.last_grasped_idx = -1
        self.fail_num = 0
        self.garment_transportation()
        for garment in self.wrapgarment.garment_group:
            garment.particle_material.set_friction(1.0)
            garment.particle_material.set_damping(10.0)
            garment.particle_material.set_lift(0.0)
        for _ in range(20):
            self.world.step(render=True)
        delete_prim(f"/World/transport_helper/transport_helper_1")
        delete_prim(f"/World/transport_helper/transport_helper_2")
        delete_prim(f"/World/transport_helper/transport_helper_3")
        delete_prim(f"/World/transport_helper/transport_helper_4")
        delete_prim(f"/World/transport_helper/transport_helper_5")
        delete_prim(f"/World/transport_helper")
        for _ in range(20):
            self.world.step(render=True)
        print("✅ Environment reset done.")
        return self.get_obs(), {}

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



    def step(self, action, eval_succ=False, flag=True):            # !!!
        """执行一步动作"""
        """eval_succ:切换train和eval的逻辑"""
        """flag: 是否启用get_point"""

        reward = 0
        done = False

        action = action.reshape(-1)
        # 增加归一化处理
        self.clip = 0.1 + self.successful_grasps * 0.1 
        scaled_action = np.clip(action, -self.clip, self.clip)
        candidate = scaled_action + self.centroid
        print(f"PPO action: {action}, scaled: {scaled_action}")
        print(f"当前质心: {self.centroid}, candidate: {candidate}")
        grasp_point = self.get_point(candidate, flag)
        print(f"最终抓取点: {grasp_point}")
        reward += self.compute_reward_pick_point(grasp_point, flag)
        
        judge_thread = threading.Thread(
            target=self.recording_camera.judge_contact_with_ground,
            args=("Data/Sofa//Record.txt",),
        )
        judge_thread.start()
        
        self.set_attach_to_garment(attach_position=grasp_point)
        target_positions = copy.deepcopy(self.config.target_positions)
        fetch_result = self.franka.fetch_garment_from_sofa(
            target_positions, self.attach, "Env_Eval/sofa_record.txt",
        )
        if not fetch_result:
            self.fail_num += 1
            cprint("failed", "red")
            self.attach.detach()
            self.franka.open()
            self.franka.return_to_initial_position(self.config.initial_position)
            for _ in range(50):
                self.world.step()
            reward += -3
            done = False
            obs = self.get_obs()
            return obs, reward, np.array([done]), {"reason": "cant fetch the point"}
        
        if self.recording_camera.contact:
            self.fail_num += 1
            cprint("触地惩罚！-3", "magenta")
            reward += -3
        else:
            self.successful_grasps += 1
        self.recording_camera.contact = False
        self.recording_camera.stop_judge_contact()

        idx, grasped_garment = self.detect_current_grasped_garment()
        if grasped_garment is not None:
            self.last_grasped_idx = idx
        self.attach.detach()
        self.franka.open()
        self.franka.return_to_initial_position(self.config.initial_position)
        for _ in range(100):
            self.world.step()
        if self.last_grasped_idx >= 0:
            is_in_basket = self.is_garment_in_basket(self.garments[self.last_grasped_idx], self.basket)
        if is_in_basket:
            cprint("🎉 成功放入篮子！奖励+20", "green")
            reward += 20
        if self.last_grasped_idx >= 0:
            self.remove_garment(self.garments[self.last_grasped_idx])
    
        print("one step\n")
        info = {"success": reward > 5}
        if self.num_garments <= 0 or self.successful_grasps >= self.target_grasp_num or self.fail_num >= 20:
            done = True
            self.reset()
        obs = self.get_obs()
        return obs, reward, np.array([done]), info

    def compute_reward_pick_point(self, grasp_point, flag):
        if flag:
            return 1
        """计算抓取动作的奖励"""
        all_points = []
        for garment in self.garments[:self.num_garments]:
            points = garment.get_world_position()
            all_points.extend(points)
        all_points = np.array(all_points)
        # 计算抓取点到最近点的距离
        dist = np.linalg.norm(all_points - grasp_point, axis=1)
        min_dist = np.min(dist)
        # 粘性半径是0.02
        reward = 0.04 - min_dist
        if reward < 0:
            reward = reward * 6  # 最低约-3分
        else:
            reward = 1/(0.2 - reward)  # 最高6.25分
        return reward


    def get_point(self, position, flag):
        """获取离给定位置最近的点"""
        if not flag:
            return position
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

    def set_attach_to_garment(self, attach_position):
        """
        push attach_block to new grasp point and attach to the garment
        """
        self.attach.set_block_position(attach_position)
        self.attach.attach()
        self.world.step(render=True)
        cprint("attach block set successfully", "green")

    def detect_current_grasped_garment(self):
        """
        遍历所有衣物，找到离 attach block 最近的衣物。
        """
        attach_pos, _ = self.attach.block.get_world_pose()  # 获取当前 attach block 的位置（世界坐标）
        min_dist = float('inf')
        target_idx = -1
        for idx, garment in enumerate(self.garments[:self.num_garments]):
            garment_points = garment.get_world_position()
            if garment_points.size == 0:
                continue
            dists = np.linalg.norm(garment_points - attach_pos, axis=1)
            min_local_dist = np.min(dists)
            if min_local_dist < min_dist and min_local_dist < 0.1:
                min_dist = min_local_dist
                target_idx = idx
        if target_idx == -1:
            print("❌ 没找到抓取衣物")
            return None, None
        print(f"✅ 检测到抓取的是第 {target_idx} 件衣物")
        return target_idx, self.garments[target_idx]
    
    def is_garment_in_basket(self, garment, basket, threshold=0.8):
        if garment is None:
            return False
        points = garment.get_world_position()
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        min_basket, max_basket = basket.get_aabb()
        inside = np.all((points >= min_basket) & (points <= max_basket), axis=1)
        ratio_inside = np.sum(inside) / len(points)
        return ratio_inside >= threshold
    
    def remove_garment(self, garment):
        if garment in self.garments:
            self.garments.remove(garment)
        prim_path = garment.get_garment_prim_path()
        delete_prim(prim_path)
        cprint(f"Garment at {prim_path} removed from scene.", "green")
        self.num_garments -= 1

    



    # 计算衣物点云包围盒
    def get_cloth_region(self):
        all_points = []
        for garment in self.garments[:self.num_garments]:
            pts = garment.get_world_position()
            all_points.extend(pts)
        all_points = np.array(all_points)
        if all_points.size == 0:
            # 如果衣物列表为空，返回空的中心和边界
            default_center = np.zeros(3)
            default_min = np.zeros(3)
            default_max = np.zeros(3)
            return default_center, default_min, default_max
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
    
    def compute_reward_result(self):
        """计算最终结果的奖励"""
        return 0

    # def _do_stir(self):
    #     """执行搅拌动作，采用随机扰动当前质心附近多个候选点"""
    #     cprint("开始搅拌...", "red")
    #     # 例如，在当前质心附近取一定范围内的 3 个随机扰动点
    #     stir_offsets = []
    #     for _ in range(3):
    #         offset = np.random.uniform(-0.1, 0.1, size=3)
    #         stir_offsets.append(offset)
    #     # 对每个扰动点执行机器人移动并等待仿真更新
    #     for offset in stir_offsets:
    #         target_point = self.centroid + offset
    #         for i in target_point:
    #             self.franka.move(end_loc=i, env_ori=None)
    #         # 快速仿真若干步
    #         for _ in range(50):
    #             self.world.step()
