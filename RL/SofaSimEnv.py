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
        self.step_num = 0

    def reset(self, stage=1):
        
        self.step_num = 0
        self.franka._robot.initialize()
        self.world.step(render=True)
        self.franka.return_to_initial_position(self.config.initial_position)
        load_sofa_transport_helper(self.world)
        for garment in self.garments:
            delete_prim(garment.get_garment_prim_path())
        self.garments.clear()

        # self.config.garment_num = random.choices([5])[0]
        # self.config.garment_num = random.choices([3, 4, 5], [0.1, 0.45, 0.45])[0]
        if stage == 1:
            self.config.garment_num = 1
            self.stage1_obs_count = 10
        elif stage == 2:
            self.config.garment_num = 2
        elif stage == 5:
            self.config.garment_num = 5
        elif stage == 8:
            self.config.garment_num = 8
        
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
        self.no_cloud_point = False

        self.garment_transportation()
        delete_prim(f"/World/transport_helper/transport_helper_1")
        delete_prim(f"/World/transport_helper/transport_helper_2")
        delete_prim(f"/World/transport_helper/transport_helper_3")
        delete_prim(f"/World/transport_helper/transport_helper_4")
        delete_prim(f"/World/transport_helper/transport_helper_5")
        delete_prim(f"/World/transport_helper")
        self.world.step(render=True)
        self.point_cloud_camera.initialize(self.num_garments)
        for _ in range(15):
            self.world.step(render=True)
        cprint("🔁 Environment reset done.", "green")
        obs = self.get_obs()
        return obs, {}

    def get_obs(self):
        point_cloud, _ = self.point_cloud_camera.get_point_cloud_data(sample_flag=True, sample_num=1024)
        if point_cloud is None or len(point_cloud) == 0:
            # 返回全 0 状态
            self.no_cloud_point = True
            return np.zeros((256, 3), dtype=np.float32)
        point_cloud = np.array(point_cloud)
        self.centroid = np.mean(point_cloud, axis=0)

        cprint(f"点云中心: {np.mean(point_cloud, axis=0)}", "magenta")

        normalized_point_cloud = point_cloud - self.centroid
        if normalized_point_cloud.shape[0] > 256:
            normalized_point_cloud, _, _ = self.fps_np(normalized_point_cloud, 256)
        elif normalized_point_cloud.shape[0] < 256:
            pad = np.zeros((256 - normalized_point_cloud.shape[0], 3), dtype=np.float32)
            normalized_point_cloud = np.concatenate([normalized_point_cloud, pad], axis=0)
        self.no_cloud_point = False
        return normalized_point_cloud.astype(np.float32)



    def step(self, action, eval_succ=False, flag=True):            # !!!
        """执行一步动作"""
        """eval_succ:切换train和eval的逻辑"""
        """flag: 是否启用get_point"""
        self.franka.return_to_initial_position(self.config.initial_position)
        self.dropped = 0
        reward = 0
        done = False
        info = {}
        action = action.reshape(-1)
        candidate = action + self.centroid
        print(f"当前质心: {self.centroid}, candidate: {candidate}")
        reward += self.compute_reward_pick_point(candidate)

        grasp_point = self.get_point(candidate, flag)
        cprint(f"最终抓取点: {grasp_point}, 得分为 {reward}", "yellow")
        

        if self.config.garment_num == 10:
            self.stage1_obs_count -= 1
            if self.stage1_obs_count <= 0:
                self.reset(self.config.garment_num)
            obs = self.get_obs()
            return obs, reward, np.array([done]), {"reason": "stage1 - only point identification"}
        

        self.set_attach_to_garment(attach_position=grasp_point)
        target_positions = copy.deepcopy(self.config.target_positions)
        fetch_result = self.franka.fetch_garment_from_sofa(
            target_positions, self.attach, "Env_Eval/sofa_record.txt",
        )
        # 失败
        if not fetch_result:
            self.fail_num += 1
            cprint("fetch failed, 惩罚-3", "red")
            info["grasp_success"] = False
            self.attach.detach()
            self.franka.open()
            try:
                self.franka.return_to_initial_position(self.config.initial_position)
            except Exception as e:
                cprint(e, "red")
                cprint("cant return to initial", "red")
                self.franka._robot.initialize()
            for _ in range(10):
                self.world.step()
            reward += -3
            done = False
            obs = self.get_obs()
            return obs, reward, np.array([done]), {"reason": "cant fetch the point",
                                                    "grasp_success": False,
                                                    "dropped": self.dropped}
        # 失败
        idx, grasped_garment = self.detect_current_grasped_garment()
        if grasped_garment is None:
            self.fail_num += 1
            cprint("fetch failed, 惩罚-3", "red")
            info["grasp_success"] = False
            self.attach.detach()
            self.franka.open()
            try:
                self.franka.return_to_initial_position(self.config.initial_position)
            except Exception as e:
                self.franka._robot.initialize()
            for _ in range(10):
                self.world.step()
            reward += -3
            done = False
            obs = self.get_obs()
            return obs, reward, np.array([done]), {"reason": "cant fetch the point",
                                                    "grasp_success": False,
                                                    "dropped": self.dropped}
        # 成功
        info["grasp_success"] = True
        self.successful_grasps += 1
        self.last_grasped_idx = idx
        self.attach.detach()
        self.franka.open()
        self.franka.return_to_initial_position(self.config.initial_position)
        for _ in range(40):
            self.world.step()
        # 衣物判断
        is_in_basket , radio= self.is_garment_in_basket(grasped_garment, self.basket)
        reward += self.delete_dropped_garments(exclude_idx=self.last_grasped_idx)
        if is_in_basket:
            cprint(f"🎉 成功放入篮子！奖励+{50*(radio-0.6)}", "green")
            info["success"] = True
            reward += 50*(radio-0.6)
        else:
            cprint(f"⚠️ 没放好进篮子！奖励+3", "yellow")
            info["success"] = False
            self.dropped += 1
            reward += 3
        if grasped_garment is not None:
            self.remove_garment(grasped_garment)
        # 后处理
        info["dropped"] = self.dropped
        if self.num_garments <= 0 or self.successful_grasps >= self.target_grasp_num or self.fail_num >= 15:
            done = True
            self.reset(self.config.garment_num)
        obs = self.get_obs()
        return obs, reward, np.array([done]), info


    def step2(self, action, eval_succ=False, flag=True):
        reward = 0
        done = True
        info = {}
        action = action.reshape(-1)
        candidate = action + self.centroid
        print(f"当前质心: {self.centroid}, candidate: {candidate}")
        reward += self.compute_reward_pick_point(candidate)
        grasp_point = self.get_point(candidate, flag)
        cprint(f"最终抓取点: {grasp_point}, 得分为 {reward}", "yellow")
        self.set_attach_to_garment(attach_position=grasp_point)
        idx, grasped_garment = self.detect_current_grasped_garment()
        points = grasped_garment.get_world_position()
        points = points.cpu().numpy()
        centroid = np.mean(points, axis=0)
        if np.linalg.norm(centroid - grasp_point) > 0.02:
            reward = min(0.2 / np.linalg.norm(centroid - grasp_point), 10)
        else :
            reward = 10
        self.reset(1)
        obs = self.get_obs()
        return obs, reward, np.array([done]), info


    def step3(self, action, eval_succ=False, flag=False):
        reward = 0
        done = True
        info = {}
        action = action.reshape(-1)
        candidate = action + self.centroid
        print(f"当前质心: {self.centroid}, candidate: {candidate}")
        reward += self.compute_reward_pick_point(candidate)
        grasp_point = self.get_point(candidate, flag)
        cprint(f"最终抓取点: {grasp_point}, 得分为 {reward}", "yellow")
        self.reset(5)
        obs = self.get_obs()
        return obs, reward, np.array([done]), info


    def compute_reward_pick_point(self, grasp_point):
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
        reward = 0.03 - min_dist
        if reward < 0:
            reward = reward*20  # 最低约-10分
        else:
            reward = 1/(0.3 - reward)  # 最高约5分
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
        nearest_point = all_points[idx]
        min_z = 0.4360
        search_radius=0.015
        if nearest_point[2] > min_z:
            return nearest_point
        else:
            print(f"⚠️ 最近点 z={nearest_point[2]:.4f} 太低，尝试往上找")
            mask = (
            (np.abs(all_points[:, 0] - nearest_point[0]) < search_radius) &
            (np.abs(all_points[:, 1] - nearest_point[1]) < search_radius) &
            (all_points[:, 2] > min_z)
            )
            candidates = all_points[mask]
            if candidates.shape[0] > 0:
                # 找离原始位置最近的那个（保证尽量贴合原位置）
                cand_dist = np.linalg.norm(candidates[:, :2] - position[:2].reshape(1, -1), axis=1)
                best_idx = np.argmin(cand_dist)
                better_point = candidates[best_idx]
                print(f"✅ 找到更高点 z={better_point[2]:.4f}")
                return better_point
            else:
                print("⚠️ 附近没有更高的点，直接用最近的")
                return all_points[idx]

    

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
            return False, 0
        points = garment.get_world_position()
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        min_basket, max_basket = basket.get_aabb()
        inside = np.all((points >= min_basket) & (points <= max_basket), axis=1)
        ratio_inside = np.sum(inside) / len(points)
        return ratio_inside >= threshold, ratio_inside
    
    def remove_garment(self, garment):
        if garment in self.garments:
            self.garments.remove(garment)
        prim_path = garment.get_garment_prim_path()
        delete_prim(prim_path)
        cprint(f"Garment at {prim_path} removed from scene.", "green")
        self.num_garments -= 1

    def delete_dropped_garments(self, exclude_idx=None, threshold_z=0.4, ratio_threshold=0.5):
        dropped_reward = 0
        for idx in reversed(range(self.num_garments)):
            if idx == exclude_idx:
                continue
            garment = self.garments[idx]
            if garment is None:
                continue
            try:
                points = garment.get_world_position()
            except Exception as e:
                cprint(f"⚠️ 无法获取衣物 {idx} 的位置，跳过: {e}", "red")
                continue
            if points.size == 0:
                continue
            if isinstance(points, torch.Tensor):
                points = points.detach().cpu().numpy()
            below_ground = points[:, 2] < threshold_z
            if np.mean(below_ground) > ratio_threshold:
                cprint(f"⚠️ 衣物 {idx} 掉落地面！惩罚 -3", "yellow")
                self.dropped += 1
                dropped_reward -= 3
                if idx >= 0 and self.garments[idx]:
                    self.remove_garment(self.garments[idx])
        return dropped_reward
            


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

    def reset_franka_completely(self):
        try:
            delete_prim(self.franka._franka_prim_path)  # 删除原始机器人
            self.world.step(render=True)  # 关键一步：让场景刷新！
            self.franka = WrapFranka(
            self.world,
            self.config.robot_position,
            self.config.robot_orientation,
            prim_path=find_unique_string_name("/World/Franka",
                is_unique_fn=lambda x: not is_prim_path_valid(x)),
            robot_name=find_unique_string_name(initial_name="franka_robot",
                is_unique_fn=lambda x: not self.world.scene.object_exists(x),),
            usd_path=self.path + "/Assets/Robot/franka.usd",
        )
            self.franka.return_to_initial_position(self.config.initial_position)
            print("✅ Franka 重建成功！")
            return True
        except Exception as e:
            print(f"❌ 重建 Franka 失败: {e}")
            return False
        
    
