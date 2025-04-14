
from typing import Tuple
import numpy as np
from omni.isaac.franka import Franka
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core import World
# 两种不同的控制器，用于运动规划和抓取放置控制。
from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController
# 基于几何和动力学的运动规划算法
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
# 用于求解机械臂运动学问题。
from omni.isaac.franka import KinematicsSolver
from Env.Utils.transforms import euler_angles_to_quat
import torch
from Env.Utils.transforms import quat_diff_rad


class MyFranka:
    def __init__(self,world:World,pos=None,ori=None,prim_path:str=None,robot_name:str=None,):
        
        self.world=world

        if prim_path is None:
            self._franka_prim_path = find_unique_string_name(
                initial_name="/World/Franka", is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
        else:
            self._franka_prim_path=prim_path

        if robot_name is None:
            self._franka_robot_name = find_unique_string_name(
                initial_name="my_franka", is_unique_fn=lambda x: not self.world.scene.object_exists(x)
            )
        else:
            self._franka_robot_name=robot_name

        self.init_position=pos
        self.init_ori=ori
        self.default_ee_ori=torch.from_numpy(np.array([0,np.pi,0]))+ori

        self.world.scene.add(Franka(prim_path=self._franka_prim_path,
                                    name=self._franka_robot_name,position=pos,
                                    orientation=euler_angles_to_quat(ori)))
        
        self._robot:Franka=self.world.scene.get_object(self._franka_robot_name)
        self._articulation_controller=self._robot.get_articulation_controller()

        """
        创建 RMPFlowController（用于连续运动规划）、KinematicsSolver（求解运动学）
        和 PickPlaceController（用于抓取放置操作），并将它们与机器人关联。
        """
        self._controller=RMPFlowController(name="rmpflow_controller",
                                           robot_articulation=self._robot)
        self._kinematic_solver=KinematicsSolver(self._robot)
        self._pick_place_controller=PickPlaceController(name="pick_place_controller",
                                                        robot_articulation=self._robot,
                                                        gripper=self._robot.gripper)
        self._controller.reset()
        self._pick_place_controller.reset()
        
    def get_prim_path(self):
        return self._franka_prim_path
    
    def get_cur_ee_pos(self):
        ee_pos, R = self._controller.get_motion_policy().get_end_effector_as_prim().get_world_pose()
        return ee_pos, R


    def pick_and_place(self,pick,place):
        self._pick_place_controller.reset()
        self._robot.gripper.open()
        while 1:
            self.world.step(render=True)
            actions=self._pick_place_controller.forward(
                picking_position=pick,
                placing_position=place,
                current_joint_positions=self._robot.get_joint_positions(),
                end_effector_offset=np.array([0,0.005,0]),
            )
            if self._pick_place_controller.is_done():
                break
            self._articulation_controller.apply_action(actions)
    
    def open(self):
        for _ in range(10):
            self._robot.gripper.open()
            self.world.step(render=True)
    
    def close(self):
        for _ in range(10):
            self._robot.gripper.close()
            self.world.step(render=True)
        

    @staticmethod
    def interpolate(start_loc, end_loc, speed):
        """
        用于在起始位置和目标位置之间生成一个插值轨迹：
        计算两个点之间的距离，根据给定的 speed 计算应分成的段数（chunks）;
        利用 np.outer 生成一个插值序列，返回一组中间点。
        """
        start_loc = np.array(start_loc)
        end_loc = np.array(end_loc)
        dist = np.linalg.norm(end_loc - start_loc)
        chunks = dist // speed
        if chunks==0:
            chunks=1
        return start_loc + np.outer(np.arange(chunks+1,dtype=float), (end_loc - start_loc) / chunks)
    
    def position_reached(self, target,thres=0.03):
        if target is None:
            return True
        
        ee_pos, R = self._controller.get_motion_policy().get_end_effector_as_prim().get_world_pose()
        pos_diff = np.linalg.norm(ee_pos- target)
        #print(pos_diff)

        if pos_diff < thres:
            return True
        else:
            return False 
    
    def rotation_reached(self, target):
        if target is None:
            return True
        
        ee_pos, R = self._controller.get_motion_policy().get_end_effector_as_prim().get_world_pose()
        angle_diff = quat_diff_rad(R, target)[0]
        # print(f'angle diff: {angle_diff}')
        if angle_diff < 0.1:
            return True
        
    def move(self,end_loc,env_ori=None):
        """
        实现单步移动操作
        """
        start_loc=self.get_cur_ee_pos()[0]

        # if env_ori is None:
        #     env_ori=self.default_ee_ori
        if env_ori is not None:
            end_effector_orientation = euler_angles_to_quat(env_ori)
        else:
            end_effector_orientation = None

        target_joint_positions = self._controller.forward(
            target_end_effector_position=end_loc, 
            target_end_effector_orientation=end_effector_orientation
        )
        self._articulation_controller.apply_action(target_joint_positions)
    
    def reach(self,end_loc,env_ori=None):
        # if env_ori is not None:
        # #     env_ori=self.default_ee_ori
        #     end_effector_orientation = euler_angles_to_quat(env_ori)
        #     if self.position_reached(end_loc) and self.rotation_reached(end_effector_orientation):
        #         return True
        # else:
        if env_ori is None:
            if self.position_reached(end_loc):
                return True
        else:
            if self.position_reached(end_loc) and self.rotation_reached(
                euler_angles_to_quat(env_ori)):
                return True
        

    def movep(self,end_loc):
        """
        实现逐步逼近目标位置的操作：

        1、先执行一步仿真更新，再进入最多 200 步的循环；

        2、每步更新末端目标位置与默认姿态，通过 forward 计算目标关节动作，并发送给机器人；

        3、当 position_reached 返回 True 时提前退出循环。
        """
        self.world.step(render=True)
        start_loc=self.get_cur_ee_pos()[0]
        cur_step=0
        for i in range(200):
            self.world.step(render=True)
            end_effector_orientation = euler_angles_to_quat(self.default_ee_ori)
            target_joint_positions = self._controller.forward(
                target_end_effector_position=end_loc, 
                target_end_effector_orientation=end_effector_orientation
            )
            self._articulation_controller.apply_action(target_joint_positions)
            cur_step+=1
            if self.position_reached(end_loc):
                break
    
    def movel(self,end_loc:Tuple[np.ndarray,torch.Tensor],
              env_ori:Tuple[np.ndarray,torch.Tensor]=None):
        """
        movel 方法类似于 movep，但在循环中同时检查位置和旋转是否均满足要求：

        1、每次调用 world.step(render=True) 进行仿真更新；

        2、根据目标位置和目标姿态（env_ori，若未提供则使用 default_ee_ori）计算目标动作；

        3、当同时满足位置与旋转到达条件时退出；

        4、如果超过 300 步仍未到达，则打印失败提示并退出。
        """
        self.world.step(render=True)
        start_loc=self.get_cur_ee_pos()[0]
        cur_step=0
        while 1:
            self.world.step(render=True)
            if env_ori is None:
                env_ori=self.default_ee_ori
            end_effector_orientation = euler_angles_to_quat(env_ori)
            target_joint_positions = self._controller.forward(
                target_end_effector_position=end_loc, target_end_effector_orientation=end_effector_orientation
            )
            self._articulation_controller.apply_action(target_joint_positions)
            cur_step+=1
            if self.position_reached(end_loc) and self.rotation_reached(end_effector_orientation):
                break
            if cur_step>300:
                print("Failed to reach target")
                break
            
            

            

# class MyFranka_GPT:
#     def __init__(self, world: World, pos=None, ori=None, prim_path: str = None, robot_name: str = None):
#         self.world = world

#         # 生成唯一的 prim 路径和机器人名称
#         if prim_path is None:
#             self._franka_prim_path = find_unique_string_name("/World/Franka", is_unique_fn=lambda x: not is_prim_path_valid(x))
#         else:
#             self._franka_prim_path = prim_path

#         if robot_name is None:
#             self._franka_robot_name = find_unique_string_name("my_franka", is_unique_fn=lambda x: not self.world.scene.object_exists(x))
#         else:
#             self._franka_robot_name = robot_name

#         self.init_position = pos
#         self.init_ori = ori
#         # 例如默认末端执行器朝向可以基于初始欧拉角进行设置
#         self.default_ee_ori = torch.from_numpy(np.array([0, np.pi, 0])) + ori if ori is not None else None

#         # 添加 Franka 机器人到场景中（请确保资源路径在机器人类内部或外部正确配置）
#         from omni.isaac.franka import Franka  # 动态导入
#         self.world.scene.add(Franka(
#             prim_path=self._franka_prim_path,
#             name=self._franka_robot_name,
#             position=pos,
#             orientation=euler_angles_to_quat(ori) if ori is not None else None
#         ))
#         self._robot: Franka = self.world.scene.get_object(self._franka_robot_name)
#         self._articulation_controller = self._robot.get_articulation_controller()
#         # 此处控制器、运动规划、抓放控制器等根据实际需求初始化
#         from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
#         from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController
#         from omni.isaac.franka import KinematicsSolver
#         self._controller = RMPFlowController(name="rmpflow_controller", robot_articulation=self._robot)
#         self._kinematic_solver = KinematicsSolver(self._robot)
#         self._pick_place_controller = PickPlaceController(
#             name="pick_place_controller", robot_articulation=self._robot, gripper=self._robot.gripper
#         )
#         self._controller.reset()
#         self._pick_place_controller.reset()

#     def get_prim_path(self):
#         return self._franka_prim_path

#     def get_cur_ee_pos(self):
#         ee_pos, R = self._controller.get_motion_policy().get_end_effector_as_prim().get_world_pose()
#         return ee_pos, R

#     def pick_and_place(self, pick, place):
#         self._pick_place_controller.reset()
#         self._robot.gripper.open()
#         while True:
#             self.world.step(render=True)
#             actions = self._pick_place_controller.forward(
#                 picking_position=pick,
#                 placing_position=place,
#                 current_joint_positions=self._robot.get_joint_positions(),
#                 end_effector_offset=np.array([0, 0.005, 0]),
#             )
#             if self._pick_place_controller.is_done():
#                 break
#             self._articulation_controller.apply_action(actions)

#     def open(self):
#         for _ in range(10):
#             self._robot.gripper.open()
#             self.world.step(render=True)

#     def close(self):
#         for _ in range(10):
#             self._robot.gripper.close()
#             self.world.step(render=True)

#     def move(self, end_loc, env_ori=None):
#         start_loc = self.get_cur_ee_pos()[0]
#         if env_ori is not None:
#             end_effector_orientation = euler_angles_to_quat(env_ori)
#         else:
#             end_effector_orientation = None
#         target_joint_positions = self._controller.forward(
#             target_end_effector_position=end_loc, target_end_effector_orientation=end_effector_orientation
#         )
#         self._articulation_controller.apply_action(target_joint_positions)

