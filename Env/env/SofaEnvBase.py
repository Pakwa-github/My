import os
import sys
sys.path.append(os.getcwd())
import copy
import threading
import cv2
import yaml
from Env.Config.SofaSceneConfig import SofaSceneConfig
from termcolor import cprint
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

class SofaSimEnvBase:
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
            prim_path=find_unique_string_name("/World/Franka",is_unique_fn=lambda x: not is_prim_path_valid(x)),
            robot_name=find_unique_string_name(initial_name="franka_robot",
                is_unique_fn=lambda x: not self.world.scene.object_exists(x),),
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
        self.room = Wrap_room(
            self.config.room_position,
            self.config.room_orientation,
            self.config.room_scale,
            self.config.room_usd_path,
            self.config.room_prim_path,
        )
        for i in range(self.config.garment_num):
            delete_prim(f"/World/Garment/garment_{i}")
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
        delete_prim(f"/World/transport_helper/transport_helper_5")
        delete_prim(f"/World/transport_helper")
        self.franka.return_to_initial_position(self.config.initial_position)
        self.create_attach_block()
        cprint("world ready!", "green", on_color="on_green")

        # self.robots = [self.franka]
        self.garments = self.wrapgarment.garment_group
        # self.control = Control(self.world, self.robots, self.garments)
        filename = "/home/pakwa/GPs/My/RL/config/config0084.yaml"
        with open(filename, 'r') as file:
            task_config = yaml.safe_load(file)
        self.task_config = task_config
        self.target_point = np.array(self.task_config["target_point"])
        self.demo_point = np.array(self.task_config["demo_point"])
        self.garment_name = self.task_config["garment_name"]
        self.task_name = self.task_config["task_name"]
        # self.garment_positions = []
        # self.garment_orientations = []
        self.num_garments = self.config.garment_num

        self.centroid = np.zeros(3)
        self.successful_grasps = 0
        self.target_grasp_num = self.config.garment_num
        self.fail_num = 0


    def garment_transportation(self):
        """
        Let the clothes fly onto the sofa
        by changing the direction of gravity.
        """
        # change gravity direction
        self.scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1))
        self.scene.CreateGravityMagnitudeAttr().Set(15)
        for i in range(100):
            self.world.step(render=True)

    def create_attach_block(self, init_position=np.array([0.0, 0.0, 1.0]), scale=None):
        """
        Create attachment block and update the collision group at the same time.
        """
        self.attach = AttachmentBlock(
            self.world,
            self.stage,
            "/World/AttachmentBlock",
            self.wrapgarment.garment_mesh_path,
        )
        from omni.isaac.core.utils.prims import is_prim_path_valid
        from omni.isaac.core.utils.string import find_unique_string_name
        self.attach.create_block(
            block_name=find_unique_string_name(initial_name="transport_helper_2",
                is_unique_fn=lambda x: not self.world.scene.object_exists(x),),
            block_position=init_position,
            block_visible=False,
            scale=scale,
        )
        self.collision.update_after_attach()
        for i in range(50):
            self.world.step(render=True)
        cprint("attach block create successfully", "green")

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