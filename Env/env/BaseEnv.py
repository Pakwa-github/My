import carb
import numpy as np
from time import gmtime, strftime

from omni.isaac.core import World
from omni.isaac.core import SimulationContext

from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims.xform_prim import XFormPrim
import omni.replicator.core as rep

from Env.Robot.Franka.MyFranka import MyFranka
from Env.Garment.Garment import Garment
from Env.Config.SceneConfig import SceneConfig
from Env.Config.FrankaConfig import FrankaConfig
from Env.Config.DeformableConfig import DeformableConfig
from Env.Config.GarmentConfig import GarmentConfig
from Env.Utils.transforms import euler_angles_to_quat
from Env.Camera.Recording_Camera import Recording_Camera
from Control.Control import Control

###############################################################################
# 环境基类，封装场景、物理、资源加载以及灯光配置
# 现在融合一下其他Env
###############################################################################
class BaseEnv:
    def __init__(self, scene_config: SceneConfig = None, rigid: bool = False,
                 deformable: bool = True, garment: bool = True, 
                 franka_config:FrankaConfig=None, Deformable_Config:DeformableConfig=None, 
                 garment_config:GarmentConfig=None ):
        self.rigid = rigid
        self.deformable = deformable
        self.garment = garment
        self.recording = False
        self.savings = []

        # 创建 World 对象与物理仿真上下文
        self.world = World()
        self.stage = self.world.scene.stage
        self.scene = self.world.scene
        self.context = SimulationContext()

        # 添加默认地面
        self.scene.add_default_ground_plane()
        carb.log_info("默认地面已添加。")

        # 加载场景配置（房间背景）
        self.scene_config = scene_config if scene_config is not None else SceneConfig()
        self.room = self.import_room(self.scene_config)

        # 设置物理仿真参数
        self.set_physics_scene()

        # 初始化灯光（此处使用 Replicator 创建 dome light，可根据需要调整）
        self.demo_light = rep.create.light(position=[0, 0, 0], light_type="dome")
        carb.log_info("灯光已初始化。")

        # 添加衣物
        if garment_config is None:
            self.garment_config=[GarmentConfig(ori=np.array([0,0,0]))]
        else:
            self.garment_config=garment_config
        self.garment:list[Garment]=[]
        for garment_config in self.garment_config:
            self.garment.append(Garment(self.world,garment_config))

        # 添加Franka
        if franka_config is None:
            self.franka_config=FrankaConfig()
        else:
            self.franka_config=franka_config
        self.robots=self.import_franka(self.franka_config)

        # Control
        self.control=Control(self.world,self.robots,[self.garment[0]])

        # 有关摄像头的配置
        self.camera = Recording_Camera(
            camera_position=np.array([0.0, 0, 6.75]),
            camera_orientation=np.array([0, 90.0, 90]),
            prim_path="/World/recording_camera",
        )

    def import_room(self, scene_config: SceneConfig):  # FlatGrid.usd
        """加载房间/场景背景"""
        room_prim_path = find_unique_string_name("/World/Room/room", is_unique_fn=lambda x: not is_prim_path_valid(x))
        room_name = find_unique_string_name(initial_name="room", is_unique_fn=lambda x: not self.world.scene.object_exists(x))
        add_reference_to_stage(usd_path=scene_config.room_usd_path, prim_path=room_prim_path)
        room = XFormPrim(
            prim_path=room_prim_path,
            name=room_name,
            position=scene_config.pos,
            orientation=euler_angles_to_quat(scene_config.ori),
            scale=scene_config.scale,
        )
        carb.log_info("房间/场景背景已加载。")
        return room

    def set_physics_scene(self):
        """配置物理仿真参数"""
        self.physics = self.world.get_physics_context()
        self.physics.enable_ccd(True)
        self.physics.enable_gpu_dynamics(True)
        self.physics.set_broadphase_type("gpu")
        self.physics.enable_stablization(True)
        if self.rigid:
            self.physics.set_solver_type("TGS")
            self.physics.set_gpu_max_rigid_contact_count(10240000)
            self.physics.set_gpu_max_rigid_patch_count(10240000)
        carb.log_info("物理仿真参数已设置。")

    def import_franka(self, franka_config: FrankaConfig):
        """根据配置导入 Franka 机器人"""
        self.franka_list: list[MyFranka] = []
        for i in range(franka_config.franka_num):
            self.franka_list.append(MyFranka(self.world, pos=franka_config.pos[i], ori=franka_config.ori[i]))
        carb.log_info("Franka 机器人已加载。")
        return self.franka_list

    def import_garment(self, garment_usd_path: str):
        """加载衣物 USD 资源"""
        cloth_prim_path = find_unique_string_name(
            "/World/Garment/garment",
            is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        add_reference_to_stage(usd_path=garment_usd_path, prim_path=cloth_prim_path)
        carb.log_info("衣物资源已加载。")
        return cloth_prim_path

    def reset(self):
        self.world.reset()
        carb.log_info("环境已重置。")

    def step(self):
        self.world.step(render=True)

    def stop(self):
        self.world.stop()

    # 以下录制与回放接口留待扩展
    def record(self):
        if not self.recording:
            self.recording = True
            self.replay_file_name = strftime("Assets/Replays/%Y%m%d-%H:%M:%S.npy", gmtime())
            self.context.add_physics_callback("record_callback", self.record_callback)

    def stop_record(self):
        if self.recording:
            self.recording = False
            self.context.remove_physics_callback("record_callback")
            np.save(self.replay_file_name, np.array(self.savings))
            self.savings = []

    def record_callback(self, step_size):
        pass

    def replay_callback(self, data):
        pass

    def __replay_callback(self, step_size):
        if self.time_ptr < self.total_ticks:
            self.replay_callback(self.data[self.time_ptr])
            self.time_ptr += 1

    def replay(self, file):
        self.data = np.load(file, allow_pickle=True)
        self.time_ptr = 0
        self.total_ticks = len(self.data)
        self.context.add_physics_callback("replay_callback", self.__replay_callback)
        if self.deformable:
            # 设置 GPU 处理软体（可变形物体）时允许的最大接触点数量，防止由于过多接触点导致内存或性能问题
            # 原作注释了此条
            self.physics.set_gpu_max_soft_body_contacts(1024000)
            # GPU 碰撞堆栈（collision stack）是物理引擎内部用于存储和处理所有碰撞检测结果的数据结构。
            # 在 GPU 加速的物理计算中，碰撞堆栈会存储所有检测到的碰撞接触数据，并在后续步骤中对这些数据进行处理，
            # 比如碰撞响应、摩擦计算等。
            self.physics.set_gpu_collision_stack_size(3000000)