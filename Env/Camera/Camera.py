import sys
import numpy as np
import time
from copy import deepcopy
import cv2
import numpy as np
from isaacsim import SimulationApp
import torch
import sys
# sys.path.append("/home/luhr/Tactile/IsaacTac/")
import open3d as o3d
from omni.isaac.core import World
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.sensor import Camera
import omni.replicator.core as rep
from omni.isaac.core.utils.rotations import euler_angles_to_quat
# from dgl.geometry import farthest_point_sampler


class MyCamera:
    def __init__(self,world:World,):
        self.world=world
        self.camera_handle_path=find_unique_string_name("/World/Camera",is_unique_fn=lambda x: not is_prim_path_valid(x))
        # 在场景中添加一个 XFormPrim 作为摄像头的“句柄”或容器。这个节点用于管理摄像头的位置和变换。
        self.camera_handle=self.world.scene.add(XFormPrim(
            prim_path=find_unique_string_name(self.camera_handle_path,is_unique_fn=lambda x: not is_prim_path_valid(x)),
            name=find_unique_string_name("Camera",is_unique_fn=lambda x: not self.world.scene.object_exists(x)),
        ))
        self.camera1_xform_path=find_unique_string_name(self.camera_handle_path+"/Camera1",is_unique_fn=lambda x: not is_prim_path_valid(x))
        self.camera_1_path=find_unique_string_name(self.camera1_xform_path+"/Camera",is_unique_fn=lambda x: not is_prim_path_valid(x))
        # 在场景中添加摄像头变换节点（XFormPrim），用于定位和管理摄像头的姿态。
        self.camera1_xform=self.world.scene.add(XFormPrim(
            prim_path=self.camera1_xform_path,
            name=find_unique_string_name("Camera1",is_unique_fn=lambda x: not self.world.scene.object_exists(x)),
        ))
        self.camera_1=Camera(
            prim_path=self.camera_1_path,
            resolution=[360,360],  # 分辨率
        )
        # self.camera_1.set_focal_length(6)
        # self.camera_1.set_focus_distance(300)

        # 将摄像头设置为单目模式（"mono"），即不使用立体视觉。
        self.camera_1.set_stereo_role("mono")


    def camera_reset(self):

        self.camera_1.initialize()
        
        """
        为摄像头添加多个后处理数据的通道：

        1、Distance to image plane：计算到图像平面的距离； 深度图

        2、Semantic segmentation：生成语义分割图；

        3、Pointcloud：生成点云数据（参数 False 表示可能不包括颜色信息或其他设置）；

        4、Distance to camera：计算到摄像头的距离。
        """
        self.camera_1.add_distance_to_image_plane_to_frame()
        self.camera_1.add_semantic_segmentation_to_frame()
        self.camera_1.add_pointcloud_to_frame(False)
        self.camera_1.add_distance_to_camera_to_frame()


        self.camera_1.post_reset()
        self.camera_1.set_local_pose(np.array([0,0,0]),euler_angles_to_quat(np.array([0,0,0])),camera_axes="usd")
        self.camera1_xform.set_world_pose(np.array([0.3,0,0]),orientation=np.array([1,0,0,0]))

        # 使用 Replicator 模块创建一个渲染产品对象，用于从摄像头采集渲染数据，分辨率为 360×360。
        self.render_product_1=rep.create.render_product(self.camera_1_path,[360,360])
        # 获取一个“pointcloud”类型的 annotator（注释器），用于从渲染产品中提取点云信息，
        # 然后将其与刚刚创建的渲染产品关联。
        self.annotator_1=rep.AnnotatorRegistry.get_annotator("pointcloud")
        self.annotator_1.attach(self.render_product_1)



    def get_pcd(self,vis:bool=False):
        """
        调用注释器的 get_data() 方法获取当前帧的点云数据及相关信息，返回一个包含数据和信息的字典。
            该字典包含以下键：
            points1：点云的坐标数据，调整成 N×3 的形状。

            colors1：点云的 RGB 颜色信息，reshape 成 N×4（可能包含 alpha 通道），并归一化到 [0,1]。

            normals1：点云法向量数据，reshape 成 N×4（可能也包含额外信息）。

            semantics1：每个点的语义标签。

            id1：这里简单生成了一个全 1 的数组，用作每个点的标识（可能用于后续处理）。
            """
        data1=self.annotator_1.get_data()
        points1=data1["data"].reshape(-1,3)
        colors1=data1["info"]["pointRgb"].reshape(-1,4)/255
        normals1=data1["info"]["pointNormals"].reshape(-1,4)
        semantics1=data1["info"]["pointSemantic"].reshape(-1)
        id1=np.ones_like(semantics1)
        if vis:
            """
            使用 Open3D 创建点云对象，并可视化采集到的点云数据（只使用 RGB 的前三个通道）
            """
            pcd=o3d.geometry.PointCloud()
            pcd.points=o3d.utility.Vector3dVector(points1)
            pcd.colors=o3d.utility.Vector3dVector(colors1[:,:3])
            o3d.visualization.draw_geometries([pcd])
        return {"pcd":{"points":points1,"id":id1,"colors":colors1,"semantics":semantics1,"normals":normals1}}


"""
    MyCamera 类 封装了在 IsaacSim 中创建摄像头节点、设置摄像头后处理功能、初始化和重置摄像头、以及通过 Replicator 注释器提取点云数据的完整流程。

    初始化时，它通过生成唯一的 prim 路径和名称，将摄像头及其变换节点添加到场景中，并创建 Camera 对象。

    camera_reset 方法对摄像头进行初始化与配置，并创建渲染产品及点云 annotator，保证摄像头采集的数据中包含距离、语义、点云等信息。

    get_pcd 方法从 annotator 中获取当前帧的点云数据，并支持可选的可视化展示，最后返回点云及其相关信息。

    这样的设计使得在 IsaacSim 中，用户可以方便地利用摄像头采集 RGB、深度、点云和语义数据，并可进一步用于任务规划、强化学习或后续数据处理。
"""