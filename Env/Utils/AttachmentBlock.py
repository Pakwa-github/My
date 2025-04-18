"""
AttachmentBlock
used to create a cube and attach the cube to the garment
in order to make Franka catch the garment smoothly
"""

import numpy as np
import torch
from omni.isaac.core.objects import DynamicCuboid
from pxr import PhysxSchema
from omni.isaac.core.utils.prims import delete_prim


class AttachmentBlock:
    def __init__(
        self, world, stage, prim_path="/World/AttachmentBlock/attach", 
        garment_path=None, collision_group=None
    ):
        """
        Args:
        - prim_path: The prim position of 'AttachmentBlock' directory
        - garment_path: the prims of all the garment in the stage
        """
        self.world = world
        self.stage = stage
        self.root_prim_path = prim_path
        self.garment_path = garment_path
        self.garment_num = len(garment_path)
        self.attachment_path_list = []
        for i in range(self.garment_num):
            self.attachment_path_list.append(garment_path[i] + f"/mesh/attachment")
        self.collision_group=collision_group

    # 你真的要害死我啦！！！！！！
    def create_block(self, block_name, block_position, block_visible, scale=None):
        self.block_path = self.root_prim_path + "/" + block_name
        if scale is None:
            scale = np.array([0.01, 0.01, 0.01])
        else:
            scale = np.array(scale)
        self.block = DynamicCuboid(
            prim_path=self.block_path,
            color=np.array([1.0, 0.0, 0.0]),
            name=block_name,
            position=block_position,
            scale=np.array([0.01, 0.01, 0.01]),
            mass=1.0,
            visible=block_visible,
        )
        self.world.scene.add(self.block)
        self.move_block_controller = self.block._rigid_prim_view
        # block can't be moved by external forces such as gravity and collisions
        self.block.disable_rigid_body_physics()
        # block can't be affected by gravity
        # self.move_block_controller.disable_gravities()
        # or you can choose to make block be affected by gravity
        # self.move_block_controller.enable_gravities()

        # me
        # self.collision_group.CreateIncludesRel().AddTarget(self.block_path)

        return self.move_block_controller

    def attach(self):
        # In this function we will try to attach the cube to all the garment
        # Actually attachment will be generated successfully only when the cube is close to the particle of clothes
        # So attach the cube to all the garment will be fine
        # It will achieve the goal that different garment may get tangled up.

        for i in range(self.garment_num):
            attachment = PhysxSchema.PhysxPhysicsAttachment.Define(
                self.stage, self.attachment_path_list[i]
            )
            attachment.GetActor0Rel().SetTargets([self.garment_path[i] + "/mesh"])
            attachment.GetActor1Rel().SetTargets([self.block_path])
            att = PhysxSchema.PhysxAutoAttachmentAPI(attachment.GetPrim())  # 自动寻找相近的 cloth 顶点并建立附着
            att.Apply(attachment.GetPrim())
            _ = att.CreateDeformableVertexOverlapOffsetAttr(defaultValue=0.02)  # 附着判定的“粘性半径”

    def detach(self):
        # delete all the attachment related to the cube
        for i in range(self.garment_num):
            delete_prim(self.attachment_path_list[i])
        # delete_prim(self.block_path)

    def set_block_position(
        self, grasp_point, grasp_orientations=torch.Tensor([1.0, 0.0, 0.0, 0.0])
    ):
        """
        use this function curiously, there may be some mistakes.
        """
        grasp_point = torch.Tensor(grasp_point)
        self.block.set_world_pose(grasp_point, grasp_orientations)

    def get_block_position(self):
        pos, rot = self.move_block_controller.get_world_poses()
        return pos

    def set_block_velocity(self, cmd):
        """
        set block velocity
        """
        self.move_block_controller.set_velocities(cmd)

    def enable_gravity(self):
        self.move_block_controller.enable_gravities()

    def disable_gravity(self):
        self.move_block_controller.disable_gravities()

    def disable_rigid_body_physics(self):
        self.block.disable_rigid_body_physics()

    def enable_rigid_body_physics(self):
        self.block.enable_rigid_body_physics()