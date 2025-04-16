from typing import Union
from Env.Config.GarmentConfig import GarmentConfig
import torch
import numpy as np
import random
import omni.kit.commands

from omni.isaac.core.materials.preview_surface import PreviewSurface
from omni.isaac.core.utils.semantics import add_update_semantics, get_semantics
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.physx.scripts import physicsUtils, particleUtils
from pxr import Gf, UsdGeom, Sdf, UsdPhysics, PhysxSchema, UsdLux, UsdShade
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
import omni.isaac.core.utils.prims as prims_utils
from omni.isaac.core.prims import (
    XFormPrim,
    ClothPrim,
    RigidPrim,
    GeometryPrim,
    ParticleSystem,
)
from omni.isaac.core.materials.particle_material import ParticleMaterial


class WrapGarment:
    def __init__(
        self,
        stage,
        scene,
        garment_num,
        garment_usd_path_dict,
        garment_position,
        garment_orientation,
        garment_scale,
    ) -> None:
        self.stage = stage
        self.scene = scene
        self.garment_mesh_path = []
        self.garment_num = garment_num
        self.garment_group = []
        random_numbers = random.sample(range(0, 100), garment_num)
        for i in range(garment_num):
            # define key to get specific garment_usd_path
            key = f"cloth{random_numbers[i]}"
            print(key)
            if i == 147:
                garment = Garment(
                    self.stage,
                    self.scene,
                    garment_usd_path_dict[key],
                    garment_position[i],
                    garment_orientation[i],
                    [0.0029, 0.0029, 0.0029],
                    garment_index=i,
                )
            else:
                garment = Garment(
                    self.stage,
                    self.scene,
                    garment_usd_path_dict[key],
                    garment_position[i],
                    garment_orientation[i],
                    garment_scale[i],
                    garment_index=i,
                )
            self.garment_mesh_path.append(garment.get_garment_prim_path())
            self.garment_group.append(garment)

    def washmachine_get_cur_poses(self, garment_index):
        cur_pose = []
        for i in range(self.garment_num):
            if garment_index[i] is False:
                pose = [-1.0, -0.55, 0.0]
            else:
                pose = self.garment_group[i].garment_mesh.get_world_pose()[0]
            cur_pose.append(pose)
        return cur_pose

    def get_cur_poses(self):
        cur_pose = []
        for i in range(self.garment_num):
            pose = self.garment_group[i].garment_mesh.get_world_pose()[0]
            cur_pose.append(pose)
        return cur_pose


"""
Class Garment
used to grnerate one piece of garment
It will be encapsulated into Class WrapGarment to generate many garments(gatment_nums can be changed)
you can also use Class Garment seperately
"""


class Garment:
    def __init__(
        self,
        stage,
        scene,
        usd_path,
        position=torch.tensor,
        orientation:Union[torch.Tensor,np.ndarray]=np.array([0.0, 0.0, 0.0]),
        scale:np.ndarray=np.array([1.0, 1.0, 1.0]),
        garment_index=int,
    ):
        self.stage = stage
        self.scene = scene
        self.usd_path = usd_path
        self.garment_view = UsdGeom.Xform.Define(self.stage, "/World/Garment")
        self.garment_name = f"garment_{garment_index}"
        self.garment_prim_path = f"/World/Garment/garment_{garment_index}"
        self.garment_mesh_prim_path = self.garment_prim_path + "/mesh"
        self.particle_system_path = "/World/Garment/particleSystem"
        self.particle_material_path = "/World/Garment/particleMaterial"
        self.garment_position = position
        self.garment_orientation = euler_angles_to_quat(orientation, degrees=True)
        self.garment_scale = scale
        self.particle_contact_offset = 0.008

        self.garment_config = GarmentConfig(usd_path=self.usd_path,pos=self.garment_position,
                                            ori=self.garment_orientation,scale=np.array(self.garment_scale),
                                            particle_contact_offset=self.particle_contact_offset)
        # when define the particle cloth initially
        # we need to define global particle system & material to control the attribute of particle

        # particle system
        self.particle_system = ParticleSystem(
            prim_path=self.particle_system_path,
            simulation_owner=self.scene.GetPath(),
            # simulation_owner=self.world.get_physics_context().prim_path,
            particle_contact_offset=0.008,
            enable_ccd=True,
            global_self_collision_enabled=True,
            non_particle_collision_enabled=True,
            solver_position_iteration_count=20,  # 求解器迭代次数 可以提高碰撞求解精度 16
            # ----optional parameter---- #
            # contact_offset=0.01,  # 接触偏移
            # rest_offset=0.008,    # 静止偏移
            # solid_rest_offset=0.01, # 固体静止偏移
            # fluid_rest_offset=0.01, # 流体静止偏移
        )

        # particle material
        self.particle_material = ParticleMaterial(
            prim_path=self.particle_material_path,
            friction=0.0, # 设为1.0
            drag=0.0,
            lift=0.3,   # 设为0
            particle_friction_scale=1.0,
            particle_adhesion_scale=1.0,
            # damping=0.0, # 设为10.0
        )

        # bind particle material to particle system
        physicsUtils.add_physics_material_to_prim(
            self.stage,
            self.stage.GetPrimAtPath(self.particle_system_path),
            self.particle_material_path,
        )

        # add garment to stage
        add_reference_to_stage(self.usd_path, self.garment_prim_path)

        # garment configuration
        # define Xform garment in stage
        self.garment = XFormPrim(
            prim_path=self.garment_prim_path,
            name=self.garment_name,
            orientation=self.garment_orientation,
            position=self.garment_position,
            scale=self.garment_scale,
        )
        # add particle cloth prim attribute
        self.garment_mesh = ClothPrim(
            name=self.garment_name + "_mesh",
            prim_path=self.garment_mesh_prim_path,
            particle_system=self.particle_system,
            particle_material=self.particle_material,
            particle_mass=0.04,  # 0.05
            stretch_stiffness=1e15,
            bend_stiffness=4.0,  # 5.0
            shear_stiffness=4.0,    # 5.0
            spring_damping=20.0,  # 10
        )
        # get particle controller
        self.particle_controller = self.garment_mesh._cloth_prim_view

    def get_garment_prim_path(self):
        return self.garment_prim_path

    def set_mass(self,mass):
        physicsUtils.add_mass(self.world.stage, self.garment_mesh_prim_path, mass)

    def get_particle_system_id(self):
        self.particle_system_api=PhysxSchema.PhysxParticleAPI.Apply(self.particle_system.prim)
        return self.particle_system_api.GetParticleGroupAttr().Get()

    def get_vertice_positions(self):
        return self.garment_mesh._get_points_pose()
    
    def get_vertices_positions(self):
        return self.garment_mesh._get_points_pose()

    def get_realvertices_positions(self):
        return self.garment_mesh._cloth_prim_view.get_world_positions()

    def apply_visual_material(self,material_path:str):
        self.visual_material_path=find_unique_string_name(self.garment_prim_path+"/visual_material",is_unique_fn=lambda x: not is_prim_path_valid(x))
        add_reference_to_stage(usd_path=material_path,prim_path=self.visual_material_path)
        self.visual_material_prim=prims_utils.get_prim_at_path(self.visual_material_path)
        self.material_prim=prims_utils.get_prim_children(self.visual_material_prim)[0]
        self.material_prim_path=self.material_prim.GetPath()
        self.visual_material=PreviewSurface(self.material_prim_path)

        self.garment_mesh_prim=prims_utils.get_prim_at_path(self.garment_mesh_prim_path)
        self.garment_submesh=prims_utils.get_prim_children(self.garment_mesh_prim)
        if len(self.garment_submesh)==0:
            omni.kit.commands.execute('BindMaterialCommand',
            prim_path=self.garment_mesh_prim_path, material_path=self.material_prim_path)
        else:
            omni.kit.commands.execute('BindMaterialCommand',
            prim_path=self.garment_mesh_prim_path, material_path=self.material_prim_path)
            for prim in self.garment_submesh:
                omni.kit.commands.execute('BindMaterialCommand',
                prim_path=prim.GetPath(), material_path=self.material_prim_path)

    def set_pose(self, pos, ori):
        self.garment.set_world_pose(position=pos, orientation=ori)

    def get_particle_system(self):
        return self.particle_system
    
    def get_world_position(self):
        particle_positions = self.get_vertices_positions()
        position, orientation = self.garment_mesh.get_world_pose()
        scale = np.array(self.garment_scale)
        position = np.array(position)
        orientation = np.array(orientation)
        particle_positions = particle_positions * scale
        particle_positions = self.rotate_point_cloud(particle_positions, orientation)
        particle_positions = particle_positions + position
        return particle_positions


    def rotate_point_cloud(self, points, quaternion):
        w, x, y, z = quaternion
        rotation_matrix = np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]
        ])
        rotated_points = points @ rotation_matrix.T
        return rotated_points
