# 这段代码主要定义了两个类：

# AttachmentBlock：用于在机器人与衣物之间创建附着块（attachment），
# 即在仿真中通过物理约束把某个刚体（块）与衣物进行绑定，实现抓取或附着效果。

# Control：提供了对多个机器人与衣物系统的综合控制接口，包含碰撞分组设置、
# 附件的生成与解除、机器人运动规划以及抓取、移动、放置等操作的控制逻辑。


import numpy as np
from omni.isaac.core import World, SimulationContext, PhysicsContext
#simulation_context=SimulationContext(set_defaults=False)
from omni.isaac.core.utils.types import ArticulationAction
from pxr import UsdGeom, UsdLux, Sdf, Gf, Vt, Usd, UsdPhysics, PhysxSchema
from omni.isaac.franka import Franka

from omni.isaac.core.utils.stage import add_reference_to_stage

from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from Env.Utils.transforms import euler_angles_to_quat
from omni.isaac.core.objects import DynamicCuboid,FixedCuboid
from omni.isaac.sensor import Camera
import omni.replicator.core as rep
from Env.Robot.Franka.MyFranka import MyFranka
from omni.isaac.core.utils.prims import get_prim_at_path, delete_prim
from pxr import Gf, UsdGeom,Sdf, UsdPhysics, PhysxSchema, UsdLux, UsdShade
from Env.Garment.Garment import Garment
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
import torch


class AttachmentBlock():
    
    def __init__(self,world:World, robot:MyFranka, init_place, collision_group=None):
        """
        init_place: 附着块初始放置的位置（通常为抓取点或附着起始位置）。
        collision_group: 指定的碰撞分组，用于后续碰撞过滤设置。
        """
        self.world=world
        self.stage=self.world.stage
        self.name="attach"
        self.init_place=init_place
        self.robot=robot
        self.robot_path=self.robot.get_prim_path()
        self.attachment_path=find_unique_string_name(
            initial_name="/World/Attachment/attach", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        self.block_path=self.attachment_path+"/block"
        self.cube_name=find_unique_string_name(initial_name="cube", is_unique_fn=lambda x: not self.world.scene.object_exists(x))
        self.collision_group=collision_group
        self.block_control=self.create()


    def create(self,):
        """
        scale 固定为 [0.03, 0.03, 0.03]，说明尺寸较小。
        mass 设置为 1000（质量较大，防止运动时受到外力影响太大）。
        visible 设置为 False，即不在渲染中显示（可能只是起到物理控制作用）。

        1、获取该刚体的控制视图，用于后续设置速度位置等
        2、禁用重力
        
        ？、将该块的路径添加到指定的碰撞组中（使用 USD API 设置碰撞过滤），
        确保在仿真中可以被碰撞检测到或者与其他对象产生碰撞关系。
        """
        prim = DynamicCuboid(prim_path=self.block_path, color=np.array([1.0, 0.0, 0.0]),
                    name=self.cube_name,
                    position=self.init_place,
                    scale=np.array([0.03, 0.03, 0.03]),
                    mass=1000,
                    visible=False)
        self.block_prim=prim
        # self.world.scene.add(prim)
        self.move_block_controller=self.block_prim._rigid_prim_view
        self.move_block_controller.disable_gravities()
        self.collision_group.CreateIncludesRel().AddTarget(self.block_path) # ？
        return self.move_block_controller


    def set_velocities(self,velocities):
        """
        设置块的线性和角速度。这里期望传入一个 6 维向量（3 维线性速度 + 3 维角速度），
        并 reshape 成 (1,6) 的形式，然后调用控制器的 set_velocities 方法。
        """
        velocities=velocities.reshape(1,6)
        self.move_block_controller.set_velocities(velocities)

    def set_position(self,grasp_point):

        self.block_prim.set_world_pose(position=grasp_point)

    def get_position(self):
        pose,orientations=self.block_prim.get_world_pose()
        return pose

    def attach(self,garment:Garment):
        """
        attach() 方法：用于建立机器人抓取衣物时的物理附着关系。
        定义一个 PhysxPhysicsAttachment 对象，并使用附件路径创建；
        将衣物的 mesh prim 路径设为附件的一端；将当前块路径设为另一端；
        创建并应用自动附着 API，使得附着关系生效；
        设置一个 deformable 顶点重叠偏移属性（可能用于柔性衣物和刚体之间的附着计算），默认值为 0.02。
        """
        attachment = PhysxSchema.PhysxPhysicsAttachment.Define(self.stage, self.attachment_path)
        attachment.GetActor0Rel().SetTargets([garment.garment_mesh_prim_path])
        print(garment.garment_mesh_prim_path)  # 打印喵
        attachment.GetActor1Rel().SetTargets([self.block_path])
        att=PhysxSchema.PhysxAutoAttachmentAPI(attachment.GetPrim())
        att.Apply(attachment.GetPrim())
        _=att.CreateDeformableVertexOverlapOffsetAttr(defaultValue=0.02)

    def detach(self):
        """
        调用 delete_prim 删除该 prim，从而解除附件和仿真中该块的存在。
        """
        print(self.block_path)
        delete_prim(self.block_path)




class Control:
    def __init__(self,world:World,robot:list[MyFranka],garment:list[Garment],rigid=None):
        self.world=world
        self.robot=robot
        self.garment=garment
        # assert len(self.robot)==len(self.garment), "The number of robot and garment must be the same"
        self.stage=self.world.stage
        self.grasp_offset=torch.tensor([0.,0.,-0.01]) # why
        self.rigid=rigid
        # 设置碰撞组关系，确保不同类型的对象之间有合适的碰撞过滤规则。
        self.collision_group()  
        # 初始化一个附件列表，与机器人数量相同，
        # 存储每个机器人当前是否有附着块（AttachmentBlock）存在。
        self.attachlist=[None]*len(self.robot)  



    def collision_group(self):
        """ 该方法用于创建和配置多个 USD 碰撞组，以管理机器人、衣物、刚体和附件之间的碰撞关系。 """
        
        # 定义刚体碰撞组，并获取用于过滤其他组的关系对象。
        self.rigid_group_path="/World/Collision/Rigid_group"
        self.rigid_group = UsdPhysics.CollisionGroup.Define(self.stage, self.rigid_group_path)
        self.filter_rigid=self.rigid_group.CreateFilteredGroupsRel()
        
        # 同理，定义机器人碰撞组和相应过滤关系。
        self.robot_group_path="/World/Collision/robot_group"
        self.robot_group = UsdPhysics.CollisionGroup.Define(self.stage, self.robot_group_path)
        self.filter_robot = self.robot_group.CreateFilteredGroupsRel()
        
        # 定义衣物碰撞组和过滤关系。
        self.garment_group_path="/World/Collision/Garment_group"
        self.garment_group = UsdPhysics.CollisionGroup.Define(self.stage, self.garment_group_path)
        self.filter_garment = self.garment_group.CreateFilteredGroupsRel()
        
        # 定义附件碰撞组和过滤关系。
        self.attach_group_path="/World/attach_group"
        self.attach_group = UsdPhysics.CollisionGroup.Define(self.stage, self.attach_group_path)
        self.filter_attach = self.attach_group.CreateFilteredGroupsRel()
        
        # 设置机器人组过滤关系，使得机器人组会与衣物、刚体和附件组发生碰撞检测。
        self.filter_robot.AddTarget(self.garment_group_path)
        self.filter_robot.AddTarget(self.rigid_group_path)
        self.filter_robot.AddTarget(self.attach_group_path)
        
        # 设置衣物组过滤关系，使其与机器人和附件组产生碰撞。
        self.filter_garment.AddTarget(self.robot_group_path)
        # self.filter_garment.AddTarget(self.rigid_group_path)
        self.filter_garment.AddTarget(self.attach_group_path)
        
        # 刚体组过滤规则，添加机器人组和附件组目标。
        self.filter_rigid.AddTarget(self.robot_group_path)
        # self.filter_rigid.AddTarget(self.garment_group_path)
        self.filter_rigid.AddTarget(self.attach_group_path)
        
        # 附件组过滤规则，设置为与其他所有组都有碰撞关系。
        self.filter_attach.AddTarget(self.robot_group_path)
        self.filter_attach.AddTarget(self.garment_group_path)
        self.filter_attach.AddTarget(self.rigid_group_path)
        
        # 应用 USD 的 CollectionAPI 为机器人组创建集合，并将每个机器人的 prim 路径添加进来。
        self.collectionAPI_robot = Usd.CollectionAPI.Apply(self.filter_robot.GetPrim(), "colliders")
        for robot in self.robot:
            self.collectionAPI_robot.CreateIncludesRel().AddTarget(robot.get_prim_path())
        
        # 为衣物组设置集合，先添加整个 /World/Garment 目录，
        # 然后遍历每个衣物对象，将其网格 prim、衣物 prim 和粒子系统路径加入集合。
        self.collectionAPI_garment = Usd.CollectionAPI.Apply(self.filter_garment.GetPrim(), "colliders")
        self.collectionAPI_garment.CreateIncludesRel().AddTarget(f"/World/Garment")
        for garment in self.garment:
            self.collectionAPI_garment.CreateIncludesRel().AddTarget(garment.garment_mesh_prim_path)
            self.collectionAPI_garment.CreateIncludesRel().AddTarget(garment.garment_prim_path)
            self.collectionAPI_garment.CreateIncludesRel().AddTarget(garment.particle_system_path)
        
        # 为附件组创建集合，并添加附件路径。
        self.collectionAPI_attach = Usd.CollectionAPI.Apply(self.filter_attach.GetPrim(), "colliders")
        self.collectionAPI_attach.CreateIncludesRel().AddTarget("/World/Attachment")

        # 为刚体组创建集合，默认加入 /World/Avatar，如果传入了刚体对象，也加入其 prim 路径。
        self.collectionAPI_rigid = Usd.CollectionAPI.Apply(self.filter_rigid.GetPrim(), "colliders")
        self.collectionAPI_rigid.CreateIncludesRel().AddTarget("/World/Avatar")
        if self.rigid is not None:
            for rigid in self.rigid:
                self.collectionAPI_rigid.CreateIncludesRel().AddTarget(rigid.get_prim_path())


    def make_attachment(self,position:list,flag:list[bool]):
        for i in range(len(self.robot)):
            if flag[i] and self.attachlist[i] is None:
                self.attachlist[i]=AttachmentBlock(self.world,self.robot[i],position[i],collision_group=self.collectionAPI_attach)
            elif flag[i] and self.attachlist[i] is not None:
                continue
            else:
                self.attachlist[i]=None

    def control_gravities(self,flag:list):
        for i in range(len(self.attachlist)):
            if not flag[i]:
                self.attachlist[i].block_control.disable_gravities()


    def robot_goto_position(self,pos:list,ori:list,flag:list[bool],max_limit=10000):
        cur_step=0
        self.world.step()
        while 1:
            for i in range(len(self.robot)):
                if flag[i]:
                    self.robot[i].move(pos[i],ori[i])
            self.world.step()
            all_reach_flag=True
            for i in range(len(self.robot)):
                if flag[i]:
                    if not self.robot[i].reach(pos[i],ori[i]):
                        all_reach_flag=False
                        break
            if all_reach_flag or cur_step>max_limit:
                break
            cur_step+=1

    def robot_step(self,pos:list,ori:list,flag:list[bool]):
        for i in range(len(self.robot)):
            if flag[i]:
                self.robot[i].move(pos[i],ori[i])
            self.world.step()

    def attach(self,object_list,flag:list[bool]):
        """
        先调用 close() 关闭机器人抓手；

        再调用对应附件对象的 attach 方法，
        将衣物（object_list 的第一个对象）附着到机器人抓手上的附件块上。
        """
        for i in range(len(flag)):
            if flag[i]:
                self.robot[i].close()
                self.attachlist[i].attach(object_list[0])

    def robot_close(self,flag:list[bool]):
        for i in range(len(self.robot)):
            if flag[i]:
                self.robot[i].close()

    def robot_open(self,flag:list[bool]):
        for i in range(len(self.robot)):
            if not flag[i]:
                self.robot[i].open()

    def robot_reset(self):
        self.robot_open([False]*len(self.robot))





    def grasp(self,pos:list,ori:list,flag:list[bool],assign_garment = None):
        '''
        grasp_function
        pos: list of robots grasp position
        ori: list of robots grasp orientation
        flag: list of bool, grasp or not
        '''
        """
        1、根据输入的位置、姿态和 flag 控制机器人移动到抓取位置；

        2、暂停仿真（可能为了保证附着过程稳定）；

        3、创建附件，并将目标衣物附着上去；

        4、恢复仿真，并等待几步稳定后关闭抓手完成抓取。

        assign_garment: 指定要抓取的衣物索引，如果为 None，则默认抓取第一个衣物。
        """
        if assign_garment is None:
            target_garment = self.garment
        else:
            target_garment = [self.garment[assign_garment]]
        self.robot_goto_position(pos,ori,flag)
        self.world.pause()
        self.make_attachment(pos,flag)
        self.attach(target_garment,flag)
        self.world.play()
        for i in range(30):
            self.world.step()
        self.robot_close(flag)

    def move(self,pos:list,ori:list,flag:list[bool],max_limit=300):
        '''
        move_function
        pos: list of robots target position
        ori: list of robots target orientation
        flag: list of bool, grasp or not
        '''
        cur_step=0
        self.world.step()
        while 1:
            self.robot_step(pos,ori,flag)
            self.next_pos_list=[]
            for id in range(len(self.robot)):
                if not flag[id]:
                    continue
                robot_pos,robot_ori=self.robot[id].get_cur_ee_pos()
                # print(robot_pos)
                if isinstance(robot_pos,np.ndarray):
                    robot_pos=torch.from_numpy(robot_pos)
                if isinstance(robot_ori,np.ndarray):
                    robot_ori=torch.from_numpy(robot_ori)
                a=self.Rotation(robot_ori,self.grasp_offset)
                block_handle:AttachmentBlock=self.attachlist[id]
                block_cur_pos=block_handle.get_position()
                # block_cur_pos=torch.from_numpy(block_cur_pos)
                block_next_pos=robot_pos+a
                block_velocity=(block_next_pos-block_cur_pos)/(self.world.get_physics_dt()*3)
                # if torch.norm(block_cur_pos-block_next_pos)<0.01:
                #     block_velocity=torch.zeros_like(block_velocity)
                orientation_ped=torch.zeros_like(block_velocity)
                cmd=torch.cat([block_velocity,orientation_ped],dim=-1)
                block_handle.set_velocities(cmd)
                # block_handle.set_position(block_next_pos)

            # self.block_reach(self.next_pos_list)
            self.world.step()
            self.world.step()
            all_reach_flag=True
            for i in range(len(self.robot)):
                if flag[i]:
                    if not self.robot[i].reach(pos[i],ori[i]):
                        all_reach_flag=False
                        break
            if all_reach_flag or cur_step>max_limit:
                cmd=torch.zeros(6,)
                for id in range(len(self.robot)):
                    if not flag[id]:
                        continue
                    block_handle:AttachmentBlock=self.attachlist[id]
                    block_handle.set_velocities(cmd)
                break
    def ungrasp(self,flag, keep=None):
        '''
        ungrasp function
        flag: list of bool, grasp or not
        grasp is True
        '''
        for i in range(len(self.attachlist)):
            if not flag[i]:
                self.robot[i].open()
            if self.attachlist[i] is not None and not flag[i]:
                if keep is None or not keep[i]:
                    self.attachlist[i].detach()
                    self.attachlist[i]=None
                else:
                    self.attachlist[i].move_block_controller.enable_gravities()
                    self.attachlist[i]=None

    def Rotation(self,q,vector):
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
        )
        vector=torch.mm(vector.unsqueeze(0),R.transpose(1,0))
        return vector.squeeze(0)

    def change_garment(self,id,garment):
        self.garment[id]=garment
