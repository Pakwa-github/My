import numpy as np
import os

import torch
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from pxr import UsdGeom, UsdLux, Sdf, Gf, Vt, Usd, UsdPhysics, PhysxSchema
from omni.isaac.core.utils.prims import delete_prim, set_prim_visibility
from termcolor import cprint
# from plyfile import PlyData, PlyElement

# 构建一条传送带，包括底座 + 两侧轨道，用来模拟衣物传输路径
def load_conveyor_belt(world, i=0, j=0):
    """
    Use Cube to generate Conveyor belt
    aim to make cubes move into the washing machine as expected
    """
    world.scene.add(
        FixedCuboid(
            name="transport_base",
            position=[-4.275 + i * 2, 0.0 + j * 2, 0.55],
            prim_path="/World/Conveyor_belt/cube1",
            scale=np.array([8, 0.56, 0.025]),
            orientation=euler_angles_to_quat([0, 0, 0], degrees=True),
            size=1.0,
            color=np.array([180, 180, 180]),
            visible=False,
        )
    )

    world.scene.add(
        FixedCuboid(
            name="transport_side_left",
            position=[-4.275 + i * 2, -0.245 + j * 2, 0.62],
            prim_path="/World/Conveyor_belt/cube2",
            scale=np.array([8, 0.17, 0.025]),
            orientation=euler_angles_to_quat([90, 0, 0], degrees=True),
            size=1.0,
            color=np.array([180, 180, 180]),
            visible=False,
        )
    )

    world.scene.add(
        FixedCuboid(
            name="transport_side_right",
            position=[-4.275 + i * 2, 0.235 + j * 2, 0.62],
            prim_path="/World/Conveyor_belt/cube3",
            scale=np.array([8, 0.17, 0.025]),
            orientation=euler_angles_to_quat([90, 0, 0], degrees=True),
            size=1.0,
            color=np.array([180, 180, 180]),
            visible=False,
        )
    )


def load_washmachine_model(world, i=0, j=0):
    """
    Use Cube to generate washmachine model
    aim to make garment stay in the right position inside the washmachine and make franka avoid potential collision.
    return cube_list
    will use cube_list to add obstacle
    """
    cube_list = []

    # cube1
    cube_list.append(
        FixedCuboid(
            name="model_1",
            position=[0.04262 + i * 2, 0.0023 + j * 2, 0.17957 - 0.05],
            prim_path="/World/Washmachine_Model/cube1",
            scale=np.array([0.80221, 0.78598, 0.4]),
            orientation=euler_angles_to_quat([0, 0, 0], degrees=True),
            size=1.0,
            color=np.array([180, 180, 180]),
            visible=False,
        )
    )
    # cube2
    cube_list.append(
        FixedCuboid(
            name="model_2",
            position=[-0.23759 + i * 2, -0.01906 + j * 2, 0.40208 - 0.05],
            prim_path="/World/Washmachine_Model/cube2",
            scale=np.array([0.1325, 0.74446, 0.22551]),
            orientation=euler_angles_to_quat([0, 0, 0], degrees=True),
            size=1.0,
            color=np.array([180, 180, 180]),
            visible=False,
        )
    )
    # cube3
    cube_list.append(
        FixedCuboid(
            name="model_3",
            position=[0.03768 + i * 2, -0.36248 + j * 2, 0.65784 - 0.05],
            prim_path="/World/Washmachine_Model/cube3",
            scale=np.array([0.79763, 0.06466, 0.73032]),
            orientation=euler_angles_to_quat([0, 0, 0], degrees=True),
            size=1.0,
            color=np.array([180, 180, 180]),
            visible=False,
        )
    )
    # cube4
    cube_list.append(
        FixedCuboid(
            name="model_4",
            position=[0.28322, 0.00161, 0.58555 - 0.05],
            prim_path="/World/Washmachine_Model/cube4",
            scale=np.array([0.25836, 0.79141, 0.87574]),
            orientation=euler_angles_to_quat([0, 0, 0], degrees=True),
            size=1.0,
            color=np.array([180, 180, 180]),
            visible=False,
        )
    )
    # cube5
    cube_list.append(
        FixedCuboid(
            name="model_5",
            position=[0.06059 + i * 2, 0.00129 + j * 2, 1.07731 - 0.05],
            prim_path="/World/Washmachine_Model/cube5",
            scale=np.array([0.78024, 0.79109, 0.1115]),
            orientation=euler_angles_to_quat([0, 0, 0], degrees=True),
            size=1.0,
            color=np.array([180, 180, 180]),
            visible=False,
        )
    )
    # cube6
    cube_list.append(
        FixedCuboid(
            name="model_6",
            position=[0.04453 + i * 2, 0.36355 + j * 2, 0.65856 - 0.05],
            prim_path="/World/Washmachine_Model/cube6",
            scale=np.array([0.79763, 0.06466, 0.72789]),
            orientation=euler_angles_to_quat([0, 0, 0], degrees=True),
            size=1.0,
            color=np.array([180, 180, 180]),
            visible=False,
        )
    )
    # cube7
    cube_list.append(
        FixedCuboid(
            name="model_7",
            position=[-0.23759 + i * 2, -0.00035 + j * 2, 1.0 - 0.05],
            prim_path="/World/Washmachine_Model/cube7",
            scale=np.array([0.1325, 0.78557, 0.16468]),
            orientation=euler_angles_to_quat([0, 0, 0], degrees=True),
            size=1.0,
            color=np.array([180, 180, 180]),
            visible=False,
        )
    )
    # cube8
    cube_list.append(
        FixedCuboid(
            name="model_8",
            position=[-0.23634 + i * 2, -0.3 + j * 2, 0.69363 - 0.05],
            prim_path="/World/Washmachine_Model/cube8",
            scale=np.array([0.1325, 0.44936, 0.24605]),
            orientation=euler_angles_to_quat([90.0, 0, 0], degrees=True),
            size=1.0,
            color=np.array([180, 180, 180]),
            visible=False,
        )
    )
    # cube9
    cube_list.append(
        FixedCuboid(
            name="model_9",
            position=[-0.23634 + i * 2, 0.3 + j * 2, 0.69363 - 0.05],
            prim_path="/World/Washmachine_Model/cube9",
            scale=np.array([0.1325, 0.44936, 0.24042]),
            orientation=euler_angles_to_quat([90.0, 0, 0], degrees=True),
            size=1.0,
            color=np.array([180, 180, 180]),
            visible=False,
        )
    )
    cube_list.append(
        FixedCuboid(
            name="slope",
            position=[-0.00998, -0.00225, 0.39 - 0.05],
            prim_path="/World/Washmachine_Model/slope",
            scale=np.array([0.7, 0.8, 0.05]),
            orientation=euler_angles_to_quat([0, 17, 0], degrees=True),
            size=1.0,
            color=np.array([180, 180, 180]),
            visible=False,
        )
    )

    return cube_list

# 根据已有文件自动编号，避免覆盖，用于保存图片或点云文件（PLY）
def get_unique_filename(base_filename, extension=".png", counter_return=False):
    counter = 0
    filename = f"{base_filename}_{counter}{extension}"
    while os.path.exists(filename):
        counter += 1
        filename = f"{base_filename}_{counter}{extension}"

    if extension == ".ply":
        return filename, counter
    if counter_return:
        return filename, counter
    else:
        return filename

# 记录成功或失败的动作到 .txt 文件，带时间戳或失败原因，便于后期数据分析
def record_success_failure(flag: bool, file_path, str=""):
    with open(file_path, "rb") as file:
        file.seek(0, 2)
        file_empty = file.tell() == 0
        if not file_empty:
            file.seek(-1, 2)
            last_char = file.read(1)
    if file_empty or last_char != b"\n":
        if flag:
            print("write success")
            with open(file_path, "a") as file:
                file.write("1 " + "success" + "\n")
        else:
            print("write failure")
            with open(file_path, "a") as file:
                file.write("0 " + str + "\n")
    else:
        print("No writing")
        return

# # 从 .ply 文件中读取裸点云
# def read_ply(filename):
#     """read XYZ point cloud from filename PLY file"""
#     plydata = PlyData.read(filename)
#     pc = plydata["vertex"].data
#     pc_array = np.array([[x, y, z] for x, y, z in pc])
#     return pc_array

# # 从 .ply 文件中读取带颜色的点云
# def read_ply_with_colors(filename):
#     plydata = PlyData.read(filename)
#     pc = plydata["vertex"].data
#     pc_array = np.array([[x, y, z] for x, y, z, r, g, b in pc])
#     colors = np.array([[r, g, b] for x, y, z, r, g, b in pc])
#     return pc_array, colors


# def write_ply(points, filename):
#     """
#     save 3D-points and colors into ply file.
#     points: [N, 3] (X, Y, Z)
#     colors: [N, 3] (R, G, B)
#     filename: output filename
#     """
#     # combine vertices and colors
#     vertices = np.array(
#         [tuple(point) for point in points],
#         dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
#     )
#     el = PlyElement.describe(vertices, "vertex")
#     # save PLY file
#     PlyData([el], text=True).write(filename)


# def write_ply_with_colors(points, colors, filename):
#     """
#     save 3D-points and colors into ply file.
#     points: [N, 3] (X, Y, Z)
#     colors: [N, 3] (R, G, B)
#     filename: output filename
#     """
#     # combine vertices and colors
#     colors = colors[:, :3]
#     vertices = np.array(
#         [tuple(point) + tuple(color) for point, color in zip(points, colors)],
#         dtype=[
#             ("x", "f4"),
#             ("y", "f4"),
#             ("z", "f4"),
#             ("red", "u1"),
#             ("green", "u1"),
#             ("blue", "u1"),
#         ],
#     )
#     # create PlyElement
#     el = PlyElement.describe(vertices, "vertex")
#     # save PLY file
#     PlyData([el], text=True).write(filename)

# 判断某个抓取动作是否影响了其他衣物的位置
def compare_position_before_and_after(pre_poses, cur_poses, index):
    nums = 0
    for i in range(len(pre_poses)):
        if i == index:
            continue
        dis = torch.norm(cur_poses[i] - pre_poses[i]).item()
        if dis > 0.2:
            nums += 1
    print(f"{nums} garments changed a lot")
    return nums

# 检查是否一次抓取多个衣物
def judge_once_per_time(cur_poses, index):
    nums = 1
    for i in range(len(cur_poses)):
        if i == index:
            continue
        dis = torch.norm(cur_poses[i] - cur_poses[index]).item()
        if dis < 0.4:
            nums += 1
    print(f"pick {nums} of garments once")
    return nums


def wm_judge_final_poses(
    position, index, garment_index, save_path: str = "Env_Eval/washmachine_record.txt"
):
    garment_nums = 0
    catch_garment_x = 0
    catch_garment_z = 0
    other_garment_x = 0
    other_garment_z = 0
    garment_retrieve_x = []
    garment_retrieve_z = []
    for i in range(len(garment_index)):
        if i == index:
            catch_garment_x = position[index][0]
            catch_garment_z = position[index][2]
            garment_retrieve_x.append(catch_garment_x)
            garment_retrieve_z.append(catch_garment_z)
            print("catch_garment_x:", catch_garment_x)
            print("catch_garment_height:", catch_garment_z)
            if catch_garment_x > -0.65 or catch_garment_z > 0.28:
                record_success_failure(False, save_path, "fail to catch out garment")
            else:
                garment_nums += 1

            delete_prim(f"/World/Garment/garment_{index}")
            garment_index[i] = False
        elif garment_index[i]:
            other_garment_x = position[i][0]
            other_garment_z = position[i][2]
            garment_retrieve_x.append(other_garment_x)
            garment_retrieve_z.append(other_garment_z)
            print("other_garment_x:", other_garment_x)
            print("other_garment_height:", other_garment_z)
            if other_garment_z < 0.38:
                garment_nums += 1
                delete_prim(f"/World/Garment/garment_{i}")
                garment_index[i] = False
    if garment_nums > 1:
        if all(x > -0.65 for x in garment_retrieve_x) and all(
            z > 0.28 for z in garment_retrieve_z
        ):
            record_success_failure(
                False,
                save_path,
                "catch more than one garment and fail to catch out garment",
            )
        else:
            record_success_failure(True, save_path)
    else:
        record_success_failure(True, save_path)

    return garment_index

# 判断衣物是否还在沙发区域（通过 YZ 坐标判断），并删除已经“放置成功”的衣物
def sofa_judge_final_poses(
    position, index, garment_index, save_path: str = "Env_Eval/sofa_record.txt"
):
    """
    position: 所有衣物位置
    index：正在抓取的衣物
    garment_index：True list真值判断列表，是否还在沙发上
    """
    for i in range(len(garment_index)):
        if i == index:
            print(f"garment_{i} position: {position[i]}")
            z = position[index][2]
            y = position[index][1]
            if z > 0.3 or y > 1.20:
                record_success_failure(False, save_path, "final pose not correct")
            delete_prim(f"/World/Garment/garment_{index}")
            garment_index[i] = False
        elif garment_index[i]: # 如果还在沙发上的剩余衣服
            print(f"garment_{i} position: {position[i]}")
            if (position[i][2] < 0.35 and position[i][2] > 0.15) or (
                position[i][1] > 1.20 and position[i][1] < 1.55
            ):
                record_success_failure(
                    False,
                    save_path,
                    "other garment final pose not correct",
                )
            if (
                position[i][2] < 0.35
                or position[i][1] < 1.55
                or position[i][0] > 0.45
                or position[i][0] < -0.45
                or position[i][1] > 2.5
            ):
                delete_prim(f"/World/Garment/garment_{i}")
                garment_index[i] = False
    record_success_failure(True, save_path, " success")
    return garment_index


def basket_judge_final_poses(
    position, index, garment_index, save_path: str = "Env_Eval/basket_record.txt"
):
    for i in range(len(garment_index)):
        if i == index:
            print(f"garment_{i} position: {position[i]}")
            z = position[index][2]
            y = position[index][1]
            x = position[index][0]
            if z > 0.35 or y > -1.2 or x < 5.27:
                record_success_failure(False, save_path, "final pose not correct")

            delete_prim(f"/World/Garment/garment_{index}")
            print(f"detele garment_{index}")
            garment_index[i] = False
        elif garment_index[i]:
            print(f"garment_{i} position: {position[i]}")
            z = position[i][2]
            y = position[i][1]
            x = position[i][0]
            if not (
                x > 4.52
                and x < 5.14
                and y > -1.01
                and y < -0.62
                and z > 0.50906
                and z < 0.84742
            ) and (z > 0.35 or y > -1.2 or x < 5.27):
                record_success_failure(
                    False,
                    save_path,
                    "other garment final pose not correct",
                )
            if not (
                x > 4.52
                and x < 5.14
                and y > -1.01
                and y < -0.62
                and z > 0.50906
                and z < 0.84742
            ):
                delete_prim(f"/World/Garment/garment_{i}")
                print(f"detele garment_{i}")
                garment_index[i] = False

    record_success_failure(True, save_path)

    return garment_index

# 运输辅助块
def load_sofa_transport_helper(world: World):
    cube_list = []
    from omni.isaac.core.utils.prims import is_prim_path_valid
    from omni.isaac.core.utils.string import find_unique_string_name
    cube_list.append(
        FixedCuboid(
            name=find_unique_string_name(initial_name="transport_helper_1",
                is_unique_fn=lambda x: not world.scene.object_exists(x),),
            position=[-0.6, 2.18897, 0.70],
            prim_path="/World/transport_helper/transport_helper_1",
            scale=np.array([2.15535, 3.38791, 0.01165]),
            orientation=euler_angles_to_quat([0, 45, 0], degrees=True),
            size=1.0,
            color=np.array([0.94118, 0.90196, 0.54902]),
            visible=False,
        )
    )
    cube_list.append(
        FixedCuboid(
            name=find_unique_string_name(initial_name="transport_helper_2",
                is_unique_fn=lambda x: not world.scene.object_exists(x),),
            position=[0.6, 2.18897, 0.70],
            prim_path="/World/transport_helper/transport_helper_2",
            scale=np.array([2.15535, 3.38791, 0.01165]),
            orientation=euler_angles_to_quat([0, -45, 0], degrees=True),
            size=1.0,
            color=np.array([0.94118, 0.90196, 0.54902]),
            visible=False,
        )
    )
    cube_list.append(
        FixedCuboid(
            name=find_unique_string_name(initial_name="transport_helper_3",
                is_unique_fn=lambda x: not world.scene.object_exists(x),),
            position=[0.0, 1.3475, 0.82],
            prim_path="/World/transport_helper/transport_helper_3",
            scale=np.array([3.1503, 1.29183, 0.01118]),
            orientation=euler_angles_to_quat([-45, 0, 0], degrees=True),
            size=1.0,
            color=np.array([0.94118, 0.90196, 0.54902]),
            visible=False,
        )
    )
    cube_list.append(
        FixedCuboid(
            name=find_unique_string_name(initial_name="transport_helper_4",
                is_unique_fn=lambda x: not world.scene.object_exists(x),),
            position=[0.0, 2.52661, 0.95755],
            prim_path="/World/transport_helper/transport_helper_4",
            scale=np.array([3.1503, 2.0, 0.01118]),
            orientation=euler_angles_to_quat([75, 0, 0], degrees=True),
            size=1.0,
            color=np.array([0.94118, 0.90196, 0.54902]),
            visible=False,
        )
    )
    cube_list.append(
        FixedCuboid(
            name=find_unique_string_name(initial_name="transport_helper_5",
                is_unique_fn=lambda x: not world.scene.object_exists(x),),
            position=[0.0, 1.62, -0.06],
            prim_path="/World/transport_helper/transport_helper_5",
            scale=np.array([2.70, 0.01, 1.0]),
            orientation=euler_angles_to_quat([0, 0, 0], degrees=True),
            size=1.0,
            color=np.array([0.94118, 0.90196, 0.54902]),
            visible=False,
        )
    )
    for cube in cube_list:
        world.scene.add(cube)
    return cube_list


def load_basket_transport_helper(world: World):
    cube_list = []
    from omni.isaac.core.utils.prims import is_prim_path_valid
    from omni.isaac.core.utils.string import find_unique_string_name

    cube_list.append(
        FixedCuboid(
            name=find_unique_string_name(initial_name="transport_helper_1",
                is_unique_fn=lambda x: not world.scene.object_exists(x),),
            position=[3.76192, -0.47743, 1.5777],
            prim_path="/World/transport_helper/transport_helper_1",
            scale=np.array([2.15535, 3.38791, 0.01165]),
            orientation=euler_angles_to_quat([0, 45, 0], degrees=True),
            size=1.0,
            color=np.array([0.94118, 0.90196, 0.54902]),
            visible=False,
        )
    )

    cube_list.append(
        FixedCuboid(
            name=find_unique_string_name(initial_name="transport_helper_2",
                is_unique_fn=lambda x: not world.scene.object_exists(x),),
            position=[5.90925, -0.47743, 1.57852],
            prim_path="/World/transport_helper/transport_helper_2",
            scale=np.array([2.15535, 3.38791, 0.01165]),
            orientation=euler_angles_to_quat([0, -45, 0], degrees=True),
            size=1.0,
            color=np.array([0.94118, 0.90196, 0.54902]),
            visible=False,
        )
    )

    cube_list.append(
        FixedCuboid(
            name=find_unique_string_name(initial_name="transport_helper_3",
                is_unique_fn=lambda x: not world.scene.object_exists(x),),
            position=[4.89684, -1.46591, 1.26201],
            prim_path="/World/transport_helper/transport_helper_3",
            scale=np.array([3.1503, 1.29183, 0.01118]),
            orientation=euler_angles_to_quat([-45, 0, 0], degrees=True),
            size=1.0,
            color=np.array([0.94118, 0.90196, 0.54902]),
            visible=False,
        )
    )

    cube_list.append(
        FixedCuboid(
            name=find_unique_string_name(initial_name="transport_helper_4",
                is_unique_fn=lambda x: not world.scene.object_exists(x),),
            position=[4.89684, 0.04995, 1.52616],
            prim_path="/World/transport_helper/transport_helper_4",
            scale=np.array([3.1503, 2.0, 0.01118]),
            orientation=euler_angles_to_quat([45, 0, 0], degrees=True),
            size=1.0,
            color=np.array([0.94118, 0.90196, 0.54902]),
            visible=False,
        )
    )

    cube_list.append(
        FixedCuboid(
            name=find_unique_string_name(initial_name="transport_helper_5",
                is_unique_fn=lambda x: not world.scene.object_exists(x),),
            position=[0.0, 1.62, -0.06],
            prim_path="/World/transport_helper/transport_helper_5",
            scale=np.array([2.70, 0.01, 1.0]),
            orientation=euler_angles_to_quat([0, 0, 0], degrees=True),
            size=1.0,
            color=np.array([0.94118, 0.90196, 0.54902]),
            visible=False,
        )
    )

    for cube in cube_list:
        world.scene.add(cube)

    return cube_list


def add_wm_door(world):
    world.scene.add(
        FixedCuboid(
            name="wm_door",
            position=[-0.39497, 0.0, 0.275],
            prim_path="/World/wm_door",
            scale=np.array([0.05, 1, 0.55]),
            orientation=euler_angles_to_quat([0, 0, 0], degrees=True),
            size=1.0,
            color=np.array([180, 180, 180]),
            visible=False,
        )
    )


def change_door_pos(obj):
    obj.set_world_pose(position=[-0.39497, 0.0, 0.5])

def delete_wm_door():
    delete_prim("/World/wm_door")

# 执行Furthest Point Sampling（FPS）
def furthest_point_sampling(points, colors=None, semantics=None, n_samples=4096):
    """
    points: [N, 3] tensor containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically &lt;&lt; N
    """
    # Convert points to PyTorch tensor if not already and move to GPU
    # print(colors)
    points = torch.Tensor(points).cuda()  # [N, 3]
    if colors is not None:
        colors = torch.Tensor(colors).cuda()
    if semantics is not None:
        semantics = semantics.astype(np.int32)
        semantics = torch.Tensor(semantics).cuda()
    # Number of points
    num_points = points.size(0)  # N
    # Initialize an array for the sampled indices
    sample_inds = torch.zeros(n_samples, dtype=torch.long).cuda()  # [S]
    # Initialize distances to inf
    dists = torch.ones(num_points).cuda() * float("inf")  # [N]
    # Select the first point randomly
    selected = torch.randint(num_points, (1,), dtype=torch.long).cuda()  # [1]
    sample_inds[0] = selected
    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        last_added = sample_inds[i - 1]  # Scalar
        dist_to_last_added_point = torch.sum(
            (points[last_added] - points) ** 2, dim=-1
        )  # [N]
        # If closer, update distances
        dists = torch.min(dist_to_last_added_point, dists)  # [N]
        # Pick the one that has the largest distance to its nearest neighbor in the sampled set
        selected = torch.argmax(dists)  # Scalar
        sample_inds[i] = selected
    if colors is not None and semantics is not None:
        return (
            points[sample_inds].cpu().numpy(),
            colors[sample_inds].cpu().numpy(),
            semantics[sample_inds].cpu().numpy(),
        )  # [S, 3]
    elif colors is not None:
        return points[sample_inds].cpu().numpy(), colors[sample_inds].cpu().numpy()


def write_rgb_image(rgb_data, filename):
    from PIL import Image

    image = Image.fromarray(rgb_data)
    image.save(filename)
    # cprint(f"write to .png file successful : {filename}", "magenta")
