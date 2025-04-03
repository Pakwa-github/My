import numpy as np

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.franka import Franka 
from omni.isaac.franka.controllers import PickPlaceController 
from omni.isaac.core.utils.prims import get_prim_at_path

world = World()
world.scene.add_default_ground_plane()

cube1 = world.scene.add(
    DynamicCuboid(
        prim_path="/World/cube1",
        name="cube1",
        position=np.array([0.0, 0.5, 2.5]),
        scale=np.array([0.05015, 0.05015, 0.05015]),
        color=np.array([0.0, 1.0, 1.0]),
        ))



franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka")) 


controller = PickPlaceController( 
            name="pick_place_controller", 
            gripper=franka.gripper, 
            robot_articulation=franka, 
        ) 

world.reset() 

franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)

cube1_prim = get_prim_at_path("/World/cube1")
print(cube1.prim_path) 
print(cube1_prim.GetPath()) 
print(cube1_prim.GetAppliedSchemas())

while simulation_app.is_running():
    position, orientation = cube1.get_world_pose() 
    goal_position = np.array([-0.3, -0.3, 0.02575]) 
    current_joint_positions = franka.get_joint_positions() 
    actions = controller.forward( 
        picking_position=position, 
        placing_position=goal_position, 
        current_joint_positions=current_joint_positions, 
    ) 
    franka.apply_action(actions) 
    world.step(render=True)

simulation_app.close()