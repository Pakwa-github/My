# def model_pick_whole_procedure(self):  # ! 
    #     """
    #     Use Aff_model to fetch garment from sofa.
    #     if affordance is not so good, then
    #         Use Pick_model and Place_model to adapt the garment.
    #     """
    #     self.stir = False

    #     while True:
    #         aff_pc, aff_color = self.point_cloud_camera.get_point_cloud_data(
    #             sample_flag=True, sample_num=4096
    #         )
    #         if aff_pc is None:
    #             cprint("Finish picking all garments", "green", on_color="on_green")
    #             simulation_app.close()

    #         if not os.path.exists("Env_Eval/sofa_record.txt"):
    #             with open("Env_Eval/sofa_record.txt", "w") as f:
    #                 f.write("result ")
    #         else:
    #             with open("Env_Eval/sofa_record.txt", "rb") as file:
    #                 file.seek(-1, 2)
    #                 last_char = file.read(1)
    #                 if last_char == b"\n":
    #                     with open("Env_Eval/sofa_record.txt", "a") as f:
    #                         f.write("result ")

    #         aff_ratio = self.point_cloud_camera.get_pc_ratio()
    #         cprint(f"aff_ratio: {aff_ratio}", "cyan")

    #         if aff_ratio >= 0.6 or self.stir:
    #             # _, self.ply_count = self.point_cloud_camera.save_pc(aff_pc, aff_color)
    #             # self.point_cloud_camera.get_rgb_graph()
    #             cprint("affordance is good, begin to fetch garment!", "green")
    #             # _, self.ply_count = self.point_cloud_camera.save_pc(aff_pc, aff_color)
    #             pick_point = self.point_cloud_camera.get_model_point()
    #             # pick_point = self.point_cloud_camera.get_random_point()[0]
    #             cprint(f"pick_point: {pick_point}", "cyan")
    #             self.cur = self.point_cloud_camera.get_cloth_picking()
    #             self.id = self.point_cloud_camera.semantic_id
    #             cprint(f"picking {self.cur}", "cyan")
    #             # define thread to judge contact with ground
    #             judge_thread = threading.Thread(
    #                 target=self.recording_camera.judge_contact_with_ground
    #             )
    #             # begin judge_final_pose thread
    #             judge_thread.start()

    #             # set attach block and pick
    #             self.set_attach_to_garment(attach_position=pick_point)

    #             print("pick target positions: ", self.config.target_positions)

    #             fetch_result = self.franka.fetch_garment_from_sofa(
    #                 self.config.target_positions, self.attach
    #             )
    #             if not fetch_result:
    #                 cprint("fetch current point failed", "red")
    #                 self.recording_camera.stop_judge_contact()
    #                 self.attach.detach()
    #                 self.franka.return_to_initial_position(self.config.initial_position)
    #                 with open("data/Record.txt", "a") as file:
    #                     file.write(f"0 point_unreachable" + "\n")
    #                 continue

    #             # stop judge contact with ground thread
    #             self.recording_camera.stop_judge_contact()

    #             # detach attach block and open franka's gripper
    #             self.attach.detach()
    #             self.franka.open()

    #             for i in range(100):
    #                 self.world.step(
    #                     render=True
    #                 )  # render the world to wait the garment fall down

    #             garment_cur_index = int(self.cur[8:])
    #             print(f"garment_cur_index: {garment_cur_index}")

    #             garment_cur_poses = self.garments.get_cur_poses()

    #             self.garment_index = sofa_judge_final_poses(
    #                 garment_cur_poses, garment_cur_index, self.garment_index
    #             )

    #             for i in range(25):
    #                 self.world.step(
    #                     render=True
    #                 )  # render the world to wait the garment disappear

    #             self.stir = False

    #         else:
    #             cprint(
    #                 "affordance is not so good, begin to adapt the garment!", "green"
    #             )
    #             pick_point = self.point_cloud_camera.get_pick_point(aff_pc)
    #             # pick_point = self.point_cloud_camera.get_random_point()[0]
    #             cprint(f"adaption pick_point: {pick_point}", "cyan")
    #             place_point = self.point_cloud_camera.get_place_point(
    #                 pick_point=pick_point, pc=aff_pc
    #             )

    #             self.set_attach_to_garment(attach_position=pick_point)

    #             target_positions = copy.deepcopy(self.config.target_positions)
    #             target_positions[1] = place_point

    #             print("adaptation target_positions: ", target_positions)

    #             fetch_result = self.franka.sofa_pick_place_procedure(
    #                 target_positions, self.attach
    #             )

    #             if not fetch_result:
    #                 cprint("failed", "red")

    #             self.attach.detach()
    #             self.franka.open()
    #             self.franka.return_to_initial_position(self.config.initial_position)

    #             self.stir = True


# def get_demo(self, assign_point, debug=False, log=False):
    #     """获取演示数据"""
    #     # self.reset()
    #     self.control.robot_reset()
    #     for _ in range(20):
    #         self.world.step()
    #     if debug:
    #         self.wait()
        
    #     point = np.array(assign_point)
    #     start_data = self.get_cloth_in_world_pose()
    #     if log:
    #         np.savetxt("start_data.txt", start_data)
            
    #     # 计算最近点的索引
    #     dist = np.linalg.norm(start_data - point[None,:], axis=-1)
    #     self.sel_particle_index = np.argmin(dist, axis=0)
        
    #     print("grasping..?")

    #     # 执行抓取
    #     self.control.grasp(pos=[point],ori=[None],flag=[True])
        
    #     # 执行放置序列
    #     for idx in range(self.target_point.shape[0]):
    #         print(self.target_point.shape[0])
    #         print("放置点：", self.target_point[idx])
    #         self.control.move(pos=[self.target_point[idx]],ori=[None],flag=[True])

    #     for _ in range(150):
    #         self.world.step()

    #     finish_data = self.get_cloth_in_world_pose()
    #     if log:
    #         np.savetxt("finish_data.txt", finish_data)

    #     self.control.ungrasp([False])
    #     for _ in range(10):
    #         self.world.step()
    #     self.succ_data = self.get_cloth_in_world_pose()

    # # 根据标准模型数据（从标准文件中读取）和当前衣物状态计算任务是否成功。
    # def get_reward(self, standard_model, garment_id = 0):
    #     succ_example = np.loadtxt(standard_model)
        
    #     # 归一化处理pts
    #     pts = self.garments[garment_id].get_vertices_positions()
    #     centroid = np.mean(pts, axis=0)
    #     pts = pts - centroid
    #     max_scale = np.max(np.abs(pts))
    #     pts = pts / max_scale

    #     dist = np.linalg.norm(pts - succ_example, ord=2)
    #     if dist < 0.1:
    #         print("################ succ ################")
    #         return True
    #     else:
    #         print("################ fail ################")
    #         return False
        
    # def get_cloth_in_world_pose(self):
    #     """
    #     获取衣物在世界坐标下的顶点位置。
    #     首先获取衣物的顶点数据（get_vertices_positions），
    #     然后结合衣物的缩放、旋转与平移（通过 garment_mesh 的 get_world_pose），
    #     对点云进行变换，返回最终位置数据。
    #     """
    #     particle_positions = self.garments[0].get_vertices_positions()
    #     position, orientation = self.garments[0].garment_mesh.get_world_pose()
    #     if True:
    #         # particle_positions = particle_positions + self.pose

    #         # print("particle_positions type:", type(particle_positions))
    #         # print("scale type:", type(self.garments[0].garment_config.scale))
    #         # print("scale value:", self.garments[0].garment_config.scale)
    #         particle_positions = particle_positions * self.garments[0].garment_config.scale
    #         particle_positions = self.rotate_point_cloud(particle_positions, orientation)
    #         particle_positions = particle_positions + position
    #         # 
    #     return particle_positions



# def exploration(self):
    #     """
    #     功能：进行多次试验（例如 800 次），在每次试验中：

    #     1、重置环境、分配抓取点（allocate_point），抓取、移动衣物；

    #     2、经过一定时间后采集最终衣物状态数据，并保存为 npy 文件；

    #     3、计算与成功状态（succ_data）的误差，利用该误差计算奖励；

    #     4、释放抓取；

    #     5、最后返回一个最终奖励。
    #     """
    #     for i in range(800):
    #         self.reset()
    #         self.control.robot_reset()
    #         for _ in range(20):
    #             self.world.step()
    #         point=self.allocate_point(i, save_path=self.trial_path)
    #         self.control.grasp(pos=[point],ori=[None],flag=[True], wo_gripper=True)
    #         self.control.move(pos=[self.target_point],ori=[None],flag=[True])
    #         # self.control.move(pos=[self.target_point+np.array([0.1,0,0])],ori=[None],flag=[True])
    #         for _ in range(100):
    #             self.world.step()
            
    #         final_data=self.garments[0].get_vertices_positions()
    #         final_path=os.path.join(self.root_path,f"final_pts_{i}.npy")
    #         error = np.linalg.norm(self.succ_data- final_data, ord = 2)/final_data.shape[0]
    #         reward = -error
    #         np.save(final_path,final_data)

    #         self.control.ungrasp([False])
    #         for _ in range(10):
    #             self.world.step()
    #         self.world.stop()
    #     return reward


    # # 为试验分配一个抓取点。
    # def allocate_point(self, index, save_path):
    #     """
    #     如果 index 为 0，则：
    #     获取衣物顶点数据，并根据衣物的缩放、旋转和平移处理，得到一个选点池；
    #     使用随机采样（torch.randperm）从选点池中选取 800 个点，并保存到文件；

    #     根据传入的 index 返回选点池中对应的抓取点。
    #     """
    #     if index==0:
    #         self.selected_pool=self.garments[0].get_vertices_positions()*self.garments[0].garment_config.scale
    #         q = euler_angles_to_quat(self.garments[0].garment_config.ori)
    #         self.selected_pool=self.Rotation(q,self.selected_pool)
    #         centroid, _ = self.garments[0].garment_mesh.get_world_pose()
    #         self.selected_pool=self.selected_pool + centroid
    #         np.savetxt("/home/pakwa/GarmentLab/select.txt",self.selected_pool)

    #         indices=torch.randperm(self.selected_pool.shape[0])[:800]
    #         self.selected_pool=self.selected_pool[indices]
    #         np.save(save_path, self.selected_pool)
    #     point=self.selected_pool[index]
    #     return point


    # # 获取衣物所有顶点数据，
    # # 经过缩放、旋转和平移后，
    # # 再对点云进行下采样（使用 FPS 采样至 256 个点），
    # # 返回下采样后的点云数据。
    # def get_all_points(self):
    #     self.selected_pool=self.garments[0].get_vertices_positions()*self.garments[0].garment_config.scale
    #     q = euler_angles_to_quat(self.garments[0].garment_config.ori)
    #     self.selected_pool=self.Rotation(q,self.selected_pool)
    #     centroid, _ = self.garments[0].garment_mesh.get_world_pose()
    #     # self.selected_pool=self.selected_pool + centroid
    #     if self.selected_pool is None:
    #         self.selected_pool = np.array(centroid).reshape(1, -1)
    #     else:
    #     # 保证 centroid 也是一个二维数组（单个点时 reshape 成 (1, dim)）
    #         centroid = np.array(centroid).reshape(1, -1)
    #         self.selected_pool = np.concatenate((self.selected_pool, centroid), axis=0)
    #     pcd, _, _ =self.fps_np(self.selected_pool, 256)

    #     return pcd



# # 以下录制与回放接口留待扩展
    # def record(self):
    #     if not self.recording:
    #         self.recording = True
    #         self.replay_file_name = strftime("Assets/Replays/%Y%m%d-%H:%M:%S.npy", gmtime())
    #         self.context.add_physics_callback("record_callback", self.record_callback)

    # def stop_record(self):
    #     if self.recording:
    #         self.recording = False
    #         self.context.remove_physics_callback("record_callback")
    #         np.save(self.replay_file_name, np.array(self.savings))
    #         self.savings = []

    # def record_callback(self, step_size):
    #     pass

    # def replay_callback(self, data):
    #     pass

    # def __replay_callback(self, step_size):
    #     if self.time_ptr < self.total_ticks:
    #         self.replay_callback(self.data[self.time_ptr])
    #         self.time_ptr += 1

    # def replay(self, file):
    #     self.data = np.load(file, allow_pickle=True)
    #     self.time_ptr = 0
    #     self.total_ticks = len(self.data)
    #     self.context.add_physics_callback("replay_callback", self.__replay_callback)
    #     if self.deformable:
    #         # 设置 GPU 处理软体（可变形物体）时允许的最大接触点数量，防止由于过多接触点导致内存或性能问题
    #         # 原作注释了此条
    #         self.physics.set_gpu_max_soft_body_contacts(1024000)
    #         # GPU 碰撞堆栈（collision stack）是物理引擎内部用于存储和处理所有碰撞检测结果的数据结构。
    #         # 在 GPU 加速的物理计算中，碰撞堆栈会存储所有检测到的碰撞接触数据，并在后续步骤中对这些数据进行处理，
    #         # 比如碰撞响应、摩擦计算等。
    #         self.physics.set_gpu_collision_stack_size(3000000)






    # def pick_whole_procedure(self):
    #     while True:
    #         pc_judge, color_judge = self.point_cloud_camera.get_point_cloud_data()
    #         # print(pc_judge)
    #         if pc_judge is None:
    #             cprint("Finish picking all garments", "green", on_color="on_green")
    #             break

    #         # save pc and rgb graph
    #         _, self.ply_count = self.point_cloud_camera.save_point_cloud(
    #             sample_flag=True, sample_num=4096
    #         )
    #         # self.point_cloud_camera.get_rgb_graph()
            
    #         # get pick point
    #         pick_point = self.point_cloud_camera.get_model_point()
    #         cprint(f"pick_point: {pick_point}", "cyan")

    #         self.cur = self.point_cloud_camera.get_cloth_picking()
    #         self.id = self.point_cloud_camera.semantic_id
    #         cprint(f"picking {self.cur}", "cyan")

    #         # with open("data/Record.txt", "a") as file:
    #         #     file.write(
    #         #         f"{pick_point[0]} {pick_point[1]} {pick_point[2]} {self.ply_count} "
    #         #     )

    #         # define thread to judge contact with ground
    #         judge_thread = threading.Thread(
    #             target=self.recording_camera.judge_contact_with_ground
    #         )
    #         judge_thread.start()

    #         # set attach block and pick
    #         self.set_attach_to_garment(attach_position=pick_point)

    #         fetch_result = self.franka.fetch_garment_from_sofa(
    #             self.config.target_positions, self.attach
    #         )
    #         if not fetch_result:
    #             cprint("fetch current point failed", "red")
    #             self.recording_camera.stop_judge_contact()
    #             self.attach.detach()
    #             self.franka.return_to_initial_position(self.config.initial_position)
    #             continue

    #         self.recording_camera.stop_judge_contact()
    #         self.attach.detach()
    #         self.franka.open()

    #         for i in range(100):
    #             self.world.step(render=True)

    #         garment_cur_index = int(self.cur[8:])
    #         print(f"garment_cur_index: {garment_cur_index}")

    #         garment_cur_poses = self.wrapgarment.get_cur_poses()

    #         self.garment_index = sofa_judge_final_poses(
    #             garment_cur_poses, garment_cur_index, self.garment_index
    #         )

    #         for i in range(25):
    #             self.world.step(render=True)

    #         # return to inital position
    #         # self.franka.return_to_initial_position(self.config.initial_position)

    # def random_pick_place(self):
    #     pick_pc, pick_color = self.point_cloud_camera.get_point_cloud_data(
    #         sample_flag=True, sample_num=4096
    #     )
    #     self.point_cloud_camera.get_rgb_graph()
    #     pick_point = self.point_cloud_camera.get_random_point()[0]
    #     place_point = self.point_cloud_camera.get_random_point()[0]
    #     distance = np.linalg.norm(pick_point - place_point)
    #     print(f"distance: {distance}")
    #     while distance < 0.2:
    #         _, self.ply_count = self.point_cloud_camera.save_pc(pick_pc, pick_color)
    #         with open("data/Record.txt", "a") as file:
    #             file.write(
    #                 f"{pick_point[0]} {pick_point[1]} {pick_point[2]} {place_point[0]} {place_point[1]} {place_point[2]} {self.ply_count} 0 distance_between_points_is_too_close"
    #                 + "\n"
    #             )
    #         pick_pc, pick_color = self.point_cloud_camera.get_point_cloud_data(
    #             sample_flag=True, sample_num=4096
    #         )
    #         pick_point = self.point_cloud_camera.get_random_point()[0]
    #         place_point = self.point_cloud_camera.get_random_point()[0]
    #         distance = np.linalg.norm(pick_point - place_point)
    #         print(f"distance: {distance}")

    #     cprint(f"pick_point: {pick_point}", "cyan")
    #     cprint(f"place_point: {place_point}", "cyan")

    #     pick_ratio = self.point_cloud_camera.get_pc_ratio()
    #     cprint(f"pick_ratio: {pick_ratio}", "cyan")

    #     self.set_attach_to_garment(attach_position=pick_point)

    #     self.config.target_positions[1] = place_point

    #     fetch_result = self.franka.sofa_pick_place_procedure(
    #         self.config.target_positions, self.attach
    #     )

    #     if not fetch_result:
    #         cprint("fetch current point failed", "red")
    #         _, self.ply_count = self.point_cloud_camera.save_pc(pick_pc, pick_color)
    #         with open("data/Record.txt", "a") as file:
    #             file.write(
    #                 f"{pick_point[0]} {pick_point[1]} {pick_point[2]} {place_point[0]} {place_point[1]} {place_point[2]} {self.ply_count} 0 point_unreachable"
    #                 + "\n"
    #             )
    #         # self.attach.detach()
    #         # self.franka.return_to_initial_position(self.config.initial_position)
    #         return

    #     self.attach.detach()
    #     self.franka.open()
    #     self.franka.return_to_initial_position(self.config.initial_position)

    #     place_pc, place_color = self.point_cloud_camera.get_point_cloud_data(
    #         sample_flag=True, sample_num=4096
    #     )
    #     self.point_cloud_camera.get_rgb_graph()
    #     place_ratio = self.point_cloud_camera.get_pc_ratio()
    #     cprint(f"place_ratio: {place_ratio}", "cyan")

    #     if pick_ratio <= place_ratio:
    #         _, self.ply_count = self.point_cloud_camera.save_pc(pick_pc, pick_color)
    #         with open("data/Record.txt", "a") as file:
    #             file.write(
    #                 f"{pick_point[0]} {pick_point[1]} {pick_point[2]} {place_point[0]} {place_point[1]} {place_point[2]} {self.ply_count} 1 pick_ratio_{pick_ratio}<=place_ratio_{place_ratio}"
    #                 + "\n"
    #             )
    #         _, self.ply_count = self.point_cloud_camera.save_pc(place_pc, place_color)
    #         with open("data/Record.txt", "a") as file:
    #             file.write(
    #                 f"{place_point[0]} {place_point[1]} {place_point[2]} {pick_point[0]} {pick_point[1]} {pick_point[2]} {self.ply_count} 0"
    #                 + "\n"
    #             )
    #     else:
    #         _, self.ply_count = self.point_cloud_camera.save_pc(pick_pc, pick_color)
    #         with open("data/Record.txt", "a") as file:
    #             file.write(
    #                 f"{pick_point[0]} {pick_point[1]} {pick_point[2]} {place_point[0]} {place_point[1]} {place_point[2]} {self.ply_count} 0"
    #                 + "\n"
    #             )
    #         _, self.ply_count = self.point_cloud_camera.save_pc(place_pc, place_color)
    #         with open("data/Record.txt", "a") as file:
    #             file.write(
    #                 f"{place_point[0]} {place_point[1]} {place_point[2]} {pick_point[0]} {pick_point[1]} {pick_point[2]} {self.ply_count} 1 pick_ratio_{pick_ratio}>place_ratio_{place_ratio}"
    #                 + "\n"
    #             )

    # def random_pick_model_place(self):
    #     pick_pc, pick_color = self.point_cloud_camera.get_point_cloud_data(
    #         sample_flag=True, sample_num=4096
    #     )
    #     # self.point_cloud_camera.get_rgb_graph()
    #     pick_ratio = self.point_cloud_camera.get_pc_ratio()
    #     cprint(f"pick_ratio: {pick_ratio}", "cyan")
    #     if pick_ratio > 0.6:
    #         return
    #     self.point_cloud_camera.get_rgb_graph()
    #     pick_point = self.point_cloud_camera.get_random_point()[0]
    #     place_point = self.point_cloud_camera.get_place_point(
    #         pick_point=pick_point, pc=pick_pc
    #     )

    #     print(f"pick_point: {pick_point}")
    #     print(f"place_point: {place_point}")

    #     self.set_attach_to_garment(attach_position=pick_point)

    #     self.config.target_positions[1] = place_point

    #     fetch_result = self.franka.sofa_pick_place_procedure(
    #         self.config.target_positions, self.attach
    #     )

    #     if not fetch_result:
    #         cprint("fetch current point failed", "red")
    #         _, self.ply_count = self.point_cloud_camera.save_pc(pick_pc, pick_color)
    #         with open("data/Record.txt", "a") as file:
    #             file.write(
    #                 f"{pick_point[0]} {pick_point[1]} {pick_point[2]} {self.ply_count} 0 point_unreachable"
    #                 + "\n"
    #             )
    #         # self.attach.detach()
    #         # self.franka.return_to_initial_position(self.config.initial_position)
    #         return

    #     self.attach.detach()
    #     self.franka.open()
    #     self.franka.return_to_initial_position(self.config.initial_position)

    #     place_pc, place_color = self.point_cloud_camera.get_point_cloud_data(
    #         sample_flag=True, sample_num=4096
    #     )
    #     self.point_cloud_camera.get_rgb_graph()
    #     place_ratio = self.point_cloud_camera.get_pc_ratio()
    #     cprint(f"place_ratio: {place_ratio}", "cyan")

    #     if place_ratio - pick_ratio > 0.1 or place_ratio > 0.6:
    #         _, self.ply_count = self.point_cloud_camera.save_pc(pick_pc, pick_color)
    #         with open("data/Record.txt", "a") as file:
    #             file.write(
    #                 f"{pick_point[0]} {pick_point[1]} {pick_point[2]} {self.ply_count} 1 pick_ratio_{pick_ratio}<=place_ratio_{place_ratio}"
    #                 + "\n"
    #             )
    #     else:
    #         _, self.ply_count = self.point_cloud_camera.save_pc(pick_pc, pick_color)
    #         with open("data/Record.txt", "a") as file:
    #             file.write(
    #                 f"{pick_point[0]} {pick_point[1]} {pick_point[2]} {self.ply_count} 0 pick_ratio_{pick_ratio}>place_ratio_{place_ratio}"
    #                 + "\n"
    #             )


        # def get_sofa_height(self):
    #     bbox = self.sofa.get_bounding_box()
    #     extent = bbox.GetRange()
    #     if extent.IsEmpty():
    #         return None
    #     min_vec = extent.GetMin()
    #     max_vec = extent.GetMax()
    #     return float(max_vec[2])

# def select_target_garment(self, grasp_point):
    #     """
    #     根据抓取点在世界坐标系下的坐标，选择离抓取点最近的衣物对象。
    #     这里我们假设每个 garment 都有一个 get_world_positions() 方法，
    #     或者你可以直接计算其所有粒子在世界坐标系下的中心点（质心）。
    #     """
    #     min_dist = 0.01
    #     target_garment = None
    #     target_idx = -1
    #     for idx, garment in enumerate(self.garments[:self.num_garments]):
    #         # 如果 garment 提供 get_world_positions()，使用它计算衣物的质心：
    #         world_points = garment.get_world_position()
    #         if world_points.size == 0:
    #             continue
    #         distances = np.linalg.norm(world_points - grasp_point, axis=1)
    #         local_min = np.min(distances)
    #         if local_min < min_dist:
    #             min_dist = local_min
    #             target_garment = garment
    #             target_idx = idx
    #     return target_idx, target_garment