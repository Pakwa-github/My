import numpy as np
import torch

###############################################################################
# 环境配置类
###############################################################################
class SceneConfig:
    def __init__(self,
                 room_usd_path="/home/pakwa/GarmentLab/Assets/Scene/FlatGrid.usd",
                 pos: np.ndarray = None,
                 ori: np.ndarray = None,
                 scale: np.ndarray = None):
        self.pos = pos if pos is not None else np.array([0, 0, 0])
        self.ori = ori if ori is not None else np.array([0, 0, 0])
        self.scale = scale if scale is not None else np.array([1, 1, 1])
        self.room_usd_path = room_usd_path
