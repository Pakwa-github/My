import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

# 自定义视觉编码器和语义编码器
# 标准 SB3 PPO 使用的是一个简单的（通常是 MLP 或 CNN）特征提取器

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            #self.mlp_bns.append(nn.BatchNorm1d(out_channel,track_running_stats=True))
            self.mlp_bns.append(nn.LayerNorm(256))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):

        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            points2 = points2.permute(0, 2, 1)
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            x=conv(new_points)
            #x=bn(x)
            new_points = F.relu(x)
        return new_points

# 这个语义编码器看上去主要用于对点云数据做语义理解，
# 也就是说，它可以尝试判别每个点属于哪个类别或者分割出不同区域的信息。
# 它也采用了 PointNet 的思想，不过通常会有一个 Softmax 层输出分类概率（或语义标签）。
# 在代码中，sem_model 的构造参数中有 num_classes（类别数），
# 表明它的设计初衷是做分类、语义分割相关任务。

class sem_model(nn.Module):
    def __init__(self, num_classes, init_ft = None ,init_pts = 256):
        super(sem_model, self).__init__()
        self.dtype = torch.float32
        input_channel = 3 + init_ft if init_ft is not None else 3
        if init_ft is not None:
            self.init_ft = True
        else:
            self.init_ft = False
        self.sa1 = PointNetSetAbstraction(init_pts, 0.025, 16, input_channel, [64, 64, 128], False)
        self.sa2 = PointNetSetAbstraction(64, 0.05, 16, 128 + 3, [128, 128, 256], False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 256, 512], group_all=True)
        self.fp2 = PointNetFeaturePropagation(640, [256, 128])
        self.conv1 = nn.Conv1d(128, 64, 1)
        self.conv2 = nn.Conv1d(64, num_classes, 1)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, xyz):
        xyz = xyz.to(self.dtype)
        xyz = xyz[:, :, :3]
        B, N, D = xyz.shape
        # if self.normal_channel:
        #     norm = xyz[:, :, 3:].reshape(B,-1,N)
            
        # else:
        #     norm = None

        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l0_points = self.fp2(l1_xyz, l3_xyz, l1_points, l3_points)

        x = F.relu(self.conv1(l0_points))
        x = self.conv2(x).view(B, -1)
        x = self.soft_max(x)
        return x.view(B, -1)


# 这个编码器主要负责将输入的原始点云数据转换成一个较小、更加精炼的特征向量
# 它使用了基于 PointNet 架构的思想：先用几个“set abstraction”
# （采样和分组）层从原始点云中提取局部特征；
# 然后将这些局部特征聚合成全局特征（后面的全连接层），生成尺寸固定的特征向量。
# 在你的代码中，encoder 是在 forward() 函数中被调用，
# 直接处理传入的观察数据，输出特征给策略网络后续的 MLP 层使用。

class encoder(nn.Module):
    def __init__(self,num_class,normal_channel=False,pts_list=[512,256,64]):
        super(encoder, self).__init__()
        self.dtype = torch.float32                  # 定义数据类型
        in_channel = 4 if normal_channel else 3     # 如果带法线信息就是 4 维，否则就是 3 维
        self.normal_channel = normal_channel
        # Set Abstraction 1：采样512个点，每个点查找半径0.02内16个邻居，特征维度从 in_channel 到 128
        self.sa1 = PointNetSetAbstraction(npoint=pts_list[0], radius=0.02, nsample=16, in_channel=in_channel, mlp=[64, 64, 128], group_all=False).to(self.dtype)
        # Set Abstraction 2：采样256个点，每个点查找半径0.05内32个邻居，特征维度从128+3到256
        self.sa2 = PointNetSetAbstraction(npoint=pts_list[1], radius=0.05, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False).to(self.dtype)
        # Set Abstraction 4：全局聚合所有点，特征维度从256+3到1024
        self.sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True).to(self.dtype)
        # 全连接层：1024 → 512 → 256 → num_class
        self.fc1 = nn.Linear(1024, 512).to(self.dtype)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256).to(self.dtype)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class).to(self.dtype)
        

    def forward(self, xyz):
        xyz = xyz.to(self.dtype)
        B, N, D = xyz.shape     # batch 大小, 点数, 维度
        if self.normal_channel:
            norm = xyz[:, :, 3:].reshape(B,-1,N)
            xyz = xyz[:, :, :3]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa4(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)

        return x

class PointNetSetAbstraction(nn.Module):
    """
    它实现了 PointNet++ 中的Set Abstraction Layer（集合抽象层）
    核心作用：
        对点云采样
        邻域分组
        用 MLP 提取局部特征
        聚合特征
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, batch_norm = False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if batch_norm:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel,track_running_stats=True))
            else:
                self.mlp_bns.append(nn.Identity())
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        if points is not None:
            points = points.permute(0, 2, 1)    # 如果有特征，把 (B, D, N) → (B, N, D)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 2, 1)     # 变成 (B, C+D, nsample, npoint)
        for i, conv in enumerate(self.mlp_convs): 
            bn = self.mlp_bns[i]
            x=conv(new_points)
            new_points =  F.relu(bn(x))

        new_points = torch.max(new_points, 2)[0]
        #new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points
    
def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn

# 继承 BaseFeaturesExtractor
class PointNetFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=1024):
        # 必须要 super 调用，告知最终输出特征维度
        super().__init__(observation_space, features_dim)
        
        # 这里直接用你原来的 encoder
        from RL.VisionEncoder import encoder
        self.encoder = encoder(features_dim).to(th.float32)
    
    def forward(self, observations):
        return self.encoder(observations)
