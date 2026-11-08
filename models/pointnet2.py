import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用你 layer.py 中已经定义的 fps (最远点采样)
from models.layer import fps 

# =========================================================================
# PointNet++ 基础张量计算算子 (原生 PyTorch 实现，避免 CUDA 编译报错)
# =========================================================================
def square_distance(src, dst):
    """
    计算两组点云之间的欧氏距离平方
    src: [B, N, C]
    dst: [B, M, C]
    返回: [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    
    # [修复] 强制下限为 0，防止 fp16 精度误差导致极小负数距离，引发后续 NaN
    dist = torch.clamp(dist, min=0.0)
    
    return dist

def index_points(points, idx):
    """
    根据索引提取点云特征
    points: [B, N, C]
    idx: [B, S] 或 [B, S, K]
    返回: [B, S, C] 或 [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    球查询 (Ball Query)
    xyz: 所有点坐标 [B, N, 3]
    new_xyz: 中心点坐标 [B, S, 3]
    返回: 局部区域内的点索引 [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    
    # 距离大于半径的点，索引设为N（后续会被替换）
    group_idx[sqrdists > radius ** 2] = N
    # 截取前 nsample 个点
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    
    # 如果球域内点不够 nsample，用第一个点来填充空白
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

# =========================================================================
# PointNet++ 核心模块: Set Abstraction (SA) & Feature Propagation (FP)
# =========================================================================
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        # 点的坐标 (3维) 会拼接到特征中一起送入 MLP
        last_channel = in_channel + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        xyz: 坐标 [B, N, 3]
        points: 特征 [B, N, C]
        """
        B, N, C = xyz.shape
        if self.group_all:
            # 最后一层，把所有点作为一个 group
            new_xyz = torch.zeros(B, 1, 3).to(xyz.device)
            grouped_xyz = xyz.view(B, 1, N, 3)
            if points is not None:
                new_points = points.view(B, 1, N, -1)
                grouped_points = torch.cat([new_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz
        else:
            npoint = min(self.npoint, N)
            # 使用最远点采样 (FPS) 提取中心点
            fps_idx = fps(xyz, npoint).long()
            new_xyz = index_points(xyz, fps_idx)
            
            # 使用球查询 (Ball Query) 划分局部簇
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)
            grouped_xyz -= new_xyz.view(B, npoint, 1, 3) # 转化为相对中心点坐标
            
            if points is not None:
                grouped_points = index_points(points, idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

        # [B, npoint, nsample, C+3] -> [B, C+3, nsample, npoint] (符合Conv2d输入规范)
        grouped_points = grouped_points.permute(0, 3, 2, 1).contiguous()

        # 局部特征提取 MLP
        for i, conv in enumerate(self.mlp_convs):
            grouped_points = F.relu(self.mlp_bns[i](conv(grouped_points)))

        # 最大池化获取局部全局特征
        new_points = torch.max(grouped_points, 2)[0] # [B, C', npoint]
        new_points = new_points.transpose(2, 1).contiguous() # 转回 [B, npoint, C']
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        xyz1: [B, N, 3] (密集层坐标)
        xyz2: [B, M, 3] (稀疏层坐标)
        points1: [B, N, C1] (密集层特征)
        points2: [B, M, C2] (稀疏层特征)
        """
        B, N, _ = xyz1.shape
        B, M, _ = xyz2.shape

        # 若稀疏层只有一个点，直接全局广播
        if M == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # 利用 KNN 寻找 3 个最近邻点，并根据距离的倒数进行加权插值
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  

            # [修复] 将 1e-8 改为 1e-5，避免 float16 (AMP) 精度下下溢出为 0，引发 NaN
            dist_recip = 1.0 / (dists + 1e-5)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        # Skip Link 连接高低层特征
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.transpose(2, 1).contiguous() # [B, C1+C2, N]
        
        # 融合特征 MLP
        for i, conv in enumerate(self.mlp_convs):
            new_points = F.relu(self.mlp_bns[i](conv(new_points)))

        new_points = new_points.transpose(2, 1).contiguous() # [B, N, C']
        return new_points


# =========================================================================
# PointNet++ 主网络 (包装以适配 LitToothSegmentationWrapper)
# =========================================================================
class PointNet2Seg(nn.Module):
    """
    PointNet++ 用于 3D 牙齿分割
    对应 train.py 中的 Method 3
    """
    def __init__(self, num_classes=17, feature_dim=24):
        super(PointNet2Seg, self).__init__()
        self.feature_dim = feature_dim

        # ----------------------------------------------------
        # 下采样路径 (Encoder: Set Abstraction)
        # 层级: N -> 1024 -> 256 -> 64 -> 16
        # ----------------------------------------------------
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, feature_dim, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256, [256, 256, 512], False)

        # ----------------------------------------------------
        # 上采样路径 (Decoder: Feature Propagation)
        # 层级: 16 -> 64 -> 256 -> 1024 -> N
        # ----------------------------------------------------
        self.fp4 = PointNetFeaturePropagation(512 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(256 + 128, [256, 256])
        self.fp2 = PointNetFeaturePropagation(256 + 64, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128 + feature_dim, [128, 128, 128])

        # ----------------------------------------------------
        # 最终逐点分类器 (Classifier)
        # ----------------------------------------------------
        self.conv1 = nn.Conv1d(128, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, x, pos):
        """
        前向传播
        :param x: 节点特征 [B, N, feature_dim]
        :param pos: 节点坐标 [B, N, 3]
        :return: 逐点分类结果 [B, N, num_classes] (配合 LitWrapper 的转置)
        """
        l0_xyz = pos
        l0_points = x

        # ==== Encoder ====
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # ==== Decoder ====
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        # ==== FC 分类 ====
        x_out = l0_points.transpose(2, 1) # 换成 [B, C, N] 送进全连接
        x_out = F.relu(self.bn1(self.conv1(x_out)))
        x_out = self.drop1(x_out)
        out = self.conv2(x_out)           # 最终输出 [B, num_classes, N]

        # 为了无缝配合你 train.py 里的 pred.transpose(2, 1) 计算损失，在此预先转置
        out = out.transpose(2, 1)         # 输出 [B, N, num_classes]

        return out