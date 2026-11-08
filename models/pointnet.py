import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用你 layer.py 中已有的空间变换网络 (T-Net)
from models.layer import STNkd

class PointNetSeg(nn.Module):
    """
    PointNet for 3D Segmentation
    对应 train.py 中的 Method 2
    """
    def __init__(self, num_classes=17, feature_dim=24):
        super(PointNetSeg, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # 1. 输入特征对齐网络 (Input Transform / T-Net)
        # 对原始的 feature_dim 维度 (如24维) 进行空间仿射变换对齐
        self.stn_in = STNkd(k=feature_dim)

        # 2. 局部特征提取 MLP (Shared MLP 1)
        self.conv1 = nn.Conv1d(feature_dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        # 3. 特征空间对齐网络 (Feature Transform / T-Net)
        # 在 64 维的高维特征空间进行对齐
        self.stn_feat = STNkd(k=64)

        # 4. 全局特征提取 MLP (Shared MLP 2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv1d(128, 1024, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        # 5. 分割网络 (Segmentation Network)
        # 拼接全局特征 (1024) 和局部特征 (64) = 1088
        self.conv6 = nn.Conv1d(1088, 512, kernel_size=1)
        self.conv7 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv8 = nn.Conv1d(256, 128, kernel_size=1)
        self.conv9 = nn.Conv1d(128, num_classes, kernel_size=1)

        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(128)

    def forward(self, x, pos=None):
        """
        前向传播
        :param x: 输入点云特征，Shape: (B, N, feature_dim)
        :param pos: 坐标位置，Shape: (B, N, 3)（PointNet内部通过全连接提取，这里兼容传参接口）
        :return: 分割预测，Shape: (B, N, num_classes)
        """
        B, N, C = x.shape

        # --- 1. 输入变换 (Input Transform) ---
        # STNkd 期望输入 shape 为 (B, N, C)
        x = self.stn_in(x) 

        # PyTorch 的 Conv1d 需要通道在第二维: (B, C, N)
        x = x.transpose(2, 1)

        # --- 2. 局部特征提取 ---
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # (B, 64, N)

        # --- 3. 特征对齐变换 (Feature Transform) ---
        x = x.transpose(2, 1) # 转回 (B, N, 64) 喂给 STNkd
        x = self.stn_feat(x)
        
        # 记录局部特征，用于后续逐点拼接
        local_feat = x.transpose(2, 1) # (B, 64, N)
        x = local_feat # 继续往前传

        # --- 4. 全局特征提取 ---
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x))) # (B, 1024, N)

        # Max Pooling 获取全局特征 (Global Feature)
        global_feat = torch.max(x, dim=2, keepdim=True)[0] # (B, 1024, 1)
        
        # 复制全局特征到每一个点上
        global_feat = global_feat.repeat(1, 1, N) # (B, 1024, N)

        # --- 5. 分割网络 (融合 Local & Global) ---
        # 按通道拼接 (B, 1024+64, N) -> (B, 1088, N)
        concat_feat = torch.cat([local_feat, global_feat], dim=1) 

        # 逐点全连接层降维分类
        x = F.relu(self.bn6(self.conv6(concat_feat)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        out = self.conv9(x) # 输出 (B, num_classes, N)

        # 转置回 (B, N, num_classes) 
        # 因为在你的 train.py 中：pred = pred.transpose(2, 1) 将把这个变回 (B, num_classes, N) 送进 CrossEntropyLoss
        out = out.transpose(2, 1)

        return out