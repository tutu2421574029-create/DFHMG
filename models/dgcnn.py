import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# 动态 KNN 构图与边特征提取 (强化版：防 FP16 下溢/溢出导致 NaN)
# =========================================================================
def knn(x, k):
    """
    计算 K 近邻。
    在 DGCNN 中，距离计算会在特征空间进行，导致数值方差极大。
    这里强制在单精度 (FP32) 下计算距离，彻底屏蔽 AMP 导致的平方溢出。
    """
    # [核心修复]: 关闭 AMP，强制 FP32 计算，规避 float16 乘方导致的 Inf/NaN
    with torch.cuda.amp.autocast(enabled=False):
        x_f = x.float()
        x_t = x_f.transpose(2, 1).contiguous() # (B, N, C)
        dist = torch.cdist(x_t, x_t, p=2.0)
        
        # 强制下限为0，防止浮点误差导致负距离
        dist = torch.clamp(dist, min=0.0)
        
    idx = dist.topk(k=k, dim=-1, largest=False)[1] # (B, N, K)
    return idx

def get_graph_feature(x, k, idx=None):
    """
    提取边特征 (Edge Feature)。
    根据 DGCNN 论文，EdgeFeature = (x_j - x_i) ⊕ x_i
    """
    B, C, N = x.size()
    
    # 动态建图：如果不提供现成的 idx，则基于当前特征 x 重新找邻居
    if idx is None:
        idx = knn(x, k=k)
        
    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx_flat = (idx + idx_base).view(-1)
    
    x_t = x.transpose(2, 1).contiguous()
    
    # 提取邻居节点特征: [B, N, K, C]
    x_feat = x_t.view(B * N, C)[idx_flat, :].view(B, N, k, C)
    
    # 扩展中心节点特征: [B, N, K, C]
    x_repeat = x_t.view(B, N, 1, C).repeat(1, 1, k, 1)
    
    # 拼接局部差异特征 (x_j - x_i) 与全局参考特征 (x_i)
    feature = torch.cat((x_feat - x_repeat, x_repeat), dim=-1) # (B, N, K, 2C)
    
    # 转置为 Conv2d 期望的格式: (B, 2C, N, K)
    feature = feature.permute(0, 3, 1, 2).contiguous()
    
    return feature, idx

# =========================================================================
# 边缘图卷积模块 (EdgeConv Block)
# =========================================================================
class EdgeConv(nn.Module):
    """
    DGCNN 的核心算子：
    在局部图上通过多层感知机 (1x1 Conv) 提取特征，然后使用 Max Pooling 聚合邻居信息。
    """
    def __init__(self, in_channels, out_channels, k=20):
        super(EdgeConv, self).__init__()
        self.k = k
        self.conv = nn.Sequential(
            # 输入通道需要乘以2，因为 get_graph_feature 拼接了 (x_j - x_i) 和 x_i
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x, idx=None):
        """
        :param x: 节点特征 [B, C, N]
        :param idx: 指定的 KNN 邻接索引。若为 None，则动态基于 x 计算。
        """
        feature, idx = get_graph_feature(x, self.k, idx)
        out = self.conv(feature)
        out = out.max(dim=-1, keepdim=False)[0] # (B, out_channels, N)
        return out, idx

# =========================================================================
# DGCNN 主网络架构 (Dynamic Graph CNN for Segmentation)
# =========================================================================
class DGCNNSeg(nn.Module):
    """
    DGCNN 用于 3D 密集点分割
    对应 train.py 中的 Method 8
    """
    def __init__(self, num_classes=17, feature_dim=24, k=20):
        super(DGCNNSeg, self).__init__()
        self.k = k
        self.num_classes = num_classes

        # ----------------------------------------------------
        # 1. 动态图卷积特征提取 (Dynamic EdgeConv Layers)
        # ----------------------------------------------------
        self.conv1 = EdgeConv(in_channels=feature_dim, out_channels=64, k=k)
        self.conv2 = EdgeConv(in_channels=64, out_channels=64, k=k)
        self.conv3 = EdgeConv(in_channels=64, out_channels=128, k=k)
        self.conv4 = EdgeConv(in_channels=128, out_channels=256, k=k)

        # ----------------------------------------------------
        # 2. 局部与全局特征聚合 (Local-Global Aggregation)
        # ----------------------------------------------------
        # 聚合四个 EdgeConv 层的输出: 64 + 64 + 128 + 256 = 512
        local_channels = 512
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(local_channels, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # ----------------------------------------------------
        # 3. 分割解码器 (Segmentation Decoder)
        # ----------------------------------------------------
        # 拼接 局部多尺度特征(512) + 全局池化特征(1024) = 1536
        self.seg_head = nn.Sequential(
            nn.Conv1d(1536, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),
            
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),
            
            nn.Conv1d(256, num_classes, kernel_size=1)
        )

    def forward(self, x, pos):
        """
        前向传播
        :param x: 节点特征 [B, N, feature_dim]
        :param pos: 节点物理坐标 [B, N, 3]
        """
        # 转置以符合 1D/2D 卷积习惯 -> [B, C, N]
        x_t = x.transpose(2, 1).contiguous()
        pos_t = pos.transpose(2, 1).contiguous()

        # ==============================================================
        # 步骤 1：构建物理空间感受野
        # 第一次 EdgeConv：使用真实的物理坐标 (pos) 来寻找 K 个邻居，
        # 确保网络第一层能感知到牙齿真实的几何拓扑结构。
        # ==============================================================
        idx_spatial = knn(pos_t, k=self.k)
        out1, _ = self.conv1(x_t, idx=idx_spatial)

        # ==============================================================
        # 步骤 2：动态语义空间特征学习 (Dynamic Graph)
        # 随着层数加深，不传入 idx，让模型自己在高维特征空间重新找邻居！
        # ==============================================================
        out2, _ = self.conv2(out1, idx=None)
        out3, _ = self.conv3(out2, idx=None)
        out4, _ = self.conv4(out3, idx=None)

        # ==============================================================
        # 步骤 3：多尺度局部特征拼接
        # ==============================================================
        local_feat = torch.cat([out1, out2, out3, out4], dim=1) # [B, 512, N]

        # ==============================================================
        # 步骤 4：全局特征提取
        # ==============================================================
        global_feat = self.conv5(local_feat)                   # [B, 1024, N]
        global_feat = global_feat.max(dim=-1, keepdim=True)[0] # Max Pool -> [B, 1024, 1]
        global_feat = global_feat.repeat(1, 1, x_t.size(-1))   # 广播至每点 -> [B, 1024, N]

        # ==============================================================
        # 步骤 5：特征融合与逐点分类
        # ==============================================================
        fusion = torch.cat([local_feat, global_feat], dim=1)   # [B, 1536, N]
        out = self.seg_head(fusion)                            # [B, num_classes, N]

        # ==============================================================
        # 适配 LitToothSegmentationWrapper [B, N, num_classes]
        # ==============================================================
        out = out.transpose(2, 1).contiguous()

        return out