import torch
import torch.nn as nn

# =========================================================================
# 动态 KNN 构图与边特征提取 (包含 FP16 半精度安全保护)
# =========================================================================
def knn(x, k):
    """
    计算 K 近邻，输入 x 形状为 (B, C, N)
    """
    x_t = x.transpose(2, 1).contiguous() # (B, N, C)
    dist = torch.cdist(x_t, x_t, p=2.0)
    
    # [FP16 保护]: 防止浮点误差产生负距离导致 NaN
    dist = torch.clamp(dist, min=0.0) 
    
    idx = dist.topk(k=k, dim=-1, largest=False)[1] # (B, N, K)
    return idx

def get_graph_feature(x, k, idx=None):
    """
    提取输入感知的图边特征 (x_j - x_i, x_i)
    """
    B, C, N = x.size()
    if idx is None:
        idx = knn(x, k=k)
        
    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx_flat = (idx + idx_base).view(-1)
    
    x_t = x.transpose(2, 1).contiguous()
    # 邻居节点特征: (B, N, K, C)
    x_feat = x_t.view(B * N, C)[idx_flat, :].view(B, N, k, C)
    
    # 中心节点特征扩展: (B, N, K, C)
    x_repeat = x_t.view(B, N, 1, C).repeat(1, 1, k, 1)
    
    # 论文标准 Edge 特征: [邻居-中心, 中心]
    feature = torch.cat((x_feat - x_repeat, x_repeat), dim=-1) # (B, N, K, 2C)
    feature = feature.permute(0, 3, 1, 2).contiguous()         # (B, 2C, N, K)
    return feature, idx


# =========================================================================
# 输入感知图卷积模块 (Input-Aware Graph Convolution)
# =========================================================================
class TSGCNet_EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=16):
        super(TSGCNet_EdgeConv, self).__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x, idx=None):
        # 提取图特征
        feature, idx = get_graph_feature(x, self.k, idx)
        
        # 卷积并聚合 (Max Pooling)
        out = self.conv(feature)
        out = out.max(dim=-1, keepdim=False)[0] # (B, out_channels, N)
        return out, idx


# =========================================================================
# 真·TSGCNet (Two-Stream Graph Convolutional Network)
# =========================================================================
class TSGCNetSeg(nn.Module):
    """
    严格按照 TSGCNet 论文实现的双流架构网络
    - Stream 1: 专门学习 Coordinates (pos) 的拓扑结构
    - Stream 2: 专门学习 Normals/Attributes (x) 的拓扑结构
    """
    def __init__(self, num_classes=17, feature_dim=24, k=16):
        super(TSGCNetSeg, self).__init__()
        self.k = k
        
        # ----------------------------------------------------
        # Stream 1: Coordinate Stream (提取空间物理几何特征)
        # 输入维度: 3 (XYZ 坐标)
        # ----------------------------------------------------
        self.coord_conv1 = TSGCNet_EdgeConv(3, 64, k)
        self.coord_conv2 = TSGCNet_EdgeConv(64, 64, k)
        self.coord_conv3 = TSGCNet_EdgeConv(64, 128, k)
        
        # ----------------------------------------------------
        # Stream 2: Attribute Stream (提取法向量等附加鉴别特征)
        # 输入维度: feature_dim (24维法向+曲率等特征)
        # ----------------------------------------------------
        self.attr_conv1 = TSGCNet_EdgeConv(feature_dim, 64, k)
        self.attr_conv2 = TSGCNet_EdgeConv(64, 64, k)
        self.attr_conv3 = TSGCNet_EdgeConv(64, 128, k)
        
        # 两条流特征拼接后的总通道数: (64+64+128) * 2 = 512
        local_channels = 512
        
        # ----------------------------------------------------
        # 全局融合层 (Global Feature Fusion)
        # ----------------------------------------------------
        self.global_conv = nn.Sequential(
            nn.Conv1d(local_channels, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # ----------------------------------------------------
        # 逐点密集预测层 (Cell-wise Dense Prediction)
        # ----------------------------------------------------
        # 拼接 局部特征(512) + 全局特征(1024) = 1536
        self.seg_head = nn.Sequential(
            nn.Conv1d(local_channels + 1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),
            nn.Conv1d(256, num_classes, kernel_size=1)
        )

    def forward(self, x, pos):
        """
        :param x: 节点属性 (法向量、曲率等) [B, N, feature_dim]
        :param pos: 节点坐标 (Coordinates) [B, N, 3]
        """
        B, N, _ = x.shape
        
        # 调整至 (B, C, N) 格式供 1D/2D 卷积使用
        x_attr = x.transpose(2, 1).contiguous()
        x_coord = pos.transpose(2, 1).contiguous()
        
        # ==============================================================
        # Stream 1 执行：Input-aware，仅基于坐标 (pos) 构建 KNN 图
        # ==============================================================
        # 注意：这里我们让第一层独立计算坐标图，后续沿用该拓扑或让特征自行学习新图
        c1, idx_c1 = self.coord_conv1(x_coord)
        c2, idx_c2 = self.coord_conv2(c1, idx_c1)  # 使用前一层的索引，保持物理拓扑稳定性
        c3, idx_c3 = self.coord_conv3(c2, idx_c2)
        
        # ==============================================================
        # Stream 2 执行：Input-aware，仅基于属性 (x) 构建 KNN 图
        # ==============================================================
        # 法向量等属性会构建出对“尖锐边缘、沟壑”等几何语义更敏感的拓扑图
        a1, idx_a1 = self.attr_conv1(x_attr)
        a2, idx_a2 = self.attr_conv2(a1, idx_a1)
        a3, idx_a3 = self.attr_conv3(a2, idx_a2)
        
        # ==============================================================
        # 多视点信息融合 (Multi-View Integration)
        # ==============================================================
        local_feat = torch.cat([c1, c2, c3, a1, a2, a3], dim=1)  # [B, 512, N]
        
        global_feat = self.global_conv(local_feat)               # [B, 1024, N]
        global_feat = global_feat.max(dim=-1, keepdim=True)[0]   # [B, 1024, 1]
        global_feat = global_feat.repeat(1, 1, N)                # [B, 1024, N]
        
        # 拼接全局与局部进行最终逐像素分类
        concat_feat = torch.cat([local_feat, global_feat], dim=1) # [B, 1536, N]
        out = self.seg_head(concat_feat)                          # [B, num_classes, N]
        
        # 适配 LitWrapper 的 CrossEntropyLoss 期望的 [B, N, num_classes] 维度
        out = out.transpose(2, 1).contiguous()
        
        return out