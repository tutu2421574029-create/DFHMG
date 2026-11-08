import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    """
    计算 K 近邻 (利用更稳定的 torch.cdist 计算欧氏距离)
    x: [B, C, N]
    返回 idx: [B, N, k]
    """
    x_t = x.transpose(2, 1) # [B, N, C]
    dist = torch.cdist(x_t, x_t, p=2.0) # [B, N, N]
    idx = dist.topk(k=k, dim=-1, largest=False)[1]
    return idx

def get_graph_feature(x, k=20, idx=None):
    """
    基于 KNN 构建局部空间图特征 (Edge Feature)
    用于 PCT 的 Neighbor Embedding 阶段
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        idx = knn(x, k=k)
        
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    # [B, 2*C, N, k]
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class SG_Layer(nn.Module):
    """
    空间图卷积层 (Spatial Graph Layer) / Neighbor Embedding
    PCT 用来提取点云局部细节的模块。
    """
    def __init__(self, in_channels, out_channels, k=20):
        super(SG_Layer, self).__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x, idx=None):
        x_graph = get_graph_feature(x, k=self.k, idx=idx)  # [B, 2*C, N, k]
        x_graph = self.conv(x_graph)
        x_graph = x_graph.max(dim=-1, keepdim=False)[0]    # [B, out_channels, N]
        return x_graph


class SA_Layer(nn.Module):
    """
    Offset-Attention 模块 (PCT 的核心)
    相比标准 Self-Attention，它计算输入特征与注意力聚合特征之间的“偏移 (Offset)”，
    被证明在点云网络中更利于特征提取 (类似于拉普拉斯平滑)。
    """
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [B, C, N]
        x_q = self.q_conv(x).permute(0, 2, 1)  # [B, N, C/4]
        x_k = self.k_conv(x)                   # [B, C/4, N]
        x_v = self.v_conv(x)                   # [B, C, N]

        # 1. 计算注意力权重
        energy = torch.bmm(x_q, x_k)           # [B, N, N]
        attention = self.softmax(energy)
        # 归一化，防止全为 0 导致的问题
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))

        # 2. 注意力特征融合
        x_r = torch.bmm(x_v, attention)        # [B, C, N]
        
        # 3. Offset-Attention: 提取输入与注意力的偏差
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        
        # 4. 加上残差
        out = x + x_r
        return out


class PCT(nn.Module):
    """
    PCT: Point Cloud Transformer 用于三维牙齿分割
    """
    def __init__(self, num_classes=17, feature_dim=24):
        super(PCT, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # 初始特征维度: pos(3) + feature_dim(24) = 27
        in_channels = feature_dim + 3

        # 1. 局部特征嵌入 (Neighbor Embedding)
        self.sg1 = SG_Layer(in_channels, 64, k=20)
        self.sg2 = SG_Layer(64, 64, k=20)

        # 2. 堆叠多层 Offset-Attention
        self.sa1 = SA_Layer(64)
        self.sa2 = SA_Layer(64)
        self.sa3 = SA_Layer(64)
        self.sa4 = SA_Layer(64)

        # 3. 全局特征映射
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(256, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # 4. 逐点分割解码器
        # 输入维度: 局部(256) + 全局(1024) = 1280
        self.convs1 = nn.Conv1d(1280, 512, 1, bias=False)
        self.bns1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        
        self.convs2 = nn.Conv1d(512, 256, 1, bias=False)
        self.bns2 = nn.BatchNorm1d(256)
        
        self.convs3 = nn.Conv1d(256, num_classes, 1)
        self.relu = nn.ReLU()

    def forward(self, x, pos):
        """
        x: [B, N, feature_dim]
        pos: [B, N, 3]
        """
        B, N, _ = x.shape
        
        # 1. 特征融合与形状调整
        x_concat = torch.cat([pos, x], dim=-1)           # [B, N, 27]
        x_concat = x_concat.transpose(2, 1).contiguous() # [B, 27, N]

        # 2. 局部 Neighbor Embedding
        x_embed1 = self.sg1(x_concat)                    # [B, 64, N]
        x_embed2 = self.sg2(x_embed1)                    # [B, 64, N]

        # 3. Self-Attention 机制
        x_sa1 = self.sa1(x_embed2)
        x_sa2 = self.sa2(x_sa1)
        x_sa3 = self.sa3(x_sa2)
        x_sa4 = self.sa4(x_sa3)

        # 4. 提取多层次局部注意力特征
        x_local = torch.cat((x_sa1, x_sa2, x_sa3, x_sa4), dim=1) # [B, 256, N]
        
        # 5. 提取全局特征
        x_global_map = self.conv_fuse(x_local)                   # [B, 1024, N]
        x_global = torch.max(x_global_map, 2, keepdim=True)[0]   # [B, 1024, 1]
        x_global_expand = x_global.repeat(1, 1, N)               # [B, 1024, N]

        # 6. 特征拼接与解码器
        out = torch.cat((x_local, x_global_expand), dim=1)       # [B, 1280, N]

        out = self.relu(self.bns1(self.convs1(out)))
        out = self.dp1(out)
        out = self.relu(self.bns2(self.convs2(out)))
        logits = self.convs3(out)                                # [B, num_classes, N]

        # 转换为适配 PyTorch Lightning Wrapper 的维度: [B, N, num_classes]
        return logits.transpose(2, 1).contiguous()