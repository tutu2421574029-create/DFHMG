import torch
import torch.nn as nn
import torch.nn.functional as F

class MeshSegNetSeg(nn.Module):
    """
    MeshSegNet 牙齿分割网络
    核心特点：多级特征提取 (Multi-level Feature Extraction) + 密集特征融合 (Dense Feature Fusion)
    保留了浅层局部信息与深层全局信息，对牙齿边界的分割有较好的效果。
    """
    def __init__(self, num_classes=17, feature_dim=24, with_dropout=True, dropout_p=0.5):
        super(MeshSegNetSeg, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # 输入维度: pos(3) + feature_dim(24) = 27
        in_channels = feature_dim + 3

        # -----------------------------------------------------------------
        # 1. 局部特征提取模块 (Local Feature Extractors)
        # -----------------------------------------------------------------
        # MLP Block 1 (提取基础局部特征)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        # MLP Block 2 (提取高级局部特征)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        # -----------------------------------------------------------------
        # 2. 全局特征提取模块 (Global Feature Extractor)
        # -----------------------------------------------------------------
        # 将高级局部特征映射到高维空间准备全局池化
        self.mlp3 = nn.Sequential(
            nn.Conv1d(64, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )

        # -----------------------------------------------------------------
        # 3. 逐点分类解码器 (Classifier / Decoder)
        # 输入维度为拼接后的维度: 1024(全局) + 64(MLP1) + 64(MLP2) = 1152
        # -----------------------------------------------------------------
        concat_channels = 1024 + 64 + 64
        
        self.decoder = nn.Sequential(
            nn.Conv1d(concat_channels, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p) if with_dropout else nn.Identity(),
            
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p) if with_dropout else nn.Identity(),
            
            nn.Conv1d(128, num_classes, kernel_size=1)
        )

    def forward(self, x, pos):
        """
        x: [B, N, feature_dim] 
        pos: [B, N, 3] 三维坐标
        """
        B, N, C = x.shape
        
        # 1. 拼接空间坐标与几何特征，并将维度调换以适配 Conv1d (B, C, N)
        x_concat = torch.cat([pos, x], dim=-1)
        x_concat = x_concat.transpose(2, 1).contiguous()  # [B, in_channels, N]
        
        # 2. 逐级特征提取
        out1 = self.mlp1(x_concat)         # [B, 64, N]
        out2 = self.mlp2(out1)             # [B, 64, N]
        out3 = self.mlp3(out2)             # [B, 1024, N]
        
        # 3. 对高维特征进行对称函数操作 (Max Pooling) 提取全局上下文
        global_feat = torch.max(out3, 2, keepdim=True)[0]  # [B, 1024, 1]
        global_feat_expand = global_feat.repeat(1, 1, N)   # [B, 1024, N]
        
        # 4. 密集特征融合 (Dense Feature Fusion)
        # 【核心操作】将不同感受野的局部特征与全局特征拼接，这比普通的 PointNet 信息更丰富
        concat_feat = torch.cat([out1, out2, global_feat_expand], dim=1) # [B, 1152, N]
        
        # 5. 解码输出
        logits = self.decoder(concat_feat) # [B, num_classes, N]
        
        # 将输出转换为 [B, N, num_classes] 以适配交叉熵函数
        return logits.transpose(2, 1).contiguous()