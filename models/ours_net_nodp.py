import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ours_net import OursEdgeGraphConvBlock, SpatiallyAnchoredSemanticGraph, OursDualDomainAttention
from models.layer import BasicPointLayer, ResidualBasicPointLayer, PointFeatureImportance, STNkd


class OursNetSegNoDP(nn.Module):
    def __init__(self, num_classes=17, feature_dim=24):
        super(OursNetSegNoDP, self).__init__()
        self.num_classes = num_classes
        self.k = 32

        self.stnkd = STNkd(k=feature_dim)

        self.local_conv1 = OursEdgeGraphConvBlock(
            in_channels=feature_dim, hidden_channels=32, out_channels=32, edge_function="local_global", k=self.k
        )
        self.local_conv2 = OursEdgeGraphConvBlock(
            in_channels=32, hidden_channels=32, out_channels=64, edge_function="local_global", k=self.k
        )
        self.local_shortcut = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64, track_running_stats=False)
        )

        # 注意：已直接去掉双频空洞图卷积模块 (nodp_conv1 和 nodp_conv2)

        # 语义图直接接收来自 l2 的 64 通道特征
        self.sahm_graph = SpatiallyAnchoredSemanticGraph(
            in_channels=64, latent_dim=32, out_channels=128, k=self.k
        )

        self.dual_domain_attn = OursDualDomainAttention(channels=128)
        
        # 多尺度级联融合通道数调整：l2(64) + x_attn(128) = 192
        self.dense_fusion = BasicPointLayer(in_channels=64 + 128, out_channels=256, dropout=0.3)

        self.feature_importance = PointFeatureImportance(in_channels=256)
        self.res_block1 = ResidualBasicPointLayer(in_channels=256, hidden_channels=128, out_channels=128, dropout=0.4)
        self.res_block2 = ResidualBasicPointLayer(in_channels=128, hidden_channels=64, out_channels=64, dropout=0.4)
        self.out = BasicPointLayer(in_channels=64, out_channels=num_classes, is_out=True)

    def forward(self, x, pos):
        B, N, _ = pos.shape

        cd_pos = torch.cdist(pos, pos, p=2)
        topk_local = torch.topk(cd_pos, self.k + 1, largest=False)[1]
        idx_local = topk_local[:, :, 1:self.k + 1].contiguous()

        x_aligned = self.stnkd(x)

        # 阶段1：浅层局部流形基元提取
        l1, _ = self.local_conv1(x_aligned, pos, idx=idx_local)
        l2, _ = self.local_conv2(l1, pos, idx=idx_local)
        l1_res = self.local_shortcut(l1.transpose(2, 1).contiguous()).transpose(2, 1).contiguous()
        l2 = 0.5 * l2 + 0.5 * l1_res

        # 阶段2：已去掉双频空洞图卷积，直接将 l2 送入深层语义构图
        x_sem, _ = self.sahm_graph(l2, pos)
        x_attn = self.dual_domain_attn(x_sem, idx_local=idx_local)

        # 阶段5：密集特征级联（仅拼接局部流形 l2 与全局注意力 x_attn）
        x_fused = torch.cat([l2, x_attn], dim=2)
        x_fused = self.dense_fusion(x_fused)

        # 阶段6：分类输出
        x_fused = self.feature_importance(x_fused)
        x_fused = self.res_block1(x_fused)
        x_fused = self.res_block2(x_fused)
        out = self.out(x_fused)

        return out