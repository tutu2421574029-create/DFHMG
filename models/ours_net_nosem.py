import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ours_net import OursEdgeGraphConvBlock, OursDualPathDilatedGCN, OursDualDomainAttention
from models.layer import BasicPointLayer, ResidualBasicPointLayer, PointFeatureImportance, STNkd


class OursNetSegNoSem(nn.Module):
    def __init__(self, num_classes=17, feature_dim=24):
        super(OursNetSegNoSem, self).__init__()
        self.num_classes = num_classes
        self.k = 32
        self.max_dilation_k = 600

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

        self.dp_dilated1 = OursDualPathDilatedGCN(in_channels=64, out_channels=64, k=self.k, dilation_k=300)
        self.dp_dilated2 = OursDualPathDilatedGCN(in_channels=64, out_channels=64, k=self.k, dilation_k=600)

        # 去掉动态构图，直接用 1x1 卷积映射通道 (64 -> 128)
        self.no_sem_proj = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.dual_domain_attn = OursDualDomainAttention(channels=128)
        self.dense_fusion = BasicPointLayer(in_channels=64 + 64 + 64 + 128, out_channels=256, dropout=0.3)

        self.feature_importance = PointFeatureImportance(in_channels=256)
        self.res_block1 = ResidualBasicPointLayer(in_channels=256, hidden_channels=128, out_channels=128, dropout=0.4)
        self.res_block2 = ResidualBasicPointLayer(in_channels=128, hidden_channels=64, out_channels=64, dropout=0.4)
        self.out = BasicPointLayer(in_channels=64, out_channels=num_classes, is_out=True)

    def forward(self, x, pos):
        B, N, _ = pos.shape

        cd_pos = torch.cdist(pos, pos, p=2)
        max_k = min(self.max_dilation_k, N)
        topk_all = torch.topk(cd_pos, max_k, largest=False)[1]
        idx_local = topk_all[:, :, 1:self.k + 1].contiguous()

        x_aligned = self.stnkd(x)

        l1, _ = self.local_conv1(x_aligned, pos, idx=idx_local)
        l2, _ = self.local_conv2(l1, pos, idx=idx_local)
        l1_res = self.local_shortcut(l1.transpose(2, 1).contiguous()).transpose(2, 1).contiguous()
        l2 = 0.5 * l2 + 0.5 * l1_res

        d1 = self.dp_dilated1(l2, pos, precomputed_topk=topk_all)
        d2 = self.dp_dilated2(d1, pos, precomputed_topk=topk_all)
        d_ms = 0.5 * d1 + 0.5 * d2

        x_sem = self.no_sem_proj(d_ms.transpose(2, 1).contiguous()).transpose(2, 1).contiguous()
        x_attn = self.dual_domain_attn(x_sem, idx_local=idx_local)

        x_fused = torch.cat([l2, d1, d2, x_attn], dim=2)
        x_fused = self.dense_fusion(x_fused)

        x_fused = self.feature_importance(x_fused)
        x_fused = self.res_block1(x_fused)
        x_fused = self.res_block2(x_fused)
        out = self.out(x_fused)

        return out