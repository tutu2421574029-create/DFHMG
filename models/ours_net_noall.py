import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ours_net import OursEdgeGraphConvBlock
from models.layer import BasicPointLayer, ResidualBasicPointLayer, PointFeatureImportance, STNkd


class OursNetSegNoAll(nn.Module):
    def __init__(self, num_classes=17, feature_dim=24):
        super(OursNetSegNoAll, self).__init__()
        self.num_classes = num_classes
        self.k = 32

        self.stnkd = STNkd(k=feature_dim)

        # 纯基础边卷积模块堆叠
        self.conv1 = OursEdgeGraphConvBlock(in_channels=feature_dim, hidden_channels=32, out_channels=64, edge_function="local_global", k=self.k)
        self.conv2 = OursEdgeGraphConvBlock(in_channels=64, hidden_channels=64, out_channels=128, edge_function="local_global", k=self.k)
        self.conv3 = OursEdgeGraphConvBlock(in_channels=128, hidden_channels=128, out_channels=256, edge_function="local_global", k=self.k)

        self.dense_fusion = BasicPointLayer(in_channels=64 + 128 + 256, out_channels=256, dropout=0.3)

        self.feature_importance = PointFeatureImportance(in_channels=256)
        self.res_block1 = ResidualBasicPointLayer(in_channels=256, hidden_channels=128, out_channels=128, dropout=0.4)
        self.res_block2 = ResidualBasicPointLayer(in_channels=128, hidden_channels=64, out_channels=64, dropout=0.4)
        self.out = BasicPointLayer(in_channels=64, out_channels=num_classes, is_out=True)

    def forward(self, x, pos):
        B, N, _ = pos.shape
        cd_pos = torch.cdist(pos, pos, p=2)
        idx_local = torch.topk(cd_pos, self.k + 1, largest=False)[1][:, :, 1:].contiguous()

        x_aligned = self.stnkd(x)

        f1, _ = self.conv1(x_aligned, pos, idx=idx_local)
        f2, _ = self.conv2(f1, pos, idx=idx_local)
        f3, _ = self.conv3(f2, pos, idx=idx_local)

        x_fused = torch.cat([f1, f2, f3], dim=2)
        x_fused = self.dense_fusion(x_fused)

        x_fused = self.feature_importance(x_fused)
        x_fused = self.res_block1(x_fused)
        x_fused = self.res_block2(x_fused)
        out = self.out(x_fused)

        return out