import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layer import (
    BasicPointLayer, 
    ResidualBasicPointLayer, 
    PointFeatureImportance, 
    STNkd,
    get_graph_feature, 
    fps, 
    batched_index_select
)


class OursEdgeGraphConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_function="local_global", k=32):
        super(OursEdgeGraphConvBlock, self).__init__()
        self.edge_function = edge_function
        self.in_channels = in_channels
        self.k = k

        if edge_function not in ["global", "local", "local_global"]:
            raise ValueError(f'Edge Function {edge_function} is not allowed.')
        if edge_function == "local_global":
            self.in_channels = self.in_channels * 2

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x, pos=None, idx=None):
        x_t = x.transpose(2, 1).contiguous()
        if pos is None:
            pos = x
        pos_t = pos.transpose(2, 1).contiguous()

        out, idx = get_graph_feature(x_t, edge_function=self.edge_function, k=self.k, idx=idx, pos=pos_t)
        out = self.conv(out)

       
        out = out.max(dim=-1, keepdim=False)[0]

        out = out.transpose(2, 1).contiguous()
        return out, idx


class OursDualPathDilatedGCN(nn.Module):
    def __init__(self, in_channels, out_channels, k=32, dilation_k=300):
        super(OursDualPathDilatedGCN, self).__init__()
        self.k = k
        self.dilation_k = dilation_k
        half_c = out_channels // 2

        self.high_freq_edge_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, half_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(half_c, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(half_c, half_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(half_c, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.laplacian_proj = nn.Sequential(
            nn.Conv1d(in_channels, half_c, kernel_size=1, bias=False),
            nn.BatchNorm1d(half_c, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.low_freq_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, half_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(half_c, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(half_c, half_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(half_c, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.low_edge_affinity = nn.Conv2d(half_c, 1, kernel_size=1, bias=False)
        self.tau = nn.Parameter(torch.tensor(0.2))

        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.gate = nn.Sequential(
            nn.Conv1d(out_channels, half_c, kernel_size=1, bias=False),
            nn.BatchNorm1d(half_c, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(half_c, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels, track_running_stats=False)
            )

    def forward(self, x, pos, cd=None, precomputed_topk=None):
        B, N, C = x.shape
        x_t = x.transpose(2, 1).contiguous()
        dilation_k = min(self.dilation_k, N)

        if precomputed_topk is not None and precomputed_topk.shape[-1] >= dilation_k:
            topk_idx = precomputed_topk[:, :, :dilation_k]
        else:
            if cd is None:
                cd = torch.cdist(pos, pos, p=2)
            topk_idx = torch.topk(cd, dilation_k, largest=False)[1]

        idx_inner = topk_idx[:, :, 1:self.k + 1].contiguous()
        edge_high, _ = get_graph_feature(x_t, edge_function="local_global", k=self.k, idx=idx_inner)
        f_high_raw = self.high_freq_edge_conv(edge_high)
        
        f_high_edge = f_high_raw.max(dim=-1)[0]

        signed_laplacian = -edge_high[:, :C, :, :].mean(dim=-1).contiguous()
        f_high_lap = self.laplacian_proj(signed_laplacian)
        f_high = 0.5 * (f_high_edge + f_high_lap)

        inner_cut = self.k // 2 if dilation_k > self.k + (self.k // 2) else 0
        idx_outer = topk_idx[:, :, inner_cut:].reshape(B * N, -1)
        idx_fps = fps(pos.reshape(B * N, -1)[idx_outer], self.k).long()
        idx_fps = batched_index_select(idx_outer, 1, idx_fps).reshape(B, N, self.k)

        edge_low, _ = get_graph_feature(x_t, edge_function="local_global", k=self.k, idx=idx_fps)
        f_low_raw = self.low_freq_conv(edge_low)

        diff_low = edge_low[:, :C, :, :]
        center_low = edge_low[:, C:, :, :]
        neighbor_low = diff_low + center_low
        sim_logits = torch.sum(
            F.normalize(center_low, p=2, dim=1) * F.normalize(neighbor_low, p=2, dim=1),
            dim=1, keepdim=True
        )
        edge_logits = self.low_edge_affinity(f_low_raw)
        attn_low = F.softmax(sim_logits / self.tau.clamp(min=0.05) + edge_logits, dim=-1)

        
        f_low = torch.sum(f_low_raw * attn_low, dim=-1)

        fused = self.fusion(torch.cat([f_high, f_low], dim=1))
        gate_weights = self.gate(fused)
        out = fused * gate_weights

        res = self.shortcut(x_t)
        return (0.5 * out + 0.5 * res).transpose(2, 1).contiguous()


class SpatiallyAnchoredSemanticGraph(nn.Module):
    def __init__(self, in_channels=64, latent_dim=32, out_channels=128, k=32):
        super(SpatiallyAnchoredSemanticGraph, self).__init__()
        self.metric_proj = nn.Sequential(
            nn.Conv1d(in_channels, latent_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(latent_dim, track_running_stats=False)
        )
        self.spatial_lambda = nn.Parameter(torch.tensor(0.5))

        self.semantic_graph_conv = OursEdgeGraphConvBlock(
            in_channels=in_channels, hidden_channels=out_channels, out_channels=out_channels,
            edge_function="local_global", k=k
        )

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x, pos):
        z = self.metric_proj(x.transpose(2, 1).contiguous()).transpose(2, 1).contiguous()
        z_hat = F.normalize(z, p=2, dim=-1)

        pos_centered = pos - pos.mean(dim=1, keepdim=True)
        pos_scale = torch.sqrt(torch.sum(pos_centered ** 2, dim=-1, keepdim=True).mean(dim=1, keepdim=True) + 1e-6)
        p_tilde = pos_centered / pos_scale

        hybrid_pos = torch.cat([z_hat, self.spatial_lambda * p_tilde], dim=-1)

        out, idx = self.semantic_graph_conv(x, pos=hybrid_pos)
        res = self.shortcut(x.transpose(2, 1).contiguous()).transpose(2, 1).contiguous()
        return 0.5 * out + 0.5 * res, idx


class OursDualDomainAttention(nn.Module):
    def __init__(self, channels):
        super(OursDualDomainAttention, self).__init__()
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels, bias=False)
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(6, 8, kernel_size=1, bias=False),
            nn.BatchNorm1d(8, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(8, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x, idx_local=None):
        B, N, C = x.shape

        avg_pool = x.mean(dim=1)
        max_pool = x.max(dim=1)[0]
        c_weight = torch.sigmoid(self.channel_mlp(avg_pool) + self.channel_mlp(max_pool)).unsqueeze(1)
        x_c = x * c_weight

        x_trans = x_c.transpose(2, 1).contiguous()
        s_avg = torch.mean(x_trans, dim=1, keepdim=True)
        s_max = torch.max(x_trans, dim=1, keepdim=True)[0]
        s_point = torch.cat([s_avg, s_max], dim=1)

        if idx_local is not None:
            k = idx_local.shape[-1]
            s_point_t = s_point.transpose(2, 1).contiguous()
            s_neighbors = batched_index_select(s_point_t, 1, idx_local.reshape(B, -1)).reshape(B, N, k, 2)
            s_neigh_mean = s_neighbors.mean(dim=2).transpose(2, 1).contiguous()
            s_diff = s_point - s_neigh_mean
            s_in = torch.cat([s_point, s_neigh_mean, s_diff], dim=1)
        else:
            s_in = torch.cat([s_point, s_point, torch.zeros_like(s_point)], dim=1)

        s_weight = self.spatial_conv(s_in)
        x_out = (x_trans * s_weight).transpose(2, 1).contiguous()

        return x_out + x


# =========================================================================
# 
# =========================================================================
class OursNetSeg(nn.Module):
    def __init__(self, num_classes=17, feature_dim=24):
        super(OursNetSeg, self).__init__()
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

        self.sahm_graph = SpatiallyAnchoredSemanticGraph(
            in_channels=64, latent_dim=32, out_channels=128, k=self.k
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

        x_sem, _ = self.sahm_graph(d_ms, pos)
        x_attn = self.dual_domain_attn(x_sem, idx_local=idx_local)

        x_fused = torch.cat([l2, d1, d2, x_attn], dim=2)
        x_fused = self.dense_fusion(x_fused)

        x_fused = self.feature_importance(x_fused)
        x_fused = self.res_block1(x_fused)
        x_fused = self.res_block2(x_fused)
        out = self.out(x_fused)

        return out