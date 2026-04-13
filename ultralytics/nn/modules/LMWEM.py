import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

def dwt(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2];
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2];
    x4 = x02[:, :, :, 1::2]
    LL = x1 + x2 + x3 + x4
    HL = -x1 - x2 + x3 + x4
    LH = -x1 + x2 - x3 + x4
    HH = x1 - x2 - x3 + x4
    return LL, HL, LH, HH

def iwt(cat):
    B, C4, H, W = cat.shape
    C = C4 // 4
    x1 = cat[:, 0:C, :, :] / 2
    x2 = cat[:, C:2 * C, :, :] / 2
    x3 = cat[:, 2 * C:3 * C, :, :] / 2
    x4 = cat[:, 3 * C:4 * C, :, :] / 2
    out = torch.zeros(B, C, H * 2, W * 2, device=cat.device)
    out[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    out[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    out[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    out[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return out

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        if isinstance(x, torch.Tensor) and x.dim() == 4 and x.size(0) == 1 and x.sum() == 0 \
                and x.size(2) == x.size(3):
            return x
        return dwt(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, ll, hl, lh, hh):
        x = torch.cat((ll, hl, lh, hh), 1)
        return iwt(x)

class WinvIWT(nn.Module):
    def __init__(self):
        super(WinvIWT, self).__init__()
        self.requires_grad = False

    def forward(self, LL, HH):
        return iwt(torch.cat([LL, HH], dim=1))

class L_transform(nn.Module):
    def __init__(self, dim):
        super(L_transform, self).__init__()

        # self.trans = nn.Conv2d(dim, out_n_feat, kernel_size=1, bias=True)
        # For Channel dimennsion
        self.conv_C = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(dim, 4 * dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            # torch.nn.Conv2d(16 * c, 16 * (16 * c), kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            torch.nn.Sigmoid()
        )

        # For Height dimennsion
        self.conv_H = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((None, 1)),
            torch.nn.Conv2d(dim, 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            # torch.nn.Conv2d(16 * c, 16, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            torch.nn.Sigmoid()
        )

        # For Width dimennsion
        self.conv_W = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, None)),
            torch.nn.Conv2d(dim, 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            # torch.nn.Conv2d(16 * c, 16, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            torch.nn.Sigmoid()
        )

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        N_, C_, H_, W_ = x.shape

        # res = x
        s0_3_c = self.conv_C(x)

        s0_3_c = s0_3_c.view(N_, 4, -1, 1, 1)

        s0_3_h = self.conv_H(x)
        s0_3_h = s0_3_h.view(N_, 4, 1, -1, 1)

        s0_3_w = self.conv_W(x)
        s0_3_w = s0_3_w.view(N_, 4, 1, 1, -1)

        cube0 = (s0_3_c * s0_3_h * s0_3_w).mean(1)

        return x * cube0

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        maxv = self.fc(self.max_pool(x))
        return torch.sigmoid(avg + maxv)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, maxv], dim=1)))

class StripAttention(nn.Module):
    def __init__(self, channels, directions=('h', 'v'), local_kernel=3, bottleneck_ratio=4):
        super().__init__()
        # validate directions
        allowed = {'h', 'v', '45', '135'}
        assert all(d in allowed for d in directions), f"directions must be subset of {allowed}"
        self.channels = channels
        self.dirs = list(directions)
        self.num_parts = len(self.dirs)

        # local encoder: applied after concat (depthwise across concatenated channels)
        # Use depthwise conv with groups = channels * num_parts
        self.local_enc = nn.Sequential(
            nn.Conv2d(channels * self.num_parts, channels * self.num_parts,
                      kernel_size=local_kernel, padding=local_kernel // 2,
                      groups=channels * self.num_parts, bias=False),
            nn.BatchNorm2d(channels * self.num_parts),
            nn.GELU()
        )

        # fuse 1x1 conv: reduce channels back to C
        self.fuse_conv = nn.Conv2d(in_channels=channels * self.num_parts,
                                   out_channels=channels,
                                   kernel_size=1, bias=False)
        self.fuse_bn = nn.BatchNorm2d(channels)
        self.fuse_act = nn.GELU()

        # small KV proj if later needed (kept for compatibility; can be used externally)
        mid = max(channels // bottleneck_ratio, 1)
        self.kv_proj = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        )

        # initialization
        nn.init.kaiming_normal_(self.fuse_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.local_enc[0].weight, mode='fan_out', nonlinearity='relu')

    def strip_pool_h(self, x):
        # horizontal strip pooling: average over height -> [B, C, 1, W]
        return F.adaptive_avg_pool2d(x, (1, x.size(3)))

    def strip_pool_v(self, x):
        # vertical strip pooling: average over width -> [B, C, H, 1]
        return F.adaptive_avg_pool2d(x, (x.size(2), 1))

    def strip_pool_diag(self, x, theta_deg):
        """
        Diagonal pooling (45 or 135 degrees) implemented via projection grouping.
        Returns tensor of shape [B, C, H, W] (values filled per pixel by its group's mean).
        This method properly handles varying lengths of diagonal strips.
        """
        # x: [B,C,H,W]
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # coordinate grid (u: row index, v: col index), using 0..H-1 and 0..W-1
        u = torch.arange(0, H, device=device, dtype=dtype).view(H, 1)  # [H,1]
        v = torch.arange(0, W, device=device, dtype=dtype).view(1, W)  # [1,W]

        # convert degrees to radians and compute projection value
        theta = float(theta_deg) * torch.pi / 180.0
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        # projection float map shape [H,W]
        proj = (u * cos_t) + (v * sin_t)  # [H,W]

        # discretize into integer bins (use floor)
        proj_floor = torch.floor(proj)  # [H,W]
        proj_min = proj_floor.min()
        idx_map = (proj_floor - proj_min).long()  # [H,W], values in 0..G-1
        G = int(idx_map.max().item()) + 1  # number of groups

        # flatten spatial dims
        N = H * W
        x_flat = x.view(B, C, N)                       # [B,C,N]
        idx_flat = idx_map.view(N).to(device)          # [N]

        # prepare index tensor expanded to [B,C,N]
        idx_expand = idx_flat.view(1, 1, N).expand(B, C, N)  # [B,C,N]

        # sums per group: shape [B, C, G]
        sums = torch.zeros(B, C, G, device=device, dtype=dtype)
        sums = sums.scatter_add_(2, idx_expand, x_flat)

        # counts per group: shape [1,1,G] or [B,1,G]
        ones = torch.ones(B, 1, N, device=device, dtype=dtype)
        counts = torch.zeros(B, 1, G, device=device, dtype=dtype)
        counts = counts.scatter_add_(2, idx_expand[:, :1, :], ones)  # use same idx for counts

        # avoid div 0
        counts = counts.clamp(min=1.0)

        # mean per group: [B, C, G]
        pooled = sums / counts  # broadcasting along channel

        # map group means back to per-pixel values
        # pooled_per_pixel: gather along group dim
        # idx_expand for gather needs shape [B, C, N]
        # but gather on dim=2: input pooled [B,C,G], index [B,C,N] -> out [B,C,N]
        idx_expand_for_gather = idx_expand
        pooled_per_pixel = torch.gather(pooled, 2, idx_expand_for_gather)  # [B,C,N]

        # reshape back to [B,C,H,W]
        out = pooled_per_pixel.view(B, C, H, W)
        return out

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: fused tensor [B, C, H, W] representing directional strip features
        (ready to serve as x to Multihead_Attention)
        """
        B, C, H, W = x.shape
        parts = []

        for d in self.dirs:
            if d == 'h':
                p = self.strip_pool_h(x)          # [B,C,1,W]
                p_up = F.interpolate(p, size=(H, W), mode='nearest')  # [B,C,H,W]
                parts.append(p_up)
            elif d == 'v':
                p = self.strip_pool_v(x)          # [B,C,H,1]
                p_up = F.interpolate(p, size=(H, W), mode='nearest')
                parts.append(p_up)
            elif d == '45':
                p_up = self.strip_pool_diag(x, 45.0)  # already [B,C,H,W]
                parts.append(p_up)
            elif d == '135':
                p_up = self.strip_pool_diag(x, 135.0)
                parts.append(p_up)
            else:
                raise NotImplementedError(f"Direction {d} not implemented.")

        # concat along channel dim
        concat = torch.cat(parts, dim=1)  # [B, C * num_parts, H, W]

        # local encoding (depthwise across concatenated channels) for local variation
        enc = self.local_enc(concat)      # [B, C * num_parts, H, W]

        # fuse and reduce to original channels
        fused = self.fuse_conv(enc)       # [B, C, H, W]
        fused = self.fuse_bn(fused)
        fused = self.fuse_act(fused)

        return fused  # this tensor plays same role as original BindAttention output

class Multihead_Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super(Multihead_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim , kernel_size=1)
        self.kv_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.q = nn.Conv2d(dim,dim,kernel_size=1)
        self.q_dwconv = nn.Conv2d(dim, dim,kernel_size=3, stride=1, padding=1, groups=dim)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x , y):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(x))   # 计算一次
        k = kv
        v = kv
        q = self.q_dwconv(self.q(y))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class LowFreqBranch(nn.Module):
    """Low-frequency branch: depthwise separable large kernels + channel & spatial attention"""

    def __init__(self, channels, reduction=2, attn_reduction=16, spatial_kernel=7):
        super().__init__()
        ks = [5, 7, 9]
        mid = max(channels // reduction, 1)
        self.branches = nn.ModuleList()
        for k in ks:
            stages = (k - 1) // 2
            layers = [nn.Conv2d(channels, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.GELU()]
            for _ in range(stages):
                layers.append(nn.Conv2d(mid, mid, (3, 1), padding=(1, 0), groups=mid, bias=False))
                layers.append(nn.Conv2d(mid, mid, (1, 3), padding=(0, 1), groups=mid, bias=False))
                layers.append(nn.BatchNorm2d(mid))
                layers.append(nn.GELU())
            layers.extend([nn.Conv2d(mid, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.GELU()])
            self.branches.append(nn.Sequential(*layers))
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // attn_reduction, 1), nn.ReLU(),
            nn.Conv2d(channels // attn_reduction, channels, 1), nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = sum(b(x) for b in self.branches)
        out = out * self.ca(out)
        avg = out.mean(1, True)
        mx, _ = out.max(1, True)
        out = out * self.sa(torch.cat([avg, mx], 1)) + x

        # b = self.b_conv(out)
        return out

class HighFreqBranch(nn.Module):
    def __init__(self, channels, use_diagonals=True):
        super().__init__()
        ks = [1, 3, 5]
        self.dw = nn.ModuleList([
            nn.Conv2d(channels * 3, channels * 3, k, padding=k // 2, groups=channels, bias=False)
            for k in ks
        ])
        self.cba = nn.Sequential(
            nn.Conv2d(3 * channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        # Attention-related modules
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()

        # Replace BindAttention (h/v) with StripAttention (supporting diagonals optionally)
        self.strip_h = StripAttention(channels, directions=('h',), local_kernel=3)
        self.strip_v = StripAttention(channels, directions=('v',), local_kernel=3)
        self.use_diagonals = use_diagonals
        if self.use_diagonals:
            self.strip_45 = StripAttention(channels, directions=('45',), local_kernel=3)
            self.strip_135 = StripAttention(channels, directions=('135',), local_kernel=3)
        else:
            self.strip_45 = None
            self.strip_135 = None

        # Fuse attended features
        self.fuse_attn = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        self.self_attn = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
        self.norm_f1 = nn.GroupNorm(num_groups=8, num_channels=channels)

        # Multihead_Attention (same as original) expects (x, y): x->kv source, y->q source
        self.multihead_attn = Multihead_Attention(channels, 4)

        # Low-rank transform and shared trunk remain the same
        self.l_transform1 = L_transform(channels)
        self.shared_trunk = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1),
        )
        self.w_head = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.b_head = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x, low):
        batch_size, channels_times3, H, W = x.shape
        channels = channels_times3 // 3

        # 1) Multi-kernel depthwise convolution fusion
        dw_feats = [conv(x) for conv in self.dw]
        mid = self.cba(torch.cat(dw_feats, dim=1))  # [B, C, H, W]
        origin = mid

        # 2) self-attn on mid to get spatial attention map
        mid_flat = mid.flatten(2).transpose(1, 2)  # [B, HW, C]
        attn_out, attn_weights = self.self_attn(mid_flat, mid_flat, mid_flat)  # attn_weights [B, HW, HW]
        attn_weights = attn_weights.mean(dim=1).unsqueeze(1)  # [B,1,HW]
        attn_weights_2d = attn_weights.view(batch_size, 1, H, W)  # [B,1,H,W]

        # 3) f1 = attention map * low  (guide low-frequency)
        f1 = attn_weights_2d * low  # [B,C,H,W]

        # 4) normalize and low-rank transform + shared trunk to produce affine params
        f2 = self.norm_f1(f1)
        tr = self.shared_trunk(self.l_transform1(f2))
        w_affine = self.w_head(tr)
        b_affine = self.b_head(tr)
        mid = w_affine * mid + b_affine  # affine modulated mid

        # 5) Generate strip representations (to be used as KV source for multihead attention)
        q_h = self.strip_h(mid)    # [B, C, H, W]
        q_v = self.strip_v(mid)    # [B, C, H, W]

        # optional diagonals
        if self.use_diagonals:
            q_45 = self.strip_45(mid)
            q_135 = self.strip_135(mid)
        else:
            q_45 = None
            q_135 = None

        # 6) cross-branch multihead attention: (x, y) where x -> kv source, y -> query (low)
        h_feature = self.multihead_attn(q_h, low)
        v_feature = self.multihead_attn(q_v, low)

        # add diagonals if used
        if self.use_diagonals:
            d45_feature = self.multihead_attn(q_45, low)
            d135_feature = self.multihead_attn(q_135, low)
            attn_sum = h_feature + v_feature + d45_feature + d135_feature
        else:
            attn_sum = h_feature + v_feature

        # 7) fuse and residual with origin
        out = self.fuse_attn(attn_sum) + origin

        return out

class LMWEM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        channels = args[0] if args else kwargs.get('channels')
        self.dwt = DWT()
        self.iwt = WinvIWT()
        self.low_branch = LowFreqBranch(channels)
        self.high_branch = HighFreqBranch(channels)
        self.resize = nn.Sequential(
            nn.Conv2d(channels // 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

    def forward(self, x):
        dtype = x.dtype
        device = x.device
        if x.dim() == 4 and x.size(0) == 1 and x.sum() == 0:
            return x


        _, _, H0, W0 = x.shape
        pad_h = H0 % 2
        pad_w = W0 % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        LL, HL, LH, HH = dwt(x)
        # print("LL:", LL.shape)
        # print("HL:", HL.shape)
        # print("LH:", LH.shape)
        # print("HH:", HH.shape)
        low = self.low_branch(LL)
        high = self.high_branch(torch.cat([HL, LH, HH], 1),low)

        out = iwt(torch.cat([low , high ], dim=1))
        # return self.resize(out)
        self.resize = self.resize.to(device=device, dtype=dtype)
        out = self.resize(out)
        if pad_h or pad_w:
            out = out[..., :H0, :W0]
        return out


if __name__ == "__main__":
    model = LMWEM(128)
    x = torch.randn(2, 128, 64, 64)  # batch=2, in_channels=3
    out = model(x)
    print("out:", out.shape)

