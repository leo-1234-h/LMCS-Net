
import torch
import torch.nn as nn
import torch.nn.functional as F


class CSDKM(nn.Module):

    def __init__(
        self,
        ch_in,
        out_channels: int,
        kernel_size: int = 3,
        groups: int = 4,
        S: int = 3,
        sd_factor: int = 2,
        sim_type: str = "dot",
    ):
        super().__init__()

        self.ch_in = ch_in
        self.out_c = int(out_channels)
        self.K = kernel_size if (kernel_size % 2 == 1) else (kernel_size + 1)
        self.K2 = self.K * self.K
        self.pad = self.K // 2

        self.G = int(groups)
        self.S = int(S)
        self.S2 = self.S * self.S
        self.sd = int(sd_factor)
        self.sim_type = sim_type

        self.to_fuse = None
        self.norm = None
        self.c4_proc_conv = None
        self.conv1 = None
        self.reshape = None
        self.proj_fused = None

        self.sim_proj_c4 = None
        self.sim_proj_c5 = None
        self.bilinear_W = None

        self.kernel_gen = None

        # learnable gating mask (raw), sigmoid before use
        self.mask_raw = nn.Parameter(torch.zeros(self.S2))

        self.act = nn.SiLU()
        self._built = False

    def build(self, c4_ch: int, c5_ch: int):
        if self._built:
            return

        # 1) to_fuse and normalization
        self.to_fuse = nn.Conv2d(c4_ch, self.out_c, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(self.out_c)

        # 2) lightweight processing of C4 (preserve spatial dims)
        # use 3x3 conv with padding to keep shape stable for any HxW
        self.c4_proc_conv = nn.Conv2d(c4_ch, c4_ch, kernel_size=3, padding=1, bias=False)

        # 3) map C5 to C4 channel dim (after upsampling/ pooling)
        self.conv1 = nn.Conv2d(c5_ch, c4_ch, kernel_size=1, bias=False)

        # 4) reduce channels for residual addition
        fused_red_ch = max(1, c4_ch // 2)
        self.reshape = nn.Conv2d(c4_ch, fused_red_ch, kernel_size=1, bias=False)

        # 5) proj_fused: map fused_red -> out_c (always created & registered here)
        # ensures no dynamic module creation in forward
        if fused_red_ch != self.out_c:
            self.proj_fused = nn.Conv2d(fused_red_ch, self.out_c, kernel_size=1, bias=False)
        else:
            # identity mapping using 1x1 conv as well (keeps consistent type)
            self.proj_fused = nn.Identity()

        # 6) groups
        if self.out_c % self.G != 0:
            # fallback to G=1 if not divisible
            print(f"[CSDKM Warning] out_c {self.out_c} not divisible by G {self.G} -> set G=1")
            self.G = 1
        self.Cg = self.out_c // self.G

        # 7) similarity projections
        sim_dim = min(64, max(8, self.out_c // 4))
        self.sim_proj_c4 = nn.Conv2d(c4_ch, sim_dim, kernel_size=1, bias=False)
        self.sim_proj_c5 = nn.Conv2d(c5_ch, sim_dim, kernel_size=1, bias=False)

        # bilinear parameter if needed
        if self.sim_type == "bilinear" or (isinstance(self.sim_type, int) and int(self.sim_type) == 2):
            self.bilinear_W = nn.Parameter(torch.randn(sim_dim, sim_dim) * 0.01)

        # 8) kernel generator MLP (scalar -> K2)
        hidden = max(16, self.K2)
        self.kernel_gen = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.K2),
        )

        # register flag
        self._built = True

    # -------------- similarity computation (supports str or int) ----------------
    def compute_similarity(self, p4, p5):
        """
        p4, p5: (B, Csim, S, S)
        returns: (B, S, S)
        """
        mode = self.sim_type
        if isinstance(mode, (int, float)):
            mode = int(mode)
            if mode == 0:
                mode = "dot"
            elif mode == 1:
                mode = "cosine"
            elif mode == 2:
                mode = "bilinear"
            else:
                raise ValueError(f"Unknown sim_type code: {mode}")

        if mode == "dot":
            return (p4 * p5).sum(dim=1)

        elif mode == "cosine":
            eps = 1e-6
            dot = (p4 * p5).sum(dim=1)
            n4 = torch.sqrt((p4 * p4).sum(dim=1) + eps)
            n5 = torch.sqrt((p5 * p5).sum(dim=1) + eps)
            return dot / (n4 * n5 + eps)

        elif mode == "bilinear":
            B, C, Ss, _ = p4.shape
            p4f = p4.view(B, C, Ss * Ss)  # (B, C, S2)
            p5f = p5.view(B, C, Ss * Ss)
            p4T = p4f.permute(0, 2, 1)    # (B, S2, C)
            Wx = self.bilinear_W         # (C, C)
            tmp = torch.matmul(p4T, Wx)  # (B, S2, C)
            out = (tmp * p5f.permute(0, 2, 1)).sum(dim=-1)  # (B, S2)
            return out.view(B, Ss, Ss)

        else:
            raise ValueError(f"Unknown sim_type: {self.sim_type}")

    # ---------------------- patch index builder --------------------------------
    def build_patch_index(self, H, W, device):
        xs = torch.arange(W, device=device)
        ys = torch.arange(H, device=device)
        x_idx = (xs * self.S) // W
        y_idx = (ys * self.S) // H
        x_idx = torch.clamp(x_idx, 0, self.S - 1)
        y_idx = torch.clamp(y_idx, 0, self.S - 1)
        yy = y_idx.unsqueeze(1).repeat(1, W)  # (H,W)
        xx = x_idx.unsqueeze(0).repeat(H, 1)  # (H,W)
        patch_idx = (yy * self.S + xx).view(-1).long()  # (HW,)
        return patch_idx

    # ---------------------------- forward -------------------------------------
    def forward(self, inputs):
        """
        inputs: [c4, c5]
          c4: (B, c4_ch, H, W)  (shallower high-res)
          c5: (B, c5_ch, H2, W2) (deeper low-res)
        returns: (B, out_c, H, W)
        """
        assert isinstance(inputs, (list, tuple)) and len(inputs) == 2, "CSDKM expects [c4, c5]"

        c4, c5 = inputs
        B, c4_ch, H, W = c4.shape

        # lazy build with actual channel numbers
        if not self._built:
            self.build(c4_ch, c5.shape[1])

        # 1) upsample c5 to c4 spatial
        c5_up = F.interpolate(c5, size=(H, W), mode="nearest")

        # 2) lightweight C4 processing (preserve spatial dims)
        c4_proc = self.c4_proc_conv(c4)  # (B, c4_ch, H, W)

        # 3) map c5 to c4_ch if needed (we pooled to HxW already)
        c5_proc = F.adaptive_avg_pool2d(c5_up, output_size=(H, W))
        c5_proc = self.conv1(c5_proc)  # (B, c4_ch, H, W)

        fused = c4_proc + c5_proc  # (B, c4_ch, H, W)

        # to_fuse -> BN -> act
        X = self.act(self.norm(self.to_fuse(fused)))  # (B, out_c, H, W)

        # ------------------ SxS similarity ------------------
        # descriptors via adaptive avg pooling to stable SxS grid
        c4_desc = F.adaptive_avg_pool2d(c4, output_size=(self.S, self.S))
        c5_desc = F.adaptive_avg_pool2d(c5_up, output_size=(self.S, self.S))

        p4 = self.sim_proj_c4(c4_desc)  # (B, Csim, S, S)
        p5 = self.sim_proj_c5(c5_desc)

        sim = self.compute_similarity(p4, p5)  # (B, S, S)
        sim_flat = sim.view(B, self.S2)        # (B, S2)

        # ---------------- mask gating ---------------------
        mask = torch.sigmoid(self.mask_raw)     # (S2,)
        gated = sim_flat * mask.unsqueeze(0)    # (B, S2)

        # ---------------- kernel generation ----------------
        k_in = gated.unsqueeze(-1).view(B * self.S2, 1)  # (B*S2, 1)
        kernels = self.kernel_gen(k_in).view(B, self.S2, self.K2)  # (B, S2, K2)
        kernels = F.softmax(kernels, dim=-1)  # (B, S2, K2)

        # ---------------- patch-wise dynamic conv -------------
        Bx, Cx, Hx, Wx = X.shape
        patches = F.unfold(X, kernel_size=self.K, padding=self.pad)  # (B, out_c*K2, HW)
        HW = Hx * Wx
        patches = patches.view(B, self.G, self.Cg, self.K2, HW)  # (B, G, Cg, K2, HW)

        device = X.device
        patch_idx = self.build_patch_index(Hx, Wx, device)   # (HW,)

        # gather kernels per-location:
        # kernels: (B, S2, K2)
        # We need kernels_loc: (B, K2, HW)
        gather = patch_idx.unsqueeze(0).expand(B, -1)         # (B, HW)
        gatherK = gather.unsqueeze(-1).expand(B, HW, self.K2) # (B, HW, K2)

        # gather expects index on dim=1: returns (B, HW, K2)
        kernels_loc = torch.gather(kernels, dim=1, index=gatherK)  # (B, HW, K2)
        kernels_loc = kernels_loc.permute(0, 2, 1).contiguous()    # (B, K2, HW)
        kernels_loc = kernels_loc.unsqueeze(1).expand(-1, self.G, -1, -1)  # (B, G, K2, HW)

        # multiply and sum over K2
        out_grp = (patches * kernels_loc.unsqueeze(2)).sum(dim=3)  # (B, G, Cg, HW)
        out = out_grp.view(B, self.out_c, Hx, Wx)

        # ---------------- residual fusion (guaranteed registered proj_fused) --------------
        fused_red = self.reshape(fused)  # (B, fused_red_ch, H, W)
        fused_red = self.proj_fused(fused_red)  # registered module or Identity -> (B, out_c, H, W)

        out = out + fused_red
        return out

    def extra_repr(self):
        return f"out_c={self.out_c}, K={self.K}, G={self.G}, S={self.S}, sim_type={self.sim_type}"
