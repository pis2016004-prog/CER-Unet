from torch import nn
from typing import Tuple, Union
from cernet.network_architecture.neural_network import SegmentationNetwork
from cernet.network_architecture.dynunet_block import UnetOutBlock, UnetResBlock
from cernet.network_architecture.synapse.model_components import UnetrPPEncoder, UnetrUpBlock

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, List, Set
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# ============================================================
# Capsule Expert
# ============================================================
class CapsuleExpert(nn.Module):
    """
    Capsule-inspired expert:
    - K parallel 3x3 convs
    - concat + squash
    - 1x1 projection back to in_channels
    """
    def __init__(self, in_channels: int, num_capsules: int = 16, capsule_dim: int = None):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim or (in_channels // num_capsules)

        self.conv_capsules = nn.ModuleList([
            nn.Conv2d(in_channels, self.capsule_dim, kernel_size=3, stride=1, padding=1)
            for _ in range(num_capsules)
        ])
        self.output_proj = nn.Conv2d(self.num_capsules * self.capsule_dim, in_channels, kernel_size=1)

    @staticmethod
    def squash(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1.0 + squared_norm)
        return scale * x / (torch.sqrt(squared_norm) + 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        caps_outputs = [caps(x) for caps in self.conv_capsules]   # each [B, cap_dim, H, W]
        u = torch.cat(caps_outputs, dim=1)                        # [B, K*cap_dim, H, W]
        u = self.squash(u, dim=1)
        return self.output_proj(u)                                # [B, in_channels, H, W]


# ============================================================
# CapMoE (4 gates, top-k routing, CV^2 balancing)
# ============================================================
class MoE(nn.Module):
    """
    CapMoE with 4 independent gates.
    Returns (y1,y2,y3,y4, loss_mean).
    """
    def __init__(self, num_experts: int, top: int = 2, emb_size: int = 256, num_gates: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.top = top
        self.emb_size = emb_size
        self.num_gates = num_gates

        self.experts = nn.ModuleList([CapsuleExpert(emb_size) for _ in range(num_experts)])
        self.gates = nn.ParameterList([nn.Parameter(torch.zeros(emb_size, num_experts)) for _ in range(num_gates)])

        for g in self.gates:
            nn.init.xavier_uniform_(g)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    @staticmethod
    def cv_squared(x: torch.Tensor) -> torch.Tensor:
        eps = 1e-10
        if x.numel() <= 1:
            return torch.zeros(1, device=x.device, dtype=x.dtype).squeeze()
        return x.float().var(unbiased=False) / (x.float().mean() ** 2 + eps)

    def _process_gate(self, x: torch.Tensor, gate_weights: nn.Parameter):
        """
        x: [B, C, H, W]
        gate_weights: [C, E]
        """
        B, C, H, W = x.shape

        x_gap = self.gap(x).view(B, C)                      # [B, C]
        gate_probs = F.softmax(x_gap @ gate_weights, dim=1)  # [B, E]

        expert_usage = gate_probs.sum(0)                    # [E]
        balance_loss = self.cv_squared(expert_usage)

        topk_weights, topk_indices = gate_probs.topk(self.top, dim=1)  # [B,k], [B,k]
        topk_weights = F.softmax(topk_weights, dim=1)

        out = torch.zeros_like(x)

        for k in range(self.top):
            expert_ids = topk_indices[:, k]              # [B]
            w = topk_weights[:, k].view(B, 1, 1, 1)      # [B,1,1,1]

            for expert_id in torch.unique(expert_ids):
                idx = (expert_ids == expert_id).nonzero(as_tuple=True)[0]
                if idx.numel() > 0:
                    x_chunk = x[idx]
                    y_chunk = self.experts[int(expert_id)](x_chunk)
                    out[idx] += y_chunk * w[idx]

        return out, balance_loss

    def forward(self, x: torch.Tensor):
        ys, losses = [], []
        for gate in self.gates:
            y, loss = self._process_gate(x, gate)
            ys.append(y)
            losses.append(loss)

        total_loss = torch.stack(losses).mean()
        return (*ys, total_loss)


# ============================================================
# OLD Docker (but parameterized by T, not hardcoded)
# ============================================================
class Docker(nn.Module):
    """
    OLD Docker:
    - 1x1 conv on each temporal slice
    - reshape to 5D
    - adaptive average pooling over (T,H,W)
    """
    def __init__(self, in_ch, out_ch, T: int, time_downscale=1, spatial_downscale=1):
        super(Docker, self).__init__()
        self.time_downscale = time_downscale
        self.spatial_downscale = spatial_downscale
        self.T = T

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B*T, C, H, W]
        returns: [B, out_ch, T', H', W']
        """
        B_T, C, H, W = x.shape
        T = self.T
        assert B_T % T == 0, f"Docker expects B*T divisible by T={T}, got B*T={B_T}"
        B = B_T // T

        x = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()  # [B,C,T,H,W]

        x2d = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)          # [B*T,C,H,W]
        x2d = self.conv(x2d)                                           # [B*T,out_ch,H,W]

        C_out = x2d.shape[1]
        x5d = x2d.view(B, T, C_out, H, W).permute(0, 2, 1, 3, 4)       # [B,out_ch,T,H,W]

        target_t = max(1, T // self.time_downscale)
        target_h = max(1, H // self.spatial_downscale)
        target_w = max(1, W // self.spatial_downscale)

        x5d = F.adaptive_avg_pool3d(x5d, output_size=(target_t, target_h, target_w))
        return x5d


# ============================================================
# UNETR_PP (SAME SIGNATURE AS ORIGINAL)
# ============================================================
class CERNET(SegmentationNetwork):
    """
    CERNET with CapMoE + OLD Docker.
    Uses Docker outputs as adaptive skip connections.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int] = (64, 128, 128),
        feature_size: int = 16,
        hidden_size: int = 256,
        num_heads: int = 4,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        dropout_rate: float = 0.0,
        depths=None,
        dims=None,
        conv_op=nn.Conv3d,
        do_ds: bool = True,
    ) -> None:
        super().__init__()

        if depths is None:
            depths = [3, 3, 3, 3]
        if dims is None:
            dims = [32, 64, 128, 256]

        self.do_ds = do_ds
        self.conv_op = conv_op
        self.hidden_size = hidden_size
        self.num_classes = out_channels

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")
        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        # --- original UNETR++ params (kept for compatibility) ---
        self.patch_size = (2, 4, 4)
        self.feat_size = (
            img_size[0] // self.patch_size[0] // 8,
            img_size[1] // self.patch_size[1] // 8,
            img_size[2] // self.patch_size[2] // 8,
        )

        # ---- We align skip fusion at enc1 scale:
        # ACDC example: img=(32,160,160) -> align=(16,40,40)
        # General: depth/2 and spatial/4
        self.align_size = (img_size[0] // 2, img_size[1] // 4, img_size[2] // 4)
        self.T_align = self.align_size[0]

        self.unetr_pp_encoder = UnetrPPEncoder(dims=dims, depths=depths, num_heads=num_heads)

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=8 * 8 * 8,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=16 * 16 * 16,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=32 * 32 * 32,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(2, 4, 4),
            norm_name=norm_name,
            out_size=64 * 128 * 128,
            conv_decoder=True,
        )

        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

        # ---- CapMoE expects 256-channel input ----
        self.moe = MoE(num_experts=4, top=2, emb_size=256, num_gates=4)

        # ---- Project enc1/enc2/enc3 to 64 each (64*4 = 256) ----
        self.conv3d_t1 = nn.Conv3d(dims[0], 64, kernel_size=1)  # 32 -> 64
        self.conv3d_t2 = nn.Conv3d(dims[1], 64, kernel_size=1)  # 64 -> 64
        self.conv3d_t3 = nn.Conv3d(dims[2], 64, kernel_size=1)  # 128 -> 64

        # Token projection (enc4 tokens)
        self.conv1d_t4 = nn.Conv1d(dims[3], 64, kernel_size=1)  # 256 -> 64

        # OLD Docker with correct T
        self.docker1 = Docker(256, 32,  T=self.T_align, time_downscale=1, spatial_downscale=1)
        self.docker2 = Docker(256, 64,  T=self.T_align, time_downscale=2, spatial_downscale=2)
        self.docker3 = Docker(256, 128, T=self.T_align, time_downscale=4, spatial_downscale=4)
        self.docker4 = Docker(256, 256, T=self.T_align, time_downscale=8, spatial_downscale=8)

    def forward(self, x_in: torch.Tensor):
        """
        x_in: [B, in_channels, D, H, W]
        returns: logits (+ deep supervision), moe_loss
        """
        _, hidden_states = self.unetr_pp_encoder(x_in)
        convBlock = self.encoder1(x_in)

        enc1, enc2, enc3, enc4 = hidden_states

        # ---- align enc1/enc2/enc3 to same size and project to 64ch ----
        t1 = F.interpolate(enc1, size=self.align_size, mode="trilinear", align_corners=False)
        t1 = self.conv3d_t1(t1)  # [B,64,T,H,W]

        t2 = F.interpolate(enc2, size=self.align_size, mode="trilinear", align_corners=False)
        t2 = self.conv3d_t2(t2)

        t3 = self.conv3d_t3(enc3)
        t3 = F.interpolate(t3, size=self.align_size, mode="trilinear", align_corners=False)

        # ---- enc4 tokens: [B,N,256] -> [B,64,N] -> [B,64,T,H,W] ----
        if enc4.dim() != 3:
            raise RuntimeError(f"enc4 expected [B,N,C], got {enc4.shape}")

        t4 = enc4.permute(0, 2, 1).contiguous()        # [B,256,N]
        t4 = self.conv1d_t4(t4)                        # [B,64,N]
        t4 = t4.unsqueeze(-1).unsqueeze(-1)            # [B,64,N,1,1]
        t4 = F.interpolate(t4, size=self.align_size, mode="trilinear", align_corners=False)

        # ---- fuse -> [B,256,T,H,W] ----
        fused = torch.cat([t1, t2, t3, t4], dim=1)

        # ---- flatten time into batch: [B*T,256,H,W] ----
        B, C, T, H, W = fused.shape
        fused2d = fused.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)

        o1, o2, o3, o4, moe_loss = self.moe(fused2d)

        # ---- docker -> adaptive multiscale skips ----
        y1 = self.docker1(o1)   # [B, 32, T,   H,   W  ]
        y2 = self.docker2(o2)   # [B, 64, T/2, H/2, W/2]
        y3 = self.docker3(o3)   # [B,128, T/4, H/4, W/4]
        y4 = self.docker4(o4)   # [B,256, T/8, H/8, W/8]

        # ---- decode with Docker skips (IMPORTANT) ----
        dec4 = y4
        dec3 = self.decoder5(dec4, y3)
        dec2 = self.decoder4(dec3, y2)
        dec1 = self.decoder3(dec2, y1)
        out  = self.decoder2(dec1, convBlock)

        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)

        return logits, moe_loss
