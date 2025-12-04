from torch import nn
from typing import Tuple, Union
from icenet.network_architecture.neural_network import SegmentationNetwork
from icenet.network_architecture.dynunet_block import UnetOutBlock, UnetResBlock
from icenet.network_architecture.lung.model_components import UnetrPPEncoder, UnetrUpBlock

import torch
import torch.nn as nn
import torch.nn.functional as F

############################
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

############################
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class CapsuleExpert(nn.Module):
    """
    Capsule-based expert module using primary capsule logic followed by squashing.
    """
    def __init__(self, in_channels: int, num_capsules: int = 16, capsule_dim: int = None, kernel_size: int = 3):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim or (in_channels // num_capsules)
        self.conv_capsules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, self.capsule_dim, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_capsules)
        ])
        self.output_proj = nn.Conv2d(self.num_capsules * self.capsule_dim, in_channels, kernel_size=1)

    def squash(self, x, dim=1):
        squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1.0 + squared_norm)
        return scale * x / (torch.sqrt(squared_norm + 1e-8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = torch.cat([caps(x) for caps in self.conv_capsules], dim=1)
        u = self.squash(u, dim=1)
        return self.output_proj(u)

class MoE(nn.Module):
    """
    Mixture of Experts (MoE) module with multiple gating mechanisms and Capsule-based experts.
    """
    def __init__(self, num_experts: int, top: int = 2, emb_size: int = 128, H: int = 224, W: int = 224):
        super().__init__()
        self.num_experts = num_experts
        self.top = top
        self.emb_size = emb_size
        self.experts = nn.ModuleList([CapsuleExpert(emb_size) for _ in range(num_experts)])
        self.gates = nn.ParameterList([nn.Parameter(torch.zeros(emb_size, num_experts)) for _ in range(4)])
        self._initialize_weights()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def _initialize_weights(self):
        for gate in self.gates:
            nn.init.xavier_uniform_(gate)

    def cv_squared(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _process_gate(self, x: torch.Tensor, gate_weights: nn.Parameter) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, emb_size, H, W = x.shape
        x_gap = self.gap(x).view(batch_size, emb_size)
        gate_probs = F.softmax(x_gap @ gate_weights, dim=1)

        topk_weights, topk_indices = gate_probs.topk(self.top, dim=1)
        topk_weights = F.softmax(topk_weights, dim=1)

        output = torch.zeros_like(x)
        expert_usage = gate_probs.sum(0)

        for k in range(self.top):
            expert_ids = topk_indices[:, k]
            weights = topk_weights[:, k].view(-1, 1, 1, 1)

            for expert_id in torch.unique(expert_ids):
                idx = (expert_ids == expert_id).nonzero(as_tuple=True)[0]
                if idx.numel() > 0:
                    x_chunk = x[idx]
                    y_chunk = self.experts[expert_id](x_chunk)
                    output[idx] += y_chunk * weights[idx]

        return output, self.cv_squared(expert_usage)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ys, losses = [], []
        for gate in self.gates:
            y, loss = self._process_gate(x, gate)
            ys.append(y)
            losses.append(loss)
        total_loss = torch.stack(losses).mean()
        return (*ys, total_loss)

def count_parameters(model: nn.Module) -> str:
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if params >= 1e6:
        return f"{params / 1e6:.2f}M parameters"
    elif params >= 1e3:
        return f"{params / 1e3:.2f}K parameters"
    else:
        return f"{params} parameters"

if __name__ == '__main__':
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MoE(num_experts=4, top=2, emb_size=128).to(device)
        model.train()

        emb = torch.randn(6, 128, 224, 224).to(device)
        out1, out2, out3, out4, loss = model(emb)

        assert out1.shape == emb.shape
        assert out2.shape == emb.shape
        assert out3.shape == emb.shape
        assert out4.shape == emb.shape

        print("\n=== MoE Module Test Passed ===")
        print(f"Input Shape: {emb.shape}")
        print(f"Output Shapes: {out1.shape}, {out2.shape}, {out3.shape}, {out4.shape}")
        print(f"Load Balancing Loss: {loss.item():.4f}")
        print(f"Model Parameters: {count_parameters(model)}")

    except Exception as e:
        print(f"Test failed: {e}")


class Docker(nn.Module):
    def __init__(self, in_ch, out_ch, time_downscale=1, spatial_downscale=1):
        super(Docker, self).__init__()
        self.time_downscale = time_downscale
        self.spatial_downscale = spatial_downscale

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x shape: [B*T, C, H, W] → reshape to [B, T, C, H, W]
        B_T, C, H, W = x.shape
        B = B_T // 16  # assuming fixed T=16; adjust as needed or infer dynamically
        T = 16
        x = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]

        # Apply 2D conv to each temporal slice
        x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)  # [B*T, C, H, W]
        x = self.conv(x)  # [B*T, out_ch, H, W]

        # Reshape back and apply adaptive pooling
        C_out = x.shape[1]
        x = x.view(B, T, C_out, H, W).permute(0, 2, 1, 3, 4)  # [B, C_out, T, H, W]

        # Dynamically compute target sizes
        target_t = max(1, T // self.time_downscale)
        target_h = max(1, H // self.spatial_downscale)
        target_w = max(1, W // self.spatial_downscale)

        x = F.adaptive_avg_pool3d(x, output_size=(target_t, target_h, target_w))
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F

class Docker(nn.Module):
    def __init__(self, in_ch, out_ch, time_downscale=1, spatial_downscale=1, target_size=None):
        """
        target_size: optional tuple (t, h, w). If provided, overrides *_downscale
        """
        super().__init__()
        self.time_downscale = max(1, int(time_downscale))
        self.spatial_downscale = max(1, int(spatial_downscale))
        self.target_size = target_size  # e.g. (4, 6, 6)

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _to_BCTHW(self, x, T=None):
        # x can be [B, T, C, H, W] or [B, C, T, H, W] or [B*T, C, H, W] (then T must be given)
        if x.dim() == 5:
            if x.shape[1] < x.shape[2]:  # assume [B, T, C, H, W]
                B, T, C, H, W = x.shape
                x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]
            else:  # already [B, C, T, H, W]
                B, C, T, H, W = x.shape
        elif x.dim() == 4:
            if T is None:
                raise ValueError("When input is [B*T, C, H, W], you must provide T.")
            B_T, C, H, W = x.shape
            if B_T % T != 0:
                raise ValueError(f"B*T={B_T} not divisible by T={T}.")
            B = B_T // T
            x = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        else:
            raise ValueError("Expected input of shape [B,T,C,H,W], [B,C,T,H,W] or [B*T,C,H,W].")
        return x  # [B, C, T, H, W]

    def forward(self, x, T=None):
        # Normalize to [B, C, T, H, W]
        T=16
        x = self._to_BCTHW(x, T=T)
        B, C, T, H, W = x.shape

        # Apply 2D conv per frame
        x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        x = self.conv(x)
        C_out = x.shape[1]
        x = x.view(B, T, C_out, H, W).permute(0, 2, 1, 3, 4)  # [B, C_out, T, H, W]
        
        # Decide target size
        if self.target_size is not None:
            target_t, target_h, target_w = self.target_size
        else:
            target_t = max(1, T // self.time_downscale)
            target_h = max(1, H // self.spatial_downscale)
            target_w = max(1, W // self.spatial_downscale)

        # Pool to target
        x = F.adaptive_avg_pool3d(x, output_size=(target_t, target_h, target_w))
        #x= x.view()
        return x  # [B, C_out, target_t, target_h, target_w]

####################################################
# UNETR_PP with SpatialMoE3D Skip Routing
####################################################
def reshape_to_volume(x, out_channels, out_shape=(16, 40, 40)):
    """
    Projects a 1D or 3D feature to 3D volume (T, H, W)
    x: Tensor [B, C, L] or [B, C, T', H', W']
    Returns: [B, out_channels, T, H, W]
    """
    if x.ndim == 3:  # [B, C, L]
        x = x.unsqueeze(-1).unsqueeze(-1)  # → [B, C, L, 1, 1]
    x = F.interpolate(x, size=out_shape, mode='trilinear', align_corners=False)
    return x


class UNETR_PP(SegmentationNetwork):
    """
    UNETR++ based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            feature_size: int = 16,
            hidden_size: int = 256,
            num_heads: int = 4,
            pos_embed: str = "perceptron",
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            depths=None,
            dims=None,
            conv_op=nn.Conv3d,
            do_ds=True,

    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimensions of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.
        """

        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.feat_size = (4, 6, 6,)
        self.hidden_size = hidden_size

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
            out_size=8*12*12,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=16*24*24,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=32*48*48,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(1, 4, 4),
            norm_name=norm_name,
            out_size=32*192*192,
            conv_decoder=True,
        )
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)
        self.moe = MoE(num_experts=4, top=2, emb_size=256)

        # 1x1 convs to adjust intermediate channels
        self.conv3_t3 = nn.Conv3d(128, 64, kernel_size=1)
        self.conv1d_t4 = nn.Conv1d(256, 16, kernel_size=1)
        self.fc_t4 = nn.Linear(50, 16 * 40 * 40)

        
        # self.docker1 = self._create_docker(64, 32, time_dim=16, spatial_downscale=1)
        # self.docker2 = self._create_docker(64, 64, time_dim=8, spatial_downscale=2)
        # self.docker3 = self._create_docker(64, 128, time_dim=4, spatial_downscale=4)
        # self.docker4 = self._create_docker(64, 256, time_dim=2, spatial_downscale=8)
        
        # self.docker1 = self._create_docker(256, 32, time_downscale=1, spatial_downscale=1)  # → 32 → 16
        # self.docker2 = self._create_docker(256, 64, time_downscale=2, spatial_downscale=2)  # → 32 → 8
        # self.docker3 = self._create_docker(256, 128, time_downscale=4, spatial_downscale=4) # → 32 → 4
        # self.docker4 = self._create_docker(256, 256, time_downscale=8, spatial_downscale=8) # → 32 → 2
        self.conv3d_t1 = nn.Conv3d(32, 64, kernel_size=1)
        self.conv1d_t4 = nn.Conv1d(256, 64, kernel_size=1)  # Reduce channel dim
        self.conv3d_t2 = nn.Conv3d(64, 64, kernel_size=1)  # in __init__()

        # self.docker1 = Docker(256,  32, time_downscale=1, spatial_downscale=1)  # -> (1, 32, 6, 40, 40)
        # self.docker2 = Docker(256,  64, time_downscale=2, spatial_downscale=2)  # -> (1, 64, 3, 20, 20)
        # self.docker3 = Docker(256, 128, time_downscale=4, spatial_downscale=4)  # -> (1,128, 1, 10, 10)
        # self.docker4 = Docker(256, 256, time_downscale=8, spatial_downscale=8)  # -> (1,256, 1,  5,  5)

        # self.docker1 = Docker(256,  32, target_size=(4,  6,  6))
        # self.docker2 = Docker(256,  64, target_size=(2, 12, 12))
        # self.docker3 = Docker(256, 128, target_size=(1, 12, 12))
        # self.docker4 = Docker(256, 256, target_size=(1,  6,  6))

        # exact targets
        self.docker4 = Docker(256, 256, target_size=(4,  6,  6))   # y4:  (1,256, 4,  6,  6)
        self.docker3 = Docker(256, 128, target_size=(8, 12, 12))   # y3:  (1,128, 8, 12, 12)
        self.docker2 = Docker(256,  64, target_size=(16,24, 24))   # y2:  (1, 64,16, 20, 20)
        self.docker1 = Docker(256,  32, target_size=(32,48, 48))   # y1:  (1, 32,16, 40, 40)


    # def _create_docker(self, in_ch, out_ch, time_dim, spatial_downscale=1):
    #     return Docker(in_ch, out_ch, time_dim, spatial_downscale)
    def _create_docker(self, in_ch, out_ch, time_downscale=1, spatial_downscale=1):
        return Docker(in_ch, out_ch, time_downscale, spatial_downscale)


    # def proj_feat(self, x, hidden_size, feat_size):
    #     x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
    #     x = x.permute(0, 4, 1, 2, 3).contiguous()
    #     return x
    def proj_feat(self, x, hidden_size, feat_size):
        if x.shape[1] != hidden_size:
            # Assume x is [B, D, H, W, C]
            x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x




    def forward(self, x_in):
        #print("#####input_shape:", x_in.shape)
        x_output, hidden_states = self.unetr_pp_encoder(x_in)

        convBlock = self.encoder1(x_in)
        print(convBlock.shape)
        enc1, enc2, enc3, enc4 = hidden_states
        
        # print("#####enc1_shape:", enc1.shape)#torch.Size([1, 32, 32, 48, 48])
        # print("#####enc2_shape:", enc2.shape)#torch.Size([1, 64, 16, 24, 24])
        # print("#####enc3_shape:", enc3.shape)#torch.Size([1, 128, 8, 12, 12])
        # print("#####enc4_shape:", enc4.shape)#torch.Size([1, 144, 256])
        t1 = self.conv3d_t1(enc1)  # [B, 64, T, H, W]
        t1 = F.interpolate(t1, size=(16, 40, 40), mode='trilinear', align_corners=False)

        t2 = self.conv3d_t2(F.interpolate(enc2, size=(16, 40, 40), mode='trilinear', align_corners=False))
        t3 = self.conv3_t3(enc3)
        t3 = F.interpolate(t3, size=(16, 40, 40), mode='trilinear', align_corners=False)

        # For enc4: [B, 50, 256] → [B, 256, 50] → Conv1D → reshape
        t4 = enc4.permute(0, 2, 1)            # [B, 256, 50]
        t4 = self.conv1d_t4(t4)              # [B, 64, 50]
        t4 = reshape_to_volume(t4, out_channels=64)  # [B, 64, 16, 40, 40]

        # print("#####t1_shape:", t1.shape)# torch.Size([1, 64, 16, 40, 40])
        # print("#####t2_shape:", t2.shape) # torch.Size([1, 64, 16, 40, 40])        
        # print("#####t3_shape:", t3.shape) #torch.Size([1, 64, 16, 40, 40])
        # print("#####t4_shape:", t4.shape) #torch.Size([1, 64, 16, 40, 40])
        fused = torch.cat([t1, t2, t3, t4], dim=1)  # [B, 64*4, 16, 40, 40]
        # print("#####fused_shape:", fused.shape)

        B, C, T, H, W = fused.shape
        fused = fused.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        #print(fused.shape)
        o1, o2, o3, o4, loss = self.moe(fused)  # each (32, 64, 40, 40)
        # print(o1.shape)
        # print(o2.shape)
        # print(o3.shape)
        # print(o4.shape)
        # === Docker transformations ===
        y1 = self.docker1(o1)  # (1,  32, 16, 40, 40)
        y2 = self.docker2(o2)  # (1,  64,  8, 20, 20)
        y3 = self.docker3(o3)  # (1, 128,  4, 10, 10)
        y4 = self.docker4(o4)  # (1, 256,  2,  5,  5)
        # y1 = self.docker1(o1, T=16)
        # y2 = self.docker2(o1, T=16)
        # y3 = self.docker3(o1, T=16)
        # y4 = self.docker4(o1, T=16)
  

        # print("y4",y4.shape)
        # print("y3",y3.shape)
        # print("y2",y2.shape)
        # print("y1",y1.shape)
        
         
        
        # print(y3.shape)
        # print(y4.shape)

        # === Decode ===
        dec4 = self.proj_feat(y4, self.hidden_size, self.feat_size)
        #print("dec4",dec4.shape)
        dec3 = self.decoder5(dec4, y3)
        #print("dec4",dec3.shape)
        dec2 = self.decoder4(dec3, y2)
        #print("dec4",dec2.shape)
        dec1 = self.decoder3(dec2, y1)
        #print("dec1",dec1.shape)
        out = self.decoder2(dec1, convBlock)


        #print(out.shape)

        
        #print("out",out.shape)

        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)

        return logits, loss

# class UNETR_PP(SegmentationNetwork):
#     """
#     UNETR++ based on: "Shaker et al.,
#     UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
#     """
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: int,
#             feature_size: int = 16,
#             hidden_size: int = 256,
#             num_heads: int = 4,
#             pos_embed: str = "perceptron",
#             norm_name: Union[Tuple, str] = "instance",
#             dropout_rate: float = 0.0,
#             depths=None,
#             dims=None,
#             conv_op=nn.Conv3d,
#             do_ds=True,

#     ) -> None:
#         """
#         Args:
#             in_channels: dimension of input channels.
#             out_channels: dimension of output channels.
#             img_size: dimension of input image.
#             feature_size: dimension of network feature size.
#             hidden_size: dimensions of  the last encoder.
#             num_heads: number of attention heads.
#             pos_embed: position embedding layer type.
#             norm_name: feature normalization type and arguments.
#             dropout_rate: faction of the input units to drop.
#             depths: number of blocks for each stage.
#             dims: number of channel maps for the stages.
#             conv_op: type of convolution operation.
#             do_ds: use deep supervision to compute the loss.
#         """

#         super().__init__()
#         if depths is None:
#             depths = [3, 3, 3, 3]
#         self.do_ds = do_ds
#         self.conv_op = conv_op
#         self.num_classes = out_channels
#         if not (0 <= dropout_rate <= 1):
#             raise AssertionError("dropout_rate should be between 0 and 1.")

#         if pos_embed not in ["conv", "perceptron"]:
#             raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

#         self.feat_size = (4, 6, 6,)
#         self.hidden_size = hidden_size

#         self.unetr_pp_encoder = UnetrPPEncoder(dims=dims, depths=depths, num_heads=num_heads)

#         self.encoder1 = UnetResBlock(
#             spatial_dims=3,
#             in_channels=in_channels,
#             out_channels=feature_size,
#             kernel_size=3,
#             stride=1,
#             norm_name=norm_name,
#         )
#         self.decoder5 = UnetrUpBlock(
#             spatial_dims=3,
#             in_channels=feature_size * 16,
#             out_channels=feature_size * 8,
#             kernel_size=3,
#             upsample_kernel_size=2,
#             norm_name=norm_name,
#             out_size=8*12*12,
#         )
#         self.decoder4 = UnetrUpBlock(
#             spatial_dims=3,
#             in_channels=feature_size * 8,
#             out_channels=feature_size * 4,
#             kernel_size=3,
#             upsample_kernel_size=2,
#             norm_name=norm_name,
#             out_size=16*24*24,
#         )
#         self.decoder3 = UnetrUpBlock(
#             spatial_dims=3,
#             in_channels=feature_size * 4,
#             out_channels=feature_size * 2,
#             kernel_size=3,
#             upsample_kernel_size=2,
#             norm_name=norm_name,
#             out_size=32*48*48,
#         )
#         self.decoder2 = UnetrUpBlock(
#             spatial_dims=3,
#             in_channels=feature_size * 2,
#             out_channels=feature_size,
#             kernel_size=3,
#             upsample_kernel_size=(1, 4, 4),
#             norm_name=norm_name,
#             out_size=32*192*192,
#             conv_decoder=True,
#         )
#         self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
#         if self.do_ds:
#             self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
#             self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

#     def proj_feat(self, x, hidden_size, feat_size):
#         x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
#         x = x.permute(0, 4, 1, 2, 3).contiguous()
#         return x

#     def forward(self, x_in):
#         #print("#####input_shape:", x_in.shape)
#         x_output, hidden_states = self.unetr_pp_encoder(x_in)

#         convBlock = self.encoder1(x_in)
#         print(convBlock.shape)
#         # Four encoders
#         enc1 = hidden_states[0]
#         #print("ENC1:",enc1.shape)
#         enc2 = hidden_states[1]
#         #print("ENC2:",enc2.shape)
#         enc3 = hidden_states[2]
#         #print("ENC3:",enc3.shape)
#         enc4 = hidden_states[3]
#         #print("ENC4:",enc4.shape)

#         # Four decoders
#         dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
#         dec3 = self.decoder5(dec4, enc3)
#         dec2 = self.decoder4(dec3, enc2)
#         dec1 = self.decoder3(dec2, enc1)
#         print(dec1.shape)

#         out = self.decoder2(dec1, convBlock)
#         print(out.shape)
#         if self.do_ds:
#             logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
#         else:
#             logits = self.out1(out)
#         loss = None
#         return logits, loss