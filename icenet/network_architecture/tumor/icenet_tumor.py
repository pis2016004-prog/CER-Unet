from torch import nn
from typing import Tuple, Union
from icenet.network_architecture.neural_network import SegmentationNetwork
from icenet.network_architecture.dynunet_block import UnetOutBlock, UnetResBlock
from icenet.network_architecture.tumor.model_components import UnetrPPEncoder, UnetrUpBlock

# Drop-in replacement for your UNETR_PP that injects a Spatial MoE over fused skips
# Assumes you already have: SegmentationNetwork, UnetrPPEncoder, UnetResBlock,
# UnetrUpBlock, UnetOutBlock, and MoE implemented/imported.

from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """
    Expert module consisting of 1x1 convolutional layers with ReLU activation.
    
    Args:
        emb_size (int): Input embedding size.
        hidden_rate (int, optional): Multiplier for hidden layer size. Defaults to 2.
    """
    def __init__(self, emb_size: int, hidden_rate: int = 2):
        super().__init__()
        hidden_emb = hidden_rate * emb_size
        self.seq = nn.Sequential(
            nn.Conv2d(emb_size, hidden_emb, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(hidden_emb, hidden_emb, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hidden_emb),
            nn.ReLU(),
            nn.Conv2d(hidden_emb, emb_size, kernel_size=1, stride=1, padding=0, bias=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the expert network."""
        return self.seq(x)
class CapsuleExpert(nn.Module):
    """
    Capsule-based expert module using primary capsule logic followed by squashing.
    This replaces traditional Conv layers in the MoE expert.
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

    def squash(self, x, dim=1):
        squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1.0 + squared_norm)
        return scale * x / (torch.sqrt(squared_norm) + 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply each capsule conv in parallel
        caps_outputs = [caps(x) for caps in self.conv_capsules]  # list of [B, capsule_dim, H, W]
        u = torch.cat(caps_outputs, dim=1)  # [B, C, H, W]
        u = self.squash(u, dim=1)
        return self.output_proj(u)  # [B, in_channels, H, W]

class MoE(nn.Module):
    """
    Mixture of Experts (MoE) module with multiple gating mechanisms.
    
    Args:
        num_experts (int): Number of expert networks.
        top (int, optional): Number of top experts to select. Defaults to 2.
        emb_size (int, optional): Embedding dimension. Defaults to 128.
        H (int, optional): Input height. Defaults to 224.
        W (int, optional): Input width. Defaults to 224.
    """
    def __init__(self, num_experts: int, top: int = 2, emb_size: int = 128, H: int = 224, W: int = 224):
        super().__init__()
        #self.experts = #nn.ModuleList([Expert(emb_size) for _ in range(num_experts)]) # REPLACE THIS LINE:

        self.experts = nn.ModuleList([CapsuleExpert(emb_size) for _ in range(num_experts)])

        self.gate1 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        self.gate2 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        self.gate3 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        self.gate4 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        self._initialize_weights()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.top = top
        
    def _initialize_weights(self) -> None:
        """Initialize gate weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.gate1)
        nn.init.xavier_uniform_(self.gate2)
        nn.init.xavier_uniform_(self.gate3)
        nn.init.xavier_uniform_(self.gate4)
        
    def cv_squared(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the squared coefficient of variation.
        
        Used as a load balancing loss to encourage uniform expert usage.
        
        Args:
            x (torch.Tensor): Tensor of expert usage values.
            
        Returns:
            torch.Tensor: Squared coefficient of variation.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)
        
    def _process_gate(self, x: torch.Tensor, gate_weights: nn.Parameter) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input through a single gating mechanism.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, emb_size).
            gate_weights (nn.Parameter): Gate weights for this gating mechanism.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and load balancing loss.
        """
        batch_size, emb_size, H, W = x.shape
        
        # Compute gating probabilities
        x0 = self.gap(x).view(batch_size, emb_size)
        gate_out = F.softmax(x0 @ gate_weights, dim=1)
        
        # Calculate expert usage for load balancing
        expert_usage = gate_out.sum(0)
        
        # Select top-k experts
        top_weights, top_index = torch.topk(gate_out, self.top, dim=1)
        used_experts = torch.unique(top_index)
        unused_experts = set(range(len(self.experts))) - set(used_experts.tolist())
        
        # Apply softmax again for normalized weights
        top_weights = F.softmax(top_weights, dim=1)
        
        # Expand input for parallel expert processing
        x_expanded = x.unsqueeze(1).expand(batch_size, self.top, emb_size, H, W).reshape(-1, emb_size, H, W)
        y = torch.zeros_like(x_expanded)
        
        # Process each expert
        for expert_i, expert_model in enumerate(self.experts):
            expert_mask = (top_index == expert_i).view(-1)
            expert_indices = expert_mask.nonzero().flatten()
            
            if expert_indices.numel() > 0:
                x_expert = x_expanded[expert_indices]
                y_expert = expert_model(x_expert)
                y = y.index_add(dim=0, index=expert_indices, source=y_expert)
            elif expert_i in unused_experts and self.training:
                # Ensure all experts are used during training
                random_sample = torch.randint(0, x.size(0), (1,), device=x.device)
                x_expert = x_expanded[random_sample]
                y_expert = expert_model(x_expert)
                y = y.index_add(dim=0, index=random_sample, source=y_expert)
        
        # Apply weights and reshape
        top_weights = top_weights.view(-1, 1, 1, 1).expand_as(y)
        y = y * top_weights
        y = y.view(batch_size, self.top, emb_size, H, W).sum(dim=1)
        
        return y, self.cv_squared(expert_usage)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through all gating mechanisms.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, emb_size, H, W).
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                Four output tensors and combined load balancing loss.
        """
        #import pdb;pdb.set_trace()
        y1, loss1 = self._process_gate(x, self.gate1)
        y2, loss2 = self._process_gate(x, self.gate2)
        y3, loss3 = self._process_gate(x, self.gate3)
        y4, loss4 = self._process_gate(x, self.gate4)
        
        # Combine losses
        loss = loss1 + loss2 + loss3 + loss4
        
        #if self.training:
            #print(f"Expert Usage - Gate1: {self._format_usage([loss1,loss])}")
            #print(f"Expert Usage - Gate2: {self._format_usage([loss2,loss])}")
            #print(f"Expert Usage - Gate3: {self._format_usage([loss3,loss])}")
            #print(f"Expert Usage - Gate4: {self._format_usage([loss4,loss])}")
        
        return y1, y2, y3, y4, loss
    
    def _format_usage(self, usage: torch.Tensor) -> str:
        """Format expert usage statistics for logging."""
        return f"Min: {usage.min():.4f}, Max: {usage.max():.4f}, CV²: {self.cv_squared(usage):.4f}"

def count_parameters(model: nn.Module) -> str:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model.
        
    Returns:
        str: Formatted string with parameter count.
    """
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if params >= 1e6:
        return f"{params / 1e6:.2f}M parameters"
    elif params >= 1e3:
        return f"{params / 1e3:.2f}K parameters"
    else:
        return f"{params} parameters"

if __name__ == '__main__':
    """Unit test for MoE module."""
    try:
        # Initialize model
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MoE(num_experts=4, top=2, emb_size=128, H=224, W=224).to(device)
        model.train()
        
        # Generate random input
        emb = torch.randn(6, 128, 224, 224).to(device)
        
        # Forward pass
        out1, out2, out3, out4, loss = model(emb)
        
        # Verify output shapes
        assert out1.shape == emb.shape, f"Output shape mismatch: {out1.shape} vs {emb.shape}"
        assert out2.shape == emb.shape, f"Output shape mismatch: {out2.shape} vs {emb.shape}"
        assert out3.shape == emb.shape, f"Output shape mismatch: {out3.shape} vs {emb.shape}"
        assert out4.shape == emb.shape, f"Output shape mismatch: {out4.shape} vs {emb.shape}"
        
        print("\n=== MoE Module Test Passed ===")

        print(f"Input Shape: {emb.shape}")
        print(f"Output Shapes: {out1.shape}, {out2.shape}, {out3.shape}, {out4.shape}")
        print(f"Load Balancing Loss: {loss.item():.4f}")
        print(f"Model Parameters: {count_parameters(model)}")
        
    except Exception as e:
        print(f"Test failed: {e}")




# class Docker(nn.Module):
#     def __init__(self, in_ch, out_ch, time_downscale=1, spatial_downscale=1):
#         super(Docker, self).__init__()
#         self.time_downscale = time_downscale
#         self.spatial_downscale = spatial_downscale

#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         # x shape: [B*T, C, H, W] → reshape to [B, T, C, H, W]
#         B_T, C, H, W = x.shape
#         T = 16
#         B = B_T // T
        
#         # B = B_T // 16  # assuming fixed T=16; adjust as needed or infer dynamically
#         # T = 16
#         x = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]

#         # Apply 2D conv to each temporal slice
#         x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)  # [B*T, C, H, W]
#         x = self.conv(x)  # [B*T, out_ch, H, W]

#         # Reshape back and apply adaptive pooling
#         C_out = x.shape[1]
#         x = x.view(B, T, C_out, H, W).permute(0, 2, 1, 3, 4)  # [B, C_out, T, H, W]

#         # Dynamically compute target sizes
#         target_t = max(1, T // self.time_downscale)
#         target_h = max(1, H // self.spatial_downscale)
#         target_w = max(1, W // self.spatial_downscale)

#         x = F.adaptive_avg_pool3d(x, output_size=(target_t, target_h, target_w))
#         x_max = F.adaptive_max_pool3d(x, output_size=(target_t, target_h, target_w))
#         x = torch.cat([x_avg, x_max], dim=1) 
#         return x
class Docker(nn.Module):
    """
    Docker module with:
    - Multi-scale convolutions (1x1, 3x3, 5x5)
    - Dual-path pooling (adaptive avg + max)
    - Feature fusion via 3D conv

    Args:
        in_ch (int): Input channels.
        out_ch (int): Output channels after fusion.
        time_downscale (int): Temporal downscaling factor.
        spatial_downscale (int): Spatial downscaling factor.
    """
    def __init__(self, in_ch, out_ch, time_downscale=1, spatial_downscale=1):
        super().__init__()
        self.time_downscale = time_downscale
        self.spatial_downscale = spatial_downscale

        # Initial projection to reduce or align channels
        self.initial_proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # Multi-scale convs: kernel sizes 1, 3, 5
        self.multi_scale = nn.ModuleList([
            nn.Conv2d(out_ch, out_ch, kernel_size=1, padding=0),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.Conv2d(out_ch, out_ch, kernel_size=5, padding=2),
        ])

        # Fusion after dual pooling
        self.fuse = nn.Sequential(
            nn.Conv3d(out_ch * 2, out_ch, kernel_size=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*T, C, H, W]
        Returns:
            Tensor: [B, out_ch, T', H', W']
        """
        B_T, C, H, W = x.shape
        T = 16  # Or infer dynamically if needed
        B = B_T // T

        # Reshape and project channels
        x = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        x2d = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # [B*T, C, H, W]
        x_proj = self.initial_proj(x2d)

        # Apply multi-scale convolutions and sum
        multi_feats = [conv(x_proj) for conv in self.multi_scale]
        x_ms = sum(multi_feats)  # [B*T, out_ch, H, W]

        # Reshape back to 5D: [B, out_ch, T, H, W]
        x_5d = x_ms.view(B, T, -1, H, W).permute(0, 2, 1, 3, 4)

        # Pool sizes
        t_out = max(1, T // self.time_downscale)
        h_out = max(1, H // self.spatial_downscale)
        w_out = max(1, W // self.spatial_downscale)

        # Dual pooling
        x_avg = F.adaptive_avg_pool3d(x_5d, (t_out, h_out, w_out))
        x_max = F.adaptive_max_pool3d(x_5d, (t_out, h_out, w_out))

        # Fuse pooled features
        x_combined = torch.cat([x_avg, x_max], dim=1)  # [B, 2*out_ch, T', H', W']
        x_fused = self.fuse(x_combined)                # [B, out_ch, T', H', W']

        return x_fused
class UNETR_PP(SegmentationNetwork):
    """
    UNETR++ with a Spatial Mixture-of-Experts (MoE) applied over fused skip features.

    Pipeline (high-level):
      enc1/2/3 (3D) + enc4 (token sequence) -> align all skips to enc1's spatial size ->
      concat (C=256) -> slice along time (2D per-slice) -> MoE (per-slice) ->
      merge back to 3D -> 1x1x1 proj to hidden_size -> AdaptivePool to (4,4,4) ->
      use as dec4 input for the UNETR++ decoder path.

    Notes:
      • Returns (logits, moe_loss). If you don't use moe_loss, just ignore it.
      • No Docker layers are required; we use adaptive pooling for robust reshaping.
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
        do_ds: bool = True,
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
    ) -> None:
        super().__init__()

        if depths is None:
            depths = [3, 3, 3, 3]
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")
        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        # UNETR++ uses a last stage of 4x4x4 tokens by default for 64x128x128 inputs.
        # Keep this aligned with your encoder settings.
        self.feat_size = (4, 4, 4)
        self.hidden_size = hidden_size
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels

        # --- Encoder ---
        self.unetr_pp_encoder = UnetrPPEncoder(dims=dims, depths=depths, num_heads=num_heads)

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        # --- Decoder ---
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,  # 256 if feature_size=16
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
            upsample_kernel_size=(4, 4, 4),
            norm_name=norm_name,
            out_size=128 * 128 * 128,
            conv_decoder=True,
        )

        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

        # --- MoE over fused skips ---
        # We fuse 4 streams (enc1/2/3 and a 3D-upsampled projection of enc4 tokens), each to 64 ch -> 256 ch total.
        self.skip_c = 64
        self.conv_t1 = nn.Conv3d(feature_size, self.skip_c, kernel_size=1)         # enc1: 16 -> 64
        self.conv_t2 = nn.Conv3d(feature_size * 2, self.skip_c, kernel_size=1)     # enc2: 32 -> 64
        self.conv_t3 = nn.Conv3d(feature_size * 4, self.skip_c, kernel_size=1)     # enc3: 64 -> 64
        self.conv_t4 = nn.Conv1d(hidden_size, self.skip_c, kernel_size=1)          # enc4 tokens: 256 -> 64

        self.moe = MoE(num_experts=moe_num_experts, top=moe_top_k, emb_size=self.skip_c * 4)  # 256

        # Project MoE features (3D) to encoder hidden width and compress to (4,4,4)
        self.moe_proj3d = nn.Conv3d(self.skip_c * 1, hidden_size, kernel_size=1)   # from C2D (64) to hidden_size (256)

    @staticmethod
    def _proj_feat_tokens_to_3d(x_tokens: torch.Tensor, hidden_size: int, feat_size: Tuple[int, int, int]):
        """x_tokens: [B, N, hidden] -> [B, hidden, D, H, W] with N=D*H*W."""
        b, n, h = x_tokens.shape
        d, hh, w = feat_size
        assert n == d * hh * w, f"Token count mismatch: {n} vs {feat_size}"
        x = x_tokens.view(b, d, hh, w, h)  # [B, D, H, W, C]
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, D, H, W]
        return x

    @staticmethod
    def _proj_feat_3d_to_tokens(x_3d: torch.Tensor):
        """[B, C, D, H, W] -> [B, N, C] with N=D*H*W."""
        b, c, d, h, w = x_3d.shape
        x = x_3d.permute(0, 2, 3, 4, 1).contiguous().view(b, d * h * w, c)  # [B, N, C]
        return x

    def forward(self, x_in: torch.Tensor):
        # Encoder trunk
        x_output, hidden_states = self.unetr_pp_encoder(x_in)
        convBlock = self.encoder1(x_in)

        enc1, enc2, enc3, enc4_tokens = hidden_states  # enc4_tokens: [B, N, hidden_size]

        # Target 3D size for fusion = enc1 spatial size
        tgt_size = enc1.shape[2:]

        # Prepare 3D skip tensors, each -> [B, 64, *tgt_size]
        t1 = self.conv_t1(enc1)

        t2 = F.interpolate(enc2, size=tgt_size, mode="trilinear", align_corners=False)
        t2 = self.conv_t2(t2)

        t3 = self.conv_t3(enc3)
        t3 = F.interpolate(t3, size=tgt_size, mode="trilinear", align_corners=False)

        # enc4 tokens [B, N, hidden] -> [B, 64, D4, H4, W4] -> upsample to tgt_size
        b = enc4_tokens.size(0)
        n_tokens = enc4_tokens.size(1)
        d4, h4, w4 = self.feat_size
        assert n_tokens == d4 * h4 * w4, (
            f"enc4 token length {n_tokens} != feat_size product {d4*h4*w4}. "
            f"Adjust self.feat_size to match your encoder."
        )
        t4_proj = self.conv_t4(enc4_tokens.permute(0, 2, 1))  # [B, 64, N]
        t4_3d = t4_proj.view(b, self.skip_c, d4, h4, w4)
        t4 = F.interpolate(t4_3d, size=tgt_size, mode="trilinear", align_corners=False)

        # Fuse -> [B, 256, T, H, W]
        fused = torch.cat([t1, t2, t3, t4], dim=1)

        # Slice along time and send per-slice 2D maps to MoE
        B, C, T, H, W = fused.shape
        fused_2d = fused.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)

        # MoE returns 4 expert outputs (same spatial size) + aux loss
        o1, o2, o3, o4, moe_loss = self.moe(fused_2d)  # each: [B*T, 64, H, W]
        o_avg = (o1 + o2 + o3 + o4) / 4.0

        # Reassemble 3D volume: [B, 64, T, H, W]
        o_3d = o_avg.view(B, T, self.skip_c, H, W).permute(0, 2, 1, 3, 4).contiguous()

        # Project to hidden_size and compress to (4,4,4) to build dec4 input
        dec4_3d = self.moe_proj3d(o_3d)  # [B, hidden, T, H, W]
        dec4_3d = F.adaptive_avg_pool3d(dec4_3d, output_size=self.feat_size)  # [B, hidden, 4, 4, 4]

        # Standard UNETR++ decoding
        dec3 = self.decoder5(dec4_3d, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)

        out = self.decoder2(dec1, convBlock)
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)

        return logits, moe_loss

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

        self.feat_size = (4, 4, 4,)
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
            out_size=8*8*8,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=16*16*16,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=32*32*32,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(4, 4, 4),
            norm_name=norm_name,
            out_size=128*128*128,
            conv_decoder=True,
        )
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        #print("###########reached forward network")
        #print("XIN",x_in.shape)
        x_output, hidden_states = self.unetr_pp_encoder(x_in)
        convBlock = self.encoder1(x_in)

        # Four encoders
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]
        print(enc1.shape)
        print(enc2.shape)
        print(enc3.shape)
        print(enc4.shape)

        # Four decoders
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)

        out = self.decoder2(dec1, convBlock)
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)

        return logits
