from torch import nn
from typing import Tuple, Union
from cernet.network_architecture.neural_network import SegmentationNetwork
from cernet.network_architecture.dynunet_block import UnetOutBlock, UnetResBlock
from cernet.network_architecture.acdc.model_components import UnetrPPEncoder, UnetrUpBlock
import os
import torch
from torch.nn import functional as F
from typing import Tuple, List, Set


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

class UNETR_PP(SegmentationNetwork):
    def __init__(self, in_channels, out_channels, feature_size=16, hidden_size=256, num_heads=4,
                 pos_embed="perceptron", norm_name="instance", dropout_rate=0.0, depths=None, dims=None,
                 conv_op=nn.Conv3d, do_ds=True):
        super().__init__()

        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        self.feat_size = (2, 5, 5)
        self.hidden_size = hidden_size
        self.dims = dims

        self.unetr_pp_encoder = UnetrPPEncoder(dims=dims, depths=depths, num_heads=num_heads)
        self.encoder1 = UnetResBlock(spatial_dims=3, in_channels=in_channels, out_channels=feature_size,
                                     kernel_size=3, stride=1, norm_name=norm_name)

        self.decoder5 = UnetrUpBlock(spatial_dims=3, in_channels=feature_size * 16, out_channels=feature_size * 8,
                                     kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, out_size=4 * 10 * 10)
        self.decoder4 = UnetrUpBlock(spatial_dims=3, in_channels=feature_size * 8, out_channels=feature_size * 4,
                                     kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, out_size=8 * 20 * 20)
        self.decoder3 = UnetrUpBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=feature_size * 2,
                                     kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, out_size=16 * 40 * 40)
        self.decoder2 = UnetrUpBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=feature_size,
                                     kernel_size=3, upsample_kernel_size=(1, 4, 4), norm_name=norm_name,
                                     out_size=16 * 160 * 160, conv_decoder=True)

        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

        self.moe = MoE(num_experts=4, top=2, emb_size=256)

        # 1x1 convs to adjust intermediate channels
        self.conv3_t3 = nn.Conv3d(128, 64, kernel_size=1)
        self.conv1d_t4 = nn.Conv1d(256, 16, kernel_size=1)
        self.fc_t4 = nn.Linear(50, 16 * 40 * 40)

        
        
        self.docker1 = self._create_docker(256, 32, time_downscale=1, spatial_downscale=1)  # → 32 → 16
        self.docker2 = self._create_docker(256, 64, time_downscale=2, spatial_downscale=2)  # → 32 → 8
        self.docker3 = self._create_docker(256, 128, time_downscale=4, spatial_downscale=4) # → 32 → 4
        self.docker4 = self._create_docker(256, 256, time_downscale=8, spatial_downscale=8) # → 32 → 2
        self.conv3d_t1 = nn.Conv3d(32, 64, kernel_size=1)
        self.conv1d_t4 = nn.Conv1d(256, 64, kernel_size=1)  # Reduce channel dim
        self.conv3d_t2 = nn.Conv3d(64, 64, kernel_size=1)  # in __init__()



    def _create_docker(self, in_ch, out_ch, time_downscale=1, spatial_downscale=1):
        return Docker(in_ch, out_ch, time_downscale, spatial_downscale)


 
    def proj_feat(self, x, hidden_size, feat_size):
        if x.shape[1] != hidden_size:
            # Assume x is [B, D, H, W, C]
            x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x



    def forward(self, x_in):
        x_output, hidden_states = self.unetr_pp_encoder(x_in)
        convBlock = self.encoder1(x_in)
        enc1, enc2, enc3, enc4 = hidden_states

        # === Prepare t1, t2, t3, t4 ===
        t1 = enc1  # (1, 16, 16, 40, 40)
      
        t1 = self.conv3d_t1(t1)
       
       
        t2_up = F.interpolate(enc2, size=(16, 40, 40), mode='trilinear', align_corners=False)
        t2_up = self.conv3d_t2(t2_up)  # → [1, 16, 16, 40, 40]

        t3_proj = self.conv3_t3(enc3)
        t3_up = F.interpolate(t3_proj, size=(16, 40, 40), mode='trilinear', align_corners=False)

     
        t4_trans = enc4.permute(0, 2, 1)                  # [B, 256, 50]
        t4_reduced = self.conv1d_t4(t4_trans)             # [B, 16, 50]

        # Add dummy spatial dims to make it 5D
        t4_expanded = t4_reduced.unsqueeze(-1).unsqueeze(-1)  # [B, 16, 50, 1, 1]

        # Interpolate to desired shape (T=16, H=40, W=40)
        t4_reshaped = F.interpolate(
            t4_expanded, size=(16, 40, 40), mode='trilinear', align_corners=False
        )  # → [B, 16, 16, 40, 40]

       

        #t4_reshaped = t4_flat.view(1, 16, 16, 40, 40)
        # === Fuse all ===
        fused = torch.cat([t1, t2_up, t3_up, t4_reshaped], dim=1)  # [1, 64, 16, 40, 40]
        # fused = fused.view(32, 64, 40, 40)                         # flatten to 2D slices
        B, C, T, H, W = fused.shape
        fused = fused.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        o1, o2, o3, o4, loss = self.moe(fused)  # each (32, 64, 40, 40)
       

        # === Docker transformations ===
        y1 = self.docker1(o1)  # (1,  32, 16, 40, 40)
        y2 = self.docker2(o2)  # (1,  64,  8, 20, 20)
        y3 = self.docker3(o3)  # (1, 128,  4, 10, 10)
        y4 = self.docker4(o4)  # (1, 256,  2,  5,  5)
       

   

        # === Decode ===
        dec4 = self.proj_feat(y4, self.hidden_size, self.feat_size)
        #print("dec4",dec4.shape)
        dec3 = self.decoder5(dec4, y3)
        dec2 = self.decoder4(dec3, y2)
        dec1 = self.decoder3(dec2, y1)
        out = self.decoder2(dec1, convBlock)

        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)

        return logits, loss



