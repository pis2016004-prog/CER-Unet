from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveAvgMaxPool3d(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(output_size)
        self.maxpool = nn.AdaptiveMaxPool3d(output_size)
    def forward(self, x):
        return torch.cat([self.avgpool(x), self.maxpool(x)], dim=1)

# class Conv2p5D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=None):
#         super().__init__()
#         if padding is None:
#             padding = kernel_size // 2
#         self.conv2d = nn.Conv3d(in_channels, out_channels, kernel_size=(1, kernel_size, kernel_size),
#                                 padding=(0, padding, padding))
#         self.conv1d = nn.Conv3d(out_channels, out_channels, kernel_size=(kernel_size, 1, 1),
#                                 padding=(kernel_size//2, 0, 0))
#     def forward(self, x):
#         x = self.conv2d(x)
#         x = self.conv1d(x)
#         return x


# class Inception3D(nn.Module):
#     def __init__(self, in_channels, c1, c2, c3, c4, stride=1):
#         super().__init__()
        
#         self.p1_1 = Conv2p5D(in_channels, c1, kernel_size=1) # 2.5D conv for 1x1 kernel
#         self.p2_1 = Conv2p5D(in_channels, c2[0], kernel_size=1)
#         self.p2_2 = Conv2p5D(c2[0], c2[1], kernel_size=3)
#         self.p3_1 = Conv2p5D(in_channels, c3[0], kernel_size=1)
#         self.p3_2 = Conv2p5D(c3[0], c3[1], kernel_size=5, padding=2)
        
#         self.p4_1 = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
#         self.p4_2= Conv2p5D(in_channels, c4, kernel_size=1)  # 2.5D conv for pooling branch
#         # for pooling branch, you may use nn.AdaptiveAvgPool3d or reduce to 2.5D
        
#     def forward(self, x):
#        #print(f"Input shape: {x.shape}")
#         p1 = F.relu(self.p1_1(x))
#         p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
#         p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
#         p4 = F.relu(self.p4_2(self.p4_1(x)))
       
#         # pooling/skip branch as before, followed by upsampling if needed
#         # align all sizes
#         target_shape = p1.shape[2:]
#         p1 = F.interpolate(p1, size=(p2.shape[2], p2.shape[3], p2.shape[4]), mode="trilinear", align_corners=False)
#         p2 = F.interpolate(p2, size=target_shape, mode="trilinear", align_corners=False)
#         p3 = F.interpolate(p3, size=target_shape, mode="trilinear", align_corners=False)
#         p4= F.interpolate(p4,size=target_shape, mode="trilinear", align_corners=False)
#         # # add pooling branch if used, then concatenate
#         concat = torch.cat([p1, p2, p3, p4], dim=1) 
       
             
#         return concat
class Conv2p5D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        
        # CHANNEL REDUCTION POINT (1x1x1 conv)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        
        # Depthwise spatial conv
        self.depthwise_spatial = nn.Conv3d(
            out_channels, out_channels, 
            kernel_size=(1, kernel_size, kernel_size),
            padding=(0, padding, padding),
            groups=out_channels  # No channel increase
        )
        
        # Depthwise temporal conv
        self.depthwise_temporal = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(padding, 0, 0),
            groups=out_channels  # No channel increase
        )

    def forward(self, x):
        x = self.pointwise(x)  # Channel reduction happens here
        x = self.depthwise_spatial(x)
        x = self.depthwise_temporal(x)
        return x
class Inception3D(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, stride=1):
        super().__init__()
        # Branch 1: Single 1x1 convolution
        self.p1_1 = Conv2p5D(in_channels, c1, kernel_size=1)
        
        # Branch 2: 1x1 conv followed by 3x3 conv
        self.p2_1 = Conv2p5D(in_channels, c2[0], kernel_size=1)
        self.p2_2 = Conv2p5D(c2[0], c2[1], kernel_size=3)
        
        # Branch 3: 1x1 conv followed by 5x5 conv
        self.p3_1 = Conv2p5D(in_channels, c3[0], kernel_size=1)
        self.p3_2 = Conv2p5D(c3[0], c3[1], kernel_size=5, padding=2)
        
        # Branch 4: Max pooling followed by 1x1 conv
        self.p4_1 = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = Conv2p5D(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        
        # Directly concatenate without interpolation
        return torch.cat([p1, p2, p3, p4], dim=1)

class UnetResBlock(nn.Module):
    """
    Inception-style residual block with downsampling support.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()

        assert spatial_dims == 3, "This Inception version only supports 3D."

        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True

        # Inception splits: total channels must sum to out_channels
        c1 = out_channels // 4
        c2 = (out_channels // 8, out_channels // 4)
        c3 = (out_channels // 8, out_channels // 4)
        c4 = out_channels // 4

        self.inception = Inception3D(in_channels, c1, c2, c3, c4, stride=1)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=3, channels=out_channels)
        self.act = get_act_layer(name=act_name)

        # Optional second Inception (as second conv) with stride=1
        self.inception2 = get_conv_layer(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, conv_only=True)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=3, channels=out_channels)

        # Residual path
        if self.downsample:
            self.res_inception = Inception3D(in_channels, c1, c2, c3, c4, stride=stride)
            self.norm_res = get_norm_layer(name=norm_name, spatial_dims=3, channels=out_channels)

    def forward(self, x):
        residual = x

        out = self.inception(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.inception2(out)
        out = self.norm2(out)

        if self.downsample:
            residual = self.res_inception(residual)
            residual = self.norm_res(residual)

        out += residual
        out = self.act(out)
        return out
class UnetBasicBlock(nn.Module):
    """
    A CNN module module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, conv_only=True
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        return out


class UnetUpBlock(nn.Module):
    """
    An upsampling module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        upsample_kernel_size: convolution kernel size for transposed convolution layers.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
        trans_bias: transposed convolution bias.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
        trans_bias: bool = False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            dropout=dropout,
            bias=trans_bias,
            conv_only=True,
            is_transposed=True,
        )
        self.conv_block = UnetBasicBlock(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
        )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class UnetOutBlock(nn.Module):
    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, dropout: Optional[Union[Tuple, str, float]] = None
    ):
        super().__init__()
        self.conv = get_conv_layer(
            spatial_dims, in_channels, out_channels, kernel_size=1, stride=1, dropout=dropout, bias=True, conv_only=True
        )

    def forward(self, inp):
        return self.conv(inp)


def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
    stride: Union[Sequence[int], int] = 1,
    act: Optional[Union[Tuple, str]] = Act.PRELU,
    norm: Union[Tuple, str] = Norm.INSTANCE,
    dropout: Optional[Union[Tuple, str, float]] = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


def get_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:

    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative, please change the kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int], padding: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding value should not be negative, please change the kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]
