from pytorchvideo.layers.swish import Swish
from pytorchvideo.layers.utils import round_width, round_repeats

# from pytorchvideo.models.net import Net
# from pytorchvideo.models.net import DetectionBBoxNetwork
from models.models_utils import create_res_roi_pooling_head, Net, DetectionBBoxNetwork
import math
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from utils.utils import initialize_weights
from pytorchvideo.models.x3d import (
    create_x3d_bottleneck_block,
    create_x3d_stem,
    create_x3d_res_stage,
    create_x3d_head,
    ProjectedPool,
)

from utils.utils import load_state_dict_flexible


def create_x3d(
    *,
    # Input clip configs.
    input_channel: int = 3,
    input_clip_length: int = 13,
    input_crop_size: int = 160,
    # Model configs.
    model_num_class: int = 400,
    dropout_rate: float = 0.5,
    width_factor: float = 2.0,
    depth_factor: float = 2.2,
    # Normalization configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    # Activation configs.
    activation: Callable = nn.ReLU,
    # Stem configs.
    stem_dim_in: int = 12,
    stem_conv_kernel_size: Tuple[int] = (5, 3, 3),
    stem_conv_stride: Tuple[int] = (1, 2, 2),
    # Stage configs.
    stage_conv_kernel_size: Tuple[Tuple[int]] = (
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
    ),
    stage_spatial_stride: Tuple[int] = (2, 2, 2, 2),
    stage_temporal_stride: Tuple[int] = (1, 1, 1, 1),
    bottleneck: Callable = create_x3d_bottleneck_block,
    bottleneck_factor: float = 2.25,
    se_ratio: float = 0.0625,
    inner_act: Callable = Swish,
    # Head configs.
    is_head=True,
    head_dim_out: int = 2048,
    head_pool_act: Callable = nn.ReLU,
    head_bn_lin5_on: bool = False,
    head_activation: Callable = nn.Sigmoid,
    head_output_with_global_average: bool = True,
) -> nn.Module:

    torch._C._log_api_usage_once("PYTORCHVIDEO.model.create_x3d")

    blocks = []
    # Create stem for X3D.
    stem_dim_out = round_width(stem_dim_in, width_factor)
    stem = create_x3d_stem(
        in_channels=input_channel,
        out_channels=stem_dim_out,
        conv_kernel_size=stem_conv_kernel_size,
        conv_stride=stem_conv_stride,
        conv_padding=[size // 2 for size in stem_conv_kernel_size],
        norm=norm,
        norm_eps=norm_eps,
        norm_momentum=norm_momentum,
        activation=activation,
    )
    blocks.append(stem)

    # Compute the depth and dimension for each stage
    stage_depths = [1, 2, 5, 3]
    exp_stage = 2.0
    stage_dim1 = stem_dim_in
    stage_dim2 = round_width(stage_dim1, exp_stage, divisor=8)
    stage_dim3 = round_width(stage_dim2, exp_stage, divisor=8)
    stage_dim4 = round_width(stage_dim3, exp_stage, divisor=8)
    stage_dims = [stage_dim1, stage_dim2, stage_dim3, stage_dim4]

    dim_in = stem_dim_out
    # Create each stage for X3D.
    for idx in range(len(stage_depths)):
        dim_out = round_width(stage_dims[idx], width_factor)
        dim_inner = int(bottleneck_factor * dim_out)
        depth = round_repeats(stage_depths[idx], depth_factor)

        stage_conv_stride = (
            stage_temporal_stride[idx],
            stage_spatial_stride[idx],
            stage_spatial_stride[idx],
        )

        stage = create_x3d_res_stage(
            depth=depth,
            dim_in=dim_in,
            dim_inner=dim_inner,
            dim_out=dim_out,
            bottleneck=bottleneck,
            conv_kernel_size=stage_conv_kernel_size[idx],
            conv_stride=stage_conv_stride,
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            se_ratio=se_ratio,
            activation=activation,
            inner_act=inner_act,
        )
        blocks.append(stage)
        dim_in = dim_out

    # Create head for X3D.
    total_spatial_stride = stem_conv_stride[1] * np.prod(stage_spatial_stride)
    total_temporal_stride = stem_conv_stride[0] * np.prod(stage_temporal_stride)

    assert (
        input_clip_length >= total_temporal_stride
    ), "Clip length doesn't match temporal stride!"
    assert (
        input_crop_size[0] >= total_spatial_stride
    ), "Crop size doesn't match spatial stride!"

    head_pool_kernel_size = (
        input_clip_length // total_temporal_stride,
        int(math.ceil(input_crop_size[0] / total_spatial_stride)),
        int(math.ceil(input_crop_size[1] / total_spatial_stride)),
    )

    if is_head:
        head = create_x3d_head(
            dim_in=dim_out,
            dim_inner=dim_inner,
            dim_out=head_dim_out,
            num_classes=model_num_class,
            pool_act=head_pool_act,
            pool_kernel_size=head_pool_kernel_size,
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            bn_lin5_on=head_bn_lin5_on,
            dropout_rate=dropout_rate,
            activation=head_activation,
            output_with_global_average=head_output_with_global_average,
        )
        blocks.append(head)
    return Net(blocks=nn.ModuleList(blocks))


def create_x3d_with_roi_head(
    *,
    # Input clip configs.
    input_channel: int = 3,
    input_clip_length: int = 13,
    input_crop_size: int = 160,
    # Model configs.
    model_num_class: int = 400,
    dropout_rate: float = 0.5,
    width_factor: float = 2.0,
    depth_factor: float = 2.2,
    # Normalization configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    # Activation configs.
    activation: Callable = nn.ReLU,
    # Stem configs.
    stem_dim_in: int = 12,
    stem_conv_kernel_size: Tuple[int] = (5, 3, 3),
    stem_conv_stride: Tuple[int] = (1, 2, 2),
    # Stage configs.
    stage_conv_kernel_size: Tuple[Tuple[int]] = (
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
    ),
    stage_spatial_stride: Tuple[int] = (2, 2, 2, 2),
    stage_temporal_stride: Tuple[int] = (1, 1, 1, 1),
    bottleneck: Callable = create_x3d_bottleneck_block,
    bottleneck_factor: float = 2.25,
    se_ratio: float = 0.0625,
    inner_act: Callable = Swish,
    # Head configs.
    head: Callable = create_res_roi_pooling_head,
    head_pool: Callable = nn.AvgPool3d,
    head_pool_act: Callable = nn.ReLU,
    head_pool_kernel_sizes: Tuple[Tuple[int]] = ((8, 1, 1), (32, 1, 1)),
    head_output_size: Tuple[int] = (1, 1, 1),
    head_activation: Callable = nn.Sigmoid,
    head_output_with_global_average: bool = False,
    head_spatial_resolution: Tuple[int] = (7, 7),
    head_spatial_scale: float = 1.0 / 16.0,
    head_sampling_ratio: int = 0,
    head_dim_out: int = 2048,
    head_bn_lin5_on: bool = False,
    # Pretrained model
    state_dict=None,
    # projected pool
    is_projected=False,
) -> nn.Module:

    model = create_x3d(
        # Input clip configs.
        input_channel=input_channel,
        input_clip_length=input_clip_length,
        input_crop_size=input_crop_size,
        # Model configs.
        model_num_class=model_num_class,
        dropout_rate=dropout_rate,
        width_factor=width_factor,
        depth_factor=depth_factor,
        # Normalization configs.
        norm=norm,
        norm_eps=norm_eps,
        norm_momentum=norm_momentum,
        # Activation configs.
        activation=activation,
        # Stem configs.
        stem_dim_in=stem_dim_in,
        stem_conv_kernel_size=stem_conv_kernel_size,
        stem_conv_stride=stem_conv_stride,
        # Stage configs.
        stage_conv_kernel_size=stage_conv_kernel_size,
        stage_spatial_stride=stage_spatial_stride,
        stage_temporal_stride=stage_temporal_stride,
        bottleneck=bottleneck,
        bottleneck_factor=bottleneck_factor,
        se_ratio=se_ratio,
        inner_act=inner_act,
        # Head configs.
        is_head=False,
    )
    if state_dict:
        load_state_dict_flexible(model, state_dict)
    else:
        initialize_weights(model)

    exp_stage = 2.0
    stage_dim1 = stem_dim_in
    stage_dim2 = round_width(stage_dim1, exp_stage, divisor=8)
    stage_dim3 = round_width(stage_dim2, exp_stage, divisor=8)
    stage_dim4 = round_width(stage_dim3, exp_stage, divisor=8)
    stage_dims = [stage_dim1, stage_dim2, stage_dim3, stage_dim4]
    stage_depths = [1, 2, 5, 3]
    # Create each stage for X3D.
    for idx in range(len(stage_depths)):
        dim_out = round_width(stage_dims[idx], width_factor)

    total_spatial_stride = stem_conv_stride[1] * np.prod(stage_spatial_stride)
    total_temporal_stride = stem_conv_stride[0] * np.prod(stage_temporal_stride)
    assert (
        input_clip_length >= total_temporal_stride
    ), "Clip length doesn't match temporal stride!"
    assert (
        input_crop_size[0] >= total_spatial_stride
    ), "Crop size doesn't match spatial stride!"

    # head_pool_kernel_size = (
    #     input_clip_length // total_temporal_stride,
    #     int(math.ceil(input_crop_size / total_spatial_stride)),
    #     int(math.ceil(input_crop_size / total_spatial_stride)),
    # )
    head_pool_kernel_size = (input_clip_length // total_temporal_stride, 1, 1)

    # pool_module = nn.AvgPool3d(head_pool_kernel_size, stride=1)
    dim_inner = int(bottleneck_factor * dim_out)
    pre_conv_module = nn.Conv3d(
        in_channels=dim_out, out_channels=dim_inner, kernel_size=(1, 1, 1), bias=False
    )

    pre_norm_module = norm(num_features=dim_inner, eps=norm_eps, momentum=norm_momentum)
    pre_act_module = None if head_pool_act is None else head_pool_act()

    if head_pool_kernel_size is None:
        pool_module = nn.AdaptiveAvgPool3d((1, 1, 1))
    else:
        pool_module = nn.AvgPool3d(head_pool_kernel_size, stride=1)

    post_conv_module = nn.Conv3d(
        in_channels=dim_inner, out_channels=head_dim_out, kernel_size=(1, 1, 1), bias=False
    )

    if head_bn_lin5_on:
        post_norm_module = norm(
            num_features=head_dim_out, eps=norm_eps, momentum=norm_momentum
        )
    else:
        post_norm_module = None
    post_act_module = None if head_pool_act is None else head_pool_act()

    projected_pool_module = ProjectedPool(
        pre_conv=pre_conv_module,
        pre_norm=pre_norm_module,
        pre_act=pre_act_module,
        pool=pool_module,
        post_conv=post_conv_module,
        post_norm=post_norm_module,
        post_act=post_act_module,
    )

    if is_projected:
        pool = projected_pool_module
        in_feat = head_dim_out
    else:
        pool = nn.AvgPool3d
        in_feat = dim_out

    head = create_res_roi_pooling_head(
        in_features=in_feat,
        out_features=model_num_class,
        pool=pool,
        pool_kernel_size=head_pool_kernel_size,
        output_size=head_output_size,
        dropout_rate=dropout_rate,
        activation=head_activation,
        output_with_global_average=head_output_with_global_average,
        resolution=head_spatial_resolution,
        spatial_scale=1 / total_spatial_stride,
        sampling_ratio=head_sampling_ratio,
    )

    initialize_weights(head)

    return DetectionBBoxNetwork(model, head)


class X3D(nn.Module):
    def __init__(
        self,
        model_type="detection",
        crop_size=160,
        clip_length=10,
        model_num_class=400,
        model_pretraining=None,
        **kwargs,
    ):
        super(X3D, self).__init__()

        state_dict = self.get_state_dict(model_pretraining)
        self.model_type = model_type

        if model_type == "detection":
            self.model = create_x3d_with_roi_head(
                input_crop_size=crop_size,
                input_clip_length=clip_length,
                model_num_class=model_num_class,
                state_dict=state_dict,
                head_activation=None,
                is_projected=False,
                # head_spatial_scale=1.0 / 32.0,
                # norm=None,
                **kwargs,
            )
        elif model_type == "classification":
            self.model = create_x3d(
                input_crop_size=crop_size,
                input_clip_length=clip_length,
                model_num_class=model_num_class,
                head_activation=None,
                **kwargs,
            )
            if state_dict:
                load_state_dict_flexible(self.model, state_dict)
            else:
                initialize_weights(self.model)
        else:
            raise ValueError(
                f"Please enter a valid model type (classification, detection) not {model_type}"
            )

    def get_state_dict(self, model_name):
        if model_name:
            root_dir = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics"
            checkpoint_paths = {
                "x3d_xs": f"{root_dir}/X3D_XS.pyth",
                "x3d_s": f"{root_dir}/X3D_S.pyth",
                "x3d_m": f"{root_dir}/X3D_M.pyth",
                "x3d_l": f"{root_dir}/X3D_L.pyth",
            }
            checkpoint = load_state_dict_from_url(
                checkpoint_paths[model_name], progress=True, map_location="cpu"
            )
            return checkpoint["model_state"]
        else:
            return None

    def forward(self, x, bboxes=None):
        if self.model_type == "detection":
            return self.model(x, bboxes)
        else:
            return self.model(x)


if __name__ == "__main__":
    model = X3D(model_num_class=1, model_pretraining="x3d_s")
    input_vid = torch.rand(4, 3, 10, 160, 455)
    bboxes = [torch.rand(2, 4), torch.rand(1, 4), torch.rand(2, 4), torch.rand(2, 4)]
    out = model(input_vid, bboxes)
    print("Output shape: ", out.shape)
