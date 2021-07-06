import torch
import torch.nn as nn

# from pytorchvideo.models.hub.resnet import slow_r50_detection
# from pytorchvideo.models.weight_init import init_net_weights
from typing import Callable, Tuple, Union

from pytorchvideo.models.resnet import create_bottleneck_block, create_res_stage
from pytorchvideo.models.head import create_res_basic_head

# from pytorchvideo.models.net import DetectionBBoxNetwork
from pytorchvideo.models.stem import create_res_basic_stem
from utils.utils import load_state_dict_flexible
from models.models_utils import Net, DetectionBBoxNetwork, create_res_roi_pooling_head

from utils.utils import initialize_weights

_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}


def create_resnet(
    *,
    # Input clip configs.
    input_channel: int = 3,
    # Model configs.
    model_depth: int = 50,
    model_num_class: int = 400,
    dropout_rate: float = 0.5,
    # Normalization configs.
    norm: Callable = nn.BatchNorm3d,
    # Activation configs.
    activation: Callable = nn.ReLU,
    # Stem configs.
    stem_dim_out: int = 64,
    stem_conv_kernel_size: Tuple[int] = (3, 7, 7),
    stem_conv_stride: Tuple[int] = (1, 2, 2),
    stem_pool: Callable = nn.MaxPool3d,
    stem_pool_kernel_size: Tuple[int] = (1, 3, 3),
    stem_pool_stride: Tuple[int] = (1, 2, 2),
    stem: Callable = create_res_basic_stem,
    # Stage configs.
    stage1_pool: Callable = None,
    stage1_pool_kernel_size: Tuple[int] = (2, 1, 1),
    stage_conv_a_kernel_size: Union[Tuple[int], Tuple[Tuple[int]]] = (
        (1, 1, 1),
        (1, 1, 1),
        (3, 1, 1),
        (3, 1, 1),
    ),
    stage_conv_b_kernel_size: Union[Tuple[int], Tuple[Tuple[int]]] = (
        (1, 3, 3),
        (1, 3, 3),
        (1, 3, 3),
        (1, 3, 3),
    ),
    stage_conv_b_num_groups: Tuple[int] = (1, 1, 1, 1),
    stage_conv_b_dilation: Union[Tuple[int], Tuple[Tuple[int]]] = (
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
    ),
    stage_spatial_h_stride: Tuple[int] = (1, 2, 2, 2),
    stage_spatial_w_stride: Tuple[int] = (1, 2, 2, 2),
    stage_temporal_stride: Tuple[int] = (1, 1, 1, 1),
    bottleneck: Union[Tuple[Callable], Callable] = create_bottleneck_block,
    # Head configs.
    head: Callable = create_res_basic_head,
    head_pool: Callable = nn.AvgPool3d,
    head_pool_kernel_size: Tuple[int] = (4, 7, 7),
    head_output_size: Tuple[int] = (1, 1, 1),
    head_activation: Callable = None,
    head_output_with_global_average: bool = True,
) -> nn.Module:
    """
    Build ResNet style models for video recognition. ResNet has three parts:
    Stem, Stages and Head. Stem is the first Convolution layer (Conv1) with an
    optional pooling layer. Stages are grouped residual blocks. There are usually
    multiple stages and each stage may include multiple residual blocks. Head
    may include pooling, dropout, a fully-connected layer and global spatial
    temporal averaging. The three parts are assembled in the following order:
    ::
                                         Input
                                           ↓
                                         Stem
                                           ↓
                                         Stage 1
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                         Stage N
                                           ↓
                                         Head
    Args:
        input_channel (int): number of channels for the input video clip.
        model_depth (int): the depth of the resnet. Options include: 50, 101, 152.
        model_num_class (int): the number of classes for the video dataset.
        dropout_rate (float): dropout rate.
        norm (callable): a callable that constructs normalization layer.
        activation (callable): a callable that constructs activation layer.
        stem_dim_out (int): output channel size to stem.
        stem_conv_kernel_size (tuple): convolutional kernel size(s) of stem.
        stem_conv_stride (tuple): convolutional stride size(s) of stem.
        stem_pool (callable): a callable that constructs resnet head pooling layer.
        stem_pool_kernel_size (tuple): pooling kernel size(s).
        stem_pool_stride (tuple): pooling stride size(s).
        stem (callable): a callable that constructs stem layer.
            Examples include: create_res_video_stem.
        stage_conv_a_kernel_size (tuple): convolutional kernel size(s) for conv_a.
        stage_conv_b_kernel_size (tuple): convolutional kernel size(s) for conv_b.
        stage_conv_b_num_groups (tuple): number of groups for groupwise convolution
            for conv_b. 1 for ResNet, and larger than 1 for ResNeXt.
        stage_conv_b_dilation (tuple): dilation for 3D convolution for conv_b.
        stage_spatial_h_stride (tuple): the spatial height stride for each stage.
        stage_spatial_w_stride (tuple): the spatial width stride for each stage.
        stage_temporal_stride (tuple): the temporal stride for each stage.
        bottleneck (callable): a callable that constructs bottleneck block layer.
            Examples include: create_bottleneck_block.
        head (callable): a callable that constructs the resnet-style head.
            Ex: create_res_basic_head
        head_pool (callable): a callable that constructs resnet head pooling layer.
        head_pool_kernel_size (tuple): the pooling kernel size.
        head_output_size (tuple): the size of output tensor for head.
        head_activation (callable): a callable that constructs activation layer.
        head_output_with_global_average (bool): if True, perform global averaging on
            the head output.
    Returns:
        (nn.Module): basic resnet.
    """

    torch._C._log_api_usage_once("PYTORCHVIDEO.model.create_resnet")

    # Given a model depth, get the number of blocks for each stage.
    assert (
        model_depth in _MODEL_STAGE_DEPTH.keys()
    ), f"{model_depth} is not in {_MODEL_STAGE_DEPTH.keys()}"
    stage_depths = _MODEL_STAGE_DEPTH[model_depth]

    # Broadcast single element to tuple if given.
    if isinstance(stage_conv_a_kernel_size[0], int):
        stage_conv_a_kernel_size = (stage_conv_a_kernel_size,) * len(stage_depths)

    if isinstance(stage_conv_b_kernel_size[0], int):
        stage_conv_b_kernel_size = (stage_conv_b_kernel_size,) * len(stage_depths)

    if isinstance(stage_conv_b_dilation[0], int):
        stage_conv_b_dilation = (stage_conv_b_dilation,) * len(stage_depths)

    if isinstance(bottleneck, Callable):
        bottleneck = [
            bottleneck,
        ] * len(stage_depths)

    blocks = []
    # Create stem for resnet.
    stem = stem(
        in_channels=input_channel,
        out_channels=stem_dim_out,
        conv_kernel_size=stem_conv_kernel_size,
        conv_stride=stem_conv_stride,
        conv_padding=[size // 2 for size in stem_conv_kernel_size],
        pool=stem_pool,
        pool_kernel_size=stem_pool_kernel_size,
        pool_stride=stem_pool_stride,
        pool_padding=[size // 2 for size in stem_pool_kernel_size],
        norm=norm,
        activation=activation,
    )
    blocks.append(stem)

    stage_dim_in = stem_dim_out
    stage_dim_out = stage_dim_in * 4

    # Create each stage for resnet.
    for idx in range(len(stage_depths)):
        stage_dim_inner = stage_dim_out // 4
        depth = stage_depths[idx]

        stage_conv_a_kernel = stage_conv_a_kernel_size[idx]
        stage_conv_a_stride = (stage_temporal_stride[idx], 1, 1)
        stage_conv_a_padding = (
            [size // 2 for size in stage_conv_a_kernel]
            if isinstance(stage_conv_a_kernel[0], int)
            else [[size // 2 for size in sizes] for sizes in stage_conv_a_kernel]
        )

        stage_conv_b_stride = (
            1,
            stage_spatial_h_stride[idx],
            stage_spatial_w_stride[idx],
        )

        stage = create_res_stage(
            depth=depth,
            dim_in=stage_dim_in,
            dim_inner=stage_dim_inner,
            dim_out=stage_dim_out,
            bottleneck=bottleneck[idx],
            conv_a_kernel_size=stage_conv_a_kernel,
            conv_a_stride=stage_conv_a_stride,
            conv_a_padding=stage_conv_a_padding,
            conv_b_kernel_size=stage_conv_b_kernel_size[idx],
            conv_b_stride=stage_conv_b_stride,
            conv_b_padding=(
                stage_conv_b_kernel_size[idx][0] // 2,
                stage_conv_b_dilation[idx][1]
                if stage_conv_b_dilation[idx][1] > 1
                else stage_conv_b_kernel_size[idx][1] // 2,
                stage_conv_b_dilation[idx][2]
                if stage_conv_b_dilation[idx][2] > 1
                else stage_conv_b_kernel_size[idx][2] // 2,
            ),
            conv_b_num_groups=stage_conv_b_num_groups[idx],
            conv_b_dilation=stage_conv_b_dilation[idx],
            norm=norm,
            activation=activation,
        )

        blocks.append(stage)
        stage_dim_in = stage_dim_out
        stage_dim_out = stage_dim_out * 2

        if idx == 0 and stage1_pool is not None:
            blocks.append(
                stage1_pool(
                    kernel_size=stage1_pool_kernel_size,
                    stride=stage1_pool_kernel_size,
                    padding=(0, 0, 0),
                )
            )
    if head is not None:
        head = head(
            in_features=stage_dim_in,
            out_features=model_num_class,
            pool=head_pool,
            output_size=head_output_size,
            pool_kernel_size=head_pool_kernel_size,
            dropout_rate=dropout_rate,
            activation=head_activation,
            output_with_global_average=head_output_with_global_average,
        )
        blocks.append(head)
    return Net(blocks=nn.ModuleList(blocks))


def create_resnet_with_roi_head(
    *,
    # Input clip configs.
    input_channel: int = 3,
    # Model configs.
    model_depth: int = 50,
    model_num_class: int = 80,
    dropout_rate: float = 0.5,
    # Normalization configs.
    norm: Callable = nn.BatchNorm3d,
    # Activation configs.
    activation: Callable = nn.ReLU,
    # Stem configs.
    stem_dim_out: int = 64,
    stem_conv_kernel_size: Tuple[int] = (1, 7, 7),
    stem_conv_stride: Tuple[int] = (1, 2, 2),
    stem_pool: Callable = nn.MaxPool3d,
    stem_pool_kernel_size: Tuple[int] = (1, 3, 3),
    stem_pool_stride: Tuple[int] = (1, 2, 2),
    stem: Callable = create_res_basic_stem,
    # Stage configs.
    stage1_pool: Callable = None,
    stage1_pool_kernel_size: Tuple[int] = (2, 1, 1),
    stage_conv_a_kernel_size: Union[Tuple[int], Tuple[Tuple[int]]] = (
        (1, 1, 1),
        (1, 1, 1),
        (3, 1, 1),
        (3, 1, 1),
    ),
    stage_conv_b_kernel_size: Union[Tuple[int], Tuple[Tuple[int]]] = (
        (1, 3, 3),
        (1, 3, 3),
        (1, 3, 3),
        (1, 3, 3),
    ),
    stage_conv_b_num_groups: Tuple[int] = (1, 1, 1, 1),
    stage_conv_b_dilation: Union[Tuple[int], Tuple[Tuple[int]]] = (
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 2, 2),
    ),
    stage_spatial_h_stride: Tuple[int] = (1, 2, 2, 1),
    stage_spatial_w_stride: Tuple[int] = (1, 2, 2, 1),
    stage_temporal_stride: Tuple[int] = (1, 1, 1, 1),
    bottleneck: Union[Tuple[Callable], Callable] = create_bottleneck_block,
    # Head configs.
    head: Callable = create_res_roi_pooling_head,
    head_pool: Callable = nn.AvgPool3d,
    head_pool_kernel_size: Tuple[int] = (4, 1, 1),
    head_output_size: Tuple[int] = (1, 1, 1),
    head_activation: Callable = nn.Sigmoid,
    head_output_with_global_average: bool = False,
    head_spatial_resolution: Tuple[int] = (7, 7),
    head_spatial_scale: float = 1.0 / 16.0,
    head_sampling_ratio: int = 0,
) -> nn.Module:
    """
    Build ResNet style models for video detection. ResNet has three parts:
    Stem, Stages and Head. Stem is the first Convolution layer (Conv1) with an
    optional pooling layer. Stages are grouped residual blocks. There are usually
    multiple stages and each stage may include multiple residual blocks. Head
    may include pooling, dropout, a fully-connected layer and global spatial
    temporal averaging. The three parts are assembled in the following order:
    ::
                            Input Clip    Input Bounding Boxes
                              ↓                       ↓
                            Stem                      ↓
                              ↓                       ↓
                            Stage 1                   ↓
                              ↓                       ↓
                              .                       ↓
                              .                       ↓
                              .                       ↓
                              ↓                       ↓
                            Stage N                   ↓
                              ↓--------> Head <-------↓
    Args:
        input_channel (int): number of channels for the input video clip.
        model_depth (int): the depth of the resnet. Options include: 50, 101, 152.
        model_num_class (int): the number of classes for the video dataset.
        dropout_rate (float): dropout rate.
        norm (callable): a callable that constructs normalization layer.
        activation (callable): a callable that constructs activation layer.
        stem_dim_out (int): output channel size to stem.
        stem_conv_kernel_size (tuple): convolutional kernel size(s) of stem.
        stem_conv_stride (tuple): convolutional stride size(s) of stem.
        stem_pool (callable): a callable that constructs resnet head pooling layer.
        stem_pool_kernel_size (tuple): pooling kernel size(s).
        stem_pool_stride (tuple): pooling stride size(s).
        stem (callable): a callable that constructs stem layer.
            Examples include: create_res_video_stem.
        stage_conv_a_kernel_size (tuple): convolutional kernel size(s) for conv_a.
        stage_conv_b_kernel_size (tuple): convolutional kernel size(s) for conv_b.
        stage_conv_b_num_groups (tuple): number of groups for groupwise convolution
            for conv_b. 1 for ResNet, and larger than 1 for ResNeXt.
        stage_conv_b_dilation (tuple): dilation for 3D convolution for conv_b.
        stage_spatial_h_stride (tuple): the spatial height stride for each stage.
        stage_spatial_w_stride (tuple): the spatial width stride for each stage.
        stage_temporal_stride (tuple): the temporal stride for each stage.
        bottleneck (callable): a callable that constructs bottleneck block layer.
            Examples include: create_bottleneck_block.
        head (callable): a callable that constructs the detection head which can
            take in the additional input of bounding boxes.
            Ex: create_res_roi_pooling_head
        head_pool (callable): a callable that constructs resnet head pooling layer.
        head_pool_kernel_size (tuple): the pooling kernel size.
        head_output_size (tuple): the size of output tensor for head.
        head_activation (callable): a callable that constructs activation layer.
        head_output_with_global_average (bool): if True, perform global averaging on
            the head output.
        head_spatial_resolution (tuple): h, w sizes of the RoI interpolation.
        head_spatial_scale (float): scale the input boxes by this number.
        head_sampling_ratio (int): number of inputs samples to take for each output
                sample interpolation. 0 to take samples densely.
    Returns:
        (nn.Module): basic resnet.
    """

    model = create_resnet(
        # Input clip configs.
        input_channel=input_channel,
        # Model configs.
        model_depth=model_depth,
        model_num_class=model_num_class,
        dropout_rate=dropout_rate,
        # Normalization configs.
        norm=norm,
        # Activation configs.
        activation=activation,
        # Stem configs.
        stem_dim_out=stem_dim_out,
        stem_conv_kernel_size=stem_conv_kernel_size,
        stem_conv_stride=stem_conv_stride,
        stem_pool=stem_pool,
        stem_pool_kernel_size=stem_pool_kernel_size,
        stem_pool_stride=stem_pool_stride,
        # Stage configs.
        stage1_pool=stage1_pool,
        stage_conv_a_kernel_size=stage_conv_a_kernel_size,
        stage_conv_b_kernel_size=stage_conv_b_kernel_size,
        stage_conv_b_num_groups=stage_conv_b_num_groups,
        stage_conv_b_dilation=stage_conv_b_dilation,
        stage_spatial_h_stride=stage_spatial_h_stride,
        stage_spatial_w_stride=stage_spatial_w_stride,
        stage_temporal_stride=stage_temporal_stride,
        bottleneck=bottleneck,
        # Head configs.
        head=None,
    )
    head = head(
        in_features=stem_dim_out * 2 ** (len(_MODEL_STAGE_DEPTH[model_depth]) + 1),
        out_features=model_num_class,
        pool=head_pool,
        output_size=head_output_size,
        pool_kernel_size=head_pool_kernel_size,
        dropout_rate=dropout_rate,
        activation=head_activation,
        output_with_global_average=head_output_with_global_average,
        resolution=head_spatial_resolution,
        spatial_scale=head_spatial_scale,
        sampling_ratio=head_sampling_ratio,
    )

    return DetectionBBoxNetwork(model, head)


root_dir = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo"
checkpoint_paths = {
    "slow_r50": f"{root_dir}/kinetics/SLOW_8x8_R50.pyth",
    "slow_r50_detection": f"{root_dir}/ava/SLOW_4x16_R50_DETECTION.pyth",
    "c2d_r50": f"{root_dir}/kinetics/C2D_8x8_R50.pyth",
    "i3d_r50": f"{root_dir}/kinetics/I3D_8x8_R50.pyth",
}


class SlowR50(nn.Module):
    def __init__(
        self,
        crop_size=160,
        clip_length=10,
        model_num_class=400,
        model_pretraining=False,
        **kwargs,
    ):
        super(SlowR50, self).__init__()

        self.model = create_resnet_with_roi_head(
            # input_crop_size=crop_size,
            # input_clip_length=clip_length,
            head_activation=None,
            model_num_class=model_num_class,
            **kwargs,
        )
        initialize_weights(self.model)
        if model_pretraining:
            print("Loading from pre-trained network")
            checkpoint = torch.hub.load_state_dict_from_url(
                checkpoint_paths["slow_r50_detection"],
                progress=True,
                map_location="cpu",
            )
            state_dict = checkpoint["model_state"]
            load_state_dict_flexible(self.model, state_dict)

    def forward(self, x, bboxes):
        return self.model(x, bboxes)


if __name__ == "__main__":
    model = SlowR50(model_num_class=1)
    print(model)
    input_vid = torch.rand(4, 3, 4, 256, 455)
    bboxes = [torch.rand(2, 4), torch.rand(1, 4), torch.rand(2, 4), torch.rand(2, 4)]
    out = model(input_vid, bboxes)
    print("Output shape: ", out.shape)
