import torch
import random
import torchvision
import numpy as np


def crop_video_bbox_follow(video, boxes_coord, height, width, scale=1.0):
    boxes_coord = np.array(boxes_coord)
    (left, top), (right, bottom) = boxes_coord[:, :2].min(0), boxes_coord[:, 2:].max(0)
    scale -= 1
    local_width = right - left
    local_height = bottom - top
    new_bottom = min(int(bottom + (local_height * scale) / 2), int(height))
    new_top = max(int(top - (local_height * scale) / 2), 0)
    new_left = max(int(left - (local_width * scale) / 2), 0)
    new_right = min(int(right + (local_width * scale) / 2), int(width))
    boxes_coord = np.array([new_left, new_top, new_right, new_bottom])
    return video[:, new_top:new_bottom, new_left:new_right, :]


def crop_video_bbox(video, boxes_coord, height, width, scale=1.0, random_fail_prob=0):
    new_video = []
    boxes_coord = np.array(boxes_coord, dtype=np.uintc)
    max_width = (boxes_coord[:, 2] - boxes_coord[:, 0]).max()
    max_height = (boxes_coord[:, 3] - boxes_coord[:, 1]).max()
    # resizing = torchvision.transforms.Resize(
    #     (int(max_height * scale), int(max_width * scale))
    # )
    if random_fail_prob > 0:
        random_crop = torchvision.transforms.RandomCrop(
            (int(max_height * scale), int(max_width * scale))
        )
    else:
        random_crop = None

    scale -= 1
    for i, bbox in enumerate(boxes_coord):
        if random_crop and random_fail_prob > random.uniform(0, 1):
            frame = random_crop(video[i].permute(2, 0, 1)).permute(1, 2, 0)
            new_video.append(frame)
            continue
        left, top, right, bottom = map(int, bbox)
        local_width = right - left
        local_height = bottom - top
        diff_width = max_width - local_width
        diff_height = max_height - local_height
        new_bottom = int(
            bottom + (diff_height // 2) + round((max_height * scale) / 2 + 0.1)
        )
        if new_bottom > int(height):
            diff = new_bottom - height
            new_bottom = height
            top -= diff
        new_top = int(top - round(diff_height / 2 + 0.1) - (max_height * scale) // 2)
        if new_top < 0:
            diff = abs(new_top)
            new_top = 0
            new_bottom += diff
        new_left = int(left - round(diff_width / 2 + 0.1) - (max_width * scale) // 2)
        if new_left < 0:
            diff = abs(new_left)
            new_left = 0
            right += diff
        new_right = int(right + (diff_width // 2) + round((max_width * scale) / 2 + 0.1))
        if new_right > int(width):
            diff = new_right - width
            new_right = width
            new_left -= diff
        frame = video[i, new_top:new_bottom, new_left:new_right, :]
        # frame = resizing(frame.permute(2, 0, 1)).permute(1, 2, 0)
        new_video.append(frame)
    return torch.stack(new_video)


def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True


def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimenions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
    """
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError("clip tensor should have data type uint8. Got %s" % str(clip.dtype))
    return clip.float().permute(3, 0, 1, 2) / 255.0


def uniform_temporal_subsample(
    x: torch.Tensor, num_samples: int, temporal_dim: int = -3
) -> torch.Tensor:
    """
    Uniformly subsamples num_samples indices from the temporal dimension of the video.
    When num_samples is larger than the size of temporal dimension of the video, it
    will sample frames based on nearest neighbor interpolation.
    Args:
        x (torch.Tensor): A video tensor with dimension larger than one with torch
            tensor type includes int, long, float, complex, etc.
        num_samples (int): The number of equispaced samples to be selected
        temporal_dim (int): dimension of temporal to perform temporal subsample.
    Returns:
        An x-like Tensor with subsampled temporal dimension.
    """
    t = x.shape[temporal_dim]
    assert num_samples > 0 and t > 0
    # Sample by nearest neighbor interpolation if num_samples > t.
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    return torch.index_select(x, temporal_dim, indices)


class UniformTemporalSubsample(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.uniform_temporal_subsample``.
    """

    def __init__(self, num_samples: int):
        super().__init__()
        self._num_samples = num_samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return uniform_temporal_subsample(x, self._num_samples)


class ToTensorVideo(object):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
        """
        return to_tensor(clip)

    def __repr__(self):
        return self.__class__.__name__
