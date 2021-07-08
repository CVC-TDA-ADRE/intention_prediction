import torch
import numpy as np


def crop_video_bbox(video, boxes_coord, height, width, scale=1.0):
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
    return video[:, new_top:new_bottom, new_left:new_right, :], boxes_coord


def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tesnor. Got %s" % type(clip))

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
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        return to_tensor(clip)

    def __repr__(self):
        return self.__class__.__name__
