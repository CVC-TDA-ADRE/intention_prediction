import torch
import torch.nn as nn
from pytorchvideo.models.hub.resnet import slow_r50_detection


class SlowR50(nn.Module):
    def __init__(
        self,
        crop_size=160,
        clip_length=10,
        model_num_class=400,
        **kwargs,
    ):
        super(SlowR50, self).__init__()

        self.model = slow_r50_detection(
            # input_crop_size=crop_size,
            # input_clip_length=clip_length,
            model_num_class=model_num_class,
            **kwargs,
        )

    def forward(self, x, bboxes):

        return self.model(x, bboxes)


if __name__ == "__main__":
    model = SlowR50(model_num_class=1)
    print(model)
    input_vid = torch.rand(4, 3, 4, 256, 455)
    bboxes = [torch.rand(2, 4), torch.rand(1, 4), torch.rand(2, 4), torch.rand(2, 4)]
    out = model(input_vid, bboxes)
    print("Output shape: ", out.shape)
