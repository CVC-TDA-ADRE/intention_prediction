import wandb
import torch.nn as nn
import numpy as np
import torch.optim as optim
import pytorch_lightning as pl
from torch.functional import Tensor
import torchmetrics

from typing import Dict

from utils.utils import video_to_stream
from utils.visualization import VideoVisualizer
from models.x3d import X3D_RoI
from models.slow_r50 import SlowR50


class IntentionPredictor(pl.LightningModule):
    def __init__(self, data_kwargs, training_kwargs, **model_kwargs) -> None:
        super(IntentionPredictor, self).__init__()

        self.training_kwargs = training_kwargs
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.bce_loss = nn.BCELoss()
        self.fps = data_kwargs["data_fps"]

        if training_kwargs["model_to_use"] == "x3d":
            self.model = X3D_RoI(
                crop_size=data_kwargs["resize"],
                clip_length=data_kwargs["input_seq_size"],
                **model_kwargs,
            )
        elif training_kwargs["model_to_use"] == "slow_r50":
            self.model = SlowR50(model_num_class=1)
        print(self.model)

        # if self.logger is not None:
        #     self.logger.experiment.watch(self.model, log="all")
        class_names = {0: "not_crossing", 1: "crossing"}
        self.visualization = VideoVisualizer(
            num_classes=1, class_names=class_names, thres=0.5, mode="binary"
        )

    def forward(self, x, boxes) -> Tensor:
        return self.model(x, boxes)

    def configure_optimizers(self):
        gen_optimiser = optim.Adam(
            self.parameters(),
            lr=self.training_kwargs["lr"],
            betas=self.training_kwargs["betas"],
        )
        return [gen_optimiser]

    def training_step(self, batch, batch_idx) -> Tensor:
        clip, boxes, labels = batch["clip"], batch["boxes"], batch["label"]
        preds = self.model(clip, boxes)
        print(preds[0], labels[0])
        self.train_accuracy(preds, labels)
        self.log("train/acc", self.train_accuracy, on_step=True)
        loss = self.bce_loss(preds, labels.float())
        self.log("train/loss", loss, on_step=True)
        if self.logger is not None and (
            batch_idx % self.training_kwargs["video_every"] == 0
        ):
            sample_clip = batch["original_clip"][0].cpu() / 255.0
            sample_preds = preds[: len(boxes[0])].detach().cpu()
            sample_boxes = batch["original_boxes"][0]

            preview_video = self.visualization.draw_clip_range(
                sample_clip, sample_preds, sample_boxes
            )
            self.logger.experiment.log(
                {
                    "train/video": wandb.Video(
                        video_to_stream(np.array(preview_video), fps=self.fps),
                        format="mp4",
                        caption=f"True label: {list(labels[: len(boxes[0])].detach().cpu().numpy())}",
                    ),
                    # "video": wandb.Video(video_to_stream(clip, fps=self.fps), format="mp4")
                    "global_step": self.global_step,
                }
            )
        return loss

    def validation_step(self, batch, batch_idx) -> Dict:
        clip, boxes, labels = batch["clip"], batch["boxes"], batch["label"]
        preds = self.model(clip, boxes)
        loss = self.bce_loss(preds, labels.float())
        self.val_accuracy(preds, labels)
        self.log("val/acc", self.val_accuracy)
        self.log("val/loss", loss)

        return loss

    # def training_epoch_end(self, outs):
    #     # additional log mean accuracy at the end of the epoch
    #     self.log("train/acc_epoch", self.train_accuracy.compute())

    # def validation_epoch_end(self, outs):
    #     # additional log mean accuracy at the end of the epoch
    #     self.log("val/acc_epoch", self.val_accuracy.compute())

    # def training_epoch_end(self, outputs) -> Dict:
    #     # Log train loss for epoch + image
    #     pass

    # def validation_epoch_end(self, outputs) -> Dict:
    #     # Log val loss for epoch + image
    #     pass
